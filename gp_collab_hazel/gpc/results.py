"""Review and plotting for gp_collab_hazel runs.

Copied from DOPE-MURI `hazel_gp/results.py` with the producer-side functions
removed (`collect_results`, `task_statuses`, `export_result_archive`) -- those
build the tables, and `gpc export-hazel` already did that. Everything kept here
is byte-identical to the original, so the figures and metrics are that
project's, not a reimplementation. `read_json` and `file_hash` are inlined from
`hazel_gp/config.py`, and `review_controls` from `hazel_gp/notebook.py`, so this
one file has no hazel_gp dependency.

Drop it next to the notebook and `import results`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET = "Product_Yield_PCT_Area_UV"


def read_json(path):
    """From hazel_gp/config.py."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path) -> str:
    """From hazel_gp/config.py."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(runs, bundle="inputs", refresh: bool = False):
    """Export `runs` into the collected/ layout `load_results` reads, if needed.

    Returns the path to pass to `load_results`. Set refresh=True to rebuild the
    tables after adding or rerunning tasks.
    """
    runs = Path(runs)
    if not any(d.name.count("__") == 1 for d in runs.iterdir() if d.is_dir()):
        # Submission scripts write to runs/<run_tag>/, so accept the parent too.
        nested = [d for d in sorted(runs.iterdir()) if d.is_dir()
                  and any(c.name.count("__") == 1 for c in d.iterdir() if c.is_dir())]
        if len(nested) == 1:
            print(f"no tasks directly in {runs}; using {nested[0]}")
            runs = nested[0]
        elif len(nested) > 1:
            raise ValueError(f"Several run folders under {runs}: "
                             f"{[d.name for d in nested]}. Set RUNS to one of them.")
        else:
            raise FileNotFoundError(f"No finished tasks under {runs}")
    root = runs / "hazel_export"
    if refresh or not (root / "collected" / "collection.json").exists():
        from gpc.export_hazel import export
        return export(bundle, runs, root, hazel_method_names=True)
    print(f"using the existing export at {root} (refresh=True to rebuild)")
    return root


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def regression_metrics(y, mean, std) -> dict:
    y, mean, std = (np.asarray(v, dtype=float).reshape(-1) for v in (y, mean, std))
    if not (len(y) == len(mean) == len(std)) or not len(y):
        raise ValueError("Metric arrays must have equal nonzero lengths")
    if not np.isfinite(np.r_[y, mean, std]).all() or np.any(std <= 0):
        raise ValueError("Predictions/uncertainties must be finite with positive predictive standard deviation")
    z = (y - mean) / std
    pit = norm.cdf(z)
    qs = np.linspace(0, 1, 101)
    pit_cdf = np.array([(pit <= q).mean() for q in qs])
    out = {"n_test": len(y), "r2": float(r2_score(y, mean)) if len(y) > 1 and np.var(y) > 0 else None,
           "rmse": float(np.sqrt(mean_squared_error(y, mean))), "mae": float(mean_absolute_error(y, mean)),
           "kendall_tau": _finite_or_none(kendalltau(y, mean).statistic) if len(y) > 1 else None,
           "predictive_nll": float(np.mean(0.5 * z**2 + np.log(std) + 0.5 * np.log(2 * np.pi))),
           "mean_predictive_std": float(std.mean()),
           "pit_cdf_mae": float(np.mean(np.abs(pit_cdf - qs))),
           "abs_residual_std_spearman": _finite_or_none(spearmanr(np.abs(y - mean), std).statistic)
                if len(y) > 1 and np.std(std) > 0 and np.std(np.abs(y-mean)) > 0 else None}
    for coverage in (0.50, 0.80, 0.95):
        q = norm.ppf((1 + coverage) / 2)
        out[f"coverage_{int(100*coverage)}"] = float((np.abs(z) <= q).mean())
        out[f"width_{int(100*coverage)}"] = float((2 * q * std).mean())
    return out



def load_results(runs):
    directory = Path(runs) / "collected"
    info = read_json(directory / "collection.json")
    for name, digest in info["output_hashes"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Collected result changed: {name}")
    tables = {}
    for name in ("predictions", "metrics_by_split", "summary", "task_status"):
        try:
            tables[name] = pd.read_csv(directory / f"{name}.csv", keep_default_na=False)
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame()
    return tables, info


def grouped_predictions(predictions, model, method="iid_matched"):
    """Return {group: {y_pred: [array per repeat], y_std: [...], test_idx: [...]}}."""
    p = predictions.loc[(predictions.model == model) & (predictions.method == method)]
    out = {}
    for group, part in p.groupby("reference_group", sort=False):
        folds = [v for _, v in part.groupby("split_id", sort=True)]
        out[group] = {"y_pred": [v.y_pred.to_numpy() for v in folds],
                      "y_std": [v.predictive_std.to_numpy() for v in folds],
                      "test_idx": [v.test_index.to_numpy() for v in folds],
                      "y_true": [v.y_true.to_numpy() for v in folds]}
    return out


MODEL_LABELS = {"pc_scores": "PC loading", "pc_scores_long": "PC loading (long)"}
# Canonical left-to-right order for every figure and table, so the hazel and
# rxnpredict reviews read the same way regardless of the order a run's config
# happened to list its models in. Anything unlisted keeps its collected order
# and follows these.
MODEL_ORDER = ["ligand_ohe", "selected_2", "selected_5", "pc_top", "pc_scores",
               "pc_scores_long",
               "rxnpredict_full", "rxnpredict_full_ohe"]
METHOD_LABELS = {"lolo": "LOLO", "iid_matched": "Matched IID",
                 "kfold": "5-fold CV", "holdout": "80:20 split"}
# Ranking direction per metric. Coverage metrics are absent on purpose: closeness to the
# nominal level is what matters there, so a plain sort would be misleading.
RANK_DIRECTION = {"rmse": "lower", "mae": "lower", "predictive_nll": "lower",
                  "pit_cdf_mae": "lower", "mean_predictive_std": "lower",
                  "width_50": "lower", "width_80": "lower", "width_95": "lower",
                  "r2": "higher", "kendall_tau": "higher",
                  "abs_residual_std_spearman": "higher"}


def model_label(model: str) -> str:
    """Display name for a representation; unlisted models keep their config key."""
    return MODEL_LABELS.get(model, model)


def method_label(method: str) -> str:
    """Display name for an evaluation type; unlisted methods keep their config key."""
    return METHOD_LABELS.get(method, method)


def _select(predictions, model, method, reference_group=None, split_id=None):
    p = predictions.loc[(predictions.model == model) & (predictions.method == method)].copy()
    if reference_group is not None:
        p = p[p.reference_group == reference_group]
    if split_id is not None:
        p = p[p.split_id == split_id]
    if p.empty:
        raise ValueError("No predictions match the selected model/method/group/split")
    return p


def _heading(model, method, reference_group=None, title=None) -> str:
    """`title`, if given, is a str.format template with `model`, `method`,
    `reference_group` fields (labelled/raw as passed in) -- e.g.
    "{model} on {method}". None keeps the default "Model | method[| ligand]"."""
    model_l, method_l = model_label(model), method_label(method)
    if title:
        return title.format(model=model_l, method=method_l, reference_group=reference_group or "")
    parts = [model_l, method_l]
    if reference_group:
        parts.append(str(reference_group))
    return " | ".join(parts)


def _score_line(y, pred) -> str:
    """Subtitle carrying the two scores for exactly the points that were drawn."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    r2 = r2_score(y, pred) if len(y) > 1 and np.var(y) > 0 else None
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return f"$R^2$ = {'n/a' if r2 is None else f'{r2:.3f}'}    RMSE = {rmse:.2f} %"


def _model_order(frame) -> list:
    """Canonical display order (MODEL_ORDER), never alphabetical as pivot/groupby
    would give. Models outside MODEL_ORDER keep the order the collector wrote them
    and follow the listed ones."""
    seen = list(dict.fromkeys(frame.model))
    listed = [m for m in MODEL_ORDER if m in seen]
    return listed + [m for m in seen if m not in MODEL_ORDER]


def _method_order(frame) -> list:
    return list(dict.fromkeys(frame.method))


def rank_models(metrics_by_split, metric="rmse"):
    """Rank representations within each evaluation type on the mean of their split metrics.

    Split metrics are averaged rather than pooled, because matched-IID repeats overlap and
    cannot be treated as independent observations. This ranks reported outcomes only. It is
    not a model-selection procedure: choosing a representation by its outer test score needs
    an additional inner validation loop.
    """
    if metric not in RANK_DIRECTION:
        raise ValueError(f"Cannot rank on {metric}; choose from {sorted(RANK_DIRECTION)}")
    ascending = RANK_DIRECTION[metric] == "lower"
    frame = metrics_by_split.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    rows = []
    for method in _method_order(frame):
        part = frame[frame.method == method]
        agg = (part.groupby("model", sort=False)[metric]
               .agg(mean_value="mean", spread="std", fits="count").reset_index())
        agg = agg.sort_values("mean_value", ascending=ascending, kind="stable")
        for rank, row in enumerate(agg.itertuples(index=False), start=1):
            rows.append({"experiment": method_label(method), "rank": rank,
                         "model": model_label(row.model),
                         f"mean_{metric}": row.mean_value,
                         "std_across_splits": row.spread, "fits": int(row.fits),
                         "better": RANK_DIRECTION[metric]})
    return pd.DataFrame(rows)


def rank_table(metrics_by_split, metric="rmse"):
    """One row per evaluation type, representations in ranked order (best in column 1)."""
    ranked = rank_models(metrics_by_split, metric)
    order = list(dict.fromkeys(ranked.experiment))
    return (ranked.pivot(index="experiment", columns="rank", values="model")
            .reindex(order).rename_axis(columns=f"rank by {metric}"))


def lolo_report(predictions, metrics_by_split, models=None):
    """The three LOLO views in one table, labelled by a `view` column.

    * `per_ligand` - one row per held-out ligand, scored on that ligand's own test rows.
    * `mean_across_ligands` - unweighted mean of those rows; every ligand counts once,
      which is fair here only because the LOLO folds are equal in size.
    * `pooled_over_folds` - all eight folds concatenated into a single out-of-fold vector,
      scored once; every reaction counts once.

    Pooled and averaged scores are not interchangeable. Pooled R2 is measured against the
    variance of the whole dataset, while a per-ligand R2 is measured against the variance
    within that one ligand, which is smaller; the averaged value is therefore systematically
    harsher and the two must not be compared. Pooled RMSE is the root of the mean squared
    error, whereas the averaged value is the mean of per-fold roots, so the pooled figure is
    the larger of the two.
    """
    metrics = metrics_by_split[metrics_by_split.method == "lolo"]
    preds = predictions[predictions.method == "lolo"]
    if metrics.empty or preds.empty:
        raise ValueError("No LOLO results to report")
    score_names = [k for k in regression_metrics([0., 1.], [0., 1.], [1., 1.]) if k != "n_test"]
    rows = []
    for model in (models or _model_order(metrics)):
        part = metrics[metrics.model == model]
        for _, row in part.iterrows():
            rows.append({"model": model_label(model), "view": "per_ligand",
                         "scope": row["reference_group"], "n_test": row.get("n_test"),
                         **{k: row[k] for k in score_names}})
        rows.append({"model": model_label(model), "view": "mean_across_ligands",
                     "scope": f"{len(part)} ligands", "n_test": None,
                     **{k: _finite_or_none(pd.to_numeric(part[k], errors="coerce").mean())
                        for k in score_names}})
        p = preds[preds.model == model]
        if p.row_id.duplicated().any():
            raise ValueError("Pooled LOLO predictions repeat rows")
        pooled = regression_metrics(p.y_true, p.y_pred, p.predictive_std)
        rows.append({"model": model_label(model), "view": "pooled_over_folds",
                     "scope": f"{len(part)} folds", "n_test": pooled["n_test"],
                     **{k: pooled[k] for k in score_names}})
    return pd.DataFrame(rows)


def plot_parity(predictions, model, method="lolo", reference_group=None, split_id=None, title=None):
    """Observed against predicted yield. Points are not separated by ligand.

    `title`, if given, overrides the default "Model | method" heading -- see `_heading`.
    """
    import matplotlib.pyplot as plt
    p = _select(predictions, model, method, reference_group, split_id)
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    ax.scatter(p.y_true, p.y_pred, s=14, alpha=0.65, color="tab:blue", edgecolors="none")
    lo, hi = min(0, p.y_pred.min()), max(100, p.y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Observed yield (%)", ylabel="Predicted yield (%)")
    fig.suptitle(_heading(model, method, reference_group, title))
    ax.set_title(_score_line(p.y_true, p.y_pred), fontsize=9, color="0.3")
    return fig


def plot_mean_trend(predictions, model, method="lolo", reference_group=None,
                    bins=10, band="empirical", title=None):
    """Mean predicted yield across observed-yield bins, shaded by one standard deviation.

    `band="empirical"` shades the spread of the predictions inside each bin.
    `band="predictive"` shades the mean predictive standard deviation the GP reported there.
    """
    import matplotlib.pyplot as plt
    if band not in ("empirical", "predictive"):
        raise ValueError("band must be empirical or predictive")
    p = _select(predictions, model, method, reference_group)
    edges = np.linspace(0.0, 100.0, int(bins) + 1)
    assigned = np.clip(np.digitize(p.y_true.to_numpy(dtype=float), edges[1:-1]), 0, int(bins) - 1)
    centre, mean, spread = [], [], []
    for b in range(int(bins)):
        part = p.iloc[np.flatnonzero(assigned == b)]
        if part.empty:
            continue
        centre.append(float(part.y_true.mean()))
        mean.append(float(part.y_pred.mean()))
        spread.append(float(part.y_pred.std(ddof=0)) if band == "empirical"
                      else float(part.predictive_std.mean()))
    if not centre:
        raise ValueError("No populated observed-yield bins")
    centre, mean, spread = (np.asarray(v, dtype=float) for v in (centre, mean, spread))
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    lo, hi = min(0, p.y_pred.min()), max(100, p.y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.fill_between(centre, mean - spread, mean + spread, color="tab:blue", alpha=0.25,
                    label="$\\pm$1 std")
    ax.plot(centre, mean, color="tab:blue", marker="o", markersize=4, linewidth=1.8,
            label="Mean")
    ax.set(xlabel="Observed yield (%)", ylabel="Predicted yield (%)")
    fig.suptitle(_heading(model, method, reference_group, title))
    ax.set_title(_score_line(p.y_true, p.y_pred), fontsize=9, color="0.3")
    ax.legend(fontsize=8, frameon=False)
    return fig


def _lolo_set(predictions, model, make_figure):
    p = _select(predictions, model, "lolo")
    out = [(ref, make_figure(ref)) for ref in sorted(p.reference_group.unique()) if ref]
    out.append(("all ligands pooled", make_figure(None)))
    return out


def plot_lolo_parity_set(predictions, model):
    """One parity figure per held-out ligand, then one pooling every LOLO fold."""
    return _lolo_set(predictions, model,
                     lambda ref: plot_parity(predictions, model, "lolo", reference_group=ref))


def plot_lolo_trend_set(predictions, model, bins=10, band="empirical"):
    """One mean-trend figure per held-out ligand, then one pooling every LOLO fold."""
    return _lolo_set(predictions, model,
                     lambda ref: plot_mean_trend(predictions, model, "lolo",
                                                 reference_group=ref, bins=bins, band=band))


def plot_lolo_comparison(metrics, metric="rmse"):
    import matplotlib.pyplot as plt
    part = metrics[metrics.method == "lolo"]
    if part.empty:
        raise ValueError("No LOLO results available")
    table = part.pivot(index="reference_group", columns="model", values=metric)
    order = [m for m in _model_order(metrics) if m in table.columns]
    table = table.reindex(columns=order).rename(columns=model_label)
    ax = table.plot.bar(figsize=(10, 4), width=0.85)
    ax.set(xlabel="Held-out ligand", ylabel=metric.upper())
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Model", fontsize=8)
    ax.figure.tight_layout()
    return ax.figure


def plot_calibration(predictions, method="lolo"):
    """Whether the predictive distributions are honest about their own uncertainty.

    Each test point gets a PIT (probability integral transform) value, the predictive CDF
    evaluated at the observation: `PIT = Phi((y_obs - mu_pred) / sigma_pred)`. If a
    predictive distribution is correct, its PIT values are uniform on [0, 1].

    The x axis is a nominal probability `q`. The y axis is the observed fraction of test
    points with `PIT <= q`. Uniform PIT values put that fraction at `q` for every `q`, which
    is the dashed diagonal. Reading a departure:

    * Above the diagonal at low `q` and below it at high `q` (an S through the middle): PIT
      values pile up at both ends, so sigma is too small and the intervals are too narrow.
      The model is overconfident.
    * The mirror image: sigma is too large, intervals too wide, underconfident.
    * The whole curve displaced to one side: the predictive mean is biased, not just its
      spread.

    At `q = 0.9`, an observed 0.8 means only 80% of points fell below their nominal 90th
    predictive percentile. For IID this averages per-split PIT CDFs, making no claim that
    the overlapping repeats are independent.
    """
    import matplotlib.pyplot as plt
    part = predictions[predictions.method == method]
    if part.empty:
        raise ValueError("No predictions for this method")
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    for model in _model_order(part):
        group = part[part.model == model]
        curves = []
        groups = [p for _, p in group.groupby("split_id")] if method == "iid_matched" else [group]
        for p in groups:
            pit = norm.cdf((p.y_true - p.y_pred) / p.predictive_std)
            curves.append(np.array([(pit <= q).mean() for q in grid]))
        ax.plot(grid, np.mean(curves, axis=0), label=model_label(model))
    ax.plot(grid, grid, "k--", linewidth=1)
    ax.set(xlabel="Nominal cumulative probability, $q$",
           ylabel="Observed fraction with PIT $\\leq q$",
           title=method_label(method), xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, frameon=False)
    return fig


def _task_dir(runs, model, method) -> Path:
    """`<runs>/<model>__<method>`, the raw per-task output `prepare` reads from.

    Also found one level down, so the same `RUNS` the notebook passes to `prepare`
    works here when it holds run-tag folders rather than tasks.
    """
    runs = Path(runs)
    direct = runs / f"{model}__{method}"
    if direct.is_dir():
        return direct
    nested = [d / f"{model}__{method}" for d in sorted(runs.iterdir()) if d.is_dir()]
    nested = [d for d in nested if d.is_dir()]
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        raise ValueError(f"Several {model}__{method} tasks under {runs}: "
                         f"{[d.parent.name for d in nested]}. Point runs at one of them.")
    raise FileNotFoundError(f"No {model}__{method} task under {runs}")


def _fold_encoder(bundle, model):
    """(reactions, prepared, X, fit_fold) for whichever clone this is.

    The rxnpredict bundle carries a published descriptor table that the hazel one
    does not: there `load_bundle` returns it as a fifth value and `feature_frame` /
    `_preprocessor` take it as a trailing argument. The signatures decide which, so
    this file stays identical in both projects.

    `fit_fold(train_idx)` returns the preprocessor fitted on those rows only -- the
    same fit the run's own pipeline made for that fold, so its output column order is
    the one the kernel indexed and its scaler never saw the held-out ligand.
    """
    import inspect
    from gpc.data import load_bundle
    from gpc.features import _preprocessor, feature_frame
    parts = load_bundle(bundle)
    reactions, ligands, reference, prepared = parts[:4]
    rxn_features = parts[4] if len(parts) > 4 else None
    rxn_columns = [c for c in rxn_features.columns if c != "row_id"] if rxn_features is not None else []
    frame_takes_rxn = len(inspect.signature(feature_frame).parameters) > 5
    prep_takes_rxn = len(inspect.signature(_preprocessor).parameters) > 4
    X = (feature_frame(reactions, ligands, model, prepared, reference, rxn_features)
         if frame_takes_rxn else feature_frame(reactions, ligands, model, prepared, reference))

    def fit_fold(train_idx):
        p = (_preprocessor(X, model, prepared, reference, rxn_columns) if prep_takes_rxn
             else _preprocessor(X, model, prepared, reference))
        return p.fit(X.iloc[train_idx])

    return reactions, prepared, X, fit_fold


def _task_folds(reactions, meta):
    """The task's own folds, rebuilt from the seed and stratify flag it recorded."""
    from gpc.splits import make_folds
    return make_folds(reactions, meta["method"], meta["seed"], meta["stratify"])


# gpc/train.py's `feature_groups` is the authority on which encoded columns each
# kernel group covers, but importing it pulls in torch and this review path is
# deliberately torch-free. The two rules below are copied from it; if the grouping
# definitions there change, change these with them.
def _ligand_columns(columns):
    """Ligand block: the numeric descriptors plus the ligand identity one-hots."""
    return [c for c in columns if c.startswith("num__") or c.startswith("cat__ligand_")]


def _kernel_groups(grouping, prepared, columns):
    """Encoded columns per kernel group, ordered as that group's lengthscales index them."""
    if grouping == "all":
        return {"fp_all": list(columns)}
    if grouping == "ligand_conditions":
        return {"fp_ligand": _ligand_columns(columns),
                "fp_conditions": [c for c in columns if c.startswith("cat__")
                                  and not c.startswith("cat__ligand_")]}
    if grouping == "per_field":
        groups = {"fp_ligand": _ligand_columns(columns)}
        for field in prepared["data"]["common_categorical"]:
            groups[f"fp_{field}"] = [c for c in columns if c.startswith(f"cat__{field}_")]
        return groups
    raise ValueError(f"Unknown grouping {grouping!r}")


def _split_lengthscale_key(key):
    """('fp_all', 7) for 'fp_all[7]'; ('fp_all', None) for one lengthscale per group."""
    if key.endswith("]") and "[" in key:
        head, _, index = key[:-1].partition("[")
        if index.isdigit():
            return head, int(index)
    return key, None


def ard_lengthscales(runs, bundle, model, method="lolo"):
    """Per-feature ARD lengthscales for one task, one row per (fold, encoded feature).

    Reads `<runs>/<model>__<method>/scores.json` and `meta.json` -- the raw GP_collab
    output, not the exported review tables, which carry no lengthscales. `bundle` is
    the prepared inputs directory.

    `relevance` is 1 / lengthscale. An RBF varies fastest along its shortest
    lengthscales, so a large relevance means the fit leans on that column. It ranks
    columns within one fold of one model and nothing further: it is not a significance
    test, and relevances from different models or kernels share no scale. Numeric
    columns are standardized on the training fold while one-hots stay raw 0/1
    indicators, so a one-hot's lengthscale is not on the same input scale as a
    descriptor's either -- read the ranking within a family.

    The kernel names lengthscales positionally (`fp_all[7]`), indexing the fitted
    preprocessor's column order. Under LOLO that order differs between folds, because
    each fold drops the held-out ligand's one-hot, so position 7 is a different
    feature in a different fold. The preprocessor is therefore refit on each fold's
    training rows and every index resolved to a name there. Aggregate by name.

    Raises if the run was fitted isotropically (`ard: false`), which reports one
    lengthscale for a whole group and holds no per-feature information.
    """
    directory = _task_dir(runs, model, method)
    meta = read_json(directory / "meta.json")
    scores = read_json(directory / "scores.json")
    seed = str(meta["seed"])
    if seed not in scores or "test_lengthscale" not in scores[seed]:
        raise ValueError(f"{directory / 'scores.json'} has no test_lengthscale for seed "
                         f"{seed}; the run must be cross-validated with return_ls=True.")
    per_fold = scores[seed]["test_lengthscale"]
    grouping = meta["config"]["gp"]["grouping"]
    reactions, prepared, X, fit_fold = _fold_encoder(bundle, model)
    folds, _ = _task_folds(reactions, meta)
    if len(per_fold) != len(folds):
        raise ValueError(f"{len(per_fold)} lengthscale records for {len(folds)} folds; "
                         f"{directory} does not match the bundle it is being read against")

    rows = []
    for fold, ((train, _), reported) in enumerate(zip(folds, per_fold)):
        names = list(fit_fold(train).get_feature_names_out())
        groups = _kernel_groups(grouping, prepared, names)
        ligand = set(_ligand_columns(names))
        isotropic = [k for k, cols in groups.items() if k in reported and len(cols) > 1]
        if isotropic:
            raise ValueError(
                f"{directory / 'scores.json'} reports a single lengthscale for "
                + ", ".join(f"{k} ({len(groups[k])} columns)" for k in isotropic)
                + f", so this run was fitted isotropically -- its meta.json records "
                f"gp.ard = {meta['config']['gp']['ard']}. Per-feature relevance needs one "
                f"lengthscale per column: set \"ard\": true under \"gp\" in the run config, "
                f"re-run {model}/{method}, and point ard_lengthscales at the new runs "
                f"directory.")
        for key, value in reported.items():
            group, index = _split_lengthscale_key(key)
            if group not in groups:
                raise ValueError(f"Fold {fold} reports lengthscale {key!r}, which is not a "
                                 f"group of grouping {grouping!r}")
            if value is None:
                raise ValueError(f"Kernel {meta['config']['gp']['kernel']!r} reports no "
                                 f"lengthscale for {key!r}; relevance is undefined for it")
            if value <= 0:
                raise ValueError(f"Fold {fold} reports a non-positive lengthscale for {key!r}")
            feature = groups[group][0] if index is None else groups[group][index]
            rows.append({"fold": fold, "group": group, "feature": feature,
                         "lengthscale": float(value), "relevance": 1.0 / float(value),
                         "family": "ligand" if feature in ligand else "condition"})
    return pd.DataFrame(rows)


def rank_features(runs, bundle, model, method="lolo", ligand_only=False, top=None):
    """`ard_lengthscales` aggregated across folds by feature NAME, most relevant first.

    The centre is the median relevance over the folds a feature appears in and the
    spread is that feature's fold quartiles; a mean and a standard deviation are easy
    to drag around on a ratio scale with this few folds. Long quartile spans mean the
    folds disagree and the ranking there is soft.

    `folds` is how many folds the feature existed in, out of `n_folds`. Under LOLO a
    ligand one-hot is absent from its own fold, so it is scored on `n_folds - 1` and
    its median is not measured over the same folds as the rest -- that is reported
    rather than smoothed over. `group` is the kernel group the lengthscale came from;
    with more than one group their relevances come from different kernels and are not
    directly comparable.

    `ligand_only=True` keeps the ligand block (numeric descriptors and ligand
    one-hots); `top` truncates the sorted table.
    """
    frame = ard_lengthscales(runs, bundle, model, method)
    n_folds = int(frame.fold.nunique())
    if ligand_only:
        frame = frame[frame.family == "ligand"]
        if frame.empty:
            raise ValueError(f"{model} encodes no ligand-block columns to rank")
    table = (frame.groupby(["feature", "family", "group"], sort=False)
             .agg(median_relevance=("relevance", "median"),
                  relevance_q1=("relevance", lambda s: s.quantile(0.25)),
                  relevance_q3=("relevance", lambda s: s.quantile(0.75)),
                  median_lengthscale=("lengthscale", "median"),
                  folds=("relevance", "count"))
             .reset_index())
    table["n_folds"] = n_folds
    table = table.sort_values("median_relevance", ascending=False, kind="stable")
    if top:
        table = table.head(int(top))
    return table.reset_index(drop=True)


def plot_feature_importance(runs, bundle, model, method="lolo", ligand_only=False, top=20):
    """Median relevance per encoded feature, most relevant at the top, coloured by family.

    Whiskers span the fold quartiles, so a long whisker means the folds disagree about
    that feature. A feature that is missing from some folds -- a ligand one-hot under
    LOLO -- says so in its label, because its median is not measured over the same
    folds as its neighbours'.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    table = rank_features(runs, bundle, model, method, ligand_only)
    total = len(table)
    if top:
        table = table.head(int(top))
    colours = {"ligand": "tab:blue", "condition": "tab:orange"}
    y = np.arange(len(table))[::-1]
    error = np.vstack([(table.median_relevance - table.relevance_q1).to_numpy(),
                       (table.relevance_q3 - table.median_relevance).to_numpy()])
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.3 * len(table) + 1.6)),
                           constrained_layout=True)
    ax.barh(y, table.median_relevance.to_numpy(), xerr=error, height=0.75,
            color=[colours.get(f, "tab:gray") for f in table.family],
            error_kw={"ecolor": "0.35", "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels([name if n == nf else f"{name}  ({n}/{nf} folds)"
                        for name, n, nf in zip(table.feature, table.folds, table.n_folds)],
                       fontsize=8)
    ax.set(xlabel="Median relevance across folds  (1 / lengthscale)")
    ax.legend(handles=[Patch(color=colours[f], label=f)
                       for f in ("ligand", "condition") if (table.family == f).any()],
              fontsize=8, frameon=False)
    fig.suptitle(_heading(model, method))
    shown = f"{len(table)} of {total} encoded columns" if len(table) < total else \
            f"all {total} encoded columns"
    ax.set_title(f"{shown}, {int(table.n_folds.iloc[0])} folds"
                 + (" | ligand block only" if ligand_only else ""),
                 fontsize=9, color="0.3")
    return fig


def ligand_distances(runs, bundle, model, method="lolo"):
    """How far each held-out ligand sits from the training ligands, one row per fold.

    One vector per ligand, not per reaction. The fold's own preprocessor standardizes
    the numeric ligand descriptors -- fitted on that fold's training rows only, so the
    held-out ligand contributes nothing to the scaler and the distance leaks nothing
    -- and each ligand's rows are averaged to one centroid. Distances are Euclidean in
    that standardized space. For the descriptor models the numeric block is constant
    within a ligand, so the mean is that ligand's own vector rather than an
    approximation; it is written as a mean so a model whose numeric block varies by
    row still reduces to one vector per ligand.

    Reaction-condition one-hots are excluded: every ligand is run over the same
    conditions, so including them would measure the design grid instead of the
    chemistry. Ligand one-hots are excluded too, for the opposite reason -- they place
    every ligand exactly the same distance apart.

    `centroid_distance` is to the mean of the training ligands' vectors, one weight
    per ligand rather than per row. `nearest_distance` is to the closest single
    training ligand, usually the better read on extrapolation: a ligand can sit near
    the training mean while resembling nothing actually trained on.

    Raises for a model whose ligand block is purely one-hot (`ligand_ohe`), which has
    no descriptor geometry to measure, and for a method whose folds do not each hold
    out one ligand.
    """
    directory = _task_dir(runs, model, method)
    meta = read_json(directory / "meta.json")
    reactions, prepared, X, fit_fold = _fold_encoder(bundle, model)
    folds, references = _task_folds(reactions, meta)
    if not all(references):
        raise ValueError(f"{meta['method']} folds do not each hold out one ligand, so there "
                         f"is no held-out ligand to measure a distance for; use method='lolo'")
    if len(folds) < 2:
        raise ValueError("Fewer than two ligands: no training ligand to measure against")

    rows = []
    for fold, (train, _) in enumerate(folds):
        preprocessor = fit_fold(train)
        names = list(preprocessor.get_feature_names_out())
        numeric = [i for i, name in enumerate(names) if name.startswith("num__")]
        if not numeric:
            raise ValueError(
                f"{model} encodes no numeric ligand descriptors -- its ligand block is "
                f"one-hot, which places every ligand the same distance apart, so this "
                f"distance would be noise. Use a descriptor model: selected_2, "
                f"selected_5, pc_top or pc_scores.")
        encoded = np.asarray(preprocessor.transform(X), dtype=float)[:, numeric]
        # Under LOLO a ligand's rows are exactly the test rows of its own fold, so the
        # folds double as the row index of each ligand and no group column is needed.
        centroid = {ref: encoded[test].mean(axis=0) for ref, (_, test) in zip(references, folds)}
        held_out = references[fold]
        training = {ref: vector for ref, vector in centroid.items() if ref != held_out}
        distance = {ref: float(np.linalg.norm(centroid[held_out] - vector))
                    for ref, vector in training.items()}
        nearest = min(distance, key=distance.get)
        rows.append({"fold": fold, "reference_group": held_out,
                     "centroid_distance": float(np.linalg.norm(
                         centroid[held_out] - np.mean(list(training.values()), axis=0))),
                     "nearest_distance": distance[nearest], "nearest_ligand": nearest,
                     "n_descriptors": len(numeric)})
    return pd.DataFrame(rows)


def _spearman(x, y):
    """(rho, p) for the rank correlation, or (None, None) when it cannot be formed."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return (None, None)
    result = spearmanr(x, y)
    return (_finite_or_none(result.statistic), _finite_or_none(result.pvalue))


def distance_vs_performance(runs, bundle, model, method="lolo", metric="rmse"):
    """`ligand_distances` joined to each fold's collected metric, one row per ligand.

    The metric is taken from the collected `metrics_by_split` table, so it is the same
    number every other table here reports; `prepare` builds that export if it is
    missing.

    The two Spearman correlations -- each distance definition against the metric --
    are returned in `frame.attrs["spearman"]` as `{"centroid": (rho, p), "nearest":
    (rho, p)}`. They are deliberately NOT columns: they describe the whole table, and
    repeating a constant down the rows reads as a per-fold number. `.attrs` does not
    survive every pandas operation, so read them off the frame this call returns.

    There is one point per ligand -- eight here, four for the four-ligand set -- so a
    correlation is suggestive of extrapolation behaviour and nothing more. Read its
    sign against the metric: for an error metric (rmse, mae, predictive_nll) positive
    rho is the expected "farther is worse"; for r2 or kendall_tau it is negative rho.
    """
    frame = ligand_distances(runs, bundle, model, method)
    tables, _ = load_results(prepare(runs, bundle))
    metrics = tables["metrics_by_split"]
    if metric not in metrics.columns:
        raise ValueError(f"No {metric!r} column in metrics_by_split; "
                         f"choose from {sorted(set(metrics.columns) & set(RANK_DIRECTION))}")
    part = metrics.loc[(metrics.model == model) & (metrics.method == method),
                       ["fold", "reference_group", metric]]
    if part.empty:
        raise ValueError(f"No collected {model}/{method} metrics to join; "
                         f"rebuild with prepare(runs, bundle, refresh=True)")
    merged = frame.merge(part, on=["fold", "reference_group"], how="left", validate="one_to_one")
    merged[metric] = pd.to_numeric(merged[metric], errors="coerce")
    if merged[metric].isna().any():
        raise ValueError(f"No {metric!r} for folds "
                         f"{merged.loc[merged[metric].isna(), 'reference_group'].tolist()}")
    merged.attrs["spearman"] = {
        "centroid": _spearman(merged.centroid_distance, merged[metric]),
        "nearest": _spearman(merged.nearest_distance, merged[metric])}
    return merged


def plot_distance_vs_performance(runs, bundle, model, method="lolo", metric="rmse"):
    """Each fold's metric against how far its held-out ligand sat from the training set.

    Two panels, one per distance definition, sharing the metric axis; every point is
    one held-out ligand and is labelled with it. Each panel's subtitle carries its
    Spearman rho, and the figure subtitle the number of folds behind them -- with one
    point per ligand the sign is a hint about extrapolation, not evidence.
    """
    import matplotlib.pyplot as plt
    frame = distance_vs_performance(runs, bundle, model, method, metric)
    correlation = frame.attrs["spearman"]
    panels = [("centroid_distance", "centroid", "to the training-ligand mean"),
              ("nearest_distance", "nearest", "to the nearest training ligand")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, constrained_layout=True)
    for ax, (column, key, label) in zip(axes, panels):
        ax.scatter(frame[column], frame[metric], s=36, color="tab:blue", edgecolors="none")
        for _, row in frame.iterrows():
            ax.annotate(row.reference_group, (row[column], row[metric]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, color="0.3")
        rho, p = correlation[key]
        ax.set_xlabel(f"Standardized distance {label}")
        ax.set_title("Spearman $\\rho$ = " + ("n/a" if rho is None else f"{rho:+.2f}  (p = {p:.2f})"),
                     fontsize=9, color="0.3")
    better = RANK_DIRECTION.get(metric)
    axes[0].set_ylabel(f"{metric} on the held-out ligand"
                       + (f" ({better} is better)" if better else ""))
    fig.suptitle(f"{_heading(model, method)}  |  {len(frame)} folds, correlation suggestive only")
    return fig


def review_controls(tables, title=None):
    """`title`, if given, overrides the parity plot's default "Model | method"
    heading -- a str.format template with `model`, `method`, `reference_group`
    fields, e.g. "{model} on {method}". See `_heading`."""
    # Local imports (were module level in hazel_gp/notebook.py) so this file
    # imports fine outside a notebook, and without ipywidgets installed.
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    pred = tables["predictions"]
    model = w.Dropdown(options=pred.model.unique().tolist(), description="Model:")
    method = w.Dropdown(options=pred.method.unique().tolist(), description="Method:")
    group = w.Dropdown(options=[("All", "")], description="Reference:")
    out = w.Output()

    def update_groups(change=None):
        part = pred[pred.method == method.value]
        values = [x for x in part.reference_group.unique().tolist() if x]
        group.options = [("All", "")] + [(x, x) for x in values]

    def redraw(change=None):
        with out:
            clear_output(wait=True)
            try:
                fig = plot_parity(pred, model.value, method.value, group.value or None, title=title)
                display(fig)
                plt.close(fig)
            except ValueError as exc:
                print(str(exc))
                return
            part = tables["metrics_by_split"]
            part = part[(part.model == model.value) & (part.method == method.value)]
            if group.value:
                part = part[part.reference_group == group.value]
            display(part)
    method.observe(update_groups, names="value")
    for control in (model, method, group):
        control.observe(redraw, names="value")
    update_groups()
    redraw()
    return w.VBox([w.HBox([model, method, group]), out])
