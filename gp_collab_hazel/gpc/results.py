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


MODEL_LABELS = {"pc_scores": "PC loading"}
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
    """Order models as the collector wrote them, not alphabetically as pivot/groupby would."""
    return list(dict.fromkeys(frame.model))


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
