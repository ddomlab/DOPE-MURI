"""Integrity-checked collection, uncertainty metrics, and local review plots."""
from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .config import TARGET, read_json, write_json, file_hash, object_hash
from .data import verify_bundle
from .splits import load_split


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


def task_statuses(bundle, runs) -> pd.DataFrame:
    """Read-only status inspection, including missing jobs; no result aggregation."""
    bundle, runs = Path(bundle), Path(runs)
    verify_bundle(bundle)
    tasks = pd.read_csv(bundle / "tasks.csv", keep_default_na=False)
    rows = []
    for task in tasks.to_dict("records"):
        path = runs / "tasks" / f"task_{task['task_id']:04d}" / "status.json"
        meta = read_json(path) if path.exists() else {"state": "missing"}
        rows.append({**task, "state": meta["state"], "error": meta.get("error", "")})
    return pd.DataFrame(rows)


def collect_results(bundle, runs, allow_partial=False):
    bundle, runs = Path(bundle), Path(runs)
    manifest = verify_bundle(bundle)
    run = read_json(runs / "run.json")
    if run["bundle_id"] != manifest["bundle_id"]:
        raise ValueError("Results and input bundle do not match")
    if object_hash({k: v for k, v in run.items() if k not in ("run_id", "n_tasks")}) != run["run_id"]:
        raise ValueError("Run metadata changed")
    tasks = pd.read_csv(bundle / "tasks.csv", keep_default_na=False)
    reactions = pd.read_csv(bundle / "reactions.csv", keep_default_na=False)
    status_table = task_statuses(bundle, runs)
    missing = status_table.loc[status_table.state != "completed", "task_id"].astype(int).tolist()
    if missing and not allow_partial:
        raise RuntimeError(f"{len(missing)} tasks incomplete; first IDs: {missing[:20]}. "
                           "Run status, finish/retry these tasks, or use --allow-partial for a labeled pilot report.")
    predictions, metric_rows = [], []
    for task in tasks.to_dict("records"):
        if task["task_id"] in missing:
            continue
        directory = runs / "tasks" / f"task_{task['task_id']:04d}"
        status = read_json(directory / "status.json")
        if status["run_id"] != run["run_id"] or status["task"] != task:
            raise ValueError(f"Task identity mismatch: {task['task_id']}")
        for name, digest in status["output_hashes"].items():
            if file_hash(directory / name) != digest:
                raise ValueError(f"Corrupt task output: {directory / name}")
        pred = pd.read_csv(directory / "predictions.csv", keep_default_na=False)
        _, _, te = load_split(bundle, task["split_id"])
        expected = reactions.iloc[te]
        if (len(pred) != len(te) or pred.row_id.duplicated().any()
                or pred.row_id.tolist() != expected.row_id.tolist()
                or pred.ligand.tolist() != expected.ligand.tolist()
                or not np.array_equal(pred.test_index.to_numpy(), te)
                or not np.allclose(pred.y_true, expected[TARGET], atol=1e-10, rtol=1e-12)):
            raise ValueError(f"Prediction rows/targets do not match split: {task['task_id']}")
        for key in ("task_id", "model", "split_id", "method", "reference_group", "repeat", "fold"):
            if not pred[key].eq(task[key]).all():
                raise ValueError(f"Prediction metadata changed: {key}")
        scores = regression_metrics(pred.y_true, pred.y_pred, pred.predictive_std)
        pred["pit"] = norm.cdf((pred.y_true - pred.y_pred) / pred.predictive_std)
        for coverage in (50, 80, 95):
            q = norm.ppf((1 + coverage / 100) / 2)
            pred[f"lower_{coverage}"] = pred.y_pred - q * pred.predictive_std
            pred[f"upper_{coverage}"] = pred.y_pred + q * pred.predictive_std
        predictions.append(pred)
        fit = read_json(directory / "fit.json")
        metric_rows.append({**task, **scores, "n_features": fit["n_features"],
                            "fit_seconds": fit["fit_seconds"],
                            "selected_stop_reason": fit["restart_summaries"][fit["selected_restart"]]["stop_reason"]})
    if not predictions:
        raise RuntimeError("No completed tasks to collect")
    pred = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    score_names = [k for k in regression_metrics([0., 1.], [0., 1.], [1., 1.]) if k != "n_test"]
    summary = []
    for (model, method), part in metrics.groupby(["model", "method"], sort=False):
        expected_count = int(((tasks.model == model) & (tasks.method == method)).sum())
        common = {"model": model, "method": method, "completed_fits": len(part),
                  "expected_fits": expected_count, "complete_method": len(part) == expected_count,
                  "run_complete": not missing}
        average = {k: _finite_or_none(pd.to_numeric(part[k], errors="coerce").mean()) for k in score_names}
        summary.append({**common, "aggregation": "mean_split_metrics", **average})
        # Every method is now a disjoint partition of the canonical rows, matched IID
        # included, so out-of-fold predictions can be pooled without repeating a reaction.
        p = pred.loc[(pred.model == model) & (pred.method == method)]
        if p.row_id.duplicated().any():
            raise ValueError("Unexpected repeated rows in pooled out-of-fold predictions")
        summary.append({**common, "aggregation": "pooled_predictions",
                        **regression_metrics(p.y_true, p.y_pred, p.predictive_std)})
    tables = {"predictions": pred, "metrics_by_split": metrics, "summary": pd.DataFrame(summary),
              "task_status": status_table}
    dest = runs / "collected"
    dest.mkdir(exist_ok=True)
    for name, frame in tables.items():
        temp = dest / (name + ".tmp.csv")
        frame.to_csv(temp, index=False)
        temp.replace(dest / (name + ".csv"))
    write_json(dest / "collection.json", {"run_id": run["run_id"], "bundle_id": manifest["bundle_id"],
               "complete": not missing, "expected_tasks": len(tasks), "completed_tasks": len(metric_rows),
               "incomplete_task_ids": missing,
               "iid_pooling": "Matched IID is a non-repeating partition, pooled like LOLO.",
               "output_hashes": {name + ".csv": file_hash(dest / (name + ".csv")) for name in tables}})
    return tables


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


def _heading(model, method, reference_group=None) -> str:
    parts = [model_label(model), method_label(method)]
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


def plot_parity(predictions, model, method="lolo", reference_group=None, split_id=None):
    """Observed against predicted yield. Points are not separated by ligand."""
    import matplotlib.pyplot as plt
    p = _select(predictions, model, method, reference_group, split_id)
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    ax.scatter(p.y_true, p.y_pred, s=14, alpha=0.65, color="tab:blue", edgecolors="none")
    lo, hi = min(0, p.y_pred.min()), max(100, p.y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Observed yield (%)", ylabel="Predicted yield (%)")
    fig.suptitle(_heading(model, method, reference_group))
    ax.set_title(_score_line(p.y_true, p.y_pred), fontsize=9, color="0.3")
    return fig


def plot_mean_trend(predictions, model, method="lolo", reference_group=None,
                    bins=10, band="empirical"):
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
    fig.suptitle(_heading(model, method, reference_group))
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


def export_result_archive(runs, destination, include_checkpoints=False):
    runs, dst = Path(runs).resolve(), Path(destination).resolve()
    if dst.exists():
        raise FileExistsError(dst)
    load_results(runs)  # Require a collected, hash-verified report.
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(runs.rglob("*")):
            if not p.is_file() or p == dst or p == tmp or p.name.startswith("."):
                continue
            if p.name in ("model.pt", "preprocessor.joblib") and not include_checkpoints:
                continue
            z.write(p, str(Path(runs.name) / p.relative_to(runs)))
    tmp.replace(dst)
    return dst
