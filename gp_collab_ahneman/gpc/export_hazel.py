"""Export a run into the schema DOPE-MURI's `02_review_results.ipynb` reads.

`hazel_gp.results.load_results` wants `<runs>/collected/` holding predictions,
metrics_by_split, summary and task_status CSVs plus a hash-verified
collection.json. This writes exactly that from the per-task output here, so the
original notebook works unchanged: point its RUNS at the directory this prints.

`regression_metrics` below is copied verbatim from hazel_gp/results.py, so every
number in the exported tables is computed by that project's own definitions
rather than re-derived here. GP_collab's own UQ metrics (ece, cdf_ama, Cv, ...)
stay in each task's scores.json; they have no column in this schema.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .config import GROUP, TARGET, read_json, write_json
from .data import load_bundle
from .features import feature_frame
from .splits import make_folds

# GP_collab trains a fixed epoch budget with no plateau test, unlike hazel_gp.
# Say so rather than reusing one of its stop reasons.
STOP_REASON = "fixed_epoch_budget_no_plateau_test"

# Only labels change; the folds are still ligand-stratified, not randomly drawn.
# Only labels change; the folds stay as their own names describe them.
# kfold_5 keeps its own name so it is never confused with the stratified one.
HAZEL_METHOD_NAMES = {"iid_stratified_4": "iid_matched",
                      "iid_stratified_8": "iid_matched",
                      "kfold_stratified_5": "kfold"}


def file_hash(path) -> str:
    """Same construction as hazel_gp.config.file_hash."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def _registry(reactions, cfg):
    """Rebuild the split/task registry hazel_gp's bundle would have held."""
    splits, index = {}, 0
    for method in cfg["run"]["methods"]:
        folds, references = make_folds(reactions, method, cfg["run"]["seed"],
                                       cfg["run"]["stratify"])
        for fold, ((train, test), reference) in enumerate(zip(folds, references)):
            splits[(method, fold)] = {
                "split_id": f"split_{index:03d}", "method": method,
                "reference_group": reference, "repeat": 0, "fold": fold,
                "seed": cfg["run"]["seed"], "n_train": len(train), "n_test": len(test),
                "fit_seed": cfg["run"]["seed"]}
            index += 1
    tasks = []
    for model in cfg["run"]["models"]:          # model-major, as hazel_gp ordered them
        for entry in splits.values():
            tasks.append({"task_id": len(tasks), "model": model, **entry})
    return splits, pd.DataFrame(tasks)


def _feature_width(reactions, ligands, prepared, reference, model: str, rxn_features=None) -> int:
    """Width of the frame `train.py` fits on, for runs whose meta.json predates
    it being recorded there."""
    return int(feature_frame(reactions, ligands, model, prepared, reference,
                             rxn_features).shape[1])


def export(bundle, runs, out=None, hazel_method_names: bool = False) -> Path:
    bundle, runs = Path(bundle), Path(runs)
    reactions, ligands, reference, prepared, rxn_features = load_bundle(bundle)
    widths: dict = {}

    def _n_features(meta, model):
        if "n_features" in meta:
            return meta["n_features"]
        if model not in widths:
            widths[model] = _feature_width(reactions, ligands, prepared, reference, model,
                                          rxn_features)
        return widths[model]

    cfg = None
    for directory in sorted(runs.iterdir()):
        if (directory / "meta.json").exists():
            cfg = read_json(directory / "meta.json")["config"]
            break
    if cfg is None:
        raise FileNotFoundError(f"No finished tasks under {runs}")

    splits, tasks = _registry(reactions, cfg)
    manifest = read_json(bundle / "manifest.json")

    predictions, metric_rows, statuses, missing = [], [], [], []
    for task in tasks.to_dict("records"):
        directory = runs / f"{task['model']}__{task['method']}"
        if not (directory / "predictions.csv").exists():
            missing.append(task["task_id"])
            statuses.append({**task, "state": "missing", "error": ""})
            continue
        statuses.append({**task, "state": "completed", "error": ""})

        meta = read_json(directory / "meta.json")
        scores = read_json(directory / "scores.json")[str(cfg["run"]["seed"])]
        frame = pd.read_csv(directory / "predictions.csv", keep_default_na=False)
        frame[TARGET] = pd.to_numeric(frame["y_true"])

        # Rows sit in canonical reaction order, so position is the test index.
        fold = frame.index[frame["fold"] == task["fold"]].to_numpy()
        if len(fold) != task["n_test"]:
            raise ValueError(f"Task {task['task_id']}: {len(fold)} rows, expected {task['n_test']}")
        part = frame.loc[fold]
        if not np.allclose(part["y_true"], reactions.iloc[fold][TARGET], atol=1e-10):
            raise ValueError(f"Task {task['task_id']}: targets do not match the bundle")

        out_frame = pd.DataFrame({
            "row_id": reactions.iloc[fold]["row_id"].to_numpy(),
            # hazel_gp's schema carries Reaction_No. The Perera bundle has that
            # column; this one identifies wells by plate/row/col instead, so fall
            # back to the canonical 1-based row number, which is what row_id
            # encodes ("reaction:N") and so keeps the two consistent.
            "Reaction_No": (reactions.iloc[fold]["Reaction_No"].to_numpy()
                            if "Reaction_No" in reactions.columns else fold + 1),
            "ligand": reactions.iloc[fold][GROUP].to_numpy(),
            "test_index": fold,
            "task_id": task["task_id"], "model": task["model"],
            "split_id": task["split_id"], "method": task["method"],
            "reference_group": task["reference_group"], "repeat": task["repeat"],
            "fold": task["fold"],
            "y_true": part["y_true"].to_numpy(),
            "y_pred": part["y_pred"].to_numpy(),
            # GP_collab predicts through the likelihood, so the only std it has
            # is the predictive one (noise included). Keep hazel_gp's column so
            # the schema still matches, but leave it empty rather than pass the
            # predictive std off as a noise-free latent std.
            "latent_std": np.nan,
            "predictive_std": part["y_std"].to_numpy()})

        score = regression_metrics(out_frame.y_true, out_frame.y_pred, out_frame.predictive_std)
        out_frame["pit"] = norm.cdf(
            (out_frame.y_true - out_frame.y_pred) / out_frame.predictive_std)
        for coverage in (50, 80, 95):
            q = norm.ppf((1 + coverage / 100) / 2)
            out_frame[f"lower_{coverage}"] = out_frame.y_pred - q * out_frame.predictive_std
            out_frame[f"upper_{coverage}"] = out_frame.y_pred + q * out_frame.predictive_std
        predictions.append(out_frame)

        metric_rows.append({**task, **score,
                            "n_features": _n_features(meta, task["model"]),
                            "fit_seconds": scores["test_run_time_sec"][task["fold"]],
                            "selected_stop_reason": STOP_REASON})

    if not predictions:
        raise RuntimeError(f"No finished tasks under {runs}")
    pred = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    score_names = [k for k in regression_metrics([0., 1.], [0., 1.], [1., 1.]) if k != "n_test"]
    summary = []
    for (model, method), part in metrics.groupby(["model", "method"], sort=False):
        expected = int(((tasks.model == model) & (tasks.method == method)).sum())
        common = {"model": model, "method": method, "completed_fits": len(part),
                  "expected_fits": expected, "complete_method": len(part) == expected,
                  "run_complete": not missing}
        summary.append({**common, "aggregation": "mean_split_metrics",
                        **{k: _finite_or_none(pd.to_numeric(part[k], errors="coerce").mean())
                           for k in score_names}})
        pooled = pred.loc[(pred.model == model) & (pred.method == method)]
        if pooled.row_id.duplicated().any():
            raise ValueError(f"{model}/{method} repeats rows in the pooled predictions")
        summary.append({**common, "aggregation": "pooled_predictions",
                        **regression_metrics(pooled.y_true, pooled.y_pred, pooled.predictive_std)})

    tables = {"predictions": pred, "metrics_by_split": metrics,
              "summary": pd.DataFrame(summary), "task_status": pd.DataFrame(statuses)}
    if hazel_method_names:
        for frame in tables.values():
            if "method" in frame:
                frame["method"] = frame["method"].replace(HAZEL_METHOD_NAMES)

    # load_results(RUNS) reads RUNS/"collected", so the tables must sit in a
    # directory of exactly that name and the notebook points at its parent.
    root = Path(out) if out else runs / "hazel_export"
    destination = root / "collected"
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    write_json(destination / "collection.json", {
        "run_id": manifest["config_hash"], "bundle_id": manifest["bundle_id"],
        "complete": not missing, "expected_tasks": len(tasks),
        "completed_tasks": len(metric_rows), "incomplete_task_ids": missing,
        "iid_pooling": "Every method is a non-repeating partition, pooled like LOLO.",
        "produced_by": "gpc export-hazel",
        "output_hashes": {f"{name}.csv": file_hash(destination / f"{name}.csv")
                          for name in tables}})

    if missing:
        print(f"WARNING: {len(missing)} of {len(tasks)} tasks missing; "
              f"exported as an incomplete run. Task ids: {missing[:20]}")
    print(f"exported {len(metric_rows)} fits -> {destination}")
    print(f'in 02_review_results.ipynb set:  RUNS = Path("{root}")')
    return root
