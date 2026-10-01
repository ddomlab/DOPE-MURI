"""Gather finished tasks into three tables for local review."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .config import read_json

METRICS = ["r2", "rmse", "mae", "nll", "ece", "cdf_ama", "cvpp_ama", "Cv", "sharpness", "RUSC"]


def _pooled(frame: pd.DataFrame) -> dict:
    y, p = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    return {"n": len(frame), "pooled_r2": r2_score(y, p),
            "pooled_rmse": root_mean_squared_error(y, p),
            "pooled_mae": mean_absolute_error(y, p)}


def collect(runs: str | Path) -> dict[str, pd.DataFrame]:
    runs = Path(runs)
    tasks = sorted(d for d in runs.iterdir() if (d / "scores.json").exists())
    if not tasks:
        raise FileNotFoundError(f"No finished tasks under {runs}")

    summary, predictions = [], []
    for task in tasks:
        meta = read_json(task / "meta.json")
        scores = read_json(task / "scores.json")
        frame = pd.read_csv(task / "predictions.csv")
        frame.insert(0, "method", meta["method"])
        frame.insert(0, "model", meta["model"])
        predictions.append(frame)

        row = {"model": meta["model"], "method": meta["method"],
               "n_folds": meta["n_folds"], "device": meta["device"],
               "run_time_sec": scores.get("run_time_sec")}
        # Fold mean/std, as GP_collab's process_scores computed them.
        for metric in METRICS:
            row[f"{metric}_mean"] = scores.get(f"{metric}_avg")
            row[f"{metric}_std"] = scores.get(f"{metric}_stdev")
        row.update(_pooled(frame))
        summary.append(row)

    predictions = pd.concat(predictions, ignore_index=True)
    lolo = predictions[predictions["method"] == "lolo"]
    by_catalyst = (
        pd.DataFrame([{"model": model, "catalyst": catalyst, **_pooled(group)}
                      for (model, catalyst), group in lolo.groupby(["model", "reference_group"])])
        if len(lolo) else pd.DataFrame()
    )
    return {"summary": pd.DataFrame(summary), "predictions": predictions,
            "lolo_by_catalyst": by_catalyst}


def write_tables(runs: str | Path, out: str | Path | None = None) -> Path:
    runs = Path(runs)
    out = Path(out) if out else runs / "collected"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in collect(runs).items():
        if len(table):
            table.to_csv(out / f"{name}.csv", index=False)
    print(f"collected {runs} -> {out}")
    return out
