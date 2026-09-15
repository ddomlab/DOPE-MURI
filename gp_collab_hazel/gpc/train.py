"""Run one (feature section, evaluation method) pair through GP_collab's CV.

The pipeline is DOPE-MURI's per-fold preprocessor followed by GP_collab's
GPytorchMAP regressor, cross-validated by GP_collab's `cross_validate`. Each job
is one model/method pair; its folds run inside that job.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import GROUP, TARGET, write_json
from .data import load_bundle
from .features import _preprocessor, feature_frame
from .splits import PrecomputedSplits, make_folds
from .vendor.kernel_mix import GPytorchMAPsklearnRegressor
from .vendor.scoring import cross_validate_regressor, process_scores

MODEL_TYPE = "GPytorchMAP"  # GP_collab keys parallel/UQ behaviour off "gp" in this


def _ligand_columns(columns):
    """Ligand block: the numeric descriptors plus ligand identity one-hots."""
    return [c for c in columns if c.startswith("num__") or c.startswith("cat__ligand_")]


def _condition_columns(columns):
    return [c for c in columns if c.startswith("cat__") and not c.startswith("cat__ligand_")]


def _field_group(field):
    def select(columns):
        return [c for c in columns if c.startswith(f"cat__{field}_")]
    return select


def feature_groups(grouping: str, prepared: dict) -> dict:
    """Map encoded columns onto GP_collab kernel groups.

    For an RBF kernel on disjoint column sets, product mixing is identical to one
    RBF whose lengthscales are shared within each group, so this choice only sets
    how many lengthscales the kernel has. "all" with ard=False is the single
    isotropic RBF over every encoded column that hazel_gp ran.
    """
    if grouping == "all":
        return {"fp_all": "__all__"}
    if grouping == "ligand_conditions":
        return {"fp_ligand": _ligand_columns, "fp_conditions": _condition_columns}
    if grouping == "per_field":
        groups = {"fp_ligand": _ligand_columns}
        for field in prepared["data"]["common_categorical"]:
            groups[f"fp_{field}"] = _field_group(field)
        return groups
    raise ValueError(f"Unknown grouping {grouping!r}")


def build_regressor(cfg: dict, prepared: dict, seed: int) -> GPytorchMAPsklearnRegressor:
    gp = cfg["gp"]
    return GPytorchMAPsklearnRegressor(
        feat_group=feature_groups(gp["grouping"], prepared),
        kernel_type={"fp": gp["kernel"], "count": gp["kernel"]},
        kernel_mixing_method=gp["mixing_method"],
        ard=gp["ard"],
        dtype=gp["dtype"],
        noise=gp["noise"],
        noise_floor=gp["noise_floor"],
        outputscale=gp["outputscale"],
        train_jitter=gp["train_jitter"],
        predict_jitter=gp["predict_jitter"],
        restarts=gp["restarts"],
        n_epoch=gp["n_epochs"],
        lr=gp["learning_rate"],
        prior=gp["prior"],
        normalize_y=gp["normalize_y"],
        random_state=seed,
        progbar=False,
        use_cuda=gp["use_cuda"],
    )


def run_task(bundle, runs, model: str, method: str, cfg: dict) -> Path:
    import torch

    reactions, ligands, reference, prepared = load_bundle(bundle)
    seed = cfg["run"]["seed"]

    X = feature_frame(reactions, ligands, model, prepared, reference)
    y = reactions[TARGET].to_numpy(dtype=float)

    folds, references = make_folds(reactions, method, seed, cfg["run"]["stratify"])
    fold_of_row = np.empty(len(reactions), dtype=np.int64)
    for i, (_, test) in enumerate(folds):
        fold_of_row[test] = i

    preprocessor = _preprocessor(X, model, prepared, reference)
    regressor = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", build_regressor(cfg, prepared, seed)),
    ])
    regressor.set_output(transform="pandas")

    use_gpu = cfg["gp"]["use_cuda"] and torch.cuda.is_available()
    torch.set_num_threads(int(cfg["run"]["threads"]))

    start = time.time()
    scores, predictions = cross_validate_regressor(
        regressor,
        MODEL_TYPE,
        X, y,
        PrecomputedSplits(folds),
        UQ=True,
        return_ls=True,
        n_jobs=1 if use_gpu else -1,
    )
    scores = process_scores({seed: dict(scores)})
    scores["run_time_sec"] = round(time.time() - start, 3)

    out = Path(runs) / f"{model}__{method}"
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "row_id": reactions["row_id"],
        "ligand": reactions[GROUP],
        "fold": fold_of_row,
        "reference_group": [references[f] for f in fold_of_row],
        "y_true": y,
        "y_pred": predictions["y_pred"],
        "y_std": predictions["y_std"],
    }).to_csv(out / "predictions.csv", index=False)

    write_json(out / "scores.json", scores)
    write_json(out / "meta.json", {
        "model": model, "method": method, "seed": seed,
        "stratify": cfg["run"]["stratify"], "n_folds": len(folds),
        "n_rows": len(reactions), "config": cfg,
        "device": "cuda" if use_gpu else "cpu",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    })
    print(f"{model}/{method}: r2={scores.get('r2_avg'):.3f} "
          f"rmse={scores.get('rmse_avg'):.3f} -> {out}")
    return out
