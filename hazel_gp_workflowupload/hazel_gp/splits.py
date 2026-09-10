"""One reproducible split registry shared by all five representations."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from .config import write_json, read_json


def validate_split(train, test, n, groups, method, reference_group=""):
    train, test = np.asarray(train), np.asarray(test)
    if (not len(train) or not len(test) or len(np.unique(train)) != len(train)
            or len(np.unique(test)) != len(test) or np.intersect1d(train, test).size
            or not np.array_equal(np.sort(np.r_[train, test]), np.arange(n))):
        raise ValueError("Split is not a complete disjoint partition of canonical rows")
    if method == "lolo":
        if set(groups[test]) != {reference_group} or reference_group in set(groups[train]):
            raise ValueError("LOLO ligand leakage")
    else:
        # With this original factorial dataset all ligand identities should be seen.
        if set(groups[test]) - set(groups[train]):
            raise ValueError("IID split has unseen ligands; revise the dataset or split protocol")
    if method == "iid_matched" and set(groups[test]) != set(groups):
        raise ValueError("Matched IID split must include every ligand in test and train")


def make_splits(reactions: pd.DataFrame, cfg: dict):
    e = cfg["evaluation"]
    n = len(reactions)
    groups = reactions["ligand"].to_numpy()
    indices = np.arange(n, dtype=np.int64)
    entries, arrays = [], {}

    def add(method, tr, te, ref="", repeat=0, seed=None, fold=0):
        validate_split(tr, te, n, groups, method, ref)
        sid = f"split_{len(entries):03d}"
        arrays[sid + "_train"] = np.asarray(tr, dtype=np.int64)
        arrays[sid + "_test"] = np.asarray(te, dtype=np.int64)
        entries.append({"split_id": sid, "method": method, "reference_group": ref,
                        "repeat": int(repeat), "fold": int(fold), "seed": seed,
                        "n_train": len(tr), "n_test": len(te)})

    for i, (tr, te) in enumerate(LeaveOneGroupOut().split(indices, groups=groups)):
        ligand = str(groups[te[0]])
        if "lolo" in e["methods"]:
            add("lolo", tr, te, ligand, fold=i)
    if "iid_matched" in e["methods"]:
        # The same fold geometry as LOLO: one disjoint fold per ligand, each sized to that
        # ligand's LOLO test set, every row tested exactly once and never repeated. Only
        # membership differs, being random instead of by ligand, so the in-distribution
        # control pools over its folds exactly the way LOLO does.
        seed = int(np.random.SeedSequence([e["seed"], 1701]).generate_state(1)[0])
        shuffled = np.random.default_rng(seed).permutation(indices)
        start = 0
        for i, size in enumerate(np.unique(groups, return_counts=True)[1]):
            te = np.sort(shuffled[start:start + int(size)])
            add("iid_matched", np.setdiff1d(indices, te), te, seed=seed, fold=i)
            start += int(size)
    if "kfold" in e["methods"]:
        kfold = KFold(n_splits=e["n_splits"], shuffle=True, random_state=e["seed"])
        for i, (tr, te) in enumerate(kfold.split(indices)):
            add("kfold", tr, te, seed=e["seed"], fold=i)
    if "holdout" in e["methods"]:
        tr, te = train_test_split(indices, test_size=e["holdout_fraction"], random_state=e["seed"])
        add("holdout", tr, te, seed=e["seed"])
    tasks = []
    for model in cfg["features"]["models"]:
        for entry in entries:
            tasks.append({"task_id": len(tasks), "model": model, **entry,
                          "fit_seed": int(e["seed"])})
    return entries, arrays, pd.DataFrame(tasks)


def save_splits(directory: Path, entries, arrays, tasks):
    write_json(directory / "splits.json", entries)
    np.savez_compressed(directory / "splits.npz", **arrays)
    tasks.to_csv(directory / "tasks.csv", index=False)


def load_split(directory: Path, split_id: str):
    entry = next(e for e in read_json(directory / "splits.json") if e["split_id"] == split_id)
    with np.load(directory / "splits.npz", allow_pickle=False) as a:
        return entry, a[split_id + "_train"], a[split_id + "_test"]


def evaluation_summary(cfg: dict, tasks=None) -> pd.DataFrame:
    desc = {"lolo": "Hold out one ligand; repeat for every ligand",
            "iid_matched": "Random partition into folds matching each LOLO test size; every row tested once",
            "kfold": f"Shuffled {cfg['evaluation']['n_splits']}-fold, seed {cfg['evaluation']['seed']}",
            "holdout": f"Random {1-cfg['evaluation']['holdout_fraction']:.0%}/{cfg['evaluation']['holdout_fraction']:.0%}, seed {cfg['evaluation']['seed']}"}
    counts = tasks.groupby("method").size().to_dict() if tasks is not None else {}
    return pd.DataFrame([{"method": m, "definition": desc[m],
                          "total_model_fits": counts.get(m, "computed after preparation")}
                         for m in cfg["evaluation"]["methods"]])
