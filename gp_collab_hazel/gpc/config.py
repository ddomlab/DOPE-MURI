"""Configuration for the GP_collab-on-DOPE runs.

`inputs/config.json` is the prepared bundle's own config and stays the authority
on how the feature sections are built. `configs/default.json` only adds what is
new here: which splits to make, and the GP settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODELS = ["ligand_ohe", "selected_2", "selected_5", "pc_top", "pc_scores"]
METHODS = ["lolo", "iid_stratified_8", "kfold_stratified_5"]
TARGET = "Product_Yield_PCT_Area_UV"
GROUP = "ligand"
# From hazel_gp/config.py; the two short ligand models are defined by these.
TWO = ["vbur_pct_boltz", "vbur_pct_min"]
FIVE = TWO + ["vbur_pct_delta", "dipolemoment_boltz", "homo_lumo_gap_eV"]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    tmp.replace(path)


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def load_config(path: str | Path) -> dict:
    cfg = read_json(path)
    unknown = set(cfg["run"]["models"]) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown feature sections: {sorted(unknown)}")
    unknown = set(cfg["run"]["methods"]) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown evaluation methods: {sorted(unknown)}")
    if cfg["gp"]["kernel"] not in ("RBF", "Matern32", "Matern52",
                                   "Tanimoto", "TanimotoRBF",
                                   "TanimotoMatern32", "TanimotoMatern52"):
        raise ValueError(f"Unknown kernel {cfg['gp']['kernel']!r}")
    if cfg["gp"]["grouping"] not in ("all", "ligand_conditions", "per_field"):
        raise ValueError(f"Unknown grouping {cfg['gp']['grouping']!r}")
    return cfg
