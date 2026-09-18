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

MODELS = ["ligand_ohe", "selected_2", "selected_5", "pc_top", "pc_scores", "pc_scores_long",
          # disabled 2026-09-16 -- user no longer runs these feature sets; uncomment to restore
          # "rxnpredict_full", "rxnpredict_full_ohe",
          ]
# Non-LOLO method names encode their own fold count and whether they stratify;
# see gpc/splits.parse_method. Any kfold_<n> / *_stratified_<n> is valid.
METHODS = ["lolo", "iid_stratified_4", "iid_stratified_8",
           "kfold_stratified_5", "kfold_5"]
# Defaults for this clone's dataset; inputs/config.json overrides both.
TARGET = "yield"
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
    from .splits import parse_method
    for method in cfg["run"]["methods"]:
        if method != "lolo":
            parse_method(method)          # raises on an unusable name
    if cfg["gp"]["kernel"] not in ("RBF", "Matern32", "Matern52",
                                   "Tanimoto", "TanimotoRBF",
                                   "TanimotoMatern32", "TanimotoMatern52"):
        raise ValueError(f"Unknown kernel {cfg['gp']['kernel']!r}")
    if cfg["gp"]["grouping"] not in ("all", "ligand_conditions", "per_field"):
        raise ValueError(f"Unknown grouping {cfg['gp']['grouping']!r}")
    # Per-model GP overrides. A silently ignored override would burn hours of cluster
    # time and produce a run indistinguishable from the default one, so typos are a
    # hard error. "walltime" is the one non-gp key allowed: it is scheduling, not model.
    overrides = cfg["run"].get("model_overrides", {})
    unknown = set(overrides) - set(cfg["run"]["models"])
    if unknown:
        raise ValueError(f"model_overrides names models that are not in run.models: {sorted(unknown)}")
    for model, block in overrides.items():
        bad = set(block) - set(cfg["gp"]) - {"walltime"}
        if bad:
            raise ValueError(f"model_overrides[{model!r}] sets unknown keys {sorted(bad)}; "
                             f"only gp keys and 'walltime' are allowed")
    return cfg
