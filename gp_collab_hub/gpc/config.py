"""Configuration for the hub.

Two configs, and they answer different questions:

* `<bundle>/config.json` -- written by a dataset's intake notebook. It is the
  authority on WHAT the dataset is: its target column, its group column, its
  one-hot condition fields, its row and group counts. Nothing here overrides it.
* `configs/<tag>.json` (or `.yaml`) -- written by `build_run.ipynb`. It is the
  authority on WHAT TO RUN: which feature sections, which evaluation methods,
  the GP hyperparameters, and the scheduling block the submit scripts read.

Keeping them apart is what lets one engine serve every dataset: swap the bundle
and nothing in the run config has to change, and vice versa.

YAML is an authoring convenience only. `build_run.ipynb` resolves whichever
format you wrote into JSON before it goes in the zip, so the cluster never needs
PyYAML installed to start a job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Canonical feature sections. The group one-hot appears under three keys because
# the three original projects named it three ways (`ligand_ohe` in Perera and
# Ahneman, `catalyst_ohe` in Gesmundo); all three mean "one-hot the group column"
# and `features.py` resolves them to the same thing. New bundles should use
# `group_ohe`, but the old names are kept so runs collected before the hub still
# load and still line up in the comparison notebook.
GROUP_OHE_KEYS = ("group_ohe", "ligand_ohe", "catalyst_ohe")

# Fallbacks for a bundle whose config.json omits them. Every bundle the hub
# writes states both explicitly, so these only matter when reading an old one.
TARGET = "Product_Yield_PCT_Area_UV"
GROUP = "ligand"

# The two short descriptor sections, from hazel_gp/config.py. Identical in all
# three original projects, which is what makes the sections comparable across
# datasets -- do not vary them per dataset.
TWO = ["vbur_pct_boltz", "vbur_pct_min"]
FIVE = TWO + ["vbur_pct_delta", "dipolemoment_boltz", "homo_lumo_gap_eV"]

# Kraken's minimum electrostatic potential near phosphorus, Boltzmann-averaged
# over conformers -- the electronic counterpart to the steric buried volumes.
# It is the only Vmin potential in the reference table (`vmin_r_boltz` beside it
# is the P-to-Vmin distance, a geometry, not a potential).
VMIN = "vmin_vmin_boltz"

# Every section defined by an explicit descriptor list. Adding an entry here is
# all it takes to make a new two-descriptor section runnable: the CLI, the
# validator, the trainer and the review path all read this one dict.
#
# The two `vbur_*_vmin` pairs swap one buried volume for Vmin, so each is one
# steric plus one electronic descriptor rather than `selected_2`'s two sterics.
# Read against `selected_2` they answer whether the second buried volume is
# carrying information that an electronic descriptor would carry better.
SELECTED_SETS = {
    "selected_2": TWO,
    "selected_5": FIVE,
    "vbur_min_vmin": ["vbur_pct_min", VMIN],
    "vbur_boltz_vmin": ["vbur_pct_boltz", VMIN],
}

MODELS = [*GROUP_OHE_KEYS, *SELECTED_SETS, "pc_top", "pc_scores", "pc_scores_long"]
# Sections whose descriptor list `run.feature_columns` may redefine.
REDEFINABLE = (*SELECTED_SETS, "pc_top")

KERNELS = ("RBF", "Matern32", "Matern52", "Tanimoto", "TanimotoRBF",
           "TanimotoMatern32", "TanimotoMatern52")
# "ligand_conditions" and "catalyst_reagents" are the two legacy spellings of
# "group_conditions"; train.feature_groups accepts all three.
GROUPINGS = ("all", "group_conditions", "ligand_conditions", "catalyst_reagents",
             "per_field")


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
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def read_any(path: str | Path) -> Any:
    """Read a JSON or YAML config by extension.

    PyYAML is imported only when a .yaml/.yml path is actually handed over, so a
    JSON-only environment (the cluster's, for instance) never needs it. The error
    names the fix rather than falling back to a half-parse, because a config read
    wrong is a run that burns GPU hours and looks fine afterwards.
    """
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                f"{path.name} is YAML but PyYAML is not installed. Either "
                f"`pip install pyyaml`, or hand this function the .json config "
                f"instead -- build_run.ipynb writes both."
            ) from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return read_json(path)


def write_yaml(path: str | Path, obj: Any) -> None:
    """Companion YAML copy of a run config, for hand editing. Optional."""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(json.dumps(obj, default=_jsonable)),
                                   sort_keys=False, default_flow_style=False),
                    encoding="utf-8")


def load_config(path: str | Path) -> dict:
    """Read and validate a run config. JSON or YAML; see `read_any`."""
    return validate_config(read_any(path))


def validate_config(cfg: dict) -> dict:
    from .splits import parse_method

    unknown = set(cfg["run"]["models"]) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown feature sections: {sorted(unknown)}; "
                         f"known sections are {MODELS}")
    for method in cfg["run"]["methods"]:
        if method != "lolo":
            parse_method(method)          # raises on an unusable name
    if cfg["gp"]["kernel"] not in KERNELS:
        raise ValueError(f"Unknown kernel {cfg['gp']['kernel']!r}; expected one of {KERNELS}")
    if cfg["gp"]["grouping"] not in GROUPINGS:
        raise ValueError(f"Unknown grouping {cfg['gp']['grouping']!r}; expected one of {GROUPINGS}")
    # Per-model GP overrides. A silently ignored override would burn hours of
    # cluster time and produce a run indistinguishable from the default one, so
    # typos are a hard error. "walltime" is the one non-gp key allowed: it is
    # scheduling, not model.
    overrides = cfg["run"].get("model_overrides", {})
    unknown = set(overrides) - set(cfg["run"]["models"])
    if unknown:
        raise ValueError(f"model_overrides names models that are not in run.models: "
                         f"{sorted(unknown)}")
    for model, block in overrides.items():
        bad = set(block) - set(cfg["gp"]) - {"walltime"}
        if bad:
            raise ValueError(f"model_overrides[{model!r}] sets unknown keys {sorted(bad)}; "
                             f"only gp keys and 'walltime' are allowed")
    # Redefining a section's descriptor list is allowed but breaks comparability
    # with every other dataset, so only the three sections that have a list at
    # all may be redefined, and each must be a non-empty list of column names.
    columns = cfg["run"].get("feature_columns") or {}
    bad = set(columns) - set(REDEFINABLE)
    if bad:
        raise ValueError(f"run.feature_columns can only redefine "
                         f"{', '.join(REDEFINABLE)}; got {sorted(bad)}")
    for section, names in columns.items():
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError(f"run.feature_columns[{section!r}] must be a non-empty "
                             f"list of descriptor column names")
    return cfg


def bundle_target(prepared: dict) -> str:
    return prepared["data"].get("target", TARGET)


def bundle_group(prepared: dict) -> str:
    return prepared["data"].get("group", GROUP)


def bundle_name(prepared: dict) -> str:
    """Short slug for this dataset, used in figure titles and folder names.

    `display_name` is what a human should see; `dataset` is the machine key the
    intake notebook wrote. Falling back to "dataset" means an old bundle still
    produces a labelled figure rather than a blank one.
    """
    return prepared.get("display_name") or prepared.get("dataset") or "dataset"
