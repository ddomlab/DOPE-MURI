"""Explicit experiment configuration and small portable I/O helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

MODELS = ["ligand_ohe", "selected_5", "selected_2", "pc_top", "pc_scores"]
METHODS = ["lolo", "iid_matched", "kfold", "holdout"]
TARGET = "Product_Yield_PCT_Area_UV"
COMMON = ["Reactant_1_Short_Hand", "Reagent_1_Short_Hand",
          "Solvent_1_Short_Hand", "substrate_pair"]
LIGANDS = {"XPhos": 1, "SPhos": 3, "P(tBu)3": 8, "P(o-Tol)3": 9,
           "CataCXium A": 10, "P(Cy)3": 11, "P(Ph)3": 17, "AmPhos": 216}
TWO = ["vbur_pct_boltz", "vbur_pct_min"]
FIVE = TWO + ["vbur_pct_delta", "dipolemoment_boltz", "homo_lumo_gap_eV"]
SOURCES = {
    "reactions": {"filename": "perera_pfizer_raw_download.csv",
                  "url": "https://raw.githubusercontent.com/Paulilein/Perera2018/main/Perera2018_Data_Original.csv"},
    "descriptors": {"filename": "kraken_features_only.csv",
                    "url": "https://raw.githubusercontent.com/doyle-lab-ucla/kraken_utils/main/kraken_features_only.csv"},
    "identifiers": {"filename": "kraken_identifiers.csv",
                    "url": "https://raw.githubusercontent.com/doyle-lab-ucla/kraken_utils/main/identifiers.csv"},
}


def default_config() -> dict:
    return {
        "schema_version": 1,
        "paths": {"raw": "data_hazel/raw", "prepared": "data_hazel/prepared/group1_v1",
                  "runs": "data_hazel/runs/experiment_v1"},
        "sources": SOURCES,
        "data": {"target": TARGET, "reaction_separator": ";", "reaction_decimal": ",",
                 "excluded_solvents": ["MeOH/H2O_V2 9:1", "THF_V2"],
                 "excluded_ligands": ["None", "dppf", "dtbpf", "Xantphos"],
                 "reactant1_codes": ["1a", "1b", "1c", "1d"],
                 "reactant2_codes": ["2a", "2b", "2c"], "ligand_mapping": LIGANDS,
                 "common_categorical": COMMON, "sphere_radius_angstrom": 3.5,
                 "hartree_to_eV": 27.211386245981,
                 "expected_rows": 3072, "expected_ligands": 8},
        "features": {"models": MODELS, "pca_reference": "full_original_kraken",
                     "pca_components": 4, "top_per_pc": 3,
                     "pc_scores_form": "loading_weighted",
                     "ohe_policy": "train_ignore_unknown"},
        "evaluation": {"methods": METHODS, "seed": 42,
                       "n_splits": 5, "holdout_fraction": 0.2},
        "gp": {"kernel": "rbf", "ard": False, "learn_outputscale": False,
               "learn_noise": False, "noise_variance": 1e-6,
               "initial_lengthscale": 1.0, "lengthscale_bounds": [0.01, 1000.0],
               "learning_rate": 0.01, "max_steps": 400, "min_steps": 100,
               "patience": 60, "relative_tolerance": 1e-6, "restarts": 5,
               "cholesky_jitter": 1e-8, "solver": "cholesky",
               "cg_tolerance": 1e-5, "max_cg_iterations": 2000,
               "prediction_batch_size": 1024, "threads": 4,
               "save_checkpoints": True, "device": "auto"},
    }


def validate_config(cfg: dict) -> dict:
    """Fail early on unsupported scientific settings rather than ignoring them."""
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema_version")
    d, f, e, g = (cfg[x] for x in ("data", "features", "evaluation", "gp"))
    if d["target"] != TARGET:
        raise ValueError(f"This workflow targets {TARGET}; audit before adapting targets.")
    for values, allowed in ((f["models"], MODELS), (e["methods"], METHODS)):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError(f"Choose unique entries from {allowed}")
    if f["pca_reference"] != "full_original_kraken" or f["pca_components"] != 4:
        raise ValueError("This protocol fixes PCA to PC1-PC4 of the full Kraken reference.")
    if f["ohe_policy"] not in ("train_ignore_unknown", "declared_vocabulary"):
        raise ValueError("Unknown OHE policy")
    # Absent in bundles prepared before this option; those keep the summed component scores.
    if f.get("pc_scores_form", "component_scores") not in ("component_scores", "loading_weighted"):
        raise ValueError("pc_scores_form must be component_scores or loading_weighted")
    # iid_repeats is unused since matched IID became a non-repeating partition; bundles
    # prepared before that change still carry the key and must keep validating.
    if not 1 <= f["top_per_pc"] <= 190 or e.get("iid_repeats", 1) < 1 or e["n_splits"] < 2:
        raise ValueError("Invalid PCA or evaluation counts")
    if not 0 < e["holdout_fraction"] < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if not d["common_categorical"] or set(d["common_categorical"]) - set(COMMON):
        raise ValueError("Common inputs must be a nonempty subset of the original four fields.")
    if g["kernel"] != "rbf" or g["solver"] not in ("cholesky", "cg"):
        raise ValueError("Supported kernel=rbf; solver=cholesky or cg")
    lo, hi = g["lengthscale_bounds"]
    if not 0 < lo < g["initial_lengthscale"] < hi:
        raise ValueError("Initial length scale must be inside positive bounds")
    for k in ("noise_variance", "learning_rate", "cholesky_jitter", "cg_tolerance"):
        if g[k] <= 0:
            raise ValueError(f"{k} must be positive")
    for k in ("max_steps", "min_steps", "patience", "restarts", "threads", "prediction_batch_size"):
        if int(g[k]) != g[k] or g[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if g["min_steps"] > g["max_steps"] or g["relative_tolerance"] < 0:
        raise ValueError("Invalid stopping settings")
    if g["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return cfg


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_config(path: str | Path) -> dict:
    return validate_config(read_json(path))


def object_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versions() -> dict:
    import platform
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "gpytorch", "linear_operator"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out
