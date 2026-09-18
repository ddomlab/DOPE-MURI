"""Load the prepared DOPE-MURI bundle."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import GROUP, TARGET, read_json
from .features import ReferencePCA


def load_bundle(bundle: str | Path):
    """Return (reactions, ligands, reference_pca, prepared_config)."""
    bundle = Path(bundle)
    prepared = read_json(bundle / "config.json")
    # keep_default_na=False matches hazel_gp/data.py: the literal "None" base
    # condition is a real category, not a missing value.
    reactions = pd.read_csv(bundle / "reactions.csv", keep_default_na=False)
    reactions[TARGET] = pd.to_numeric(reactions[TARGET], errors="raise")
    ligands = pd.read_csv(bundle / "ligand_features.csv")

    expected = prepared["data"]["expected_rows"]
    if len(reactions) != expected:
        raise ValueError(f"{len(reactions)} rows in bundle, config expects {expected}")
    if reactions[GROUP].nunique() != prepared["data"]["expected_ligands"]:
        raise ValueError("Unexpected ligand count in bundle")
    if reactions[TARGET].isna().any():
        raise ValueError("Missing target values in bundle")
    return reactions, ligands, ReferencePCA.load(bundle), prepared
