"""Load the prepared Cernak bundle.

One table. Every column the model sees is in `reactions.csv`, and `config.json`
is the authority on which of those columns is the target, which is the group and
which are the one-hot condition fields. Nothing else in `inputs/` is read.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import ReferencePCA
from .config import GROUP, TARGET, read_json


def load_bundle(bundle: str | Path):
    """Return (reactions, ligands, reference_pca, prepared_config)."""
    bundle = Path(bundle)
    prepared = read_json(bundle / "config.json")
    target = prepared["data"].get("target", TARGET)
    group = prepared["data"].get("group", GROUP)

    reactions = pd.read_csv(bundle / "reactions.csv", keep_default_na=False)
    reactions[target] = pd.to_numeric(reactions[target], errors="raise")

    expected = prepared["data"]["expected_rows"]
    if len(reactions) != expected:
        raise ValueError(f"{len(reactions)} rows in bundle, config expects {expected}")
    if reactions[group].nunique() != prepared["data"]["expected_groups"]:
        raise ValueError("Unexpected catalyst count in bundle")
    if reactions[target].isna().any():
        raise ValueError("Missing target values in bundle")
    missing = [c for c in prepared["data"]["common_categorical"] if c not in reactions]
    if missing:
        raise ValueError(f"Bundle is missing condition columns: {missing}")
    ligands = pd.read_csv(bundle / "ligand_features.csv")
    return reactions, ligands, ReferencePCA.load(bundle), prepared
