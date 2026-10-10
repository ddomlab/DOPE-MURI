"""Load the prepared rxnpredict bundle."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import GROUP, TARGET, read_json
from .features import ReferencePCA


def load_bundle(bundle: str | Path):
    """Return (reactions, ligands, reference_pca, prepared_config, rxn_features).

    `rxn_features` is the published DFT descriptor table, or None for a bundle
    that does not ship one.
    """
    bundle = Path(bundle)
    prepared = read_json(bundle / "config.json")
    target = prepared["data"].get("target", TARGET)
    group = prepared["data"].get("group", GROUP)

    reactions = pd.read_csv(bundle / "reactions.csv", keep_default_na=False)
    reactions[target] = pd.to_numeric(reactions[target], errors="raise")
    ligands = pd.read_csv(bundle / "ligand_features.csv")

    expected = prepared["data"]["expected_rows"]
    if len(reactions) != expected:
        raise ValueError(f"{len(reactions)} rows in bundle, config expects {expected}")
    if reactions[group].nunique() != prepared["data"]["expected_ligands"]:
        raise ValueError("Unexpected ligand count in bundle")
    if reactions[target].isna().any():
        raise ValueError("Missing target values in bundle")

    path = bundle / "rxnpredict_features.csv"
    rxn_features = pd.read_csv(path) if path.exists() else None
    if rxn_features is not None and list(rxn_features.row_id) != list(reactions.row_id):
        raise ValueError("rxnpredict_features.csv is not aligned with reactions.csv")
    return reactions, ligands, ReferencePCA.load(bundle), prepared, rxn_features
