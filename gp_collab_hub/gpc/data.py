"""Load a prepared bundle, whichever dataset wrote it.

A bundle is a directory holding:

    config.json          what the dataset is -- target, group, condition fields,
                         expected row and group counts
    reactions.csv        one row per reaction/well; carries row_id, the group
                         column, the condition columns, the target and kraken_id
    ligand_features.csv  Kraken descriptors keyed by kraken_id, for the groups
                         this dataset uses (or the whole reference; both work)
    pca_reference.json   the frozen reference PCA, copied never refitted, so
    pca_reference.npz    PC1..PC4 mean the same axes in every dataset
    ligand_mapping.csv   group name -> kraken_id, for figures and provenance

Everything the model sees is in `reactions.csv` and `ligand_features.csv`, and
`config.json` is the authority on which column plays which role. Nothing here
hardcodes a dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import bundle_group, bundle_target, read_json
from .features import ReferencePCA


def load_bundle(bundle: str | Path):
    """Return (reactions, ligands, reference_pca, prepared_config)."""
    bundle = Path(bundle)
    prepared = read_json(bundle / "config.json")
    target = bundle_target(prepared)
    group = bundle_group(prepared)

    # keep_default_na=False matches hazel_gp/data.py: the literal "None" base
    # condition is a real category in several of these screens, not a missing
    # value, and pandas would otherwise silently turn it into NaN.
    reactions = pd.read_csv(bundle / "reactions.csv", keep_default_na=False)
    reactions[target] = pd.to_numeric(reactions[target], errors="raise")

    expected = prepared["data"]["expected_rows"]
    if len(reactions) != expected:
        raise ValueError(f"{len(reactions)} rows in bundle, config expects {expected}")
    # Old bundles say expected_ligands, newer ones expected_groups. Accept either
    # rather than make every existing bundle a migration.
    n_expected = prepared["data"].get("expected_groups",
                                      prepared["data"].get("expected_ligands"))
    if n_expected is not None and reactions[group].nunique() != n_expected:
        raise ValueError(f"{reactions[group].nunique()} distinct {group}s in bundle, "
                         f"config expects {n_expected}")
    if reactions[target].isna().any():
        raise ValueError(f"Missing {target} values in bundle")
    missing = [c for c in prepared["data"]["common_categorical"] if c not in reactions]
    if missing:
        raise ValueError(f"Bundle is missing condition columns: {missing}")
    if "kraken_id" not in reactions:
        raise ValueError("reactions.csv has no kraken_id column to join descriptors on")

    ligands = pd.read_csv(bundle / "ligand_features.csv")
    unmapped = sorted(set(reactions.kraken_id) - set(ligands.kraken_id))
    if unmapped:
        raise ValueError(f"kraken_ids in reactions.csv with no descriptor row: {unmapped}")
    return reactions, ligands, ReferencePCA.load(bundle), prepared


def group_mapping(bundle: str | Path) -> pd.DataFrame:
    """`ligand_mapping.csv` as (group_name, kraken_id), whatever its first column
    happens to be called. Figures label points by name and join on the id, so
    they need this without caring that Perera called it `ligand` and Gesmundo
    `catalyst`."""
    bundle = Path(bundle)
    path = bundle / "ligand_mapping.csv"
    if not path.exists():
        # Derivable from reactions.csv, so a bundle without the file still plots.
        reactions, *_ = load_bundle(bundle)
        group = bundle_group(read_json(bundle / "config.json"))
        frame = reactions[[group, "kraken_id"]].drop_duplicates()
        return frame.rename(columns={group: "name"}).reset_index(drop=True)
    frame = pd.read_csv(path)
    name_column = next(c for c in frame.columns if c != "kraken_id" and c != "id")
    id_column = "kraken_id" if "kraken_id" in frame.columns else "id"
    return (frame[[name_column, id_column]]
            .rename(columns={name_column: "name", id_column: "kraken_id"})
            .reset_index(drop=True))
