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
    reaction_features.csv  OPTIONAL: per-reaction descriptors keyed by row_id
                         (only bundles whose config.json declares
                         "reaction_features"; read by the doyle_* sections)

Everything the model sees is in `reactions.csv`, `ligand_features.csv` and, when
present, `reaction_features.csv`, and
`config.json` is the authority on which column plays which role. Nothing here
hardcodes a dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (bundle_group, bundle_target, condition_numeric, reaction_columns,
                     read_json)
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
    numeric = condition_numeric(prepared)
    missing = [c for c in numeric if c not in reactions]
    if missing:
        raise ValueError(f"Bundle is missing continuous condition columns: {missing}")
    for column in numeric:
        reactions[column] = pd.to_numeric(reactions[column], errors="raise")
    if numeric and not np.isfinite(reactions[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or non-finite values in {numeric}")
    if "kraken_id" not in reactions:
        raise ValueError("reactions.csv has no kraken_id column to join descriptors on")

    ligands = pd.read_csv(bundle / "ligand_features.csv")
    unmapped = sorted(set(reactions.kraken_id) - set(ligands.kraken_id))
    if unmapped:
        raise ValueError(f"kraken_ids in reactions.csv with no descriptor row: {unmapped}")
    reactions = _attach_reaction_features(bundle, reactions, ligands, prepared)
    return reactions, ligands, ReferencePCA.load(bundle), prepared


def _attach_reaction_features(bundle: Path, reactions: pd.DataFrame,
                              ligands: pd.DataFrame, prepared: dict) -> pd.DataFrame:
    """Join the bundle's reaction-level descriptor table onto `reactions`, if it has one.

    Only bundles that declare "reaction_features" in config.json carry one
    (`ahneman_doyle`: Doyle's 120 DFT descriptors). It is joined here, once, on
    row_id, so every caller of load_bundle -- training, export, review -- sees the
    same columns, and the 4-tuple this function returns never changes shape.
    Strict on purpose: a row it cannot match, a non-finite value or a column name
    that collides with a reaction or Kraken column is an error, not a NaN.
    """
    columns = reaction_columns(prepared)
    if not columns:
        return reactions
    block = prepared["reaction_features"]
    path = bundle / block.get("file", "reaction_features.csv")
    if not path.exists():
        raise FileNotFoundError(f"config.json declares reaction_features but {path.name} "
                                f"is not in {bundle}")
    table = pd.read_csv(path)
    absent = [c for c in columns if c not in table.columns]
    if absent:
        raise ValueError(f"{path.name} lacks declared columns, e.g. {absent[:3]}")
    clash = sorted((set(columns) & set(reactions.columns))
                   | (set(columns) & (set(ligands.columns) - {"kraken_id"})))
    if clash:
        raise ValueError(f"{path.name} columns collide with reaction/Kraken columns: {clash[:5]}")
    if list(table["row_id"]) != list(reactions["row_id"]):
        raise ValueError(f"{path.name} is not row-aligned with reactions.csv")
    values = table[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} has missing or non-finite descriptor values")
    return reactions.merge(table[["row_id"] + columns], on="row_id", how="left",
                           sort=False, validate="one_to_one")


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
