"""Intake: turn one screen's own table into a bundle the engine can read.

The per-dataset notebooks under `datasets/` do the part that is genuinely
specific -- downloading, reshaping, deciding what to drop -- and then call
`write_bundle` here. Everything below that line is shared, so a new dataset type
needs a notebook and nothing else.

The canonical schema a bundle's `reactions.csv` must have:

    row_id        "reaction:1", "reaction:2", ... in table order
    <group>       the varied component's name (ligand, catalyst, ...)
    kraken_id     that component's Kraken id, the join key for descriptors
    <target>      the numeric response
    <categorical> one column per shared condition field

Everything else is carried through untouched and simply ignored by the models,
so keeping plate/row/column or the original names costs nothing.

The reference PCA is copied, never refitted. PC1..PC4 have to mean the same axes
in every dataset or the `pc_top` and `pc_scores` sections stop being comparable,
which is the whole point of running them side by side.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FIVE, SELECTED_SETS, TWO, write_json
from .features import ReferencePCA

ROOT = Path(__file__).resolve().parent.parent
KRAKEN_DIR = ROOT / "reference" / "kraken"

# Files copied verbatim from the reference into every bundle.
PCA_FILES = ("pca_reference.json", "pca_reference.npz")


def kraken_features() -> pd.DataFrame:
    """The full Kraken descriptor table: 1,223 ligands, PC1..PC4 included."""
    return pd.read_csv(KRAKEN_DIR / "ligand_features.csv")


def kraken_identifiers() -> pd.DataFrame:
    """Kraken's own name/id/SMILES table, for resolving a screen's ligand names."""
    return pd.read_csv(KRAKEN_DIR / "kraken_identifiers.csv")


def reference_pca() -> ReferencePCA:
    """The frozen reference PCA, as every dataset must use it."""
    return ReferencePCA.load(KRAKEN_DIR)


def find_kraken_id(name: str, identifiers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate Kraken rows whose name contains `name`, case-insensitively.

    A lookup helper for writing a new dataset's mapping by hand -- Kraken spells
    several of these differently from the papers (`PtBu3` for P(tBu)3,
    `PAd2nBu, CataCXium A` for CataCXium A), so the mapping is a judgement call
    that belongs in the notebook and not in an automatic match.
    """
    identifiers = kraken_identifiers() if identifiers is None else identifiers
    hit = identifiers[identifiers["ligand"].str.contains(name, case=False, na=False)]
    return hit[["ligand", "id", "can_smiles"]]


def attach_kraken(reactions: pd.DataFrame, group: str, mapping: dict) -> pd.DataFrame:
    """Add `kraken_id` from a {group name: kraken id} mapping, strictly.

    Every group value must be in the mapping. A silent NaN here becomes a
    missing descriptor row at training time, hundreds of lines further on.
    """
    unmapped = sorted(set(reactions[group]) - set(mapping))
    if unmapped:
        raise ValueError(f"No kraken_id for {group} values {unmapped}; "
                         f"add them to the mapping")
    out = reactions.copy()
    out["kraken_id"] = out[group].map(mapping).astype(int)
    return out


def add_row_ids(reactions: pd.DataFrame) -> pd.DataFrame:
    """`row_id` as "reaction:N", 1-based, in current table order."""
    out = reactions.reset_index(drop=True).copy()
    out.insert(0, "row_id", [f"reaction:{i + 1}" for i in range(len(out))])
    return out


def bundle_config(dataset: str, display_name: str, target: str, group: str,
                  categorical, reactions: pd.DataFrame, models=None, methods=None,
                  source: str = "", seed: int = 42, extra: dict | None = None) -> dict:
    """The bundle's config.json: what this dataset IS.

    Row and group counts are recorded from the frame being written, so
    `load_bundle` can refuse a bundle that was edited after the fact.
    """
    models = list(models or ["group_ohe", "selected_2", "vbur_min_vmin",
                             "vbur_boltz_vmin", "selected_5", "pc_top", "pc_scores"])
    n_groups = int(reactions[group].nunique())
    if methods is None:
        # LOLO is already an n-fold partition over n groups, so the matched
        # in-distribution control is the stratified fold of the SAME count --
        # same splitter family, differing only in whether the held-out group was
        # seen in training. kfold_5 is the unstratified 5-fold, for contrast.
        methods = ["lolo", f"iid_stratified_{n_groups}", "kfold_stratified_5", "kfold_5"]
    cfg = {
        "schema_version": 2,
        "dataset": dataset,
        "display_name": display_name,
        "source": source,
        "data": {
            "target": target,
            "group": group,
            "common_categorical": list(categorical),
            "expected_rows": int(len(reactions)),
            "expected_groups": n_groups,
        },
        "features": {
            "models": models,
            "pca_reference": "full_original_kraken",
            "pca_components": 4,
            "top_per_pc": 3,
            "pc_scores_form": "component_scores",
            "ohe_policy": "train_ignore_unknown",
        },
        "evaluation": {"methods": list(methods), "seed": seed, "stratify_on": group},
    }
    if extra:
        cfg.update(extra)
    return cfg


def check_bundle(reactions: pd.DataFrame, cfg: dict, ligands: pd.DataFrame) -> pd.DataFrame:
    """Every check `load_bundle` will make, run here so the notebook fails first.

    Returns a table of what passed, which is worth displaying in the notebook:
    it is the record of what the bundle claims about itself.
    """
    data = cfg["data"]
    target, group = data["target"], data["group"]
    checks = []

    def record(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    record("row_id present and unique",
           "row_id" in reactions and reactions["row_id"].is_unique,
           f"{len(reactions)} rows")
    record("target is numeric and complete",
           target in reactions
           and pd.to_numeric(reactions[target], errors="coerce").notna().all(),
           target)
    record("group column present", group in reactions,
           f"{reactions[group].nunique() if group in reactions else 0} distinct")
    record("condition columns present",
           all(c in reactions for c in data["common_categorical"]),
           ", ".join(data["common_categorical"]))
    record("kraken_id on every row",
           "kraken_id" in reactions and reactions["kraken_id"].notna().all(),
           f"{reactions['kraken_id'].nunique() if 'kraken_id' in reactions else 0} ids")
    have = set(ligands.kraken_id) if "kraken_id" in ligands else set()
    want = set(reactions["kraken_id"]) if "kraken_id" in reactions else set()
    record("descriptors for every kraken_id", not (want - have),
           f"missing {sorted(want - have)}" if want - have else "all present")
    # Every descriptor any explicit section needs, deduplicated. Checking only
    # selected_5 would have passed a bundle that cannot run the Vmin pairs.
    needed = list(dict.fromkeys(c for cols in SELECTED_SETS.values() for c in cols))
    present = [c for c in needed if c in ligands.columns]
    subset = ligands[ligands.kraken_id.isin(want)][present] if present else pd.DataFrame()
    record("section descriptors present and finite",
           len(present) == len(needed)
           and (subset.empty or np.isfinite(subset.to_numpy(dtype=float)).all()),
           f"{len(present)}/{len(needed)} columns"
           + (f"; missing {sorted(set(needed) - set(present))}"
              if len(present) != len(needed) else ""))
    record("PC1..PC4 present",
           all(f"PC{i}" in ligands.columns for i in range(1, 5)), "reference projection")
    record("row count matches config", len(reactions) == data["expected_rows"],
           f"{len(reactions)} vs {data['expected_rows']}")
    record("group count matches config",
           group in reactions and reactions[group].nunique() == data["expected_groups"],
           f"{reactions[group].nunique() if group in reactions else 0} vs "
           f"{data['expected_groups']}")

    table = pd.DataFrame(checks)
    failed = table[~table.ok]
    if len(failed):
        raise ValueError("Bundle checks failed:\n" + failed.to_string(index=False))
    return table


def write_bundle(destination, reactions: pd.DataFrame, cfg: dict, mapping: dict,
                 ligands: pd.DataFrame | None = None, audit: dict | None = None,
                 full_kraken: bool = False) -> Path:
    """Write a complete bundle and return its directory.

    `ligands` defaults to the full Kraken table; `full_kraken=False` trims it to
    the ids this screen uses, which is what the training zip needs and is all
    `feature_frame` ever joins against. The figures read the full reference from
    `reference/kraken/` regardless, so trimming costs the report nothing.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ligands = kraken_features() if ligands is None else ligands

    check_bundle(reactions, cfg, ligands)

    kept = (ligands if full_kraken
            else ligands[ligands.kraken_id.isin(set(reactions["kraken_id"]))])
    reactions.to_csv(destination / "reactions.csv", index=False)
    kept.to_csv(destination / "ligand_features.csv", index=False)

    group = cfg["data"]["group"]
    identifiers = kraken_identifiers().set_index("id")
    pd.DataFrame([
        {group: name, "kraken_id": kid,
         "kraken_name": identifiers["ligand"].get(kid, ""),
         "can_smiles": identifiers["can_smiles"].get(kid, "")}
        for name, kid in mapping.items()
    ]).to_csv(destination / "ligand_mapping.csv", index=False)

    for name in PCA_FILES:
        shutil.copyfile(KRAKEN_DIR / name, destination / name)
    write_json(destination / "config.json", cfg)
    if audit is not None:
        write_json(destination / "audit.json", audit)

    print(f"bundle -> {destination}")
    print(f"  {len(reactions)} rows, {reactions[group].nunique()} {group}s, "
          f"{len(kept)} descriptor rows")
    return destination


def describe(reactions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows and target statistics per group -- the table to eyeball before writing."""
    target, group = cfg["data"]["target"], cfg["data"]["group"]
    out = (reactions.groupby(group)[target]
           .agg(n="size", mean="mean", sd="std", min="min", max="max")
           .reset_index())
    out.insert(1, "kraken_id", out[group].map(
        reactions.drop_duplicates(group).set_index(group)["kraken_id"]))
    return out.sort_values("n", ascending=False).reset_index(drop=True)
