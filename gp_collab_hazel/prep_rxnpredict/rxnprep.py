"""Build a gp_collab_hazel input bundle from the Doyle rxnpredict screen.

Source: https://github.com/doylelab/rxnpredict -- the Ahneman et al. (Science
2018) Buchwald-Hartwig C-N coupling screen. Reaction conditions and yields come
from `data_table.csv`; the DFT descriptors come from the four per-component
tables in `R/`, joined on component name rather than by row position. That join
reproduces Doyle's own `R/output_table.csv` exactly (same 120 columns, same
3,960-row multiset), which is the correctness check `verify_against_doyle` runs.

Ligand chemistry is attached from the frozen Kraken reference already in
`inputs/`, so `pc_top` and `pc_scores` use the same components, the same top-12
descriptors and the same loadings as the Perera runs, with no refitting.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/doylelab/rxnpredict/master/"
SOURCES = {
    "data_table.csv": RAW + "data_table.csv",
    "ligand.csv": RAW + "R/ligand.csv",
    "base.csv": RAW + "R/base.csv",
    "aryl_halide.csv": RAW + "R/aryl_halide.csv",
    "additive.csv": RAW + "R/additive.csv",
    "ligand-list.csv": RAW + "smiles/ligand-list.csv",
    "output_table.csv": RAW + "R/output_table.csv",      # only for verification
    # Kraken identifiers, to check the ligand mapping by SMILES rather than name.
    "kraken_identifiers.csv":
        "https://raw.githubusercontent.com/doyle-lab-ucla/kraken_utils/main/identifiers.csv",
}
COMPONENTS = ["ligand", "base", "aryl_halide", "additive"]
CATEGORICAL = ["aryl_halide", "base", "additive"]         # shared block; ligand is the group
TARGET = "yield"
GROUP = "ligand"

# Doyle ligand -> Kraken id. Verified by canonical SMILES in `verify_ligand_map`.
LIGAND_KRAKEN = {"XPhos": 1, "t-BuXPhos": 90, "t-BuBrettPhos": 89, "AdBrettPhos": 347}

MODELS = ["ligand_ohe", "selected_2", "selected_5", "pc_top", "pc_scores",
          "rxnpredict_full", "rxnpredict_full_ohe"]
TWO = ["vbur_pct_boltz", "vbur_pct_min"]
FIVE = TWO + ["vbur_pct_delta", "dipolemoment_boltz", "homo_lumo_gap_eV"]


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float) + "\n", encoding="utf-8")


def download_sources(raw_dir, refresh: bool = False) -> dict:
    """Cache the rxnpredict files locally and record their hashes."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, url in SOURCES.items():
        path = raw_dir / name
        if refresh or not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
            print(f"downloaded {name}")
        record[name] = {"url": url, "sha256": file_hash(path), "bytes": path.stat().st_size}
    return record


def load_components(raw_dir) -> dict:
    return {name: pd.read_csv(Path(raw_dir) / f"{name}.csv") for name in COMPONENTS}


def build_reactions(raw_dir) -> tuple[pd.DataFrame, dict]:
    """Conditions and yield only -- no descriptors of any kind.

    Drops the no-aryl-halide control wells and the one additive that has no
    computed descriptors, leaving the 3,955 complete reactions.
    """
    table = pd.read_csv(Path(raw_dir) / "data_table.csv")
    components = load_components(raw_dir)
    audit = {"raw_rows": len(table)}

    described = set(components["additive"]["name"])
    no_descriptors = table[~table.additive.isin(described)]
    audit["dropped_additive_without_descriptors"] = {
        "rows": int(len(no_descriptors)),
        "names": sorted(str(x) for x in no_descriptors.additive.dropna().unique()),
        "rows_with_blank_additive": int(table.additive.isna().sum())}
    table = table[table.additive.isin(described)]

    controls = table[table.aryl_halide.isna()]
    audit["dropped_no_aryl_halide_controls"] = {
        "rows": int(len(controls)),
        "max_yield": float(controls[TARGET].max()) if len(controls) else None}
    table = table.dropna(subset=["aryl_halide"])

    if table.duplicated(COMPONENTS).any():
        raise ValueError("Repeated condition tuples; row-random splits would leak")
    for name, frame in components.items():
        unknown = set(table[name]) - set(frame["name"])
        if unknown:
            raise ValueError(f"{name} values without descriptors: {sorted(unknown)}")
    if table[TARGET].isna().any():
        raise ValueError("Missing yields")

    reactions = table[COMPONENTS + [TARGET, "plate", "row", "col"]].copy()
    reactions = reactions.sort_values(["plate", "row", "col"]).reset_index(drop=True)
    reactions.insert(0, "row_id", [f"reaction:{i + 1}" for i in range(len(reactions))])
    reactions["kraken_id"] = reactions[GROUP].map(LIGAND_KRAKEN)
    if reactions.kraken_id.isna().any():
        raise ValueError("A ligand has no Kraken id")
    reactions["kraken_id"] = reactions["kraken_id"].astype(int)

    grid = len(set(table.ligand)) * len(set(table.base)) * len(set(table.aryl_halide)) * len(set(table.additive))
    audit["kept_rows"] = int(len(reactions))
    audit["full_factorial"] = int(grid)
    audit["missing_combinations"] = sorted(
        list(map(list, set(itertools.product(*[sorted(set(table[c])) for c in COMPONENTS]))
                 - set(map(tuple, table[COMPONENTS].values)))))
    audit["rows_per_ligand"] = reactions[GROUP].value_counts().sort_index().to_dict()
    return reactions, audit


def build_rxnpredict_features(reactions, raw_dir) -> pd.DataFrame:
    """The 120 DFT descriptors per reaction, joined on component name."""
    frame = reactions[["row_id"] + COMPONENTS].copy()
    for name, table in load_components(raw_dir).items():
        frame = frame.merge(table, left_on=name, right_on="name",
                            how="left", validate="many_to_one").drop(columns="name")
    frame = frame.drop(columns=COMPONENTS)
    if frame.isna().any().any():
        raise ValueError("Missing descriptor values after the component join")
    return frame


def verify_against_doyle(raw_dir) -> dict:
    """Rebuild Doyle's full-factorial descriptor table from the same join.

    Proves the join is right without relying on row order: the published table
    has no key columns, and its R script attaches yields positionally.
    """
    components = load_components(raw_dir)
    published = pd.read_csv(Path(raw_dir) / "output_table.csv")
    grid = pd.DataFrame(list(itertools.product(*[components[c]["name"] for c in COMPONENTS])),
                        columns=COMPONENTS)
    for name, table in components.items():
        grid = grid.merge(table, left_on=name, right_on="name",
                          how="left", validate="many_to_one").drop(columns="name")
    mine = grid.drop(columns=COMPONENTS)
    order = sorted(published.columns)
    same_columns = sorted(mine.columns) == order
    rows_mine = set(map(tuple, np.round(mine[order].to_numpy(float), 8))) if same_columns else set()
    rows_published = set(map(tuple, np.round(published[order].to_numpy(float), 8)))
    return {"published_shape": list(published.shape), "rebuilt_shape": list(mine.shape),
            "same_columns": bool(same_columns),
            "identical_row_multiset": bool(same_columns and rows_mine == rows_published)}


def verify_ligand_map(raw_dir) -> pd.DataFrame:
    """Doyle ligand, its Kraken entry, and both SMILES side by side.

    Compare the two SMILES columns yourself: they are written by different
    groups so they differ in kekulisation and atom order, and matching them is
    a chemistry judgement, not a string comparison.
    """
    raw_dir = Path(raw_dir)
    listed = pd.read_csv(raw_dir / "ligand-list.csv")
    kraken = pd.read_csv(raw_dir / "kraken_identifiers.csv", keep_default_na=False)
    rows = []
    for ligand, kid in LIGAND_KRAKEN.items():
        match = kraken.loc[kraken.id == kid]
        rows.append({
            "ligand": ligand, "kraken_id": kid,
            "kraken_name": match["ligand"].squeeze() if len(match) else "NOT FOUND",
            "rxnpredict_smiles": listed.loc[listed.name == ligand, "Ligand_SMILES"].squeeze(),
            "kraken_smiles": match["can_smiles"].squeeze() if len(match) else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- feature sets

def model_columns(model: str, cfg: dict, reference, rxn_columns) -> tuple[list, list, list]:
    """(ligand-descriptor columns, rxnpredict columns, categorical fields).

    The three ligand-chemistry sets are the same definitions the Perera bundle
    uses. The two rxnpredict sets carry the published DFT descriptors instead,
    with `rxnpredict_full` holding no one-hot block at all, as Doyle modelled it.
    """
    categorical = list(cfg["data"]["common_categorical"])
    scores = list(reference.columns) if cfg["features"]["pc_scores_form"] == "loading_weighted" \
        else [f"PC{i}" for i in range(1, 5)]
    ligand = {"ligand_ohe": [], "selected_5": FIVE, "selected_2": TWO,
              "pc_top": list(reference.top_features), "pc_scores": scores,
              "rxnpredict_full": [], "rxnpredict_full_ohe": []}[model]
    rxn = list(rxn_columns) if model.startswith("rxnpredict") else []
    if model == "ligand_ohe":
        categorical = [GROUP] + categorical
    if model == "rxnpredict_full":
        categorical = []          # descriptors only, exactly the published baseline
    return list(ligand), rxn, categorical


def feature_frame(reactions, ligands, rxn_features, model, cfg, reference) -> pd.DataFrame:
    ligand_cols, rxn_cols, categorical = model_columns(
        model, cfg, reference, [c for c in rxn_features.columns if c != "row_id"])
    frame = reactions.merge(ligands[["kraken_id"] + ligand_cols], on="kraken_id",
                            how="left", sort=False, validate="many_to_one")
    if rxn_cols:
        frame = frame.merge(rxn_features[["row_id"] + rxn_cols], on="row_id",
                            how="left", sort=False, validate="one_to_one")
    if len(frame) != len(reactions):
        raise ValueError(f"{model}: the feature merge changed the row count")
    numeric = ligand_cols + rxn_cols
    if numeric and not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError(f"{model}: missing or non-finite features")
    if categorical and frame[categorical].isna().any().any():
        raise ValueError(f"{model}: missing categorical inputs")
    return frame[numeric + categorical]


def encoded_width(reactions, ligands, rxn_features, model, cfg, reference) -> tuple[int, int, int]:
    """(numeric, one-hot, total) input counts, one-hot fitted on all rows."""
    ligand_cols, rxn_cols, categorical = model_columns(
        model, cfg, reference, [c for c in rxn_features.columns if c != "row_id"])
    numeric = len(ligand_cols) + len(rxn_cols)
    if model == "pc_scores":
        numeric *= cfg["features"]["pca_components"]      # loadings expand each descriptor
    onehot = int(sum(reactions[c].nunique() for c in categorical))
    return numeric, onehot, numeric + onehot


def model_table(reactions, ligands, rxn_features, cfg, reference) -> pd.DataFrame:
    described = {
        "ligand_ohe": "Ligand identity one-hot",
        "selected_2": "Boltzmann-average and minimum buried volume",
        "selected_5": "Buried volume boltz/min/range, dipole, HOMO-LUMO gap",
        "pc_top": "Top 3 Kraken descriptors from each of PC1-PC4, deduplicated",
        "pc_scores": "Every Kraken descriptor times its loading in each of PC1-PC4",
        "rxnpredict_full": "Published DFT descriptors for all four components, no one-hot block",
        "rxnpredict_full_ohe": "The same DFT descriptors plus the shared one-hot block"}
    rows = []
    for model in cfg["features"]["models"]:
        ligand_cols, rxn_cols, categorical = model_columns(
            model, cfg, reference, [c for c in rxn_features.columns if c != "row_id"])
        numeric, onehot, total = encoded_width(reactions, ligands, rxn_features, model, cfg, reference)
        rows.append({"model": model, "ligand_features": described[model],
                     "ligand_descriptors": len(ligand_cols), "rxnpredict_descriptors": len(rxn_cols),
                     "numeric_inputs": numeric, "onehot_inputs": onehot, "encoded_inputs": total,
                     "categorical_fields": ", ".join(categorical) or "(none)"})
    return pd.DataFrame(rows)


def model_features(reactions, ligands, rxn_features, cfg, reference) -> pd.DataFrame:
    """Every encoded input name, per model, for review."""
    from sklearn.preprocessing import OneHotEncoder
    rows = []
    for model in cfg["features"]["models"]:
        ligand_cols, rxn_cols, categorical = model_columns(
            model, cfg, reference, [c for c in rxn_features.columns if c != "row_id"])
        names = []
        if model == "pc_scores":
            names += [f"num__PC{i}_x_{c}" for i in range(1, cfg["features"]["pca_components"] + 1)
                      for c in ligand_cols]
        else:
            names += [f"num__{c}" for c in ligand_cols]
        names += [f"num__{c}" for c in rxn_cols]
        if categorical:
            encoder = OneHotEncoder(sparse_output=False).fit(reactions[categorical])
            names += [f"cat__{n}" for n in encoder.get_feature_names_out(categorical)]
        block = (["ligand"] * (len(names) - len(rxn_cols) - sum(
            reactions[c].nunique() for c in categorical)) + ["rxnpredict"] * len(rxn_cols)
            + ["categorical"] * int(sum(reactions[c].nunique() for c in categorical)))
        rows += [{"model": model, "position": i + 1, "feature": n, "block": b}
                 for i, (n, b) in enumerate(zip(names, block))]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- the bundle

def default_config() -> dict:
    return {
        "schema_version": 1,
        "dataset": "rxnpredict_ahneman_2018",
        "source": "https://github.com/doylelab/rxnpredict",
        "data": {
            "target": TARGET, "group": GROUP,
            "common_categorical": CATEGORICAL,
            "ligand_mapping": LIGAND_KRAKEN,
            "expected_rows": 3955, "expected_ligands": 4,
            "sphere_radius_angstrom": 3.5, "hartree_to_eV": 27.211386245981},
        "features": {
            "models": MODELS, "pca_reference": "full_original_kraken",
            "pca_components": 4, "top_per_pc": 3,
            "pc_scores_form": "loading_weighted", "ohe_policy": "train_ignore_unknown"},
        "evaluation": {
            "methods": ["lolo", "iid_stratified_4", "kfold_stratified_5"],
            "seed": 42, "stratify_on": GROUP},
    }


def export_bundle(destination, reactions, ligands, rxn_features, cfg, reference,
                  audit, sources, perera_inputs) -> Path:
    """Write the bundle gp_collab_hazel reads, mirroring inputs/ file for file."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    perera_inputs = Path(perera_inputs)

    used = sorted(set(reactions.kraken_id))
    ligands[ligands.kraken_id.isin(used)].to_csv(destination / "ligand_features.csv", index=False)
    reactions.to_csv(destination / "reactions.csv", index=False)
    rxn_features.to_csv(destination / "rxnpredict_features.csv", index=False)
    model_table(reactions, ligands, rxn_features, cfg, reference).to_csv(
        destination / "model_table.csv", index=False)
    model_features(reactions, ligands, rxn_features, cfg, reference).to_csv(
        destination / "model_features.csv", index=False)
    pd.DataFrame([{"ligand": k, "kraken_id": v} for k, v in LIGAND_KRAKEN.items()]).to_csv(
        destination / "ligand_mapping.csv", index=False)

    # The frozen PCA is copied, never refitted, so pc_top and pc_scores mean the
    # same thing here as in the Perera bundle.
    for name in ("pca_reference.json", "pca_reference.npz", "pca_loadings.csv"):
        (destination / name).write_bytes((perera_inputs / name).read_bytes())

    write_json(destination / "config.json", cfg)
    write_json(destination / "audit.json", {"sources": sources, **audit})

    files = sorted(p.name for p in destination.iterdir() if p.name != "manifest.json")
    hashes = {name: file_hash(destination / name) for name in files}
    write_json(destination / "manifest.json", {
        "schema_version": 1, "dataset": cfg["dataset"],
        "n_rows": int(len(reactions)), "n_models": len(cfg["features"]["models"]),
        "n_methods": len(cfg["evaluation"]["methods"]),
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "files": hashes,
        "bundle_id": hashlib.sha256("".join(hashes[n] for n in files).encode()).hexdigest()})
    return destination
