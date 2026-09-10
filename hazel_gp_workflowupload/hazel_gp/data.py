"""Original downloads, audited Group1 cleanup, and immutable portable bundles."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import urllib.request
import uuid
import zipfile

import numpy as np
import pandas as pd

from .config import TARGET, FIVE, TWO, file_hash, object_hash, read_json, write_json, validate_config, versions
from .features import (ReferencePCA, ligand_table, feature_frame, model_summary,
                       model_feature_records)
from .splits import make_splits, save_splits, evaluation_summary


def download_sources(cfg: dict, project_root: str | Path = ".", refresh=False) -> dict:
    """Download exact original URLs. Existing cached files are hash-checked and reused."""
    root = Path(project_root).resolve()
    raw = root / cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    previous = read_json(raw / "sources.json") if (raw / "sources.json").exists() else {}
    records = {}
    for key, spec in cfg["sources"].items():
        name = spec["filename"]
        if Path(name).name != name:
            raise ValueError("Source filename must be a basename")
        path = raw / name
        if path.exists() and not refresh:
            prior = previous.get(key)
            if prior and (file_hash(path) != prior["sha256"] or prior["url"] != spec["url"]):
                raise ValueError(f"Cached source changed: {path}. Use a new raw directory or explicit refresh.")
            records[key] = prior or {"filename": name, "url": spec["url"],
                                     "origin": "existing_local_file", "sha256": file_hash(path)}
            continue
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "Hazel-GP-workflow/1.0"})
        tmp = path.with_suffix(path.suffix + ".download")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as dst:
                shutil.copyfileobj(response, dst)
            if tmp.stat().st_size < 100:
                raise ValueError(f"Downloaded source is unexpectedly small: {spec['url']}")
            tmp.replace(path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {spec['url']}. Retry or place the original CSV at {path}.") from exc
        records[key] = {"filename": name, "url": spec["url"], "origin": "download",
                        "downloaded_utc": datetime.now(timezone.utc).isoformat(), "sha256": file_hash(path)}
    write_json(raw / "sources.json", records)
    return records


@dataclass
class PreparedData:
    config: dict
    reactions: pd.DataFrame
    descriptors: pd.DataFrame
    identifiers: pd.DataFrame
    ligands: pd.DataFrame
    reference: ReferencePCA
    excluded: pd.DataFrame
    mapping: pd.DataFrame
    audit: dict
    sources: dict
    split_entries: list
    split_arrays: dict
    tasks: pd.DataFrame


def prepare_data(cfg: dict, project_root: str | Path = ".") -> PreparedData:
    """Pure preparation after download: inspect the return value before exporting."""
    validate_config(cfg)
    raw = Path(project_root).resolve() / cfg["paths"]["raw"]
    d = cfg["data"]
    paths = {key: raw / s["filename"] for key, s in cfg["sources"].items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Run download_sources first; missing {path}")
    source_meta = read_json(raw / "sources.json") if (raw / "sources.json").exists() else {}
    sources = {}
    for key, path in paths.items():
        digest = file_hash(path)
        prior = source_meta.get(key, {})
        if prior and prior["sha256"] != digest:
            raise ValueError(f"Source hash mismatch: {path}")
        sources[key] = {**prior, "filename": path.name, "url": cfg["sources"][key]["url"],
                        "sha256": digest}
    df = pd.read_csv(paths["reactions"], sep=d["reaction_separator"], decimal=d["reaction_decimal"],
                     encoding="utf-8-sig", keep_default_na=False)
    raw_rows = len(df)
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].astype(str).str.strip()
    unnamed = [c for c in df if c.startswith("Unnamed")]
    if any((df[c].notna() & df[c].astype(str).ne("")).any() for c in unnamed):
        raise ValueError("Unexpected nonempty Unnamed columns; review the source delimiter/schema")
    df = df.drop(columns=unnamed)
    required = ["Reaction_No", "Reactant_1_Short_Hand", "Reactant_2_Name",
                "Ligand_Short_Hand", "Reagent_1_Short_Hand", "Solvent_1_Short_Hand", TARGET]
    if set(required) - set(df):
        raise ValueError(f"Missing reaction fields: {set(required) - set(df)}")
    excluded = []

    def remove(mask, reason):
        nonlocal df
        part = df.loc[mask].copy()
        part["exclusion_reason"] = reason
        excluded.append(part)
        df = df.loc[~mask].copy()

    remove(df.duplicated(), "exact_duplicate_import")
    if df["Reaction_No"].duplicated().any():
        raise ValueError("Conflicting repeated Reaction_No values; resolve before splitting")
    remove(df["Solvent_1_Short_Hand"].isin(d["excluded_solvents"]), "original_solvent_exclusion")
    remove(df["Ligand_Short_Hand"].isin(d["excluded_ligands"]), "outside_monodentate_ligand_scope")
    after_exclusions = len(df)
    df["reactant1_code"] = df["Reactant_1_Short_Hand"].str.extract(r"^(1[a-z])", expand=False)
    df["reactant2_code"] = df["Reactant_2_Name"].str.extract(r"^(2[a-z])", expand=False)
    if df[["reactant1_code", "reactant2_code"]].isna().any().any():
        raise ValueError("Unrecognized reactant labels; revise the documented code extraction")
    keep = df["reactant1_code"].isin(d["reactant1_codes"]) & df["reactant2_code"].isin(d["reactant2_codes"])
    remove(~keep, "outside_Group1_scope")
    df = df.reset_index(drop=True)
    df["substrate_pair"] = df["reactant1_code"] + "_" + df["reactant2_code"]
    df["ligand"] = df["Ligand_Short_Hand"]
    if any(df[c].eq("").any() for c in required):
        raise ValueError("Blank required reaction fields; literal 'None' base is retained as a category")
    # Explicitly convert source decimal strings, avoiding accidental object targets.
    df[TARGET] = pd.to_numeric(df[TARGET].astype(str).str.replace(",", ".", regex=False), errors="raise")
    if not np.isfinite(df[TARGET]).all() or not df[TARGET].between(0, 100).all():
        raise ValueError("Missing, nonfinite, or out-of-range UV-area yields; review without clipping")
    df["row_id"] = "reaction:" + df["Reaction_No"].astype(str)
    df["kraken_id"] = df["ligand"].map(d["ligand_mapping"])
    if df["kraken_id"].isna().any():
        raise ValueError(f"Unmapped ligands: {df.loc[df.kraken_id.isna(), 'ligand'].unique()}")
    df["kraken_id"] = df["kraken_id"].astype(int)
    condition_keys = ["ligand", "substrate_pair", "Reagent_1_Short_Hand", "Solvent_1_Short_Hand"]
    if df.duplicated(condition_keys, keep=False).any():
        raise ValueError("Repeated experimental conditions detected. This original-data protocol requires "
                         "one row per condition; define replicate grouping before using row-random splits.")
    for key, observed in (("expected_rows", len(df)), ("expected_ligands", df.ligand.nunique())):
        if d[key] is not None and observed != d[key]:
            raise ValueError(f"{key}: expected {d[key]}, found {observed}; inspect the new source snapshot")
    desc = pd.read_csv(paths["descriptors"])
    ids = pd.read_csv(paths["identifiers"], keep_default_na=False)
    for frame in (desc, ids):
        frame["id"] = pd.to_numeric(frame["id"], errors="raise")
        if frame["id"].isna().any() or (frame["id"] % 1 != 0).any() or frame["id"].duplicated().any():
            raise ValueError("Kraken IDs must be unique nonmissing integers")
        frame["id"] = frame["id"].astype(int)
    desc = desc.astype({c: "float64" for c in desc if c != "id"})
    if set(df.kraken_id) - set(desc.id):
        raise ValueError("One or more modeled ligands are missing from the descriptor reference")
    mapping = pd.DataFrame({"ligand": list(d["ligand_mapping"]), "id": list(d["ligand_mapping"].values())})
    mapping = mapping.merge(ids[["id", "ligand", "can_smiles"]].rename(columns={"ligand": "kraken_name"}),
                            on="id", how="left", validate="one_to_one")
    if mapping[["kraken_name", "can_smiles"]].isna().any().any() or mapping.can_smiles.eq("").any():
        raise ValueError("Incomplete identifier/structure mapping")
    reference = ReferencePCA.fit(desc, cfg["features"]["top_per_pc"])
    ligands = ligand_table(desc, reference, cfg)
    for model in cfg["features"]["models"]:
        feature_frame(df, ligands, model, cfg, reference)
    entries, arrays, tasks = make_splits(df, cfg)
    audit = {"raw_rows": raw_rows, "after_solvent_ligand_exclusions": after_exclusions,
             "model_rows": len(df), "ligand_count": int(df.ligand.nunique()),
             "rows_per_ligand": {str(k): int(v) for k, v in df.ligand.value_counts().items()},
             "reference_ligands": len(desc), "reference_descriptors": len(reference.columns),
             "pca_variance_ratio": reference.variance_ratio.tolist(),
             "model_fit_count": len(tasks), "source_outcomes_excluded_from_features": [TARGET, "Product_Yield_Mass_Ion_Count"],
             "common_inputs": d["common_categorical"], "versions": versions(),
             "units": {"vbur_raw": "angstrom^3", "vbur_pct_*": "percent of 3.5 angstrom-radius sphere",
                       "homo_lumo_gap_eV": "(LUMO - HOMO) * hartree_to_eV; positive gap",
                       "target": "UV area percent, not isolated yield"}}
    return PreparedData(cfg, df, desc, ids, ligands, reference, pd.concat(excluded, ignore_index=True),
                        mapping, audit, sources, entries, arrays, tasks)


def export_prepared(prepared: PreparedData, destination: str | Path) -> Path:
    """Explicit final export. Never overwrite an existing prepared experiment."""
    dst = Path(destination).resolve()
    if dst.exists():
        raise FileExistsError(f"Prepared bundle already exists: {dst}. Choose a new version folder.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = dst.parent / (".preparing_" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        for name, frame in (("reactions", prepared.reactions), ("kraken_reference", prepared.descriptors),
                            ("ligand_features", prepared.ligands), ("ligand_mapping", prepared.mapping),
                            ("excluded_rows", prepared.excluded)):
            frame.to_csv(stage / f"{name}.csv", index=False)
        write_json(stage / "config.json", prepared.config)
        write_json(stage / "audit.json", prepared.audit)
        write_json(stage / "sources.json", prepared.sources)
        prepared.reference.save(stage)
        save_splits(stage, prepared.split_entries, prepared.split_arrays, prepared.tasks)
        model_summary(prepared.config, prepared.reference).to_csv(stage / "model_table.csv", index=False)
        model_feature_records(prepared).to_csv(stage / "model_features.csv", index=False)
        evaluation_summary(prepared.config, prepared.tasks).to_csv(stage / "evaluation_table.csv", index=False)
        manifest = {"schema_version": 1, "n_rows": len(prepared.reactions),
                    "n_tasks": len(prepared.tasks), "config_hash": object_hash(prepared.config),
                    "files": {p.name: file_hash(p) for p in sorted(stage.iterdir()) if p.is_file()}}
        manifest["bundle_id"] = object_hash(manifest)
        write_json(stage / "manifest.json", manifest)
        stage.rename(dst)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return dst


def verify_bundle(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
    if object_hash(payload) != manifest["bundle_id"]:
        raise ValueError("Bundle manifest was modified")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError(f"Prepared input changed or is missing: {name}")
    cfg = validate_config(read_json(directory / "config.json"))
    if object_hash(cfg) != manifest["config_hash"]:
        raise ValueError("Configuration hash mismatch")
    return manifest


def load_bundle(directory: str | Path):
    directory = Path(directory)
    manifest = verify_bundle(directory)
    cfg = read_json(directory / "config.json")
    reactions = pd.read_csv(directory / "reactions.csv", keep_default_na=False)
    if reactions.row_id.duplicated().any() or len(reactions) != manifest["n_rows"]:
        raise ValueError("Invalid canonical reaction rows")
    return (manifest, cfg, reactions, pd.read_csv(directory / "ligand_features.csv"),
            ReferencePCA.load(directory))


def create_upload_archive(project_root: str | Path, bundle: str | Path, destination: str | Path) -> Path:
    """Code plus verified inputs; no environments, runs, or unrelated local files."""
    root, bundle, dst = Path(project_root).resolve(), Path(bundle).resolve(), Path(destination).resolve()
    verify_bundle(bundle)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.zip")
    selected = [p for p in root.iterdir() if p.is_file() and
                (p.suffix in (".md", ".txt", ".toml", ".ipynb") or p.name == "environment_local.yml")]
    for folder in ("hazel_gp", "hazel", "configs", "tests"):
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file() and
                        "__pycache__" not in p.parts and p.suffix != ".pyc")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(selected)):
                z.write(p, str(Path("hazel_gp_workflow") / p.relative_to(root)))
            for p in sorted(bundle.iterdir()):
                if p.is_file():
                    z.write(p, str(Path("hazel_gp_workflow/inputs") / p.name))
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dst
