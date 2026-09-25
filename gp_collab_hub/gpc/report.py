"""Turn a finished run into `results/<dataset>_<date>/`.

One call produces the folder the hub hands back: the saved figures, the tables
they were drawn from, and a `report.json` recording which run and which bundle
they came from. Every drawing decision lives in `gpc/figures.py`; this module
only decides what to draw, what to call it, and where to put it.

    results/perera_20260923/
        figures/
            pooled_r2_by_featureset.png
            lolo_by_held_out_ligand.png
            kraken_space.png
        tables/
            summary.csv  metrics_by_split.csv  predictions.csv  lolo_report.csv
        report.json
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path

import pandas as pd

from . import figures
from .config import bundle_group, bundle_name, read_json, write_json
from .data import group_mapping
from .results import lolo_report, load_results, prepare

ROOT = Path(__file__).resolve().parent.parent
# The full 1,223-ligand Kraken table. The reference cloud must be the same in
# every dataset's report, and a bundle is allowed to ship only the ligands it
# actually uses (Ahneman's carries four rows), so the population is read from
# the hub rather than from the bundle whenever the hub is available.
KRAKEN_REFERENCE = ROOT / "reference" / "kraken" / "ligand_features.csv"

# Tables copied into every results folder. `predictions` is the row-level source
# every figure is derived from, and it is also by far the largest file here --
# about 10 MB for a 3,000-reaction screen across five sections. Drop it from
# this tuple, or pass `tables=`, if the folder is being kept only for figures.
REPORT_TABLES = ("summary", "metrics_by_split", "predictions")

FIGURE_NAMES = {
    "pooled": "pooled_{metric}_by_featureset",
    "lolo": "lolo_by_held_out_{group}",
    "kraken": "kraken_space",
}


def _slug(text: str) -> str:
    """Folder-safe lower-case token: 'Perera (Suzuki)' -> 'perera_suzuki'."""
    keep = [c.lower() if (c.isalnum() or c in "-_") else " " for c in str(text)]
    return "_".join("".join(keep).split()) or "dataset"


def kraken_population(bundle) -> pd.DataFrame:
    """The reference cloud the space figures plot behind the dataset's ligands."""
    if KRAKEN_REFERENCE.exists():
        return pd.read_csv(KRAKEN_REFERENCE)
    # Running from inside an HPC zip, where reference/ was not shipped: the
    # bundle's own descriptor table is the only population available. Say which
    # one was used in report.json rather than quietly drawing a four-point cloud.
    return pd.read_csv(Path(bundle) / "ligand_features.csv")


def dataset_ligands(bundle, population: pd.DataFrame) -> pd.DataFrame:
    """This dataset's own ligands, as rows of the reference table plus a name."""
    mapping = group_mapping(bundle)
    merged = mapping.merge(population, on="kraken_id", how="left", validate="many_to_one")
    missing = merged[merged[figures.STYLE["kraken"]["pc_x"]].isna()]
    if len(missing):
        raise ValueError(f"No descriptor row for kraken_id "
                         f"{sorted(missing.kraken_id.tolist())}; the reference table "
                         f"and this bundle's ligand_mapping.csv disagree")
    return merged


def write_report(runs, bundle, out=None, date=None, dataset_name=None,
                 metric: str = "r2", formats=("png",), refresh: bool = False,
                 tables_to_save=REPORT_TABLES) -> Path:
    """Build every figure and table for one run and write them under `out`.

    `runs` is the folder the training jobs wrote to; `bundle` is the prepared
    inputs they were trained against. `out` defaults to
    `results/<dataset>_<YYYYMMDD>`, which is also what the run tag is called, so
    a results folder and the run that produced it carry the same name.
    """
    runs, bundle = Path(runs), Path(bundle)
    prepared = read_json(bundle / "config.json")
    group = bundle_group(prepared)
    display_name = dataset_name or bundle_name(prepared)
    stamp = date or _date.today().strftime("%Y%m%d")

    destination = Path(out) if out else ROOT / "results" / f"{_slug(display_name)}_{stamp}"
    figure_dir, table_dir = destination / "figures", destination / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    export = prepare(runs, bundle=bundle, refresh=refresh)
    tables, collection = load_results(export)

    written, skipped = [], []

    def _emit(key, name, build):
        """Draw one figure, or record why it could not be drawn.

        A section that a run does not support (no LOLO folds, say) must not cost
        the rest of the report, and a blank file would be worse than an absence,
        so the reason goes in report.json instead.
        """
        import matplotlib.pyplot as plt
        try:
            fig = build()
        except (ValueError, KeyError) as exc:
            skipped.append({"figure": key, "reason": str(exc)})
            return
        for path in figures.save(fig, figure_dir, name, formats):
            written.append(str(path.relative_to(destination)))
        plt.close(fig)

    _emit("pooled", FIGURE_NAMES["pooled"].format(metric=metric),
          lambda: figures.pooled_by_featureset(tables["summary"], display_name, metric))
    _emit("lolo", FIGURE_NAMES["lolo"].format(group=group),
          lambda: figures.lolo_by_group(tables["metrics_by_split"], display_name,
                                        metric, group))

    population = kraken_population(bundle)
    _emit("kraken", FIGURE_NAMES["kraken"],
          lambda: figures.kraken_space(population, dataset_ligands(bundle, population),
                                       display_name))

    for name in tables_to_save:
        if name in tables and len(tables[name]):
            tables[name].to_csv(table_dir / f"{name}.csv", index=False)
    try:
        lolo_report(tables["predictions"], tables["metrics_by_split"]).to_csv(
            table_dir / "lolo_report.csv", index=False)
    except ValueError as exc:
        skipped.append({"table": "lolo_report", "reason": str(exc)})

    write_json(destination / "report.json", {
        "dataset": prepared.get("dataset"), "display_name": display_name,
        "group": group, "date": stamp, "metric": metric,
        "runs": str(runs), "bundle": str(bundle), "export": str(export),
        "kraken_population": str(KRAKEN_REFERENCE if KRAKEN_REFERENCE.exists()
                                 else bundle / "ligand_features.csv"),
        "n_population": int(len(population)),
        "complete": collection.get("complete"),
        "completed_tasks": collection.get("completed_tasks"),
        "expected_tasks": collection.get("expected_tasks"),
        "figures": written, "tables": list(tables_to_save), "skipped": skipped,
        "style_source": "gpc/figures.py",
    })

    print(f"report -> {destination}")
    for path in written:
        print(f"  {path}")
    for entry in skipped:
        print(f"  skipped {entry.get('figure') or entry.get('table')}: {entry['reason']}")
    return destination
