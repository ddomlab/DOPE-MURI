"""Build the zip that goes to the HPC.

What ships: the engine, the chosen dataset's prepared inputs, the submit
scripts, and the resolved run config. What does not: the intake notebooks, the
Kraken reference cloud, previous runs and results, every other dataset, and the
raw downloads each bundle was built from. None of those are read by a training
job, and together they are most of the repository's size.

The submit scripts are rewritten on the way in, so the run tag, conda prefix,
walltime, GPU type, memory and CPU count in the zip are the ones chosen in
`build_run.ipynb` rather than whatever the template happened to say. The
resolved config is written to `configs/default.json` inside the zip because that
is the path both `gpc` and the scripts read by default -- JSON, never YAML, so
the cluster needs no PyYAML to start a job.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import date as _date
from pathlib import Path

import pandas as pd

from .config import read_json, validate_config
from .splits import parse_method

ROOT = Path(__file__).resolve().parent.parent

# Engine files a training job actually executes. The whole package ships rather
# than a hand-picked subset: it is ~150 KB, and a partially-copied package fails
# at import time on the cluster instead of here.
ENGINE_GLOBS = ("gpc/*.py", "gpc/vendor/*.py")
SUPPORT_FILES = ("requirements.txt", ".gitignore")
SCRIPT_GLOBS = ("hazel/*.sh", "hazel/slurm/*.sh")
TEST_GLOBS = ("tests/*.py",)

# Bundle files load_bundle reads. model_features.csv / model_table.csv /
# pca_loadings.csv / audit.json are documentation of how the bundle was built,
# so they are carried only when `full_inputs` is set.
BUNDLE_REQUIRED = ("config.json", "reactions.csv", "ligand_features.csv",
                   "pca_reference.json", "pca_reference.npz")
BUNDLE_OPTIONAL = ("ligand_mapping.csv", "manifest.json")
BUNDLE_EXTRA = ("audit.json", "model_features.csv", "model_table.csv", "pca_loadings.csv")

# Slurm backfills against REQUESTED time, not actual time, so an over-generous
# walltime is not free -- it is the thing that keeps a short job out of the small
# gaps it would otherwise slot into. Measured ceiling across every task of the
# three completed benchmarks (cuda, ard=true, 400 epochs) is 171.6 s, so 20
# minutes is roughly 7x headroom on the slowest task ever observed here.
MAX_WALLTIME = "00:20:00"

DEFAULT_HPC = {
    "conda_env": "/usr/local/usrapps/ddomlab/kagoble/gp_collab_hub_py312",
    "walltime": MAX_WALLTIME,
    "cpu_walltime": MAX_WALLTIME,
    "collect_walltime": MAX_WALLTIME,
    "gpu_request": "gpu:l40:1",
    "mem": "16G",
    "cpus_per_task": 4,
    "pilot": 0,
    "submit_collection": 1,
}

WALLTIME_KEYS = ("walltime", "cpu_walltime", "collect_walltime")


def walltime_seconds(value: str) -> int:
    """Seconds in an HH:MM:SS (or MM:SS, or D-HH:MM:SS) Slurm time string."""
    text = str(value).strip()
    days, _, rest = text.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return ((int(days) if days else 0) * 86400) + hours * 3600 + minutes * 60 + seconds


def cap_walltime(value: str, label: str = "walltime") -> str:
    """Clamp one time string to MAX_WALLTIME, saying so when it bites."""
    if walltime_seconds(value) <= walltime_seconds(MAX_WALLTIME):
        return value
    print(f"  {label} {value} exceeds the {MAX_WALLTIME} cap -> {MAX_WALLTIME} "
          f"(raise gpc.bundle.MAX_WALLTIME if a task really needs longer)")
    return MAX_WALLTIME


def hpc_block(cfg: dict) -> dict:
    """The scheduling settings, with anything unset falling back to the defaults.

    Every time request is capped at MAX_WALLTIME, including a per-model override,
    so a stale config carried over from an earlier run cannot quietly ask for
    four hours again.
    """
    block = {**DEFAULT_HPC, **(cfg.get("hpc") or {})}
    for key in WALLTIME_KEYS:
        block[key] = cap_walltime(block[key], f"hpc.{key}")
    return block


def run_tag(cfg: dict) -> str:
    """`<dataset>_<YYYYMMDD>` unless the config names one explicitly.

    The run tag, the folder the cluster writes into and the results folder the
    report writes all carry this same name, so a result can be traced back to
    the job that made it without consulting a log.
    """
    if cfg.get("run_tag"):
        return cfg["run_tag"]
    return f"{cfg.get('dataset', 'dataset')}_{_date.today().strftime('%Y%m%d')}"


def _retarget_scripts(text: str, tag: str, hpc: dict) -> str:
    """Rewrite the settings block of a submit script in place.

    Anchored at the start of a line so only the assignment block is touched and
    nothing inside the heredoc'd SBATCH body is rewritten by accident.
    """
    substitutions = {
        "run_tag": f'"{tag}"',
        "conda_env": f'"{hpc["conda_env"]}"',
        "gpu_request": f'"{hpc["gpu_request"]}"',
        "pilot": str(int(hpc["pilot"])),
        "submit_collection": str(int(hpc["submit_collection"])),
    }
    for key, value in substitutions.items():
        text = re.sub(rf'^{key}=.*$', f"{key}={value}", text, flags=re.MULTILINE)
    # The CPU fallback needs its own, longer budget -- the same fits take longer
    # without a GPU, and handing it the GPU walltime means the jobs are killed
    # partway through. A script is the CPU one exactly when it never asks for a
    # GPU, which is the `gpu_request=` assignment. (Testing for
    # "--partition=compute" instead does not work: that string is inside the
    # SBATCH heredoc of both scripts' collection job.)
    is_cpu = re.search(r'^gpu_request=', text, flags=re.MULTILINE) is None
    walltime = hpc["cpu_walltime"] if is_cpu else hpc["walltime"]
    text = re.sub(r'^walltime=.*$', f'walltime="{walltime}"', text, flags=re.MULTILINE)
    # The chained collection job carries its own fixed request inside the SBATCH
    # heredoc, so it is not covered by the `walltime` variable above and would
    # otherwise stay at the template's 30 minutes.
    text = re.sub(r'^#SBATCH --time=00:30:00$',
                  f"#SBATCH --time={hpc['collect_walltime']}",
                  text, flags=re.MULTILINE)
    # The training job's resources. The collection job further down asks for 1
    # CPU and 8G and is left alone; only the first occurrence is the trainer.
    text = re.sub(r'^#SBATCH --cpus-per-task=4$',
                  f"#SBATCH --cpus-per-task={int(hpc['cpus_per_task'])}",
                  text, count=1, flags=re.MULTILINE)
    text = re.sub(r'^#SBATCH --mem=16G$', f"#SBATCH --mem={hpc['mem']}",
                  text, count=1, flags=re.MULTILINE)
    return text


def _subset_ligand_features(path: Path, kraken_ids) -> bytes:
    """Only the rows for ligands this dataset actually screens.

    Features are joined per reaction on kraken_id and the reference PCA is
    already fitted and shipped separately, so the other 1,200-odd Kraken rows
    are dead weight in a training zip -- about 2.8 MB of it.
    """
    frame = pd.read_csv(path)
    subset = frame[frame.kraken_id.isin(list(kraken_ids))]
    buffer = io.StringIO()
    subset.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def dataset_dir(cfg: dict, dataset: str | None = None) -> Path:
    name = dataset or cfg.get("dataset")
    if not name:
        raise ValueError("No dataset: set 'dataset' in the run config or pass --dataset")
    directory = Path(name)
    if not directory.is_absolute():
        directory = ROOT / "datasets" / name
    inputs = directory / "inputs"
    if not (inputs / "config.json").exists():
        raise FileNotFoundError(
            f"No prepared bundle at {inputs}. Run that dataset's intake notebook "
            f"(datasets/{name}/prepare_{name}.ipynb) first.")
    return inputs


def check_methods(methods, prepared: dict) -> None:
    """Refuse a method list that does not fit this bundle.

    `iid_stratified_<n>` exists to be LOLO's matched in-distribution control:
    same fold count, same splitter family, differing only in whether the
    held-out group was seen in training. Carrying an 8-fold control over to a
    5-group screen still runs, and still produces a number -- one that quietly
    is not the control it is labelled as. That is worth failing over.
    """
    n_groups = prepared["data"].get("expected_groups",
                                    prepared["data"].get("expected_ligands"))
    if not n_groups:
        return
    for method in methods:
        if method == "lolo":
            continue
        n_splits, _ = parse_method(method)
        if method.startswith("iid_stratified_") and n_splits != n_groups:
            raise ValueError(
                f"{method} is not LOLO's matched control for this bundle: LOLO is "
                f"{n_groups} folds here ({n_groups} groups), so the matched control "
                f"is iid_stratified_{n_groups}. Change run.methods, or drop the "
                f"matched control if that is not the comparison you want.")
    # Nothing is checked about the ordinary k-folds. StratifiedKFold needs n
    # members of each class, not n classes, so `kfold_stratified_5` on a
    # four-ligand screen is fine -- Ahneman runs exactly that. sklearn raises
    # its own clear error if a group really is too small.


def write_zip(cfg: dict, dataset: str | None = None, out=None, config_path=None,
              full_inputs: bool = False, include_reference: bool = False) -> Path:
    """Write the HPC zip and return its path."""
    cfg = validate_config(dict(cfg))
    inputs = dataset_dir(cfg, dataset)
    prepared = read_json(inputs / "config.json")
    if dataset and dataset != cfg.get("dataset"):
        # `--dataset` picks the inputs, so it has to pick the name too. Letting
        # them disagree produces a zip full of one screen's data labelled with
        # another's, which survives all the way into a results folder. The
        # bundle's own method list comes with it, since a fold count that suited
        # the previous dataset rarely suits this one.
        cfg = {**cfg, "dataset": dataset, "run_tag": None}
        cfg["run"] = {**cfg["run"], "methods": list(prepared["evaluation"]["methods"])}
        print(f"--dataset {dataset}: methods taken from its bundle -> "
              f"{cfg['run']['methods']}")
    check_methods(cfg["run"]["methods"], prepared)
    tag = run_tag(cfg)
    hpc = hpc_block(cfg)

    # Inside the zip the bundle is always `inputs/` and the runs go to
    # `runs/<tag>/`, so the scripts and the CLI agree without any extra flags.
    resolved = dict(cfg)
    resolved["run_tag"] = tag
    resolved["paths"] = {"bundle": "inputs", "runs": f"runs/{tag}"}
    resolved["hpc"] = hpc
    # A per-model walltime override is read straight out of this config by the
    # submit script, so it bypasses the script rewriting above -- cap it here or
    # one section could still ask for hours.
    overrides = resolved.get("run", {}).get("model_overrides") or {}
    if any("walltime" in block for block in overrides.values()):
        resolved["run"] = {**resolved["run"], "model_overrides": {
            model: ({**block, "walltime": cap_walltime(
                        block["walltime"], f"model_overrides.{model}.walltime")}
                    if "walltime" in block else block)
            for model, block in overrides.items()}}
    resolved["built_from"] = str(Path(config_path).name) if config_path else None
    resolved["built_on"] = _date.today().isoformat()

    reactions = pd.read_csv(inputs / "reactions.csv", keep_default_na=False)
    used_ids = sorted(set(pd.to_numeric(reactions["kraken_id"], errors="coerce").dropna().astype(int)))

    destination = Path(out) if out else ROOT / "exports" / f"{tag}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        def add_bytes(arcname: str, payload: bytes):
            archive.writestr(f"{tag}/{arcname}", payload)
            manifest.append({"path": arcname, "bytes": len(payload)})

        def add_file(source: Path, arcname: str):
            add_bytes(arcname, source.read_bytes())

        for pattern in ENGINE_GLOBS:
            for source in sorted(ROOT.glob(pattern)):
                add_file(source, source.relative_to(ROOT).as_posix())
        for name in SUPPORT_FILES:
            if (ROOT / name).exists():
                add_file(ROOT / name, name)
        for pattern in TEST_GLOBS:
            for source in sorted(ROOT.glob(pattern)):
                add_file(source, source.relative_to(ROOT).as_posix())

        for pattern in SCRIPT_GLOBS:
            for source in sorted(ROOT.glob(pattern)):
                text = source.read_text(encoding="utf-8")
                add_bytes(source.relative_to(ROOT).as_posix(),
                          _retarget_scripts(text, tag, hpc).encode("utf-8"))

        names = list(BUNDLE_REQUIRED) + list(BUNDLE_OPTIONAL)
        if full_inputs:
            names += list(BUNDLE_EXTRA)
        for name in names:
            source = inputs / name
            if not source.exists():
                if name in BUNDLE_REQUIRED:
                    raise FileNotFoundError(f"{inputs} is missing {name}")
                continue
            if name == "ligand_features.csv" and not full_inputs:
                add_bytes("inputs/ligand_features.csv",
                          _subset_ligand_features(source, used_ids))
            else:
                add_file(source, f"inputs/{name}")

        if include_reference:
            for source in sorted((ROOT / "reference" / "kraken").glob("*")):
                if source.is_file():
                    add_file(source, f"reference/kraken/{source.name}")

        # configs/default.json is what `gpc` and the scripts read with no flags.
        # The tagged copy beside it is for the record, so a zip found later still
        # says which config it was built from.
        payload = (json.dumps(resolved, indent=2) + "\n").encode("utf-8")
        add_bytes("configs/default.json", payload)
        add_bytes(f"configs/{tag}.json", payload)
        add_bytes("RUN_ON_HPC.md", _instructions(tag, hpc, resolved).encode("utf-8"))
        archive.writestr(f"{tag}/MANIFEST.json",
                         json.dumps({"run_tag": tag, "dataset": cfg.get("dataset"),
                                     "built_on": resolved["built_on"],
                                     "n_tasks": len(cfg["run"]["models"]) * len(cfg["run"]["methods"]),
                                     "models": cfg["run"]["models"],
                                     "methods": cfg["run"]["methods"],
                                     "ligand_features_rows": len(used_ids) if not full_inputs else None,
                                     "files": manifest}, indent=2) + "\n")

    total = destination.stat().st_size / 1024
    print(f"{destination}  ({total:,.0f} KB, {len(manifest) + 1} files)")
    print(f"  run tag   {tag}")
    print(f"  tasks     {len(cfg['run']['models']) * len(cfg['run']['methods'])} "
          f"({len(cfg['run']['models'])} sections x {len(cfg['run']['methods'])} methods)")
    print(f"  walltime  {hpc['walltime']} per task on {hpc['gpu_request']}")
    return destination


def _instructions(tag: str, hpc: dict, cfg: dict) -> str:
    n_tasks = len(cfg["run"]["models"]) * len(cfg["run"]["methods"])
    return f"""# {tag}

Built by the DOPE-MURI hub on {cfg['built_on']} for dataset `{cfg.get('dataset')}`.
{n_tasks} jobs: {len(cfg['run']['models'])} feature sections x {len(cfg['run']['methods'])} evaluation methods.

Upload, unzip, and from inside `{tag}/`:

    bash hazel/setup_environment.sh            # once, on a login node
    bash hazel/slurm/GP_preflight_GPU_arg.sh   # GPU visibility + float64 Cholesky
    bash hazel/slurm/GP_GPU_arg.sh             # the {n_tasks} jobs, collection chained behind

Settings already written into the scripts:

    run tag      {tag}          -> runs/{tag}/
    conda env    {hpc['conda_env']}
    GPU          {hpc['gpu_request']}
    walltime     {hpc['walltime']} per task
    memory       {hpc['mem']}, {hpc['cpus_per_task']} CPUs per task
    pilot        {hpc['pilot']}   (1 = 10 epochs into runs/{tag}_pilot, to size walltime)

Bring back the whole `runs/{tag}/` folder. In the hub:

    python -m gpc --runs <that folder> --bundle datasets/{cfg.get('dataset')}/inputs report

which writes `results/{cfg.get('dataset')}_<date>/` with the figures and tables.

`inputs/ligand_features.csv` holds only the ligands this screen uses; the full
Kraken reference stays in the hub, where the figures are drawn.
"""
