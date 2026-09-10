"""Run from the project root: python -m hazel_gp --help."""
import argparse
import json
from pathlib import Path
from .config import load_config, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/default.json")
        p.add_argument("--root", default=".")
        if name == "download":
            p.add_argument("--refresh", action="store_true")
        else:
            p.add_argument("--output", help="New prepared directory; defaults to config")
    for name in ("verify", "tasks", "status", "run", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", default="inputs")
        if name in ("status", "run", "collect"):
            p.add_argument("--runs", default="runs/experiment_v1")
        if name == "run":
            p.add_argument("--task-id", type=int, required=True)
            p.add_argument("--device", choices=["auto", "cpu", "cuda"])
            p.add_argument("--max-steps", type=int, help="Pilot override; use a separate runs directory")
            p.add_argument("--retry", action="store_true")
        if name == "tasks":
            p.add_argument("--count", action="store_true")
        if name == "collect":
            p.add_argument("--allow-partial", action="store_true")
    p = sub.add_parser("check-env")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p = sub.add_parser("pack-inputs")
    p.add_argument("--bundle", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    p = sub.add_parser("pack-results")
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--include-checkpoints", action="store_true")
    a = parser.parse_args(argv)
    if a.command == "download":
        from .data import download_sources
        print(json.dumps(download_sources(load_config(a.config), a.root, a.refresh), indent=2))
    elif a.command == "prepare":
        from .data import prepare_data, export_prepared
        cfg = load_config(a.config)
        prepared = prepare_data(cfg, a.root)
        print(json.dumps(prepared.audit, indent=2))
        print(export_prepared(prepared, a.output or Path(a.root) / cfg["paths"]["prepared"]))
    elif a.command == "verify":
        from .data import verify_bundle
        print(json.dumps(verify_bundle(a.bundle), indent=2))
    elif a.command == "tasks":
        import pandas as pd
        from .data import verify_bundle
        meta = verify_bundle(a.bundle)
        print(meta["n_tasks"] if a.count else pd.read_csv(Path(a.bundle) / "tasks.csv").to_string(index=False))
    elif a.command == "status":
        from .results import task_statuses
        print(task_statuses(a.bundle, a.runs).to_string(index=False))
    elif a.command == "run":
        from .train import run_task
        print(json.dumps(run_task(a.bundle, a.runs, a.task_id, a.device, a.max_steps, a.retry), indent=2))
    elif a.command == "collect":
        from .results import collect_results
        tables = collect_results(a.bundle, a.runs, a.allow_partial)
        print(tables["summary"].to_string(index=False))
    elif a.command == "check-env":
        from .train import check_environment
        print(json.dumps(check_environment(a.device), indent=2))
    elif a.command == "pack-inputs":
        from .data import create_upload_archive
        print(create_upload_archive(a.root, a.bundle, a.output))
    elif a.command == "pack-results":
        from .results import export_result_archive
        print(export_result_archive(a.runs, a.output, a.include_checkpoints))
