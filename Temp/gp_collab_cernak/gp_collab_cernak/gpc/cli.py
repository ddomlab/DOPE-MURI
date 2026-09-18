"""Command line: tasks, splits, train, collect, check-env."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .collect import write_tables
from .config import GROUP, load_config
from .data import load_bundle
from .splits import fold_table

ROOT = Path(__file__).resolve().parent.parent


def task_table(cfg: dict) -> pd.DataFrame:
    """The job registry: one row per (feature section, evaluation method)."""
    rows = [{"task_id": i, "model": model, "method": method}
            for i, (model, method) in enumerate(
                (m, v) for m in cfg["run"]["models"] for v in cfg["run"]["methods"])]
    return pd.DataFrame(rows)


def _paths(args, cfg):
    bundle = Path(args.bundle or ROOT / cfg["paths"]["bundle"])
    runs = Path(args.runs or ROOT / cfg["paths"]["runs"])
    return bundle, runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gpc")
    parser.add_argument("--config", default=str(ROOT / "configs/default.json"))
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--runs", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tasks", help="print the task registry")
    sub.add_parser("splits", help="print the fold table for every method")
    sub.add_parser("check-env", help="report torch/CUDA and a float Cholesky")

    train = sub.add_parser("train", help="run one task")
    train.add_argument("--task-id", type=int)
    train.add_argument("--model")
    train.add_argument("--method")
    train.add_argument("--epochs", type=int, help="override n_epochs (pilot runs)")

    collect = sub.add_parser("collect", help="gather finished tasks into tables")
    collect.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    bundle, runs = _paths(args, cfg)

    if args.command == "tasks":
        print(task_table(cfg).to_string(index=False))
    elif args.command == "splits":
        reactions, prepared = load_bundle(bundle)
        group = prepared["data"].get("group", GROUP)
        table = pd.concat([fold_table(reactions, m, cfg["run"]["seed"],
                                      cfg["run"]["stratify"], group)
                           for m in cfg["run"]["methods"]], ignore_index=True)
        print(table.to_string(index=False))
    elif args.command == "check-env":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(64, 64, dtype=torch.float64, device=device)
        torch.linalg.cholesky(a @ a.T + 64 * torch.eye(64, dtype=torch.float64, device=device))
        print(f"torch {torch.__version__} cuda={torch.version.cuda} device={device} "
              f"gpus={torch.cuda.device_count()} cholesky=ok")
    elif args.command == "train":
        if args.task_id is not None:
            row = task_table(cfg).iloc[args.task_id]
            model, method = row["model"], row["method"]
        elif args.model and args.method:
            model, method = args.model, args.method
        else:
            parser.error("train needs --task-id, or both --model and --method")
        if args.epochs:
            cfg["gp"]["n_epochs"] = args.epochs
        from .train import run_task  # torch is only needed to train

        run_task(bundle, runs, model, method, cfg)
    elif args.command == "collect":
        write_tables(runs, args.out)
    return 0
