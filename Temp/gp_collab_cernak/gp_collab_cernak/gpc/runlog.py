"""A dated, append-only log of the hyperparameters every training task ran with.

Each task already writes its own `meta.json` holding the full config, so the
per-task record exists. What is missing -- and what this file adds -- is one
place to compare parameters *between* runs:

    <runs_dir>/run_parameters_<YYYY-MM-DD>.jsonl

one JSON object per line. A record is appended as a single whole-line write and
no line is ever rewritten, so concurrent SLURM tasks cannot clobber each other.
Each record carries `gp_hash`, the first 12 hex characters of the sha256 of the
canonical JSON of the `gp` block, so identical hyperparameter sets group with a
plain equality test.

Writer side is `append_run_log`, called from `gpc.train.run_task`; its whole
body is guarded so a logging failure warns and returns instead of killing a
fit. Reader side is `read_run_logs` plus `differing_columns` / `run_log_diff`,
which reduce the log to the columns that actually changed -- the "what is
different between these runs" view the review notebook wants.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

LOG_PREFIX = "run_parameters_"
LOG_GLOB = LOG_PREFIX + "*.jsonl"

# What identifies a record rather than describing its hyperparameters. The diff
# view always shows these, whether or not they vary.
ID_COLUMNS = ("timestamp", "model", "method", "gp_hash")

# Bookkeeping that differs between records by construction, so it is never
# interesting as a "what changed" answer.
DIFF_IGNORE = ("timestamp", "log_file", "line_no")


def _flatten(value, key=""):
    """Nested dicts -> dotted keys; scalars and lists are left as they are."""
    if isinstance(value, dict):
        flat = {}
        for name, sub in value.items():
            flat.update(_flatten(sub, f"{key}.{name}" if key else str(name)))
        return flat
    return {key: value}


def gp_hash(gp: dict, length: int = 12) -> str:
    """Short stable fingerprint of a `gp` config block."""
    canonical = json.dumps(gp, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def build_record(model: str, method: str, cfg: dict, extra: dict | None = None,
                 now: datetime | None = None) -> dict:
    """The one JSON object a task contributes to the log.

    The full `gp` block is flattened under `gp.*`, and `run.seed` /
    `run.stratify` are kept beside it because they change what was fitted just
    as much as a kernel setting does. Anything in `extra` (`n_features`, say)
    is merged in last.
    """
    now = now or datetime.now()
    gp = dict(cfg.get("gp") or {})
    run = dict(cfg.get("run") or {})
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "model": model,
        "method": method,
        "gp_hash": gp_hash(gp),
        "seed": run.get("seed"),
        "stratify": run.get("stratify"),
    }
    record.update(_flatten(gp, "gp"))
    for name, value in (extra or {}).items():
        record[str(name)] = value
    return record


def log_path(runs_dir, now: datetime | None = None) -> Path:
    """Today's log file inside `runs_dir` (not created here)."""
    now = now or datetime.now()
    return Path(runs_dir) / f"{LOG_PREFIX}{now.strftime('%Y-%m-%d')}.jsonl"


def append_run_log(runs_dir, model: str, method: str, cfg: dict,
                   extra: dict | None = None):
    """Append one line describing the hyperparameters this task ran with.

    Returns the log path, or None if anything went wrong. This sits in the
    training path, so it never raises: a broken log must not cost a fit.
    """
    try:
        now = datetime.now()
        record = build_record(model, method, cfg, extra, now)
        path = log_path(runs_dir, now)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        # "a" is O_APPEND, and one write of one whole line keeps parallel tasks
        # from interleaving. newline="\n" so a Windows and a SLURM run agree.
        with open(path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
        return path
    except Exception as exc:  # never break training over a log line
        print(f"[runlog] could not write the run parameter log: {exc!r}",
              file=sys.stderr)
        return None


def log_files(runs_dir) -> list:
    """Every run_parameters_*.jsonl at or below `runs_dir`, sorted by path."""
    directory = Path(runs_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.rglob(LOG_GLOB))


def read_run_logs(runs_dir):
    """One row per logged task, across every dated log under `runs_dir`."""
    import pandas as pd

    rows = []
    for path in log_files(runs_dir):
        try:
            # Text mode: universal newlines reads an LF or a CRLF log alike.
            with open(path, "r", encoding="utf-8") as stream:
                lines = stream.readlines()
        except OSError as exc:
            print(f"[runlog] could not read {path}: {exc!r}", file=sys.stderr)
            continue
        for number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                print(f"[runlog] skipping unreadable line {number} of {path}",
                      file=sys.stderr)
                continue
            if not isinstance(record, dict):
                continue
            record = dict(record)
            record["log_file"] = path.name
            record["line_no"] = number
            rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    front = [c for c in ID_COLUMNS if c in frame.columns]
    back = [c for c in ("log_file", "line_no") if c in frame.columns]
    middle = [c for c in frame.columns if c not in front and c not in back]
    frame = frame.loc[:, front + middle + back]
    sort_on = [c for c in ("timestamp", "log_file", "line_no") if c in frame.columns]
    if sort_on:
        frame = frame.sort_values(sort_on, kind="stable").reset_index(drop=True)
    return frame


def _comparable(series):
    """Values as canonical JSON text, so lists and None compare like scalars."""
    return series.map(lambda value: json.dumps(value, sort_keys=True, default=str))


def differing_columns(frame, ignore=DIFF_IGNORE) -> list:
    """The columns whose value is not the same in every record."""
    if frame is None or len(frame) == 0:
        return []
    skip = set(ignore or ())
    return [c for c in frame.columns
            if c not in skip and _comparable(frame[c]).nunique(dropna=False) > 1]


def run_log_diff(logs, ignore=DIFF_IGNORE, id_columns=ID_COLUMNS):
    """The "what changed between these runs" view.

    `logs` is either a runs directory or a frame from `read_run_logs`. The
    result keeps the identifying columns plus only those that actually vary, so
    a wide, mostly constant log collapses to the handful of settings that moved.
    """
    import pandas as pd

    frame = logs if isinstance(logs, pd.DataFrame) else read_run_logs(logs)
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    front = [c for c in id_columns if c in frame.columns]
    changed = [c for c in differing_columns(frame, ignore) if c not in front]
    return frame.loc[:, front + changed]
