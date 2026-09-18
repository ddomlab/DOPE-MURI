"""Load the prepared Cernak bundle.

One table. Every column the model sees is in `reactions.csv`; the chemistry that
identifies a catalyst sits in `catalyst_mapping.csv` and is never a
feature, so it is not loaded here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import GROUP, TARGET, read_json


def load_bundle(bundle: str | Path):
    """Return (reactions, prepared_config)."""
    bundle = Path(bundle)
    prepared = read_json(bundle / "config.json")
    target = prepared["data"].get("target", TARGET)
    group = prepared["data"].get("group", GROUP)

    reactions = pd.read_csv(bundle / "reactions.csv", keep_default_na=False)
    reactions[target] = pd.to_numeric(reactions[target], errors="raise")

    expected = prepared["data"]["expected_rows"]
    if len(reactions) != expected:
        raise ValueError(f"{len(reactions)} rows in bundle, config expects {expected}")
    if reactions[group].nunique() != prepared["data"]["expected_groups"]:
        raise ValueError("Unexpected catalyst count in bundle")
    if reactions[target].isna().any():
        raise ValueError("Missing target values in bundle")
    missing = [c for c in prepared["data"]["common_categorical"] if c not in reactions]
    if missing:
        raise ValueError(f"Bundle is missing condition columns: {missing}")
    return reactions, prepared
