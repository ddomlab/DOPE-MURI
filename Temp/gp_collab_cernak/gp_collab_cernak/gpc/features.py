"""The one feature section, and the per-fold preprocessor that encodes it.

The Cernak screen has no computed chemistry: it varies the catalyst and
four reaction conditions, each of them a discrete screen level. So there is no
numeric block to impute or scale, and the whole design is one-hot --
`catalyst_ohe` is the catalyst plus every condition.

The preprocessor is handed to the sklearn Pipeline rather than fitted up front,
so GP_collab's cross-validation fits the encoder on the training fold of each
split. That matters for `lolo`: the held-out catalyst is absent from the
training fold, so its column does not exist there, and `handle_unknown="ignore"`
encodes it as all-zeros at predict time rather than failing.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def model_columns(model: str, cfg: dict) -> tuple[list, list]:
    """(numeric columns, categorical fields) for a feature section."""
    categorical = list(cfg["data"]["common_categorical"])
    group = cfg["data"]["group"]
    if model == "catalyst_ohe":
        return [], [group] + categorical
    raise ValueError(f"Unknown feature section {model!r}")


def feature_frame(reactions: pd.DataFrame, model: str, cfg: dict) -> pd.DataFrame:
    """The columns `model` is fitted on, in the bundle's canonical row order."""
    num, cat = model_columns(model, cfg)
    missing = [c for c in num + cat if c not in reactions.columns]
    if missing:
        raise ValueError(f"{model}: bundle has no column(s) {missing}")
    if cat and reactions[cat].isna().any().any():
        raise ValueError(f"{model}: missing categorical inputs; resolve before training")
    return reactions[num + cat].copy()


def _preprocessor(frame: pd.DataFrame, model: str, cfg: dict) -> ColumnTransformer:
    """The single definition of model inputs, shared by training folds and review."""
    num, cat = model_columns(model, cfg)
    vocab = "auto"
    if cfg["features"]["ohe_policy"] == "declared_vocabulary":
        vocab = [sorted(frame[c].unique().tolist()) for c in cat]
    transformers = []
    # There is no numeric block in this bundle today. An empty ColumnTransformer
    # entry would fit a scaler on nothing, so only add one if a section ever has
    # numeric columns.
    if num:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="mean")),
                                              ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(categories=vocab, handle_unknown="ignore",
                                  sparse_output=False), cat))
    if not transformers:
        raise ValueError(f"{model}: no inputs to encode")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
