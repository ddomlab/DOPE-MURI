"""Feature sections for the Cernak screen, and the per-fold preprocessor.

Ported from gp_collab_hazel/gpc/features.py so the five sections mean exactly
the same thing across all three benchmarks. The Cernak catalysts are Buchwald
precatalysts of monodentate phosphines, so they carry Kraken descriptors joined
on kraken_id, and the reference PCA is the one fitted on the full Kraken set for
the Perera runs -- reused unrefitted, exactly as the rxnpredict bundle does, so
PC1..PC4 mean the same axes in every project.

The group column here is `catalyst`, not `ligand`; the one-hot section is named
`catalyst_ohe` for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FIVE, TWO, read_json


@dataclass
class ReferencePCA:
    columns: list[str]
    impute_mean: np.ndarray
    scale_mean: np.ndarray
    scale_std: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    variance_ratio: np.ndarray
    top_by_pc: dict[str, list[str]]

    @classmethod
    def fit(cls, descriptors: pd.DataFrame, top_k: int = 3):
        x = descriptors.drop(columns="id").apply(pd.to_numeric, errors="raise")
        x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        if min(x.shape) < 4:
            raise ValueError("PCA needs at least four descriptor columns and four reference ligands")
        means = x.mean().to_numpy()
        filled = np.where(np.isfinite(x.to_numpy()), x.to_numpy(), means)
        scaler = StandardScaler().fit(filled)
        z = scaler.transform(filled)
        pca = PCA(n_components=4, svd_solver="full").fit(z)
        # Stable absolute-loading order; break exact ties by original column order.
        tops = {f"PC{i+1}": x.columns[np.argsort(-np.abs(c), kind="stable")[:top_k]].tolist()
                for i, c in enumerate(pca.components_)}
        return cls(x.columns.tolist(), means, scaler.mean_, scaler.scale_, pca.mean_,
                   pca.components_, pca.explained_variance_ratio_, tops)

    @property
    def top_features(self) -> list[str]:
        return list(dict.fromkeys(c for cols in self.top_by_pc.values() for c in cols))

    def transform(self, descriptors: pd.DataFrame) -> np.ndarray:
        x = descriptors[self.columns].to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, self.impute_mean)
        return ((x - self.scale_mean) / self.scale_std - self.pca_mean) @ self.components.T

    @classmethod
    def load(cls, directory: Path):
        meta = read_json(directory / "pca_reference.json")
        with np.load(directory / "pca_reference.npz", allow_pickle=False) as a:
            return cls(meta["columns"], **{k: a[k] for k in (
                "impute_mean", "scale_mean", "scale_std", "pca_mean", "components", "variance_ratio")},
                       top_by_pc=meta["top_by_pc"])


class LoadingExpansion(BaseEstimator, TransformerMixin):
    """Apply each PC loading to each descriptor separately, without summing them.

    Output column `PC{i}_x_{descriptor}` is `loading[i, j] * z_j` for the fold-standardized
    descriptor `z_j`. Adding the descriptor columns of one component reproduces that
    component's score exactly, so this keeps every quantity the PC scores are built from
    and performs no dimensional reduction: `d` descriptors become `d * n_components`
    inputs rather than `n_components`.

    Loadings are applied after the fold scaler on purpose. A StandardScaler placed after
    this step would rescale every column to unit variance and erase the loading weights.
    """

    def __init__(self, components=None, descriptors=None):
        self.components = components
        self.descriptors = descriptors

    def fit(self, X, y=None):
        components = np.asarray(self.components, dtype=float)
        if components.ndim != 2 or components.shape[1] != np.asarray(X).shape[1]:
            raise ValueError("Loading matrix does not match the descriptor block it weights")
        if self.descriptors is not None and len(self.descriptors) != components.shape[1]:
            raise ValueError("Descriptor names do not match the loading matrix")
        self.n_features_in_ = components.shape[1]
        return self

    def transform(self, X):
        z = np.asarray(X, dtype=float)
        components = np.asarray(self.components, dtype=float)
        if z.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected descriptor count at transform time")
        return (z[:, None, :] * components[None, :, :]).reshape(len(z), -1)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else list(self.descriptors)
        return np.asarray([f"PC{i + 1}_x_{name}"
                           for i in range(len(np.asarray(self.components))) for name in names],
                          dtype=object)


def pc_scores_form(cfg: dict) -> str:
    """Absent key means the original four summed component scores, so old bundles are stable."""
    return cfg["features"].get("pc_scores_form", "component_scores")


# A long-budget variant must differ in GP hyperparameters ONLY, never in features --
# that is what keeps it comparable to its short-budget twin. Resolving the alias here
# means the pc_scores branches below (and _expands_loadings) need no duplication.
MODEL_FEATURE_ALIAS = {"pc_scores_long": "pc_scores"}


def feature_alias(model: str) -> str:
    """The model whose FEATURES `model` uses; identity for everything unaliased."""
    return MODEL_FEATURE_ALIAS.get(model, model)


def model_columns(model: str, cfg: dict, reference: ReferencePCA) -> tuple[list, list]:
    model = feature_alias(model)
    categorical = list(cfg["data"]["common_categorical"])
    group = cfg["data"].get("group", "catalyst")
    needs_reference = model == "pc_top" or (model == "pc_scores" and pc_scores_form(cfg) == "loading_weighted")
    if needs_reference and reference is None:
        raise ValueError(f"{model} requires the fitted reference PCA")
    # loading_weighted feeds the raw descriptors so the fold scaler standardizes them
    # before LoadingExpansion applies each component's loadings.
    scores = (list(reference.columns) if reference else []) \
        if pc_scores_form(cfg) == "loading_weighted" else [f"PC{i}" for i in range(1, 5)]
    numeric = {"catalyst_ohe": [], "selected_5": FIVE, "selected_2": TWO,
               "pc_top": reference.top_features if reference else [],
               "pc_scores": scores}[model]
    if model == "catalyst_ohe":
        categorical = [group] + categorical
    return list(numeric), categorical


def feature_frame(reactions: pd.DataFrame, ligands: pd.DataFrame, model: str,
                  cfg: dict, reference: ReferencePCA) -> pd.DataFrame:
    num, cat = model_columns(model, cfg, reference)
    merged = reactions.merge(ligands[["kraken_id"] + num], on="kraken_id", how="left",
                             sort=False, validate="many_to_one")
    if merged[cat].isna().any().any() or len(merged) != len(reactions):
        raise ValueError("Incomplete categorical inputs or expanding feature merge")
    if num and not np.isfinite(merged[num].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing/nonfinite selected ligand features for {model}; resolve before training")
    return merged[num + cat]


def _expands_loadings(model: str, cfg: dict) -> bool:
    return feature_alias(model) == "pc_scores" and pc_scores_form(cfg) == "loading_weighted"


def _numeric_pipeline(model: str, cfg: dict, reference: ReferencePCA) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    if _expands_loadings(model, cfg):
        steps.append(("loadings", LoadingExpansion(reference.components, list(reference.columns))))
    return Pipeline(steps)


def _preprocessor(frame: pd.DataFrame, model: str, cfg: dict, reference: ReferencePCA) -> ColumnTransformer:
    """The single definition of model inputs, shared by training folds and review."""
    num, cat = model_columns(model, cfg, reference)
    vocab = "auto"
    if cfg["features"]["ohe_policy"] == "declared_vocabulary":
        vocab = [sorted(frame[c].unique().tolist()) for c in cat]
    return ColumnTransformer([
        ("num", _numeric_pipeline(model, cfg, reference), num),
        ("cat", OneHotEncoder(categories=vocab, handle_unknown="ignore", sparse_output=False), cat),
    ], remainder="drop", sparse_threshold=0)

