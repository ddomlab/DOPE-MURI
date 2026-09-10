"""Frozen original-population PCA and train-only reaction transformations."""
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

from .config import FIVE, TWO, write_json, read_json


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

    def save(self, directory: Path) -> None:
        np.savez_compressed(directory / "pca_reference.npz", impute_mean=self.impute_mean,
                            scale_mean=self.scale_mean, scale_std=self.scale_std,
                            pca_mean=self.pca_mean, components=self.components,
                            variance_ratio=self.variance_ratio)
        write_json(directory / "pca_reference.json", {
            "scope": "full_original_kraken", "columns": self.columns,
            "top_by_pc": self.top_by_pc, "top_features": self.top_features,
            "meaning": "External unlabeled reference; may include held-out ligand covariates."})
        pd.DataFrame(self.components.T, index=self.columns,
                     columns=[f"PC{i}" for i in range(1, 5)]).rename_axis("descriptor").to_csv(
                         directory / "pca_loadings.csv")

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


def ligand_table(descriptors: pd.DataFrame, reference: ReferencePCA, cfg: dict) -> pd.DataFrame:
    """Keep raw quantities untouched; add consistent derived units and four scores."""
    out = descriptors.copy()
    sphere = (4.0 / 3.0) * np.pi * cfg["data"]["sphere_radius_angstrom"] ** 3
    for suffix in ("boltz", "min", "delta"):
        out[f"vbur_pct_{suffix}"] = 100.0 * out[f"vbur_vbur_{suffix}"] / sphere
    out["homo_lumo_gap_eV"] = (out["fmo_e_lumo_boltz"] - out["fmo_e_homo_boltz"]) * cfg["data"]["hartree_to_eV"]
    out[[f"PC{i}" for i in range(1, 5)]] = reference.transform(descriptors)
    return out.rename(columns={"id": "kraken_id"})


def model_columns(model: str, cfg: dict, reference: ReferencePCA) -> tuple[list, list]:
    categorical = list(cfg["data"]["common_categorical"])
    needs_reference = model == "pc_top" or (model == "pc_scores" and pc_scores_form(cfg) == "loading_weighted")
    if needs_reference and reference is None:
        raise ValueError(f"{model} requires the fitted reference PCA")
    # loading_weighted feeds the raw descriptors so the fold scaler standardizes them
    # before LoadingExpansion applies each component's loadings.
    scores = (list(reference.columns) if reference else []) \
        if pc_scores_form(cfg) == "loading_weighted" else [f"PC{i}" for i in range(1, 5)]
    numeric = {"ligand_ohe": [], "selected_5": FIVE, "selected_2": TWO,
               "pc_top": reference.top_features if reference else [],
               "pc_scores": scores}[model]
    if model == "ligand_ohe":
        categorical = ["ligand"] + categorical
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
    return model == "pc_scores" and pc_scores_form(cfg) == "loading_weighted"


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


def fit_transform_fold(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray,
                       model: str, cfg: dict, reference: ReferencePCA):
    num, cat = model_columns(model, cfg, reference)
    pre = _preprocessor(frame, model, cfg, reference)
    xtr = np.asarray(pre.fit_transform(frame.iloc[train]), dtype=np.float64)
    xte = np.asarray(pre.transform(frame.iloc[test]), dtype=np.float64)
    unknown = {c: sorted(set(frame.iloc[test][c]) - set(frame.iloc[train][c])) for c in cat}
    return xtr, xte, pre, pre.get_feature_names_out().tolist(), unknown


def pca_loading_summary(reference: ReferencePCA) -> pd.DataFrame:
    """Loading reach of each PC score, and how much of it the displayed top descriptors carry.

    `pc_scores` uses the complete unit-norm loading vector, so every descriptor with a
    nonzero loading contributes; `top_squared_loading_share` is the fraction of that
    vector's squared length held by the three descriptors `pc_top` selects.
    """
    rows = []
    for i, component in enumerate(reference.components):
        pc = f"PC{i + 1}"
        weights = pd.Series(component, index=reference.columns)
        top = reference.top_by_pc[pc]
        rows.append({"component": pc,
                     "explained_variance_fraction": float(reference.variance_ratio[i]),
                     "descriptors_with_nonzero_loading": int((weights != 0).sum()),
                     "descriptors_available": len(reference.columns),
                     "top_squared_loading_share": float((weights[top] ** 2).sum() / (weights ** 2).sum()),
                     "top_descriptors": ", ".join(top)})
    return pd.DataFrame(rows)


def pca_loading_table(reference: ReferencePCA, top: int = 25) -> pd.DataFrame:
    """Per-descriptor PC loadings, largest absolute weight in any PC first."""
    loadings = pd.DataFrame(reference.components.T, index=reference.columns,
                            columns=[f"PC{i}" for i in range(1, 5)]).rename_axis("descriptor")
    order = loadings.abs().max(axis=1).sort_values(ascending=False, kind="stable").index
    return loadings.loc[order[:top]]


def model_feature_blocks(model: str, reactions: pd.DataFrame, ligands: pd.DataFrame,
                         cfg: dict, reference: ReferencePCA) -> pd.DataFrame:
    """Every encoded input of one model, named, with the field and block it came from.

    Categories are taken from all prepared rows, so this is the matched-IID column set.
    A LOLO training fold omits the held-out ligand's one-hot column, leaving `ligand_ohe`
    one column shorter; the shared categorical block is identical for every model.
    """
    frame = feature_frame(reactions, ligands, model, cfg, reference)
    num, cat = model_columns(model, cfg, reference)
    pre = _preprocessor(frame, model, cfg, reference).fit(frame)
    # One numeric input per descriptor, or one per (component, descriptor) when expanded.
    repeats = len(reference.components) if _expands_loadings(model, cfg) else 1
    sources = [name for _ in range(repeats) for name in num] + \
              [field for field, values in
               zip(cat, pre.named_transformers_["cat"].categories_) for _ in values]
    shared = set(cfg["data"]["common_categorical"])
    return pd.DataFrame({
        "model": model, "feature": pre.get_feature_names_out().tolist(), "source_field": sources,
        "block": ["shared_categorical" if s in shared else "ligand" for s in sources]})


def model_feature_records(prepared) -> pd.DataFrame:
    """Complete untruncated input names for every model, for the prepared bundle."""
    frames = [model_feature_blocks(model, prepared.reactions, prepared.ligands,
                                   prepared.config, prepared.reference)
              for model in prepared.config["features"]["models"]]
    out = pd.concat(frames, ignore_index=True)
    return out.assign(position=out.groupby("model").cumcount() + 1)[
        ["model", "position", "feature", "source_field", "block"]]


def model_feature_table(prepared, max_features: int = 25) -> pd.DataFrame:
    """Side-by-side review grid of named inputs, including the shared constant fields."""
    records = model_feature_records(prepared)
    columns = {}
    for model, part in records.groupby("model", sort=False):
        names = part.feature.tolist()
        shown = names[:max_features]
        counts = part.block.value_counts()
        columns[model] = shown + [""] * (max_features - len(shown)) + [
            f"{len(names)} inputs = {counts.get('ligand', 0)} ligand"
            f" + {counts.get('shared_categorical', 0)} shared"
            + (f"; {len(names) - max_features} not shown" if len(names) > max_features else "")]
    index = [str(i) for i in range(1, max_features + 1)] + ["totals"]
    return pd.DataFrame(columns, index=index).rename_axis("input")


def model_summary(cfg: dict, reference: ReferencePCA | None = None) -> pd.DataFrame:
    rows = []
    labels = {"ligand_ohe": "Ligand identity OHE", "selected_5": ", ".join(FIVE),
              "selected_2": ", ".join(TWO),
              "pc_top": "Top 3 original descriptors per PC1-PC4 (deduplicated)",
              "pc_scores": "PC1-PC4 scores; every reference descriptor at its own loading"}
    available = len(reference.columns) if reference else 190
    components = len(reference.components) if reference else cfg["features"]["pca_components"]
    if pc_scores_form(cfg) == "loading_weighted":
        labels["pc_scores"] = f"Every descriptor times its loading in each of PC1-PC{components}"
    # Columns handed to the GP, which is not the number of descriptors behind them.
    sources = {"ligand_ohe": "none; ligand identity only",
               "selected_5": "5 raw/derived descriptors",
               "selected_2": "2 raw/derived descriptors",
               "pc_top": f"{len(reference.top_features) if reference else 12} raw descriptors"
                         " (top 3 per PC; the rest are discarded)",
               "pc_scores": f"all {available} reference descriptors, each weighted by its own"
                            + (f" loading in every component; no summing, {available} x {components} inputs"
                               if pc_scores_form(cfg) == "loading_weighted"
                               else f" PC loading, compressed into {components} coordinates")}
    for model in cfg["features"]["models"]:
        n = {"ligand_ohe": "8 known categories; 7 training columns in LOLO by default",
             "selected_5": 5, "selected_2": 2,
             "pc_scores": available * components if pc_scores_form(cfg) == "loading_weighted" else components,
             "pc_top": len(reference.top_features) if reference else "up to 12"}[model]
        rows.append({"model": model, "ligand_features": labels[model],
                     "ligand_columns_into_gp": n, "descriptors_behind_those_columns": sources[model],
                     "common_inputs": ", ".join(cfg["data"]["common_categorical"])})
    return pd.DataFrame(rows)
