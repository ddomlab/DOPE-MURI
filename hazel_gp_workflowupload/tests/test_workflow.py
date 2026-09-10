"""Focused correctness checks; real data integration is optional after download/prepare."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from hazel_gp.config import default_config, load_config, FIVE, TWO
from hazel_gp.features import (ReferencePCA, ligand_table, fit_transform_fold, feature_frame,
                               model_columns, pc_scores_form)
from hazel_gp.splits import make_splits, validate_split, load_split
from hazel_gp.train import build_model, solver_context, fit_exact, predict_exact, run_task, predict_from_checkpoint
from hazel_gp.results import regression_metrics, collect_results, grouped_predictions
from hazel_gp.data import load_bundle, verify_bundle


def descriptor_fixture():
    rng = np.random.default_rng(91)
    names = ["vbur_vbur_boltz", "vbur_vbur_min", "vbur_vbur_delta", "dipolemoment_boltz",
             "fmo_e_homo_boltz", "fmo_e_lumo_boltz", "a", "b", "c", "d"]
    df = pd.DataFrame(rng.normal(size=(24, 10)), columns=names)
    df.insert(0, "id", np.arange(24))
    return df


def test_original_population_pca_and_reload(tmp_path):
    d = descriptor_fixture()
    ref = ReferencePCA.fit(d)
    z = StandardScaler().fit_transform(d.drop(columns="id"))
    expected = PCA(n_components=4, svd_solver="full").fit_transform(z)
    np.testing.assert_allclose(ref.transform(d), expected, atol=1e-12)
    ref.save(tmp_path)
    restored = ReferencePCA.load(tmp_path)
    np.testing.assert_allclose(restored.transform(d), expected, atol=1e-12)
    assert len(ref.top_features) == len(set(ref.top_features))
    assert set(ref.top_features) <= set(d.columns) - {"id"}


def test_raw_values_preserved_and_units_consistent():
    d = descriptor_fixture()
    original = d.copy()
    cfg = default_config()
    table = ligand_table(d, ReferencePCA.fit(d), cfg)
    pd.testing.assert_frame_equal(d, original)
    sphere = 4 / 3 * np.pi * 3.5**3
    for suffix in ("boltz", "min", "delta"):
        np.testing.assert_allclose(table[f"vbur_pct_{suffix}"], d[f"vbur_vbur_{suffix}"] * 100 / sphere)
    np.testing.assert_allclose(table.homo_lumo_gap_eV,
                               (d.fmo_e_lumo_boltz - d.fmo_e_homo_boltz) * cfg["data"]["hartree_to_eV"])
    assert TWO == ["vbur_pct_boltz", "vbur_pct_min"]


def test_shared_splits_counts_identity_and_repeat_independence():
    cfg = default_config()
    reactions = pd.DataFrame({"ligand": np.repeat(sorted(cfg["data"]["ligand_mapping"]), 384)})
    entries, arrays, tasks = make_splits(reactions, cfg)
    assert len(entries) == 22 and len(tasks) == 110
    assert tasks.groupby("model").size().eq(22).all()
    assert tasks.groupby("method").size().to_dict() == {"holdout": 5, "iid_matched": 40, "kfold": 25, "lolo": 40}
    again = make_splits(reactions, cfg)
    pd.testing.assert_frame_equal(tasks, again[2])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], again[1][key])
    lolo_test, kfold_test, iid_test, iid_signatures = [], [], [], set()
    for entry in entries:
        tr, te = arrays[entry["split_id"] + "_train"], arrays[entry["split_id"] + "_test"]
        validate_split(tr, te, len(reactions), reactions.ligand.to_numpy(), entry["method"], entry["reference_group"])
        if entry["method"] == "lolo":
            assert len(tr) == 2688 and len(te) == 384
            lolo_test.extend(te)
        if entry["method"] == "iid_matched":
            assert len(tr) == 2688 and len(te) == 384
            iid_test.extend(te)
            iid_signatures.add(tuple(sorted(te)))
        if entry["method"] == "kfold":
            kfold_test.extend(te)
        if entry["method"] == "holdout":
            assert len(tr) == 2457 and len(te) == 615
    # Matched IID mirrors the LOLO fold geometry: eight disjoint folds covering every row
    # exactly once, so it pools the same way and never reuses a reaction.
    assert len(iid_signatures) == 8
    np.testing.assert_array_equal(sorted(lolo_test), np.arange(3072))
    np.testing.assert_array_equal(sorted(iid_test), np.arange(3072))
    np.testing.assert_array_equal(sorted(kfold_test), np.arange(3072))
    # Random membership, not the by-ligand partition it is size-matched to.
    lolo_sets = {tuple(sorted(arrays[e["split_id"] + "_test"])) for e in entries if e["method"] == "lolo"}
    assert not (iid_signatures & lolo_sets)


def test_test_features_do_not_change_fitted_scaling():
    cfg = default_config()
    cfg["data"]["common_categorical"] = ["substrate_pair"]
    frame = pd.DataFrame({"vbur_pct_boltz": [1., 2., 3., 1000.],
                          "vbur_pct_min": [3., 5., 7., -1000.], "substrate_pair": ["a", "b", "a", "c"]})
    tr, te = np.array([0, 1, 2]), np.array([3])
    a, b, p, names, unknown = fit_transform_fold(frame, tr, te, "selected_2", cfg, None)
    changed = frame.copy()
    changed.loc[3, TWO] = [-99999, 99999]
    aa, bb, pp, _, _ = fit_transform_fold(changed, tr, te, "selected_2", cfg, None)
    np.testing.assert_array_equal(a, aa)
    assert unknown["substrate_pair"] == ["c"]
    assert not np.allclose(b, bb)
    np.testing.assert_allclose(p.named_transformers_["num"].named_steps["scale"].mean_, [2., 5.])


def test_loading_weighted_scores_keep_every_descriptor_and_its_weight():
    d = descriptor_fixture()
    cfg = default_config()
    cfg["features"]["pc_scores_form"] = "loading_weighted"
    cfg["data"]["common_categorical"] = ["substrate_pair"]
    ref = ReferencePCA.fit(d)
    ligands = ligand_table(d, ref, cfg)
    reactions = pd.DataFrame({"kraken_id": np.repeat(np.arange(12), 2),
                              "substrate_pair": np.tile(["p1", "p2"], 12)})
    frame = feature_frame(reactions, ligands, "pc_scores", cfg, ref)
    num, _ = model_columns("pc_scores", cfg, ref)
    assert num == list(ref.columns)  # every descriptor enters; nothing is summed away
    tr, te = np.arange(20), np.arange(20, 24)
    xtr, xte, pre, names, _ = fit_transform_fold(frame, tr, te, "pc_scores", cfg, ref)
    n_desc, n_pc = len(ref.columns), len(ref.components)
    assert xtr.shape[1] == n_desc * n_pc + 2 and names[0] == f"num__PC1_x_{ref.columns[0]}"
    z = StandardScaler().fit(frame.iloc[tr][num].to_numpy(float)).transform(
        frame.iloc[te][num].to_numpy(float))
    block = xte[:, :n_desc * n_pc].reshape(len(te), n_pc, n_desc)
    np.testing.assert_allclose(block, z[:, None, :] * ref.components[None, :, :], atol=1e-10)
    # Summing one component's columns returns that component's score on the same descriptors.
    np.testing.assert_allclose(block.sum(axis=2), z @ ref.components.T, atol=1e-10)
    # The loadings must survive: a scaler applied after them would leave unit variance.
    assert not np.allclose(xtr[:, :n_desc * n_pc].std(axis=0), 1.0)


def test_component_scores_form_is_unchanged():
    d = descriptor_fixture()
    cfg = default_config()
    cfg["features"]["pc_scores_form"] = "component_scores"
    cfg["data"]["common_categorical"] = ["substrate_pair"]
    ref = ReferencePCA.fit(d)
    num, _ = model_columns("pc_scores", cfg, ref)
    assert num == ["PC1", "PC2", "PC3", "PC4"]
    del cfg["features"]["pc_scores_form"]  # Bundles prepared before the option behave the same.
    assert model_columns("pc_scores", cfg, ref)[0] == num


def test_fixed_hyperparameter_posterior_matches_sklearn():
    rng = np.random.default_rng(18)
    x, test = rng.normal(size=(35, 3)), rng.normal(size=(9, 3))
    y = np.sin(x[:, 0]) + 0.1 * x[:, 1]
    cfg = default_config()["gp"]
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor((y - y.mean()) / y.std(), dtype=torch.float64)
    model, likelihood = build_model(xt, yt, cfg)
    model.eval(); likelihood.eval()
    summary = {"target_mean": float(y.mean()), "target_std": float(y.std())}
    mean, latent, predictive = predict_exact(model, likelihood, test, summary, cfg)
    other = GaussianProcessRegressor(kernel=RBF(1.0), optimizer=None, alpha=1e-6, normalize_y=True).fit(x, y)
    expected_mean, expected_std = other.predict(test, return_std=True)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(latent, expected_std, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(predictive**2 - latent**2, 1e-6 * y.std()**2, atol=1e-10)


def test_training_runs_and_keeps_fixed_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 3)); y = 40 + 10 * np.sin(x[:, 0])
    cfg = default_config()["gp"]
    cfg.update(max_steps=12, min_steps=5, patience=5, threads=1)
    model, likelihood, summary, history = fit_exact(x, y, cfg)
    mean, latent, predictive = predict_exact(model, likelihood, x[:4], summary, cfg)
    assert np.isfinite(mean).all() and (predictive >= latent).all()
    assert len(history) >= 5
    assert summary["best_negative_mll_per_point"] <= history.negative_mll_per_point.iloc[0]
    assert summary["noise_variance_standardized"] == pytest.approx(1e-6)
    assert summary["target_std"] == pytest.approx(y.std(ddof=0))


def test_metrics_use_predictive_uncertainty_and_undefined_r2():
    scores = regression_metrics([1, 1], [1, 1], [2, 2])
    assert scores["r2"] is None and scores["coverage_95"] == 1
    assert scores["predictive_nll"] == pytest.approx(np.log(2) + 0.5 * np.log(2 * np.pi))
    with pytest.raises(ValueError):
        regression_metrics([1], [1], [0])


_ROOT = Path(__file__).resolve().parents[1]
# Follow the configured prepared directory rather than a pinned version, so the
# integration check always runs against the bundle the current code produces.
REAL_BUNDLE = _ROOT / load_config(_ROOT / "configs/default.json")["paths"]["prepared"]


@pytest.mark.skipif(not REAL_BUNDLE.exists(), reason="Prepare original data first for integration checks")
def test_original_data_cleanup_and_all_representations():
    m, cfg, reactions, ligands, ref = load_bundle(REAL_BUNDLE)
    assert len(reactions) == 3072 and len(ligands) == 1223
    assert len(ref.columns) == 190 and len(ref.top_features) == 12
    _, tr, te = load_split(REAL_BUNDLE, "split_000")
    scores = 190 * 4 + 28 if pc_scores_form(cfg) == "loading_weighted" else 32
    dimensions = {"ligand_ohe": 35, "selected_5": 33, "selected_2": 30, "pc_top": 40, "pc_scores": scores}
    for model, dim in dimensions.items():
        f = feature_frame(reactions, ligands, model, cfg, ref)
        assert TARGET_NOT_IN(f)
        a, b, _, _, unknown = fit_transform_fold(f, tr, te, model, cfg, ref)
        assert a.shape == (2688, dim) and b.shape == (384, dim)
        if model == "ligand_ohe":
            assert unknown["ligand"] == ["AmPhos"]


def TARGET_NOT_IN(frame):
    return not ({"Product_Yield_PCT_Area_UV", "Product_Yield_Mass_Ion_Count", "Reaction_No", "kraken_id"} & set(frame))
