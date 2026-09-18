"""Checks that the bundle, the feature sections and the splits are intact.

Run with `python -m pytest -q tests`. The training test needs torch and is
skipped when it is absent, so the data checks still run on a login node.
"""
from pathlib import Path

import numpy as np
import pytest

from gpc.config import load_config
from gpc.data import load_bundle
from gpc.features import _preprocessor, feature_frame
from gpc.splits import make_folds

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "inputs"

# Encoded input counts published in the DOPE-MURI hazel_gp README (LOLO / IID).
# pc_scores is the exception: it is now the PCA reduction (PC1..PC4 + the 28
# shared one-hot columns), not the 760-column loading expansion the old README
# tabulated as 788. Set features.pc_scores_form to loading_weighted to get 788 back.
WIDTHS = {"ligand_ohe": (35, 36), "selected_5": (33, 33), "selected_2": (30, 30),
          "pc_top": (40, 40), "pc_scores": (32, 32)}


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(BUNDLE)


def test_bundle_shape(bundle):
    reactions, _, _, prepared = bundle
    assert len(reactions) == 3072
    assert reactions["ligand"].nunique() == 8
    assert reactions["ligand"].value_counts().unique().tolist() == [384]
    # The literal "None" base must survive the CSV round trip as a category.
    assert "None" in set(reactions["Reagent_1_Short_Hand"])


@pytest.mark.parametrize("model", list(WIDTHS))
def test_encoded_width_matches_hazel_gp(bundle, model):
    reactions, ligands, reference, prepared = bundle
    frame = feature_frame(reactions, ligands, model, prepared, reference)
    pre = _preprocessor(frame, model, prepared, reference)
    lolo_train = make_folds(reactions, "lolo", 42)[0][0][0]
    iid_train = make_folds(reactions, "iid_stratified_8", 42)[0][0][0]
    assert pre.fit_transform(frame.iloc[lolo_train]).shape[1] == WIDTHS[model][0]
    assert pre.fit_transform(frame.iloc[iid_train]).shape[1] == WIDTHS[model][1]


def test_lolo_holds_out_one_whole_ligand(bundle):
    reactions, *_ = bundle
    folds, references = make_folds(reactions, "lolo", 42)
    groups = reactions["ligand"].to_numpy()
    assert len(folds) == 8
    for (train, test), ref in zip(folds, references):
        assert set(groups[test]) == {ref}
        assert ref not in set(groups[train])


@pytest.mark.parametrize("method,n_splits", [("iid_stratified_8", 8), ("kfold_stratified_5", 5)])
def test_stratified_folds_balance_ligands(bundle, method, n_splits):
    reactions, *_ = bundle
    folds, _ = make_folds(reactions, method, 42, stratify=True)
    groups = reactions["ligand"].to_numpy()
    assert len(folds) == n_splits
    for train, test in folds:
        assert set(groups[test]) == set(groups)      # in-distribution
        assert set(groups[test]) <= set(groups[train])
        counts = np.unique(groups[test], return_counts=True)[1]
        assert counts.max() - counts.min() <= 1      # stratified


def test_every_row_tested_exactly_once(bundle):
    reactions, *_ = bundle
    for method in ("lolo", "iid_stratified_8", "kfold_stratified_5"):
        folds, _ = make_folds(reactions, method, 42)
        tested = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(tested), np.arange(len(reactions)))


def test_config_rejects_unknown_names(tmp_path):
    import json
    cfg = json.loads((ROOT / "configs/default.json").read_text())
    cfg["run"]["methods"] = ["holdout"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Unknown evaluation methods"):
        load_config(path)


def test_kernel_is_one_isotropic_rbf(tmp_path):
    """grouping=all with ard=False must give a single shared lengthscale."""
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    from gpc.train import run_task

    cfg = load_config(ROOT / "configs/default.json")
    # Pin ard explicitly: this test states an invariant about ard=False, so it must not
    # silently change meaning when the shipped config flips ard on.
    cfg["gp"].update(n_epochs=2, use_cuda=False, ard=False)
    out = run_task(BUNDLE, tmp_path, "selected_2", "kfold_stratified_5", cfg)

    import json
    scores = json.loads((out / "scores.json").read_text())
    per_fold = scores[str(cfg["run"]["seed"])]["test_lengthscale"]
    assert all(list(fold) == ["fp_all"] for fold in per_fold)

    import pandas as pd
    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == 3072
    assert predictions["y_pred"].notna().all()
    assert predictions["y_std"].notna().all()


def test_ard_gives_one_lengthscale_per_feature(tmp_path):
    """ard=True must give a per-column lengthscale, keyed fp_all[i], and the
    per-model override must reach the fit."""
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    import json
    from gpc.train import run_task

    cfg = load_config(ROOT / "configs/default.json")
    cfg["gp"].update(n_epochs=2, use_cuda=False, ard=True)
    out = run_task(BUNDLE, tmp_path, "selected_2", "kfold_stratified_5", cfg)

    scores = json.loads((out / "scores.json").read_text())
    per_fold = scores[str(cfg["run"]["seed"])]["test_lengthscale"]
    for fold in per_fold:
        keys = list(fold)
        assert len(keys) > 1, "ard=True still produced a single lengthscale"
        assert all(k.startswith("fp_all[") for k in keys)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["config"]["gp"]["ard"] is True


def test_model_override_mechanism(tmp_path):
    """run.model_overrides still folds into the gp block for whichever model names it.

    pc_scores_long was removed once pc_scores became the 4-column PC1..PC4
    projection and no longer needed a longer budget, so this exercises the
    mechanism with a synthetic override rather than a shipped one.
    """
    import copy
    from gpc.train import effective_config

    cfg = load_config(ROOT / "configs/default.json")
    assert effective_config(cfg, "pc_scores")["gp"]["n_epochs"] == cfg["gp"]["n_epochs"]

    cfg2 = copy.deepcopy(cfg)
    cfg2["run"]["model_overrides"] = {"pc_scores": {"n_epochs": 999, "walltime": "01:00:00"}}
    eff = effective_config(cfg2, "pc_scores")["gp"]
    assert eff["n_epochs"] == 999
    assert eff["restarts"] == cfg["gp"]["restarts"]      # untouched keys fall through
    assert "walltime" not in eff                          # scheduling, never a GP parameter
    assert effective_config(cfg2, "selected_2")["gp"]["n_epochs"] == cfg["gp"]["n_epochs"]
