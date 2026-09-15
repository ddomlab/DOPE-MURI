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
WIDTHS = {"ligand_ohe": (35, 36), "selected_5": (33, 33), "selected_2": (30, 30),
          "pc_top": (40, 40), "pc_scores": (788, 788)}


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
    cfg["gp"].update(n_epochs=2, use_cuda=False)
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
