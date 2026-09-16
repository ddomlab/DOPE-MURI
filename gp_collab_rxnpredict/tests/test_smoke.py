"""Checks the rxnpredict bundle, its feature sets and its splits are intact.

Run with `python -m pytest -q tests`. The training test needs torch and is
skipped when it is absent, so the data checks still run on a login node.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpc.config import load_config
from gpc.data import load_bundle
from gpc.features import _preprocessor, feature_frame
from gpc.splits import make_folds, parse_method

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "inputs"

# Encoded input counts recorded in inputs/model_table.csv.
WIDTHS = {"ligand_ohe": 44, "selected_2": 42, "selected_5": 45, "pc_top": 52,
          "pc_scores": 800, "rxnpredict_full": 120, "rxnpredict_full_ohe": 160}


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(BUNDLE)


def test_bundle_shape(bundle):
    reactions, _, _, prepared, rxn = bundle
    assert len(reactions) == 3955
    assert reactions["ligand"].nunique() == 4
    assert prepared["data"]["target"] == "yield"
    assert rxn is not None and rxn.shape[1] - 1 == 120
    assert not reactions.duplicated(["ligand", "base", "aryl_halide", "additive"]).any()


@pytest.mark.parametrize("model", list(WIDTHS))
def test_encoded_width(bundle, model):
    reactions, ligands, reference, prepared, rxn = bundle
    frame = feature_frame(reactions, ligands, model, prepared, reference, rxn)
    columns = [c for c in rxn.columns if c != "row_id"]
    pre = _preprocessor(frame, model, prepared, reference, columns)
    assert pre.fit_transform(frame).shape[1] == WIDTHS[model]


def test_rxnpredict_full_has_no_onehot_block(bundle):
    """The published baseline is descriptors only, as Doyle modelled it."""
    reactions, ligands, reference, prepared, rxn = bundle
    columns = [c for c in rxn.columns if c != "row_id"]
    frame = feature_frame(reactions, ligands, "rxnpredict_full", prepared, reference, rxn)
    pre = _preprocessor(frame, "rxnpredict_full", prepared, reference, columns)
    names = pre.fit(frame).get_feature_names_out()
    assert not any(n.startswith("cat__") for n in names)
    assert len(names) == 120


def test_lolo_holds_out_one_whole_ligand(bundle):
    reactions = bundle[0]
    folds, references = make_folds(reactions, "lolo", 42)
    groups = reactions["ligand"].to_numpy()
    assert len(folds) == 4                      # four ligands, not eight
    for (train, test), ref in zip(folds, references):
        assert set(groups[test]) == {ref}
        assert ref not in set(groups[train])


def test_stratified_balances_ligands_and_plain_kfold_does_not(bundle):
    reactions = bundle[0]
    groups = reactions["ligand"].to_numpy()
    spread = lambda folds: max(
        int(pd.Series(groups[te]).value_counts().max() - pd.Series(groups[te]).value_counts().min())
        for _, te in folds)
    stratified, _ = make_folds(reactions, "kfold_stratified_5", 42)
    plain, _ = make_folds(reactions, "kfold_5", 42)
    assert spread(stratified) <= 1
    assert spread(plain) > 5                    # unstratified really is unbalanced
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(stratified, plain))


def test_method_names_parse():
    assert parse_method("kfold_5") == (5, False)
    assert parse_method("kfold_stratified_5") == (5, True)
    assert parse_method("iid_stratified_4") == (4, True)
    with pytest.raises(ValueError):
        parse_method("holdout")


def test_every_row_tested_exactly_once(bundle):
    reactions = bundle[0]
    for method in load_config(ROOT / "configs/default.json")["run"]["methods"]:
        folds, _ = make_folds(reactions, method, 42)
        tested = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(tested), np.arange(len(reactions)))


def test_training_runs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    from gpc.train import run_task
    cfg = load_config(ROOT / "configs/default.json")
    cfg["gp"].update(n_epochs=2, use_cuda=False)
    out = run_task(BUNDLE, tmp_path, "selected_2", "kfold_5", cfg)
    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == 3955
    assert predictions["y_pred"].notna().all()
    assert json.loads((out / "meta.json").read_text())["n_folds"] == 5
