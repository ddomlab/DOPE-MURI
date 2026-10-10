"""Checks the Cernak Suzuki informer bundle, its feature section and its splits.

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
from gpc.features import _preprocessor, feature_frame, model_columns
from gpc.splits import make_folds, parse_method

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "inputs"

# Encoded input count per feature section. The one-hot block is
# 5 catalysts + 4 bases + 3 cosolvents + 12 cores + 10 monomers = 34;
# the descriptor sections replace the catalyst one-hot with numeric columns.
# Original note:
# 5 catalysts + 4 bases + 3 cosolvents + 12 cores + 10 monomers.
WIDTHS = {"catalyst_ohe": 34, "selected_2": 31, "selected_5": 34,
          "pc_top": 41, "pc_scores": 33}


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(BUNDLE)


def test_bundle_shape(bundle):
    reactions, ligands, reference, prepared = bundle
    assert len(reactions) == 1320
    assert reactions["catalyst"].nunique() == 5
    assert prepared["data"]["target"] == "conversion"
    assert prepared["data"]["group"] == "catalyst"


def test_the_five_catalysts_are_the_published_precatalysts(bundle):
    reactions = bundle[0]
    assert set(reactions["catalyst"]) == {
        "Aphos G3", "RuPhos G3", "Xphos G3", "tBu3P G2", "tBuXphos G3"}


def test_reagent_cardinalities_match_the_screen(bundle):
    """12 cores x 10 monomers x 12 (catalyst, base, cosolvent) condition sets."""
    reactions, ligands, reference, prepared = bundle
    assert reactions["core"].nunique() == 12
    assert reactions["monomer"].nunique() == 10
    condition_sets = reactions[["catalyst", "base", "cosolvent"]].drop_duplicates()
    assert len(condition_sets) == 12


def test_no_assay_plumbing_leaked_into_the_bundle(bundle):
    """Only the catalyst, the reagents, the row id and the target."""
    reactions, ligands, reference, prepared = bundle
    expected = {"row_id", prepared["data"]["group"], prepared["data"]["target"],
                "kraken_id",  # join key to ligand_features.csv for the descriptor sections
                *prepared["data"]["common_categorical"]}
    assert set(reactions.columns) == expected


@pytest.mark.parametrize("model", list(WIDTHS))
def test_encoded_width(bundle, model):
    reactions, ligands, reference, prepared = bundle
    frame = feature_frame(reactions, ligands, model, prepared, reference)
    pre = _preprocessor(frame, model, prepared, reference)
    assert pre.fit_transform(frame).shape[1] == WIDTHS[model]


def test_catalyst_ohe_has_no_numeric_block(bundle):
    """catalyst_ohe is pure one-hot; the descriptor sections are not, and are
    covered by test_encoded_width."""
    reactions, ligands, reference, prepared = bundle
    numeric, categorical = model_columns("catalyst_ohe", prepared, reference)
    assert numeric == []
    frame = feature_frame(reactions, ligands, "catalyst_ohe", prepared, reference)
    names = _preprocessor(frame, "catalyst_ohe", prepared, reference).fit(frame).get_feature_names_out()
    assert all(n.startswith("cat__") for n in names)


def test_lolo_holds_out_one_whole_catalyst(bundle):
    """Five catalysts, five folds, one precatalyst absent from training in each."""
    reactions = bundle[0]
    folds, references = make_folds(reactions, "lolo", 42)
    groups = reactions["catalyst"].to_numpy()
    assert len(folds) == 5
    for (train, test), ref in zip(folds, references):
        assert set(groups[test]) == {ref}
        assert ref not in set(groups[train])


def test_lolo_folds_are_unbalanced_because_aphos_has_double_coverage(bundle):
    """Aphos G3 carries 4 of the 12 condition sets; the others carry 2 each."""
    reactions = bundle[0]
    folds, references = make_folds(reactions, "lolo", 42)
    sizes = {ref: len(test) for (_, test), ref in zip(folds, references)}
    assert sizes["Aphos G3"] == 441
    assert all(200 < n < 240 for k, n in sizes.items() if k != "Aphos G3")
    assert sum(sizes.values()) == 1320


def test_held_out_catalyst_encodes_to_zeros_not_a_crash(bundle):
    """LOLO drops the test catalyst's column from the training fold."""
    reactions, ligands, reference, prepared = bundle
    frame = feature_frame(reactions, ligands, "catalyst_ohe", prepared, reference)
    (train, test) = make_folds(reactions, "lolo", 42)[0][0]
    encoder = _preprocessor(frame, "catalyst_ohe", prepared, reference).fit(frame.iloc[train])
    names = encoder.get_feature_names_out()
    assert len(names) == WIDTHS["catalyst_ohe"] - 1
    encoded = encoder.transform(frame.iloc[test])
    catalyst = [i for i, n in enumerate(names) if n.startswith("cat__catalyst_")]
    assert encoded[:, catalyst].sum() == 0
    # The reagents are still fully encoded: one column per reagent field.
    other = [i for i, n in enumerate(names) if i not in set(catalyst)]
    assert (encoded[:, other].sum(axis=1) == len(prepared["data"]["common_categorical"])).all()


def test_stratified_balances_catalysts_and_plain_kfold_does_not(bundle):
    reactions = bundle[0]
    groups = reactions["catalyst"].to_numpy()
    spread = lambda folds: max(
        int(pd.Series(groups[te]).value_counts().max() - pd.Series(groups[te]).value_counts().min())
        for _, te in folds)
    stratified, _ = make_folds(reactions, "kfold_stratified_5", 42)
    plain, _ = make_folds(reactions, "kfold_5", 42)
    # Aphos G3 is twice as common, so even a stratified fold is not flat -- what
    # stratification holds is each catalyst's SHARE, not an equal count.
    shares = [pd.Series(groups[te]).value_counts(normalize=True) for _, te in stratified]
    overall = pd.Series(groups).value_counts(normalize=True)
    assert max(float((s - overall).abs().max()) for s in shares) < 0.01
    assert spread(plain) > spread(stratified)
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(stratified, plain))


def test_method_names_parse():
    assert parse_method("kfold_5") == (5, False)
    assert parse_method("kfold_stratified_5") == (5, True)
    with pytest.raises(ValueError):
        parse_method("holdout")
    with pytest.raises(ValueError):
        parse_method("lolo")                    # handled separately, never parsed


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
    out = run_task(BUNDLE, tmp_path, "catalyst_ohe", "kfold_5", cfg)
    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == 1320
    assert predictions["y_pred"].notna().all()
    assert json.loads((out / "meta.json").read_text())["n_folds"] == 5
