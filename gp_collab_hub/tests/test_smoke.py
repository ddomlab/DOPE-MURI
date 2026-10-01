"""Checks that run without torch, on whichever bundles are present.

These are the assertions worth having in a hub that serves several datasets:
that every bundle still loads, that the folds are honest partitions, that the
feature sections encode to the widths they are supposed to, and that the parts
which used to be hardcoded to one dataset really are not any more.

    python -m pytest -q tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpc import figures                                         # noqa: E402
from gpc.config import (FIVE, KRAKEN_ALL, REACTION_FAMILIES, REACTION_SECTIONS,  # noqa: E402
                        SELECTED_SETS, TWO, VMIN, bundle_group, bundle_target,
                        check_reaction_sections, check_run_against_bundle,
                        condition_numeric, load_config, reaction_blocks,
                        reaction_columns)
from gpc.data import group_mapping, load_bundle                 # noqa: E402
from gpc.features import (apply_feature_overrides, feature_alias, feature_frame,  # noqa: E402
                          _preprocessor, _section_columns, model_columns)
from gpc.splits import make_folds, parse_method                 # noqa: E402

# In the hub a bundle sits at datasets/<name>/inputs; inside an HPC zip there is
# exactly one, at inputs/. Looking for both means `pytest -q tests` is a usable
# check on the cluster as well, which is where it matters most.
DATASETS = sorted(d for d in (ROOT / "datasets").glob("*/inputs")
                  if (d / "config.json").exists())
if not DATASETS and (ROOT / "inputs" / "config.json").exists():
    DATASETS = [ROOT / "inputs"]
SECTIONS = ["group_ohe", "selected_2", "vbur_min_vmin", "vbur_boltz_vmin",
            "selected_5", "pc_top", "pc_scores"]


def _ids(paths):
    return [p.parent.name for p in paths]


@pytest.fixture(scope="module", params=DATASETS, ids=_ids(DATASETS))
def bundle(request):
    return request.param


def test_at_least_one_dataset():
    assert DATASETS, "no prepared bundles under datasets/*/inputs"


def test_bundle_loads(bundle):
    reactions, ligands, reference, prepared = load_bundle(bundle)
    assert len(reactions) == prepared["data"]["expected_rows"]
    assert reactions[bundle_target(prepared)].notna().all()
    assert set(reactions.kraken_id) <= set(ligands.kraken_id)
    assert reference.components.shape[0] == 4


def test_reference_pca_is_shared(bundle):
    """PC1..PC4 must mean the same axes everywhere, or pc_top/pc_scores stop
    being comparable across datasets -- which is the whole point of them."""
    _, _, reference, _ = load_bundle(bundle)
    _, _, first, _ = load_bundle(DATASETS[0])
    assert reference.columns == first.columns
    assert np.allclose(reference.components, first.components)


def test_sections_encode(bundle):
    reactions, ligands, reference, prepared = load_bundle(bundle)
    for model in SECTIONS:
        frame = feature_frame(reactions, ligands, model, prepared, reference)
        encoded = _preprocessor(frame, model, prepared, reference).fit_transform(frame)
        assert encoded.shape[0] == len(reactions)
        assert encoded.shape[1] > 0
        assert np.isfinite(np.asarray(encoded, dtype=float)).all()


def test_group_ohe_aliases_agree(bundle):
    """ligand_ohe, catalyst_ohe and group_ohe must all mean one thing, or a
    Gesmundo run and a Perera run stop being the same experiment."""
    _, _, reference, prepared = load_bundle(bundle)
    widths = {name: model_columns(name, prepared, reference)
              for name in ("group_ohe", "ligand_ohe", "catalyst_ohe")}
    assert len(set(map(str, widths.values()))) == 1
    assert all(feature_alias(n) == "group_ohe" for n in widths)


def test_group_ohe_adds_the_group_column(bundle):
    _, _, reference, prepared = load_bundle(bundle)
    group = bundle_group(prepared)
    _, categorical = model_columns("group_ohe", prepared, reference)
    assert categorical[0] == group
    _, plain = model_columns("selected_2", prepared, reference)
    assert group not in plain


def test_selected_sections_use_the_shared_descriptors(bundle):
    _, _, reference, prepared = load_bundle(bundle)
    # The section's own descriptors -- continuous conditions are appended after
    # them by model_columns and checked in test_continuous_conditions_*.
    for section, columns in SELECTED_SETS.items():
        assert _section_columns(section, prepared, reference)[0] == columns
    assert _section_columns("selected_2", prepared, reference)[0] == TWO
    assert _section_columns("selected_5", prepared, reference)[0] == FIVE


def test_vmin_pairs_are_one_steric_plus_one_electronic(bundle):
    """Each vbur_*_vmin section is a buried volume paired with Vmin -- two
    descriptors like selected_2, but not two sterics."""
    _, ligands, reference, prepared = load_bundle(bundle)
    for section in ("vbur_min_vmin", "vbur_boltz_vmin"):
        columns = _section_columns(section, prepared, reference)[0]
        assert len(columns) == 2
        assert VMIN in columns
        assert sum(c.startswith("vbur_pct_") for c in columns) == 1
        values = ligands[columns].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{section} has a non-finite descriptor"


def test_feature_columns_override(bundle):
    _, _, reference, prepared = load_bundle(bundle)
    patched = apply_feature_overrides(prepared, {
        "run": {"feature_columns": {"selected_2": ["vbur_pct_delta"]}}})
    assert _section_columns("selected_2", patched, reference)[0] == ["vbur_pct_delta"]
    # The bundle config itself must not be mutated, or one task's override would
    # leak into the next one in the same process.
    assert _section_columns("selected_2", prepared, reference)[0] == TWO


def test_folds_partition_every_row(bundle):
    reactions, _, _, prepared = load_bundle(bundle)
    group = bundle_group(prepared)
    n_groups = reactions[group].nunique()
    for method in ("lolo", f"iid_stratified_{n_groups}", "kfold_stratified_5", "kfold_5"):
        folds, references = make_folds(reactions, method, 42, True, group)
        covered = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(covered), np.arange(len(reactions)))
        for train, test in folds:
            assert not np.intersect1d(train, test).size
        if method == "lolo":
            assert len(folds) == n_groups
            # The point of LOLO: the held-out group is absent from training.
            for (train, test), reference in zip(folds, references):
                assert reference not in set(reactions[group].to_numpy()[train])
        else:
            # The point of the controls: it is not.
            for train, test in folds:
                assert not set(reactions[group].to_numpy()[test]) - set(
                    reactions[group].to_numpy()[train])


def test_method_names_parse():
    assert parse_method("iid_stratified_8") == (8, True)
    assert parse_method("kfold_stratified_5") == (5, True)
    assert parse_method("kfold_5") == (5, False)
    for bad in ("kfold", "kfold_1", "lolo", "nonsense_3"):
        with pytest.raises(ValueError):
            parse_method(bad)


def test_group_mapping_is_complete(bundle):
    """The figures label points from this table, so a gap here is a silently
    unlabelled ligand rather than an error."""
    reactions, _, _, prepared = load_bundle(bundle)
    mapping = group_mapping(bundle)
    assert set(mapping.name) == set(reactions[bundle_group(prepared)])
    assert set(mapping.kraken_id) == set(reactions.kraken_id)


def test_default_run_config_is_valid():
    cfg = load_config(ROOT / "configs" / "default.json")
    assert cfg["run"]["models"] and cfg["run"]["methods"]
    assert cfg["gp"]["kernel"] and "hpc" in cfg


def test_figure_style_covers_every_section():
    """A section with no display label would print its raw config key on a
    figure heading for a talk, which is the kind of thing nobody notices until
    the talk."""
    for section in [*SECTIONS, *REACTION_SECTIONS]:
        assert section in figures.FEATURESET_LABELS
    for method in ("lolo", "iid_matched", "kfold"):
        assert method in figures.METHOD_DISPLAY
        assert method in figures.STYLE["pooled"]["method_colors"]


# ---------------------------------------------------- reaction-level descriptors
# Only a bundle that declares "reaction_features" (ahneman_doyle_dft: Doyle's 120
# DFT descriptors) can run the doyle_* sections; every other bundle must refuse
# them up front rather than fail inside a cluster job.

def test_reaction_sections_follow_the_bundle(bundle):
    reactions, ligands, reference, prepared = load_bundle(bundle)
    rxn = reaction_columns(prepared)
    if not rxn:
        for model in REACTION_SECTIONS:
            with pytest.raises(ValueError):
                model_columns(model, prepared, reference)
        with pytest.raises(ValueError):
            check_reaction_sections(list(REACTION_SECTIONS), prepared)
        return
    assert set(rxn) <= set(reactions.columns)
    assert np.isfinite(reactions[rxn].to_numpy(dtype=float)).all()
    blocks = reaction_blocks(prepared)
    group = bundle_group(prepared)
    assert blocks["group"] and blocks["conditions"]
    assert sorted(blocks["group"] + blocks["conditions"]) == sorted(rxn)
    for model, spec in REACTION_SECTIONS.items():
        numeric, categorical = model_columns(model, prepared, reference)
        doyle = blocks[spec["doyle"]]
        assert numeric[:len(doyle)] == doyle, f"{model} must lead with its Doyle block"
        if spec["ligand"] in (None, "group_ohe"):
            extra = []
        elif spec["ligand"] == KRAKEN_ALL:
            extra = list(reference.columns)
        else:
            extra = model_columns(spec["ligand"], prepared, reference)[0]
        assert numeric[len(doyle):] == extra
        expected_cat = (([group] if spec["ligand"] == "group_ohe" else [])
                        + (list(prepared["data"]["common_categorical"])
                           if spec["condition_ohe"] else []))
        assert categorical == expected_cat, model
        # The in-place family must carry none of Doyle's ligand descriptors, and
        # the added family all of them -- the comparison depends on exactly that.
        if model.startswith("doyle_cond_"):
            assert not set(numeric) & set(blocks["group"]), model
        if model.startswith("doyle_full"):
            assert set(blocks["group"]) <= set(numeric), model
        frame = feature_frame(reactions, ligands, model, prepared, reference)
        encoded = _preprocessor(frame, model, prepared, reference).fit_transform(frame)
        width = len(doyle) + len(extra) + sum(reactions[c].nunique() for c in expected_cat)
        assert encoded.shape == (len(reactions), width), model
        assert np.isfinite(np.asarray(encoded, dtype=float)).all()
    # Every in-place section has an added twin with the same ligand representation.
    for model in REACTION_FAMILIES["in_place"]:
        twin = model.replace("doyle_cond_", "doyle_full_")
        assert twin in REACTION_FAMILIES["added"], f"{model} has no added twin"
    with pytest.raises(ValueError):
        check_reaction_sections(["doyle_full"], prepared, {"grouping": "per_field"})


def test_reaction_family_labels(bundle):
    """ARD ranking with ligand_only=True must keep the ligand's own Doyle
    descriptors and drop the aryl halide/base/additive ones."""
    from gpc.results import _family_ligand_columns
    _, _, _, prepared = load_bundle(bundle)
    rxn = reaction_columns(prepared)
    if not rxn:
        pytest.skip("bundle has no reaction-level descriptors")
    group = bundle_group(prepared)
    names = [f"num__{c}" for c in rxn] + ["num__vbur_pct_boltz"]
    kept = set(_family_ligand_columns(names, prepared))
    assert "num__vbur_pct_boltz" in kept
    for c in rxn:
        assert (f"num__{c}" in kept) == c.startswith(f"{group}_")


# ------------------------------------------------------- continuous conditions
# Reizman_Summit's temperature / loading / residence time are continuous. A bundle declares
# them in data.common_numeric and every section must carry them as numeric
# inputs, never as one-hots and never as ligand chemistry.

def test_continuous_conditions_in_every_section(bundle):
    reactions, ligands, reference, prepared = load_bundle(bundle)
    numeric_conditions = condition_numeric(prepared)
    if not numeric_conditions:
        pytest.skip("bundle has no continuous conditions")
    assert np.isfinite(reactions[numeric_conditions].to_numpy(dtype=float)).all()
    sections = list(prepared["features"]["models"])
    for model in sections:
        numeric, categorical = model_columns(model, prepared, reference)
        assert numeric[-len(numeric_conditions):] == numeric_conditions, model
        assert not set(numeric_conditions) & set(categorical), model
        frame = feature_frame(reactions, ligands, model, prepared, reference)
        names = list(_preprocessor(frame, model, prepared, reference)
                     .fit(frame).get_feature_names_out())
        for c in numeric_conditions:
            assert f"num__{c}" in names, (model, c)
    # Split kernel groupings would model a temperature as ligand chemistry.
    with pytest.raises(ValueError):
        check_run_against_bundle(sections, prepared, {"grouping": "group_conditions"})
    check_run_against_bundle(sections, prepared, {"grouping": "all"})


def test_continuous_conditions_are_not_ligand_family(bundle):
    from gpc.results import _family_ligand_columns
    _, _, _, prepared = load_bundle(bundle)
    numeric_conditions = condition_numeric(prepared)
    if not numeric_conditions:
        pytest.skip("bundle has no continuous conditions")
    group = bundle_group(prepared)
    names = ([f"num__{c}" for c in numeric_conditions]
             + ["num__vbur_pct_boltz", f"cat__{group}_x", "cat__other_y"])
    kept = set(_family_ligand_columns(names, prepared))
    assert kept == {"num__vbur_pct_boltz", f"cat__{group}_x"}
