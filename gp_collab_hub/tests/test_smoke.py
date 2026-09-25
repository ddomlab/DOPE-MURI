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
from gpc.config import (FIVE, SELECTED_SETS, TWO, VMIN, bundle_group,  # noqa: E402
                        bundle_target, load_config)
from gpc.data import group_mapping, load_bundle                 # noqa: E402
from gpc.features import (apply_feature_overrides, feature_alias, feature_frame,  # noqa: E402
                          _preprocessor, model_columns)
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
    for section, columns in SELECTED_SETS.items():
        assert model_columns(section, prepared, reference)[0] == columns
    assert model_columns("selected_2", prepared, reference)[0] == TWO
    assert model_columns("selected_5", prepared, reference)[0] == FIVE


def test_vmin_pairs_are_one_steric_plus_one_electronic(bundle):
    """Each vbur_*_vmin section is a buried volume paired with Vmin -- two
    descriptors like selected_2, but not two sterics."""
    _, ligands, reference, prepared = load_bundle(bundle)
    for section in ("vbur_min_vmin", "vbur_boltz_vmin"):
        columns = model_columns(section, prepared, reference)[0]
        assert len(columns) == 2
        assert VMIN in columns
        assert sum(c.startswith("vbur_pct_") for c in columns) == 1
        values = ligands[columns].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{section} has a non-finite descriptor"


def test_feature_columns_override(bundle):
    _, _, reference, prepared = load_bundle(bundle)
    patched = apply_feature_overrides(prepared, {
        "run": {"feature_columns": {"selected_2": ["vbur_pct_delta"]}}})
    assert model_columns("selected_2", patched, reference)[0] == ["vbur_pct_delta"]
    # The bundle config itself must not be mutated, or one task's override would
    # leak into the next one in the same process.
    assert model_columns("selected_2", prepared, reference)[0] == TWO


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
    for section in SECTIONS:
        assert section in figures.FEATURESET_LABELS
    for method in ("lolo", "iid_matched", "kfold"):
        assert method in figures.METHOD_DISPLAY
        assert method in figures.STYLE["pooled"]["method_colors"]
