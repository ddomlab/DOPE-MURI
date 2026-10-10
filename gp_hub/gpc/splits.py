"""The evaluation methods, as explicit (train, test) index pairs.

Every method is a disjoint partition of all rows, so each yields one pooled
out-of-fold prediction per reaction and they are directly comparable.

  lolo                  -- hold out one group at a time (LeaveOneGroupOut)
  iid_stratified_<n>    -- n folds, StratifiedKFold on the group; the matched
                           in-distribution control, so pick n to match LOLO's
                           fold count
  kfold_stratified_<n>  -- n folds, StratifiedKFold on the group
  kfold_<n>             -- n folds, plain KFold, no stratification

The fold count and whether to stratify are read from the method name, so
`kfold_5` and `kfold_stratified_5` can run side by side in one sweep and be
compared directly. Stratification is on group identity and is not applicable to
LOLO, whose whole point is that the test group is absent from training.

"Group" is the varied component -- the ligand in Perera and Ahneman, the
catalyst in Gesmundo. The bundle's config.json names it; nothing here assumes
which it is.

Two sizing notes carried over from the original projects, because both change
how the numbers should be read:

* When a screen has exactly n groups, LOLO is already an n-fold partition, so
  `kfold_stratified_n` IS its matched in-distribution control -- same splitter
  family, same fold count, differing only in whether the held-out group was seen
  during training. Adding `iid_stratified_n` on top would be the identical
  splitter.
* LOLO folds are NOT equal sized whenever the design is unbalanced (Gesmundo's
  Aphos G3 carries four of twelve condition sets, so its fold removes 441 of
  1320 rows against ~215-226 for the others). Compare pooled metrics across
  methods, not fold means.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold

from .config import GROUP

METHOD_PATTERN = re.compile(r"(iid_stratified|kfold_stratified|kfold)_(\d+)\Z")


def parse_method(method: str) -> tuple[int, bool]:
    """(n_splits, stratified) for any non-LOLO method name."""
    match = METHOD_PATTERN.fullmatch(method)
    if not match:
        raise ValueError(
            f"Unknown method {method!r}; expected 'lolo' or one of "
            f"iid_stratified_<n> / kfold_stratified_<n> / kfold_<n>")
    n_splits = int(match.group(2))
    if n_splits < 2:
        raise ValueError(f"{method!r} needs at least two folds")
    return n_splits, match.group(1).endswith("stratified")


class PrecomputedSplits:
    """Minimal splitter so GP_collab's `cross_validate` can consume any method."""

    def __init__(self, folds):
        self.folds = [(np.asarray(tr), np.asarray(te)) for tr, te in folds]

    def split(self, X=None, y=None, groups=None):
        return iter(self.folds)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


def _validate(folds, n):
    tested = np.concatenate([te for _, te in folds])
    if not np.array_equal(np.sort(tested), np.arange(n)):
        raise ValueError("Folds are not a disjoint partition of every row")
    for train, test in folds:
        if np.intersect1d(train, test).size or len(train) + len(test) != n:
            raise ValueError("A fold leaks rows between train and test")


def make_folds(reactions: pd.DataFrame, method: str, seed: int, stratify: bool = True,
               group: str = GROUP):
    """Return [(train_idx, test_idx), ...] plus the held-out group per fold."""
    n = len(reactions)
    indices = np.arange(n, dtype=np.int64)
    groups = reactions[group].to_numpy()

    if method == "lolo":
        folds = list(LeaveOneGroupOut().split(indices, groups=groups))
        references = [str(groups[te[0]]) for _, te in folds]
    else:
        n_splits, stratified = parse_method(method)
        # stratify=False forces plain KFold even for a *_stratified_* method,
        # which is how the run-wide config switch turns stratification off.
        splitter = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    if (stratified and stratify)
                    else KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        folds = list(splitter.split(indices, groups))
        # An in-distribution fold must see every group on both sides.
        for train, test in folds:
            if set(groups[test]) - set(groups[train]):
                raise ValueError(f"{method} fold has a {group} absent from training")
        references = ["" for _ in folds]

    _validate(folds, n)
    return folds, references


def fold_table(reactions: pd.DataFrame, method: str, seed: int, stratify: bool = True,
               group: str = GROUP):
    folds, references = make_folds(reactions, method, seed, stratify, group)
    return pd.DataFrame([
        {"method": method, "fold": i, "reference_group": ref,
         "n_train": len(tr), "n_test": len(te),
         "n_test_groups": int(pd.Series(reactions[group].to_numpy()[te]).nunique())}
        for i, ((tr, te), ref) in enumerate(zip(folds, references))
    ])
