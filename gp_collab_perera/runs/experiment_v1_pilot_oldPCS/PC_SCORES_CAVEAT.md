# pc_scores in this run is not the intended representation

Added 2026-09-18, after the run finished. The run itself was not re-executed.

**What `pc_scores` meant here.** `inputs/config.json` carried
`features.pc_scores_form = "loading_weighted"`: the 190 standardized Kraken
descriptors were expanded elementwise against the 4 reference-PCA loading
vectors, giving 190 x 4 = 760 numeric columns (788 after the categorical
one-hots). That is not the PC1..PC4 projection the name implies.

**Why it is wrong.** This run trained with `ard = false`, so under the isotropic
RBF the expansion was mathematically equivalent to weighting each descriptor by
the L2 norm of its 4 loadings -- the weighting did survive here, but the model is
still not PC1..PC4, and it is not comparable with the replacement. Under
`ard = true` (now standard) those constants are absorbed and the weighting is
lost outright. With 8 ligands there are at most 7 independent ligand
dimensions, so the 760 columns are overwhelmingly redundant either way.

**Scope -- do not discard the whole run.** Only the `pc_scores` row is
affected. `ligand_ohe`, `selected_2`, `selected_5` and `pc_top` in this run are
UNAFFECTED and remain valid. (This folder is a 10-epoch smoke test, not a scientific result.)

**Replacement.** `pc_scores` is now `pc_scores_form = "component_scores"`: the
PC1..PC4 projection, 4 numeric columns. Retrained under the run tag `pcs_v3`.
