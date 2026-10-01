# pc_scores in this run is not the intended representation

Added 2026-09-18, after the run finished. The run itself was not re-executed.

**What `pc_scores` meant here.** `inputs/config.json` carried
`features.pc_scores_form = "loading_weighted"`: the 190 standardized Kraken
descriptors were expanded elementwise against the 4 reference-PCA loading
vectors, giving 190 x 4 = 760 numeric columns (800 after the categorical
one-hots). That is not the PC1..PC4 projection the name implies.

**Why it is wrong.** This run trained with `ard = true` (see any task's
`meta.json`). ARD fits one lengthscale per column, so the constant loading
factors are absorbed into those lengthscales and the PCA weighting they were
meant to carry is discarded. With 4 ligands there are at most 3
independent ligand dimensions, so the 760 columns are overwhelmingly
redundant and their lengthscales are unidentifiable.

**Scope -- do not discard the whole run.** Only the `pc_scores` and `pc_scores_long` rows are
affected. `ligand_ohe`, `selected_2`, `selected_5` and `pc_top` in this run are
UNAFFECTED and remain valid.

**Replacement.** `pc_scores` is now `pc_scores_form = "component_scores"`: the
PC1..PC4 projection, 4 numeric columns. Retrained under the run tag `rxnpredict_pcs_v3`.
