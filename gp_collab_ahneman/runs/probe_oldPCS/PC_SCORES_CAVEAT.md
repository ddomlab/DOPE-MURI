# pc_scores caveat -- not applicable to this run

Added 2026-09-18. This directory was renamed with the `_oldPCS` suffix only so
that every pre-2026-09-18 run folder carries the same marker. It contains a
single task, `rxnpredict_full__kfold_5`, and **no `pc_scores` task at all**, so
nothing here is affected by the pc_scores redefinition.

For context: in runs that do contain it, `pc_scores` used
`features.pc_scores_form = "loading_weighted"` -- 190 descriptors expanded
against 4 PCA loading vectors, 760 numeric columns (800 with the one-hots) --
instead of the PC1..PC4 projection the name implies. With `ard = true` the
loading constants are absorbed into the per-column lengthscales, so the PCA
weighting is lost, and with 4 ligands at most 3 independent ligand dimensions
exist. From 2026-09-18 `pc_scores` is `pc_scores_form = "component_scores"`:
PC1..PC4, 4 numeric columns, retrained under the tag `rxnpredict_pcs_v3`.

`rxnpredict_full` in this probe is UNAFFECTED and remains valid.
