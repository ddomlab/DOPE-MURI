# gp_collab_rxnpredict

A sibling of `gp_collab_hazel` that runs the same GP and the same
cross-validation on the **Doyle rxnpredict** screen instead of the Perera one.
Self-contained: its own `inputs/`, its own `runs/`, its own log directory, its
own Slurm run tag. Nothing here touches the Perera folder or its results, and
both can sit in `DOPE-MURI/` at once.

Production is **28 jobs** -- 7 feature sections x 4 evaluation methods.

## The dataset

[doylelab/rxnpredict](https://github.com/doylelab/rxnpredict), the Ahneman et al.
(*Science* 2018) Buchwald-Hartwig C-N coupling screen. 3,955 reactions:
4 ligands x 3 bases x 15 aryl halides x 22 additives, minus 5 combinations that
were never run. Dropped upstream: no-aryl-halide controls (263, all 0% yield),
no-additive wells (192), and one additive with no computed descriptors (189).
No condition tuple repeats, so row-level splits do not leak.

`prep_rxnpredict/` holds the notebook and module that built `inputs/`, including
the check that rebuilds Doyle's own descriptor table from the component join and
compares it column for column.

## Feature sections

| Section | Ligand | rxn | Numeric | One-hot | Encoded |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ligand_ohe` | 0 | 0 | 0 | 44 | 44 |
| `selected_2` | 2 | 0 | 2 | 40 | 42 |
| `selected_5` | 5 | 0 | 5 | 40 | 45 |
| `pc_top` | 12 | 0 | 12 | 40 | 52 |
| `pc_scores` | 4 | 0 | 4 | 40 | 44 |
| `rxnpredict_full` | 0 | 120 | 120 | 0 | 120 |
| `rxnpredict_full_ohe` | 0 | 120 | 120 | 40 | 160 |

The shared one-hot block is 15 aryl halides + 3 bases + 22 additives = 40.
The five ligand-chemistry sets use the **same preprocessor and the same frozen
Kraken PCA** as the Perera runs -- the PCA files are copied, never refitted, so
`pc_top` and `pc_scores` mean the same thing in both folders.

`pc_scores` is the PCA reduction itself -- the four component scores PC1-PC4.
It previously held each of the 190 descriptors times its loading in each
component (760 numeric columns, no reduction), which `ard: true` renders
unidentifiable. `features.pc_scores_form` in `inputs/config.json` selects the
form: `component_scores` is the current default, `loading_weighted` reproduces
the old one. **Runs recorded before this change carry the old 760-column
`pc_scores`.** `inputs/model_table.csv` still records the old 190/760/800 row --
it is a frozen provenance artifact, not a live input.

`rxnpredict_full` is the published baseline: the 120 DFT descriptors and **no**
one-hot block, as Doyle modelled it. `rxnpredict_full_ohe` adds the shared block
so the only thing changing across the other six is the ligand representation.

## Evaluation

| Method | Folds | Geometry | Stratified |
| --- | ---: | --- | --- |
| `lolo` | 4 | 2965-2969 / 986-990 | n/a |
| `iid_stratified_4` | 4 | 2966-2967 / 988-989 | yes |
| `kfold_stratified_5` | 5 | 3164 / 791 | yes |
| `kfold_5` | 5 | 3164 / 791 | **no** |

Seed 42, stratification on ligand identity, no 80/20 holdout. With four ligands
LOLO holds out about a quarter of the data per fold, so expect weaker and
noisier LOLO numbers than the Perera runs for reasons unrelated to the
representations. `kfold_5` and `kfold_stratified_5` share geometry and seed but
differ in membership, so they isolate what stratification is worth.

## Running it on Hazel

Same procedure as `gp_collab_hazel`, and it reuses that folder's conda
environment -- no second install. Check `conda_env`, `account` and `gpu_request`
in `hazel/slurm/*.sh` against your allocation first.

```bash
bash hazel/slurm/GP_preflight_GPU_arg.sh     # GPU + float64 Cholesky, 28-task registry
# set pilot=1 in GP_GPU_arg.sh, submit, size the walltime, then set it back
bash hazel/slurm/GP_GPU_arg.sh               # 28 jobs, collection chained behind them
```

Logs go to `/share/.../working_space/gp_collab_rxnpredict/HPC_history/hpc_DATE`
and results to `runs/rxnpredict_v1`, both separate from the Perera run.

## Reviewing

```bash
python -m gpc --runs runs/rxnpredict_v1 collect
```

writes `collected/summary.csv`, `predictions.csv` and `lolo_by_ligand.csv`.
`review_gpc_results.ipynb` reads a run through `gpc/results.py` exactly as it
does for Perera; point its `RUNS` at `runs/rxnpredict_v1`.

## Before you trust the two rxnpredict baselines

A 5-epoch smoke test gave `rxnpredict_full` r2 = 0.007 where `selected_2` gave
0.798. That is most likely the epoch budget -- one isotropic lengthscale starting
at 1.0 is small for 120 standardised dimensions, and squared distances grow with
dimensionality -- but it has **not** been confirmed at the full 400-epoch budget.
Run one job first:

```bash
python -m gpc --runs runs/probe train --model rxnpredict_full --method kfold_5
```

and read `test_lengthscale` in its `scores.json`. Grown well above 1 with a
sensible r2 means it was only the budget. Still pinned near 1 with r2 about 0
means the high-dimensional sections want `ard: true` or a larger initial
lengthscale in `configs/default.json` -- a config change, not a code one.
