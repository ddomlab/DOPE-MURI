# gp_collab_cernak

A sibling of `gp_collab_hazel` and `gp_collab_rxnpredict` that runs the same GP
and the same cross-validation on the **Cernak Suzuki informer** screen.
Self-contained: its own `inputs/`, its own `runs/`, its own Slurm run tag.
Nothing here touches the other two folders, and all three can sit in
`DOPE-MURI/` at once.

Production is **3 jobs** — 1 feature section × 3 evaluation methods, 15 GP fits.

## The dataset

[cernaklab/medchem-reaction-miniaturization](https://github.com/cernaklab/medchem-reaction-miniaturization),
the `suzuki-ez.xlsx` workbook published with *Nature Synthesis* 2023
([10.1038/s44160-023-00351-1](https://doi.org/10.1038/s44160-023-00351-1)).
This bundle uses the **`suzuki_informer`** sheet: a Suzuki–Miyaura informer
library screened across five palladium precatalysts.

```
12 cores x 10 monomers x 12 condition sets = 1440 wells
```

A condition set is a (catalyst, base, cosolvent) triple. There are 12, and they
are **not** a factorial over the three — each precatalyst is paired only with the
bases and cosolvents it actually works with:

| Catalyst | Condition sets | Bases | Cosolvent | Wells (valid) |
| --- | ---: | --- | --- | ---: |
| `Aphos G3` | 4 | BTMG, BTTP, P2-Et | none / t-AmOH | 441 |
| `tBu3P G2` | 2 | BTMG, BTTP | t-AmOH | 226 |
| `RuPhos G3` | 2 | BTMG, K3PO4 | H2O | 221 |
| `Xphos G3` | 2 | BTMG, K3PO4 | H2O | 217 |
| `tBuXphos G3` | 2 | BTMG, P2-Et | none | 215 |

`Rel. %Conv.` is missing in **120** wells, which are dropped — 89 of them on
`Core_02` alone. That leaves **1320** rows. `inputs/audit.json` records the
count, the per-core and per-catalyst breakdown, and every workbook column
dropped with the reason.

Target is `conversion`, the sheet's `Rel. %Conv.` (0–100, mean 40.2).

## Feature sections

| Section | Numeric | One-hot | Encoded |
| --- | ---: | ---: | ---: |
| `catalyst_ohe` | 0 | 34 | 34 |

5 catalysts + 4 bases + 3 cosolvents + 12 cores + 10 monomers = 34.

One section, no numeric block: this screen ships no computed chemistry, so the
catalyst and every reagent is a discrete choice and every one is one-hot
encoded. It is the direct analogue of the other folders' `ligand_ohe`. Adding a
chemistry-derived section means computing descriptors the published dataset does
not carry — new work, not a config change.

## Evaluation

| Method | Folds | Test fold | Stratified |
| --- | ---: | --- | --- |
| `lolo` | 5 | one whole precatalyst, 215–441 rows | n/a |
| `kfold_stratified_5` | 5 | 264 rows, all 5 catalysts | yes |
| `kfold_5` | 5 | 264 rows, all 5 catalysts | **no** |

Seed 42, stratification on catalyst identity, no 80/20 holdout.

`lolo` holds out one of the five precatalysts at a time — the question this
folder exists to ask. With five catalysts that is already a 5-fold partition, so
**`kfold_stratified_5` is its matched in-distribution control**: same splitter
family, same fold count, differing only in whether the held-out catalyst was seen
in training. A separate `iid_stratified_5` would be the identical splitter and is
deliberately not offered.

Two asymmetries to keep in mind:

- **LOLO folds are unequal.** Aphos G3 carries four of the twelve condition sets,
  so holding it out removes 441 of 1320 rows against ~215–226 for the others.
  Read `pooled_r2` in `collected/summary.csv`, not the fold mean.
- **48 design cells were run twice** and are told apart in the sheet only by
  `Condition Order`, which is run order rather than chemistry and is dropped. So
  94 rows share a feature row with a replicate twin. They are kept — replicate
  spread is real information about the assay — but a *random* fold can put one
  twin in train and the other in test. `lolo` cannot: a whole catalyst moves
  together.

## What the pilot says

A 60-epoch CPU pilot over all three methods:

| Method | pooled r² | pooled RMSE | fold-mean r² |
| --- | ---: | ---: | ---: |
| `kfold_5` | 0.73 | 21.2 | 0.73 ± 0.03 |
| `kfold_stratified_5` | 0.73 | 21.3 | 0.73 ± 0.02 |
| `lolo` | **0.58** | **26.5** | 0.53 ± 0.23 |

Unlike a held-out *substrate*, a held-out *catalyst* is still largely
predictable: dropping one precatalyst costs about 0.15 r², not all of it. The
reason is that 29 of the 34 one-hot columns describe the core, monomer, base and
cosolvent, and those carry most of the signal — the model can still tell that a
given core/monomer pair is reactive, it just loses which precatalyst is driving
it. Catalyst identity is a real effect here, not the dominant one.

Per catalyst (`collected/lolo_by_catalyst.csv`):

| Held out | n | pooled r² | pooled RMSE |
| --- | ---: | ---: | ---: |
| `tBu3P G2` | 226 | 0.72 | 22.7 |
| `RuPhos G3` | 221 | 0.68 | 21.9 |
| `Aphos G3` | 441 | 0.61 | 26.2 |
| `Xphos G3` | 217 | 0.56 | 26.8 |
| `tBuXphos G3` | 215 | **0.08** | **33.7** |

`tBuXphos G3` is the one that does not transfer, and it is also the outlier in
the raw data — mean conversion 26.4% against 41–47% for the other four. It is
the only precatalyst paired with (BTMG, P2-Et) and no cosolvent, so when it is
held out the model has never seen that condition set at all. That fold, not the
average, is the interesting result in this folder.

## Running it on Hazel

Same procedure as the sibling folders, and it reuses that conda environment —
no second install. Check `conda_env`, `account` and `gpu_request` in
`hazel/slurm/*.sh` against your allocation first.

```bash
bash hazel/slurm/GP_preflight_GPU_arg.sh     # GPU + float64 Cholesky, 3-task registry
# set pilot=1 in GP_GPU_arg.sh, submit, size the walltime, then set it back
bash hazel/slurm/GP_GPU_arg.sh               # 3 jobs, collection chained behind them
```

This is a small run — at most an 1105×1105 Cholesky per fit — so the walltimes in
those scripts are deliberately loose and worth tightening after one pilot.
`GP_CPU_arg.sh` is the no-GPU fallback and is realistic for a screen this size:
the whole pilot above ran in 23 seconds on CPU.

## Reviewing

```bash
python -m gpc --runs runs/cernak_v1 collect
```

writes `collected/summary.csv`, `predictions.csv` and `lolo_by_catalyst.csv`.

## Rebuilding the bundle

```bash
python prep_cernak/cernakprep.py
```

reads `prep_cernak/raw_cernak/suzuki-ez.xlsx` and rewrites `inputs/`. The raw
workbook is committed here so the build is reproducible offline;
`inputs/audit.json` records its sha256, the upstream URL, the sheet used, the
design check, the dropped-row accounting and every dropped column with a reason.

## What this folder deliberately does not carry

The transfer zip is what the cluster needs to train and nothing else. Absent by
design:

- **No PCA.** No `pca_reference.*`, no `pc_top` or `pc_scores` — there are no
  descriptors to reduce.
- **No `ligand_features.csv`.** No per-reagent chemistry exists for this screen.
- **No `export_hazel.py` / `results.py`.** Those export into `hazel_gp`'s review
  schema, which is keyed on ligands and a Perera-shaped bundle. `gpc collect`
  covers review here.
- **The assay plumbing is dropped at prep time** — plate position, retention
  times, peak areas, IS ratios, product mass and formula, plus `Solvent` and
  `Chemistry` (constant across all 1440 wells) and the `Condition Order` /
  `Monomer Order` run indices. Core and monomer SMILES live in
  `reagent_smiles.csv` so they cannot be mistaken for features.
- **`prep_cernak/` is not in the zip.** `inputs/` is prebuilt; the cluster never
  rebuilds it.
