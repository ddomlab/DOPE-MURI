# gp_collab_cernak

The same GP and the same cross-validation as `gp_collab_hazel` and
`gp_collab_rxnpredict`, run on the **Cernak Suzuki informer** screen. Training
only: its own `inputs/`, its own `runs/`, its own Slurm run tag (`cernak_v1`),
and nothing that the cluster does not need. Review happens locally from the
collected tables.

Production is **3 jobs** -- 1 feature section x 3 evaluation methods, 15 GP fits.

## The dataset

[cernaklab/medchem-reaction-miniaturization](https://github.com/cernaklab/medchem-reaction-miniaturization),
the `suzuki-ez.xlsx` workbook published with *Nature Synthesis* 2023
([10.1038/s44160-023-00351-1](https://doi.org/10.1038/s44160-023-00351-1)),
`suzuki_informer` sheet: a Suzuki-Miyaura informer library screened across five
palladium precatalysts.

```
12 cores x 10 monomers x 12 condition sets = 1440 wells
minus 120 wells with no Rel. %Conv.        = 1320 rows
```

A condition set is a (catalyst, base, cosolvent) triple. The 12 are **not** a
factorial: each precatalyst is paired only with the bases and cosolvents it
actually works with, so `Aphos G3` carries four condition sets (441 rows) and
the other four carry two each (215-226 rows).

Target is `conversion` -- the sheet's `Rel. %Conv.`, 0-100, mean 40.2.
`inputs/audit.json` is the provenance record: the source URL, the workbook's
sha256, the dropped-row accounting and every workbook column dropped with its
reason.

## Feature sections

| Section | Raw columns | Numeric | One-hot | Encoded |
| --- | ---: | ---: | ---: | ---: |
| `catalyst_ohe` | 5 | 0 | 34 | 34 |

5 catalysts + 4 bases + 3 cosolvents + 12 cores + 10 monomers = 34.

One section, and no numeric block: this screen ships no computed chemistry, so
the catalyst and every reagent is a discrete choice and all of it is one-hot.
It is the direct analogue of the sibling folders' `ligand_ohe`. There is no
`pc_top` / `pc_scores` here and no PCA reference -- there are no descriptors to
reduce. Adding a chemistry-derived section means computing descriptors the
published dataset does not carry: new work, not a config change.

Under LOLO the training fold has only 33 encoded columns, because the held-out
catalyst's one-hot column does not exist there; `handle_unknown="ignore"`
encodes it as all-zeros at predict time.

## Evaluation

| Method | Folds | Test fold | Stratified |
| --- | ---: | --- | --- |
| `lolo` | 5 | one whole precatalyst, 215-441 rows | n/a |
| `kfold_stratified_5` | 5 | 264 rows, all 5 catalysts | yes |
| `kfold_5` | 5 | 264 rows, all 5 catalysts | **no** |

Seed 42, stratification on catalyst identity, no 80/20 holdout. Every method is
a disjoint partition of all 1320 rows, so the three are directly comparable on
pooled metrics.

`lolo` holds out one of the five precatalysts at a time -- the question this
folder exists to ask. With five catalysts that is already a 5-fold partition, so
**`kfold_stratified_5` is its matched in-distribution control**: same splitter
family, same fold count, differing only in whether the held-out catalyst was seen
in training. A separate `iid_stratified_5` would be the identical splitter and is
deliberately not offered. `kfold_5` is the same thing unstratified, and isolates
what stratification alone is worth.

Two asymmetries to keep in mind:

- **LOLO folds are unequal.** Holding out `Aphos G3` removes 441 of 1320 rows
  against ~215-226 for the others. Read `pooled_r2` in `collected/summary.csv`,
  not the fold mean.
- **48 design cells were run twice.** They are told apart in the workbook only by
  `Condition Order`, which is run order rather than chemistry and is dropped, so
  94 rows share a feature row with a replicate twin. They are kept -- replicate
  spread is real information about the assay -- but a *random* fold can put one
  twin in train and the other in test. `lolo` cannot: a whole catalyst moves
  together.

## Running it on Hazel

Reuses the sibling folders' conda environment -- no second install. Check
`conda_env`, `account` and `gpu_request` in `hazel/slurm/*.sh` against your
allocation first.

```bash
bash hazel/setup_environment.sh              # once, only if the env does not exist
bash hazel/slurm/GP_preflight_GPU_arg.sh     # GPU + float64 Cholesky, 3-task registry
# set pilot=1 in GP_GPU_arg.sh, submit, size the walltime, then set it back
bash hazel/slurm/GP_GPU_arg.sh               # 3 jobs, collection chained behind them
```

This is a small run -- at most an 1105x1105 Cholesky per fit -- so the walltimes
in those scripts are deliberately loose and worth tightening after one pilot.
`GP_CPU_arg.sh` is the no-GPU fallback and is realistic at this size: a 2-epoch
LOLO task fits in about 2 seconds per fold on one CPU.

Each task writes `runs/cernak_v1/<model>__<method>/` with `predictions.csv`,
`scores.json` (fold metrics plus a per-encoded-column ARD lengthscale,
`fp_all[i]`) and `meta.json`. The chained collect job writes
`runs/cernak_v1/collected/{summary,predictions,lolo_by_catalyst}.csv`.

## Reviewing

Pull `runs/cernak_v1/` back and read the collected tables locally. To re-collect
after retrying a task:

```bash
python -m gpc --runs runs/cernak_v1 collect
```

## Rebuilding `inputs/`

You do not. The bundle is shipped prepared: `inputs/reactions.csv` and
`inputs/config.json` are the only files the training path reads, and there is no
prep script or raw workbook in this folder. To rebuild from scratch, fetch the
workbook recorded in `inputs/audit.json` (`source.url`, sha256
`bc61800940c839ed037979d05610d25c1b6a8c0a7788c8bcb8fffe8a7bf99bb8`), take the
`suzuki_informer` sheet, drop the rows and columns that `audit.json` lists, and
rename `Rel. %Conv.` to `conversion`.

## What this folder deliberately does not carry

Everything needed to train and collect on the cluster, and nothing else:

- **No review notebook, no `gpc/results.py`, no `gpc/export_hazel.py`.** Those
  export into `hazel_gp`'s ligand-keyed review schema, which does not apply here.
  `gpc collect` covers review.
- **No PCA artefacts and no ligand/descriptor tables.** No per-reagent chemistry
  exists for this screen.
- **No `model_features.csv` / `model_table.csv` / `reagent_smiles.csv` /
  `manifest.json`.** Nothing in the code reads them; the encoded feature count
  lands in `meta.json` and the reagent SMILES were never features.
- **No prep script and no raw workbook.** The cluster never rebuilds `inputs/`.
