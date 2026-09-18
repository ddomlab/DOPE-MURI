# gp_collab_hazel

GP_collab's Gaussian process and cross-validation code, run on the DOPE-MURI
feature sections. Self-contained: the prepared input bundle is copied into
`inputs/`, so this folder can be zipped, uploaded to Hazel and submitted without
`hazel_gp` or `GP_collab` being present.

Production is **105 GP fits** — 5 feature sections × 21 folds — submitted as
**15 jobs**, one per (feature section, evaluation method) pair.

## What is reused, and from where

| Piece | Source | Changed? |
| --- | --- | --- |
| `gpc/features.py` | DOPE-MURI `hazel_gp/features.py` | No. PCA fitting/saving and the review tables were dropped; the transformations are byte-identical. |
| `inputs/` | DOPE-MURI `data_hazel/prepared/group1_v2` | No. Copied verbatim, hashes in `manifest.json`. |
| `gpc/vendor/kernel_mix.py` | GP_collab `GPytorch_kernel_mix.py` | The pyro/HMC regressor and the SSK kernel were dropped. New keyword arguments, all marked `# DOPE:`, default to GP_collab's original behaviour. |
| `gpc/vendor/scoring.py` | GP_collab `scoring_validation.py` | Trimmed to the per-fold worker, the non-grouped CV branch and `process_scores`. Kept code is unchanged. |
| `gpc/vendor/uq.py` | GP_collab `utils_uncertainty_calibration.py` | No. |
| `gpc/{config,data,splits,train,collect,cli}.py` | new | The glue: load the bundle, build the folds, assemble the pipeline, write tables. |

## Feature sections

Unchanged from DOPE-MURI. Encoded input counts (LOLO / in-distribution) are
asserted against that project's published table in `tests/test_smoke.py`:

| Section | Ligand inputs | Encoded inputs |
| --- | --- | ---: |
| `ligand_ohe` | ligand identity one-hot | 35 / 36 |
| `selected_5` | Boltzmann-average, minimum and range of buried volume; dipole; HOMO–LUMO gap | 33 / 33 |
| `selected_2` | Boltzmann-average and minimum buried volume | 30 / 30 |
| `pc_top` | top 3 original descriptors from each of PC1–PC4, deduplicated | 40 / 40 |
| `pc_scores` | the PC1–PC4 component scores: the frozen reference PCA's dimension reduction | 32 / 32 |

> **`pc_scores` changed on 2026-09-18.** It is now the PCA dimension reduction
> proper: the four component scores `PC1`-`PC4`, read straight from the frozen
> reference PCA. It previously used `pc_scores_form: "loading_weighted"`, which
> hands over all 190 reference descriptors and applies each PC loading to each of
> them separately -- 760 numeric columns and no reduction at all. Runs made with
> that form are tagged `INVALID_pc_scores*` under `runs/` and are not comparable
> to anything current; re-run those tasks. The form is set by
> `inputs/config.json` -> `features.pc_scores_form`, and both values still work.

All sections keep the same four categorical fields (28 one-hot columns).
Scaling, one-hot fitting and target standardisation happen inside each training
fold. A held-out ligand gets an all-zero ligand block (`handle_unknown='ignore'`).

## Evaluation methods

Stratification is on **ligand identity**, seed 42. The 80/20 holdout of the
original DOPE-MURI run is deliberately **not** reproduced.

| Method | Definition | Folds |
| --- | --- | ---: |
| `lolo` | hold out one ligand, train on the other seven | 8 |
| `iid_stratified_8` | `StratifiedKFold(8)` on ligand — LOLO's fold geometry (2688/384) with every ligand in train and test | 8 |
| `kfold_stratified_5` | `StratifiedKFold(5)` on ligand | 5 |

Stratification does not apply to LOLO, whose point is that the test ligand is
absent from training. Set `run.stratify` to `false` to fall back to plain
`KFold` for the other two. Each method is a disjoint partition of all 3,072
rows, so every method yields one pooled out-of-fold prediction per reaction.

## The kernel

`GPytorchMAPsklearnRegressor` with `kernel: "RBF"` and `grouping: "all"`,
`ard: false` — **one isotropic RBF over every encoded column**, which is the
kernel geometry `hazel_gp` ran.

For RBF kernels on disjoint column sets, GP_collab's `product` mixing is
identical to a single RBF with lengthscales shared within each group:

```
∏ᵢ exp(−‖xᵢ−x'ᵢ‖² / 2ℓᵢ²) = exp(−Σᵢ ‖xᵢ−x'ᵢ‖² / 2ℓᵢ²)
```

So `grouping` only sets **how many lengthscales the one kernel has**, and
`pc_scores` costs one kernel evaluation, not 4:

- `all` — 1 lengthscale (with `ard: false`); per-column with `ard: true`
- `ligand_conditions` — 2: the ligand block vs. the shared one-hot block
- `per_field` — one per categorical field plus the ligand block

This equivalence holds only for RBF and only for `product` mixing; `sum` is a
genuinely different additive GP.

## Differences from the original hazel_gp fits

Kernel geometry matches, but GP_collab's fit does not, so **scores here are not
numerically comparable to the earlier `hazel_gp` run** unless the config is
changed. Every difference is a key in `configs/default.json`:

| | `hazel_gp` ran | default here (GP_collab's) | key |
| --- | --- | --- | --- |
| Signal variance | fixed 1.0 | learned, LogNormal(0,1) prior | `outputscale: 1.0` to fix |
| Noise | fixed 1e-6 | learned, floor 1e-2 | `noise: 1e-6` to fix |
| Lengthscale priors | none | Gamma(5,5) | `prior: false` |
| Precision | float64 | float32 | `dtype: "float64"` |
| Cholesky jitter | 1e-8 / 1e-4 | 1e-1 / 1e-4 | `train_jitter`, `predict_jitter` |
| Optimiser | Adam, plateau stop, 5 restarts | Adam, 400 fixed epochs, 1 init | `n_epochs`, `restarts` |

To approximate the earlier run: `dtype: "float64"`, `noise: 1e-6`,
`outputscale: 1.0`, `prior: false`, `train_jitter: 1e-8`, `restarts: 5`.
GP_collab has no plateau stop, so `n_epochs` stays a fixed budget.

## Running it

Locally, for a check:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m gpc tasks     # the 15-job registry
python -m gpc splits    # fold sizes and ligand balance
python -m gpc --runs runs/local train --model selected_2 --method kfold_stratified_5 --epochs 10
python -m gpc --runs runs/local collect
```

On Hazel (Slurm is the production scheduler). Edit `conda_env` and
`output_root` in each script if your paths differ from `ddomlab/kagoble`:

```bash
bash hazel/setup_environment.sh          # once, on a login node
bash hazel/slurm/GP_preflight_GPU_arg.sh # GPU visibility + float64 Cholesky
# set pilot=1 in GP_GPU_arg.sh, submit, and size walltime from the result
bash hazel/slurm/GP_GPU_arg.sh           # 15 jobs, collection chained behind them
```

`GP_CPU_arg.sh` is the CPU fallback and writes to its own run folder.
`GP_collect_CPU_arg.sh` re-collects a run by hand. Job logs go to a dated
`HPC_history/hpc_YYYYMMDD` folder under `/share`, which each script creates.

## Output

Each task writes `runs/<tag>/<section>__<method>/`:

- `predictions.csv` — pooled out-of-fold `y_pred`/`y_std` per reaction, with its
  fold and held-out ligand
- `scores.json` — per-fold rmse/mae/r²/ece/RUSC/cdf_ama/cvpp_ama/nll/Cv/sharpness
  and fitted lengthscales, plus GP_collab's `*_avg`/`*_stdev` aggregates
- `meta.json` — config, device, torch/CUDA versions

`gpc collect` gathers these into `collected/summary.csv` (fold mean ± sd and
pooled metrics per section × method), `collected/predictions.csv`, and
`collected/lolo_by_ligand.csv`. Per-ligand R² is measured against that ligand's
own variance and pooled R² against the whole dataset's, so the two are not
interchangeable — the per-ligand figure is systematically harsher.
