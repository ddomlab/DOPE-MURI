| Model | Ligand inputs | Total encoded inputs in LOLO / IID |
| --- | --- | --- |
| `ligand_ohe` | Ligand one-hot encoding | 35 / 36 |
| `selected_5` | Boltzmann-average, minimum and range of buried volume; dipole; HOMO-LUMO gap | 33 / 33 |
| `selected_2` | Boltzmann-average and minimum buried volume | 30 / 30 |
| `pc_top` | Top 3 original descriptors from each of PC1-PC4, deduplicated | 40 / 40 (12 ligand descriptors in this snapshot) |
| `pc_scores` | Each of the 190 reference descriptors times its loading in each of PC1-PC4, unsummed | 788 / 788 |

| Evaluation | Definition | Fits across five models |
| --- | --- | ---: |
| LOLO / OOD | Train on seven ligands, test all rows of the eighth; all eight ligands | 40 |
| Matched IID | The same eight-fold geometry with random membership: folds sized to each LOLO test set, every row tested exactly once, all ligands in every fold | 40 |
| Five-fold | `KFold(5, shuffle=True, random_state=42)` | 25 |
| 80/20 | `train_test_split(test_size=0.2, random_state=42)` | 5 |
| LOLO reporting | Individual ligands, mean across ligand-left-out models, pooled after the eight folds; reuse the same fits | 0 additional |

This package prepares the original Perera/Pfizer Suzuki data locally, fits five GPyTorch ExactGP representations on NC State Hazel, and returns predictions and uncertainty for local analysis. Production is **110 fits**, each with five optimizer restarts. No fitting is performed by opening either notebook. The supplied validation material documents checks, not a completed benchmark or a Hazel run.

All models retain the original four common categorical fields: `Reactant_1_Short_Hand`, `substrate_pair`, `Reagent_1_Short_Hand`, and `Solvent_1_Short_Hand`. They contribute 28 one-hot columns. Keeping reactant-1 alongside substrate pair intentionally preserves the original redundant encoding. Ligand identity is a predictor only for `ligand_ohe`; it is retained as metadata for all models.

## 1. Install locally on Windows / VS Code

Extract the ZIP into a new folder. Open that folder in VS Code. Use Python 3.12. From the project root, either create the supplied Conda environment:

```powershell
conda env create -f environment_local.yml
conda activate hazel-gp-local
```

Or use a standard virtual environment, without requiring PowerShell activation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name hazel-gp-local --display-name "Python (Hazel GP local)"
```

Choose that interpreter/kernel in VS Code. The local preparation and review workflow does not require a GPU or PyTorch. Optional local CPU training/tests:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m hazel_gp check-env --device cpu
```

When using the virtual environment without activation, replace `python` with `.\.venv\Scripts\python.exe`. Cluster and local training use the same pinned scientific packages. Notebook/UI dependencies have bounded versions. The worker records actual versions and rejects mixing software versions in one run directory.

## 2. Local download, cleanup, review, and export

Open **`01_prepare_local.ipynb`** and run its cells in order. It downloads from the same three URLs found in `data_workup_notebook(1).ipynb`:

- [Original reaction CSV](https://raw.githubusercontent.com/Paulilein/Perera2018/main/Perera2018_Data_Original.csv)
- [Kraken descriptor CSV](https://raw.githubusercontent.com/doyle-lab-ucla/kraken_utils/main/kraken_features_only.csv)
- [Kraken identifiers CSV](https://raw.githubusercontent.com/doyle-lab-ucla/kraken_utils/main/identifiers.csv)

The new folders are `data_hazel/raw`, `data_hazel/prepared/group1_v2`, `data_hazel/runs`, and `exports`. Your original `data/` is never modified. Downloads are cached with SHA-256 hashes. Existing cached files are reused. For a new upstream snapshot, choose a new raw directory in the config; `download --refresh` is an explicit alternative that replaces the cached raw files.

The notebook displays the cleanup audit, exclusions, ligand mapping/SMILES, model definitions, PCA variance/loadings, and the actual split counts before exporting. A feature grid names each encoded input for all five models, ligand features first and then the shared categorical block held constant across models; it shows the first `MAX_FEATURES_SHOWN` (25 by default) with a totals row, and the bundle records every name in `model_features.csv`. Its final **Export reviewed inputs** button writes the prepared folder and creates the upload ZIP. Change both output fields for a new version. Existing prepared folders and ZIP files are never overwritten. The original code remains in modules; the notebook holds settings and review controls.

The preparation result is also accessible from Python:

```python
from pathlib import Path
from hazel_gp.config import load_config
from hazel_gp.data import download_sources, prepare_data, export_prepared, create_upload_archive

root = Path.cwd()
cfg = load_config(root / "configs/default.json")
download_sources(cfg, root)
prepared = prepare_data(cfg, root)  # computes in memory; inspect before the next two lines
bundle = export_prepared(prepared, root / cfg["paths"]["prepared"])
create_upload_archive(root, bundle, root / "exports/hazel_gp_upload_v2.zip")
```

Equivalent noninteractive commands (the `prepare` command explicitly exports immediately):

```powershell
python -m hazel_gp download
python -m hazel_gp prepare
python -m hazel_gp pack-inputs --bundle data_hazel/prepared/group1_v2 --output exports/hazel_gp_upload_v2.zip
```

For the verified source snapshot, cleanup returns **5,760 raw -> 3,712 after original solvent/ligand exclusions -> 3,072 Group1 rows**, with eight ligands and 384 rows per ligand. The configuration checks these counts and stops if the data unexpectedly change. It keeps Group1 reactant codes 1a-1d and 2a-2c, drops the original V2 solvent labels, and excludes the no-ligand/bidentate conditions. These are scope rules, not yield-based filtering.

Literal `None` base conditions remain explicit categories. Exact duplicate imports are audited; conflicting reaction IDs, repeated experimental conditions, missing inputs, unrecognized mappings, and out-of-range targets cause an error. True replicates would require a new grouping protocol before applying these row-random splits. Only explicitly listed predictors reach the model; IDs, UV yield, and mass-ion response cannot leak into features.

The complete source package includes `reference_snapshot/` with the verified downloaded CSV bytes and hashes, and `examples/verified_inputs/` with a prepared reference bundle. They do not populate or overwrite the new working folders. The usual notebook downloads from the original links. To use the preserved snapshot offline, copy its contents into `data_hazel/raw` before running the notebook. To skip preparation for a first cluster pilot, use the verified example with `pack-inputs --bundle examples/verified_inputs`.

## 3. Ligand representations and PCA

The short model is **two buried-volume descriptors**, not electronic Vmin. Derived fields are `vbur_pct_boltz`, `vbur_pct_min`, `vbur_pct_delta`, and `homo_lumo_gap_eV`. Raw Kraken columns remain unmodified. All buried-volume percentages use `100 * V / ((4/3) * pi * 3.5**3)`; the positive gap is `(LUMO - HOMO) * 27.211386245981`. Standardizing each numerical feature makes these consistent unit/sign changes distance-equivalent to the original columns under the same RBF geometry.

PCA is fitted once to **all 1,223 ligands and 190 descriptors in the original linked Kraken file**. This interprets the accepted original-data population as the full linked Kraken reference, rather than the eight reaction ligands or a new expanded database. Missing reference values use reference means; descriptors are standardized and four PCs are fitted with deterministic full SVD. No reaction yields enter PCA. Its reference means/scales/components and source hashes are saved. Held-out ligand descriptors may occur in this external reference; this is reaction-outcome LOLO, not strict exclusion of held-out covariates from representation construction.

The recovered top descriptors match the saved notebook selections:

- PC1: `vbur_vtot_boltz`, `volume_boltz`, `vbur_near_vtot_vburminconf`.
- PC2: `pyr_P_max`, `qpole_amp_max`, `vbur_qvbur_min_min`.
- PC3: `vbur_near_vbur_delta`, `vbur_vbur_delta`, `vbur_qvbur_min_delta`.
- PC4: `nbo_bds_occ_avg_boltz`, `efgtens_zz_P_boltz`, `nmr_P_boltz`.

The first four explained-variance fractions in this source snapshot are approximately **29.00%, 13.01%, 11.18%, and 6.15%**. These need not equal the paper's 1,558-ligand reference. See [Gensch et al., JACS 2022](https://doi.org/10.1021/jacs.1c09718). `pc_top` uses the original selected descriptor values. `pc_scores` applies the loadings without reducing dimension: input `PC{i}_x_{descriptor}` is that descriptor's fold-standardized value times its loading in component `i`, so all 190 descriptors are kept at their own PCA weight and 190 x 4 = 760 ligand inputs reach the model. Summing one component's 190 columns reproduces that component's score exactly, so no quantity the scores are built from is lost. Loadings are applied *after* the fold scaler; a scaler placed afterwards would rescale each column to unit variance and erase the weights.

`configs/default.json` sets `features.pc_scores_form`. `loading_weighted` is the default described above; `component_scores` restores the earlier four summed PC scores (32 encoded inputs). A bundle prepared before this option existed has no such key and keeps `component_scores`, so previously prepared inputs are unaffected. Because the ligand block grows from 4 to 760 columns against the same 28 shared categorical columns, the isotropic RBF distance is now dominated by ligand descriptors; this is a deliberate statistical change, so use a new prepared configuration and run directory, and do not compare its scores against a `component_scores` run as if they were the same model. Each `pc_scores` checkpoint also stores a 2,688 x 760 float64 training matrix, roughly 17 MB per task against well under 1 MB before, so allow a few extra GB under `/share` for a full run; the default result ZIP still excludes checkpoints. The selected columns are recomputed from the pinned source snapshot, not hardcoded to the names above.

Reaction-feature scaling, one-hot encoder fitting, and target standardization occur within each training fold. The default OHE policy preserves `handle_unknown='ignore'`: a held-out ligand has an all-zero ligand block and supplies no chemical similarity. `declared_vocabulary` is an optional intentional alternative. It uses a vocabulary from the declared dataset; it still cannot learn chemical similarity for an unseen ligand. Unknown categories are logged in every fit.

## 4. Model and optimization settings

Default: GPyTorch `ExactGP`, zero mean on the train-standardized target, isotropic RBF, fixed unit signal variance, fixed Gaussian noise variance `1e-6` in standardized target units, float64. This matches the original sklearn model's statistical structure. Numerical jitter (`1e-8`) is separate from observation noise.

Adam optimizes the training marginal likelihood, initially learning rate 0.01, maximum 400 steps, minimum 100 steps, with a training-objective plateau check. This optimizer differs from sklearn's L-BFGS-B; identical fitted length scales or final scores are not promised. There are no hyperparameter priors or HMC sampling. `restarts=5` performs five independent initializations per fit and keeps the one with the best **training** marginal likelihood; no test data is consulted in that choice. Restarts multiply optimizer work per fit by roughly five, which dominates the walltime estimate a pilot should produce. Objective traces and stop reasons are returned. Reaching 400 steps does not assert convergence; examine the histories before publishing results. The plateau check can only fire at a step at or past `patience`, so a cap at or below it leaves the check unreachable and the stop reason is `max_steps_plateau_unreachable` rather than `max_steps`; `fit.json` records the same distinction as `plateau_reachable`. A short pilot always reports the unreachable form, so its stop reasons measure walltime and memory only and say nothing about convergence.

`configs/default.json` exposes optional ARD, learned signal scale, and learned Gaussian noise. Keep the same settings across all five models for a comparison. These options are statistical changes, so create a new prepared configuration and run directory when changing them. Do not select kernels or settings based on outer test performance; a performance-driven search requires an additional inner validation procedure.

The initial solver uses Cholesky for transparent comparison with sklearn at this dataset size. `solver='cg'` is an optional iterative solver with explicit tolerances. ExactGP training uses the complete training fold, not minibatches. CPU/GPU pilots should determine walltime and memory requests. One GPU serves one task; the package uses neither MPI nor nested joblib workers.

## 5. Transfer and prepare Hazel

Upload the archive using SFTP, Open OnDemand, or Globus and extract it under `/share/<group>/<user>/...`. The upload archive contains a `hazel_gp_workflow` folder with code and an **`inputs/`** bundle. All commands below run from that extracted project root.

NC State reports Slurm as the production scheduler since August 17, 2026, so use `hazel/slurm`. `hazel/lsf` holds the same six submissions written as `bsub` scripts, following the `training/hpc_submit_training_GP` conventions, for a cluster still running LSF. Slurm requires a typed GPU request such as `--gres=gpu:a100:1`; the LSF scripts express the equivalent as a `select[a10 || a30 || ...]` list. Training and GPU tests must run on allocated compute nodes. [HPC home](https://hpc.ncsu.edu/main.php), [GPU jobs](https://hpc.ncsu.edu/RunningJobs/Gpu.php).

Inspect your allocation and modules on the login node:

```bash
groups
sa
sqos
si --gpus
module avail conda
```

Every script carries its configuration in a block at the top, the way the training scripts do; nothing is exported into the environment before submitting. The shipped paths are `ddomlab/kagoble`, so edit `conda_env` and `output_root` in `setup_environment.sh` and in the scripts you submit if yours differ. Install the environment once, then preflight an allocated GPU:

```bash
bash hazel/setup_environment.sh
bash hazel/slurm/GP_preflight_GPU_arg.sh
```

Job logs go to `output_root`, a dated folder under `/share` named like the training sweeps' `HPC_history/hpc_YYYYMMDD`, which every script creates before submitting.

The script loads `conda`, creates Python 3.12, installs PyTorch 2.8.0 using its CUDA 12.6 wheel index, then installs the pinned training requirements. The preflight checks GPU visibility and a float64 Cholesky operation. Official PyTorch documents the 2.8.0 CUDA 12.6 wheel, and NC State's machine-learning page currently recommends CUDA 12.6. These do not replace testing the actual allocated GPU/driver. A generic CUDA module is unnecessary for the packaged PyTorch wheel unless your local environment requires it; compiling custom CUDA extensions is outside this workflow. [PyTorch wheels](https://pytorch.org/get-started/previous-versions/), [NC State installation guidance](https://hpc.ncsu.edu/Software/Apps-slurm.php?app=gputools).

Environment and Conda cache paths are separate because compute nodes cannot write to usrapps and home has a small quota. All runtime output goes to writable run folders. General external downloads are not assumed on compute nodes: the complete data bundle is uploaded in advance. Retrieve important results from `/share`; it is not backed up and files unaccessed for 30 days are deleted. [Conda](https://hpc.ncsu.edu/Software/Apps.php?app=Conda), [storage](https://hpc.ncsu.edu/Storage/Storage.php), [network](https://hpc.ncsu.edu/QuickStart/LinuxCluster.php).

## 6. Pilot, then production

The GPU scripts initially request one A100, four CPU cores, 16 GB host memory, and just under two hours. Adjust the GPU type, resources, account/QOS, or time using your actual allocation. The CPU script uses the `compute` partition. These are starter requests, not measured Hazel requirements.

First pilot one full-size task per representation, with ten optimizer steps, in its own run folder:

```bash
bash hazel/slurm/GP_pilot_GPU_arg.sh
```

The pilot reads the first task ID of each representation from `inputs/tasks.csv` instead of hardcoding them, because the registry is model-major and its split count changes whenever the bundle is re-prepared. The current bundle holds 110 tasks, 22 splits per representation, so those IDs are 0, 22, 44, 66 and 88. The retired `data_hazel/prepared/group1_v1` holds 270 because it predates matched IID becoming a non-repeating partition; its matched-IID folds test some rows up to 17 times and leave 20 untested, so collection rejects it. Re-prepare rather than reuse it. Set `pilot_task_ids` to pilot a chosen set instead. Monitor with `squeue -u "$USER"`, or `bjobs` under LSF. After completion inspect fit times, convergence histories, warnings and memory usage. To report the partial pilot, set `allow_partial=1` and `run_tag="pilot_v1"` in `GP_collect_CPU_arg.sh` and submit it; full collection intentionally rejects a partial pilot.

Production, after choosing sufficient walltime from the pilot:

```bash
bash hazel/slurm/GP_GPU_arg.sh
```

This submits one job per task, the way the training scripts submit one job per kernel combination. It reads the task count from the verified bundle manifest and chains collection behind the training jobs; set `submit_collection=0` to submit training alone and collect later. Change `gpu_request` and the `#SBATCH --time` line for a different allocation, or `#BSUB -q` and `#BSUB -W` in the LSF variant.

Unlike the previous job array, the loop submits every task at once with no concurrency cap, matching how `training/hpc_submit_training_GP` fires its roughly hundred-job sweeps. Narrow the loop range if your queue caps total submissions.

For a short partner GPU pilot, set `partition="gpu_partners"` in `GP_pilot_GPU_arg.sh`, add `--qos=short_gpu`, and keep to at most two hours. Account/partner QOS privileges must match your account. The collection job uses the default CPU account; edit it separately if your account needs an explicit association. [Partitions and resources](https://hpc.ncsu.edu/RunningJobs/Resources.php).

CPU fallback is `bash hazel/slurm/GP_CPU_arg.sh`, which writes to its own `experiment_v1_cpu` run folder. The same environment can execute the CUDA-enabled PyTorch wheel on CPU. Thread counts are fixed at four in the default configuration and in every submission script; change both together if benchmarking another allocation.

## 7. Resume, collect, download, review

Every task has its own directory and completion marker. Completed tasks are hash-checked and skipped on rerun. Errors preserve tracebacks. A manifest hash ties every result to its config, source data, split IDs, code and software versions. Use a new run folder after edits.

```bash
conda activate /usr/local/usrapps/ddomlab/kagoble/hazel_gp_py312
python -m hazel_gp status --bundle inputs --runs runs/experiment_v1
```

Put the reported failed or missing IDs in `retry_task_ids` in `GP_retry_GPU_arg.sh`, then submit that script; it passes `--retry` and chains collection like the production script. A scheduler kill can leave a `.running` lock; first confirm the old job has ended, then remove only that task's stale lock before retrying. An interrupted run initializer can similarly leave `.initializing`. Failed/incomplete fits restart from their seed; **completed checkpoints are reloadable, but this version does not resume Adam halfway through a fit**. This makes ordinary non-preemptable short arrays the intended route, rather than scavenger jobs.

The collector verifies exact test row identities, observed values, metadata and output hashes. It reports each split, unweighted mean split metrics, and pooled out-of-fold predictions for every method, matched IID included: each method is now a disjoint partition of the 3,072 rows, so no reaction is counted twice. `lolo_report` assembles the three LOLO views in one table - each ligand alone, the mean across the eight ligand-left-out models, and the pooled score after all eight folds. Pooled and averaged values are not interchangeable: pooled R² is measured against the variance of the whole dataset while a per-ligand R² is measured against the variance within that ligand, which is smaller, so the averaged figure is systematically harsher. Partial reports carry incomplete flags and are not silently ranked as a full benchmark.

Collection exports `exports/results_JOBID.zip`, or `exports/results_RUNTAG_JOBID.zip` from the standalone collect script. Download and extract it to `data_hazel/returned/`, then open **`02_review_results.ipynb`** and set its `RUNS` path to the extracted experiment directory. Local review does not require PyTorch. The notebook supports model/method/ligand selection, parity and calibration plots, per-ligand comparisons, and an explicit final figure/table export cell.

Returned data include observed/predicted UV-area yield, latent and predictive standard deviation, 50/80/95% intervals, PIT values, row IDs and split assignments. Metrics include R², RMSE, MAE, Kendall tau, predictive NLL, coverage, interval widths, PIT-CDF mean absolute error, and residual/uncertainty Spearman correlation. Undefined constant-target R² remains blank. Predictions are not clipped to 0-100.

The default result ZIP excludes checkpoints and preprocessors for a smaller download; its collected files can be read directly with `load_results`. To rerun collection locally or reload models, archive the full run instead:

```bash
python -m hazel_gp pack-results --runs runs/experiment_v1 \
    --output exports/results_with_models.zip --include-checkpoints
```

Setting `include_checkpoints=1` in `GP_collect_CPU_arg.sh` does the same from a job.

Only load model/preprocessor files that you created and trust. Checkpoints retain training inputs/targets, model state, feature ordering and target normalization; `predict_from_checkpoint` handles the saved preprocessing and restores target units. Supply its raw feature frame with the same named columns as the selected model (see `feature_frame`). It does not scrape or calculate quantum-chemical descriptors for a new ligand.

For the earlier group-array interface:

```python
from hazel_gp.results import load_results, grouped_predictions
tables, info = load_results("data_hazel/returned/experiment_v1")
grouped = grouped_predictions(tables["predictions"], "selected_2", "lolo")
# grouped['SPhos']['y_pred'] is a list of arrays, one per split; seeds remain in metadata.
# Matched IID no longer has per-ligand reference groups, so it returns a single group.
```

With the preserved seed 42, the 80/20 holdout has the same test membership as the first shuffled five-fold test set for this 3,072-row dataset. It is retained to match the original notebook and is not an independent validation experiment. The saved row IDs make this overlap explicit.

## Files and validation

- `01_prepare_local.ipynb`, `02_review_results.ipynb`: settings, review and export.
- `hazel_gp/config.py`: defaults, validation, hashes, version records.
- `hazel_gp/data.py`: original downloads, cleanup, input bundles.
- `hazel_gp/features.py`: frozen reference PCA and train-only transformations.
- `hazel_gp/splits.py`: shared LOLO/IID/K-fold/holdout assignments.
- `hazel_gp/train.py`: ExactGP, task execution and checkpoint reload.
- `hazel_gp/results.py`: collection, metrics, plots and result archives.
- `hazel_gp/notebook.py`, `cli.py`: small notebook/command-line interfaces.
- `hazel/`: `setup_environment.sh`, plus `slurm/` and `lsf/` variants of the same six submissions - preflight, pilot, GPU, CPU, retry and collect.
- `tests/`, `VALIDATION.md`: correctness tests and what was actually verified.

Run `python -m pytest -q tests` after installing the test requirements. The real-data integration check runs when the prepared directory named in `configs/default.json` is present. Use `python -m hazel_gp --help` for the complete command interface. See [GPyTorch's ExactGP tutorial](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html) for the mean/kernel/likelihood structure.
