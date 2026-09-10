# Validation of version 1.0.0

Validated on September 9, 2026. This file records implementation checks, not final model performance. No Hazel account was accessed and no Hazel job was submitted.

## Original data

The downloader successfully retrieved all three original GitHub CSV URLs into the new `data_hazel/raw` directory. Fresh cleanup reproduced:

- 5,760 original reaction rows.
- 3,712 rows after the original solvent/ligand exclusions.
- 3,072 Group1 rows, with eight ligands and 384 rows per ligand.
- 1,223 original-reference ligands and 190 descriptors.
- The same 12 top-loading descriptors in the saved original notebook outputs.
- 54 unique split definitions, shared across five representations: 270 tasks.

The source snapshot and hashes are in `reference_snapshot/`. The immutable prepared example is in `examples/verified_inputs/`. The normal notebook still creates new working folders and downloads from the same original links.

## Automated checks

`python -m pytest -q tests` completed with **9 passed**. The checks cover:

1. Full-reference PCA calculations and serialization/reload against a separately fitted sklearn PCA.
2. Consistent buried-volume units, correct orbital-gap sign, and preservation of raw source values.
3. Shared/reproducible folds, correct counts, disjoint train/test rows, LOLO ligand exclusion, IID ligand coverage, and distinct IID repeat assignments.
4. Training-only scaling unaffected by extreme test-feature changes.
5. Fixed-hyperparameter GPyTorch posterior means and latent standard deviations matching sklearn to 1e-7 tolerance; observation-noise variance accounted for separately.
6. Training objective execution, fixed-noise behavior and target normalization.
7. Predictive uncertainty metrics and undefined R2 for constant targets.
8. Original-data cleanup and exact encoded dimensions for all five models.
9. A complete small synthetic benchmark testing full collection, pooled versus mean-split R2, omitted pooled IID scores, completion skipping, incompatible settings rejection, and input/output corruption detection.

Synthetic benchmark data used in automated tests are generated inside temporary directories and are never mixed with the original scientific dataset.

## Full-size CPU integration pilots

Twenty full-size original-data fits completed: **each of five representations × one LOLO fold, one matched IID split, one five-fold split, and the 80/20 split**. Each used five optimization steps to exercise the execution path. They are smoke tests, not converged performance estimates or the full 270-fit benchmark.

- All 20 tasks wrote predictions, metrics, objective histories, preprocessors and checkpoints.
- All five representation checkpoints were reloaded; predicted means and predictive standard deviations matched the saved values within 1e-9 tolerance.
- A repeated completed task was recognized and skipped without retraining.
- Full collection correctly rejected the pilot because 250 tasks were missing.
- Explicit partial collection succeeded and labeled incomplete methods/run status.
- A result ZIP was created, extracted, and read successfully; grouped IID arrays were recovered.
- The final input-export button callback created a verified 270-task bundle and upload archive.

The pilots used Python 3.12.14, PyTorch 2.8.0+cpu, GPyTorch 1.15.2, linear_operator 0.6.1, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and scikit-learn 1.8.0. Five-step fit times ranged from about 3.34 to 5.29 seconds in this environment, excluding queueing and much of data loading/prediction. Do not use these as a Hazel GPU runtime estimate.

## Notebook, shell and archive checks

- Both notebook files validate against the notebook schema and are delivered with no stored outputs or execution counts.
- Every code cell in the preparation notebook executed sequentially in a fresh Python/IPython process, including cached downloads, cleanup, model/split tables and export-widget construction.
- Every code cell in the review notebook executed sequentially in a fresh process against the returned pilot data. Plot generation, control construction, grouped arrays and the default disabled final export passed.
- Calibration and ligand-comparison plots were rendered and visually inspected.
- The Python modules compile and all six shell/batch scripts pass Bash syntax checks.
- Input/result ZIP creation and input hash verification were exercised.

The runtime used for development blocks the socket transports required to start a Jupyter kernel. Therefore notebook cell execution and controls were checked in fresh Python/IPython processes, not through a live Jupyter frontend. Interactive rendering in VS Code, Windows-specific environment installation, Hazel software installation, Slurm scheduling and actual GPU execution remain user-environment checks. The supplied allocated-node preflight is intended to verify the GPU environment before training.

The optimizer's default 400-step budget and memory/walltime requests are starting settings. Review convergence and resource consumption in the supplied Hazel pilot before a production submission. No relative model ranking is inferred from the five-step validation runs.
