# gp_collab_hub

The hub for DOPE-MURI GP benchmarking: **one** engine, **many** datasets.

`gp_collab_hazel`, `gp_collab_cernak` and `gp_collab_rxnpredict` were three
clones of the same package, each edited to suit one screen. That worked, but it
meant every fix had to be made three times, the three copies had drifted, and
comparing them meant unloading and re-importing `gpc` from `sys.modules` between
datasets. Here the dataset-specific facts live in the bundle's `config.json` —
its target column, its group column, its condition fields — and the code reads
them. Adding a screen is a folder under `datasets/`, not a fork.

The three original folders are left alone. Their runs are readable from here.

## The loop

```
datasets/<name>/prepare_<name>.ipynb     1. intake:  a screen -> a bundle
build_run.ipynb                          2. plan:    choices -> config + zip
        |  upload, run on the cluster, bring runs/<tag>/ back
make_report.ipynb                        3. report:  a run -> results/<name>_<date>/
compare_datasets.ipynb                   4. compare: across datasets
```

### 1. Intake — `datasets/<name>/prepare_<name>.ipynb`

One notebook per dataset *type*, tailored to that screen and meant to be
edited. It reshapes the screen's own table into the canonical schema and calls
`gpc.prep.write_bundle`.

- `perera/` — Perera flow Suzuki, 3,072 reactions, 8 ligands
- `ahneman/` — Ahneman Buchwald–Hartwig, 3,955 reactions, 4 ligands.
  Rebuilds genuinely from the cached raw download in `raw/`.
- `gesmundo/` — Gesmundo Suzuki informer, 1,320 wells, 5 catalysts
- `_template/` — copy this for a new screen

Perera and Gesmundo run in `SOURCE = "adopt"` mode: they re-emit the verified
reactions table and validate it against the hub schema. Their `SOURCE = "raw"`
cells document every recorded decision and raise rather than guess — the code
that first built those tables is not in this repository.

A bundle is:

| file | what it is |
| --- | --- |
| `config.json` | what the dataset IS: target, group, condition fields, row and group counts |
| `reactions.csv` | one row per reaction: `row_id`, group, `kraken_id`, target, conditions |
| `ligand_features.csv` | Kraken descriptors for the ligands this screen uses |
| `pca_reference.json/.npz` | the frozen reference PCA, **copied, never refitted** |
| `ligand_mapping.csv` | group name → Kraken id, for figures and provenance |

### 2. Plan — `build_run.ipynb`

Asks which dataset, which feature sections, which evaluation methods, which GP
hyperparameters and which cluster resources; writes `configs/<tag>.json` (and a
`.yaml` copy for hand editing) and packs `exports/<tag>.zip`.

The zip carries the engine, that dataset's inputs and the submit scripts — about
120–170 KB, against several MB for the old per-project folders. The run tag,
conda prefix, walltime, GPU type and memory are written **into** the scripts, so
nothing needs editing on the cluster. The config inside is always JSON, so the
cluster never needs PyYAML to start a job.

Two mistakes are refused here rather than three hours into a job: a feature
section or GP key that does not exist, and an `iid_stratified_<n>` whose fold
count does not match the screen's group count — that method exists to be LOLO's
matched control, and with the wrong `n` it silently is not one.

### 3. Report — `make_report.ipynb`, or `python -m gpc ... report`

```bash
python -m gpc --bundle datasets/perera/inputs --runs runs/perera_20260923 report
```

writes `results/<dataset>_<date>/`:

```
figures/  pooled_r2_by_featureset.png
          lolo_by_held_out_<group>.png
          kraken_space.png
tables/   summary.csv  metrics_by_split.csv  predictions.csv  lolo_report.csv
report.json
```

**Every saved graphic is defined in one file: `gpc/figures.py`.** Titles, axis
labels, colours, figure sizes, bar widths, tick angles, label positions, axis
limits — all in the `STYLE` dict at the top. Change it there and re-run the
report; nothing else in the package draws anything that gets written to disk.

The three figures:

- **`pooled_r2_by_featureset`** — titled `<dataset>: Pooled R² by feature
  selection`, y axis just `R²`. Heights are the pooled out-of-fold figures, so
  they are not the mean of the per-fold bars in the next chart.
- **`lolo_by_held_out_<group>`** — one bar per held-out ligand/catalyst, one
  colour per feature section.
- **`kraken_space`** — two panels: PC1 vs PC2, and the two buried-volume
  descriptors, with the full 1,223-ligand Kraken cloud behind this dataset's own
  ligands. Point labels are placed automatically and then pushed apart until
  their measured text boxes stop overlapping; pin one by hand with
  `STYLE["kraken"]["offsets"] = {"XPhos": (18, -14)}`.

Both bar charts pin the y axis at zero. A negative R² therefore has nowhere to
draw, so its value is written at the baseline in red instead of the bar silently
vanishing.

### 4. Compare — `compare_datasets.ipynb`

The across-datasets charts: 5-fold CV by feature set, matched IID against LOLO,
every screen's ligands in one Kraken space, and ARD relevance by feature set.
Same shape as the old `compare_featuresets.ipynb` at the repo root, without the
`sys.modules` juggling.

## Layout

```
gpc/            the engine
  config.py     run + bundle configs, JSON or YAML
  data.py       load a bundle, whichever dataset wrote it
  features.py   the five feature sections and the per-fold encoder
  splits.py     lolo / iid_stratified_<n> / kfold_stratified_<n> / kfold_<n>
  train.py      one (section, method) task  [needs torch]
  collect.py    finished tasks -> tables
  export_hazel.py  -> the collected/ schema results.py reads
  results.py    review and interactive plotting
  figures.py    >>> every saved figure, and every setting that controls it <<<
  report.py     a run -> results/<dataset>_<date>/
  bundle.py     the HPC zip
  prep.py       shared intake helpers
  vendor/       GP_collab's kernel, scoring and UQ code, unchanged
reference/kraken/   the 1,223-ligand reference and the frozen PCA
datasets/<name>/    an intake notebook and the bundle it writes
configs/            run configs
exports/            zips to upload
results/            what comes back
runs/               raw task output from the cluster
```

## Feature sections

Identical in meaning across every dataset, which is what makes them comparable.

| section | ligand inputs |
| --- | --- |
| `group_ohe` | group identity one-hot — the baseline that carries no chemistry |
| `selected_2` | `vbur_pct_boltz` + `vbur_pct_min` — two sterics |
| `vbur_min_vmin` | `vbur_pct_min` + `vmin_vmin_boltz` — one steric, one electronic |
| `vbur_boltz_vmin` | `vbur_pct_boltz` + `vmin_vmin_boltz` — one steric, one electronic |
| `selected_5` | `selected_2` plus buried-volume range, dipole, HOMO–LUMO gap |
| `pc_top` | top 3 original descriptors from each of PC1–PC4, deduplicated |
| `pc_scores` | the four reference-PCA component scores |

The three two-descriptor sections encode to the same width (30 inputs on
Perera), so they differ only in which two descriptors they carry. Read against
`selected_2`, the two `vbur_*_vmin` pairs ask whether the second buried volume
is carrying information an electronic descriptor would carry better.
`vmin_vmin_boltz` is Kraken's minimum electrostatic potential near phosphorus,
Boltzmann-averaged over conformers — the only Vmin *potential* in the table
(`vmin_r_boltz` beside it is the P-to-Vmin distance, a geometry).

Sections are defined in one place, `config.SELECTED_SETS`. Adding an entry there
is all it takes to make a new descriptor section runnable end to end.

`ligand_ohe` and `catalyst_ohe` are accepted as names for `group_ohe`, so runs
collected before the hub still load and still line up in the comparison charts.

`run.feature_columns` in a run config redefines which descriptors `selected_2`,
`selected_5` or `pc_top` feed. That is a legitimate experiment, but it breaks
comparability with every other dataset, so it is recorded in the run config and
in every task's `meta.json` rather than being a silent default.

## Evaluation methods

| method | definition |
| --- | --- |
| `lolo` | hold out one group at a time — the out-of-distribution question |
| `iid_stratified_<n>` | LOLO's matched control: same fold count, every group present in training |
| `kfold_stratified_<n>` | ordinary stratified CV |
| `kfold_<n>` | the same without stratification |

Two sizing notes that change how the numbers read:

- When a screen has exactly *n* groups, LOLO is already an *n*-fold partition,
  so `kfold_stratified_n` **is** its matched control — Gesmundo's matched-IID
  and 5-fold columns are identical by construction, not by accident.
- LOLO folds are not equal sized when the design is unbalanced. Gesmundo's
  Aphos G3 fold removes 441 of 1,320 rows against ~215–226 for the others, so
  compare pooled metrics across methods, not fold means.

## Running it

Locally (review only — no PyTorch needed):

```bash
python -m pytest -q tests
python -m gpc tasks                     # the job registry
python -m gpc splits                    # fold sizes and group balance
python -m gpc --bundle datasets/perera/inputs --runs runs/<tag> report
```

Training needs torch; see `hazel/setup_environment.sh` and `RUN_ON_HPC.md`
inside each zip.

## Where PC1–PC4 come from

They are **computed in this project, not shipped by Kraken.** Kraken provides
190 raw descriptors per ligand; the PCA over them is ours.

`ReferencePCA.fit` standardizes all 190 descriptors across all **1,223** Kraken
ligands and takes the first four components — 29.0%, 13.0%, 11.2% and 6.2% of
descriptor variance, 59.3% together. The fitted mean, scale, mean-centre and
loading matrix are frozen into `reference/kraken/pca_reference.npz`, and
`prep.write_bundle` **copies** them into every bundle rather than refitting. The
`PC1..PC4` columns in `ligand_features.csv` are that frozen projection applied to
each ligand, reproducible to 1e-14 by `ReferencePCA.transform`.

Two consequences worth holding onto:

- A ligand's PC scores do not depend on which screen it appears in. That is what
  makes `pc_top` and `pc_scores` comparable across datasets, and it is why the
  PCA is never refitted per dataset.
- The PCA was fitted on the **whole** Kraken population, including ligands no
  screen used. It is an unlabelled external reference, so this is not target
  leakage — but it does mean the components describe Kraken's chemical space,
  not any one screen's.

`vbur_pct_boltz`, `vbur_pct_min`, `vbur_pct_delta` and `homo_lumo_gap_eV` are
**derived** columns added by the DOPE-MURI prep and are *not* among the 190 PCA
inputs. `dipolemoment_boltz` and `vmin_vmin_boltz` are raw Kraken descriptors and
*are* PCA inputs.

## What changed from the three original folders

- **One engine.** The group column, target and condition fields come from the
  bundle. `gpc/results.py` used to hardcode the one-hot prefix `cat__ligand_`,
  so `rank_features(..., ligand_only=True)` matched nothing on a
  catalyst-grouped screen and failed with a misleading "encodes no ligand-block
  columns to rank". That is fixed here; the bug is still present in
  `gp_collab_cernak`.
- **One reference.** Every bundle's descriptors are re-derived from
  `reference/kraken/`, and the migration verified they match each original
  bundle's own values exactly.
- **One figure file.** `gpc/figures.py` owns every saved graphic.
- **Smaller zips.** `ligand_features.csv` is trimmed to the ligands a screen
  actually uses; the 2.8 MB reference stays here, where the figures are drawn.
