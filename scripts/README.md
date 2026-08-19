# scripts/

This folder holds all entry-point scripts for the project. The reusable
library code (data generation, model architectures, RC formulas) lives in
`src/spect/baseline/`. Every file here is a thin command-line driver that
imports from there and does one job.

The files are not grouped into subfolders. Most are standalone, and
several `.sh` submission scripts under `scripts/hpc/` call them by a fixed
path, so moving them would mean updating those paths too. Instead, files
are grouped by naming convention: `quantify_*` computes RC metrics,
`visualize_*` and `plot_*` generate figures, `run_inference_*` applies a
trained checkpoint, `print_*` summarises an existing CSV or log without
doing new computation, `generate_*` produces input files for the
SIRF/HPC pipeline, and `validate_*` / `inspect_*` are sanity checks that
are not part of the main results pipeline.

## Data generation

- `generate_xcat_parfiles.py`: randomises N `.par` files from an XCAT
  template, following the published randomisation protocol for body,
  cardiac, respiratory, and organ-activity variation. Its output feeds
  into `scripts/hpc/generate_data/run_xcat_parfiles.sh`, which calls the
  `dxcat2` binary and is not runnable from this Python environment
  directly.

## Dataset and environment sanity checks

- `inspect_dataloader.py`: a quick smoke check. It loads the train, val,
  and test splits through `SPECTDataset`, pulls one batch, and prints
  the shapes and value ranges. Run this first when setting up the repo,
  to confirm the data path and environment are working.
- `validate_dataset.py`: deeper dataset self-consistency checks that do
  not involve a model. It checks completeness across all 5 alpha levels,
  the sum ratio against the expected alpha scaling, the noise level
  trend across alpha, and label consistency across alpha folders. It
  auto-detects which phantom indices are present, so it works against
  the full dataset or a smaller sample. Figures are saved to `logs/` and
  used by `notebooks/02_dataset_inspection.ipynb`.

## Methods-section illustrative figures

These show what the raw data and noise model look like. No trained
checkpoint is involved; they read straight from `data/`.

- `plot_ellipsoid_phantom_example.py`: one ellipsoidal phantom, shown as
  a clean label plus all 5 noisy inputs.
- `plot_xcat_phantom_example.py`: one XCAT phantom, shown as a clean
  label plus its one assigned noisy input.
- `plot_earl_phantom_example.py`: the EARL phantom, shown as a clean
  label plus all 5 noisy inputs.

## Model inference (dump denoised output to disk)

These run a trained checkpoint over a dataset and save its restored
count-domain output as `.npy` files, so the quantification and
visualization scripts below can read the results without reloading the
model.

- `run_inference_dump.py`: for the ellipsoid or XCAT datasets. Iterates
  over (phantom_idx, alpha) pairs from a split, or a fixed list of
  phantom indices.
- `run_inference_nema_earl.py`: for the NEMA/EARL phantom. Iterates over
  (alpha, seed) pairs.

## Quantitative evaluation (Recovery Coefficient)

Each script computes RC against the label, and separately compares the
label against ground truth to isolate reconstruction-only bias from CNN
error. There is one script per dataset, all using the same three-way
comparison logic.

- `quantify_noisy_baseline.py`: the ellipsoidal dataset, using
  seed-generated VOI masks.
- `quantify_xcat.py`: the XCAT dataset, using real per-tissue VOI masks
  taken from the original XCAT activity map. This gives a more exact
  ground truth than the ellipsoid version.
- `quantify_nema_earl.py`: the NEMA/EARL phantom, using real fixed
  sphere masks.
- `print_earl_size_alpha_table.py`: re-aggregates an existing
  `quant_earl_*.csv` into a sphere-diameter by alpha grid. Pure stdlib,
  no re-computation from the `.npy` volumes.
- `print_history_table.py`: reads `history.npz` training logs and prints
  the final and best-epoch PSNR/SSIM.

## Qualitative comparison figures

These read the `.npy` files dumped by the inference scripts above and
plot grids of noisy input, error, model output, and ground truth.

- `visualize_predictions.py`: one checkpoint at a time, across all 5
  alpha levels, for the ellipsoid or XCAT dataset. Imported by the three
  scripts below.
- `visualize_earl_predictions.py`: the same, for the EARL phantom.
  Imported by `visualize_earl_full_comparison.py`.
- `visualize_full_comparison.py`: two checkpoints side by side in one
  figure, for example old-method vs label x alpha, or U-Net vs Swin.
- `visualize_old_vs_labelalpha.py`: a compact 4-column old-vs-label x
  alpha comparison, showing only output and error, without repeating the
  noisy-input and ground-truth columns.
- `visualize_earl_full_comparison.py`: U-Net vs Swin UNETR side by side,
  for one EARL background variant.

## Tests

Assert-based checks live in `tests/`, not here. Keeping them separate
makes it clear which files are the actual pipeline and which are
verification code.