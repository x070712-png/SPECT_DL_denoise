# SPECT_DL_denoise

**Author**: Xingyu Liu

**Supervisors**: Kris Thielemans, Efstathios Varzakis, Cate Gascoigne

**Affiliation**: MSc Scientific and Data Intensive Computing (SDIC), University College London (UCL)

**Project type**: MSc Dissertation, PHAS0077 (2025/26)

This repository implements and evaluates deep-learning denoising of SPECT
images reconstructed at reduced count levels. Two image-domain
architectures are trained and compared:

- **3D U-Net**
- **Swin UNETR**

Both are trained under two target formulations -- the **baseline method**
(fixed-scale label) and the **label x alpha method** (label scaled by the count level, correcting an implicit low-count scale inflation in the training target that otherwise causes the Recovery Coefficient to collapse at low count levels). And evaluated across five simulated count levels (alpha = 1.0, 0.5, 0.25, 0.125, 0.05) plus a simulated EARL-style NEMA phantom.

The SPECT simulation and reconstruction pipeline (SIRF/STIR forward
projection, Poisson noise, OSEM reconstruction) and the two network
architectures are adapted from Wei Miao's dissertation codebase
([ucapwmi/SPECT_codes](https://github.com/ucapwmi/SPECT_codes)); see
**Acknowledgements** below for what is reused vs. original to this
project.

---

## 1) Environment

**SIRF v3.7+ (STIR backend)** is required for all simulation,
reconstruction, and inference steps, and is not installable via pip --
follow the [official SIRF installation instructions](https://github.com/SyneRBI/SIRF)
(SuperBuild, conda package, or the SIRF-SuperBuild Docker image) first.
This project was developed and run inside a Docker container with SIRF
pre-installed, on UCL's Myriad HPC cluster for GPU training/inference.

Once SIRF is available in your environment, install the remaining Python
dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/varzakis/phantomgen.git
```

`phantomgen` (Stathis Varzakis's package) is only needed for the EARL/NEMA
phantom scripts (`src/spect/baseline/generate_earl_phantom.py`). You can skip it
if you only need the ellipsoid/XCAT pipeline.

All commands in this README are run from the repository root, with `src/`
on the Python path:

```bash
export PYTHONPATH=src:$PYTHONPATH
```

## 2) Repository layout

```
.
├── src/spect/baseline/                     # library code (imported by everything below)
│   ├── config.py                            # count levels, acquisition/UMAP/OSEM/normalisation config
│   ├── sirf_bridge.py                       # SIRF/STIR bridge: forward projection, Poisson noise, OSEM recon
│   ├── dataset.py                           # SPECTDataset, build_split() train/val/test partitioning
│   ├── model.py                             # CustomUNet3D, get_swin_unetr() network definitions
│   ├── quantification.py                    # VOI masks + Recovery Coefficient / bias math (ellipsoid ground truth)
│   ├── generate_ellipsoids.py               # random-ellipsoid phantom generator
│   ├── generate_dataset.py                  # batch-generates the 500-phantom ellipsoid dataset
│   ├── generate_xcat_dataset.py             # runs XCAT activity phantoms through the sim/recon pipeline
│   ├── generate_earl_phantom.py             # builds the EARL/NEMA phantom activity/attenuation/sphere masks
│   └── generate_eval_phantom_dataset.py     # forward-project + reconstruct one fixed eval phantom across count levels
├── train/
│   ├── train_unet.py                        # 3D U-Net pre-training / fine-tuning entry point
│   └── train_swin_unetr.py                  # Swin UNETR pre-training / fine-tuning entry point
├── scripts/                                 # entry-point scripts: data generation, inference,
│   │                                          #   quantification, visualisation (see scripts/README.md)
│   └── hpc/                                  # SGE/qsub submission scripts for Myriad (see scripts/hpc/README.md)
├── notebooks/                                # demo/inspection notebooks
├── tests/                                    # pytest unit tests
├── data/                                     # datasets (not tracked by git except data/template/, see Data below)
├── checkpoints/                              # trained model weights (not tracked by git, see Pre-trained Weights below)
├── requirements.txt
├── LICENSE                                   # Apache 2.0 license
├── NOTICE                                    # copyright and third-party attribution
├── general.samp.par                          # XCAT generation template (from Wei Miao's repo, see Acknowledgements)
└── README.md
```

`scripts/` and `scripts/hpc/` each have their own README listing every
individual file and its role -- start there if you're looking for a
specific script.

## 3) Data

Full dataset generation (500 ellipsoid phantoms x 5 count levels, the
500-phantom XCAT set, and the EARL/NEMA phantoms) requires SIRF/STIR and,
for XCAT, a licensed copy of the XCAT phantom generator (`dxcat2`) -- see
`scripts/hpc/generate_data/` and its README for the generation pipeline.

`general.samp.par` (repo root) is the base XCAT parameter template that
`scripts/generate_xcat_parfiles.py` randomises from -- taken directly from
Wei Miao's dissertation codebase (see Acknowledgements), used as-is.

Because the full datasets are large, a **sample data package** is
provided for quick reproduction of the headline results without
regenerating everything from scratch:

- **Sample data (download)**: [Google Drive folder](https://drive.google.com/drive/folders/1wPSAYj6ed-_VLEANd4L5KxNB0ucvVpET?usp=drive_link)
  -- also contains the checkpoints (see Pre-trained Weights below).
- **Placement**: download the `data/` subfolder from the link above and
  place it at the repository root, so the paths below exist as-is.

Contents of the sample package:

```
data/
  template/
    temp_sino.hs, temp_sino.s       # acquisition-geometry template (also in git)
  dataset/
    alpha_1p0/ ... alpha_0p05/
      input_0090.npy ... input_0099.npy   # ellipsoid phantoms 90-99 (held-out test set)
      label_0090.npy ... label_0099.npy
  xcat_dataset/
    alpha_1p0/    input_0090.npy ... input_0099.npy, label_0090.npy ... label_0099.npy
    alpha_0p5/    input_0190.npy ... input_0199.npy, label_0190.npy ... label_0199.npy
    alpha_0p25/   input_0290.npy ... input_0299.npy, label_0290.npy ... label_0299.npy
    alpha_0p125/  input_0390.npy ... input_0399.npy, label_0390.npy ... label_0399.npy
    alpha_0p05/   input_0490.npy ... input_0499.npy, label_0490.npy ... label_0499.npy
    # 50 held-out XCAT test-split phantoms -- each alpha has its own distinct
    # 10 indices, unlike the ellipsoid set above where 90-99 is shared across all 5 alphas.
  earl_dataset_v3_bg0/
    alpha_1p0/ ... alpha_0p05/            # EARL phantom, background condition bg0
  earl_dataset_v3_bg_ratio10/
    alpha_1p0/ ... alpha_0p05/            # EARL phantom, background condition bg_ratio10
  earl_phantom_v3_bg0/
    activity.npy, att_map.npy, EARL_sphere_{13,17,22,28,37,60}mm.npy   # source phantom + VOI masks, bg0
  earl_phantom_v3_bg_ratio10/
    activity.npy, att_map.npy, EARL_sphere_{13,17,22,28,37,60}mm.npy   # source phantom + VOI masks, bg_ratio10
```

Phantom indices 90-99 are the fixed 10-phantom subset used throughout this
project for the alpha-by-alpha figures and tables (see
`--phantom_indices` / `--fixed_phantom` in `scripts/README.md`); they are
a subset of the full 500-phantom ellipsoid test split.

`data/template/` (the acquisition geometry template) is small enough to
be tracked directly in git, so it does not need to be downloaded
separately.

## 4) Pre-trained weights

Four checkpoints are needed to reproduce the main results (2 architectures
x label-alpha method, ellipsoid-pretrained and XCAT-fine-tuned):

- **Checkpoints (download)**: same [Google Drive folder](https://drive.google.com/drive/folders/1wPSAYj6ed-_VLEANd4L5KxNB0ucvVpET?usp=drive_link)
  as the sample data above, in the `checkpoints/` subfolder.

Download the `checkpoints/` subfolder and place it at the repository root
so the following paths exist:

```
checkpoints/
  3d_unet_label_alpha/best_model.pth                     # U-Net, ellipsoid-pretrained
  swin_unetr_label_alpha/best_model.pth                  # Swin UNETR, ellipsoid-pretrained
  3d_unet_xcat_finetune_label_alpha/best_model.pth        # U-Net, XCAT-fine-tuned (used for EARL/XCAT results)
  swin_unetr_xcat_finetune_label_alpha/best_model.pth     # Swin UNETR, XCAT-fine-tuned (used for EARL/XCAT results)
```

(The old-method checkpoints -- `checkpoints/3d_unet/`,
`checkpoints/swin_unetr/`, and their `_xcat_finetune` counterparts -- are
only needed to reproduce the old-vs-label-alpha comparison figures, and
are not included in the sample package; retrain them with
`train/train_unet.py` / `train/train_swin_unetr.py` without
`--scale_label_by_alpha` if needed.)

## 5) How to run

### 5.1 Sanity check the environment and data

```bash
python3 scripts/inspect_dataloader.py
```

Loads the train/val/test splits and prints shapes/value ranges -- run
this first after setting up data to confirm paths are correct.

### 5.2 Training (from scratch)

```bash
python3 train/train_unet.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/3d_unet_label_alpha \
    --scale_label_by_alpha

python3 train/train_swin_unetr.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/swin_unetr_label_alpha \
    --scale_label_by_alpha
```

Drop `--scale_label_by_alpha` to train the old-method variant instead.
Full training requires the full 500-phantom `data/dataset`, not the
90-99 sample subset above.

### 5.3 Fine-tuning on XCAT

```bash
python3 train/train_unet.py \
    --data_dir data/xcat_dataset \
    --init_checkpoint checkpoints/3d_unet_label_alpha/best_model.pth \
    --checkpoint_dir checkpoints/3d_unet_xcat_finetune_label_alpha \
    --scale_label_by_alpha
```

See `scripts/hpc/finetune/` for the equivalent Swin UNETR command and the
exact hyperparameters (learning rate, epochs) used for the reported
results.

### 5.4 Reproducing the fixed-10-phantom results (using the sample data package)

Dump denoised output for phantoms 90-99, all 5 alpha levels (output
directory name must end in `_fixed10` to match `visualize_predictions.py`'s
convention, see below):

```bash
python3 scripts/run_inference_dump.py \
    --data_dir data/dataset \
    --checkpoint checkpoints/3d_unet_label_alpha/best_model.pth \
    --model unet \
    --phantom_indices 90,91,92,93,94,95,96,97,98,99 \
    --out_dir logs/denoised/3d_unet_label_alpha_fixed10
```

Compute Recovery Coefficient / bias against the label:

```bash
python3 scripts/quantify_noisy_baseline.py \
    --data_dir logs/denoised/3d_unet_label_alpha_fixed10 \
    --phantom_indices 90,91,92,93,94,95,96,97,98,99 \
    --input_prefix denoised \
    --label_dir data/dataset \
    --out_csv logs/quant_3d_unet_label_alpha_fixed10.csv
```

Generate the qualitative comparison figures (checkpoint paths are
pre-configured in `CHECKPOINTS_BY_DATASET` inside the script -- edit there
if your checkpoints live somewhere else):

```bash
python3 scripts/visualize_predictions.py \
    --checkpoint_key unet_label_alpha \
    --fixed_phantom 90 \
    --fixed10_dirs
```

### 5.5 Reproducing the EARL phantom results
 
Same three-stage pattern (dump -> quantify -> visualise), for the EARL
phantom's `bg0` background condition. Note the explicit `--label_dir` on
the quantify step below -- it defaults to `--data_dir`, but
`run_inference_nema_earl.py` never copies `label.npy` into its `--out_dir`,
so leaving it unset here (pointed at the CNN-output directory) would make
every alpha print `[skip] ... missing .../label.npy` and produce an empty
output CSV:
 
```bash
python3 scripts/run_inference_nema_earl.py \
    --data_dir data/earl_dataset_v3_bg0 \
    --checkpoint checkpoints/3d_unet_xcat_finetune_label_alpha/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet_xcat_labelalpha_earl_v3_bg0
 
python3 scripts/quantify_nema_earl.py \
    --data_dir logs/denoised/3d_unet_xcat_labelalpha_earl_v3_bg0 \
    --label_dir data/earl_dataset_v3_bg0 \
    --activity data/earl_phantom_v3_bg0/activity.npy \
    --sphere_dir data/earl_phantom_v3_bg0 \
    --sphere_prefix EARL_sphere_ \
    --input_prefix denoised \
    --out_csv logs/quant_earl_v3_bg0_3d_unet_labelalpha.csv
 
python3 scripts/visualize_earl_predictions.py \
    --variant v3_bg0 \
    --checkpoint_key unet_xcat_labelalpha
```
 
Swap `v3_bg0` / `earl_dataset_v3_bg0` / `earl_phantom_v3_bg0` for
`v3_bg_ratio10` throughout to reproduce the 10:1 background condition
instead. See `scripts/README.md` for every script's exact role and
arguments, and `src/spect/baseline/generate_earl_phantom.py`'s docstring
for what `bg0` vs `bg_ratio10` and the sphere activity concentration mean
physically.

### 5.6 Running on Myriad (HPC)

All of the above have matching `qsub` submission scripts under
`scripts/hpc/`, organised by pipeline stage (`generate_data/`, `train/`,
`finetune/`, `inference/`, `visualize/`). See `scripts/hpc/README.md` for
the full mapping and submission conventions.

## 6) Tests

```bash
pytest tests/
```

Covers the Recovery Coefficient / VOI-mask math in
`src/spect/baseline/quantification.py`, helper functions in
`scripts/quantify_noisy_baseline.py`, and slice-selection logic in
`scripts/visualize_predictions.py`. `tests/test_sirf_stir_minimal.py`
additionally checks that the SIRF/STIR environment itself is set up
correctly (requires SIRF to be installed; skip if only checking the
pure-Python logic above).

## 7) Acknowledgements

The SPECT simulation/reconstruction bridge (`src/spect/baseline/sirf_bridge.py`),
the 3D U-Net and Swin UNETR architectures (`src/spect/baseline/model.py`),
and the training loop structure (`train/train_unet.py`,
`train/train_swin_unetr.py`) are adapted from Wei Miao's dissertation
codebase, [ucapwmi/SPECT_codes](https://github.com/ucapwmi/SPECT_codes).
`general.samp.par`, the base XCAT parameter template used by
`scripts/generate_xcat_parfiles.py`, is also taken directly from that
repository; the randomisation ranges applied on top of it follow the
protocol described in his dissertation.

The EARL/NEMA phantom geometry is generated using Stathis Varzakis's
`phantomgen` package ([varzakis/phantomgen](https://github.com/varzakis/phantomgen)).

The ellipsoid-phantom denoising approach builds on Cate Gascoigne's MRes
thesis (UCL, 2024) on 2D CNN-based SPECT phantom denoising, extended here
to 3D architectures and a wider range of count levels.

I would like to thank my supervisors, Kris Thielemans, Efstathios Varzakis,
and Cate Gascoigne, for their guidance and support throughout this
project.
