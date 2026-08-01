# src/spect/baseline/generate_eval_phantom_dataset.py
"""
Forward-project + OSEM-reconstruct a SINGLE fixed evaluation phantom
(NEMA or EARL) across all count levels, using its own REAL attenuation
map -- NOT the uniform mu=0.12 used for the virtual ellipsoid training
data. Per Stathis (30 Jul email): "You should only use the uniform
attenuation map with your virtual ellipsoid phantoms" -- NEMA/EARL/
clinical should all use their own provided attenuation maps.

Unlike generate_xcat_dataset.py (500 DIFFERENT phantoms, one alpha each,
for fine-tuning), NEMA/EARL are each a SINGLE fixed physical phantom used
for EVALUATION -- so this script runs the same phantom through every
count level in COUNT_LEVELS (not just one), to characterise CNN
performance across the full noise range on a known-geometry object.

Voxel size confirmed 4.42mm isotropic for NEMA/EARL/clinical (Stathis, 30
Jul) -- matches PHANTOM_CONFIG exactly, so NO resampling needed, the
raw .npy arrays go straight into sirf_bridge.py as-is.

Sphere masks (NEMA_sphere_10mm.npy etc.) are NOT touched by this script --
they don't need forward-projecting, they're VOIs used later at
quantification time (same idea as quantify_noisy_baseline.py's VOI masks
for the ellipsoid phantoms, just already given as file-per-sphere instead
of computed from ellipsoid parameters). Point the quantification script
directly at the original nema/EARL_sphere_*mm.npy files, no copy needed.
"""

import argparse
import os

import numpy as np

from src.spect.baseline.config import COUNT_LEVELS
from src.spect.baseline.sirf_bridge import (
    load_template_sinogram,
    acquire_data,
    reconstruct_data,
    make_custom_umap,
)

VOLUME_SHAPE = (128, 128, 128)  # confirmed by Stathis: 4.42mm isotropic, no resampling needed

# same folder-name-safe encoding as the ellipsoid/XCAT pipeline
ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=str, required=True,
                    help="path to the phantom's activity map .npy (e.g. nema/NEMA_activity.npy)")
    p.add_argument("--att_map", type=str, required=True,
                    help="path to the phantom's REAL attenuation map .npy (e.g. nema/NEMA_att_map.npy) "
                         "-- used for BOTH forward projection and reconstruction, unlike the "
                         "ellipsoid/XCAT pipeline's uniform umap")
    p.add_argument("--out_dir", type=str, required=True,
                    help="output dir -- gets alpha_{alpha_str}/{input,label}.npy subfolders "
                         "(no phantom_idx needed, this is always a single fixed phantom)")
    p.add_argument("--alphas", type=float, nargs="+", default=COUNT_LEVELS,
                    help=f"count levels to run (default: all of {COUNT_LEVELS}, matching "
                         f"the ellipsoid/XCAT training count levels for comparability)")
    p.add_argument("--seed", type=int, default=42,
                    help="seed for the Poisson noise draw")
    return p.parse_args()


def load_array(path, expected_shape=VOLUME_SHAPE):
    arr = np.load(path).astype(np.float32)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{path}: shape={arr.shape}, expected {expected_shape}. "
            f"If this is clinical data (256x128x128), it must be cropped to 128^3 "
            f"FIRST (see the clinical cropping script) -- this script does not crop."
        )
    return arr


def main():
    args = parse_args()
    np.random.seed(args.seed)

    activity = load_array(args.activity)
    att_map_array = load_array(args.att_map)
    print(f"Loaded activity: {args.activity} (min={activity.min():.4f}, max={activity.max():.4f})")
    print(f"Loaded att_map:  {args.att_map} (min={att_map_array.min():.4f}, max={att_map_array.max():.4f})")

    templ_sino = load_template_sinogram()
    print("Template sinogram dimensions:", templ_sino.dimensions())

    # build the REAL attenuation ImageData ONCE -- reused across all alpha levels
    # (same physical phantom / same attenuation regardless of count level)
    umap = make_custom_umap(templ_sino, att_map_array)

    os.makedirs(args.out_dir, exist_ok=True)
    for alpha in args.alphas:
        alpha_str = ALPHA_STR.get(alpha, str(alpha).replace(".", "p"))
        out_subdir = os.path.join(args.out_dir, f"alpha_{alpha_str}")
        os.makedirs(out_subdir, exist_ok=True)
        label_path = os.path.join(out_subdir, "label.npy")
        input_path = os.path.join(out_subdir, "input.npy")

        if os.path.exists(label_path) and os.path.exists(input_path):
            print(f"[skip] alpha_{alpha_str}: already exists")
            continue

        print(f"[alpha_{alpha_str}] forward projecting (real attenuation)...")
        clean_sino, noisy_sino = acquire_data(activity, templ_sino, alpha=alpha, umap=umap)

        print(f"[alpha_{alpha_str}] reconstructing label (clean, real attenuation)...")
        label_img = reconstruct_data(clean_sino, templ_sino, umap=umap)

        print(f"[alpha_{alpha_str}] reconstructing input (noisy, real attenuation)...")
        input_img = reconstruct_data(noisy_sino, templ_sino, umap=umap)

        np.save(label_path, label_img.as_array().astype(np.float32))
        np.save(input_path, input_img.as_array().astype(np.float32))
        print(f"[alpha_{alpha_str}] done -> {input_path}, {label_path}")

    print(f"\nAll done. {args.out_dir}/alpha_*/{{input,label}}.npy ready. "
          f"For per-sphere RC quantification, point at the ORIGINAL sphere mask "
          f"files (e.g. nema/NEMA_sphere_*mm.npy) directly -- they weren't copied, "
          f"same VOIs apply to every alpha level.")


if __name__ == "__main__":
    main()