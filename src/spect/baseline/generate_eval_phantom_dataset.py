# src/spect/baseline/generate_eval_phantom_dataset.py
"""
Forward-project + OSEM-reconstruct a SINGLE fixed evaluation phantom
(NEMA or EARL) across all count levels, using its own REAL attenuation
map -- NOT the uniform mu=0.12 used for the virtual ellipsoid training
data. NEMA/EARL/clinical phantoms should use their own provided
attenuation maps rather than the uniform one, since they are real
physical objects rather than the synthetic ellipsoid geometry the
uniform map was chosen for.

Unlike generate_xcat_dataset.py (500 DIFFERENT phantoms, one alpha each,
for fine-tuning), NEMA/EARL are each a SINGLE fixed physical phantom used
for EVALUATION -- so this script runs the same phantom through every
count level in COUNT_LEVELS (not just one), to characterise CNN
performance across the full noise range on a known-geometry object.

Voxel size is 4.42mm isotropic for NEMA/EARL/clinical, matching
PHANTOM_CONFIG exactly, so NO resampling is needed -- the raw .npy
arrays go straight into sirf_bridge.py as-is.

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

VOLUME_SHAPE = (128, 128, 128)  # 4.42mm isotropic, no resampling needed

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
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one or more seeds for independent Poisson noise draws -- "
                         "e.g. --seeds 42 43 44 45 46 for 5 independent noise "
                         "realizations per alpha (RECOMMENDED for NEMA/EARL RC "
                         "stability, since it's a single fixed phantom, not 10+ "
                         "different phantoms like the ellipsoid/XCAT test split). "
                         "Default is a single seed (42) for backward compatibility.")
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

        # ---- clean sinogram / label: deterministic, no seed dependence,
        # compute ONCE per alpha and reuse across every seed below ----
        if os.path.exists(label_path):
            print(f"[skip] alpha_{alpha_str}: label.npy already exists")
            need_clean_sino = False
        else:
            need_clean_sino = True
 
        clean_sino = None  # lazily computed below only if needed
 
        for seed in args.seeds:
            input_path = os.path.join(out_subdir, f"input_seed{seed}.npy")
            if os.path.exists(input_path) and not need_clean_sino:
                print(f"[skip] alpha_{alpha_str} seed={seed}: already exists")
                continue
 
            if clean_sino is None:
                print(f"[alpha_{alpha_str}] forward projecting clean sinogram "
                      f"(real attenuation, seed-independent)...")
                # alpha/umap fully determine the clean sinogram -- the noise draw
                # (seeded via np.random.seed below) only affects the noisy branch
                clean_sino, _ = acquire_data(activity, templ_sino, alpha=alpha, umap=umap)
 
                if need_clean_sino:
                    print(f"[alpha_{alpha_str}] reconstructing label (clean, real attenuation)...")
                    label_img = reconstruct_data(clean_sino, templ_sino, umap=umap)
                    np.save(label_path, label_img.as_array().astype(np.float32))
                    need_clean_sino = False
 
            if os.path.exists(input_path):
                continue
 
            # ---- noisy branch: reseed per (alpha, seed) pair for a fresh,
            # reproducible, INDEPENDENT Poisson draw ----
            print(f"[alpha_{alpha_str}] seed={seed}: drawing noisy sinogram...")
            np.random.seed(seed)
            scaled = clean_sino.as_array() * alpha
            noisy_array = np.random.poisson(scaled).astype("float32")
            noisy_sino = clean_sino.clone()
            noisy_sino.fill(noisy_array)
 
            print(f"[alpha_{alpha_str}] seed={seed}: reconstructing noisy input "
                  f"(real attenuation)...")
            input_img = reconstruct_data(noisy_sino, templ_sino, umap=umap)
            np.save(input_path, input_img.as_array().astype(np.float32))
            print(f"[alpha_{alpha_str}] seed={seed}: done -> {input_path}")
 
    print(f"\nAll done. {args.out_dir}/alpha_*/label.npy (shared) + "
          f"input_seed{{seed}}.npy (one per --seeds entry) ready. "
          f"For per-sphere RC quantification, point at the ORIGINAL sphere mask "
          f"files (e.g. nema/NEMA_sphere_*mm.npy) directly -- they weren't copied, "
          f"same VOIs apply to every alpha level. Use quantify_nema_earl.py's "
          f"--seeds option to average RC across the realizations generated here.")


if __name__ == "__main__":
    main()