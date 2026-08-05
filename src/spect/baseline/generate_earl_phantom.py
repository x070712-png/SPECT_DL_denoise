# src/spect/baseline/generate_earl_phantom.py
"""
Generates the EARL evaluation phantom's activity map, attenuation map, and
per-sphere VOI masks using Stathis's phantomgen package
(https://github.com/varzakis/phantomgen), and CALIBRATES the overall
activity concentration (act_conc_MBq_ml) so that the CLEAN (noise-free)
reconstructed image at alpha=1.0 lands on the same scale as the
ellipsoid/XCAT training data.

Why calibration is needed: phantomgen's default EARL preset
(act_conc_MBq_ml=2.0 for all 6 spheres) produces an activity map whose
reconstructed values are ~10,000x smaller than the ellipsoid/XCAT training
scale (mean label ~= 3.1 vs ~0.00027 for the previous NEMA/EARL data
source). Since Poisson noise statistics depend on the
ABSOLUTE count level (not just relative image structure), feeding the
network data at the wrong scale means "alpha=1.0" here does not
correspond to the same noise regime as "alpha=1.0" during training --
the network fails even at the highest nominal count level. This script
fixes that at the SOURCE (before forward-projection), rather than trying
to compensate after the fact.

Calibration approach: activity -> forward-projection -> OSEM reconstruction
is approximately LINEAR in activity (ignoring noise, which is not present
in this calibration step -- we only reconstruct the CLEAN sinogram). So:
  1. Generate the phantom at a trial act_conc_MBq_ml.
  2. Forward-project (no noise) + reconstruct at alpha=1.0.
  3. Measure the reconstructed image's mean value.
  4. Rescale act_conc_MBq_ml by (target_mean / measured_mean) and repeat.
Converges in 2-3 iterations given the near-linear relationship.
"""

import argparse
import os

import numpy as np

from phantomgen import create_nema

from src.spect.baseline.sirf_bridge import (
    load_template_sinogram,
    acquire_data,
    reconstruct_data,
    make_custom_umap,
)

VOLUME_SHAPE = (128, 128, 128)
VOXEL_SIZE_MM = (4.42, 4.42, 4.42)  # matches CONFIG["pixel_size_mm"] in generate_ellipsoids.py

SPHERE_DIAMETERS_MM = [13, 17, 22, 28, 37, 60]  # order must match ANGLE_LOC below
ANGLE_LOC = [270, 150, 30, 90, 330, 210]


def build_earl_dict(act_conc):
    return {
        "mu_values": {
            "perspex_mu_value": 0.15,
            "fill_mu_value": 0.14,
            "lung_mu_value": 0.043,
        },
        "activity_concentration_background": 0.0,
        "include_lung_insert": False,
        "sphere_dict": {
            "ring_R": 57,
            "ring_z": -37,
            "spheres": {
                "diametre_mm": SPHERE_DIAMETERS_MM,
                "angle_loc": ANGLE_LOC,
                "act_conc_MBq_ml": [act_conc] * 6,
            },
        },
    }


def generate_activity_and_umap(act_conc):
    earl_dict = build_earl_dict(act_conc)
    act_vol, ctac_vol, masks = create_nema(
        matrix_size=VOLUME_SHAPE,
        voxel_size_mm=VOXEL_SIZE_MM,
        nema_dict=earl_dict,
        supersample=4,
    )
    return act_vol.astype(np.float32), ctac_vol.astype(np.float32), masks


def measure_clean_reconstructed_mean(act_vol, ctac_vol):
    """Forward-project (noise-free) + OSEM reconstruct at alpha=1.0, return
    the mean of the reconstructed image -- used ONLY to calibrate the
    overall activity/count scale against the training data's label scale.
    No Poisson noise is involved here (that's handled separately, later,
    by generate_eval_phantom_dataset.py once the activity map is fixed)."""
    templ_sino = load_template_sinogram()
    umap = make_custom_umap(templ_sino, ctac_vol)
    clean_sino, _ = acquire_data(act_vol, templ_sino, alpha=1.0, umap=umap)
    label_img = reconstruct_data(clean_sino, templ_sino, umap=umap)
    return float(label_img.as_array().mean())


def calibrate(target_mean, init_act_conc, max_iters=3, tol=0.05):
    act_conc = init_act_conc
    act_vol = ctac_vol = masks = None
    for i in range(max_iters):
        act_vol, ctac_vol, masks = generate_activity_and_umap(act_conc)
        measured = measure_clean_reconstructed_mean(act_vol, ctac_vol)
        print(f"[calibrate] iter {i}: act_conc_MBq_ml={act_conc:.6g} -> "
              f"reconstructed mean={measured:.6g} (target={target_mean})")
        if measured <= 0:
            raise RuntimeError("Reconstructed mean is <= 0 -- check umap/activity units "
                                "before continuing (something is wrong upstream, not just scale).")
        ratio = target_mean / measured
        if abs(ratio - 1.0) < tol:
            print(f"[calibrate] converged within {tol*100:.0f}% after {i+1} iteration(s).")
            return act_conc, act_vol, ctac_vol, masks
        act_conc *= ratio  # near-linear activity -> reconstructed-mean assumption
    print(f"[calibrate] stopped after {max_iters} iterations without full convergence "
          f"(last ratio={ratio:.3f}) -- using the last generated activity map anyway; "
          f"rerun with a higher --max_iters if you need tighter convergence.")
    return act_conc, act_vol, ctac_vol, masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--target_mean", type=float, default=3.143,
                    help="target mean of the CLEAN reconstructed image at alpha=1.0, to "
                         "match the ellipsoid training data scale (mean true_val_label "
                         "= 3.143 for ellipsoid test split alpha_1p0, 8/3 quantify run)")
    p.add_argument("--init_act_conc", type=float, default=2.0,
                    help="starting act_conc_MBq_ml (uniform across all 6 spheres) before "
                         "calibration -- 2.0 is phantomgen's EARL preset default")
    p.add_argument("--max_iters", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    act_conc, act_vol, ctac_vol, masks = calibrate(
        args.target_mean, args.init_act_conc, max_iters=args.max_iters)

    np.save(os.path.join(args.out_dir, "activity.npy"), act_vol)
    np.save(os.path.join(args.out_dir, "att_map.npy"), ctac_vol)

    for i, d in enumerate(SPHERE_DIAMETERS_MM, start=1):
        key = f"sphere_{i}"
        if key not in masks:
            print(f"[warn] mask key '{key}' not found in phantomgen output -- "
                  f"available keys: {list(masks.keys())}")
            continue
        mask = masks[key].astype(np.uint8)
        np.save(os.path.join(args.out_dir, f"EARL_sphere_{d}mm.npy"), mask)
        print(f"  saved EARL_sphere_{d}mm.npy from mask key '{key}' "
              f"(n_voxels={mask.sum()})")

    print(f"\nFinal act_conc_MBq_ml = {act_conc:.6g}")
    print(f"Saved activity.npy, att_map.npy, and sphere masks to {args.out_dir}")
    print(f"\nNext step -- feed into the existing eval pipeline unchanged:")
    print(f"python3 src/spect/baseline/generate_eval_phantom_dataset.py \\")
    print(f"    --activity {args.out_dir}/activity.npy \\")
    print(f"    --att_map {args.out_dir}/att_map.npy \\")
    print(f"    --out_dir data/earl_dataset \\")
    print(f"    --seeds 42 43 44 45 46 47 48 49 50 51")


if __name__ == "__main__":
    main()