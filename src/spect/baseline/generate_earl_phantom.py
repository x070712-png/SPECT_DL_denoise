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
 
 
def build_earl_dict(sphere_act_conc, background_act_conc):
    return {
        "mu_values": {
            "perspex_mu_value": 0.15,
            "fill_mu_value": 0.14,
            "lung_mu_value": 0.043,
        },
        "activity_concentration_background": background_act_conc,
        "include_lung_insert": False,
        "sphere_dict": {
            "ring_R": 57,
            "ring_z": -37,
            "spheres": {
                "diametre_mm": SPHERE_DIAMETERS_MM,
                "angle_loc": ANGLE_LOC,
                "act_conc_MBq_ml": [sphere_act_conc] * 6,
            },
        },
    }
 
 
def generate_activity_and_umap(sphere_act_conc, background_act_conc):
    earl_dict = build_earl_dict(sphere_act_conc, background_act_conc)
    act_vol, ctac_vol, masks = create_nema(
        matrix_size=VOLUME_SHAPE,
        voxel_size_mm=VOXEL_SIZE_MM,
        nema_dict=earl_dict,
        supersample=4,
    )
    return act_vol.astype(np.float32), ctac_vol.astype(np.float32), masks
 
 
def measure_clean_stats(act_vol, ctac_vol):
    """Forward-project (noise-free) + OSEM reconstruct at alpha=1.0, return
    (mean, max, max/mean) of the reconstructed image -- used to calibrate
    BOTH the overall scale (mean) and the intensity distribution SHAPE
    (max/mean ratio) against the ellipsoid/XCAT training data. No Poisson
    noise involved here."""
    templ_sino = load_template_sinogram()
    umap = make_custom_umap(templ_sino, ctac_vol)
    clean_sino, _ = acquire_data(act_vol, templ_sino, alpha=1.0, umap=umap)
    label_img = reconstruct_data(clean_sino, templ_sino, umap=umap)
    arr = label_img.as_array()
    mean = float(arr.mean())
    max_ = float(arr.max())
    ratio = max_ / mean if mean > 0 else float("inf")
    return mean, max_, ratio
 
 
def calibrate_sphere_for_mean(background_act_conc, target_mean, init_sphere_act_conc,
                                max_iters=3, tol=0.05):
    """INNER loop: for a FIXED background level, ratio-scale sphere activity
    until the whole-volume mean hits target_mean (same style as v1's
    single-parameter calibration)."""
    s = init_sphere_act_conc
    mean = max_ = ratio = None
    act_vol = ctac_vol = masks = None
    for i in range(max_iters):
        act_vol, ctac_vol, masks = generate_activity_and_umap(s, background_act_conc)
        mean, max_, ratio = measure_clean_stats(act_vol, ctac_vol)
        print(f"    [inner {i}] sphere_conc={s:.6g} (bg={background_act_conc:.6g}) -> "
              f"mean={mean:.4f} (target {target_mean}) max={max_:.4f} ratio={ratio:.2f}")
        if mean <= 0:
            raise RuntimeError("Reconstructed mean is <= 0 -- check activity/umap units.")
        r = target_mean / mean
        if abs(r - 1.0) < tol:
            break
        s *= r
    return s, act_vol, ctac_vol, masks, mean, max_, ratio
 
 
def calibrate_joint(target_mean, target_ratio, init_sphere_act_conc=2.0,
                      max_outer=6, ratio_tol=0.15, inner_max_iters=3, inner_tol=0.05):
    """OUTER loop: bisection-search background_act_conc to hit
    target_ratio, re-running the INNER mean-calibration at each trial
    background level (since sphere activity has to be re-tuned every time
    background changes, to keep the mean on target)."""
    bg_lo, bg_hi = 0.0, None
    bg = 0.0
    s = init_sphere_act_conc
    best = None
    for outer in range(max_outer):
        s, act_vol, ctac_vol, masks, mean, max_, ratio = calibrate_sphere_for_mean(
            bg, target_mean, s, max_iters=inner_max_iters, tol=inner_tol)
        print(f"  [outer {outer}] background_act_conc={bg:.6g} -> "
              f"mean={mean:.4f} (target {target_mean})  "
              f"max/mean={ratio:.2f} (target {target_ratio})")
        best = (bg, s, act_vol, ctac_vol, masks, mean, max_, ratio)
 
        rel_err = abs(ratio - target_ratio) / target_ratio
        if rel_err < ratio_tol:
            print(f"  [outer] converged: max/mean ratio within "
                  f"{ratio_tol*100:.0f}% of target after {outer+1} outer iteration(s).")
            break
 
        if ratio > target_ratio:
            # too concentrated (background too weak) -> raise background
            bg_lo = bg
            bg = (bg * 2) if bg > 0 else (s / 100.0)  # first nonzero guess: 1% of sphere conc
            if bg_hi is not None:
                bg = (bg_lo + bg_hi) / 2
        else:
            # background too strong (ratio undershot) -> lower background
            bg_hi = bg
            bg = (bg_lo + bg_hi) / 2 if bg_lo is not None else bg / 2
    else:
        print(f"  [outer] stopped after {max_outer} outer iterations without full "
              f"convergence (last rel_err={rel_err:.2%}) -- using last generated map anyway.")
    return best
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--target_mean", type=float, default=0.461,
                    help="target WHOLE-VOLUME mean of the clean reconstructed image at "
                         "alpha=1.0 -- measured from ellipsoid training data's CLEAN "
                         "labels (data/dataset/alpha_1p0/label_*.npy), NOT the per-VOI "
                         "'true_val_label' figure used by quantify_noisy_baseline.py, "
                         "and NOT the noisy input_*.npy (see docstring note)")
    p.add_argument("--target_max_mean_ratio", type=float, default=25.20,
                    help="target whole-volume max/mean ratio -- also measured from "
                         "ellipsoid CLEAN labels (not noisy input, which inflates max "
                         "via single-realisation Poisson spikes), controls background "
                         "activity")
    p.add_argument("--init_sphere_act_conc", type=float, default=2.0)
    p.add_argument("--max_outer", type=int, default=6)
    p.add_argument("--ratio_tol", type=float, default=0.15,
                    help="relative tolerance for max/mean ratio convergence -- default "
                         "15%% since the ellipsoid target itself has ~23%% relative "
                         "spread (10.5/45.16) across samples")
    args = p.parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
 
    bg, s, act_vol, ctac_vol, masks, mean, max_, ratio = calibrate_joint(
        args.target_mean, args.target_max_mean_ratio,
        init_sphere_act_conc=args.init_sphere_act_conc,
        max_outer=args.max_outer, ratio_tol=args.ratio_tol)
 
    np.save(os.path.join(args.out_dir, "activity.npy"), act_vol)
    np.save(os.path.join(args.out_dir, "att_map.npy"), ctac_vol)
 
    for i, d in enumerate(SPHERE_DIAMETERS_MM, start=1):
        key = f"sphere_{i}"
        if key not in masks:
            print(f"[warn] mask key '{key}' not found -- available: {list(masks.keys())}")
            continue
        mask = masks[key].astype(np.uint8)
        np.save(os.path.join(args.out_dir, f"EARL_sphere_{d}mm.npy"), mask)
        print(f"  saved EARL_sphere_{d}mm.npy from mask key '{key}' (n_voxels={mask.sum()})")
 
    print(f"\nFinal sphere_act_conc_MBq_ml = {s:.6g}")
    print(f"Final background_act_conc_MBq_ml = {bg:.6g}")
    print(f"Final reconstructed mean = {mean:.4f} (target {args.target_mean})")
    print(f"Final reconstructed max/mean ratio = {ratio:.2f} (target {args.target_max_mean_ratio})")
    print(f"Saved activity.npy, att_map.npy, and sphere masks to {args.out_dir}")
    print(f"\nNext step -- feed into the existing eval pipeline unchanged:")
    print(f"python3 src/spect/baseline/generate_eval_phantom_dataset.py \\")
    print(f"    --activity {args.out_dir}/activity.npy \\")
    print(f"    --att_map {args.out_dir}/att_map.npy \\")
    print(f"    --out_dir data/earl_dataset_v2 \\")
    print(f"    --seeds 42 43 44 45 46 47 48 49 50 51")
 
 
if __name__ == "__main__":
    main()
 