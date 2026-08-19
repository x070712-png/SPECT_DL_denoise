# src/spect/baseline/generate_earl_phantom.py
"""
Generates the EARL evaluation phantom's activity map, attenuation map, and
per-sphere VOI masks using Stathis Varzakis's phantomgen package
(https://github.com/varzakis/phantomgen).

WHAT'S ACTUALLY USED: the final EARL evaluation data comes from
--calibration_mode fixed (v4): sphere_act_conc=126.457 fixed across both
conditions, background_act_conc set explicitly to 0.0 (no background) or
12.6457 (10:1 sphere:background) -- a direct comparison of two physically
meaningful ratios, no calibration/solving involved.

sphere_act_conc=126.457 was obtained separately, by running this script's
other mode once (--calibration_mode joint, the default, calibrate_joint()
below), which jointly solves for sphere AND background activity so the
clean reconstructed image's whole-volume mean and max/mean ratio match
target statistics from the ellipsoid/XCAT training data (see targets
below). Only the solved sphere value was carried over into the fixed-mode
runs above -- the background value joint mode solves for (84.4954) was
NOT used in the final experiments; fixed mode's explicit ratios are a
simpler, more direct comparison instead.

Target statistics for joint mode, measured from the ellipsoid training
data's CLEAN labels (data/dataset/alpha_1p0/label_*.npy, 20-sample check):
  whole-volume mean       = 0.461 +/- 0.118
  whole-volume max/mean   = 25.20 +/- 4.18

JOINT CALIBRATION (calibrate_joint / solve_joint): sphere activity affects
both the reconstructed mean and max; background affects only the mean
(too dilute to create a new hottest voxel). Treating (mean, max) as
locally linear in (sphere_conc, background_conc), the script probes 3
points to estimate the local Jacobian, then solves the resulting 2x2
linear system directly for the (sphere, background) hitting both targets,
confirming (and re-linearising if needed) at the solved point. This
probe-and-solve approach replaced an earlier nested-bisection attempt
that oscillated and never converged.

"max" is always the max over sphere-mask voxels only, not whole-volume
np.max() -- the global max can sit on a reconstruction/attenuation edge
artifact rather than real sphere signal once background is non-trivial,
which would corrupt the calibration. measure_clean_stats() restricts to
the union of the 6 sphere masks and cross-checks against the global max
so a mismatch is visible rather than silently corrupting results.

Usage (fixed mode -- generates the final evaluation data):
    export PYTHONPATH=<repo_root>:$PYTHONPATH
    python3 src/spect/baseline/generate_earl_phantom.py \
        --out_dir data/earl_phantom_v3_bg0 \
        --calibration_mode fixed \
        --sphere_act_conc 126.457 \
        --background_act_conc 0.0

    python3 src/spect/baseline/generate_earl_phantom.py \
        --out_dir data/earl_phantom_v3_bg_ratio10 \
        --calibration_mode fixed \
        --sphere_act_conc 126.457 \
        --background_act_conc 12.6457

Usage (joint mode -- only to re-derive sphere_act_conc from scratch, not
part of the final evaluation pipeline):
    python3 src/spect/baseline/generate_earl_phantom.py \
        --out_dir data/earl_phantom_v2 \
        --target_mean 0.461 \
        --target_max_mean_ratio 25.20

Outputs (in --out_dir):
    activity.npy            -- calibrated activity map, (128,128,128)
    att_map.npy              -- attenuation map (cm^-1), (128,128,128)
    EARL_sphere_{d}mm.npy    -- one binary VOI mask per sphere
                                (d in 13,17,22,28,37,60)
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


def measure_clean_stats(sphere_act_conc, background_act_conc):
    """Forward-project (noise-free) + OSEM reconstruct at alpha=1.0, return
    (mean, max, act_vol, ctac_vol, masks) -- used to calibrate BOTH the
    overall scale (mean) and the intensity distribution SHAPE (max) against
    the ellipsoid/XCAT training data. No Poisson noise involved here.

    "max" here is NOT whole-volume np.max(), but restricted to voxels
    inside the union of the 6 sphere masks (the only physically
    meaningful definition of "peak sphere signal") -- the global max can
    sit on a reconstruction/attenuation edge artifact rather than real
    sphere signal once background is non-trivial. Cross-checks against
    the global max + its coordinates so a mismatch is visible in the log
    rather than silently corrupting the calibration."""
    act_vol, ctac_vol, masks = generate_activity_and_umap(sphere_act_conc, background_act_conc)
    templ_sino = load_template_sinogram()
    umap = make_custom_umap(templ_sino, ctac_vol)
    clean_sino, _ = acquire_data(act_vol, templ_sino, alpha=1.0, umap=umap)
    label_img = reconstruct_data(clean_sino, templ_sino, umap=umap)
    arr = label_img.as_array()
    mean = float(arr.mean())

    global_max = float(arr.max())
    global_argmax_coords = np.unravel_index(np.argmax(arr), arr.shape)

    sphere_keys = [f"sphere_{i}" for i in range(1, 7)]
    combined_mask = None
    for k in sphere_keys:
        if k in masks:
            m = masks[k].astype(bool)
            combined_mask = m if combined_mask is None else (combined_mask | m)
    if combined_mask is None or not combined_mask.any():
        raise RuntimeError(f"No sphere mask voxels found (keys checked: {sphere_keys}, "
                            f"available: {list(masks.keys())}) -- cannot compute sphere-"
                            f"restricted max.")

    sphere_vals = arr[combined_mask]
    max_ = float(sphere_vals.max())
    sphere_argmax_flat = np.argmax(np.where(combined_mask, arr, -np.inf))
    sphere_argmax_coords = np.unravel_index(sphere_argmax_flat, arr.shape)

    in_sphere = bool(combined_mask[global_argmax_coords])
    flag = "" if in_sphere else "  [WARN] global max is OUTSIDE all sphere masks -- likely an edge/reconstruction artifact, not sphere signal"
    print(f"    probe: sphere={sphere_act_conc:.6g} bg={background_act_conc:.6g} "
          f"-> mean={mean:.4f} sphere_max={max_:.4f} ratio={(max_/mean if mean>0 else float('inf')):.2f} "
          f"(global_max={global_max:.4f} @ {global_argmax_coords}, sphere_argmax @ {sphere_argmax_coords}){flag}")
    return mean, max_, act_vol, ctac_vol, masks


def calibrate_sphere_only(target_mean, init_sphere_act_conc=2.0, max_iters=4, tol=0.03):
    """Single-parameter ratio-scaling calibration (background fixed at 0)
    -- used here just to get a sensible baseline probe point s0 where
    mean(s0, 0) ~= target_mean before estimating the Jacobian."""
    s = init_sphere_act_conc
    mean = max_ = None
    for i in range(max_iters):
        mean, max_, act_vol, ctac_vol, masks = measure_clean_stats(s, 0.0)
        if mean <= 0:
            raise RuntimeError("Reconstructed mean is <= 0 -- check activity/umap units.")
        r = target_mean / mean
        if abs(r - 1.0) < tol:
            break
        s *= r
    return s, mean, max_


def solve_joint(target_mean, target_ratio, s0, mean0, max0, s1, mean_s1, max_s1,
                  b1, mean_b1, max_b1):
    """Given 3 probe points -- baseline (s0,0), sphere-perturbed (s1,0),
    background-perturbed (s0,b1) -- fit a local linear model and solve
    directly for the (sphere, background) that hits (target_mean,
    target_max) to first order."""
    target_max = target_mean * target_ratio

    ms = (mean_s1 - mean0) / (s1 - s0)   # d(mean)/d(sphere)
    xs = (max_s1 - max0) / (s1 - s0)     # d(max)/d(sphere)
    mb = (mean_b1 - mean0) / b1          # d(mean)/d(background)
    xb = (max_b1 - max0) / b1            # d(max)/d(background)

    A = np.array([[ms, mb], [xs, xb]])
    rhs = np.array([target_mean - mean0, target_max - max0])
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise RuntimeError(f"Jacobian is singular (det={det:.3e}) -- probes were not "
                            f"informative enough to solve for both parameters.")
    ds, b = np.linalg.solve(A, rhs)
    s = s0 + ds
    print(f"  Linear model: d(mean)/d(sphere)={ms:.3e}  d(max)/d(sphere)={xs:.4f}")
    print(f"                d(mean)/d(bg)={mb:.4f}      d(max)/d(bg)={xb:.4f}")
    print(f"  Solved: sphere={s:.6g}  background={b:.6g}  (target_max={target_max:.4f})")
    return max(s, 1e-6), max(b, 0.0)  # clip to physically valid (non-negative) range


def calibrate_joint(target_mean, target_ratio, init_sphere_act_conc=2.0,
                      sphere_probe_factor=1.5, bg_probe_frac_of_sphere=0.01,
                      max_refine=2, mean_tol=0.05, ratio_tol=0.15):
    print("Step 1/3: single-parameter baseline (background=0)...")
    s0, mean0, max0 = calibrate_sphere_only(target_mean, init_sphere_act_conc)

    for refine in range(max_refine + 1):
        print(f"\nStep 2/3 (refine {refine}): probing local Jacobian around "
              f"sphere={s0:.6g}, background={0 if refine == 0 else 'prev solved value'}...")
        s1 = s0 * sphere_probe_factor
        b1 = max(s0 * bg_probe_frac_of_sphere, 1e-3)

        mean_s1, max_s1, _, _, _ = measure_clean_stats(s1, 0.0)
        mean_b1, max_b1, _, _, _ = measure_clean_stats(s0, b1)

        print("Step 3/3: solving linear system for both targets...")
        s_sol, b_sol = solve_joint(target_mean, target_ratio,
                                     s0, mean0, max0, s1, mean_s1, max_s1,
                                     b1, mean_b1, max_b1)

        mean_final, max_final, act_vol, ctac_vol, masks = measure_clean_stats(s_sol, b_sol)
        ratio_final = max_final / mean_final if mean_final > 0 else float("inf")
        mean_rel_err = abs(mean_final - target_mean) / target_mean
        ratio_rel_err = abs(ratio_final - target_ratio) / target_ratio
        print(f"  Confirmation: sphere={s_sol:.6g} bg={b_sol:.6g} -> "
              f"mean={mean_final:.4f} (rel_err={mean_rel_err:.1%}, tol={mean_tol:.0%})  "
              f"ratio={ratio_final:.2f} (rel_err={ratio_rel_err:.1%}, tol={ratio_tol:.0%})")

        if mean_rel_err < mean_tol and ratio_rel_err < ratio_tol:
            print(f"  Converged after {refine + 1} refinement round(s).")
            return s_sol, b_sol, act_vol, ctac_vol, masks, mean_final, max_final, ratio_final

        # re-linearise around the new solved point for another round
        s0, mean0, max0 = s_sol, mean_final, max_final

    print(f"  [warn] did not fully converge after {max_refine + 1} refinement round(s) -- "
          f"using last solved point anyway (mean_rel_err={mean_rel_err:.1%}, "
          f"ratio_rel_err={ratio_rel_err:.1%}).")
    return s_sol, b_sol, act_vol, ctac_vol, masks, mean_final, max_final, ratio_final


def generate_fixed(sphere_act_conc, background_act_conc):
    """v4 fixed mode: no calibration/solving at all -- just generate +
    reconstruct at the EXPLICIT (sphere, background) values given, so the
    resulting mean/max/ratio are whatever they turn out to be (reported
    via measure_clean_stats' printout, not targeted). Reuses
    measure_clean_stats() purely for its forward-project + OSEM
    reconstruct + sphere-restricted-max logic -- the calibration loop
    machinery (calibrate_sphere_only / calibrate_joint / solve_joint)
    isn't involved here at all."""
    mean, max_, act_vol, ctac_vol, masks = measure_clean_stats(sphere_act_conc, background_act_conc)
    ratio = max_ / mean if mean > 0 else float("inf")
    return act_vol, ctac_vol, masks, mean, max_, ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--calibration_mode", type=str, default="joint", choices=["joint", "fixed"],
                    help="'joint' (default): solve for (sphere, background) to hit "
                         "--target_mean/--target_max_mean_ratio -- only needed to "
                         "re-derive sphere_act_conc from scratch. 'fixed' (v4, what "
                         "generates the final evaluation data): skip all solving, "
                         "generate directly at --sphere_act_conc/--background_act_conc "
                         "as given.")
    p.add_argument("--sphere_act_conc", type=float, default=126.457,
                    help="[fixed mode only] sphere activity concentration -- default is "
                         "the value already solved via joint mode")
    p.add_argument("--background_act_conc", type=float, default=0.0,
                    help="[fixed mode only] background activity concentration -- default "
                         "0.0 (no background). Try sphere_act_conc/10 for a 10:1 ratio.")
    p.add_argument("--target_mean", type=float, default=0.461,
                    help="target WHOLE-VOLUME mean of the clean reconstructed image at "
                         "alpha=1.0 -- measured from ellipsoid CLEAN labels "
                         "(data/dataset/alpha_1p0/label_*.npy)")
    p.add_argument("--target_max_mean_ratio", type=float, default=25.20,
                    help="target whole-volume max/mean ratio -- also from ellipsoid "
                         "CLEAN labels, controls background activity")
    p.add_argument("--init_sphere_act_conc", type=float, default=2.0)
    p.add_argument("--max_refine", type=int, default=2,
                    help="extra probe-and-solve rounds if the first solve doesn't hit "
                         "tolerance (the system is only locally linear, so large jumps "
                         "may need 1-2 re-linearisations)")
    p.add_argument("--mean_tol", type=float, default=0.05)
    p.add_argument("--ratio_tol", type=float, default=0.15,
                    help="relative tolerance for max/mean ratio -- default 15%% since "
                         "the ellipsoid target itself has ~17%% relative spread "
                         "(4.18/25.20) across samples")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.calibration_mode == "fixed":
        print(f"[fixed mode] generating directly at sphere={args.sphere_act_conc:.6g}, "
              f"background={args.background_act_conc:.6g} -- no calibration/solving, "
              f"whatever mean/max/ratio results is just reported below.")
        act_vol, ctac_vol, masks, mean, max_, ratio = generate_fixed(
            args.sphere_act_conc, args.background_act_conc)
        s, b = args.sphere_act_conc, args.background_act_conc
    else:
        s, b, act_vol, ctac_vol, masks, mean, max_, ratio = calibrate_joint(
            args.target_mean, args.target_max_mean_ratio,
            init_sphere_act_conc=args.init_sphere_act_conc,
            max_refine=args.max_refine, mean_tol=args.mean_tol, ratio_tol=args.ratio_tol)

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
    print(f"Final background_act_conc_MBq_ml = {b:.6g}")
    if args.calibration_mode == "fixed":
        print(f"Resulting reconstructed mean = {mean:.4f} (not targeted -- fixed mode)")
        print(f"Resulting reconstructed max/mean ratio = {ratio:.2f} (not targeted -- fixed mode)")
    else:
        print(f"Final reconstructed mean = {mean:.4f} (target {args.target_mean})")
        print(f"Final reconstructed max/mean ratio = {ratio:.2f} (target {args.target_max_mean_ratio})")
    print(f"Saved activity.npy, att_map.npy, and sphere masks to {args.out_dir}")
    print(f"\nNext step -- feed into the existing eval pipeline unchanged:")
    print(f"python3 src/spect/baseline/generate_eval_phantom_dataset.py \\")
    print(f"    --activity {args.out_dir}/activity.npy \\")
    print(f"    --att_map {args.out_dir}/att_map.npy \\")
    print(f"    --out_dir data/earl_dataset_v3_bg0  (pick a name that matches --out_dir above) \\")
    print(f"    --seeds 42 43 44 45 46 47 48 49 50 51")


if __name__ == "__main__":
    main()