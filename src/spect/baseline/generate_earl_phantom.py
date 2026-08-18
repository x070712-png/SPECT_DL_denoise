# src/spect/baseline/generate_earl_phantom.py
"""
Generates the EARL evaluation phantom's activity map, attenuation map, and
per-sphere VOI masks using Stathis's phantomgen package
(https://github.com/varzakis/phantomgen), and JOINTLY CALIBRATES TWO
parameters -- sphere activity concentration AND background activity
concentration -- so that the CLEAN (noise-free) reconstructed image at
alpha=1.0 matches the ellipsoid/XCAT training data on TWO statistics at
once:
  1. whole-volume mean          (target ~0.461, see NOTE below)
  2. whole-volume max/mean ratio (target ~25.2, see NOTE below)

BACKGROUND (why a second parameter was needed, 8/7 finding): the first
version of this script only calibrated sphere activity, matching the
reconstructed image's mean to a target value -- but because the CNN's
input normalisation is `inp / inp.mean()`, matching only the overall mean
is INSUFFICIENT: multiplying the whole activity map (spheres AND
background) by a constant leaves the normalised image completely
unchanged, since the scale cancels out exactly in `inp / inp.mean()`.
What actually matters for whether the network has seen anything like this
image before is the SHAPE of the intensity distribution, not its overall
scale. With EARL's background hard-set to 0 (phantomgen's default,
possibly a simplification of the true EARL/NEMA IQ protocol, which
typically has a non-zero background at some sphere:background ratio --
worth confirming with Stathis), essentially all signal concentrates into
6 small spheres, giving a whole-volume max/mean ratio of ~1170 -- vs ~25
for the ellipsoid/XCAT training data. This ~46x shape mismatch is what
caused the CNN (both U-Net and Swin, XCAT-finetune label-alpha
checkpoints) to systematically under-recover EARL sphere activity by a
size-dependent ~5-10x, REGARDLESS of alpha -- i.e. NOT a noise/count-level
problem (which label x alpha already fixed), but a distinct
geometric/distributional domain gap.

NOTE on target statistics: measured directly from the ellipsoid training
data's CLEAN labels (data/dataset/alpha_1p0/label_*.npy, 20-sample check,
8/7), NOT the noisy inputs (an earlier check used noisy input_*.npy,
giving max/mean=45.16 -- inflated relative to the clean value because max
is an extreme-value statistic that a single Poisson realisation can push
up substantially; label_*.npy is consistent with this script's own
noise-free calibration reconstructions):
  whole-volume mean       = 0.461 +/- 0.118
  whole-volume max/mean   = 25.20 +/- 4.18

CALIBRATION STRATEGY (v3, 8/7 -- replaces the nested-bisection v2
approach, which got stuck oscillating and never converged): PROBE-AND-
SOLVE. Sphere activity and background activity both affect the
reconstructed mean, but only sphere activity meaningfully affects the
max (background is far too dilute to create a new hottest voxel) --
confirmed empirically from the v2 run's own log (background changing
60x barely moved max: 531.25 -> 531.61). This means (mean, max) is
approximately a LINEAR function of (sphere_conc, background_conc) over a
reasonably local range, so:
  1. Run 3 probe reconstructions to estimate the local Jacobian:
     - (s0, 0)      -- baseline, s0 chosen s.t. mean(s0,0) ~= target_mean
       (i.e. reuse the single-parameter v1/v2-style calibration first)
     - (s1, 0)       -- perturb sphere only, to get d(mean)/d(sphere),
       d(max)/d(sphere)
     - (s0, b1)      -- perturb background only, to get d(mean)/d(bg),
       d(max)/d(bg)
  2. Solve the resulting 2x2 linear system directly for the (sphere, bg)
     that hits (target_mean, target_mean*target_ratio) exactly (to first
     order).
  3. Run ONE confirmation reconstruction at the solved point; if still
     outside tolerance, re-linearise around it and repeat (usually
     converges in 1 extra step since the system is close to linear).
This needs ~4-6 reconstructions total, vs up to 18 for the nested
bisection, AND actually converges (the nested approach's outer/inner
loops made contradictory assumptions about independence that broke down
once background started dominating the mean -- see project log, 8/7).

"max" DEFINITION FIX (8/7, caught during review of the v2 log): "max" is
now the max over sphere-mask voxels only, NOT whole-volume np.max(). The
v2 log showed max staying frozen at an identical value across a ~3x
sphere_conc change, which is more consistent with the global argmax
sitting on a reconstruction/attenuation edge artifact than on real sphere
signal once background is non-trivial. measure_clean_stats() now prints
both the global max (with coordinates) and the sphere-restricted max, and
flags explicitly if the global argmax falls outside every sphere mask.
Caveat: the target ratio (25.20) was itself measured from the ellipsoid
data's GLOBAL max, not a VOI-restricted max -- for ellipsoids this is
very likely equivalent in practice (the ellipsoid VOIs are the brightest
regions by construction, background is 0.1-0.5 vs VOI 1.0-5.0), but
hasn't been explicitly re-verified; worth a quick sanity check later if
time allows.

FIXED MODE (v4, 8/10 -- Stathis's finding at the T3 meeting10 8/10
meeting): the v2/v3 joint calibration above matches the ellipsoid
training domain's intensity SHAPE (mean + max/mean ratio) by solving for
a background activity of 84.4954 -- but Stathis pointed out this makes
sphere:background only ~1.5:1, whereas a real EARL/NEMA IQ phantom
protocol normally has NO background at all (spheres only) or, if
non-zero, something like a 10:1 sphere:background ratio -- nothing close
to 1.5:1. His hypothesis: this unrealistically strong background is what
is causing the CNN's ~systematic, alpha-independent overestimation seen
in the EARL v2 CNN-output RC numbers (quant_earl_v2_{unet,swin}_output),
NOT a general CNN bias as originally written into the workbook notes.

Fixed mode tests this directly: sphere_act_conc is kept at the ALREADY
v2-calibrated value (126.457 by default -- Stathis's point was about
background, not sphere activity), and background_act_conc is set
EXPLICITLY (no solving) -- start with 0.0 (a "true" EARL phantom, no
background at all), then optionally try intermediate ratios (e.g.
sphere/10 = 12.6, for a 10:1 ratio) if time allows, per Stathis's
suggested progression. No mean/ratio target-matching is involved here;
whatever mean/max/ratio the reconstruction ends up with is simply
reported (via measure_clean_stats' existing printout) for the record.

Usage (v2/v3 joint calibration, unchanged, DEFAULT if --calibration_mode
not given):
    export PYTHONPATH=<repo_root>:$PYTHONPATH
    python3 src/spect/baseline/generate_earl_phantom.py \
        --out_dir data/earl_phantom_v2 \
        --target_mean 0.461 \
        --target_max_mean_ratio 25.20

Usage (v4 fixed mode -- background=0 "true EARL" test):
    python3 src/spect/baseline/generate_earl_phantom.py \
        --out_dir data/earl_phantom_v3_bg0 \
        --calibration_mode fixed \
        --sphere_act_conc 126.457 \
        --background_act_conc 0.0

Usage (v4 fixed mode -- 10:1 sphere:background ratio, if time allows):
    python3 src/spect/baseline/generate_earl_phantom.py \
        --out_dir data/earl_phantom_v3_bg_ratio10 \
        --calibration_mode fixed \
        --sphere_act_conc 126.457 \
        --background_act_conc 12.6457

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

    IMPORTANT (8/7, bug caught by user review of the v2 log): "max" here is
    NOT whole-volume np.max(). An earlier version used the global max, but
    the v2 nested-bisection log showed a case (outer3: sphere_conc dropped
    ~3x, from 184.626 to 65.1949) where the reported max stayed IDENTICAL
    to 4 decimal places (21.9817) -- far too exact to be a real "barely
    moved" case, and suspicious enough to suggest the global argmax voxel
    wasn't inside a sphere at all (most likely a reconstruction/attenuation
    -correction edge artifact at the phantom boundary, which doesn't scale
    with sphere activity, and can end up hotter than a shrunk-down sphere
    peak once background gets large). If that's what's happening, the
    global max is not actually tracking sphere signal, which would corrupt
    both the old bisection AND this script's own Jacobian probes.
    Fix: restrict "max" to voxels inside the union of the 6 sphere masks
    (the only physically meaningful definition of "peak sphere signal"),
    and cross-check against the global max + its coordinates so a mismatch
    is visible in the log rather than silently corrupting the calibration."""
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
    -- same style as v1, used here just to get a sensible baseline probe
    point s0 where mean(s0, 0) ~= target_mean before estimating the
    Jacobian."""
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
    """v4 fixed mode (8/10, Stathis's finding): no calibration/solving at
    all -- just generate + reconstruct at the EXPLICIT (sphere, background)
    values given, so the resulting mean/max/ratio are whatever they turn
    out to be (reported via measure_clean_stats' printout, not targeted).
    Reuses measure_clean_stats() purely for its forward-project + OSEM
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
                    help="'joint' (default, unchanged v2/v3 behaviour): solve for "
                         "(sphere, background) to hit --target_mean/--target_max_mean_ratio. "
                         "'fixed' (v4, 8/10 Stathis finding): skip all solving, generate "
                         "directly at --sphere_act_conc/--background_act_conc as given -- "
                         "use this to test whether a more realistic (near-zero) background "
                         "removes the CNN-output overestimation seen with the joint-"
                         "calibrated background of 84.4954.")
    p.add_argument("--sphere_act_conc", type=float, default=126.457,
                    help="[fixed mode only] sphere activity concentration -- default is "
                         "the already-solved v2 joint-calibration value, since Stathis's "
                         "point was specifically about background, not sphere activity")
    p.add_argument("--background_act_conc", type=float, default=0.0,
                    help="[fixed mode only] background activity concentration -- default "
                         "0.0 (a 'true' EARL phantom per Stathis, no background at all). "
                         "Try e.g. sphere_act_conc/10 for a 10:1 sphere:background ratio "
                         "as a follow-up test if time allows.")
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