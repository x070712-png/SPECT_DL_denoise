# scripts/visualize_predictions.py
"""
Qualitative comparison grid across ALL 5 count levels, for one checkpoint
at a time -- built to support the label-alpha vs old-method / U-Net vs
Swin comparison need:

  1. Whole-image difference maps (no VOI-mask restriction) -- diff panels
     are computed on the whole central slice, never masked, so background
     bias is visible.
  2. Shared colour scales across EVERY panel/row/figure (not recomputed
     per-row, and not even per-checkpoint) -- computed once in a first
     pass over ALL data before any plotting, so brightness and error
     magnitude are directly comparable across samples, count levels, AND
     checkpoints. Pre-CNN (noisy-label) and post-CNN (output-label) diff
     panels use TWO SEPARATE shared scales rather than one: pre-CNN error
     is an order of magnitude larger by construction, so one shared scale
     across both would wash the post-CNN panel out to near-white
     regardless of real differences between checkpoints. Each diff type
     still shares ONE scale across all checkpoints/alphas, so old-method
     vs label-alpha and U-Net vs Swin remain directly comparable within
     that diff type.
  3. One representative test-split phantom per alpha level (5 rows), not
     just alpha_1p0 -- ellipsoid phantoms only exist at ONE alpha each
     (500 phantoms / 5 fixed groups), so "more count levels" means picking
     a different representative phantom per alpha, not the same phantom
     repeated at different alphas (that's only possible for NEMA/EARL).
  4. A pre-CNN baseline difference panel (noisy input - label) alongside
     the post-CNN one (model output - label), so the reader sees at a
     glance how much error the network actually removed.

Reads ALREADY-DUMPED denoised outputs (written by run_inference_dump.py,
the same step quantify_noisy_baseline.py's RC numbers were computed from)
instead of loading the model and running inference itself -- guarantees
the qualitative figure and the quantitative RC table agree, and this
script has no torch/model dependency (safe on the login node, no GPU
needed).

VOI-mask-overlay panel from the older version is DROPPED here -- that was
a separate methodology check (on how the mask is built/reused), not part
of this whole-image background-bias comparison; keep it as a separate,
one-off figure if still needed.

Run ONCE PER CHECKPOINT (see CHECKPOINTS_BY_DATASET config below -- edit
paths if yours differ), or all four in one go:

    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/visualize_predictions.py --checkpoint_key unet_old
    python3 scripts/visualize_predictions.py --checkpoint_key unet_label_alpha
    python3 scripts/visualize_predictions.py --checkpoint_key swin_old
    python3 scripts/visualize_predictions.py --checkpoint_key swin_label_alpha
    python3 scripts/visualize_predictions.py --checkpoint_key all   # 4 PNGs, one global scale

Also supports the XCAT dataset/checkpoints (same 5-column layout, same
old-method vs label-alpha comparison, just a different --dataset and
--data_dir) via --dataset xcat:

    python3 scripts/visualize_predictions.py --dataset xcat --checkpoint_key all
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spect.baseline.dataset import build_split

ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}
ALPHAS_ORDERED = [1.0, 0.5, 0.25, 0.125, 0.05]

# ---- EDIT THESE if your denoised-dump directories are named differently ----
# each entry: denoised .npy files live at {denoised_dir}/alpha_*/denoised_NNNN.npy
# NOTE: unet_old was previously dumped with --split val, not test -- if the
# test-split denoised_NNNN.npy files for that checkpoint don't exist yet,
# rerun run_inference_dump.py with --split test first.
#
# alpha_correction: label-alpha checkpoints were trained to predict
# label*alpha (see dataset.py's scale_label_by_alpha), so their RAW saved
# output is approximately true_scale*alpha, not true_scale. Comparing that
# raw output directly against the un-scaled label makes the diff panels
# look like the network gets WORSE as alpha shrinks, when that's just the
# known label-scaling artifact, not a real regression (same trap already
# handled in quantify_noisy_baseline.py's RC/alpha column). Set True here
# for label-alpha checkpoints so this script divides by alpha right after
# loading, before computing anything else -- old-method checkpoints are
# untouched (already alpha-independent by design).
#
# NOISY-INPUT alpha-division (applies to EVERY checkpoint below, NOT
# gated by alpha_correction): the raw noisy reconstruction is also
# naturally ~alpha x dimmer in absolute units at low alpha -- the noisy
# sinogram is Poisson-thinned by alpha BEFORE reconstruction (see
# sirf_bridge.py's acquire_data()), so fewer total counts go into OSEM.
# The label is always full-scale, so comparing it against a raw noisy
# input that's ~alpha x dimmer would balloon the pre-CNN diff panel at
# low alpha for a reason unrelated to noise. Dividing the noisy input by
# alpha for display removes this, applied uniformly regardless of
# alpha_correction.
#
# Keyed by --dataset. Both sub-dicts use the SAME four checkpoint_key
# names (unet_old / unet_label_alpha / swin_old / swin_label_alpha) so
# --checkpoint_key works identically regardless of --dataset -- only the
# label text and denoised_dir paths differ. XCAT paths confirmed against
# scripts/hpc/inference/submit_inference_dump_{unet,swin}_xcat{,_labelalpha}.sh.
CHECKPOINTS_BY_DATASET = {
    "ellipsoid": {
        "unet_old":          {"label": "U-Net (old method)",        "denoised_dir": "logs/denoised/3d_unet",                     "alpha_correction": False},
        "unet_label_alpha":  {"label": "U-Net (label x alpha)",      "denoised_dir": "logs/denoised/3d_unet_label_alpha",         "alpha_correction": True},
        "swin_old":          {"label": "Swin UNETR (old method)",    "denoised_dir": "logs/denoised/swin_unetr",                  "alpha_correction": False},
        "swin_label_alpha":  {"label": "Swin UNETR (label x alpha)", "denoised_dir": "logs/denoised/swin_unetr_label_alpha",      "alpha_correction": True},
    },
    "xcat": {
        "unet_old":          {"label": "U-Net (XCAT finetune, old method)",        "denoised_dir": "logs/denoised/3d_unet_xcat_finetune",                    "alpha_correction": False},
        "unet_label_alpha":  {"label": "U-Net (XCAT finetune, label x alpha)",      "denoised_dir": "logs/denoised/3d_unet_xcat_finetune_label_alpha",        "alpha_correction": True},
        "swin_old":          {"label": "Swin UNETR (XCAT finetune, old method)",    "denoised_dir": "logs/denoised/swin_unetr_xcat_finetune",                 "alpha_correction": False},
        "swin_label_alpha":  {"label": "Swin UNETR (XCAT finetune, label x alpha)", "denoised_dir": "logs/denoised/swin_unetr_xcat_finetune_label_alpha",     "alpha_correction": True},
    },
}
DATA_DIR_BY_DATASET = {"ellipsoid": "data/dataset", "xcat": "data/xcat_dataset"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ellipsoid", choices=["ellipsoid", "xcat"],
                    help="which CHECKPOINTS_BY_DATASET / default --data_dir to use")
    p.add_argument("--data_dir", type=str, default=None,
                    help="root dir with alpha_*/{input,label}_NNNN.npy (noisy input + GT label) "
                         "-- defaults to DATA_DIR_BY_DATASET[--dataset] if not given")
    p.add_argument("--checkpoint_key", type=str, default="all",
                    choices=list(CHECKPOINTS_BY_DATASET["ellipsoid"].keys()) + ["all"])
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                    help="ignored if --fixed_phantom is given")
    p.add_argument("--fixed_phantom", type=int, default=None,
                    help="if given, show this SAME phantom index at all 5 alphas instead "
                         "of pick_representative_phantoms()'s one-different-phantom-per-"
                         "alpha choice -- matches the fixed-10-phantom x 5-alpha "
                         "evaluation. Use an index in 90-99 (the fixed-10 set Table "
                         "4.2/4.4 were regenerated on). MUST be paired with "
                         "--fixed10_dirs.")
    p.add_argument("--fixed10_dirs", action="store_true",
                    help="redirect every checkpoint's denoised_dir to the "
                         "'<denoised_dir>_fixed10' variant (e.g. logs/denoised/3d_unet -> "
                         "logs/denoised/3d_unet_fixed10) produced by run_inference_dump.py "
                         "--phantom_indices 90,...,99 -- required for --fixed_phantom to "
                         "find any denoised output beyond alpha_1p0 (only alpha_1p0's "
                         "plain --split test set happens to also cover 90-99)")
    p.add_argument("--out_dir", type=str, default="logs/qualitative")
    p.add_argument("--slice_axis", type=int, default=0, help="0=axial, 1=coronal, 2=sagittal")
    p.add_argument("--vmax_headroom", type=float, default=1.2,
                    help="multiply the global label max by this factor for the shared "
                         "intensity colour scale -- ignored if --intensity_vmax_override "
                         "is given")
    p.add_argument("--intensity_vmax_override", type=float, default=None,
                    help="if given, use this directly as the shared intensity vmax "
                         "instead of computing it from data. Useful when the "
                         "auto-computed max is set by one unusually bright "
                         "representative phantom, making everything else look too "
                         "dark/near-black under the shared scale.")
    p.add_argument("--diff_pre_vmax_override", type=float, default=None,
                    help="if given, use this directly as the shared pre-CNN diff "
                         "+/-scale instead of the auto-computed 99th-percentile max. "
                         "NOTE: pixels beyond this get clipped/saturated -- shrinking "
                         "it trades away the worst checkpoint/alpha's true magnitude "
                         "for more visible mid-range detail elsewhere.")
    p.add_argument("--diff_post_vmax_override", type=float, default=None,
                    help="same as --diff_pre_vmax_override but for the post-CNN "
                         "(output - label) diff panel.")
    return p.parse_args()


def central_slice(vol, axis):
    idx = vol.shape[axis] // 2
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def pick_representative_phantoms(split):
    """
    One (phantom_idx, alpha_str) per alpha level, from the given split
    -- first phantom encountered in each alpha group, consistent choice
    across all checkpoints since it only depends on data_dir's split, not
    on which checkpoint is being visualised.
    Returns {alpha_str: phantom_idx}.
    """
    pairs = build_split(split)
    picked = {}
    for phantom_idx, alpha_str in pairs:
        if alpha_str not in picked:
            picked[alpha_str] = phantom_idx
    return picked  # {alpha_str: phantom_idx}


def pick_fixed_phantom(phantom_idx):
    """SAME phantom_idx at all 5 alphas -- matches the fixed-10-phantom x
    5-alpha evaluation design. Only valid together with --fixed10_dirs
    (denoised_dir must point at the *_fixed10 directories produced by
    run_inference_dump.py --phantom_indices, since the plain denoised_dir
    only ever contains ONE alpha's worth of test-split output for
    phantom 90-99, not all 5)."""
    return {ALPHA_STR[a]: phantom_idx for a in ALPHAS_ORDERED}


def load_triplet(data_dir, denoised_dir, phantom_idx, alpha_str):
    """Load (noisy input, GT label, model output) for one (phantom, alpha),
    all still in raw count-domain units (as saved on disk -- no extra
    normalisation needed here, run_inference_dump.py already restored
    count-domain units before saving denoised_NNNN.npy)."""
    inp_path = os.path.join(data_dir, f"alpha_{alpha_str}", f"input_{phantom_idx:04d}.npy")
    lbl_path = os.path.join(data_dir, f"alpha_{alpha_str}", f"label_{phantom_idx:04d}.npy")
    den_path = os.path.join(denoised_dir, f"alpha_{alpha_str}", f"denoised_{phantom_idx:04d}.npy")

    missing = [p for p in (inp_path, lbl_path, den_path) if not os.path.exists(p)]
    if missing:
        return None, missing

    inp = np.load(inp_path).astype(np.float32)
    lbl = np.load(lbl_path).astype(np.float32)
    den = np.load(den_path).astype(np.float32)
    return (inp, lbl, den), []


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    CHECKPOINTS = CHECKPOINTS_BY_DATASET[args.dataset]
    if args.data_dir is None:
        args.data_dir = DATA_DIR_BY_DATASET[args.dataset]
    print(f"Dataset = {args.dataset}, data_dir = {args.data_dir}")

    keys = list(CHECKPOINTS.keys()) if args.checkpoint_key == "all" else [args.checkpoint_key]

    if args.fixed_phantom is not None:
        representative = pick_fixed_phantom(args.fixed_phantom)
        print(f"[FIXED-PHANTOM MODE] Using phantom {args.fixed_phantom:04d} at all 5 alphas "
              f"(--split={args.split} ignored)")
        if not args.fixed10_dirs:
            print("[warn] --fixed_phantom given without --fixed10_dirs -- denoised lookups "
                  "for alphas other than 1p0 will likely 404 (see --fixed10_dirs help)")
    else:
        representative = pick_representative_phantoms(args.split)
        print(f"Representative phantoms ({args.split} split): {representative}")

    # ------------------------------------------------------------------
    # Pass 1: load everything for every checkpoint x alpha up front, so a
    # SINGLE global colour scale (intensity + diff) can be computed across
    # the whole deliverable before any plotting happens (see docstring
    # item 2 for why).
    # ------------------------------------------------------------------
    loaded = {}   # {checkpoint_key: {alpha_str: (phantom_idx, inp, lbl, den) or None}}
    all_intensity_vals = []
    all_diff_pre_vals = []   # noisy input - label (pre-CNN), own shared scale
    all_diff_post_vals = []  # model output - label (post-CNN), own shared scale

    for key in keys:
        denoised_dir = CHECKPOINTS[key]["denoised_dir"]
        if args.fixed10_dirs:
            denoised_dir = denoised_dir + "_fixed10"
        loaded[key] = {}
        for alpha in ALPHAS_ORDERED:
            alpha_str = ALPHA_STR[alpha]
            phantom_idx = representative.get(alpha_str)
            if phantom_idx is None:
                print(f"[warn] no {args.split}-split phantom found for alpha_{alpha_str}, skipping")
                loaded[key][alpha_str] = None
                continue

            data, missing = load_triplet(args.data_dir, denoised_dir, phantom_idx, alpha_str)
            if data is None:
                print(f"[skip] {key} alpha_{alpha_str} phantom {phantom_idx:04d}: missing {missing}")
                loaded[key][alpha_str] = None
                continue

            inp, lbl, den = data
            # see CHECKPOINTS comment above for why both of these divisions
            # happen
            inp = inp / alpha
            if CHECKPOINTS[key]["alpha_correction"]:
                den = den / alpha
            loaded[key][alpha_str] = (phantom_idx, inp, lbl, den)

            inp_s = central_slice(inp, args.slice_axis)
            lbl_s = central_slice(lbl, args.slice_axis)
            den_s = central_slice(den, args.slice_axis)
            # Intensity vmax comes from label + model output ONLY, not the
            # noisy input -- at low alpha the alpha-divided noisy input's
            # Poisson noise is amplified across a wide swath of pixels, so
            # even its 99th percentile stays inflated well above the
            # anatomical range visible in label/output. All three
            # quantities below use the 99th percentile rather than the
            # true max for the same reason: isolated noise-spike pixels
            # would otherwise pin the shared scale and crush every other
            # row to near-invisible. (value, source) pairs are kept only
            # so the diagnostic print below can name which
            # checkpoint/alpha/panel set the scale.
            all_intensity_vals.append((np.percentile(lbl_s, 99), f"{key} alpha={alpha} label"))
            all_intensity_vals.append((np.percentile(den_s, 99), f"{key} alpha={alpha} model_output"))
            all_diff_post_vals.append(np.percentile(np.abs(den_s - lbl_s), 99))
            all_diff_pre_vals.append(np.percentile(np.abs(inp_s - lbl_s), 99))

    if not all_intensity_vals:
        raise RuntimeError("Nothing loaded -- check --data_dir / CHECKPOINTS paths and that "
                            "run_inference_dump.py has been run (with --split matching --split "
                            "here) for the requested checkpoint(s).")

    # global, shared colour scales -- see docstring item 2 for why pre-CNN
    # and post-CNN diffs get two separate scales rather than one.
    if args.intensity_vmax_override is not None:
        global_vmax = args.intensity_vmax_override
    else:
        worst_val, worst_source = max(all_intensity_vals, key=lambda t: t[0])
        print(f"[diagnostic] intensity vmax set by: {worst_source} (99th pct = {worst_val:.3f})")
        global_vmax = worst_val * args.vmax_headroom
    global_diff_pre_absmax = (args.diff_pre_vmax_override if args.diff_pre_vmax_override is not None
                               else max(all_diff_pre_vals))
    global_diff_post_absmax = (args.diff_post_vmax_override if args.diff_post_vmax_override is not None
                                else max(all_diff_post_vals))
    print(f"Global intensity vmax = {global_vmax:.3f}")
    print(f"Pre-CNN diff scale  = +/-{global_diff_pre_absmax:.3f}")
    print(f"Post-CNN diff scale = +/-{global_diff_post_absmax:.3f}")

    # ------------------------------------------------------------------
    # Pass 2: plot, one figure per checkpoint, one row per alpha, using
    # the global scales computed above throughout.
    # ------------------------------------------------------------------
    col_titles = ["Noisy input", "Pre-CNN error",
                  "Model output", "Post-CNN error", "Ground truth"]

    for key in keys:
        n_rows = sum(1 for v in loaded[key].values() if v is not None)
        if n_rows == 0:
            print(f"[skip figure] {key}: no data loaded")
            continue

        fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4.2 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]

        row = 0
        for alpha in ALPHAS_ORDERED:
            alpha_str = ALPHA_STR[alpha]
            entry = loaded[key][alpha_str]
            if entry is None:
                continue
            phantom_idx, inp, lbl, den = entry

            inp_s = central_slice(inp, args.slice_axis)
            lbl_s = central_slice(lbl, args.slice_axis)
            den_s = central_slice(den, args.slice_axis)
            diff_pre = inp_s - lbl_s
            diff_post = den_s - lbl_s

            axes[row, 0].imshow(inp_s, cmap="gray", vmin=0, vmax=global_vmax)
            im_pre = axes[row, 1].imshow(diff_pre, cmap="coolwarm",
                                          vmin=-global_diff_pre_absmax, vmax=global_diff_pre_absmax)
            axes[row, 2].imshow(den_s, cmap="gray", vmin=0, vmax=global_vmax)
            im_post = axes[row, 3].imshow(diff_post, cmap="coolwarm",
                                           vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
            im_lbl = axes[row, 4].imshow(lbl_s, cmap="gray", vmin=0, vmax=global_vmax)

            axes[row, 0].set_ylabel(f"alpha={alpha}\nphantom {phantom_idx:04d}",
                                     fontsize=16, rotation=0, labelpad=75, va="center")

            if row == 0:
                for c, title in enumerate(col_titles):
                    axes[row, c].set_title(title, fontsize=15)

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])

            row += 1

        # pre/post diff colorbars use DIFFERENT scales (see docstring item
        # 2), so both need their own colorbar rather than sharing one.
        fig.colorbar(im_lbl, ax=axes[:, 4].tolist(), fraction=0.02, pad=0.02,
                     label="count-domain activity")
        fig.colorbar(im_pre, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02,
                     label="pre-CNN difference")
        fig.colorbar(im_post, ax=axes[:, 3].tolist(), fraction=0.02, pad=0.02,
                     label="post-CNN difference")

        split_tag = f"fixed10_p{args.fixed_phantom:04d}" if args.fixed_phantom is not None else args.split
        out_path = os.path.join(args.out_dir, f"qualitative_{key}_{args.dataset}_{split_tag}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()