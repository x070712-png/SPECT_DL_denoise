# scripts/visualize_old_vs_labelalpha.py
"""
Compact "old-method vs label x alpha" comparison figure, for Section 4.3.

Rationale: 4.2 already shows the full 5-column layout (noisy input,
pre-CNN diff, model output, post-CNN diff, ground truth) for the baseline
formulation, on the same 5 representative phantoms (one per alpha level).
Noisy input / pre-CNN diff / ground truth do NOT depend on which
checkpoint denoised the data -- they're the same images whether you're
looking at the old-method or label x alpha output. Repeating all 5
columns again in 4.3 to show the correction is therefore redundant and
makes new-vs-old harder to compare (reader has to flip between two
separate 5-column figures).

This script instead produces ONE 4-column x 5-row figure per architecture:

    [ Baseline output | Baseline error | Label x alpha output | Label x alpha error ]

with the two OUTPUT columns sharing one intensity colour scale, and the
two ERROR (post-CNN diff) columns sharing one diff colour scale -- this
is required, not optional: once old and new are side by side in the same
figure, a reader will visually compare "how red/blue" each panel is, and
that comparison is only valid if both diff panels are on the identical
scale. (Mirrors the shared-scale requirement from the 8/X supervisor
meeting, applied within this one figure rather than across all figures.)

4.2's two 5-column figures are untouched -- they still need the full
context since they're the reader's first exposure to this qualitative
comparison. This script only replaces what would otherwise have been
4.3's 5-column figures.

Reads the SAME already-dumped denoised .npy files as visualize_predictions.py
(via run_inference_dump.py) -- no torch/GPU needed, safe on the login node.

Run ONE PER ARCHITECTURE (2 total for ellipsoid, 2 more for xcat if you
want that comparison too):

    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/visualize_old_vs_labelalpha.py --dataset ellipsoid --arch unet
    python3 scripts/visualize_old_vs_labelalpha.py --dataset ellipsoid --arch swin
    python3 scripts/visualize_old_vs_labelalpha.py --dataset xcat --arch unet
    python3 scripts/visualize_old_vs_labelalpha.py --dataset xcat --arch swin
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_predictions import (
    CHECKPOINTS_BY_DATASET,
    DATA_DIR_BY_DATASET,
    ALPHA_STR,
    ALPHAS_ORDERED,
    central_slice,
    pick_representative_phantoms,
    pick_fixed_phantom,
    load_triplet,
)

ARCH_KEYS = {
    "unet": ("unet_old", "unet_label_alpha"),
    "swin": ("swin_old", "swin_label_alpha"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ellipsoid", choices=["ellipsoid", "xcat"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--arch", type=str, required=True, choices=["unet", "swin"])
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                    help="ignored if --fixed_phantom is given")
    p.add_argument("--fixed_phantom", type=int, default=None,
                    help="if given, show this SAME phantom index at all 5 alphas instead of "
                         "pick_representative_phantoms()'s one-different-phantom-per-alpha "
                         "choice -- matches the fixed-10-phantom x 5-alpha evaluation "
                         "(Stathis, 8/14 review). Use an index in 90-99. MUST be paired with "
                         "--fixed10_dirs. See visualize_predictions.py's pick_fixed_phantom().")
    p.add_argument("--fixed10_dirs", action="store_true",
                    help="redirect both checkpoints' denoised_dir to the '<denoised_dir>_fixed10' "
                         "variant produced by run_inference_dump.py --phantom_indices 90,...,99 "
                         "-- required for --fixed_phantom to find denoised output beyond alpha_1p0")
    p.add_argument("--out_dir", type=str, default="logs/qualitative")
    p.add_argument("--slice_axis", type=int, default=0)
    p.add_argument("--vmax_headroom", type=float, default=1.2)
    p.add_argument("--intensity_vmax_override", type=float, default=None)
    p.add_argument("--diff_vmax_override", type=float, default=None,
                    help="single shared +/- scale for BOTH the baseline and "
                         "label x alpha post-CNN error columns (deliberately "
                         "one value, not two -- see module docstring).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    CHECKPOINTS = CHECKPOINTS_BY_DATASET[args.dataset]
    data_dir = args.data_dir or DATA_DIR_BY_DATASET[args.dataset]
    old_key, new_key = ARCH_KEYS[args.arch]
    old_cfg, new_cfg = dict(CHECKPOINTS[old_key]), dict(CHECKPOINTS[new_key])
    if args.fixed10_dirs:
        old_cfg["denoised_dir"] = old_cfg["denoised_dir"] + "_fixed10"
        new_cfg["denoised_dir"] = new_cfg["denoised_dir"] + "_fixed10"

    print(f"Dataset = {args.dataset}, arch = {args.arch}, data_dir = {data_dir}")
    print(f"  old: {old_key} -> {old_cfg['denoised_dir']}")
    print(f"  new: {new_key} -> {new_cfg['denoised_dir']}")

    if args.fixed_phantom is not None:
        representative = pick_fixed_phantom(args.fixed_phantom)
        print(f"[FIXED-PHANTOM MODE] Using phantom {args.fixed_phantom:04d} at all 5 alphas "
              f"(--split={args.split} ignored)")
        if not args.fixed10_dirs:
            print("[warn] --fixed_phantom given without --fixed10_dirs -- denoised lookups "
                  "for alphas other than 1p0 will likely 404")
    else:
        representative = pick_representative_phantoms(args.split)
        print(f"Representative phantoms ({args.split} split): {representative}")

    # ---- Pass 1: load everything, compute shared scales ----
    rows = {}  # alpha_str -> dict with phantom_idx, lbl, old_out, new_out
    intensity_vals = []
    diff_vals = []  # BOTH old and new post-CNN errors go into ONE pool

    for alpha in ALPHAS_ORDERED:
        alpha_str = ALPHA_STR[alpha]
        phantom_idx = representative.get(alpha_str)
        if phantom_idx is None:
            print(f"[warn] no {args.split}-split phantom for alpha_{alpha_str}, skipping")
            continue

        old_data, old_missing = load_triplet(data_dir, old_cfg["denoised_dir"], phantom_idx, alpha_str)
        new_data, new_missing = load_triplet(data_dir, new_cfg["denoised_dir"], phantom_idx, alpha_str)
        if old_data is None or new_data is None:
            print(f"[skip] alpha_{alpha_str} phantom {phantom_idx:04d}: "
                  f"missing old={old_missing} new={new_missing}")
            continue

        _, lbl, old_out = old_data
        _, _, new_out = new_data  # label is identical between old/new loads, only need it once

        if old_cfg["alpha_correction"]:
            old_out = old_out / alpha
        if new_cfg["alpha_correction"]:
            new_out = new_out / alpha

        lbl_s = central_slice(lbl, args.slice_axis)
        old_s = central_slice(old_out, args.slice_axis)
        new_s = central_slice(new_out, args.slice_axis)

        rows[alpha_str] = dict(phantom_idx=phantom_idx, lbl_s=lbl_s, old_s=old_s, new_s=new_s)

        intensity_vals.append(np.percentile(lbl_s, 99))
        intensity_vals.append(np.percentile(old_s, 99))
        intensity_vals.append(np.percentile(new_s, 99))
        diff_vals.append(np.percentile(np.abs(old_s - lbl_s), 99))
        diff_vals.append(np.percentile(np.abs(new_s - lbl_s), 99))

    if not rows:
        raise RuntimeError("Nothing loaded -- check denoised_dir paths / run_inference_dump.py "
                            "has been run for both checkpoints with the requested --split.")

    global_vmax = (args.intensity_vmax_override if args.intensity_vmax_override is not None
                   else max(intensity_vals) * args.vmax_headroom)
    global_diff_absmax = (args.diff_vmax_override if args.diff_vmax_override is not None
                           else max(diff_vals))
    print(f"Global intensity vmax = {global_vmax:.3f}")
    print(f"Shared post-CNN error scale (old AND new) = +/-{global_diff_absmax:.3f}")

    # ---- Pass 2: plot ----
    col_titles = ["Baseline output", "Baseline error", "Label x alpha output", "Label x alpha error"]

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    row = 0
    im_out = im_diff = None
    for alpha in ALPHAS_ORDERED:
        alpha_str = ALPHA_STR[alpha]
        if alpha_str not in rows:
            continue
        d = rows[alpha_str]
        old_diff = d["old_s"] - d["lbl_s"]
        new_diff = d["new_s"] - d["lbl_s"]

        axes[row, 0].imshow(d["old_s"], cmap="gray", vmin=0, vmax=global_vmax)
        axes[row, 1].imshow(old_diff, cmap="coolwarm",
                             vmin=-global_diff_absmax, vmax=global_diff_absmax)
        im_out = axes[row, 2].imshow(d["new_s"], cmap="gray", vmin=0, vmax=global_vmax)
        im_diff = axes[row, 3].imshow(new_diff, cmap="coolwarm",
                                       vmin=-global_diff_absmax, vmax=global_diff_absmax)

        axes[row, 0].set_ylabel(f"alpha={alpha}\nphantom {d['phantom_idx']:04d}",
                                 fontsize=16, rotation=0, labelpad=75, va="center")

        if row == 0:
            for c, title in enumerate(col_titles):
                axes[row, c].set_title(title, fontsize=15)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        row += 1

    fig.colorbar(im_out, ax=axes[:, 2].tolist(), fraction=0.02, pad=0.02,
                 label="count-domain activity")
    fig.colorbar(im_diff, ax=axes[:, 3].tolist(), fraction=0.02, pad=0.02,
                 label="post-CNN error (shared scale)")

    split_tag = f"fixed10_p{args.fixed_phantom:04d}" if args.fixed_phantom is not None else args.split
    out_path = os.path.join(args.out_dir, f"compare_old_vs_labelalpha_{args.arch}_{args.dataset}_{split_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()