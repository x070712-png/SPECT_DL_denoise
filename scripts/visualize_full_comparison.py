# scripts/visualize_full_comparison.py
"""
One-figure-does-everything version: show noisy input / pre-CNN error /
ground truth ONCE, plus TWO checkpoints' output and post-CNN error side
by side. Generic over WHICH two checkpoints -- originally built to
compare old-method vs label x alpha for one architecture (Section 4.3),
also works for comparing U-Net vs Swin UNETR within the SAME training
formulation (e.g. both old-method, for Section 4.1's pipeline
reproduction figures) -- any two of the four checkpoint_key entries in
CHECKPOINTS_BY_DATASET can go on the left/right.

Columns (5 rows, one per alpha level):

    [ Noisy input | Pre-CNN error | <left> output | <left> error |
      <right> output | <right> error | Ground truth ]

Shared colour scales, same discipline as the other qualitative scripts:
  - ONE intensity scale across noisy input / left output / right output /
    ground truth (all count-domain, all comparable).
  - ONE pre-CNN diff scale (noisy/alpha - label) -- only one such column
    here, but kept as its own scale rather than folding into the post-CNN
    one, since pre-CNN error is an order of magnitude larger by
    construction (same reasoning as visualize_predictions.py).
  - ONE post-CNN diff scale shared between left error AND right error --
    the whole point of putting two checkpoints side by side is that
    colour intensity is directly comparable between them.

Reads the same already-dumped denoised .npy files as the other two
qualitative scripts -- no torch/GPU needed, safe on the login node.

Two ways to pick the two checkpoints:

  1. --arch {unet,swin} -- shortcut for the original use case, old-method
     vs label x alpha for ONE architecture (Section 4.3):
       python3 scripts/visualize_full_comparison.py --dataset xcat --arch swin

  2. --left_key / --right_key -- any two of unet_old / unet_label_alpha /
     swin_old / swin_label_alpha directly. E.g. Section 4.1's "old-method,
     U-Net vs Swin" comparison:
       python3 scripts/visualize_full_comparison.py --dataset ellipsoid \\
           --left_key unet_old --right_key swin_old \\
           --left_label "U-Net" --right_label "Swin UNETR"
       python3 scripts/visualize_full_comparison.py --dataset xcat \\
           --left_key unet_old --right_key swin_old \\
           --left_label "U-Net" --right_label "Swin UNETR"

--left_key/--right_key take priority over --arch if both are given.
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
ALL_CHECKPOINT_KEYS = ["unet_old", "unet_label_alpha", "swin_old", "swin_label_alpha"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="xcat", choices=["ellipsoid", "xcat"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--arch", type=str, default="swin", choices=["unet", "swin"],
                    help="shortcut for --left_key/--right_key = (<arch>_old, <arch>_label_alpha); "
                         "ignored if --left_key/--right_key are given")
    p.add_argument("--left_key", type=str, default=None, choices=ALL_CHECKPOINT_KEYS)
    p.add_argument("--right_key", type=str, default=None, choices=ALL_CHECKPOINT_KEYS)
    p.add_argument("--left_label", type=str, default=None,
                    help="column title prefix for the left checkpoint (default: 'Baseline' if "
                         "--arch shortcut used, otherwise the checkpoint's own label)")
    p.add_argument("--right_label", type=str, default=None,
                    help="column title prefix for the right checkpoint (default: 'Label x alpha' "
                         "if --arch shortcut used, otherwise the checkpoint's own label)")
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
    p.add_argument("--diff_pre_vmax_override", type=float, default=None)
    p.add_argument("--diff_post_vmax_override", type=float, default=None,
                    help="single shared +/- scale for BOTH left and right post-CNN error columns.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    CHECKPOINTS = CHECKPOINTS_BY_DATASET[args.dataset]
    data_dir = args.data_dir or DATA_DIR_BY_DATASET[args.dataset]

    if args.left_key and args.right_key:
        old_key, new_key = args.left_key, args.right_key
        default_left_label, default_right_label = old_key, new_key
        tag = f"{old_key}_vs_{new_key}"
    else:
        old_key, new_key = ARCH_KEYS[args.arch]
        default_left_label, default_right_label = "Baseline", "Label x alpha"
        tag = args.arch

    left_label = args.left_label or default_left_label
    right_label = args.right_label or default_right_label
    old_cfg, new_cfg = dict(CHECKPOINTS[old_key]), dict(CHECKPOINTS[new_key])
    if args.fixed10_dirs:
        old_cfg["denoised_dir"] = old_cfg["denoised_dir"] + "_fixed10"
        new_cfg["denoised_dir"] = new_cfg["denoised_dir"] + "_fixed10"

    print(f"Dataset = {args.dataset}, data_dir = {data_dir}")
    print(f"  left  ({left_label}): {old_key} -> {old_cfg['denoised_dir']}")
    print(f"  right ({right_label}): {new_key} -> {new_cfg['denoised_dir']}")

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
    rows = {}
    intensity_vals = []
    diff_pre_vals = []
    diff_post_vals = []  # BOTH old and new post-CNN errors pooled here

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

        inp, lbl, old_out = old_data
        _, _, new_out = new_data  # inp/lbl identical to old_data's, only loaded once

        # noisy input naturally ~alpha x dimmer -- divide for display (see
        # visualize_predictions.py's CHECKPOINTS comment for the full reasoning)
        inp = inp / alpha
        if old_cfg["alpha_correction"]:
            old_out = old_out / alpha
        if new_cfg["alpha_correction"]:
            new_out = new_out / alpha

        inp_s = central_slice(inp, args.slice_axis)
        lbl_s = central_slice(lbl, args.slice_axis)
        old_s = central_slice(old_out, args.slice_axis)
        new_s = central_slice(new_out, args.slice_axis)

        rows[alpha_str] = dict(phantom_idx=phantom_idx, inp_s=inp_s, lbl_s=lbl_s,
                                old_s=old_s, new_s=new_s)

        # intensity scale: label + both model outputs only (not noisy input --
        # see visualize_predictions.py's comment on why noisy input is excluded)
        intensity_vals.append(np.percentile(lbl_s, 99))
        intensity_vals.append(np.percentile(old_s, 99))
        intensity_vals.append(np.percentile(new_s, 99))

        diff_pre_vals.append(np.percentile(np.abs(inp_s - lbl_s), 99))
        diff_post_vals.append(np.percentile(np.abs(old_s - lbl_s), 99))
        diff_post_vals.append(np.percentile(np.abs(new_s - lbl_s), 99))

    if not rows:
        raise RuntimeError("Nothing loaded -- check denoised_dir paths / run_inference_dump.py "
                            "has been run for both checkpoints with the requested --split.")

    global_vmax = (args.intensity_vmax_override if args.intensity_vmax_override is not None
                   else max(intensity_vals) * args.vmax_headroom)
    global_diff_pre_absmax = (args.diff_pre_vmax_override if args.diff_pre_vmax_override is not None
                               else max(diff_pre_vals))
    global_diff_post_absmax = (args.diff_post_vmax_override if args.diff_post_vmax_override is not None
                                else max(diff_post_vals))
    print(f"Global intensity vmax = {global_vmax:.3f}")
    print(f"Pre-CNN diff scale = +/-{global_diff_pre_absmax:.3f}")
    print(f"Post-CNN diff scale (old AND new, shared) = +/-{global_diff_post_absmax:.3f}")

    # ---- Pass 2: plot ----
    col_titles = ["Noisy input", "Pre-CNN error", f"{left_label} output", f"{left_label} error",
                  f"{right_label} output", f"{right_label} error", "Ground truth"]

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 7, figsize=(30, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    row = 0
    im_intensity = im_pre = im_post = None
    for alpha in ALPHAS_ORDERED:
        alpha_str = ALPHA_STR[alpha]
        if alpha_str not in rows:
            continue
        d = rows[alpha_str]
        pre_diff = d["inp_s"] - d["lbl_s"]
        old_diff = d["old_s"] - d["lbl_s"]
        new_diff = d["new_s"] - d["lbl_s"]

        axes[row, 0].imshow(d["inp_s"], cmap="gray", vmin=0, vmax=global_vmax)
        im_pre = axes[row, 1].imshow(pre_diff, cmap="coolwarm",
                                      vmin=-global_diff_pre_absmax, vmax=global_diff_pre_absmax)
        axes[row, 2].imshow(d["old_s"], cmap="gray", vmin=0, vmax=global_vmax)
        axes[row, 3].imshow(old_diff, cmap="coolwarm",
                             vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
        axes[row, 4].imshow(d["new_s"], cmap="gray", vmin=0, vmax=global_vmax)
        im_post = axes[row, 5].imshow(new_diff, cmap="coolwarm",
                                       vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
        im_intensity = axes[row, 6].imshow(d["lbl_s"], cmap="gray", vmin=0, vmax=global_vmax)

        axes[row, 0].set_ylabel(f"alpha={alpha}\nphantom {d['phantom_idx']:04d}",
                                 fontsize=16, rotation=0, labelpad=75, va="center")

        if row == 0:
            for c, title in enumerate(col_titles):
                axes[row, c].set_title(title, fontsize=15)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        row += 1

    fig.colorbar(im_intensity, ax=axes[:, 6].tolist(), fraction=0.02, pad=0.02,
                 label="count-domain activity")
    fig.colorbar(im_pre, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02,
                 label="pre-CNN error")
    fig.colorbar(im_post, ax=axes[:, 5].tolist(), fraction=0.02, pad=0.02,
                 label="post-CNN error (shared, old & new)")

    split_tag = f"fixed10_p{args.fixed_phantom:04d}" if args.fixed_phantom is not None else args.split
    out_path = os.path.join(args.out_dir, f"full_comparison_{tag}_{args.dataset}_{split_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()