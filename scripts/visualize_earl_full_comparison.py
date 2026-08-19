# scripts/visualize_earl_full_comparison.py
"""
Compact U-Net vs Swin UNETR comparison for ONE EARL background variant,
same idea as visualize_full_comparison.py / visualize_old_vs_labelalpha.py
applied to the EARL qualitative figures: noisy input / pre-CNN error /
ground truth only need to be shown ONCE per variant (they don't depend on
which architecture denoised the data), so U-Net and Swin's outputs and
post-CNN errors can go side by side in one 7-column figure instead of two
separate 5-column ones.

Columns (5 rows, one per alpha level):

    [ Noisy input | Pre-CNN error | U-Net output | U-Net error |
      Swin output | Swin error | Ground truth ]

Shared colour scales:
  - ONE intensity scale across noisy input / U-Net output / Swin output /
    ground truth.
  - ONE pre-CNN diff scale.
  - ONE post-CNN diff scale shared between U-Net error AND Swin error.
    It required once when they're side by side, same reasoning as the other
    comparison scripts.

Single representative seed (default 42), same z-slice selection (max
sphere-mask coverage) as visualize_earl_predictions.py.

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/visualize_earl_full_comparison.py --variant v3_bg0
    python3 scripts/visualize_earl_full_comparison.py --variant v3_bg_ratio10
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_earl_predictions import (
    EARL_VARIANTS,
    ALPHA_STR,
    ALPHAS_ORDERED,
    find_sphere_slice,
    load_triplet,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="v3_bg0", choices=list(EARL_VARIANTS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="logs/qualitative_earl")
    p.add_argument("--vmax_headroom", type=float, default=1.2)
    p.add_argument("--intensity_vmax_override", type=float, default=None)
    p.add_argument("--diff_pre_vmax_override", type=float, default=None)
    p.add_argument("--diff_post_vmax_override", type=float, default=None,
                    help="single shared +/- scale for BOTH U-Net and Swin post-CNN error columns.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    variant = EARL_VARIANTS[args.variant]
    data_dir = variant["data_dir"]
    sphere_dir = variant["sphere_dir"]
    unet_cfg = variant["checkpoints"]["unet_xcat_labelalpha"]
    swin_cfg = variant["checkpoints"]["swin_xcat_labelalpha"]

    print(f"variant = {args.variant}, data_dir = {data_dir}, sphere_dir = {sphere_dir}")
    print(f"  U-Net: {unet_cfg['denoised_dir']}")
    print(f"  Swin:  {swin_cfg['denoised_dir']}")

    z = find_sphere_slice(sphere_dir)

    # ---- Pass 1: load everything, compute shared scales ----
    rows = {}
    intensity_vals = []
    diff_pre_vals = []
    diff_post_vals = []  # BOTH U-Net and Swin post-CNN errors pooled here

    for alpha in ALPHAS_ORDERED:
        alpha_str = ALPHA_STR[alpha]
        unet_data, unet_missing = load_triplet(data_dir, unet_cfg["denoised_dir"], alpha_str, args.seed)
        swin_data, swin_missing = load_triplet(data_dir, swin_cfg["denoised_dir"], alpha_str, args.seed)
        if unet_data is None or swin_data is None:
            print(f"[skip] alpha_{alpha_str} seed={args.seed}: "
                  f"missing unet={unet_missing} swin={swin_missing}")
            continue

        inp, lbl, unet_out = unet_data
        _, _, swin_out = swin_data  # inp/lbl identical, only loaded once

        inp = inp / alpha
        if unet_cfg["alpha_correction"]:
            unet_out = unet_out / alpha
        if swin_cfg["alpha_correction"]:
            swin_out = swin_out / alpha

        inp_s, lbl_s, unet_s, swin_s = inp[z], lbl[z], unet_out[z], swin_out[z]

        rows[alpha_str] = dict(inp_s=inp_s, lbl_s=lbl_s, unet_s=unet_s, swin_s=swin_s)

        intensity_vals.append(lbl_s.max())
        intensity_vals.append(unet_s.max())
        intensity_vals.append(swin_s.max())
        diff_pre_vals.append(np.percentile(np.abs(inp_s - lbl_s), 99))
        diff_post_vals.append(np.percentile(np.abs(unet_s - lbl_s), 99))
        diff_post_vals.append(np.percentile(np.abs(swin_s - lbl_s), 99))

    if not rows:
        raise RuntimeError(f"Nothing loaded -- check EARL_VARIANTS['{args.variant}'] paths and "
                            "that run_inference_nema_earl.py has been run for the requested seed.")

    global_vmax = (args.intensity_vmax_override if args.intensity_vmax_override is not None
                   else max(intensity_vals) * args.vmax_headroom)
    global_diff_pre_absmax = (args.diff_pre_vmax_override if args.diff_pre_vmax_override is not None
                               else max(diff_pre_vals))
    global_diff_post_absmax = (args.diff_post_vmax_override if args.diff_post_vmax_override is not None
                                else max(diff_post_vals))
    print(f"Global intensity vmax = {global_vmax:.3f}")
    print(f"Pre-CNN diff scale = +/-{global_diff_pre_absmax:.3f}")
    print(f"Post-CNN diff scale (U-Net AND Swin, shared) = +/-{global_diff_post_absmax:.3f}")

    # ---- Pass 2: plot ----
    col_titles = ["Noisy input", "Pre-CNN error", "U-Net output", "U-Net error",
                  "Swin output", "Swin error", "Ground truth"]

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
        unet_diff = d["unet_s"] - d["lbl_s"]
        swin_diff = d["swin_s"] - d["lbl_s"]

        axes[row, 0].imshow(d["inp_s"], cmap="gray", vmin=0, vmax=global_vmax)
        im_pre = axes[row, 1].imshow(pre_diff, cmap="coolwarm",
                                      vmin=-global_diff_pre_absmax, vmax=global_diff_pre_absmax)
        axes[row, 2].imshow(d["unet_s"], cmap="gray", vmin=0, vmax=global_vmax)
        axes[row, 3].imshow(unet_diff, cmap="coolwarm",
                             vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
        axes[row, 4].imshow(d["swin_s"], cmap="gray", vmin=0, vmax=global_vmax)
        im_post = axes[row, 5].imshow(swin_diff, cmap="coolwarm",
                                       vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
        im_intensity = axes[row, 6].imshow(d["lbl_s"], cmap="gray", vmin=0, vmax=global_vmax)

        axes[row, 0].set_ylabel(f"alpha={alpha}\nEARL, z={z}, seed={args.seed}",
                                 fontsize=16, rotation=0, labelpad=90, va="center")

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
                 label="post-CNN error (shared, U-Net & Swin)")

    out_path = os.path.join(args.out_dir, f"full_comparison_earl_{args.variant}_seed{args.seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()