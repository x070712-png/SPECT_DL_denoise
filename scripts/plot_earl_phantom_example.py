# scripts/plot_earl_phantom_example.py
"""
Methods-section figure for the EARL Phantom Dataset: the single EARL
phantom shown as clean label + all 5 alpha noisy inputs, same 2x3 grid
layout as plot_ellipsoid_phantom_example.py. Unlike XCAT, EARL genuinely
has all 5 alpha versions of the SAME underlying activity map
(generate_eval_phantom_dataset.py forward-projects the one calibrated
phantom at all 5 count levels), so the "same phantom, noise degrading
across count levels" story applies here too.

Two things this does differently from plot_ellipsoid_phantom_example.py:
  1. Slice selection: EARL's spheres sit off-centre (ring_z=-37mm per
     phantomgen's geometry), so a naive central-index slice could miss
     them. Reuses the same find_sphere_slice() logic as
     visualize_earl_predictions.py (picks the z-slice with the most
     total sphere-mask coverage, summed across all 6 spheres).
  2. Colour scale: each panel is scaled to its OWN max (not one shared
     scale across all 6 panels) -- same fix as
     plot_xcat_phantom_example.py, for the same reason: at alpha=0.05
     the reconstructed activity is genuinely ~20x lower in absolute
     units than the alpha=1.0 label, so a shared scale calibrated to
     the label would crush the low-alpha panels to near-black.

Usage:
    python3 scripts/plot_earl_phantom_example.py \
        --data_dir data/earl_dataset_v2 \
        --sphere_dir data/earl_phantom_v2 \
        --sphere_prefix EARL_sphere_ \
        --seed 42 \
        --out_path logs/qualitative/earl_phantom_example.png
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}
ALPHAS_ORDERED = [1.0, 0.5, 0.25, 0.125, 0.05]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/earl_dataset_v2")
    p.add_argument("--sphere_dir", type=str, default="data/earl_phantom_v2")
    p.add_argument("--sphere_prefix", type=str, default="EARL_sphere_")
    p.add_argument("--seed", type=int, default=42,
                    help="which single noise realisation to display -- RC numbers "
                         "are seed-averaged separately (quantify_nema_earl.py); "
                         "this is for visual inspection of one representative draw")
    p.add_argument("--vmax_headroom", type=float, default=1.2)
    p.add_argument("--out_path", type=str,
                    default="logs/qualitative/earl_phantom_example.png")
    return p.parse_args()


def find_sphere_slice(sphere_dir, sphere_prefix):
    """Same logic as visualize_earl_predictions.py: pick the z-slice with
    the most total sphere-mask coverage (summed across all 6 spheres),
    since EARL's spheres sit off-centre (ring_z=-37mm) and a naive
    central-index slice could miss them entirely."""
    pattern = os.path.join(sphere_dir, f"{sphere_prefix}*mm.npy")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No sphere mask files found matching {pattern}")

    total = None
    for p in paths:
        mask = np.load(p).astype(np.int32)
        total = mask if total is None else total + mask

    per_slice_counts = total.sum(axis=(1, 2))
    z = int(np.argmax(per_slice_counts))
    print(f"Selected z-slice {z} (total sphere voxels in-slice = {per_slice_counts[z]})")
    return z


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    z = find_sphere_slice(args.sphere_dir, args.sphere_prefix)

    # label.npy is identical across all 5 alpha_*/ dirs (same underlying
    # calibrated phantom, forward-projected once at alpha=1.0) -- load
    # once from alpha_1p0.
    label_path = os.path.join(args.data_dir, "alpha_1p0", "label.npy")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path} not found -- check --data_dir")
    label = np.load(label_path).astype(np.float32)
    label_s = label[z]

    inputs_s = {}
    missing = []
    for alpha in ALPHAS_ORDERED:
        alpha_str = ALPHA_STR[alpha]
        inp_path = os.path.join(args.data_dir, f"alpha_{alpha_str}", f"input_seed{args.seed}.npy")
        if not os.path.exists(inp_path):
            missing.append(inp_path)
            continue
        inp = np.load(inp_path).astype(np.float32)
        inputs_s[alpha_str] = inp[z]

    if missing:
        raise FileNotFoundError(f"missing {len(missing)} input file(s): {missing}")

    # each panel independently scaled to its own max -- NOT a shared
    # scale (see module docstring for why: low-alpha panels would
    # otherwise be crushed to near-black relative to the label).
    panels = [("Clean label\n(noise-free)", label_s)]
    for alpha in ALPHAS_ORDERED:
        panels.append((f"Noisy input\n(alpha = {alpha})", inputs_s[ALPHA_STR[alpha]]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    axes_flat = axes.flatten()

    for ax, (title, img) in zip(axes_flat, panels):
        vmax = img.max() * args.vmax_headroom
        im = ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="reconstructed activity")

    fig.suptitle(f"EARL Phantom -- central sphere-containing slice (z={z}), "
                 f"seed={args.seed}, each panel independently scaled", fontsize=13)
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()