# scripts/plot_ellipsoid_phantom_example.py
"""
Methods-section figure for the Ellipsoidal Phantom Dataset: one fixed
phantom (--phantom_idx, default 0) shown as a clean label + all 5 noisy
inputs side by side, to illustrate how OSEM reconstruction noise degrades
as the count level alpha decreases. Extends Miao's Figure 2.1 (single
count level) to show the full noise progression in one figure.

Usage (no PYTHONPATH / SIRF env needed):
    python3 scripts/plot_ellipsoid_phantom_example.py \
        --phantom_idx 0 \
        --data_dir data/dataset \
        --slice_axis 1 \
        --out_path logs/qualitative/ellipsoid_phantom_example.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}
ALPHAS_ORDERED = [1.0, 0.5, 0.25, 0.125, 0.05]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--phantom_idx", type=int, default=0)
    p.add_argument("--slice_axis", type=int, default=1,
                    help="0=axial, 1=coronal, 2=sagittal -- default 1 (coronal) "
                         "to match Miao's Figure 2.1 caption")
    p.add_argument("--vmax_headroom", type=float, default=1.2,
                    help="shared intensity vmax = label_slice.max() * this "
                         "(same convention as visualize_predictions.py)")
    p.add_argument("--out_path", type=str,
                    default="logs/qualitative/ellipsoid_phantom_example.png")
    return p.parse_args()


def central_slice(vol, axis):
    idx = vol.shape[axis] // 2
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # label.npy is identical across all 5 alpha_*/ dirs for a given
    # phantom_idx (generate_dataset.py computes it once at alpha=1.0 and
    # copies it into every alpha subfolder) -- load it once from alpha_1p0.
    label_path = os.path.join(args.data_dir, "alpha_1p0", f"label_{args.phantom_idx:04d}.npy")
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"{label_path} not found -- pick a different --phantom_idx "
            f"(check with: ls data/dataset/alpha_*/label_{args.phantom_idx:04d}.npy)")
    label = np.load(label_path).astype(np.float32)
    label_s = central_slice(label, args.slice_axis)

    inputs_s = {}
    missing = []
    for alpha in ALPHAS_ORDERED:
        alpha_str = ALPHA_STR[alpha]
        inp_path = os.path.join(args.data_dir, f"alpha_{alpha_str}", f"input_{args.phantom_idx:04d}.npy")
        if not os.path.exists(inp_path):
            missing.append(inp_path)
            continue
        inp = np.load(inp_path).astype(np.float32)
        inputs_s[alpha_str] = central_slice(inp, args.slice_axis)

    if missing:
        raise FileNotFoundError(
            f"phantom_idx {args.phantom_idx}: missing {len(missing)} input file(s): "
            f"{missing} -- pick a different --phantom_idx.")

    vmax = label_s.max() * args.vmax_headroom
    axis_name = {0: "axial", 1: "coronal", 2: "sagittal"}[args.slice_axis]

    # 2 rows x 3 cols: row 1 = label, alpha=1.0, alpha=0.5 -- row 2 =
    # alpha=0.25, alpha=0.125, alpha=0.05
    panels = [
        ("Clean label\n(alpha = 1.0, noise-free)", label_s),
        ("Noisy input\n(alpha = 1.0)", inputs_s[ALPHA_STR[1.0]]),
        ("Noisy input\n(alpha = 0.5)", inputs_s[ALPHA_STR[0.5]]),
        ("Noisy input\n(alpha = 0.25)", inputs_s[ALPHA_STR[0.25]]),
        ("Noisy input\n(alpha = 0.125)", inputs_s[ALPHA_STR[0.125]]),
        ("Noisy input\n(alpha = 0.05)", inputs_s[ALPHA_STR[0.05]]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    axes_flat = axes.flatten()

    im = None
    for ax, (title, img) in zip(axes_flat, panels):
        im = ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label="reconstructed activity")
    fig.suptitle(f"Synthetic Ellipsoidal Phantom {args.phantom_idx:04d} -- "
                 f"central {axis_name} slice, shared intensity scale", fontsize=13)
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()