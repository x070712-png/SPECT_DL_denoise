# scripts/plot_xcat_phantom_example.py
"""
Methods-section figure for the XCAT Dataset: a single representative
XCAT phantom shown as clean label + noisy input, mirroring Miao's
Figure 2.2 (single example at one count level).

Usage:
    python3 scripts/plot_xcat_phantom_example.py \
        --phantom_idx 0 \
        --alpha_str 1p0 \
        --data_dir data/xcat_dataset \
        --slice_axis 1 \
        --out_path logs/qualitative/xcat_phantom_example.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_STR_TO_FLOAT = {"1p0": 1.0, "0p5": 0.5, "0p25": 0.25, "0p125": 0.125, "0p05": 0.05}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/xcat_dataset")
    p.add_argument("--phantom_idx", type=int, default=0)
    p.add_argument("--alpha_str", type=str, default="1p0",
                    choices=list(ALPHA_STR_TO_FLOAT.keys()),
                    help="must match the count-level block --phantom_idx belongs to "
                         "(0-99=1p0, 100-199=0p5, 200-299=0p25, 300-399=0p125, 400-499=0p05)")
    p.add_argument("--slice_axis", type=int, default=1,
                    help="0=axial, 1=coronal, 2=sagittal -- default 1 (coronal) "
                         "to match Miao's Figure 2.2 caption")
    p.add_argument("--vmax_headroom", type=float, default=1.2)
    p.add_argument("--out_path", type=str,
                    default="logs/qualitative/xcat_phantom_example.png")
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

    alpha_val = ALPHA_STR_TO_FLOAT[args.alpha_str]
    alpha_dir = os.path.join(args.data_dir, f"alpha_{args.alpha_str}")
    label_path = os.path.join(alpha_dir, f"label_{args.phantom_idx:04d}.npy")
    input_path = os.path.join(alpha_dir, f"input_{args.phantom_idx:04d}.npy")

    for p in (label_path, input_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- check --phantom_idx belongs to the "
                f"--alpha_str={args.alpha_str} block (0-99=1p0, 100-199=0p5, "
                f"200-299=0p25, 300-399=0p125, 400-499=0p05).")

    label = np.load(label_path).astype(np.float32)
    inp = np.load(input_path).astype(np.float32)
    label_s = central_slice(label, args.slice_axis)
    inp_s = central_slice(inp, args.slice_axis)


    label_vmax = label_s.max() * args.vmax_headroom
    input_vmax = inp_s.max() * args.vmax_headroom
    axis_name = {0: "axial", 1: "coronal", 2: "sagittal"}[args.slice_axis]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    im0 = axes[0].imshow(label_s, cmap="hot", vmin=0, vmax=label_vmax)
    axes[0].set_title("Clean label\n(noise-free)", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="reconstructed activity")

    im1 = axes[1].imshow(inp_s, cmap="hot", vmin=0, vmax=input_vmax)
    axes[1].set_title(f"Noisy input\n(alpha = {alpha_val})", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="reconstructed activity")
    fig.suptitle(f"XCAT Phantom {args.phantom_idx:04d} -- central {axis_name} slice, "
                 f"count level alpha = {alpha_val}", fontsize=13)
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()