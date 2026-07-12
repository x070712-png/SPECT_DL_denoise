#scripts/visualize_predictions.py

"""
Qualitative sanity check for a trained checkpoint: pick a few validation
samples, run inference, and plot noisy input / model output / ground-truth
label side by side (central axial slice), with per-sample PSNR/SSIM
annotated. Same normalisation + metric definitions as train_unet.py, so the
numbers you see here are directly comparable to the val_psnr/val_ssim
printed during training.

Run on the LOGIN NODE (no GPU needed):

    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/visualize_predictions.py \
        --data_dir data/dataset \
        --checkpoint checkpoints/3d_unet/best_model.pth \
        --out_dir logs/3d_unet/qualitative \
        --num_samples 4
"""

import argparse
import os
 
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from monai.losses import SSIMLoss
 
from spect.baseline.model import CustomUNet3D
from spect.baseline.dataset import SPECTDataset
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--checkpoint", type=str, default="checkpoints/3d_unet/best_model.pth")
    p.add_argument("--out_dir", type=str, default="logs/3d_unet/qualitative")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--slice_axis", type=int, default=0, help="0=axial, 1=coronal, 2=sagittal (over the D,H,W dims)")
    return p.parse_args()
 
 
def per_volume_scale(gt, eps=1e-8):
    # FIXED 12 Jul: matches train_unet.py — peak (max) of the ground-truth
    # label, not the mean of the input. See train_unet.py's per_volume_scale
    # docstring for why this matters.
    return gt.max().clamp(min=eps)
 
 
def compute_psnr(pred_cnt, tgt_cnt, eps=1e-8):
    peak = tgt_cnt.max() + eps
    pred_n, tgt_n = pred_cnt / peak, tgt_cnt / peak
    mse = torch.mean((pred_n - tgt_n) ** 2)
    if mse == 0:
        return float("inf")
    return (10.0 * torch.log10(1.0 / mse)).item()
 
 
def central_slice(vol, axis):
    # vol: (D, H, W) tensor
    idx = vol.shape[axis] // 2
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]
 
 
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    model = CustomUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
 
    ds = SPECTDataset(args.data_dir, split=args.split)
    n = min(args.num_samples, len(ds))
    print(f"Visualising {n} samples from '{args.split}' split")
 
    ssim_metric = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=7, reduction="mean")
 
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]
 
    for i in range(n):
        inp, lbl = ds[i]  # (1, D, H, W) each, raw counts
        inp_b, lbl_b = inp.unsqueeze(0).to(device), lbl.unsqueeze(0).to(device)  # add batch dim
 
        scale = per_volume_scale(lbl_b)
        inp_n, lbl_n = inp_b / scale, lbl_b / scale
 
        with torch.no_grad():
            out_n = model(inp_n)
        out_cnt, lbl_cnt, inp_cnt = out_n * scale, lbl_n * scale, inp_b
 
        psnr = compute_psnr(out_cnt, lbl_cnt)
        peak = lbl_cnt.max() + 1e-8
        ssim = 1.0 - ssim_metric(out_cnt / peak, lbl_cnt / peak).item()
 
        # also report the noisy input's own PSNR/SSIM vs GT, for a before/after comparison
        psnr_noisy = compute_psnr(inp_cnt, lbl_cnt)
        ssim_noisy = 1.0 - ssim_metric(inp_cnt / peak, lbl_cnt / peak).item()
 
        # squeeze to (D, H, W) for slicing, move to CPU numpy
        inp_slice = central_slice(inp_cnt.squeeze(0).squeeze(0).cpu(), args.slice_axis).numpy()
        out_slice = central_slice(out_cnt.squeeze(0).squeeze(0).cpu(), args.slice_axis).numpy()
        lbl_slice = central_slice(lbl_cnt.squeeze(0).squeeze(0).cpu(), args.slice_axis).numpy()
 
        # shared colour scale across the three panels so brightness is comparable
        vmax = lbl_slice.max()
 
        axes[i, 0].imshow(inp_slice, cmap="gray", vmin=0, vmax=vmax)
        axes[i, 0].set_title(f"Noisy input\nPSNR={psnr_noisy:.2f} SSIM={ssim_noisy:.3f}")
        axes[i, 1].imshow(out_slice, cmap="gray", vmin=0, vmax=vmax)
        axes[i, 1].set_title(f"Model output\nPSNR={psnr:.2f} SSIM={ssim:.3f}")
        axes[i, 2].imshow(lbl_slice, cmap="gray", vmin=0, vmax=vmax)
        axes[i, 2].set_title("Ground truth (label)")
 
        for ax in axes[i]:
            ax.axis("off")
 
        print(f"[sample {i}] noisy: PSNR={psnr_noisy:.2f} SSIM={ssim_noisy:.3f}  "
              f"-> denoised: PSNR={psnr:.2f} SSIM={ssim:.3f}")
 
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"qualitative_{args.split}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison grid to {out_path}")
 
 
if __name__ == "__main__":
    main()