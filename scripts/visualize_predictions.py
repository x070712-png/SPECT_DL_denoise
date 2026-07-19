#scripts/visualize_predictions.py

"""
Qualitative sanity check for a trained checkpoint: pick a few validation
samples, run inference, and plot noisy input / model output / ground-truth
label side by side (central axial slice), with per-sample PSNR/SSIM
annotated. Same normalisation + metric definitions as train_unet.py, so the
numbers you see here are directly comparable to the val_psnr/val_ssim
printed during training.

Run on a GPU node (submit as a job, not on the login node -- see the 16 Jul
Myriad usage warning; model inference on 3D volumes is too heavy for
login13):
 
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
import re
 
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from monai.losses import SSIMLoss
 
from spect.baseline.model import CustomUNet3D
from spect.baseline.dataset import SPECTDataset
from spect.baseline.quantification import build_voi_masks
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--checkpoint", type=str, default="checkpoints/3d_unet/best_model.pth")
    p.add_argument("--out_dir", type=str, default="logs/3d_unet/qualitative")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--slice_axis", type=int, default=0, help="0=axial, 1=coronal, 2=sagittal (over the D,H,W dims)")
    p.add_argument("--vmax_headroom", type=float, default=1.2,
                    help="multiply label's max by this factor for the shared colour scale (Kris, 7/15 meeting)")
    p.add_argument("--seed_base", type=int, default=42,
                    help="must match the seed_base used everywhere else (dataset generation, "
                         "quantify_noisy_baseline.py) so the reconstructed VOI masks line up")
    return p.parse_args()
 
 
 
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
    
 
def parse_phantom_alpha_from_path(inp_path):
    """Extract (phantom_idx, alpha_str) from an input file path like
    '.../alpha_0p5/input_0083.npy'. SPECTDataset doesn't separately expose
    which phantom a given __getitem__(i) came from, but ds.samples[i]
    still holds the original file paths, so we just parse those instead
    of touching dataset.py."""
    inp_path = str(inp_path)
    alpha_match = re.search(r"alpha_([0-9p]+)", inp_path)
    idx_match = re.search(r"input_(\d+)\.npy", inp_path)
    if alpha_match is None or idx_match is None:
        raise ValueError(f"couldn't parse phantom_idx/alpha from path: {inp_path}")
    return int(idx_match.group(1)), alpha_match.group(1)


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

    # 5 panels now: noisy input, model output, ground truth, VOI mask
    # overlay, difference image.
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        inp_n, lbl_n, scale = ds[i]  # (1, D, H, W) each, already mean-normalised by SPECTDataset
        inp_n, lbl_n = inp_n.unsqueeze(0).to(device), lbl_n.unsqueeze(0).to(device)  # add batch dim
        scale = scale.to(device).view(1, 1, 1, 1, 1)

        with torch.no_grad():
            out_n = model(inp_n)
        # restore to true count-domain via the SAME mean scale used going in
        out_cnt, lbl_cnt, inp_cnt = out_n * scale, lbl_n * scale, inp_n * scale

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

        # shared colour scale across the first three panels so brightness is
        # comparable -- vmax has headroom above the label's max (Kris,
        # 7/15 meeting) so bright voxels aren't pinned at the top of the
        # range, which would hide whether the network is over/under-shooting.
        vmax = lbl_slice.max() * args.vmax_headroom

        axes[i, 0].imshow(inp_slice, cmap="gray", vmin=0, vmax=vmax)
        axes[i, 0].set_title(f"Noisy input\nPSNR={psnr_noisy:.2f} SSIM={ssim_noisy:.3f}")
        axes[i, 1].imshow(out_slice, cmap="gray", vmin=0, vmax=vmax)
        axes[i, 1].set_title(f"Model output\nPSNR={psnr:.2f} SSIM={ssim:.3f}")
        im = axes[i, 2].imshow(lbl_slice, cmap="gray", vmin=0, vmax=vmax)
        axes[i, 2].set_title("Ground truth (label)")

        # one colorbar for the first three panels -- vmax is row-specific
        # (based on that row's own label peak), so a single figure-wide
        # colorbar would be misleading; this makes the shared 0..vmax scale
        # for THIS row explicit (Kris, 7/15 meeting).
        fig.colorbar(im, ax=axes[i, :3].tolist(), fraction=0.025, pad=0.02,
                     label="count-domain activity")

        # ---- panel 4: VOI mask overlay ----
        # recover which phantom this sample is, from the dataset's own
        # stored file paths, then rebuild the same ground-truth VOI mask
        # quantify_noisy_baseline.py uses (deterministic given seed_base).
        inp_path, _ = ds.samples[i]
        phantom_idx, alpha_str = parse_phantom_alpha_from_path(inp_path)
        combined_mask, per_voi, _ = build_voi_masks(phantom_idx, seed_base=args.seed_base)
        mask_slice = central_slice(combined_mask, args.slice_axis)

        axes[i, 3].imshow(lbl_slice, cmap="gray", vmin=0, vmax=vmax)
        # semi-transparent red overlay wherever the combined VOI mask is
        # True; overlapping-ellipsoid regions aren't distinguished here
        # (this is a coverage check, not an overlap diagnostic -- see
        # compute_isolation_flags() in quantify_noisy_baseline.py for that).
        overlay = np.zeros((*mask_slice.shape, 4))
        overlay[mask_slice, 0] = 1.0   # red channel
        overlay[mask_slice, 3] = 0.35  # alpha channel
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"VOI mask overlay\nphantom {phantom_idx:04d} alpha_{alpha_str} "
                              f"({len(per_voi)} VOIs)")

        # ---- panel 5: difference image (model output - label) ----
        diff_slice = out_slice - lbl_slice
        diff_absmax = max(np.abs(diff_slice).max(), 1e-8)  # avoid vmin=vmax=0
        im_diff = axes[i, 4].imshow(diff_slice, cmap="coolwarm", vmin=-diff_absmax, vmax=diff_absmax)
        axes[i, 4].set_title("Difference (output - label)\nred=over-estimate, blue=under-estimate")
        fig.colorbar(im_diff, ax=axes[i, 4], fraction=0.046, pad=0.04,
                     label="count-domain difference")

        for ax in axes[i]:
            ax.axis("off")

        print(f"[sample {i}] phantom {phantom_idx:04d} alpha_{alpha_str}  "
              f"noisy: PSNR={psnr_noisy:.2f} SSIM={ssim_noisy:.3f}  "
              f"-> denoised: PSNR={psnr:.2f} SSIM={ssim:.3f}")

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"qualitative_{args.split}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison grid to {out_path}")


if __name__ == "__main__":
    main()