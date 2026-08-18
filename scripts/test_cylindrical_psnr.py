# scripts/test_cylindrical_psnr.py
"""
What this does: run the current best checkpoint over the validation set,
compute PSNR/SSIM/MSE two ways per sample --
  (a) "whole volume"     -- exactly what train_unet.py already reports
  (b) "cylinder-masked"  -- same prediction/label, but MSE/PSNR computed
                             only over voxels where cylindrical_mask==True
-- and report both, plus the delta, so we know how much of the 10dB gap
this choice alone could account for. Uses the SAME per_volume_scale
(label-peak) normalisation as train_unet.py, so the (a) numbers here should
closely match the logged val_psnr/val_ssim from training -- if they don't,
that's a bug in this script, not a new finding.

Myriad the same way as train_unet.py:

    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/test_cylindrical_psnr.py \
        --checkpoint checkpoints/3d_unet/best_model.pth \
        --data_dir data/dataset
"""

import argparse

import numpy as np
import torch

from monai.losses import SSIMLoss

from spect.baseline.model import CustomUNet3D
from spect.baseline.dataset import SPECTDataset
from spect.baseline.generate_ellipsoids import cylindrical_mask, CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/3d_unet/best_model.pth")
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--out_csv", type=str, default="logs/cylindrical_psnr_test.csv")
    return p.parse_args()


def per_volume_scale(gt, eps=1e-8):
    """Identical to train_unet.py: peak of the ground-truth label, per
    volume -- keep this in sync so the "whole volume" numbers below are a
    true apples-to-apples match to what was logged during training."""
    b = gt.shape[0]
    return gt.view(b, -1).amax(dim=1).clamp(min=eps).view(b, 1, 1, 1, 1)


def compute_psnr_masked(pred_cnt, tgt_cnt, mask=None, eps=1e-8):
    """Same peak-normalised PSNR definition as train_unet.py/
    test_resolution_model_recon.py, with an optional boolean mask to
    restrict the MSE (and the peak) to a subset of voxels. mask is a numpy
    bool array broadcastable to pred_cnt/tgt_cnt's spatial dims (assumes
    batch size 1 here, called per-sample)."""
    pred = pred_cnt.squeeze().cpu().numpy()
    tgt = tgt_cnt.squeeze().cpu().numpy()

    if mask is not None:
        pred = pred[mask]
        tgt = tgt[mask]

    peak = tgt.max() + eps
    pred_n, tgt_n = pred / peak, tgt / peak
    mse = np.mean((pred_n - tgt_n) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    return psnr, mse


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- the cylinder, built exactly like at dataset-generation time ----
    cyl_mask = cylindrical_mask(CONFIG["shape"], CONFIG["mask_radius_mm"], CONFIG["pixel_size_mm"])
    frac_inside = cyl_mask.mean()
    print(f"Cylinder mask: {frac_inside:.1%} of voxels inside "
          f"(radius={CONFIG['mask_radius_mm']}mm, pixel={CONFIG['pixel_size_mm']}mm)")

    # ---- model ----
    model = CustomUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ssim_metric_eval = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=7, reduction="mean")

    ds = SPECTDataset(args.data_dir, split=args.split)
    print(f"{args.split}: {len(ds)} samples")

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            inp, lbl = ds[i]
            inp, lbl = inp.unsqueeze(0).to(device), lbl.unsqueeze(0).to(device)

            scale = per_volume_scale(lbl)
            inp_n, lbl_n = inp / scale, lbl / scale
            out_n = model(inp_n)
            out_cnt, lbl_cnt = out_n * scale, lbl_n * scale

            # (a) whole volume -- should match train_unet.py's logged numbers
            psnr_full, mse_full = compute_psnr_masked(out_cnt, lbl_cnt, mask=None)

            # (b) cylinder-masked
            psnr_cyl, mse_cyl = compute_psnr_masked(out_cnt, lbl_cnt, mask=cyl_mask)

            # SSIM only really makes sense on the full volume (window-based,
            # not voxel-independent) -- reported for reference only.
            peak = lbl_cnt.amax(dim=tuple(range(1, lbl_cnt.dim()))).view(-1, 1, 1, 1, 1) + 1e-8
            ssim_full = float(1.0 - ssim_metric_eval(out_cnt / peak, lbl_cnt / peak))

            rows.append({
                "idx": i,
                "psnr_full": psnr_full,
                "psnr_cylinder": psnr_cyl,
                "delta_psnr": psnr_cyl - psnr_full,
                "mse_full": mse_full,
                "mse_cylinder": mse_cyl,
                "ssim_full": ssim_full,
            })
            print(f"[{i:03d}] PSNR full={psnr_full:6.2f}dB  cylinder={psnr_cyl:6.2f}dB  "
                  f"delta={psnr_cyl - psnr_full:+.2f}dB  SSIM(full)={ssim_full:.4f}")

    # ---- summary ----
    import csv, os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_full = np.mean([r["psnr_full"] for r in rows])
    mean_cyl = np.mean([r["psnr_cylinder"] for r in rows])
    print("\n=== Summary ===")
    print(f"Mean PSNR, whole volume:      {mean_full:.2f} dB  (should match training's logged val_psnr)")
    print(f"Mean PSNR, cylinder-masked:   {mean_cyl:.2f} dB")
    print(f"Delta (cylinder - full):      {mean_cyl - mean_full:+.2f} dB")
    print(f"\nWei Miao's reported PSNR: ~45 dB. Our whole-volume: {mean_full:.2f} dB "
          f"(gap ~{45 - mean_full:.1f} dB).")
    print("If |delta| above is a big fraction of that gap, cylinder-masking choice "
          "is a real contributor and worth flagging/matching. If delta is small "
          "(a dB or less), this hypothesis is likely NOT the main driver -- rule it "
          "out same as the resolution-model test, and let the Swin UNETR comparison "
          "be the next real signal.")
    print(f"\nSaved per-sample results to {args.out_csv}")


if __name__ == "__main__":
    main()