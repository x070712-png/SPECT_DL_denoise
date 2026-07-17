# scripts/train_swin_unetr.py
"""
Swin UNETR (Wei Miao baseline) pre-training on the 500-ellipsoid dataset.

Adapted from Wei Miao's experiments/Swin_UNETR/pre_training/train.py (read
directly from the SPECT_codes GitHub repo, same as the U-Net port). Same
substitution as train_unet.py: his make_data_list_pre_training() + MONAI
CacheDataset pipeline is replaced with SPECTDataset(data_dir, split), and
the SaveMeand/DivideByScaled normalisation his transform pipeline would
otherwise handle is done explicitly here -- IDENTICAL logic to train_unet.py
(mean_volume_scale from the noisy input + combined_loss's own internal
peak renorm), because both his U-Net and Swin UNETR scripts share the same
core/transforms.py and core/metrics.py.
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt

from monai.losses import SSIMLoss

from spect.baseline.model import get_swin_unetr
from spect.baseline.dataset import SPECTDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset",
                    help="root dir containing the per-alpha subfolders")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/swin_unetr")
    p.add_argument("--log_dir", type=str, default="logs/swin_unetr")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = SPECTDataset(args.data_dir, split="train")
    val_ds = SPECTDataset(args.data_dir, split="val")
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ------------------------------------------------------------------
    # Model / optimiser / scheduler -- hyperparameters match his Swin
    # UNETR script exactly (see module docstring for the diffs vs U-Net)
    # ------------------------------------------------------------------
    model = get_swin_unetr(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3, min_lr=3e-6)

    mse_loss = torch.nn.MSELoss()
    ssim_loss_train = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=5, reduction="mean")
    ssim_metric_eval = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=7, reduction="mean")

    # ---- SAME normalisation as train_unet.py -- his core/transforms.py

    def combined_loss(pred_cnt, gt_cnt, alpha=0.5, eps=1e-8):
        peak = gt_cnt.view(gt_cnt.size(0), -1).amax(dim=1).clamp(min=eps).view(-1, 1, 1, 1, 1)
        pred_norm = pred_cnt / peak
        gt_norm = gt_cnt / peak
        mse_part = mse_loss(pred_norm, gt_norm)
        ssim_part = ssim_loss_train(pred_norm, gt_norm)
        return alpha * mse_part + (1 - alpha) * ssim_part

    def compute_psnr(pred_cnt, tgt_cnt, eps=1e-8):
        peak = tgt_cnt.amax(dim=tuple(range(1, tgt_cnt.dim()))).view(-1, 1, 1, 1, 1) + eps
        pred_n, tgt_n = pred_cnt / peak, tgt_cnt / peak
        mse = torch.mean((pred_n - tgt_n) ** 2, dim=list(range(1, tgt_n.dim())))
        psnr = torch.where(mse > 0, 10.0 * torch.log10(1.0 / mse), torch.full_like(mse, float("inf")))
        return psnr

    # ------------------------------------------------------------------
    # Training loop -- structurally identical to train_unet.py
    # ------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [],
               "val_psnr": [], "val_ssim": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss, running_mse = 0.0, 0.0

        for inp_n, lbl_n, scale in tqdm(train_loader, desc=f"[Epoch {epoch:03d}] train"):
            inp_n, lbl_n = inp_n.to(device), lbl_n.to(device)
            scale = scale.to(device).view(-1, 1, 1, 1, 1)

            optimizer.zero_grad()
            out_n = model(inp_n)
            loss = combined_loss(out_n, lbl_n)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                out_cnt, lbl_cnt = out_n * scale, lbl_n * scale
                running_mse += mse_loss(out_cnt, lbl_cnt).item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_mse = running_mse / len(train_loader)

        # ---- validate ----
        model.eval()
        running_val_loss, running_val_mse = 0.0, 0.0
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for inp_n, lbl_n, scale in tqdm(val_loader, desc=f"[Epoch {epoch:03d}] val"):
                inp_n, lbl_n = inp_n.to(device), lbl_n.to(device)
                scale = scale.to(device).view(-1, 1, 1, 1, 1)

                out_n = model(inp_n)
                loss_v = combined_loss(out_n, lbl_n)
                running_val_loss += loss_v.item()

                out_cnt, lbl_cnt = out_n * scale, lbl_n * scale
                running_val_mse += mse_loss(out_cnt, lbl_cnt).item()

                psnr_list.extend(compute_psnr(out_cnt, lbl_cnt).cpu().tolist())
                peak = lbl_cnt.amax(dim=tuple(range(1, lbl_cnt.dim()))).view(-1, 1, 1, 1, 1) + 1e-8
                ssim_val = 1.0 - ssim_metric_eval(out_cnt / peak, lbl_cnt / peak)
                ssim_list.append(ssim_val.item())

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_mse = running_val_mse / len(val_loader)
        avg_val_psnr = float(np.mean(psnr_list))
        avg_val_ssim = float(np.mean(ssim_list))

        lr_scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_mse"].append(avg_train_mse)
        history["val_mse"].append(avg_val_mse)
        history["val_psnr"].append(avg_val_psnr)
        history["val_ssim"].append(avg_val_ssim)

        print(f"[Epoch {epoch:03d}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} "
              f"val_mse={avg_val_mse:.6e} val_psnr={avg_val_psnr:.3f} val_ssim={avg_val_ssim:.4f}")

        # ---- checkpoint / early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
            print(f"  -> saved new best checkpoint (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ---- final save ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"final_model_{timestamp}.pth"))

    # ---- save metrics + curves (no plt.show() — headless cluster) ----
    np.savez(os.path.join(args.log_dir, "history.npz"), **history)

    epochs_ran = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs_ran, history["train_loss"], label="train")
    axes[0, 0].plot(epochs_ran, history["val_loss"], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()

    axes[0, 1].plot(epochs_ran, history["train_mse"], label="train")
    axes[0, 1].plot(epochs_ran, history["val_mse"], label="val")
    axes[0, 1].set_title("MSE"); axes[0, 1].legend()

    axes[1, 0].plot(epochs_ran, history["val_psnr"])
    axes[1, 0].set_title("Val PSNR")

    axes[1, 1].plot(epochs_ran, history["val_ssim"])
    axes[1, 1].set_title("Val SSIM")

    fig.tight_layout()
    fig.savefig(os.path.join(args.log_dir, "training_curves.png"), dpi=150)
    print(f"Saved history + curves to {args.log_dir}")


if __name__ == "__main__":
    main()