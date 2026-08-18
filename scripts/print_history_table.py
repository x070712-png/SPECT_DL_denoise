# scripts/print_history_table.py
"""
Print final-epoch and best-epoch val_psnr/val_ssim from one or more
history.npz files, in the same format as Table 4.1 (baseline formulation).

Run on the login node, no torch/GPU needed:
    python3 scripts/print_history_table.py \
        logs/3d_unet_label_alpha/history.npz \
        logs/3d_unet_xcat_finetune_label_alpha/history.npz \
        logs/swin_unetr_label_alpha/history.npz \
        logs/swin_unetr_xcat_finetune_label_alpha/history.npz

If the expected keys (val_psnr / val_ssim, case-insensitive substring match)
aren't found, this prints ALL keys in the file instead of guessing -- rerun
after checking which keys actually hold the metric before trusting any
numbers below.
"""

import sys
import numpy as np


def find_key(keys, substr):
    matches = [k for k in keys if substr.lower() in k.lower()]
    return matches


def summarize(path):
    data = np.load(path)
    keys = list(data.keys())

    psnr_keys = find_key(keys, "psnr")
    ssim_keys = find_key(keys, "ssim")

    # prefer a key that also contains "val"
    def pick(cands):
        val_cands = [k for k in cands if "val" in k.lower()]
        return val_cands[0] if val_cands else (cands[0] if cands else None)

    psnr_key = pick(psnr_keys)
    ssim_key = pick(ssim_keys)

    print(f"\n=== {path} ===")
    print(f"all keys: {keys}")

    if psnr_key is None or ssim_key is None:
        print("[!] could not confidently find val_psnr/val_ssim keys -- "
              "inspect the key list above and rerun with the right names.")
        return

    psnr = np.asarray(data[psnr_key], dtype=float)
    ssim = np.asarray(data[ssim_key], dtype=float)
    n_epochs = len(psnr)

    final_epoch = n_epochs
    final_psnr = psnr[-1]
    final_ssim = ssim[-1]

    best_psnr_idx = int(np.argmax(psnr))
    best_ssim_idx = int(np.argmax(ssim))

    print(f"using psnr key = '{psnr_key}', ssim key = '{ssim_key}'")
    print(f"epochs run        = {n_epochs}")
    print(f"final PSNR / SSIM = {final_psnr:.3f} / {final_ssim:.4f}  (epoch {final_epoch})")
    print(f"best PSNR (epoch) = {psnr[best_psnr_idx]:.3f} (epoch {best_psnr_idx + 1})")
    print(f"best SSIM (epoch) = {ssim[best_ssim_idx]:.4f} (epoch {best_ssim_idx + 1})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 print_history_table.py <history1.npz> [history2.npz ...]")
        sys.exit(1)

    for p in sys.argv[1:]:
        summarize(p)