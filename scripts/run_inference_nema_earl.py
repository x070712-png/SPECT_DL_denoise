# scripts/run_inference_nema_earl.py
"""
Run a trained checkpoint (3D U-Net or Swin UNETR, old-method or
label-alpha) over the NEMA/EARL evaluation dataset and dump restored
count-domain outputs, for later quantification with quantify_nema_earl.py
(--input_prefix denoised).

For each (alpha, seed) pair, load input_seed{seed}.npy, apply the SAME stage-1
mean normalisation SPECTDataset.__getitem__ applies (divide by the
volume's own mean) before feeding it to the model, then multiply the
model's output back by that same scale to restore count-domain units --
matching exactly what training/quantify_noisy_baseline.py do, so the
output is directly comparable.

Usage:
    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/run_inference_nema_earl.py \
        --data_dir data/nema_dataset \
        --checkpoint checkpoints/3d_unet/best_model.pth \
        --model unet \
        --out_dir logs/denoised/3d_unet_nema \
        --seeds 42 43 44 45 46

    python3 scripts/run_inference_nema_earl.py \
        --data_dir data/nema_dataset \
        --checkpoint checkpoints/swin_unetr_label_alpha/best_model.pth \
        --model swin \
        --out_dir logs/denoised/swin_label_alpha_nema \
        --seeds 42 43 44 45 46

Then quantify with:
    python3 scripts/quantify_nema_earl.py \
        --activity nema/NEMA_activity.npy \
        --data_dir logs/denoised/3d_unet_nema \
        --sphere_dir nema --sphere_prefix NEMA_sphere_ \
        --input_prefix denoised --seeds 42 43 44 45 46 \
        --out_csv logs/quant_nema_unet_output.csv
"""

import argparse
import os

import numpy as np
import torch

from spect.baseline.config import COUNT_LEVELS

ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                    help="e.g. data/nema_dataset -- reads alpha_*/input_seed{seed}.npy")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, required=True, choices=["unet", "swin"])
    p.add_argument("--out_dir", type=str, required=True,
                    help="writes alpha_*/denoised_seed{seed}.npy here")
    p.add_argument("--alphas", type=float, nargs="+", default=COUNT_LEVELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    return p.parse_args()


def load_model(model_name, checkpoint_path, device):
    if model_name == "unet":
        from spect.baseline.model import CustomUNet3D
        model = CustomUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    else:
        from spect.baseline.model import get_swin_unetr
        model = get_swin_unetr(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model, args.checkpoint, device)
    print(f"Loaded {args.model} checkpoint from {args.checkpoint}")

    n_done, n_skipped, n_missing = 0, 0, 0

    for alpha in args.alphas:
        alpha_str = ALPHA_STR.get(alpha, str(alpha).replace(".", "p"))
        in_dir = os.path.join(args.data_dir, f"alpha_{alpha_str}")
        out_dir = os.path.join(args.out_dir, f"alpha_{alpha_str}")
        os.makedirs(out_dir, exist_ok=True)

        for seed in args.seeds:
            input_path = os.path.join(in_dir, f"input_seed{seed}.npy")
            out_path = os.path.join(out_dir, f"denoised_seed{seed}.npy")

            if os.path.exists(out_path):
                n_skipped += 1
                continue
            if not os.path.exists(input_path):
                print(f"[skip] alpha_{alpha_str} seed={seed}: missing {input_path}")
                n_missing += 1
                continue

            inp = np.load(input_path).astype(np.float32)

            # ---- stage 1 mean normalisation, SAME as SPECTDataset.__getitem__:
            # divide by the volume's own mean before feeding the network ----
            scale = max(float(inp.mean()), 1e-8)
            inp_n = inp / scale

            inp_t = torch.from_numpy(inp_n).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
            with torch.no_grad():
                out_n = model(inp_t)

            # ---- restore count-domain units via the SAME scale used going in ----
            out_cnt = (out_n.squeeze(0).squeeze(0).cpu().numpy() * scale).astype(np.float32)
            np.save(out_path, out_cnt)
            n_done += 1
            print(f"[alpha_{alpha_str}] seed={seed}: done -> {out_path}")

    print(f"\nDone. {n_done} denoised, {n_skipped} already existed, "
          f"{n_missing} missing input files.")


if __name__ == "__main__":
    main()