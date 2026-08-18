"""
scripts/run_inference_dump.py

Runs a trained checkpoint over every (phantom_idx, alpha) pair in a given
split and saves the restored count-domain model output as .npy, using the
SAME filename convention as the noisy inputs (input_NNNN.npy) so that
quantify_noisy_baseline.py can be pointed at either one via --input_prefix
without any change to its quantification logic. This is what makes the
"before" (quantify_noisy_baseline.py --input_prefix input) and "after"
(--input_prefix denoised) recovery-coefficient numbers directly comparable
-- identical masks, identical RC formula, only the array being measured
differs.

Same normalisation as train_unet.py / visualize_predictions.py: load npy,
scale = input.mean(), normalise input, forward pass, restore to count
domain by multiplying the output by that same scale.

Needs GPU -- submit via qsub, don't run on the login node (see
submit_inference_dump.sh).

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/run_inference_dump.py \
        --data_dir data/dataset --split val \
        --checkpoint checkpoints/3d_unet/best_model.pth \
        --model unet \
        --out_dir logs/denoised/3d_unet
"""

import argparse
import os

import numpy as np
import torch

from spect.baseline.dataset import build_split
from spect.baseline.model import CustomUNet3D, get_swin_unetr

ALPHAS_ORDERED = ["1p0", "0p5", "0p25", "0p125", "0p05"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                    help="ignored if --phantom_indices is given")
    p.add_argument("--phantom_indices", type=str, default=None,
                    help="comma-separated phantom indices, e.g. '90,91,...,99' -- if given, "
                         "runs inference on these SAME indices at all 5 alphas instead of "
                         "using --split's block-based (index,alpha) pairing. See module "
                         "docstring 'FIXED-PHANTOM MODE'.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="unet", choices=["unet", "swin"],
                    help="which architecture the checkpoint belongs to")
    p.add_argument("--out_dir", type=str, required=True,
                    help="root dir to write alpha_*/input_NNNN.npy denoised volumes into "
                         "(named 'input_' to match quantify_noisy_baseline.py's default "
                         "pattern -- pass --input_prefix denoised there if you'd rather "
                         "rename these; see script docstring)")
    p.add_argument("--input_prefix", type=str, default="denoised",
                    help="filename prefix for the saved output, e.g. 'denoised' for "
                         "denoised_0083.npy -- pass the same value to "
                         "quantify_noisy_baseline.py's --input_prefix")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "unet":
        model = CustomUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    else:
        model = get_swin_unetr(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.phantom_indices:
        indices = [int(x) for x in args.phantom_indices.split(",")]
        pairs = [(idx, a) for idx in indices for a in ALPHAS_ORDERED]
        print(f"[FIXED-PHANTOM MODE] Running inference on {len(indices)} phantom(s) "
              f"{indices} x {len(ALPHAS_ORDERED)} alphas = {len(pairs)} pairs "
              f"(--split={args.split} ignored)")
    else:
        pairs = build_split(args.split)  # [(phantom_idx, alpha_str), ...]
        print(f"Running inference on {len(pairs)} (phantom, alpha) pairs from split={args.split}")

    n_done, n_skipped = 0, 0
    with torch.no_grad():
        for phantom_idx, alpha_str in pairs:
            inp_path = os.path.join(args.data_dir, f"alpha_{alpha_str}", f"input_{phantom_idx:04d}.npy")
            if not os.path.exists(inp_path):
                print(f"[skip] phantom {phantom_idx:04d} alpha_{alpha_str}: missing {inp_path}")
                n_skipped += 1
                continue

            inp = np.load(inp_path).astype(np.float32)
            inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)

            # stage 1 normalisation -- same as SPECTDataset.__getitem__
            scale = inp_t.mean().clamp(min=1e-8)
            inp_n = inp_t / scale

            out_n = model(inp_n)
            out_cnt = (out_n * scale).squeeze(0).squeeze(0).cpu().numpy()

            out_subdir = os.path.join(args.out_dir, f"alpha_{alpha_str}")
            os.makedirs(out_subdir, exist_ok=True)
            out_path = os.path.join(out_subdir, f"{args.input_prefix}_{phantom_idx:04d}.npy")
            np.save(out_path, out_cnt.astype(np.float32))
            n_done += 1

            if n_done % 50 == 0:
                print(f"  {n_done}/{len(pairs)} done")

    print(f"Done. Saved {n_done} denoised volumes to {args.out_dir}/alpha_*/{args.input_prefix}_NNNN.npy "
          f"({n_skipped} skipped due to missing input files).")
    if args.phantom_indices:
        print(f"\nNow run: python3 scripts/quantify_noisy_baseline.py --data_dir {args.out_dir} "
              f"--phantom_indices {args.phantom_indices} --input_prefix {args.input_prefix} "
              f"--label_dir {args.data_dir} --out_csv logs/quant_{args.model}_fixed10_output.csv")
    else:
        print(f"\nNow run: python3 scripts/quantify_noisy_baseline.py --data_dir {args.out_dir} "
              f"--split {args.split} --input_prefix {args.input_prefix} "
              f"--out_csv logs/quant_{args.model}_output.csv")


if __name__ == "__main__":
    main()