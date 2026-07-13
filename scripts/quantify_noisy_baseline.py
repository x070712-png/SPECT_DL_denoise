# scripts/quantify_noisy_baseline.py
"""
"Before training" activity-recovery baseline: how well does the NOISY
INPUT (before any denoising) recover the true activity in each VOI,
compared to what the model output will later be measured against?

Needs no GPU, no checkpoint — data/dataset already has the noisy inputs
on disk. Run on the login node:

    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/quantify_noisy_baseline.py \
        --data_dir data/dataset --split val --out_csv logs/quant_noisy_baseline.csv

Reads phantom_idx/alpha pairs the same way SPECTDataset does (via
build_split), so the CSV lines up with whatever split you point it at.
"""

import argparse
import csv
import os

import numpy as np

from spect.baseline.dataset import build_split
from spect.baseline.quantification import build_voi_masks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--out_csv", type=str, default="logs/quant_noisy_baseline.csv")
    p.add_argument("--seed_base", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    pairs = build_split(args.split)  # [(phantom_idx, alpha_str), ...]

    rows = []
    for phantom_idx, alpha_str in pairs:
        inp_path = os.path.join(args.data_dir, f"alpha_{alpha_str}", f"input_{phantom_idx:04d}.npy")
        if not os.path.exists(inp_path):
            print(f"[skip] phantom {phantom_idx:04d} alpha_{alpha_str}: missing {inp_path}")
            continue

        noisy = np.load(inp_path).astype(np.float32)

        combined_mask, per_voi, background = build_voi_masks(phantom_idx, seed_base=args.seed_base)

        # --- combined (all-VOI) recovery ---
        if combined_mask.sum() > 0:
            # true value for the combined mask: approximate as background +
            # mean ellipsoid intensity (overlap regions are an approximation
            # — see quantification.py note). Fine for a first-pass baseline.
            mean_intensity = float(np.mean([v["intensity"] for v in per_voi])) if per_voi else 0.0
            true_val_combined = background + mean_intensity
            vals = noisy[combined_mask]
            combined_mean_rc = float(vals.mean()) / true_val_combined
            combined_bias_pct = (float(vals.mean()) - true_val_combined) / true_val_combined * 100.0
        else:
            combined_mean_rc, combined_bias_pct = float("nan"), float("nan")

        rows.append({
            "phantom_idx": phantom_idx,
            "alpha": alpha_str,
            "n_voi": len(per_voi),
            "combined_mean_rc": combined_mean_rc,
            "combined_bias_pct": combined_bias_pct,
        })

        # --- per-VOI recovery, so you can group by size later ---
        for i, v in enumerate(per_voi):
            true_val = background + v["intensity"]
            vals = noisy[v["mask"]]
            mean_rc = float(vals.mean()) / true_val
            bias_pct = (float(vals.mean()) - true_val) / true_val * 100.0
            rows[-1].setdefault("per_voi", []).append({
                "voi_idx": i,
                "mean_radius_vox": v["mean_radius_vox"],
                "n_voxels": v["n_voxels"],
                "mean_rc": mean_rc,
                "bias_pct": bias_pct,
            })

        print(f"phantom {phantom_idx:04d} alpha_{alpha_str}: "
              f"combined RC={combined_mean_rc:.3f} bias={combined_bias_pct:+.1f}% "
              f"({len(per_voi)} VOIs)")

    # ---- write a flat summary CSV (combined-level numbers) ----
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phantom_idx", "alpha", "n_voi", "combined_mean_rc", "combined_bias_pct"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # ---- also print an overall summary, grouped by alpha ----
    print("\n=== Summary by alpha (noisy input, before denoising) ===")
    alphas = sorted(set(r["alpha"] for r in rows))
    for a in alphas:
        subset = [r["combined_mean_rc"] for r in rows if r["alpha"] == a and not np.isnan(r["combined_mean_rc"])]
        if subset:
            print(f"  alpha_{a}: mean RC = {np.mean(subset):.3f} (n={len(subset)})")

    print(f"\nSaved {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()