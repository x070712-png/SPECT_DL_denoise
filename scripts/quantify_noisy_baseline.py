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

def alpha_to_float(alpha_str):
    """Convert a folder-name-safe alpha string like '0p125' back to the
    float count-level (0.125). Reverses the 'p'-for-'.' encoding used
    throughout data/dataset's alpha_* folder names.
"""
    return float(alpha_str.replace("p", "."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--out_csv", type=str, default="logs/quant_noisy_baseline.csv")
    p.add_argument("--per_voi_csv", type=str, default=None,
                    help="where to save the per-ellipsoid breakdown "
                         "(default: same dir as --out_csv, suffixed _per_voi.csv)")
    p.add_argument("--n_size_bins", type=int, default=3,
                    help="number of equal-COUNT size bins (terciles by default) "
                         "for the per-ellipsoid-size RC summary")
    p.add_argument("--seed_base", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.per_voi_csv is None:
        base, ext = os.path.splitext(args.out_csv)
        args.per_voi_csv = f"{base}_per_voi{ext}"

    pairs = build_split(args.split)  # [(phantom_idx, alpha_str), ...]

    rows = []
    per_voi_rows = []
    for phantom_idx, alpha_str in pairs:
        inp_path = os.path.join(args.data_dir, f"alpha_{alpha_str}", f"input_{phantom_idx:04d}.npy")
        if not os.path.exists(inp_path):
            print(f"[skip] phantom {phantom_idx:04d} alpha_{alpha_str}: missing {inp_path}")
            continue

        noisy = np.load(inp_path).astype(np.float32)

        combined_mask, per_voi, background = build_voi_masks(phantom_idx, seed_base=args.seed_base)
        alpha_val = alpha_to_float(alpha_str)

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
            # remove the expected/known count-level scaling -- see
            # alpha_to_float() docstring above for why this matters.
            combined_mean_rc_over_alpha = combined_mean_rc / alpha_val
        else:
            combined_mean_rc, combined_bias_pct = float("nan"), float("nan")
            combined_mean_rc_over_alpha = float("nan")

        rows.append({
            "phantom_idx": phantom_idx,
            "alpha": alpha_str,
            "n_voi": len(per_voi),
            "combined_mean_rc": combined_mean_rc,
            "combined_bias_pct": combined_bias_pct,
            "combined_mean_rc_over_alpha": combined_mean_rc_over_alpha,
        })

        # --- per-VOI recovery, so you can group by size later ---
        for i, v in enumerate(per_voi):
            true_val = background + v["intensity"]
            vals = noisy[v["mask"]]
            mean_rc = float(vals.mean()) / true_val
            bias_pct = (float(vals.mean()) - true_val) / true_val * 100.0
            mean_rc_over_alpha = mean_rc / alpha_val

            per_voi_entry = {
                "voi_idx": i,
                "mean_radius_vox": v["mean_radius_vox"],
                "n_voxels": v["n_voxels"],
                "mean_rc": mean_rc,
                "bias_pct": bias_pct,
                "mean_rc_over_alpha": mean_rc_over_alpha,
            }
            rows[-1].setdefault("per_voi", []).append(per_voi_entry)

            per_voi_rows.append({
                "phantom_idx": phantom_idx,
                "alpha": alpha_str,
                "alpha_val": alpha_val,
                **per_voi_entry,
            })

        print(f"phantom {phantom_idx:04d} alpha_{alpha_str}: "
              f"combined RC={combined_mean_rc:.3f} (RC/alpha={combined_mean_rc_over_alpha:.3f}) "
              f"bias={combined_bias_pct:+.1f}% ({len(per_voi)} VOIs)")

    # ---- write a flat summary CSV (combined-level numbers) ----
    fieldnames = ["phantom_idx", "alpha", "n_voi", "combined_mean_rc",
                  "combined_bias_pct", "combined_mean_rc_over_alpha"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # ---- also print an overall summary, grouped by alpha ----
    print("\n=== Summary by alpha (noisy input, before denoising) ===")
    alphas = sorted(set(r["alpha"] for r in rows))
    for a in alphas:
        subset = [r["combined_mean_rc"] for r in rows if r["alpha"] == a and not np.isnan(r["combined_mean_rc"])]
        subset_norm = [r["combined_mean_rc_over_alpha"] for r in rows
                        if r["alpha"] == a and not np.isnan(r["combined_mean_rc_over_alpha"])]
        if subset:
            print(f"  alpha_{a}: mean RC = {np.mean(subset):.3f}  "
                  f"mean RC/alpha = {np.mean(subset_norm):.3f} (n={len(subset)})")

    all_norm = [r["combined_mean_rc_over_alpha"] for r in rows if not np.isnan(r["combined_mean_rc_over_alpha"])]
    if all_norm:
        print(f"\nRC/alpha across all groups: mean={np.mean(all_norm):.3f}, "
              f"std={np.std(all_norm):.3f}, min={np.min(all_norm):.3f}, max={np.max(all_norm):.3f}")

    print(f"\nSaved {len(rows)} rows to {args.out_csv}")

    # ---- write the flat per-VOI CSV ----
    per_voi_fieldnames = ["phantom_idx", "alpha", "alpha_val", "voi_idx",
                           "mean_radius_vox", "n_voxels", "mean_rc",
                           "bias_pct", "mean_rc_over_alpha"]
    with open(args.per_voi_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_voi_fieldnames)
        writer.writeheader()
        for r in per_voi_rows:
            writer.writerow({k: r[k] for k in per_voi_fieldnames})
    print(f"Saved {len(per_voi_rows)} per-VOI rows to {args.per_voi_csv}")

    # ---- per-ellipsoid RC grouped by size (equal-count bins) ----
    if per_voi_rows:
        radii = np.array([r["mean_radius_vox"] for r in per_voi_rows])
        edges = np.quantile(radii, np.linspace(0, 1, args.n_size_bins + 1))
        edges[-1] += 1e-6  # make the top edge inclusive

        print(f"\n=== Per-ellipsoid RC grouped by size ({args.n_size_bins} "
              f"equal-count bins, radius range {radii.min():.1f}-{radii.max():.1f} vox) ===")
        for b in range(args.n_size_bins):
            lo, hi = edges[b], edges[b + 1]
            in_bin = [r for r in per_voi_rows if lo <= r["mean_radius_vox"] < hi]
            if not in_bin:
                continue
            mean_rc_bin = np.mean([r["mean_rc"] for r in in_bin])
            mean_rc_over_alpha_bin = np.mean([r["mean_rc_over_alpha"] for r in in_bin])
            print(f"  radius [{lo:.1f}, {hi:.1f}) vox: n={len(in_bin):3d}  "
                  f"mean RC={mean_rc_bin:.3f}  mean RC/alpha={mean_rc_over_alpha_bin:.3f}")

        print("\n  -- same bins, broken down by alpha (checks whether the "
              "size effect is stable across noise levels) --")
        alphas = sorted(set(r["alpha"] for r in per_voi_rows))
        for a in alphas:
            line = f"  alpha_{a}: "
            parts = []
            for b in range(args.n_size_bins):
                lo, hi = edges[b], edges[b + 1]
                in_bin = [r for r in per_voi_rows
                          if r["alpha"] == a and lo <= r["mean_radius_vox"] < hi]
                if in_bin:
                    parts.append(f"bin{b}(n={len(in_bin)}) RC/alpha="
                                 f"{np.mean([r['mean_rc_over_alpha'] for r in in_bin]):.3f}")
            print(line + "  ".join(parts))


if __name__ == "__main__":
    main()