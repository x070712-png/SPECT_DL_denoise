# scripts/print_earl_size_alpha_table.py
"""
Double-grouped (sphere diameter x alpha) RC summary for the EARL quantify
CSVs. The other summary tables in the dissertation each collapse ONE of the
two dimensions (grouping by alpha alone, or by sphere diameter alone).
This script keeps BOTH dimensions, producing a 5 (alpha) x 6 (sphere_mm)
grid, averaged over the 10 noise realisations (seed 42-51) per cell.

Does NOT re-run inference or re-derive anything from the .npy volumes.
It reads the existing per-sphere-per-seed rows already sitting in the
quant_earl_*.csv files (columns: sphere_mm, alpha, alpha_val, seed,
n_voxels, true_val_gt, true_val_label, measured_mean,
recon_rc_label_over_gt, recon_bias_pct, mean_rc, bias_pct,
mean_rc_over_alpha), just re-aggregates with a different groupby key.

Writes two things per input CSV:
  1. A pivot-style table printed to stdout (rows=alpha, cols=sphere_mm).
  2. A long-format CSV (alpha, sphere_mm, n, mean_rc_mean, mean_rc_std,
     mean_rc_over_alpha_mean, mean_rc_over_alpha_std) -- this is also
     exactly the shape of data needed for an "RC vs sphere diameter,
     coloured by alpha" figure, so it doubles as that figure's data
     source.

Usage:
    python3 scripts/print_earl_size_alpha_table.py \
        --csv logs/quant_earl_v3bgr10_unet.csv \
        --out_csv logs/earl_size_alpha_table_bgr10_unet.csv
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True,
                    help="one of the quant_earl_*.csv files, e.g. "
                         "logs/quant_earl_v3bgr10_unet.csv")
    p.add_argument("--out_csv", type=str, default=None,
                    help="where to write the long-format (alpha, sphere_mm, "
                         "...) summary. Default: alongside --csv, suffixed "
                         "'_size_alpha_table.csv'")
    p.add_argument("--value_col", type=str, default="mean_rc",
                    choices=["mean_rc", "mean_rc_over_alpha", "recon_rc_label_over_gt"],
                    help="which column to average per (alpha, sphere_mm) "
                         "cell for the printed pivot table -- the output "
                         "CSV always includes both mean_rc and "
                         "mean_rc_over_alpha regardless of this choice")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_csv is None:
        base, ext = os.path.splitext(args.csv)
        args.out_csv = f"{base}_size_alpha_table{ext}"

    groups = defaultdict(list)  # (alpha, sphere_mm) -> list of row dicts
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["alpha"], row["sphere_mm"])
            groups[key].append(row)

    if not groups:
        raise RuntimeError(f"No rows read from {args.csv} -- check the path/columns")

    alphas = sorted(set(k[0] for k in groups), key=lambda a: -float(a.replace("p", ".")))
    spheres = sorted(set(k[1] for k in groups), key=lambda s: float(s))

    # ---- compute mean/std per cell for both mean_rc and mean_rc_over_alpha ----
    cell_stats = {}
    for key, rows in groups.items():
        mean_rc_vals = [float(r["mean_rc"]) for r in rows]
        rc_over_alpha_vals = [float(r["mean_rc_over_alpha"]) for r in rows]
        cell_stats[key] = {
            "n": len(rows),
            "mean_rc_mean": statistics.mean(mean_rc_vals),
            "mean_rc_std": statistics.pstdev(mean_rc_vals) if len(mean_rc_vals) > 1 else 0.0,
            "rc_over_alpha_mean": statistics.mean(rc_over_alpha_vals),
            "rc_over_alpha_std": statistics.pstdev(rc_over_alpha_vals) if len(rc_over_alpha_vals) > 1 else 0.0,
        }

    # ---- printed pivot table (rows=alpha, cols=sphere_mm) ----
    print(f"\n=== {args.csv}: {args.value_col} by (alpha, sphere_mm), "
          f"averaged over seeds ===")
    header = "alpha".ljust(10) + "".join(f"{s+'mm':>12}" for s in spheres)
    print(header)
    stat_key = {
        "mean_rc": "mean_rc_mean",
        "mean_rc_over_alpha": "rc_over_alpha_mean",
        "recon_rc_label_over_gt": None,  # handled separately below if needed
    }[args.value_col]
    for a in alphas:
        line = a.ljust(10)
        for s in spheres:
            key = (a, s)
            if key in cell_stats:
                line += f"{cell_stats[key][stat_key]:>12.3f}"
            else:
                line += f"{'--':>12}"
        print(line)

    n_header = "n (seeds)".ljust(10) + "".join(f"{s+'mm':>12}" for s in spheres)
    print("\n" + n_header)
    for a in alphas:
        line = a.ljust(10)
        for s in spheres:
            key = (a, s)
            line += f"{cell_stats[key]['n']:>12d}" if key in cell_stats else f"{'--':>12}"
        print(line)

    # ---- long-format output CSV ----
    fieldnames = ["alpha", "sphere_mm", "n", "mean_rc_mean", "mean_rc_std",
                  "mean_rc_over_alpha_mean", "mean_rc_over_alpha_std"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in alphas:
            for s in spheres:
                key = (a, s)
                if key not in cell_stats:
                    continue
                st = cell_stats[key]
                writer.writerow({
                    "alpha": a,
                    "sphere_mm": s,
                    "n": st["n"],
                    "mean_rc_mean": st["mean_rc_mean"],
                    "mean_rc_std": st["mean_rc_std"],
                    "mean_rc_over_alpha_mean": st["rc_over_alpha_mean"],
                    "mean_rc_over_alpha_std": st["rc_over_alpha_std"],
                })
    print(f"\nSaved long-format (alpha, sphere_mm) table to {args.out_csv} "
          f"({len(alphas)} alphas x {len(spheres)} spheres = "
          f"{len(cell_stats)} cells populated)")


if __name__ == "__main__":
    main()