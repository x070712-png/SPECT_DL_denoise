# scripts/check_peak_mean_ratio.py
"""
Diagnostic for the "mean-normalisation made training noisier" question.

Idea: SaveMean/DivideByScaled normalises each volume by its own MEAN, not its
peak. If the peak-to-mean ratio varies a lot from sample to sample (heavy
tail, e.g. because overlapping ellipsoids create extra-bright local hot
spots), the effective input/target magnitude seen by the network swings a
lot between samples/batches, which is a plausible root cause of the noisier
val curves after switching normalisation.

This script just walks the label .npy files, computes mean/peak/ratio per
volume, and reports the distribution overall and broken down by alpha
(count level), so you can see whether it's wide/heavy-tailed and whether it
differs across count levels.

No model, no GPU needed -- run this directly on the login node.

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/check_peak_mean_ratio.py --data_dir data/dataset --split train
"""

import argparse
import glob
import os
import re

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--split", type=str, default="",
                    help="subfolder name if your dataset is split into train/val/test "
                         "subdirectories on disk; leave as '' (default) if labels live "
                         "directly under data_dir/alpha_*/, which is the case for this "
                         "project -- SPECTDataset does the train/val/test split by index, "
                         "not by physical subfolder")
    p.add_argument("--label_glob", type=str, default="label_*.npy",
                    help="filename pattern for label volumes (adjust if yours differs)")
    return p.parse_args()


def parse_alpha_from_path(path):
    m = re.search(r"alpha_([0-9p]+)", str(path))
    return m.group(1) if m else "unknown"


def main():
    args = parse_args()
    search_root = os.path.join(args.data_dir, args.split) if args.split else args.data_dir
    pattern = os.path.join(search_root, "**", args.label_glob)
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        print(f"No files matched {pattern} -- check --data_dir/--split/--label_glob.")
        return

    print(f"Found {len(paths)} label volumes under {search_root}\n")

    by_alpha = {}
    all_ratios = []

    for path in paths:
        vol = np.load(path)
        mean = vol.mean()
        peak = vol.max()
        if mean <= 1e-8:
            continue  # skip degenerate empty volumes
        ratio = peak / mean
        alpha = parse_alpha_from_path(path)
        by_alpha.setdefault(alpha, []).append(ratio)
        all_ratios.append(ratio)

    def report(name, ratios):
        ratios = np.array(ratios)
        print(f"{name:>12s}  n={len(ratios):4d}  "
              f"mean={ratios.mean():6.2f}  median={np.median(ratios):6.2f}  "
              f"std={ratios.std():6.2f}  min={ratios.min():6.2f}  max={ratios.max():6.2f}")

    print("Peak-to-mean ratio (peak/mean) per volume, by count level:\n")
    for alpha in sorted(by_alpha.keys()):
        report(f"alpha={alpha}", by_alpha[alpha])
    print()
    report("ALL", all_ratios)

    print(
        "\nHow to read this:\n"
        "  - If std is small relative to mean (say < ~20-30% of the mean) and max isn't\n"
        "    wildly larger than the median, the ratio is fairly tight -- mean-normalisation\n"
        "    shouldn't be causing much extra instability on its own, and the LR/scheduler is\n"
        "    the more likely culprit.\n"
        "  - If std/max are large relative to the median (heavy tail), that directly explains\n"
        "    noisier training: some volumes end up with much larger normalised magnitude than\n"
        "    others, so batches are effectively inconsistently scaled.\n"
        "  - Also compare across alpha groups: if low-alpha (noisier, more Poisson noise)\n"
        "    volumes have a different ratio distribution than high-alpha ones, that's worth\n"
        "    noting too, since it could interact with the per-alpha weighted loss."
    )


if __name__ == "__main__":
    main()