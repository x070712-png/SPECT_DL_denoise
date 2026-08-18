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
import os
 
import numpy as np
 
# Must match GROUP_TO_ALPHA in src/spect/baseline/dataset.py exactly.
GROUP_TO_ALPHA = {
    0: '1p0',   # phantom 0-99    -> alpha 1.0
    1: '0p5',   # phantom 100-199 -> alpha 0.5
    2: '0p25',  # phantom 200-299 -> alpha 0.25
    3: '0p125', # phantom 300-399 -> alpha 0.125
    4: '0p05',  # phantom 400-499 -> alpha 0.05
}
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--label_prefix", type=str, default="label",
                    help="filename prefix for label volumes, e.g. 'label' for "
                         "label_0083.npy (adjust if yours differs)")
    return p.parse_args()
 
 
def actual_used_paths(data_dir, label_prefix):
    """Yield (path, alpha_str, phantom_idx) for exactly the 500 (phantom, alpha)
    pairs SPECTDataset.build_split() would ever select across train+val+test
    combined -- i.e. the real, disjoint, no-repeat set."""
    for group, alpha_str in GROUP_TO_ALPHA.items():
        base = group * 100
        for i in range(100):
            phantom_idx = base + i
            path = os.path.join(data_dir, f"alpha_{alpha_str}", f"{label_prefix}_{phantom_idx:04d}.npy")
            yield path, alpha_str, phantom_idx
 
 
def main():
    args = parse_args()
 
    by_alpha = {}
    all_ratios = []
    missing = []
 
    for path, alpha, phantom_idx in actual_used_paths(args.data_dir, args.label_prefix):
        if not os.path.exists(path):
            missing.append((phantom_idx, alpha))
            continue
        vol = np.load(path)
        mean = vol.mean()
        peak = vol.max()
        if mean <= 1e-8:
            continue  # skip degenerate empty volumes
        ratio = peak / mean
        by_alpha.setdefault(alpha, []).append(ratio)
        all_ratios.append(ratio)
 
    if missing:
        print(f"WARNING: {len(missing)} expected (phantom, alpha) files were missing, e.g. {missing[:5]}\n")
 
    print(f"Found {len(all_ratios)} label volumes actually used by SPECTDataset "
          f"(disjoint, 100 per alpha, no repeats)\n")
 
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