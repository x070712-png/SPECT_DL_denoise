# scripts/validate_dataset.py
"""
Dataset validation script -- runs a set of sanity checks over whatever
phantoms are present under --data_dir (auto-detected, NOT assumed to be
a fixed count or a fixed set of indices), so this works the same whether
pointed at the full 500-phantom dataset or a smaller submitted sample:

  1. Completeness check -- which phantom indices have all 5 alpha_*
     input files present.
  2. Sample visualisation -- clean label + all 5 noisy inputs, for a
     handful of the phantoms found in step 1.
  3. Sum ratio validation -- does sum(noisy input) / sum(label) track
     the expected alpha scaling factor.
  4. Noise level vs alpha -- does relative noise (input-label)/label
     increase as alpha decreases, as expected.
  5. Label consistency -- label_{idx}.npy should be IDENTICAL across
     every alpha_*/ subfolder for the same phantom_idx (same clean
     reconstruction, only the noisy input differs by alpha).

Usage:
    python3 scripts/validate_dataset.py
    python3 scripts/validate_dataset.py --data_dir data/dataset --n_samples 10
"""

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt

ALPHAS = {'1p0': 1.0, '0p5': 0.5, '0p25': 0.25, '0p125': 0.125, '0p05': 0.05}
ALPHA_STRS = list(ALPHAS.keys())
ALPHA_VALS = list(ALPHAS.values())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset",
                    help="root dir with alpha_*/{input,label}_NNNN.npy -- works "
                         "against the full dataset or a smaller submitted sample, "
                         "since phantom indices are auto-detected, not assumed")
    p.add_argument("--out_dir", type=str, default="logs")
    p.add_argument("--n_samples", type=int, default=10,
                    help="how many phantoms to save qualitative figures for in "
                         "Section 2 -- picked from whichever indices are found "
                         "complete in Section 1 (up to 5 evenly spaced + up to 5 "
                         "random, capped by however many are actually available)")
    p.add_argument("--label_consistency_stride", type=int, default=50,
                    help="check every Nth complete phantom's label consistency "
                         "(Section 5) -- smaller = more thorough but slower")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_complete_phantoms(data_dir, max_scan=1000):
    """Scan alpha_1p0/ for label_NNNN.npy files to discover which phantom
    indices exist at all, then return only those that have ALL 5 alphas'
    input files present too. max_scan bounds how far up the index search
    goes -- 1000 comfortably covers the full 500-phantom dataset with
    headroom, while still terminating quickly against a small sample."""
    complete = []
    for i in range(max_scan):
        label_path = os.path.join(data_dir, "alpha_1p0", f"label_{i:04d}.npy")
        if not os.path.exists(label_path):
            continue
        if all(os.path.exists(os.path.join(data_dir, f"alpha_{a}", f"input_{i:04d}.npy"))
               for a in ALPHA_STRS):
            complete.append(i)
    return complete


def check_completeness(data_dir):
    print("=== Section 1: Completeness Check ===")
    complete = find_complete_phantoms(data_dir)
    print(f"Complete phantoms found: {len(complete)} (indices {complete[0]:04d}-{complete[-1]:04d})"
          if complete else "Complete phantoms found: 0")
    return complete


def save_sample_visualisations(data_dir, out_dir, complete, n_samples, rng):
    print("\n=== Section 2: Sample Visualisation ===")
    if not complete:
        print("  (no complete phantoms to visualise)")
        return

    n_each = max(1, n_samples // 2)
    step = max(1, len(complete) // n_each)
    evenly_spaced = complete[::step][:n_each]
    remaining = [i for i in complete if i not in evenly_spaced]
    random_sample = rng.sample(remaining, min(n_each, len(remaining))) if remaining else []
    sample_indices = sorted(set(evenly_spaced + random_sample))

    for idx in sample_indices:
        lbl = np.load(f"{data_dir}/alpha_1p0/label_{idx:04d}.npy")
        z = lbl.shape[0] // 2

        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        axes[0].imshow(lbl[z], cmap='hot')
        axes[0].set_title('Label (clean)')
        axes[0].axis('off')

        for j, (alpha_str, alpha_val) in enumerate(ALPHAS.items()):
            inp = np.load(f"{data_dir}/alpha_{alpha_str}/input_{idx:04d}.npy")
            axes[j + 1].imshow(inp[z], cmap='hot')
            axes[j + 1].set_title(f'Input α={alpha_val}')
            axes[j + 1].axis('off')

        plt.suptitle(f'Phantom {idx:04d}')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_phantom_{idx:04d}.png", dpi=150)
        plt.close()
        print(f"Saved sample_phantom_{idx:04d}.png")


def validate_sum_ratios(data_dir, out_dir, complete):
    print("\n=== Section 3: Sum Ratio Validation ===")
    if not complete:
        print("  (no complete phantoms to validate)")
        return

    ratios = {a: [] for a in ALPHA_STRS}
    for i in complete:
        lbl = np.load(f"{data_dir}/alpha_1p0/label_{i:04d}.npy")
        lbl_sum = lbl.sum()
        for alpha_str in ALPHA_STRS:
            inp = np.load(f"{data_dir}/alpha_{alpha_str}/input_{i:04d}.npy")
            ratios[alpha_str].append(inp.sum() / lbl_sum)

    print("\nSum ratio stats:")
    for alpha_str, alpha_val in ALPHAS.items():
        r = np.array(ratios[alpha_str])
        print(f"  alpha={alpha_val}: mean={r.mean():.4f}, std={r.std():.4f}, error={abs(r.mean() - alpha_val):.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [np.array(ratios[a]) for a in ALPHA_STRS]
    positions = list(range(1, len(ALPHA_STRS) + 1))
    bp = ax.boxplot(data, positions=positions, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in ALPHA_VALS])
    ax.set_xlabel("Alpha level")
    ax.set_ylabel("Sum ratio (input/label)")
    ax.set_title(f"Sum Ratio Validation -- {len(complete)} Phantoms")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/validation_sum_ratio.png", dpi=150)
    plt.close()
    print("Saved validation_sum_ratio.png")


def validate_noise_level(data_dir, out_dir, complete):
    print("\n=== Section 4: Noise Level vs Alpha ===")
    if not complete:
        print("  (no complete phantoms to validate)")
        return

    noise_levels = {a: [] for a in ALPHA_STRS}
    for i in complete:
        lbl = np.load(f"{data_dir}/alpha_1p0/label_{i:04d}.npy")
        lbl_std = lbl.std()
        if lbl_std == 0:
            continue
        for alpha_str in ALPHA_STRS:
            inp = np.load(f"{data_dir}/alpha_{alpha_str}/input_{i:04d}.npy")
            noise_levels[alpha_str].append((inp - lbl).std() / lbl_std)

    means = [np.mean(noise_levels[a]) for a in ALPHA_STRS]
    stds = [np.std(noise_levels[a]) for a in ALPHA_STRS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ALPHA_VALS, means, yerr=stds, fmt='o-', capsize=5, color='steelblue')
    ax.set_xlabel("Alpha level")
    ax.set_ylabel("Relative noise level")
    ax.set_title(f"Noise Level vs Alpha -- {len(complete)} Phantoms")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/validation_noise_level.png", dpi=150)
    plt.close()
    print("Saved validation_noise_level.png")


def check_label_consistency(data_dir, complete, stride):
    print("\n=== Section 5: Label Consistency ===")
    if not complete:
        print("  (no complete phantoms to check)")
        return

    max_diffs = []
    for i in complete[::stride]:
        labels = [np.load(f"{data_dir}/alpha_{a}/label_{i:04d}.npy") for a in ALPHA_STRS]
        diffs = [abs(labels[0] - l).max() for l in labels[1:]]
        max_diffs.append(max(diffs))

    print(f"Checked {len(max_diffs)} phantom(s) (every {stride}th complete index).")
    print(f"Max label difference across all alpha levels: {max(max_diffs):.6f}")
    print("Label consistency check passed!" if max(max_diffs) == 0 else "WARNING: labels differ!")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    complete = check_completeness(args.data_dir)
    save_sample_visualisations(args.data_dir, args.out_dir, complete, args.n_samples, rng)
    validate_sum_ratios(args.data_dir, args.out_dir, complete)
    validate_noise_level(args.data_dir, args.out_dir, complete)
    check_label_consistency(args.data_dir, complete, args.label_consistency_stride)

    print(f"\nValidation complete! All figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()