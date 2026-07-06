"""
Dataset validation script for full 500-phantom dataset.
Run on Myriad: python3 scripts/validate_dataset.py
Results saved to logs/ as PNG files for notebook embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Auto-detect environment
if os.path.exists('/home/ucapiuw/SPECT_DL_denoise/data/dataset'):
    DATA_DIR = '/home/ucapiuw/SPECT_DL_denoise/data/dataset'
else:
    DATA_DIR = 'data/dataset'

ALPHAS = {'1p0': 1.0, '0p5': 0.5, '0p25': 0.25, '0p125': 0.125, '0p05': 0.05}
ALPHA_STRS = list(ALPHAS.keys())
ALPHA_VALS = list(ALPHAS.values())
N_PHANTOMS = 500

os.makedirs("logs", exist_ok=True)

# Section 1: Completeness check
print("=== Section 1: Completeness Check ===")
completed = [
    i for i in range(N_PHANTOMS)
    if all(os.path.exists(f"{DATA_DIR}/alpha_{a}/input_{i:04d}.npy") for a in ALPHA_STRS)
]
print(f"Complete phantoms: {len(completed)}/{N_PHANTOMS}")

# Section 2: Sample visualisation
import random
random.seed(42)

print("\n=== Section 2: Sample Visualisation ===")

# fixed + random indices for visualisation
fixed_indices = [0, 100, 200, 300, 400]
random_indices = random.sample(range(500), 5)
sample_indices = sorted(set(fixed_indices + random_indices))

for idx in sample_indices:
    lbl = np.load(f"{DATA_DIR}/alpha_1p0/label_{idx:04d}.npy")
    z = lbl.shape[0] // 2

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    axes[0].imshow(lbl[z], cmap='hot')
    axes[0].set_title('Label (clean)')
    axes[0].axis('off')

    for j, (alpha_str, alpha_val) in enumerate(ALPHAS.items()):
        inp = np.load(f"{DATA_DIR}/alpha_{alpha_str}/input_{idx:04d}.npy")
        axes[j+1].imshow(inp[z], cmap='hot')
        axes[j+1].set_title(f'Input α={alpha_val}')
        axes[j+1].axis('off')

    plt.suptitle(f'Phantom {idx:04d}')
    plt.tight_layout()
    plt.savefig(f"logs/sample_phantom_{idx:04d}.png", dpi=150)
    plt.close()
    print(f"Saved sample_phantom_{idx:04d}.png")

# Section 3: Sum ratio validation
print("\n=== Section 3: Sum Ratio Validation ===")
ratios = {a: [] for a in ALPHA_STRS}

for i in range(N_PHANTOMS):
    lbl = np.load(f"{DATA_DIR}/alpha_1p0/label_{i:04d}.npy")
    lbl_sum = lbl.sum()
    for alpha_str in ALPHA_STRS:
        inp = np.load(f"{DATA_DIR}/alpha_{alpha_str}/input_{i:04d}.npy")
        ratios[alpha_str].append(inp.sum() / lbl_sum)

print("\nSum ratio stats:")
for alpha_str, alpha_val in ALPHAS.items():
    r = np.array(ratios[alpha_str])
    print(f"  alpha={alpha_val}: mean={r.mean():.4f}, std={r.std():.4f}, error={abs(r.mean()-alpha_val):.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
data = [ratios[a] for a in ALPHA_STRS]
bp = ax.boxplot(data, labels=[str(v) for v in ALPHA_VALS], patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
ax.set_xlabel("Alpha level")
ax.set_ylabel("Sum ratio (input/label)")
ax.set_title("Sum Ratio Validation — 500 Phantoms")
ax.set_ylim(0, 1.6)
plt.tight_layout()
plt.savefig("logs/validation_sum_ratio.png", dpi=150)
plt.close()
print("Saved validation_sum_ratio.png")

# Section 4: Noise level vs alpha
print("\n=== Section 4: Noise Level vs Alpha ===")
noise_levels = {a: [] for a in ALPHA_STRS}

for i in range(N_PHANTOMS):
    lbl = np.load(f"{DATA_DIR}/alpha_1p0/label_{i:04d}.npy")
    lbl_std = lbl.std()
    if lbl_std == 0:
        continue
    for alpha_str in ALPHA_STRS:
        inp = np.load(f"{DATA_DIR}/alpha_{alpha_str}/input_{i:04d}.npy")
        noise_levels[alpha_str].append((inp - lbl).std() / lbl_std)

means = [np.mean(noise_levels[a]) for a in ALPHA_STRS]
stds = [np.std(noise_levels[a]) for a in ALPHA_STRS]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ALPHA_VALS, means, yerr=stds, fmt='o-', capsize=5, color='steelblue')
ax.set_xlabel("Alpha level")
ax.set_ylabel("Relative noise level")
ax.set_title("Noise Level vs Alpha — 500 Phantoms")
ax.invert_xaxis()
plt.tight_layout()
plt.savefig("logs/validation_noise_level.png", dpi=150)
plt.close()
print("Saved validation_noise_level.png")

# Section 5: Label consistency
print("\n=== Section 5: Label Consistency ===")
max_diffs = []
for i in range(0, N_PHANTOMS, 50):  # check every 50th phantom
    labels = [np.load(f"{DATA_DIR}/alpha_{a}/label_{i:04d}.npy") for a in ALPHA_STRS]
    diffs = [abs(labels[0] - l).max() for l in labels[1:]]
    max_diffs.append(max(diffs))

print(f"Max label difference across all alpha levels: {max(max_diffs):.6f}")
print("Label consistency check passed!" if max(max_diffs) == 0 else "WARNING: labels differ!")

print("\nValidation complete! All figures saved to logs/")