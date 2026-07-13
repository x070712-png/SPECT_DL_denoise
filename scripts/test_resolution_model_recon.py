# scripts/test_resolution_model_recon.py
"""
One-off A/B test: does including the resolution model (collimator/detector
response) in OSEM reconstruction -- Wei Miao's actual approach per his
thesis -- meaningfully change how close the reconstruction gets to the
true phantom activity, compared to our current setup (no resolution model
in recon, to avoid inverse crime)?

Motivated by: after fixing the training-loss normalisation bug, SSIM
converged to ~0.98 (very close to Wei Miao's ~0.998) but PSNR stayed at
~34.5dB (far from his ~45dB) -- a split result that points at something
in the reconstruction/data-generation pipeline rather than the training
loss itself.

Cheap, single-phantom test -- no GPU, no training, just one extra
reconstruction of an already-generated sinogram.

Run on the login node, same SIRF environment used to originally generate
data/dataset (see the array-job script that runs generate_dataset.py):

    cd ~/SPECT_DL_denoise
    source ~/scripts/sirf_build/sirf_requirements.sh
    source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/test_resolution_model_recon.py --phantom_idx 0
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spect.baseline.generate_ellipsoids import generate_phantom
from spect.baseline.sirf_bridge import load_template_sinogram, acquire_data, reconstruct_data


def compute_psnr(pred, tgt, eps=1e-8):
    """Same peak-normalised PSNR definition used everywhere else in this
    project (train_unet.py, quantification.py) -- peak comes from the
    target/true array."""
    peak = tgt.max() + eps
    pred_n, tgt_n = pred / peak, tgt / peak
    mse = np.mean((pred_n - tgt_n) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phantom_idx", type=int, default=0)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="logs/resolution_model_test")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    templ_sino = load_template_sinogram()

    # The true activity map -- not a reconstruction, this is the exact
    # phantom used to generate the sinogram, our best available ground
    # truth for this comparison.
    true_phantom = generate_phantom(seed=args.seed_base + args.phantom_idx)
    print(f"True phantom peak activity: {true_phantom.max():.4f}")

    # alpha=1.0 (clean/noiseless) sinogram -- same as how `label` is made
    # in generate_dataset.py. Forward projection always uses the
    # resolution model (unchanged, that part isn't in question).
    clean_sino, _ = acquire_data(true_phantom, templ_sino, alpha=1.0)

    print("Reconstructing WITHOUT resolution model (current pipeline setting)...")
    recon_no_res = reconstruct_data(clean_sino, templ_sino, use_resolution_model_recon=False).as_array()

    print("Reconstructing WITH resolution model (Wei Miao's actual approach)...")
    recon_with_res = reconstruct_data(clean_sino, templ_sino, use_resolution_model_recon=True).as_array()

    psnr_no_res = compute_psnr(recon_no_res, true_phantom)
    psnr_with_res = compute_psnr(recon_with_res, true_phantom)

    print(f"\nPeak activity -- true phantom:                   {true_phantom.max():.4f}")
    print(f"Peak activity -- recon WITHOUT resolution model: {recon_no_res.max():.4f}")
    print(f"Peak activity -- recon WITH resolution model:    {recon_with_res.max():.4f}")
    print(f"\nPSNR vs true phantom -- WITHOUT resolution model: {psnr_no_res:.2f} dB")
    print(f"PSNR vs true phantom -- WITH resolution model:    {psnr_with_res:.2f} dB")
    print(f"Difference (with minus without): {psnr_with_res - psnr_no_res:+.2f} dB")
    print("\n(If this difference is close to the ~10-11dB gap between our "
          "34.5dB and Wei Miao's ~45dB, that's strong evidence this is the "
          "main remaining cause. If it's small (a couple dB or less), look "
          "elsewhere.)")

    # ---- quick visual comparison, central axial slice ----
    D = true_phantom.shape[0]
    idx = D // 2
    vmax = float(true_phantom[idx].max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(true_phantom[idx], cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("True phantom")
    axes[1].imshow(recon_no_res[idx], cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title(f"Recon, no res. model (current)\nPSNR={psnr_no_res:.2f}")
    axes[2].imshow(recon_with_res[idx], cmap="hot", vmin=0, vmax=vmax)
    axes[2].set_title(f"Recon, with res. model (Wei Miao)\nPSNR={psnr_with_res:.2f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"resolution_model_comparison_phantom{args.phantom_idx:04d}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison figure to {out_path}")


if __name__ == "__main__":
    main()