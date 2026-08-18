# src/spect/baseline/visualization.py
"""
Visualisation utilities for baseline pipeline outputs.
Saves .png figures to a given directory (no display needed).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as stir


def _middle_slice(img: stir.ImageData) -> np.ndarray:
    arr = img.as_array()
    z = arr.shape[0] // 2
    return arr[z]


def save_all_figures(out, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Phantom (activity image)
    fig, ax = plt.subplots()
    ax.imshow(_middle_slice(out.recon_clean), cmap='hot')
    ax.set_title('Phantom - middle slice')
    ax.axis('off')
    fig.savefig(os.path.join(fig_dir, '01_phantom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Clean sinogram
    fig, ax = plt.subplots()
    sino_arr = out.clean_sino.as_array()[0, 0]  # shape: (views, bins)
    ax.imshow(sino_arr, cmap='hot', aspect='auto')
    ax.set_title('Clean sinogram')
    ax.set_xlabel('Tangential bins')
    ax.set_ylabel('Views')
    fig.savefig(os.path.join(fig_dir, '02_clean_sino.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Noisy sinograms (all alphas in one figure)
    alphas = sorted(out.noisy_sinos.keys())
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 3))
    for ax, a in zip(axes, alphas):
        arr = out.noisy_sinos[a].as_array()[0, 0]
        ax.imshow(arr, cmap='hot', aspect='auto')
        ax.set_title(f'alpha={a}')
        ax.axis('off')
    fig.suptitle('Noisy sinograms')
    fig.savefig(os.path.join(fig_dir, '03_noisy_sinos.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Clean recon vs noisy recons
    n = 1 + len(alphas)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    axes[0].imshow(_middle_slice(out.recon_clean), cmap='hot')
    axes[0].set_title('Recon (clean)')
    axes[0].axis('off')
    for ax, a in zip(axes[1:], alphas):
        ax.imshow(_middle_slice(out.recon_noisy[a]), cmap='hot')
        ax.set_title(f'Recon alpha={a}')
        ax.axis('off')
    fig.suptitle('Reconstructions: clean vs noisy')
    fig.savefig(os.path.join(fig_dir, '04_recons.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'[SAVED] figures -> {fig_dir}')