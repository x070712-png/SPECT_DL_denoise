# src/spect/baseline/generate_dataset.py

import os
import numpy as np
from pathlib import Path

from .config import COUNT_LEVELS, OSEM_CONFIG
from .generate_ellipsoids import generate_phantom
from .sirf_bridge import (
    load_template_sinogram,
    acquire_data,
    reconstruct_data,
)

TEMPLATE_SINO_PATH = "data/template/temp_sino.hs"
OUT_DIR = "data/dataset"
NUM_PHANTOMS = 500


def generate_pair(phantom_idx: int, templ_sino, out_dir: str):
    """Generate one phantom × all alpha levels → (input, label) pairs."""

    phantom = generate_phantom(seed=42 + phantom_idx)

    # only need to reconstruct the clean sinogram once for the label
    clean_sino, _ = acquire_data(phantom, templ_sino, alpha=1.0)
    label = reconstruct_data(clean_sino, templ_sino).as_array().astype("float32")

    # generate noisy sinograms and reconstructions for each alpha level
    for alpha in COUNT_LEVELS:
        alpha_str = str(alpha).replace(".", "p")
        pair_dir = os.path.join(out_dir, f"alpha_{alpha_str}")
        os.makedirs(pair_dir, exist_ok=True)

        input_path = os.path.join(pair_dir, f"input_{phantom_idx:04d}.npy")
        label_path = os.path.join(pair_dir, f"label_{phantom_idx:04d}.npy")

        # Resume support
        if os.path.exists(input_path) and os.path.exists(label_path):
            print(f"[{phantom_idx:04d}] alpha={alpha} already exists, skipping.")
            continue

        _, noisy_sino = acquire_data(phantom, templ_sino, alpha=alpha)
        inp = reconstruct_data(noisy_sino, templ_sino).as_array().astype("float32")

        np.save(input_path, inp)
        np.save(label_path, label)  # label is the same for all alpha levels

    print(f"[{phantom_idx:04d}] done.")


def main():
    import sirf.STIR as spect
    spect.MessageRedirector('info.txt', 'warnings.txt', 'errors.txt')

    templ_sino = load_template_sinogram(TEMPLATE_SINO_PATH)
    print("Template loaded. Starting test run...")

    for i in range(3):
        generate_pair(i, templ_sino, OUT_DIR)

    print("Test run complete.")


if __name__ == "__main__":
    main()