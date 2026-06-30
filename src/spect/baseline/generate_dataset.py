# src/spect/baseline/generate_dataset.py

import os
import numpy as np
from pathlib import Path

from src.spect.baseline.config import COUNT_LEVELS, OSEM_CONFIG
from src.spect.baseline.generate_ellipsoids import generate_phantom
from src.spect.baseline.sirf_bridge import (
    load_template_sinogram,
    acquire_data,
    reconstruct_data,
)

TEMPLATE_SINO_PATH = "data/template/temp_sino.hs"
OUT_DIR = "data/dataset"
NUM_PHANTOMS = 500


def generate_pair(phantom_idx: int, templ_sino, out_dir: str):

    phantom = generate_phantom(seed=42 + phantom_idx)
    print(f"[{phantom_idx:04d}] Phantom generated.")

    print(f"[{phantom_idx:04d}] Forward projecting clean sinogram...")
    clean_sino, _ = acquire_data(phantom, templ_sino, alpha=1.0)
    
    print(f"[{phantom_idx:04d}] Reconstructing label...")
    label = reconstruct_data(clean_sino, templ_sino).as_array().astype("float32")
    print(f"[{phantom_idx:04d}] Label done. min={label.min():.4f}, max={label.max():.4f}")

    for alpha in COUNT_LEVELS:
        alpha_str = str(alpha).replace(".", "p")
        pair_dir = os.path.join(out_dir, f"alpha_{alpha_str}")
        os.makedirs(pair_dir, exist_ok=True)

        input_path = os.path.join(pair_dir, f"input_{phantom_idx:04d}.npy")
        label_path = os.path.join(pair_dir, f"label_{phantom_idx:04d}.npy")

        if os.path.exists(input_path) and os.path.exists(label_path):
            print(f"[{phantom_idx:04d}] alpha={alpha} already exists, skipping.")
            continue

        print(f"[{phantom_idx:04d}] alpha={alpha} — forward projecting noisy sinogram...")
        _, noisy_sino = acquire_data(phantom, templ_sino, alpha=alpha)

        print(f"[{phantom_idx:04d}] alpha={alpha} — reconstructing input...")
        inp = reconstruct_data(noisy_sino, templ_sino).as_array().astype("float32")

        np.save(input_path, inp)
        np.save(label_path, label)
        print(f"[{phantom_idx:04d}] alpha={alpha} done. input max={inp.max():.4f}")

    print(f"[{phantom_idx:04d}] ALL DONE.")


def main():
    import sirf.STIR as spect
    spect.MessageRedirector('info.txt', 'warnings.txt', 'errors.txt')

    templ_sino = load_template_sinogram(TEMPLATE_SINO_PATH)
    print(f"Template loaded. Starting dataset generation...")
    print(f"  {NUM_PHANTOMS} phantoms × {len(COUNT_LEVELS)} alpha levels")

    for i in range(NUM_PHANTOMS):
        generate_pair(i, templ_sino, OUT_DIR)

    print("Dataset generation complete.")


if __name__ == "__main__":
    import sys
    import sirf.STIR as spect

    task_idx = int(sys.argv[1])  # 0-based task index
    PHANTOMS_PER_TASK = 5
    start = task_idx * PHANTOMS_PER_TASK
    end = min(start + PHANTOMS_PER_TASK, NUM_PHANTOMS)

    spect.MessageRedirector(
        f'logs/info_{task_idx}.txt',
        f'logs/warnings_{task_idx}.txt',
        f'logs/errors_{task_idx}.txt'
    )

    os.makedirs("logs", exist_ok=True)
    templ_sino = load_template_sinogram(TEMPLATE_SINO_PATH)

    print(f"Starting task {task_idx}: phantoms {start} to {end-1}")
    for phantom_idx in range(start, end):
        print(f"--- Processing phantom {phantom_idx:04d} ---")
        generate_pair(phantom_idx, templ_sino, OUT_DIR)

    print(f"Task {task_idx} complete: phantoms {start}-{end-1}")