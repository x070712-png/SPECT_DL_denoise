# src/spect/baseline/dataset.py
"""
PyTorch Dataset for the ellipsoid/XCAT SPECT data. Reads pre-generated
(noisy input, clean label) volume pairs from data_dir, organised into
per-alpha subfolders (alpha_1p0/, alpha_0p5/, ...), and applies the
per-volume mean normalisation needed before feeding into the model.

build_split() defines the fixed train/val/test partition and is also
imported directly (without SPECTDataset) by scripts that need the same
(phantom_idx, alpha) pairing without loading the actual arrays, e.g.
run_inference_dump.py, quantify_noisy_baseline.py, quantify_xcat.py,
visualize_predictions.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from monai.transforms import Compose, RandFlipd, RandRotate90d, Rand3DElasticd
 
from spect.baseline.config import AUGMENTATION_CONFIG

# Wei Miao's setup: each group of 100 phantoms maps to one count level.
GROUP_TO_ALPHA = {
    0: '1p0',   # Group 0 (phantom 0-99)    -> alpha 1.0
    1: '0p5',   # Group 1 (phantom 100-199) -> alpha 0.5
    2: '0p25',  # Group 2 (phantom 200-299) -> alpha 0.25
    3: '0p125', # Group 3 (phantom 300-399) -> alpha 0.125
    4: '0p05',  # Group 4 (phantom 400-499) -> alpha 0.05
}

ALPHA_STR_TO_FLOAT = {
    '1p0': 1.0,
    '0p5': 0.5,
    '0p25': 0.25,
    '0p125': 0.125,
    '0p05': 0.05,
}

def build_split(split):
    """
    Return a list of (phantom_idx, alpha_str) pairs for the given split.
    Within each group of 100: 0-79 train, 80-89 val, 90-99 test.
    """
    assert split in ('train', 'val', 'test')
    pairs = []
    for group, alpha_str in GROUP_TO_ALPHA.items():
        base = group * 100
        if split == 'train':
            local = range(0, 80)
        elif split == 'val':
            local = range(80, 90)
        else:  # test
            local = range(90, 100)
        for i in local:
            pairs.append((base + i, alpha_str))
    return pairs

def build_train_augmentation():
    """
    MONAI dict-transform pipeline, applied only to the train split.
    Using passing keys=["input","label"] together is what guarantees 
    the same random flip/rotation/deformation is applied to both 
    input and label.
    """
    flip_cfg = AUGMENTATION_CONFIG["rand_flip"]
    rot_cfg = AUGMENTATION_CONFIG["rand_rotate90"]
    elastic_cfg = AUGMENTATION_CONFIG["rand_3d_elastic"]
 
    return Compose([
        RandFlipd(
            keys=["input", "label"],
            spatial_axis=flip_cfg["spatial_axis"],
            prob=flip_cfg["prob"],
        ),
        RandRotate90d(
            keys=["input", "label"],
            spatial_axes=rot_cfg["spatial_axes"],
            max_k=rot_cfg["max_k"],
            prob=rot_cfg["prob"],
        ),
        Rand3DElasticd(
            keys=["input", "label"],
            sigma_range=elastic_cfg["sigma_range"],
            magnitude_range=elastic_cfg["magnitude_range"],
            prob=elastic_cfg["prob"],
        ),
    ])

class SPECTDataset(Dataset):
    """
    Loads (input, label, scale) triples for one split. Missing or
    truncated files (below 1MB) are skipped with a warning rather than
    raising, so a partial/sample dataset still loads whatever is present
    (see build_split() docstring for how phantom_idx/alpha pairs are
    generated for each split).
    """

    def __init__(self, data_dir, split, scale_label_by_alpha=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.scale_label_by_alpha = scale_label_by_alpha
        self.pairs = build_split(split)
        self.samples = []
 
        skipped = []
        for phantom_idx, alpha_str in self.pairs:
            inp_path = self.data_dir / f"alpha_{alpha_str}" / f"input_{phantom_idx:04d}.npy"
            lbl_path = self.data_dir / f"alpha_{alpha_str}" / f"label_{phantom_idx:04d}.npy"

            if not (inp_path.exists() and lbl_path.exists()):
                skipped.append((phantom_idx, alpha_str, "missing"))
                continue

            if inp_path.stat().st_size < 1_000_000 or lbl_path.stat().st_size < 1_000_000:
                skipped.append((phantom_idx, alpha_str, "truncated/empty"))
                continue

            self.samples.append((inp_path, lbl_path, alpha_str))

        if skipped:
            print(f"[{split}] WARNING: skipped {len(skipped)} corrupted/missing sample(s):")
            for phantom_idx, alpha_str, reason in skipped:
                print(f"    phantom {phantom_idx:04d} (alpha_{alpha_str}): {reason}")
 
        # Augmentation only for train; val/test stay exactly as before
        self.transform = build_train_augmentation() if split == "train" else None
 
        tag = " (with augmentation)" if self.transform is not None else ""
        print(f"[{split}] Dataset loaded: {len(self.samples)} samples{tag}")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        """
        Returns (input, label, scale), all as (1, D, H, W) float32 tensors
        except scale (scalar). Normalisation is two stages, split across
        this codebase:

        Stage 1 (here): mean normalisation. Both input and label are
        divided by the noisy input's own mean, so the model always sees
        inputs of a similar scale regardless of alpha. scale is returned
        so callers can undo this and get back to true count-domain values
        (e.g. train_unet.py, run_inference_dump.py both do
        `x_cnt = x_normalised * scale`).

        Stage 2 (elsewhere, e.g. combined_loss() in train_unet.py): a
        second, peak-based normalisation applied on top of stage 1's
        output, specifically for the loss/metric computation.

        If scale_label_by_alpha is set, the label is additionally scaled
        by the known alpha value -- an alternative training target where
        the model must also learn to predict the count-level scaling
        rather than just the denoised shape.
        """
        inp_path, lbl_path, alpha_str = self.samples[idx]
        inp = np.load(inp_path).astype(np.float32)
        lbl = np.load(lbl_path).astype(np.float32)

        inp = torch.from_numpy(inp).unsqueeze(0)  # (1, D, H, W)
        lbl = torch.from_numpy(lbl).unsqueeze(0)

        scale = inp.mean().clamp(min=1e-8)
        inp = inp / scale
        lbl = lbl / scale

        if self.scale_label_by_alpha:
            lbl = lbl * ALPHA_STR_TO_FLOAT[alpha_str]

        if self.transform is not None:
            data = {"input": inp, "label": lbl}
            data = self.transform(data)
            inp, lbl = data["input"], data["label"]
 
        return inp, lbl, scale