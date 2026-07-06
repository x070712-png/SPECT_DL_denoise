import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# Wei Miao's setup: each group of 100 phantoms maps to one count level.
GROUP_TO_ALPHA = {
    0: '1p0',   # Group 0 (phantom 0-99)    -> alpha 1.0
    1: '0p5',   # Group 1 (phantom 100-199) -> alpha 0.5
    2: '0p25',  # Group 2 (phantom 200-299) -> alpha 0.25
    3: '0p125', # Group 3 (phantom 300-399) -> alpha 0.125
    4: '0p05',  # Group 4 (phantom 400-499) -> alpha 0.05
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


class SPECTDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)
        self.pairs = build_split(split)
        self.samples = []

        for phantom_idx, alpha_str in self.pairs:
            inp_path = self.data_dir / f"alpha_{alpha_str}" / f"input_{phantom_idx:04d}.npy"
            lbl_path = self.data_dir / f"alpha_{alpha_str}" / f"label_{phantom_idx:04d}.npy"
            if inp_path.exists() and lbl_path.exists():
                self.samples.append((inp_path, lbl_path))

        print(f"[{split}] Dataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp = np.load(self.samples[idx][0]).astype(np.float32)
        lbl = np.load(self.samples[idx][1]).astype(np.float32)

        inp = torch.from_numpy(inp).unsqueeze(0)
        lbl = torch.from_numpy(lbl).unsqueeze(0)

        return inp, lbl