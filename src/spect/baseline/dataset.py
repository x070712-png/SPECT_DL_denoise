import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


ALPHA_LEVELS = ['1p0', '0p5', '0p25', '0p125', '0p05']

TRAIN_INDICES = list(range(0, 400))
VAL_INDICES = list(range(400, 450))
TEST_INDICES = list(range(450, 500))


class SPECTDataset(Dataset):
    def __init__(self, data_dir, phantom_indices, alpha_levels=ALPHA_LEVELS):
        self.samples = []
        data_dir = Path(data_dir)

        for idx in phantom_indices:
            for alpha in alpha_levels:
                inp_path = data_dir / f"alpha_{alpha}" / f"input_{idx:04d}.npy"
                lbl_path = data_dir / f"alpha_{alpha}" / f"label_{idx:04d}.npy"
                if inp_path.exists() and lbl_path.exists():
                    self.samples.append((inp_path, lbl_path))

        print(f"Dataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp = np.load(self.samples[idx][0]).astype(np.float32)
        lbl = np.load(self.samples[idx][1]).astype(np.float32)

        inp = torch.from_numpy(inp).unsqueeze(0)
        lbl = torch.from_numpy(lbl).unsqueeze(0)

        return inp, lbl