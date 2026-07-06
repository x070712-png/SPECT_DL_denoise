# scripts/test_dataloader.py

import sys
sys.path.insert(0, '.')

from torch.utils.data import DataLoader
from src.spect.baseline.dataset import SPECTDataset

DATA_DIR = "data/dataset"

# Create datasets
train_dataset = SPECTDataset(DATA_DIR, 'train')
val_dataset = SPECTDataset(DATA_DIR, 'val')
test_dataset = SPECTDataset(DATA_DIR, 'test')

print(f"Train: {len(train_dataset)} samples")
print(f"Val:   {len(val_dataset)} samples")
print(f"Test:  {len(test_dataset)} samples")

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load one batch and check shapes
inp, lbl = next(iter(train_loader))
print(f"\nBatch input shape: {inp.shape}")
print(f"Batch label shape: {lbl.shape}")
print(f"Input  min={inp.min():.4f}, max={inp.max():.4f}")
print(f"Label  min={lbl.min():.4f}, max={lbl.max():.4f}")