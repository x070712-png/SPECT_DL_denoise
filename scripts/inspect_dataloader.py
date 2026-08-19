# scripts/inspect_dataloader.py
"""
Quick smoke check for SPECTDataset / DataLoader -- loads train/val/test
splits, pulls one training batch, and prints sample counts + batch shapes
+ value ranges for manual inspection. No assertions -- eyeball the
numbers against what you expect (non-empty splits, matching input/label
shapes, sane min/max ranges) before trusting anything downstream.

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/inspect_dataloader.py
"""

from torch.utils.data import DataLoader
from spect.baseline.dataset import SPECTDataset

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
inp, lbl, scale = next(iter(train_loader))
print(f"\nBatch input shape: {inp.shape}")
print(f"Batch label shape: {lbl.shape}")
print(f"Batch scale shape: {scale.shape}")
print(f"Input  min={inp.min():.4f}, max={inp.max():.4f}")
print(f"Label  min={lbl.min():.4f}, max={lbl.max():.4f}")