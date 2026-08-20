# scripts/inspect_dataloader.py
"""
Quick smoke check for SPECTDataset / DataLoader -- loads train/val/test
splits, pulls one batch from the first non-empty split, and prints sample
counts + batch shapes + value ranges for manual inspection. No assertions
-- eyeball the numbers against what you expect (non-empty splits, matching
input/label shapes, sane min/max ranges) before trusting anything
downstream.

Against the FULL 500-phantom data/dataset (see README.md's Data section),
all three splits are non-empty (train=0-79, val=80-89, test=90-99 within
each of the 5 alpha groups -- see build_split()'s docstring). Against only
the sample data package (phantoms 90-99 only), train and val are correctly
EMPTY -- this is expected, not a bug -- so this script pulls its example
batch from the first split that actually has data (test, in that case)
rather than assuming train is always populated.

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/inspect_dataloader.py
"""

from torch.utils.data import DataLoader
from spect.baseline.dataset import SPECTDataset

DATA_DIR = "data/dataset"

# Create datasets
splits = {
    "Train": SPECTDataset(DATA_DIR, 'train'),
    "Val":   SPECTDataset(DATA_DIR, 'val'),
    "Test":  SPECTDataset(DATA_DIR, 'test'),
}

for name, ds in splits.items():
    print(f"{name}: {len(ds)} samples")

# Pull an example batch from the first non-empty split (with only the
# sample data package present, train/val are empty by design -- see
# module docstring -- so don't hardcode "train" here).
example_split = next((name for name, ds in splits.items() if len(ds) > 0), None)
if example_split is None:
    raise SystemExit(
        "All splits are empty -- check --data_dir / DATA_DIR points at a "
        "populated data/dataset (either the full 500-phantom set or the "
        "sample package's phantoms 90-99, see README.md's Data section)."
    )

print(f"\nPulling an example batch from '{example_split}' (first non-empty split)...")
loader = DataLoader(splits[example_split], batch_size=2, shuffle=True)
inp, lbl, scale = next(iter(loader))
print(f"Batch input shape: {inp.shape}")
print(f"Batch label shape: {lbl.shape}")
print(f"Batch scale shape: {scale.shape}")
print(f"Input  min={inp.min():.4f}, max={inp.max():.4f}")
print(f"Label  min={lbl.min():.4f}, max={lbl.max():.4f}")
