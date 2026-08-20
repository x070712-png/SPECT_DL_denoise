# tests/test_visualize_earl_slice_selection.py
"""
Fast, independent assert-based check for find_sphere_slice() in
scripts/visualize_earl_predictions.py . Verifies it actually picks the
z-slice with the most combined sphere-mask coverage, using small
synthetic masks instead of real EARL data (no dependency on the dataset
being generated).

Run from the repo root:
    python3 tests/test_visualize_earl_slice_selection.py
"""

import os
import sys
import shutil
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from visualize_earl_predictions import find_sphere_slice, SPHERE_DIAMETERS_MM, SPHERE_PREFIX


def _make_synthetic_sphere_dir(peak_z, shape=(10, 5, 5)):
    """Write tiny fake EARL_sphere_*mm.npy masks where the combined
    coverage is deliberately made highest at z=peak_z, and 0 elsewhere,
    so the correct answer is known ahead of time."""
    tmp_dir = tempfile.mkdtemp()
    for i, d in enumerate(SPHERE_DIAMETERS_MM):
        mask = np.zeros(shape, dtype=np.int32)
        # first sphere alone marks peak_z as the busiest slice; the rest
        # are empty, so the combined sum is still maximised at peak_z
        if i == 0:
            mask[peak_z] = 1
        np.save(os.path.join(tmp_dir, f"{SPHERE_PREFIX}{d}mm.npy"), mask)
    return tmp_dir


def test_find_sphere_slice_picks_max_coverage_z():
    tmp_dir = _make_synthetic_sphere_dir(peak_z=6)
    try:
        z = find_sphere_slice(tmp_dir)
        assert z == 6, f"expected z=6 (the deliberately-marked peak slice), got z={z}"
    finally:
        shutil.rmtree(tmp_dir)


def test_find_sphere_slice_is_zero_when_all_empty():
    tmp_dir = _make_synthetic_sphere_dir(peak_z=0)  # all-zero masks except z=0
    try:
        z = find_sphere_slice(tmp_dir)
        assert z == 0
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_find_sphere_slice_picks_max_coverage_z()
    test_find_sphere_slice_is_zero_when_all_empty()
    print("All find_sphere_slice() checks passed.")()
    test_find_sphere_slice_is_zero_when_all_empty()
    print("All find_sphere_slice() checks passed.")
