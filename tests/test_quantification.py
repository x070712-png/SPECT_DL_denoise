# tests/test_quantification.py
"""
Tests for the RC (recovery coefficient) formulas in
src/spect/baseline/quantification.py. These numbers feed directly into
the reported results, so a silent formula error here (e.g. wrong
numerator/denominator, a missing *100 on bias_pct) would produce
plausible-looking but wrong numbers with no crash -- exactly the kind of
bug worth catching with a test.

How to run: 
export PYTHONPATH=src:$PYTHONPATH
python3 tests/test_quantification.py 2>&1 || pytest tests/test_quantification.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spect.baseline.quantification import recovery_stats, ellipsoid_mask


def test_recovery_stats_matches_hand_calculation():
    # 2x2x2 volume, VOI = first 4 voxels (mean=10, max=13), true_value=10
    volume = np.array([[[9, 10], [11, 12]], [[13, 20], [21, 22]]], dtype=np.float32)
    mask = np.array([[[True, True], [True, True]], [[False, False], [False, False]]])
    true_value = 10.0

    stats = recovery_stats(volume, mask, true_value)

    expected_mean = (9 + 10 + 11 + 12) / 4  # 10.5
    expected_max = 12.0
    assert np.isclose(stats["mean_rc"], expected_mean / true_value)
    assert np.isclose(stats["max_rc"], expected_max / true_value)
    assert np.isclose(stats["bias_pct"], (expected_mean - true_value) / true_value * 100.0)
    assert stats["n_voxels"] == 4


def test_recovery_stats_empty_mask_returns_nan_not_crash():
    volume = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=bool)

    stats = recovery_stats(volume, mask, true_value=5.0)

    assert np.isnan(stats["mean_rc"])
    assert np.isnan(stats["max_rc"])
    assert np.isnan(stats["bias_pct"])
    assert stats["n_voxels"] == 0


def test_ellipsoid_mask_voxel_count_matches_sphere_volume():
    # A sphere (equal radii) of radius 10 voxels, centered in a 40^3 volume.
    shape = (40, 40, 40)
    center = (20, 20, 20)
    radius = 10
    mask = ellipsoid_mask(shape, center, (radius, radius, radius))

    analytical_volume = (4.0 / 3.0) * np.pi * radius ** 3
    voxel_count = mask.sum()

    # Voxel-grid approximation of a sphere; allow generous tolerance since
    # this is a discretisation, not an exact match.
    assert abs(voxel_count - analytical_volume) / analytical_volume < 0.15