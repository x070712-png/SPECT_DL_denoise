# tests/test_quantify_noisy_baseline_helpers.py
"""
Fast, independent assert-based checks for the pure-logic helpers in
scripts/quantify_noisy_baseline.py.

Run from the repo root:
    python3 tests/test_quantify_noisy_baseline_helpers.py
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))  # quantify_noisy_baseline.py
                                                        # imports `spect...` directly,
                                                        # same as PYTHONPATH=src at runtime
from quantify_noisy_baseline import alpha_to_float, compute_isolation_flags


def test_alpha_to_float():
    assert alpha_to_float("1p0") == 1.0
    assert alpha_to_float("0p125") == 0.125
    assert alpha_to_float("0p05") == 0.05


def test_compute_isolation_flags_detects_overlap():
    # 3 VOIs on a small grid: voi0 and voi1 overlap at one voxel, voi2 is
    # fully separate -- expect [False, False, True]
    m0 = np.zeros((4, 4, 4), dtype=bool)
    m0[0:2, 0:2, 0:2] = True
    m1 = np.zeros((4, 4, 4), dtype=bool)
    m1[1:3, 1:3, 1:3] = True  # overlaps m0 at voxel (1,1,1)
    m2 = np.zeros((4, 4, 4), dtype=bool)
    m2[3, 3, 3] = True  # disjoint from both

    per_voi = [{"mask": m0}, {"mask": m1}, {"mask": m2}]
    flags = compute_isolation_flags(per_voi)
    assert flags == [False, False, True], f"got {flags}"


def test_compute_isolation_flags_all_isolated():
    m0 = np.zeros((4, 4, 4), dtype=bool)
    m0[0, 0, 0] = True
    m1 = np.zeros((4, 4, 4), dtype=bool)
    m1[3, 3, 3] = True
    per_voi = [{"mask": m0}, {"mask": m1}]
    flags = compute_isolation_flags(per_voi)
    assert flags == [True, True], f"got {flags}"


if __name__ == "__main__":
    test_alpha_to_float()
    test_compute_isolation_flags_detects_overlap()
    test_compute_isolation_flags_all_isolated()
    print("All quantify_noisy_baseline helper checks passed.")