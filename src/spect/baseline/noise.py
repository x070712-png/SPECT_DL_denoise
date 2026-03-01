# src/spect/baseline/noise.py
"""
Noise utilities for SPECT projection data (sinograms).

WeiMiao-style count-level simulation:
- Scale counts: y_scaled = alpha * y_clean
- Poisson sampling: y_noisy ~ Poisson(y_scaled)
- Scale back: y_noisy_rescaled = y_noisy / alpha

Interpretation:
- Larger alpha -> higher counts -> lower relative noise (cleaner).
- Smaller alpha -> lower counts -> higher relative noise (noisier).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import sirf.STIR as stir


def add_noise_alpha(
    clean_proj_data: stir.AcquisitionData,
    alpha: float = 1.0,
    *,
    seed: Optional[int] = None,
    clip_negative: bool = True,
) -> stir.AcquisitionData:
    """
    Add Poisson noise with a count-scaling factor alpha (WeiMiao-style).

    Parameters
    ----------
    clean_proj_data:
        Clean acquisition data (projection domain).
    alpha:
        Count scaling factor. Larger -> cleaner. Must be > 0.
    seed:
        RNG seed for reproducibility.
    clip_negative:
        If True, clip negative values to 0 before Poisson (safety).

    Returns
    -------
    noisy_proj_data:
        Noisy acquisition data (same metadata as input).
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    rng = np.random.default_rng(seed)

    arr = clean_proj_data.as_array().astype(np.float32)
    if clip_negative:
        arr = np.clip(arr, 0, None)

    scaled = arr * float(alpha)
    noisy_counts = rng.poisson(scaled).astype(np.float32)
    noisy_arr = noisy_counts / float(alpha)

    noisy = clean_proj_data.clone()
    noisy.fill(noisy_arr)
    return noisy


def make_noisy_sinos(
    clean_sino: stir.AcquisitionData,
    alphas: Iterable[float],
    *,
    seed: Optional[int] = None,
) -> Dict[float, stir.AcquisitionData]:
    """
    Convenience helper: generate multiple noisy sinograms keyed by alpha.

    Note: if seed is provided, we use a deterministic per-alpha seed so runs
    are reproducible but different alphas don't share identical noise.
    """
    noisy_dict: Dict[float, stir.AcquisitionData] = {}
    for i, a in enumerate(alphas):
        a = float(a)
        this_seed = None if seed is None else int(seed + i * 10007)
        noisy_dict[a] = add_noise_alpha(clean_sino, alpha=a, seed=this_seed)
    return noisy_dict


def sinogram_stats(sino: stir.AcquisitionData) -> Dict[str, float]:
    """Return quick stats for logging/debugging."""
    arr = sino.as_array()
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def smoke_test_noise(
    clean_sino: stir.AcquisitionData,
    alphas=(5.0, 1.0, 0.5, 0.05),
    *,
    seed: int = 0,
) -> None:
    """
    Minimal smoke test: generate noisy sinos and print stats.
    (No plotting; safe for compute nodes.)
    """
    print("clean stats:", sinogram_stats(clean_sino))
    noisy = make_noisy_sinos(clean_sino, alphas, seed=seed)
    for a in sorted(noisy.keys()):
        print(f"alpha={a:g} stats:", sinogram_stats(noisy[a]))