# src/spect/baseline/quantification.py
"""
VOI-based quantification utilities for the SPECT_DL_denoise contribution plan.

Ground-truth VOI masks are derived from the ellipsoid parameters used
to generate each phantom (center/radii from generate_ellipsoids.py), not
from the reconstructed label image. Two consequences:

  1. This works right now, independent of any trained checkpoint — the
     mask-building half of the pipeline doesn't need a model at all.
  2. The "true" activity used for recovery/bias calculations is the
     PHANTOM's designed activity, not the alpha=1.0 reconstruction — the
     stricter of the two candidate ground-truth definitions. This also means any reconstruction-
     level bias (e.g. from the OSEM subset/iteration mismatch or the
     resolution-model-in-recon question flagged when comparing against
     Wei Miao's thesis) will show up here too — which is exactly the kind
     of thing this analysis is meant to catch, not something to work
     around.
"""

import numpy as np

from spect.baseline.generate_ellipsoids import generate_phantom, CONFIG


def get_phantom_ellipsoids(phantom_idx, seed_base=42, cfg=CONFIG):
    """Regenerate phantom `phantom_idx` and return (volume, background,
    ellipsoid_params_list) without touching any files in data/dataset.
    """
    volume, meta = generate_phantom(seed=seed_base + phantom_idx, cfg=cfg, return_params=True)
    return volume, meta["background"], meta["ellipsoids"]


def ellipsoid_mask(shape, center_zyx, radii_zyx):
    """
    Boolean mask (full volume shape) for a single ellipsoid. Mirrors the
    exact bbox + inequality logic in generate_ellipsoids.add_random_ellipsoid
    so the mask lines up voxel-for-voxel with what's actually baked into
    the phantom volume.
    """
    D, H, W = shape
    cz, cy, cx = center_zyx
    rz, ry, rx = radii_zyx

    z0, z1 = max(0, int(cz - rz)), min(D, int(cz + rz) + 1)
    y0, y1 = max(0, int(cy - ry)), min(H, int(cy + ry) + 1)
    x0, x1 = max(0, int(cx - rx)), min(W, int(cx + rx) + 1)

    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1) - cz,
        np.arange(y0, y1) - cy,
        np.arange(x0, x1) - cx,
        indexing="ij",
    )
    inside = (zz / rz) ** 2 + (yy / ry) ** 2 + (xx / rx) ** 2 <= 1

    mask = np.zeros(shape, dtype=bool)
    mask[z0:z1, y0:y1, x0:x1] = inside
    return mask


def build_voi_masks(phantom_idx, shape=(128, 128, 128), seed_base=42, cfg=CONFIG):
    """
    Return (combined_mask, per_voi, background) for one phantom.

    combined_mask : bool ndarray, True wherever ANY ellipsoid is present.
        Use for a single aggregate "all VOIs" recovery number.

    per_voi : list of dicts, one per ellipsoid — each has its own boolean
        mask plus the radii/intensity that generated it. Use this to group
        VOIs by size (e.g. mean_radius_vox) and check whether small
        targets recover worse than large ones — this is the evidence base
        for the class-imbalance loss-function contribution direction.

    background : the uniform background activity value for this phantom.
    """
    _, background, ellipsoids = get_phantom_ellipsoids(phantom_idx, seed_base, cfg)

    combined_mask = np.zeros(shape, dtype=bool)
    per_voi = []
    for e in ellipsoids:
        m = ellipsoid_mask(shape, e["center_zyx"], e["radii_zyx"])
        combined_mask |= m
        per_voi.append({
            "mask": m,
            "radii_zyx": e["radii_zyx"],
            "mean_radius_vox": float(np.mean(e["radii_zyx"])),
            "intensity": e["intensity"],
            "n_voxels": int(m.sum()),
        })

    return combined_mask, per_voi, background


def recovery_stats(volume, mask, true_value):
    """
    Given a reconstructed/denoised volume, a boolean VOI mask, and the
    known true activity for that VOI, return the standard recovery
    metrics used in the contribution plan:

      mean_recovery_coefficient = mean(volume[mask]) / true_value
      max_recovery_coefficient  = max(volume[mask])  / true_value
      bias_pct                  = (mean(volume[mask]) - true_value) / true_value * 100
    """
    if mask.sum() == 0:
        return {"mean_rc": float("nan"), "max_rc": float("nan"), "bias_pct": float("nan"), "n_voxels": 0}

    vals = volume[mask]
    mean_val = float(vals.mean())
    max_val = float(vals.max())

    return {
        "mean_rc": mean_val / true_value,
        "max_rc": max_val / true_value,
        "bias_pct": (mean_val - true_value) / true_value * 100.0,
        "n_voxels": int(mask.sum()),
    }

