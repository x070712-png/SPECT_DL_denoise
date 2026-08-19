# src/spect/baseline/generate_ellipsoids.py
"""
Generates one synthetic phantom volume: a uniform cylindrical background
with a random number of overlapping ellipsoids of varying size/intensity
placed inside it. Used as the "ground truth" activity map before forward
projection + reconstruction (see generate_dataset.py, which calls
generate_phantom() as its first step for each phantom).
"""

import numpy as np

CONFIG = {
    "shape": (128, 128, 128),
    "pixel_size_mm": 4.42,
    "mask_radius_mm": 180,
    "bg_range": (0.1, 0.5),
    "n_ellipsoids_mean": 10,
    "radius_range": (8, 30),
    "intensity_range": (1.0, 5.0),
}


def cylindrical_mask(shape, radius_mm, pixel_size_mm):
    """
    Boolean mask (True inside the FOV), a cylinder of the given radius
    running the full length of the D axis, centered on the H/W plane.
    Used both to constrain where ellipsoids can be placed and to zero out
    everything outside the FOV in the finished phantom.
    """
    D, H, W = shape
    radius_vox = radius_mm / pixel_size_mm

    y = np.arange(H) - H / 2
    x = np.arange(W) - W / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")

    mask2d = xx**2 + yy**2 <= radius_vox**2
    return np.broadcast_to(mask2d[None, :, :], shape)


def add_random_ellipsoid(volume, mask, rng, cfg, params_list=None):
    """
    Draws one random ellipsoid (random radii, intensity, and a center
    point resampled up to 100 times until it falls inside the cylindrical
    mask) and adds its intensity to volume in place. If params_list is
    given, appends a dict recording exactly what was drawn (center, radii,
    intensity, bounding box) -- used by quantification.py's
    ellipsoid_mask() to rebuild the same VOI mask later without needing to
    re-run the random draw.
    """
    D, H, W = volume.shape

    rx, ry, rz = rng.uniform(*cfg["radius_range"], size=3)
    intensity = rng.uniform(*cfg["intensity_range"])

    margin = int(max(rx, ry, rz)) + 1

    for _ in range(100):
        cz = rng.integers(margin, D - margin)
        cy = rng.integers(margin, H - margin)
        cx = rng.integers(margin, W - margin)
        if mask[cz, cy, cx]:
            break
    else:
        return

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

    region = volume[z0:z1, y0:y1, x0:x1]
    region[inside] += intensity

    if params_list is not None:
        # NOTE: no rng calls here — purely recording what was already drawn.
        params_list.append({
            "center_zyx": (float(cz), float(cy), float(cx)),
            "radii_zyx": (float(rz), float(ry), float(rx)),
            "intensity": float(intensity),
            "bbox_zyx": ((z0, z1), (y0, y1), (x0, x1)),
        })


def generate_phantom(seed=42, cfg=CONFIG, return_params=False):
    """
    Build one phantom volume: uniform background inside the cylindrical
    FOV, plus a Poisson-random number of overlapping random ellipsoids.
    Deterministic given seed. If return_params is True, also returns the
    background value and the full list of ellipsoid parameters actually
    drawn (needed to rebuild ground-truth VOI masks later without
    re-simulating).
    """
    rng = np.random.default_rng(seed)
 
    mask = cylindrical_mask(
        cfg["shape"],
        cfg["mask_radius_mm"],
        cfg["pixel_size_mm"],
    )
 
    bg = rng.uniform(*cfg["bg_range"])
    volume = np.zeros(cfg["shape"], dtype=np.float32)
    volume[mask] = bg
 
    n = rng.poisson(cfg["n_ellipsoids_mean"])
    print(f"n_ellipsoids = {n}")
 
    params_list = [] if return_params else None
    for _ in range(n):
        add_random_ellipsoid(volume, mask, rng, cfg, params_list=params_list)
 
    volume[~mask] = 0
 
    if return_params:
        return volume, {"background": float(bg), "ellipsoids": params_list}
    return volume