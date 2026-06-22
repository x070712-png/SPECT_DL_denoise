# src/spect/baseline/generate_ellipsoids.py

import os
import numpy as np
import matplotlib.pyplot as plt

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
    D, H, W = shape
    radius_vox = radius_mm / pixel_size_mm

    y = np.arange(H) - H / 2
    x = np.arange(W) - W / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")

    mask2d = xx**2 + yy**2 <= radius_vox**2
    return np.broadcast_to(mask2d[None, :, :], shape)


def add_random_ellipsoid(volume, mask, rng, cfg):
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


def generate_phantom(seed=42, cfg=CONFIG):
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
    for _ in range(n):
        add_random_ellipsoid(volume, mask, rng, cfg)

    volume[~mask] = 0
    return volume



def save_preview(volume, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    D, H, W = volume.shape
    slices = [
        volume[D // 2],
        volume[:, H // 2, :],
        volume[:, :, W // 2],
    ]

    titles = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, sl, title in zip(axes, slices, titles):
        im = ax.imshow(sl, cmap="hot", origin="lower")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(base, "../../../outputs/phantoms_500"))
    os.makedirs(out_dir, exist_ok=True)
    for i in range(500):
        vol = generate_phantom(seed=42 + i)
        np.save(f"{out_dir}/phantom_{i:03d}.npy", vol)
        if i < 5:   # only save preview for the first 5 phantoms to avoid clutter
            save_preview(vol, f"{out_dir}/phantom_{i:03d}.png")
        if (i + 1) % 50 == 0:
            print(f"{i+1}/500 done")
    print(f"Saved 500 phantoms to {out_dir}")