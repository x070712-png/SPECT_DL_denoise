# src/spect/baseline/phantom_umap.py
"""
Create a consistent image grid, activity phantom, and attenuation map (uMap)
for SPECT OSEM baseline.

Key idea:
- The SPECTUBMatrix projector is sensitive to z-sampling.
- Therefore we build ONE common image grid from templ_sino, optionally apply a
  zoom (e.g. (0.5,1,1)), and then clone all images from this grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import sirf.STIR as stir

from .baseline_setup import create_sample_image, make_cylindrical_FOV


@dataclass
class PhantomBundle:
    """Convenient container for baseline images."""
    img_grid: stir.ImageData          # geometry template
    activity: stir.ImageData          # phantom image (activity domain)
    umap: stir.ImageData              # attenuation map (mu-map)


def build_common_image_grid(
    templ_sino: stir.AcquisitionData,
    zooms: Optional[Tuple[float, float, float]] = (0.5, 1.0, 1.0),
) -> stir.ImageData:
    """
    Build a single common image grid derived from the acquisition template.

    zooms:
      - If None: no zoom.
      - If tuple: apply zoom_image(zooms=...) once.
    """
    img_grid = templ_sino.create_uniform_image()
    if zooms is not None:
        img_grid = img_grid.zoom_image(zooms=zooms)
    return img_grid


def make_activity_image(
    img_grid: stir.ImageData,
    *,
    use_cyl_fov: bool = True,
) -> stir.ImageData:
    """Create the activity phantom on the provided grid."""
    image = img_grid.clone()
    image.fill(0)
    create_sample_image(image)
    if use_cyl_fov:
        make_cylindrical_FOV(image)
    return image


def make_uniform_umap(
    img_grid: stir.ImageData,
    mu: float = 0.12,
    *,
    use_cyl_fov: bool = True,
) -> stir.ImageData:
    """Create a uniform attenuation map on the provided grid."""
    umap = img_grid.clone()
    umap.fill(mu)
    if use_cyl_fov:
        make_cylindrical_FOV(umap)
    return umap


def make_phantom_and_umap(
    templ_sino: stir.AcquisitionData,
    *,
    zooms: Optional[Tuple[float, float, float]] = (0.5, 1.0, 1.0),
    mu: float = 0.12,
    use_cyl_fov: bool = True,
) -> PhantomBundle:
    """One-call helper that matches the notebook logic."""
    img_grid = build_common_image_grid(templ_sino, zooms=zooms)
    activity = make_activity_image(img_grid, use_cyl_fov=use_cyl_fov)
    umap = make_uniform_umap(img_grid, mu=mu, use_cyl_fov=use_cyl_fov)
    return PhantomBundle(img_grid=img_grid, activity=activity, umap=umap)

import os
from sirf.Utilities import examples_data_path

def load_template_sinogram(data_path: str | None = None,
                           filename: str = "template_sinogram.hs") -> stir.AcquisitionData:
    """
    Load the template sinogram used as acquisition metadata template.
    If data_path is None, use SIRF examples_data_path('SPECT').
    """
    if data_path is None:
        data_path = examples_data_path("SPECT")
    return stir.AcquisitionData(os.path.join(data_path, filename))