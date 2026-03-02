# src/spect/baseline/acquisition_model.py
"""
Acquisition model + forward projection for SPECT OSEM baseline.

Implements:
- build_ubmatrix_acq_model(): SPECTUBMatrix + attenuation + resolution model
- forward_project(): activity -> clean sinogram (AcquisitionData)

Notes:
- SPECTUBMatrix is sensitive to z-sampling consistency between image and projection data.
  Make sure your activity/uMap are created from a grid derived from templ_sino
  (see phantom_umap.make_phantom_and_umap()).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sirf.STIR as stir


@dataclass
class AcquisitionBundle:
    """
    Container for acquisition objects used later by reconstruction.
    """
    acq_model: stir.AcquisitionModelUsingMatrix
    acq_matrix: stir.SPECTUBMatrix

    @property
    def model(self) -> stir.AcquisitionModelUsingMatrix:
        """Alias for acq_model (so downstream code can use bundle.model)."""
        return self.acq_model


def build_ubmatrix_acq_model(
    templ_sino: stir.AcquisitionData,
    umap: stir.ImageData,
    *,
    resol_slope: float = 0.1,
    resol_sigma0: float = 0.1,
    full_3D: bool = False,
) -> AcquisitionBundle:
    """
    Build an acquisition model using SPECTUBMatrix ray-tracing.

    Args:
        templ_sino: acquisition template (geometry/metadata).
        umap: attenuation image (same grid as activity image).
        resol_slope/resol_sigma0: depth-dependent resolution model parameters.
        full_3D: keep False to match your notebook setup.

    Returns:
        AcquisitionBundle(acq_model, acq_matrix)
    """
    acq_matrix = stir.SPECTUBMatrix()
    acq_matrix.set_attenuation_image(umap)
    acq_matrix.set_resolution_model(resol_slope, resol_sigma0, full_3D=full_3D)

    acq_model = stir.AcquisitionModelUsingMatrix(acq_matrix)
    acq_model.set_up(templ_sino, umap)  # img_templ can be umap; must match image grid
    return AcquisitionBundle(acq_model=acq_model, acq_matrix=acq_matrix)


def forward_project(
    bundle: AcquisitionBundle,
    activity: stir.ImageData,
    templ_sino: stir.AcquisitionData,
    *,
    subset_num: int = 0,
    num_subsets: int = 1,
) -> stir.AcquisitionData:
    """
    Forward project activity image -> clean sinogram.

    Args:
        bundle: output of build_ubmatrix_acq_model().
        activity: activity image (phantom) on the correct grid.
        templ_sino: acquisition template (used to allocate output sinogram).
        subset_num / num_subsets: keep (0,1) for full projection.

    Returns:
        clean_sino: AcquisitionData
    """
    clean_sino = templ_sino.get_uniform_copy()
    bundle.acq_model.forward(activity, subset_num, num_subsets, clean_sino)
    return clean_sino


def show_sino_slice(
    sino: stir.AcquisitionData,
    *,
    view: int = 0,
    seg: int = 0,
    title: str = "Sinogram slice",
):
    """
    Convenience viewer (works in notebooks). Safe to call even if show_2D_array not available.
    """
    try:
        from sirf.STIR import show_2D_array
    except Exception:
        show_2D_array = None

    arr = sino.as_array()
    if show_2D_array is None:
        print(f"{title}: array shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}")
        return

    show_2D_array(title, arr[0, seg, view, :, :])  # (tof, seg, view, axial?, tangential?) depending on template


def smoke_test_acq_and_project(
    templ_sino: stir.AcquisitionData,
    activity: stir.ImageData,
    umap: stir.ImageData,
    *,
    resol_slope: float = 0.1,
    resol_sigma0: float = 0.1,
):
    """
    Minimal end-to-end sanity test:
    - build model
    - forward project
    - print shapes/stats
    """
    bundle = build_ubmatrix_acq_model(
        templ_sino, umap, resol_slope=resol_slope, resol_sigma0=resol_sigma0
    )
    clean = forward_project(bundle, activity, templ_sino)

    arr = clean.as_array()
    print("clean_sino shape:", arr.shape)
    print("clean_sino stats: min=", float(arr.min()), "max=", float(arr.max()), "mean=", float(arr.mean()))
    return bundle, clean