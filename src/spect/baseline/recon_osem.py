# src/spect/baseline/recon_osem.py
"""
OSEM reconstruction utilities for SPECT baseline.

Goal:
- Reconstruct an ImageData from a given AcquisitionData sinogram (clean or noisy)
  using STIR's OSMAPOSLReconstructor (OSEM style).

Notes:
- Objective: Poisson log-likelihood
- Uses an AcquisitionModel (e.g. AcquisitionModelUsingMatrix) consistent with forward projection
- Initial image should match geometry (same grid) and typically truncated to cylindrical FOV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import sirf.STIR as stir

from .baseline_setup import make_cylindrical_FOV


@dataclass
class ReconConfig:
    num_subsets: int = 21
    num_subiters: int = 42          # "subiterations" in STIR (e.g. 42 = 2 full iters if subsets=21)
    init_value: float = 1.0
    # Optional: if you want to change resolution modelling inside recon vs proj
    resol_slope: Optional[float] = None
    resol_sigma0: Optional[float] = None
    full_3d: bool = False


def make_objective_function(
    acq_data: stir.AcquisitionData,
    acq_model: stir.AcquisitionModel,
) -> stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData:
    """
    Create Poisson log-likelihood objective and attach acquisition model.
    """
    obj_fun = stir.make_Poisson_loglikelihood(acq_data)
    obj_fun.set_acquisition_model(acq_model)
    return obj_fun


def make_init_image(
    img_template: stir.ImageData,
    init_value: float = 1.0,
    use_cyl_fov: bool = True,
) -> stir.ImageData:
    """
    Create an initial image for iterative reconstruction from an image template.
    """
    x0 = img_template.get_uniform_copy(init_value)
    if use_cyl_fov:
        make_cylindrical_FOV(x0)
    return x0


def osem_reconstruct(
    acq_data: stir.AcquisitionData,
    *,
    acq_model: stir.AcquisitionModel,
    img_template: stir.ImageData,
    config: ReconConfig = ReconConfig(),
    use_cyl_fov: bool = True,
) -> stir.ImageData:
    """
    Run OSEM reconstruction and return reconstructed ImageData.
    """
    # Optionally adjust resolution model inside the same matrix object
    # (Only works if acq_model is AcquisitionModelUsingMatrix and exposes the matrix)
    if (config.resol_slope is not None) and (config.resol_sigma0 is not None):
        # Try to access underlying matrix if present
        try:
            mat = acq_model.get_matrix()  # might not exist depending on wrapper
            mat.set_resolution_model(config.resol_slope, config.resol_sigma0, full_3D=config.full_3d)
        except Exception:
            # Safe fallback: ignore if not supported; keep acq_model as-is
            pass

    obj_fun = make_objective_function(acq_data, acq_model)

    recon = stir.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(config.num_subsets)
    recon.set_num_subiterations(config.num_subiters)

    x0 = make_init_image(img_template, init_value=config.init_value, use_cyl_fov=use_cyl_fov)

    recon.set_up(x0)
    recon.reconstruct(x0)
    out = recon.get_current_estimate()
    return out


def image_stats(img: stir.ImageData) -> Dict[str, float]:
    arr = img.as_array()
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }