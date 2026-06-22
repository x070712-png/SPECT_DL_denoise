# src/spect/baseline/sirf_bridge.py
"""
Bridge between our NumPy ellipsoid phantoms and SIRF/STIR.
"""

from __future__ import annotations

import numpy as np
import sirf.STIR as spect


TEMPLATE_SINO_PATH = "data/template/temp_sino.hs" 


def load_template_sinogram(path: str | None = None) -> spect.AcquisitionData:
    """
    Load the template sinogram that defines acquisition geometry
    (matrix size, num projections, pixel size, orbit radius, etc.)
    If path is None, use TEMPLATE_SINO_PATH (which must be set).
    """
    path = path or TEMPLATE_SINO_PATH
    if path is None:
        raise ValueError(
            "No template sinogram path set. "
            "Fill in TEMPLATE_SINO_PATH once Cate sends the .hs/.s file."
        )
    return spect.AcquisitionData(path)


def build_image_from_template(templ_sino: spect.AcquisitionData) -> spect.ImageData:
    """
    Build an empty SIRF ImageData object whose dimensions match the
    template sinogram (same logic as phantom_umap.py / Cate's notebook).
    """
    image = templ_sino.create_uniform_image()
    image = image.zoom_image(
        zooms=(0.5, 1.0, 1.0),
        size=(
            templ_sino.dimensions()[1],
            templ_sino.dimensions()[3],
            templ_sino.dimensions()[3],
        ),
    )
    return image


def acquire_data(
    phantom_data: np.ndarray,
    templ_sino: spect.AcquisitionData,
    scale_factor: float = 0.5,
) -> tuple[spect.AcquisitionData, spect.AcquisitionData]:
    """
    Forward-project a phantom to obtain a clean and a noisy sinogram.

    Args:
        phantom_data: numpy array (our ellipsoid phantom), must match
            the dimensions derived from templ_sino.
        templ_sino: template AcquisitionData defining the geometry.
        scale_factor: controls Poisson noise level (mimics count level /
            acquisition time). Equivalent to Wei Miao's alpha.

    Returns:
        (true_sinogram, noisy_sinogram)
    """
    acq_model_matrix_sim = spect.SPECTUBMatrix()
    acq_model_sim = spect.AcquisitionModelUsingMatrix(acq_model_matrix_sim)

    image = build_image_from_template(templ_sino)

    acq_model_sim.set_up(templ_sino, image)

    # Fill the template image with our phantom data
    phantom = image.fill(phantom_data)

    # Poisson noise, scaled
    noisy_array = np.random.poisson(phantom.as_array() * scale_factor).astype("float64")
    noisy_phantom = phantom.clone()
    noisy_phantom = noisy_phantom.fill(noisy_array)

    true_sinogram = acq_model_sim.forward(phantom)
    noisy_sinogram = acq_model_sim.forward(noisy_phantom)

    return true_sinogram, noisy_sinogram


def reconstruct_data(
    sinogram: spect.AcquisitionData,
    templ_sino: spect.AcquisitionData,
    num_subsets: int = 2,
    num_subiterations: int = 24,
) -> spect.ImageData:
    """
    OSEM reconstruction from a sinogram.

    Uses a separate acquisition model (without resolution modelling)
    to avoid 'inverse crime'.
    """
    image = build_image_from_template(templ_sino)

    acq_model_matrix_recon = spect.SPECTUBMatrix()
    acq_model_matrix_recon.set_keep_all_views_in_cache(True)
    acq_model_recon = spect.AcquisitionModelUsingMatrix(acq_model_matrix_recon)

    obj_fun = spect.make_Poisson_loglikelihood(sinogram)
    obj_fun.set_acquisition_model(acq_model_recon)

    recon = spect.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiterations)

    init_image = image.get_uniform_copy(1)
    recon.set_current_estimate(init_image)
    recon.set_up(init_image)
    recon.process()

    return recon.get_output()


# Quick smoke test
if __name__ == "__main__":
    import os
    from .generate_ellipsoids import generate_phantom

    if TEMPLATE_SINO_PATH is None:
        print("⚠️  TEMPLATE_SINO_PATH not set yet — nothing to run.")
        print("    Waiting on template sinogram file from Cate.")
    else:
        templ_sino = load_template_sinogram()
        print("Template sinogram dimensions:", templ_sino.dimensions())

        phantom = generate_phantom(seed=42)
        print("Phantom shape:", phantom.shape, phantom.dtype)

        true_sino, noisy_sino = acquire_data(phantom, templ_sino, scale_factor=0.5)
        print("Forward projection done.")

        recon = reconstruct_data(noisy_sino, templ_sino)
        print("Reconstruction done. Output shape:", recon.as_array().shape)