# src/spect/baseline/sirf_bridge.py
"""
Bridge between our NumPy ellipsoid phantoms and SIRF/STIR.
"""

from __future__ import annotations

import numpy as np
import sirf.STIR as spect

from .config import UMAP_CONFIG, ACQUISITION_CONFIG, OSEM_CONFIG


TEMPLATE_SINO_PATH = "data/template/temp_sino.hs"


def load_template_sinogram(path: str | None = None) -> spect.AcquisitionData:
    """Load the template sinogram defining acquisition geometry."""
    path = path or TEMPLATE_SINO_PATH
    if path is None:
        raise ValueError("No template sinogram path set.")
    return spect.AcquisitionData(path)


def build_image_from_template(templ_sino: spect.AcquisitionData) -> spect.ImageData:
    """Build an empty SIRF ImageData matching the template sinogram dimensions."""
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


def make_uniform_umap(templ_sino: spect.AcquisitionData) -> spect.ImageData:
    """Create a uniform attenuation map (mu = 0.12 cm^-1). """
    umap = build_image_from_template(templ_sino)
    umap.fill(UMAP_CONFIG["mu_cm_inv"])
    return umap

def make_custom_umap(
    templ_sino: spect.AcquisitionData,
    att_map_array: np.ndarray,
) -> spect.ImageData:
    """
    Build an attenuation ImageData from a REAL (non-uniform) attenuation
    map array (e.g. NEMA_att_map.npy, EARL_att_map.npy, or a cropped
    clinical att_map{1,2}.npy) instead of the uniform mu=0.12 used for the
    virtual ellipsoid training phantoms.
    """
    umap = build_image_from_template(templ_sino)
    umap.fill(att_map_array)
    return umap

def make_acquisition_model(
    templ_sino: spect.AcquisitionData,
    image: spect.ImageData,
    use_resolution_model: bool = True,
    umap: spect.ImageData | None = None,
):
    """
    Build the SPECT acquisition model with attenuation and
    (optionally) collimator-detector response.

    Args:
        templ_sino: template sinogram defining geometry.
        image: image template for set_up.
        use_resolution_model: True for forward projection (simulation),
            False for reconstruction (avoid inverse crime).
    """
    ubm = spect.SPECTUBMatrix()

    # Attenuation -- uniform by default (unchanged), custom if given
    if umap is None:
        umap = make_uniform_umap(templ_sino)
    ubm.set_attenuation_image(umap)

    # Collimator-detector response (forward projection only)
    if use_resolution_model:
        ubm.set_resolution_model(
            ACQUISITION_CONFIG["collimator_sigma"],
            ACQUISITION_CONFIG["collimator_slope"],
            full_3D=False,
        )

    acq_model = spect.AcquisitionModelUsingMatrix(ubm)
    acq_model.set_up(templ_sino, image)
    return acq_model


def acquire_data(
    phantom_data: np.ndarray,
    templ_sino: spect.AcquisitionData,
    alpha: float = 1.0,
    umap: spect.ImageData | None = None,
) -> tuple[spect.AcquisitionData, spect.AcquisitionData]:
    """
    Forward-project a phantom to obtain clean and noisy sinograms.

    Noise is applied in the sinogram domain (physically correct):
        y_scaled = alpha * y_clean
        y_noisy  = Poisson(y_scaled)

    Args:
        phantom_data: numpy array (our ellipsoid phantom).
        templ_sino: template AcquisitionData defining geometry.
        alpha: count level scaling factor (Wei Miao's alpha).
        umap: optional custom attenuation map.
    Returns:
        (clean_sinogram, noisy_sinogram)
    """
    image = build_image_from_template(templ_sino)
    acq_model = make_acquisition_model(templ_sino, image, use_resolution_model=True, umap=umap)

    # Fill template image with phantom data
    phantom = image.fill(phantom_data)

    # Forward project to get clean sinogram
    clean_sino = acq_model.forward(phantom)

    # Apply Poisson noise in sinogram domain
    # When uses the global (unseeded) np.random.poisson, not a seeded
    # generator -- unlike generate_ellipsoids.py's phantom shapes, the
    # noise realisation here is NOT reproducible across separate runs of
    # the data-generation pipeline. Doesn't affect anything downstream of
    # the already-generated .npy files (inference/quantification only
    # read those, they never call this function), only "regenerate the
    # dataset from scratch and get bit-identical noise" reproducibility.
    scaled = clean_sino.as_array() * alpha
    noisy_array = np.random.poisson(scaled).astype("float32")
    noisy_sino = clean_sino.clone()
    noisy_sino.fill(noisy_array)

    return clean_sino, noisy_sino


def reconstruct_data(
    sinogram: spect.AcquisitionData,
    templ_sino: spect.AcquisitionData,
    use_resolution_model_recon: bool = False,
    umap: spect.ImageData | None = None,
) -> spect.ImageData:
    """
    OSEM reconstruction from a sinogram.
    No collimator model in reconstruction (avoid inverse crime).
    """
    image = build_image_from_template(templ_sino)
    acq_model = make_acquisition_model(templ_sino, image, use_resolution_model=use_resolution_model_recon, umap=umap)

    obj_fun = spect.make_Poisson_loglikelihood(sinogram)
    obj_fun.set_acquisition_model(acq_model)

    recon = spect.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(OSEM_CONFIG["num_subsets"])
    recon.set_num_subiterations(OSEM_CONFIG["num_subiterations"])

    init_image = image.get_uniform_copy(1)
    recon.set_current_estimate(init_image)
    recon.set_up(init_image)
    recon.process()

    return recon.get_output()


# Quick test
if __name__ == "__main__":
    from .generate_ellipsoids import generate_phantom

    templ_sino = load_template_sinogram()
    print("Template sinogram dimensions:", templ_sino.dimensions())

    phantom = generate_phantom(seed=42)
    print("Phantom shape:", phantom.shape, phantom.dtype)

    clean_sino, noisy_sino = acquire_data(phantom, templ_sino, alpha=1.0)
    print("Forward projection done.")
    print("Clean sinogram shape:", clean_sino.as_array().shape)
    print("Noisy sinogram shape:", noisy_sino.as_array().shape)

    label = reconstruct_data(clean_sino, templ_sino)
    inp = reconstruct_data(noisy_sino, templ_sino)
    print("Reconstruction done.")
    print("Label shape:", label.as_array().shape)
    print("Input shape:", inp.as_array().shape)