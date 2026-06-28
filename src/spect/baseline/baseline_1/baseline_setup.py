# src/spect/baseline/baseline_setup.py
"""
Setup + helper utilities.

What this script does:
- Defines helper functions used in the baseline notebook:
  - create_sample_image(image)
  - make_cylindrical_FOV(image)
  - add_noise(proj_data, noise_factor)
"""

import os
import numpy as np

import sirf.STIR as spect
#from sirf.Utilities import examples_data_path
#from sirf_exercises import exercises_working_path

def enable_stir_logging(info="info.txt", warnings="warnings.txt", errors="errors.txt"):
    """Redirect STIR output to files (created in current working directory)."""
    return spect.MessageRedirector(info, warnings, errors)


def create_sample_image(image):
    """Fill the image with some simple geometric shapes."""
    image.fill(0)

    shape = spect.EllipticCylinder()
    shape.set_length(400)

    # Shape 1
    shape.set_radii((100, 40))
    shape.set_origin((0, 60, 10))
    image.add_shape(shape, scale=1)

    # Shape 2
    shape.set_radii((30, 30))
    shape.set_origin((60, -30, 10))
    image.add_shape(shape, scale=1.5)

    # Shape 3
    shape.set_origin((-60, -30, 10))
    image.add_shape(shape, scale=0.75)


def make_cylindrical_FOV(image):
    """Truncate to cylindrical FOV (in-place)."""
    cyl_filter = spect.TruncateToCylinderProcessor()
    cyl_filter.apply(image)
    return image


def add_noise(proj_data, noise_factor=1.0, seed=None):
    """Add Poisson noise to acquisition (projection) data."""
    if noise_factor <= 0:
        raise ValueError("noise_factor must be > 0")

    rng = np.random.default_rng(seed)

    proj_data_arr = proj_data.as_array() / float(noise_factor)
    proj_data_arr = np.abs(proj_data_arr)  # just in case

    noisy_arr = rng.poisson(proj_data_arr).astype(np.float32)

    noisy_proj_data = proj_data.clone()
    noisy_proj_data.fill(noisy_arr)
    return noisy_proj_data

