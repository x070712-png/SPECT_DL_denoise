"""
Setup + helper utilities.

What this script does:
- Imports SIRF/STIR and checks the backend is reachable
- Sets up SIRF exercises working directory (optional)
- Creates MessageRedirector to capture STIR output (optional)
- Defines helper functions used in the baseline notebook:
  - create_sample_image(image)
  - make_cylindrical_FOV(image)
  - add_noise(proj_data, noise_factor)

Run (Myriad):
  source ~/scripts/sirf_build/sirf_requirements.sh
  source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh
  python3 scripts/smoke_baseline_setup.py
"""

import os
import numpy as np

# Notebook utilities used by SIRF exercises
try:
    import notebook_setup  # noqa: F401
except Exception:
    notebook_setup = None

import sirf.STIR as spect
#from sirf.Utilities import examples_data_path
#from sirf_exercises import exercises_working_path

# Only used in notebooks for quick plotting
try:
    from sirf.STIR import show_2D_array  # noqa: F401
except Exception:
    show_2D_array = None



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


def main():
    print("STIR version:", spect.get_STIR_version_string())

    # 1) Optionally move into exercises working dir (matches notebook behaviour)
    #data_path, workdir = set_working_dir()
    #print("examples_data_path('SPECT') =", data_path)
    #print("workdir =", workdir)

    # 2) Optional logging (creates info.txt/warnings.txt/errors.txt in workdir)
    _redir = enable_stir_logging()

    # 3) Minimal sanity: create an ImageData and apply helper functions
    img = spect.ImageData()
    img.initialise((16, 16, 16), (2.0, 2.0, 2.0))
    create_sample_image(img)
    make_cylindrical_FOV(img)

    arr = img.as_array()
    print("ImageData OK. shape:", arr.shape, "mean:", float(arr.mean()))
    print("Baseline setup helpers loaded OK.")


if __name__ == "__main__":
    main()