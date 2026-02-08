"""
Minimal sanity check for SIRF + STIR environment.

This script verifies:
- SIRF Python module can be imported
- STIR backend is available
- ImageData can be initialised and filled from NumPy

"""

import numpy as np
import sirf.STIR as STIR

# Print STIR version for record
print("STIR version:", STIR.get_STIR_version_string())

# 1. Create an empty ImageData object
img = STIR.ImageData()

# 2. Initialise with image dimensions and voxel size
# NOTE: both arguments must be tuples, not lists
img.initialise((16, 16, 16), (2.0, 2.0, 2.0))
print("ImageData initialised")

# 3. Create a NumPy array (z, y, x ordering for STIR)
arr = np.zeros((16, 16, 16), dtype=np.float32)

# 4. Fill the ImageData object from NumPy
img.fill(arr)
print("ImageData filled from numpy")

# 5. Convert back to NumPy and verify
out = img.as_array()
print("Output shape:", out.shape)
print("Mean value:", out.mean())

print("STIR ImageData minimal test PASSED")