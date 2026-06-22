# src/spect/baseline/config.py

# Count levels used by Wei Miao
COUNT_LEVELS = [1.0, 0.5, 0.25, 0.125, 0.05]

# Synthetic ellipsoidal dataset parameters
PHANTOM_CONFIG = {
    "num_volumes": 500,
    "volume_shape": (128, 128, 128),

    # background activity
    "bg_intensity_range": (0.1, 0.5),

    # random ellipsoids
    "ellipsoid_count_mean": 10,
    "ellipsoid_radius_range": (8, 30),
    "ellipsoid_intensity_range": (1.0, 5.0),

    # cylindrical mask / FOV
    "mask_radius_mm": 180,

    # attenuation
    "mu_cm_inv": 0.12,

    # assumptions to confirm
    "radii_independent_axes": True,
    "allow_overlap": True,
}

# Dataset split per count level
SPLIT_PER_ALPHA = {
    "train": 80,
    "val": 10,
    "test": 10,
}

# Output format aligned with Wei Miao
OUTPUT_FORMAT = {
    "input_prefix": "input_",
    "label_prefix": "label_",
    "extension": ".npy",
    "dtype": "float32",
}

# OSEM parameters for reconstruction
OSEM_CONFIG = {
    "num_iterations": 2,
    "num_subsets": 24,
}