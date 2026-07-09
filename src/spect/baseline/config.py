# src/spect/baseline/config.py

# Count levels used by Wei Miao
COUNT_LEVELS = [1.0, 0.5, 0.25, 0.125, 0.05]

# Synthetic ellipsoidal dataset parameters
PHANTOM_CONFIG = {
    "num_volumes": 500,
    "volume_shape": (128, 128, 128),
    "voxel_size_mm": 4.42,

    # background activity
    "bg_intensity_range": (0.1, 0.5),

    # random ellipsoids
    "ellipsoid_count_mean": 10,
    "ellipsoid_radius_range": (8, 30),
    "ellipsoid_intensity_range": (1.0, 5.0),

    # cylindrical mask / FOV
    "mask_radius_mm": 180,


    "radii_independent_axes": True,  # true ellipsoids, not spheres
    "allow_overlap": True,  # overlapping permitted
}

# Attenuation map
UMAP_CONFIG = {
    "mu_cm_inv": 0.12,         # uniform attenuation for water at 140 keV
}

# Acquisition / collimator parameters (from Wei Miao's Data128.ipynb)
ACQUISITION_CONFIG = {
    "collimator_sigma": 2.35598,     # used in forward projection only
    "collimator_slope": 0.01771,     # used in forward projection only
    "use_resolution_model_in_recon": False,  # inverse crime prevention
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
    "num_subsets": 2,
    "num_subiterations": 24,
}

# Data augmentation (train split only, disabled for val/test) — core/transforms.py
AUGMENTATION_CONFIG = {
    "rand_flip": {"spatial_axis": [0], "prob": 0.5},
    "rand_rotate90": {"spatial_axes": [1, 2], "max_k": 3, "prob": 0.5},
    "rand_3d_elastic": {"sigma_range": (3, 5), "magnitude_range": (3, 5), "prob": 0.2},
}

# 3D U-Net pre-training hyperparameters — Table 2.8
UNET_TRAINING_CONFIG = {
    "batch_size": 4,
    "num_epochs": 150,
    "early_stop_patience": 6,
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-6,
    },
    "seed": 42,
}

# Loss — 0.5*MSE + 0.5*SSIM;
# window=5 for the training loss (avoids checkerboard artefactsat low counts)
# window=7 only for the reported SSIM metric.
LOSS_CONFIG = {
    "mse_weight": 0.5,
    "ssim_weight": 0.5,
    "ssim_win_size_train": 5,
    "ssim_win_size_eval": 7,
    "ssim_data_range": 1.0,
}