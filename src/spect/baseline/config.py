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
 
# Normalisation — see notes/normalisation.md (or this block) for the write-up
NORMALIZATION_CONFIG = {
    # CURRENT / correct method — matches Wei Miao exactly. Two stages:
    "method": "two_stage_mean_then_peak",
    "stage_1": {
        "name": "mean_volume_scale",
        "source": "core/transforms.py: SaveMeand + DivideByScaled",
        "scale": "input.mean()  — per-volume mean of the NOISY INPUT only "
                 "(not the label)",
        "applied_to": "both input and label, divided by this same scale, "
                       "BEFORE the network sees them",
        "deployable": True,  # only needs the input — no label required,
                              # so this is computable at real inference time
                              # (resolves the concern Kris raised: label-only
                              # scales like peak-of-label aren't available
                              # when there's no ground truth)
    },
    "stage_2": {
        "name": "combined_loss internal peak renorm",
        "source": "core/metrics.py: combined_loss()",
        "scale": "gt_cnt.amax() per volume — peak of the STAGE-1-NORMALISED "
                 "label, recomputed fresh inside the loss function",
        "applied_to": "pred and gt, only for computing the 0.5*MSE + 0.5*SSIM "
                       "training loss — NOT used for the network's "
                       "input/output scaling, and not needed at inference",
        "deployable": False,  # doesn't matter — training-only, needs labels
                              # by definition (supervised loss)
    },
    "count_domain_metrics": "out_cnt = out_normalised * stage_1_scale, "
                             "lbl_cnt = lbl_normalised * stage_1_scale — "
                             "restores true count-domain values for "
                             "MSE_cnt/PSNR/eval-SSIM using the SAME stage-1 "
                             "(mean) scale used going in",
 
 
    # Candidate future direction — NOT currently used,
    # deprioritised until after reproduction + VOI quantification:
    "future_candidate": {
        "name": "Mean Y (Imraj's paper / Cate's MRes project)",
        "relationship_to_current_method": (
            "Effectively the same core idea as stage_1 above (normalise by "
            "the input's own mean, save it, use it to de-normalise the "
            "prediction) — Cate's version doesn't have Wei Miao's stage_2 "
            "internal peak step for the loss, since her project used a "
            "different loss setup. Not a totally separate direction to "
            "explore later; correctly replicating Wei Miao's method already "
            "gets most of the way there and is already deployment-safe."
        ),
        "open_question_from_meeting": (
            "Stathis noted max-based normalisation can be unstable for "
            "very noisy inputs — worth keeping in mind if peak-based scales "
            "come up again anywhere (e.g. compute_psnr's internal peak "
            "step uses the LABEL's peak, which is fine since it's only "
            "used for evaluation on data where the label is known)."
        ),
    },
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

UNET_MODEL_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 32,
}


# Swin UNETR pre-training hyperparameters
SWIN_UNETR_TRAINING_CONFIG = {
    "batch_size": 2,
    "num_epochs": 150,
    "early_stop_patience": 6,
    "optimizer": "AdamW",
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.8,
        "patience": 3,
        "min_lr": 3e-6,
    },
    "seed": 42,
}
# Swin UNETR model architecture
# Uses MONAI's built-in SwinUNETR directly, no custom architecture.
SWIN_UNETR_MODEL_CONFIG = {
    "img_size": (128, 128, 128),
    "in_channels": 1,
    "out_channels": 1,
    "feature_size": 48,
    "use_checkpoint": True,   # gradient checkpointing, trades compute for memory
    "use_v2": True,
}


# Qualitative visualisation (visualize_predictions.py). vmax_headroom added
VISUALIZATION_CONFIG = {
    "vmin": 0,
    "vmax_headroom": 1.2,   # vmax = label_slice.max() * vmax_headroom
    "shared_scale_across_panels": True,
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