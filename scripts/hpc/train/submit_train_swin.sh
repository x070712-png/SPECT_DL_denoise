#!/bin/bash -l
# scripts/hpc/train/submit_train_swin.sh
#
# GPU job submission script for Myriad (UCL Research Computing, SGE scheduler).
# Adapted from submit_train.sh (U-Net) for train_swin_unetr.py. Based on:
#           https://www.rc.ucl.ac.uk/docs/Example_Jobscripts/#gpu-job-script-example
#           https://www.rc.ucl.ac.uk/docs/Supplementary/GPU_Nodes/
#
# Submit with:  qsub scripts/hpc/train/submit_train_swin.sh
# Monitor with: qstat
#
# NOTE on wallclock time: no per-epoch timing for Swin UNETR yet (unlike
# U-Net, where Wei Miao's reported ~1min15s/epoch on an RTX 5090 gave a
# starting estimate). Swin UNETR is a heavier, transformer-based model with
# a smaller batch_size (2 vs 4), so per-epoch time is unknown until the
# first few epochs actually run on Myriad's hardware -- 48h kept as a
# conservative first guess, same as the U-Net job; check the log after a
# few epochs and adjust for future runs / fine-tuning jobs.
#
# NOTE on memory: use_checkpoint=True (gradient checkpointing) is already
# enabled in get_swin_unetr() to reduce VRAM, and batch_size=2 is smaller
# than U-Net's 4 -- but SwinUNETR (feature_size=48, 3D, 128^3 volumes) is
# still a much heavier model than the U-Net baseline. If this OOMs on VRAM,
# either request a specific higher-memory GPU node (check
# https://www.rc.ucl.ac.uk/docs/Supplementary/GPU_Nodes/ for what's
# available -- A100 80G if offered) or drop --batch_size to 1. Host RAM
# (mem= below) bumped up a bit vs the U-Net job as a precaution since this
# is untested; reduce back down once you've confirmed actual usage.

# Request 1 GPU (any node type first attempt -- see memory note above if it OOMs)
#$ -l gpu=1

# Wallclock time (hours:minutes:seconds) — generous upper bound, early stopping
# will normally finish well before this. Same conservative guess as U-Net job.
#$ -l h_rt=48:00:00

# RAM per core (host RAM, not VRAM). Bumped from U-Net's 16G as a precaution
# for the heavier model -- untested, adjust after seeing real usage.
#$ -l mem=24G

# Local scratch space on the compute node.
#$ -l tmpfs=20G

# Job name.
#$ -N spect_swin_pretrain

#$ -wd /home/ucapiuw/SPECT_DL_denoise


echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

# ---- load modules (same as the U-Net GPU job) ----
module unload gcc-libs
module load pytorch/2.1.0/gpu


export PYTHONPATH=src:$PYTHONPATH

export PYTHONUNBUFFERED=1

# Sanity check that CUDA is visible before committing to a long run.
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# ---- run training ----
# batch_size=2 and lr=5e-5 (not U-Net's 4 / 1e-4) -- matches the reported
# Swin UNETR pretraining hyperparameters, see config.py
# SWIN_UNETR_TRAINING_CONFIG for the full hyperparameter set.
python3 -u train/train_swin_unetr.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/swin_unetr \
    --log_dir logs/swin_unetr \
    --epochs 150 \
    --patience 6 \
    --batch_size 2 \
    --lr 5e-5 \
    --num_workers 1

echo "Job finished at $(date)"