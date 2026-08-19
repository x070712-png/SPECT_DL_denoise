#!/bin/bash -l
# scripts/hpc/visualize/submit_visualize.sh
#
# Quick GPU job to run visualize_predictions.py against the finished U-Net
# checkpoint (checkpoints/3d_unet/best_model.pth, early-stopped epoch 144,
# val_psnr~32.5dB / val_ssim~0.961). Short wallclock since this is just a
# few forward passes, not a training loop.
#
# Submit with:  qsub scripts/hpc/visualize/submit_visualize.sh
# Watch with:   qstat -u $(whoami)
# Result:       logs/3d_unet/qualitative/qualitative_val.png

#$ -l gpu=1
#$ -l h_rt=0:20:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_unet_visualize

#$ -wd /home/ucapiuw/SPECT_DL_denoise

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

module unload gcc-libs
module load pytorch/2.1.0/gpu

export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

python3 scripts/visualize_predictions.py \
    --data_dir data/dataset \
    --checkpoint checkpoints/3d_unet/best_model.pth \
    --out_dir logs/3d_unet/qualitative \
    --split val \
    --num_samples 4

echo "Job finished at $(date)"
echo "Result: logs/3d_unet/qualitative/qualitative_val.png"