#!/bin/bash -l
# scripts/hpc/inference/submit_inference_earl_unet.sh
#
# Runs the U-Net XCAT-finetune LABEL-ALPHA checkpoint (best-performing
# checkpoint per the ellipsoid + XCAT RC results, and the checkpoint used
# exclusively for NEMA/EARL going forward) over the 10-seed EARL dataset.
# Raw output is uncorrected label*alpha -- same convention as everywhere
# else -- quantify_nema_earl.py's RC/alpha column applies the correction.
#
# Submit with: qsub scripts/hpc/inference/submit_inference_earl_unet.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_unet_infer_earl
#$ -pe smp 1

#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

module unload gcc-libs
module load pytorch/2.1.0/gpu

export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

python3 -u scripts/run_inference_nema_earl.py \
    --data_dir data/earl_dataset \
    --checkpoint checkpoints/3d_unet_xcat_finetune_label_alpha/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet_xcat_labelalpha_earl \
    --seeds 42 43 44 45 46 47 48 49 50 51

echo "Job finished at $(date)"