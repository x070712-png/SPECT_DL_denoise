#!/bin/bash -l
# scripts/submit_inference_earl_v3_bg0_unet.sh
#
# Runs the U-Net XCAT-finetune LABEL-ALPHA checkpoint over the 10-seed
# EARL v3-bg0 dataset (data/earl_dataset_v3_bg0 -- SAME sphere activity as
# v2 (126.457), background set to 0.0 instead of v2's joint-calibrated
# 84.4954, per Stathis's 8/10 finding that v2's background was ~1.5:1
# sphere:background, unrealistic for a real EARL/NEMA IQ phantom). See
# generate_earl_phantom.py's "FIXED MODE" docstring.
#
# Submit with: qsub scripts/submit_inference_earl_v3_bg0_unet.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_unet_infer_earl_v3bg0
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
    --data_dir data/earl_dataset_v3_bg0 \
    --checkpoint checkpoints/3d_unet_xcat_finetune_label_alpha/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet_xcat_labelalpha_earl_v3_bg0 \
    --seeds 42 43 44 45 46 47 48 49 50 51

echo "Job finished at $(date)"