#!/bin/bash -l
# scripts/hpc/inference/submit_inference_dump_unet_xcat.sh
#
# Dumps the U-Net XCAT-finetune (OLD method) checkpoint's denoised output
# for the XCAT test split, so quantify_noisy_baseline.py can compute RC on
# it (same pattern as the ellipsoid-dataset dumps in logs/denoised/3d_unet).
#
# Submit with: qsub scripts/hpc/inference/submit_inference_dump_unet_xcat.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=1:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_unet_infer_xcat
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

python3 -u scripts/run_inference_dump.py \
    --data_dir data/xcat_dataset \
    --split test \
    --checkpoint checkpoints/3d_unet_xcat_finetune/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet_xcat_finetune \
    --input_prefix denoised

echo "Job finished at $(date)"