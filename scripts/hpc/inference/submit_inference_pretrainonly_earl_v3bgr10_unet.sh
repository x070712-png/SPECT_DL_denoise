#!/bin/bash -l
# scripts/hpc/inference/submit_inference_pretrainonly_earl_v3bgr10_unet.sh
#
# Runs the ellipsoid-pretrained (NOT XCAT-finetuned) 3D U-Net label-alpha
# checkpoint over the 10-seed EARL v3-bg_ratio10 dataset, so its recovery
# coefficients can be compared against the XCAT-finetuned checkpoint's --
# isolates the effect of the XCAT fine-tuning stage on EARL
# generalisation.
#
# Submit with: qsub scripts/hpc/inference/submit_inference_pretrainonly_earl_v3bgr10_unet.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_unet_infer_pretrainonly_earl_v3bgr10
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
    --data_dir data/earl_dataset_v3_bg_ratio10 \
    --checkpoint checkpoints/3d_unet_label_alpha/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet_pretrainonly_earl_v3_bg_ratio10 \
    --seeds 42 43 44 45 46 47 48 49 50 51

echo "Job finished at $(date)"