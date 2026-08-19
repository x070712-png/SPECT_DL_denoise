#!/bin/bash -l
# scripts/hpc/inference/submit_inference_dump.sh
#
# Runs the trained 3D U-Net checkpoint over the val split and dumps
# restored count-domain model outputs to logs/denoised/3d_unet/, so
# quantify_noisy_baseline.py can measure RC on model OUTPUT the same
# way it already measures RC on the noisy INPUT (before/after comparison
# using identical VOI masks and RC formula -- see run_inference_dump.py
# docstring).
#
# Inference only, no training -- should be quick (a handful of minutes
# for 50 val volumes), but still needs the GPU + pytorch module, so this
# goes through qsub like everything else that touches torch.
#
# Submit with:  qsub scripts/hpc/inference/submit_inference_dump.sh
# Watch with:   qstat -u $(whoami)
# Then run (on the login node, no GPU needed):
#   python3 scripts/quantify_noisy_baseline.py \
#       --data_dir logs/denoised/3d_unet --split val \
#       --input_prefix denoised --out_csv logs/quant_unet_output.csv
 
#$ -l gpu=1
#$ -l h_rt=0:30:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_inference_dump
 
#$ -wd /home/ucapiuw/SPECT_DL_denoise
 
echo "Job started on $(hostname) at $(date)"
nvidia-smi
 
cd /home/ucapiuw/SPECT_DL_denoise
 
module unload gcc-libs
module load pytorch/2.1.0/gpu
 
export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1
 
python3 scripts/run_inference_dump.py \
    --data_dir data/dataset \
    --split test \
    --checkpoint checkpoints/3d_unet/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet \
    --input_prefix denoised
 
echo "Job finished at $(date)"
 
