#!/bin/bash -l
# scripts/hpc/inference/submit_inference_fixed10_unet_old.sh
#
# Same as submit_inference_fixed10_unet_labelalpha.sh but for the
# baseline (labels-only) 3D U-Net checkpoint instead of the label-alpha
# checkpoint -- fixed 10-phantom test subset (indices 90-99), all count
# levels.
#
# Submit with: qsub scripts/hpc/inference/submit_inference_fixed10_unet_old.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=0:30:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_infer_fixed10_unet_old
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
    --phantom_indices 90,91,92,93,94,95,96,97,98,99 \
    --checkpoint checkpoints/3d_unet/best_model.pth \
    --model unet \
    --out_dir logs/denoised/3d_unet_fixed10 \
    --input_prefix denoised

echo "Job finished at $(date)"