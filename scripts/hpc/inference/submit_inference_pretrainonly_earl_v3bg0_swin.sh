#!/bin/bash -l
# scripts/hpc/inference/submit_inference_pretrainonly_earl_v3bg0_swin.sh
#
# Runs the ellipsoid-pretrained (NOT XCAT-finetuned) Swin UNETR
# label-alpha checkpoint over the 10-seed EARL v3-bg0 dataset, so its
# recovery coefficients can be compared against the XCAT-finetuned
# checkpoint's (submit_inference_earl_v3_bg0_swin.sh) -- isolates the
# effect of the XCAT fine-tuning stage on EARL generalisation.
#
# Submit with: qsub scripts/hpc/inference/submit_inference_pretrainonly_earl_v3bg0_swin.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_swin_infer_pretrainonly_earl_v3bg0
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
    --checkpoint checkpoints/swin_unetr_label_alpha/best_model.pth \
    --model swin \
    --out_dir logs/denoised/swin_pretrainonly_earl_v3_bg0 \
    --seeds 42 43 44 45 46 47 48 49 50 51

echo "Job finished at $(date)"