#!/bin/bash -l
# scripts/hpc/inference/submit_inference_pretrainonly_swin_xcat.sh
#
# Runs the ellipsoid-pretrained (NOT XCAT-finetuned) Swin UNETR
# label-alpha checkpoint over the XCAT test split, so its recovery
# coefficients can be compared against the XCAT-finetuned checkpoint's
# (submit_inference_dump_swin_xcat_labelalpha.sh) -- isolates the effect
# of the XCAT fine-tuning stage on XCAT test-set performance.
#
# Submit with: qsub scripts/hpc/inference/submit_inference_pretrainonly_swin_xcat.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=0:30:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_infer_pretrainonly_swin_xcat
#$ -wd /home/ucapiuw/SPECT_DL_denoise

echo "Job started on $(hostname) at $(date)"
nvidia-smi
cd /home/ucapiuw/SPECT_DL_denoise
module unload gcc-libs
module load pytorch/2.1.0/gpu
export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

python3 scripts/run_inference_dump.py \
    --data_dir data/xcat_dataset \
    --split test \
    --checkpoint checkpoints/swin_unetr_label_alpha/best_model.pth \
    --model swin \
    --out_dir logs/denoised/swin_unetr_pretrainonly_xcat \
    --input_prefix denoised

echo "Job finished at $(date)"