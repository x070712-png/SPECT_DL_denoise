#!/bin/bash -l
# scripts/submit_inference_dump_swin_xcat_labelalpha.sh
#
# Dumps the Swin UNETR XCAT-finetune LABEL-ALPHA checkpoint's denoised
# output for the XCAT test split. Raw output is label*alpha (uncorrected)
# -- same convention as logs/denoised/swin_unetr_label_alpha --
# quantify_noisy_baseline.py's RC/alpha column handles the correction
# downstream, no change needed here.
#
# Submit with: qsub scripts/submit_inference_dump_swin_xcat_labelalpha.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=1:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_swin_infer_xcat_la
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
    --checkpoint checkpoints/swin_unetr_xcat_finetune_label_alpha/best_model.pth \
    --model swin \
    --out_dir logs/denoised/swin_unetr_xcat_finetune_label_alpha \
    --input_prefix denoised

echo "Job finished at $(date)"