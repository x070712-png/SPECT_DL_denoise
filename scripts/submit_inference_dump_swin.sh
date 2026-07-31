#!/bin/bash -l
# scripts/submit_inference_dump_swin.sh
#
# Same as submit_inference_dump.sh but for the Swin UNETR checkpoint
# instead of 3D U-Net -- checkpoints/swin_unetr/best_model.pth confirmed
# present (7/19), so training is done, this is a straight rerun of the
# same inference-dump step with --model swin.
#
# Uses --split test (NOT val) -- matches the val/test leakage fix from the
# 7/27 meeting (val was used for early-stopping/LR scheduling during
# training, so it's not a clean held-out set; test was never touched).
# Do the same for Swin's own quantification -- no reason to repeat the
# val-then-test back-and-forth U-Net went through, go straight to test.
#
# Submit with:  qsub scripts/submit_inference_dump_swin.sh
# Watch with:   qstat -u $(whoami)
# Then run (on the login node, no GPU needed):
#   python3 scripts/quantify_noisy_baseline.py \
#       --data_dir logs/denoised/swin_unetr --split test \
#       --input_prefix denoised \
#       --out_csv logs/quant_swin_output_vs_label_test.csv

#$ -l gpu=1
#$ -l h_rt=0:30:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_inference_dump_swin

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
    --checkpoint checkpoints/swin_unetr/best_model.pth \
    --model swin \
    --out_dir logs/denoised/swin_unetr \
    --input_prefix denoised

echo "Job finished at $(date)"