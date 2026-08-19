#!/bin/bash -l
# scripts/hpc/inference/submit_inference_dump_swin_label_alpha.sh
#
# Same as submit_inference_dump_swin.sh but for the label-x-alpha
# retrained Swin UNETR checkpoint
# (checkpoints/swin_unetr_label_alpha/best_model.pth -- see
# submit_train_swin_label_alpha.sh). Straight rerun of the same
# inference-dump step, pointed at the new checkpoint and a separate
# output dir so it doesn't overwrite logs/denoised/swin_unetr/ from the
# baseline run.
#
# IMPORTANT: same caveat as the U-Net label-alpha inference job -- this
# checkpoint's raw output is trained against label*alpha, not the
# full-scale label. Check the printed quantify summary after running:
# if alpha=0.05's mean RC lands near 0.05 (not near 1), use the RC/alpha
# column instead of raw RC for this checkpoint's leg.
#
# Submit with:  qsub scripts/hpc/inference/submit_inference_dump_swin_label_alpha.sh
# Watch with:   qstat -u $(whoami)
# Then run (on the login node, no GPU needed):
#   python3 scripts/quantify_noisy_baseline.py \
#       --data_dir logs/denoised/swin_unetr_label_alpha --split test \
#       --input_prefix denoised --label_dir data/dataset \
#       --out_csv logs/quant_swin_label_alpha_output_vs_label_test.csv \
#       --per_voi_csv logs/quant_swin_label_alpha_output_vs_label_test_per_voi.csv

#$ -l gpu=1
#$ -l h_rt=0:30:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_inference_dump_swin_labelalpha

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
    --checkpoint checkpoints/swin_unetr_label_alpha/best_model.pth \
    --model swin \
    --out_dir logs/denoised/swin_unetr_label_alpha \
    --input_prefix denoised

echo "Job finished at $(date)"