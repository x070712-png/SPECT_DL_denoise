#!/bin/bash -l
# scripts/submit_lr_sweep.sh
#
# LR sensitivity check, all in one job: does training under the corrected
# SaveMean/DivideByScaled (mean-based) normalisation stay noisy at the LR
# copied verbatim from Wei Miao's thesis (1e-4), or does it stabilise at a
# lower LR? Runs all three LR values sequentially in a single job so this
# doesn't turn into three near-duplicate scripts.
#
# 30 epochs each, early stopping disabled (patience=999) so we get the
# full curve shape to compare, not an early-stopped partial one. If one
# arm fails (e.g. OOM, bad LR causing NaN), the loop keeps going to the
# next LR rather than aborting the whole job -- check the per-LR log dir
# to see which ones actually completed.
#
# Submit with:  qsub scripts/submit_lr_sweep.sh
# Watch with:   qstat -u $(whoami)
# Result:       logs/lr_sweep_<lr>/  for each of the LR values below --
#               compare val_psnr/val_ssim curves across the three dirs.

#$ -l gpu=1
#$ -l h_rt=6:00:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -N spect_lr_sweep

#$ -wd /home/ucapiuw/SPECT_DL_denoise

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

module unload gcc-libs
module load pytorch/2.1.0/gpu

export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Add/remove LR values here -- everything else (epochs, batch_size, data_dir)
# stays fixed so LR is the only manipulated variable.
LRS=(1e-4 5e-5 2e-5)

for lr in "${LRS[@]}"; do
    echo ""
    echo "=== Starting run with lr=${lr} at $(date) ==="
    python3 scripts/train_unet.py \
        --data_dir data/dataset \
        --checkpoint_dir "checkpoints/lr_sweep_${lr}" \
        --log_dir "logs/lr_sweep_${lr}" \
        --epochs 30 \
        --patience 999 \
        --batch_size 4 \
        --lr "${lr}" \
        --num_workers 1
    echo "=== Finished run with lr=${lr} at $(date) (exit code $?) ==="
done

echo ""
echo "Job finished at $(date)"
echo "Results: logs/lr_sweep_<lr>/ for lr in ${LRS[*]}"