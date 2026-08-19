#!/bin/bash -l
# scripts/hpc/train/submit_train_label_alpha.sh
#
# GPU job submission script for Myriad (UCL Research Computing, SGE scheduler).
# Same as submit_train.sh (U-Net baseline) but with --scale_label_by_alpha,
# writing to a SEPARATE checkpoint/log dir so the existing baseline
# checkpoint (checkpoints/3d_unet/best_model.pth) is not touched.
#
# Label is now also multiplied by alpha
# after normalisation, so input and label sit on the same scale across all
# alpha groups -- tests whether the network was implicitly learning a
# 1/alpha rescale alongside denoising (see RC_vs_alpha_cutoff plot: CNN
# underperforms a trivial "divide noisy input by alpha" baseline at
# alpha=0.05/0.125, only beats it once alpha >= 0.25).
#
# Submit with:  qsub scripts/hpc/train/submit_train_label_alpha.sh
# Monitor with: qstat
#
# NOTE on wallclock time: same conservative 48h guess as the baseline U-Net
# job -- this experiment doesn't change per-epoch cost (same model, same
# data volume, only the label tensor's scale changes), so reuse whatever
# actual per-epoch time you saw from the original run's log.

#$ -l gpu=1
#$ -l h_rt=48:00:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -pe smp 1

#$ -N spect_unet_pretrain_labelalpha

#$ -wd /home/ucapiuw/SPECT_DL_denoise


echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

module unload gcc-libs
module load pytorch/2.1.0/gpu

export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# ---- run training (label x alpha experiment) ----
# checkpoint_dir/log_dir point at a NEW directory -- does not overwrite
# checkpoints/3d_unet/best_model.pth from the baseline run.
python3 train/train_unet.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/3d_unet_label_alpha \
    --log_dir logs/3d_unet_label_alpha \
    --epochs 150 \
    --patience 6 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_workers 1 \
    --scale_label_by_alpha

echo "Job finished at $(date)"