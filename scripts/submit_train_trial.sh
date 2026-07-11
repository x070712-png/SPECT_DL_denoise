#!/bin/bash -l
# scripts/submit_train_trial.sh
#
# SMOKE TEST — run this before submit_train.sh (the full 150-epoch job).
# Same GPU job as submit_train.sh but with a short wallclock and only
# 2 epochs, so you find out in ~10-15 min (not 2 days) whether:
#   - the GPU actually gets allocated and modules load
#   - your SPECTDataset finds the data and loads without errors
#   - a forward/backward pass runs without OOM
#   - a checkpoint gets written
# If this finishes cleanly, submit the real job with submit_train.sh.
#
# Submit with:  qsub scripts/submit_train_trial.sh
# Watch it with: qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=0:30:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -N spect_unet_trial

#$ -wd /home/ucapiuw/SPECT_DL_denoise

echo "Trial job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

export PYTHONPATH=src

module unload gcc-libs
module load pytorch/2.1.0/gpu

python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# Only 2 epochs, separate checkpoint/log dirs so this doesn't clobber
# a real run's outputs.
python3 scripts/train_unet.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/3d_unet_trial \
    --log_dir logs/3d_unet_trial \
    --epochs 2 \
    --patience 6 \
    --batch_size 4 \
    --lr 1e-4

echo "Trial job finished at $(date)"
echo "If you see a 2-epoch printout above with no errors and a checkpoint"
echo "in checkpoints/3d_unet_trial/, you're good to submit submit_train.sh."
