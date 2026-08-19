#!/bin/bash -l
# scripts/hpc/train/submit_train_swin_label_alpha.sh
#
# GPU job submission script for Myriad (UCL Research Computing, SGE scheduler).
# Same as submit_train_swin.sh (Swin UNETR baseline) but with
# --scale_label_by_alpha, writing to a SEPARATE checkpoint/log dir so the
# existing baseline checkpoint (checkpoints/swin_unetr/best_model.pth) is
# not touched. Requires the validate-loop fix in train/train_swin_unetr.py
# (mean_volume_scale matching SPECTDataset's (inp, lbl, scale) return) --
# without that fix this job will crash at the first validation epoch
# regardless of this flag.
#
# Label is now also multiplied by alpha
# after normalisation, so input and label sit on the same scale across all
# alpha groups -- tests whether the network was implicitly learning a
# 1/alpha rescale alongside denoising, same experiment as the U-Net job
# (submit_train_label_alpha.sh).
#
# Submit with:  qsub scripts/hpc/train/submit_train_swin_label_alpha.sh
# Monitor with: qstat
#
# NOTE on wallclock/memory: same conservative guesses as the Swin baseline
# job (48h, mem=24G) -- this experiment doesn't change per-epoch cost or
# model size, only the label tensor's scale changes.

#$ -l gpu=1
#$ -l h_rt=48:00:00
#$ -l mem=24G
#$ -l tmpfs=20G
#$ -pe smp 1

#$ -N spect_swin_pretrain_labelalpha

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
# checkpoints/swin_unetr/best_model.pth from the baseline run.
python3 train/train_swin_unetr.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/swin_unetr_label_alpha \
    --log_dir logs/swin_unetr_label_alpha \
    --epochs 150 \
    --patience 6 \
    --batch_size 2 \
    --lr 5e-5 \
    --num_workers 1 \
    --scale_label_by_alpha

echo "Job finished at $(date)"