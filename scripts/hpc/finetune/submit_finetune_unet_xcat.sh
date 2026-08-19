#!/bin/bash -l
# scripts/hpc/finetune/submit_finetune_unet_xcat.sh
#
# Fine-tunes the 3D U-Net ellipsoid-pretrained checkpoint on the 500-phantom
# XCAT dataset -- OLD method (scale_label_by_alpha OFF), matching the method
# the base checkpoint (checkpoints/3d_unet/best_model.pth) was itself
# pretrained with. Do NOT mix methods between --init_checkpoint and this
# run's own --scale_label_by_alpha setting, or the model's already-learned
# output scale won't match the fine-tuning targets.
#
# LR set to 1/5 of the pretraining LR (1e-4 -> 2e-5) -- fine-tuning from a
# converged checkpoint on a new (but related) domain, standard practice is
# a lower LR than the original pretraining to avoid catastrophically
# forgetting what was already learned on the ellipsoid data.
#
# CHECK: --data_dir below assumes generate_xcat_dataset.py wrote its output
# to data/xcat_dataset in the same alpha_*/{input,label}_NNNN.npy layout as
# data/dataset. Update if the actual path differs.
#
# Submit with: qsub scripts/submit_finetune_unet_xcat.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -l mem=24G
#$ -l tmpfs=20G
#$ -N spect_unet_finetune_xcat
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

python3 -u train/train_unet.py \
    --data_dir data/xcat_dataset \
    --init_checkpoint checkpoints/3d_unet/best_model.pth \
    --checkpoint_dir checkpoints/3d_unet_xcat_finetune \
    --log_dir logs/3d_unet_xcat_finetune \
    --epochs 100 \
    --patience 6 \
    --batch_size 4 \
    --lr 2e-5 \
    --num_workers 1

echo "Job finished at $(date)"