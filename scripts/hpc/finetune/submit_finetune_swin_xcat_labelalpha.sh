#!/bin/bash -l
# scripts/submit_finetune_swin_xcat_labelalpha.sh
#
# Fine-tunes the Swin UNETR LABEL-ALPHA-pretrained checkpoint on the
# 500-phantom XCAT dataset -- keeps --scale_label_by_alpha ON throughout,
# matching the method the base checkpoint
# (checkpoints/swin_unetr_label_alpha/best_model.pth) was itself
# pretrained with.
#
# Label x alpha counterpart of submit_finetune_swin_xcat.sh (old method) --
# run both to get a clean old-vs-new comparison on XCAT, same as already
# done on the ellipsoid test split.
#
# LR set to 1/5 of the label-alpha pretraining LR (5e-5 -> 1e-5), same
# rationale as the old-method Swin XCAT fine-tuning script.
#
# CHECK: --data_dir below assumes generate_xcat_dataset.py wrote its
# output to data/xcat_dataset in the same alpha_*/{input,label}_NNNN.npy
# layout as data/dataset. Update if the actual path differs.
#
# Submit with: qsub scripts/submit_finetune_swin_xcat_labelalpha.sh
# Watch with:  qstat -u $(whoami)

#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -l mem=24G
#$ -l tmpfs=20G
#$ -N spect_swin_finetune_xcat_labelalpha
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

python3 -u train/train_swin_unetr.py \
    --data_dir data/xcat_dataset \
    --init_checkpoint checkpoints/swin_unetr_label_alpha/best_model.pth \
    --checkpoint_dir checkpoints/swin_unetr_xcat_finetune_label_alpha \
    --log_dir logs/swin_unetr_xcat_finetune_label_alpha \
    --epochs 100 \
    --patience 6 \
    --batch_size 2 \
    --lr 1e-5 \
    --num_workers 1 \
    --scale_label_by_alpha

echo "Job finished at $(date)"