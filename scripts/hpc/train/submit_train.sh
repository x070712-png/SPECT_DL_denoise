#!/bin/bash -l
# scripts/hpc/train/submit_train.sh
#
# GPU job submission script for Myriad (UCL Research Computing, SGE scheduler).
# Based on: https://www.rc.ucl.ac.uk/docs/Example_Jobscripts/#gpu-job-script-example
#           https://www.rc.ucl.ac.uk/docs/Supplementary/GPU_Nodes/
#
# Submit with:  qsub scripts/hpc/train/submit_train.sh
# Monitor with: qstat
#
# NOTE on wallclock time: Wei Miao's numbers (~1min15s/epoch, 150 epochs max,
# early-stop patience=6) were on an RTX 5090. Myriad's GPU nodes (A100 40/80G
# or V100 32G) will run at a different speed — unknown until you've run a
# few epochs. 48h is a conservative first guess; check actual per-epoch time
# from the first job's log and adjust for future runs / fine-tuning jobs.

# Request 1 GPU (any node type — A100 or V100 is fine for batch_size=4, 128^3 vols)
#$ -l gpu=1

# Wallclock time (hours:minutes:seconds) — generous upper bound, early stopping
# will normally finish well before this.
#$ -l h_rt=48:00:00

# RAM per core. 128^3 float32 volumes + MONAI SSIM windows are not huge,
# but bump this if you hit OOM (this is host RAM, not VRAM).
#$ -l mem=16G

# Local scratch space on the compute node.
#$ -l tmpfs=20G

# Job name.
#$ -N spect_unet_pretrain

#$ -wd /home/ucapiuw/SPECT_DL_denoise


echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /home/ucapiuw/SPECT_DL_denoise

# ---- load modules (as you've been using for the CPU jobs, GPU variant) ----
module unload gcc-libs
module load pytorch/2.1.0/gpu


export PYTHONPATH=src:$PYTHONPATH

export PYTHONUNBUFFERED=1

# Sanity check that CUDA is visible before committing to a long run.
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# ---- run training ----
python3 train/train_unet.py \
    --data_dir data/dataset \
    --checkpoint_dir checkpoints/3d_unet \
    --log_dir logs/3d_unet \
    --epochs 150 \
    --patience 6 \
    --batch_size 4 \
    --lr 1e-4\
    --num_workers 1

echo "Job finished at $(date)"
