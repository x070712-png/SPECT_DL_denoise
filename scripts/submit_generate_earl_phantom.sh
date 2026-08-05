#!/bin/bash -l
# scripts/submit_generate_earl_phantom.sh
#
# Generates + calibrates the EARL activity/attenuation maps (via
# generate_earl_phantom.py) so the reconstructed scale matches the
# ellipsoid/XCAT training data. SIRF-side script -- CPU only, no GPU
# needed. Runs up to 3 forward-projection+OSEM reconstructions
# (calibration iterations), so it's much lighter than the full 10-seed
# generate_eval_phantom_dataset.py run, but still real STIR/SIRF compute
# -- submit via qsub, don't run on the login node.
#
# Submit with: qsub scripts/submit_generate_earl_phantom.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=2:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_earl_phantom
#$ -pe smp 1

#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

echo "Job started on $(hostname) at $(date)"

cd /home/ucapiuw/SPECT_DL_denoise

# SIRF-side convention: absolute imports (src.spect...), plain script-path
# invocation, PYTHONPATH=repo_root -- NOT the pytorch/module-load convention
# used by the train/inference GPU scripts.
export PYTHONPATH=.:$PYTHONPATH

python3 -u src/spect/baseline/generate_earl_phantom.py \
    --out_dir data/earl_phantom \
    --target_mean 3.143 \
    --init_act_conc 2.0 \
    --max_iters 3

echo "Job finished at $(date)"