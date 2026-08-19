#!/bin/bash -l
# scripts/hpc/generate_data/submit_generate_earl_phantom.sh
#
# Generates + calibrates the EARL activity/attenuation maps (via
# generate_earl_phantom.py) so the reconstructed scale matches the
# ellipsoid/XCAT training data. SIRF-side script -- CPU only, no GPU
# needed. Runs up to 3 forward-projection+OSEM reconstructions
# (calibration iterations), so it's much lighter than the full 10-seed
# generate_eval_phantom_dataset.py run, but still real STIR/SIRF compute
# -- submit via qsub, don't run on the login node.
#
# Submit with: qsub scripts/hpc/generate_data/submit_generate_earl_phantom.sh
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

# Load SIRF environment (identical to submit_generate_eval_dataset.sh /
# submit_dataset.sh / submit_generate_xcat_dataset.sh) -- required before
# any SIRF/STIR import, otherwise the job fails with ModuleNotFoundError.
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python3 -u src/spect/baseline/generate_earl_phantom.py \
    --out_dir data/earl_phantom \
    --target_mean 3.143 \
    --init_act_conc 2.0 \
    --max_iters 3

echo "Job finished at $(date)"