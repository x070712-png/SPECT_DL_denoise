#!/bin/bash -l
# scripts/hpc/generate_data/submit_generate_eval_dataset_earl_v3_bg0.sh
#
# Generates the 10-seed EARL v3-bg0 dataset from the FIXED-mode phantom
# (sphere_conc=126.457 -- same as v2 -- background_conc=0.0, since v2's
# joint-calibrated background (84.4954, only ~1.5:1 sphere:background) is
# unrealistic for a real EARL/NEMA IQ phantom -- see
# generate_earl_phantom.py / submit_generate_earl_phantom_v3_bg0.sh).
# This background=0 phantom reproduces the original v1-style domain gap
# at the clean-reconstruction level (mean=0.0096, max/mean ratio=1152.38),
# so it's an open question what the CNN does with it -- that's the point
# of this test.
#
# Same SIRF env + seed convention as submit_generate_eval_dataset_earl_v2.sh.
#
# Submit with: qsub scripts/hpc/generate_data/submit_generate_eval_dataset_earl_v3_bg0.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -N spect_earl_dataset_v3bg0
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -pe smp 1

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "Starting earl_v3_bg0 at $(date)"
python3 -u src/spect/baseline/generate_eval_phantom_dataset.py \
    --activity data/earl_phantom_v3_bg0/activity.npy \
    --att_map data/earl_phantom_v3_bg0/att_map.npy \
    --out_dir data/earl_dataset_v3_bg0 \
    --seeds 42 43 44 45 46 47 48 49 50 51
echo "Finished earl_v3_bg0 at $(date)"