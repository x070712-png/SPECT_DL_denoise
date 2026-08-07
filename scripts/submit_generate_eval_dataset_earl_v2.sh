#!/bin/bash -l
# scripts/submit_generate_eval_dataset_earl_v2.sh
#
# Generates the 10-seed EARL v2 dataset from the jointly-calibrated
# phantom (sphere_conc=126.457, background_conc=84.4954, converged 8/7 --
# see generate_earl_phantom.py / submit_generate_earl_phantom_v2.sh,
# job 110057) that fixes the background=0 domain-gap problem found in
# the original EARL phantom (data/earl_dataset).
#
# Same SIRF env + seed convention as submit_generate_eval_dataset.sh.
#
# Submit with: qsub scripts/submit_generate_eval_dataset_earl_v2.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -N spect_earl_dataset_v2
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -pe smp 1

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "Starting earl_v2 at $(date)"
python3 -u src/spect/baseline/generate_eval_phantom_dataset.py \
    --activity data/earl_phantom_v2/activity.npy \
    --att_map data/earl_phantom_v2/att_map.npy \
    --out_dir data/earl_dataset_v2 \
    --seeds 42 43 44 45 46 47 48 49 50 51
echo "Finished earl_v2 at $(date)"