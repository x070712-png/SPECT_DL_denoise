#!/bin/bash -l
# scripts/submit_generate_eval_dataset_earl_v3_bg_ratio10.sh
#
# Generates the 10-seed EARL v3-bg_ratio10 dataset from the FIXED-mode
# phantom (sphere_conc=126.457 -- same as v2 -- background_conc=12.6457,
# a 10:1 sphere:background ratio, Stathis's suggested intermediate test
# between v2's background=84.4954 and bg0's background=0.0). Clean-
# reconstruction stats: mean=0.0772, max/mean ratio=146.24 (job 121328) --
# still well off the ellipsoid target (mean=0.461, ratio=25.2), but much
# closer than bg0 (mean=0.0096, ratio=1152.38).
#
# Same SIRF env + seed convention as submit_generate_eval_dataset_earl_v2.sh /
# submit_generate_eval_dataset_earl_v3_bg0.sh.
#
# Submit with: qsub scripts/submit_generate_eval_dataset_earl_v3_bg_ratio10.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -N spect_earl_dataset_v3bgr10
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -pe smp 1

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "Starting earl_v3_bg_ratio10 at $(date)"
python3 -u src/spect/baseline/generate_eval_phantom_dataset.py \
    --activity data/earl_phantom_v3_bg_ratio10/activity.npy \
    --att_map data/earl_phantom_v3_bg_ratio10/att_map.npy \
    --out_dir data/earl_dataset_v3_bg_ratio10 \
    --seeds 42 43 44 45 46 47 48 49 50 51
echo "Finished earl_v3_bg_ratio10 at $(date)"