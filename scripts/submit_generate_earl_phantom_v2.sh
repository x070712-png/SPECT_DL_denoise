#!/bin/bash -l
# scripts/submit_generate_earl_phantom_v2.sh
#
# Joint 2-parameter calibration (sphere activity + background activity) of
# the EARL phantom, fixing the max/mean-ratio domain-gap found on 8/7 (see
# generate_earl_phantom.py docstring for full explanation). Up to
# max_outer(6) x inner_max_iters(3) = 18 forward-projection+OSEM
# reconstructions worst case -- more compute than the single-parameter v1
# calibration (job 99416, 2 iterations), so h_rt bumped up accordingly.
#
# Submit with: qsub scripts/submit_generate_earl_phantom_v2.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=8:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_earl_phantom_v2
#$ -pe smp 1

#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

echo "Job started on $(hostname) at $(date)"

# Load SIRF environment (same as submit_generate_eval_dataset.sh)
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python3 -u src/spect/baseline/generate_earl_phantom.py \
    --out_dir data/earl_phantom_v2 \
    --target_mean 0.461 \
    --target_max_mean_ratio 25.20 \
    --init_sphere_act_conc 2.0 \
    --max_outer 6 \
    --ratio_tol 0.15

echo "Job finished at $(date)"