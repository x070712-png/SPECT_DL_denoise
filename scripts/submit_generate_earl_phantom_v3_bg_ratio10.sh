#!/bin/bash -l
# scripts/submit_generate_earl_phantom_v3_bg_ratio10.sh
#
# v4 FIXED-mode EARL phantom generation, 10:1 sphere:background ratio
# follow-up test (8/10, Stathis's suggested progression at T3 meeting10):
# after the background=0 test (submit_generate_earl_phantom_v3_bg0.sh)
# reproduces the original v1-style catastrophic domain gap (mean=0.0096,
# max/mean ratio=1152.38 -- virtually all signal concentrated into the 6
# tiny spheres, ~0.12% of the volume), this tries a more realistic
# NON-ZERO background at a 10:1 sphere:background ratio -- background =
# sphere_act_conc/10 = 12.6457 -- to see whether an intermediate value
# (rather than either extreme: v2's 84.4954, or bg0's 0.0) is what
# actually brings the CNN-output RC close to 1.0.
#
# sphere_act_conc is kept at the v2 value (126.457), same as bg0 -- only
# background changes. See generate_earl_phantom.py's "FIXED MODE
# (v4, 8/10 ...)" docstring section.
#
# Submit with: qsub scripts/submit_generate_earl_phantom_v3_bg_ratio10.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=3:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_earl_phantom_v3bgr10
#$ -pe smp 1

#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

echo "Job started on $(hostname) at $(date)"

# Load SIRF environment (same as submit_generate_earl_phantom_v2.sh /
# submit_generate_earl_phantom_v3_bg0.sh)
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python3 -u src/spect/baseline/generate_earl_phantom.py \
    --out_dir data/earl_phantom_v3_bg_ratio10 \
    --calibration_mode fixed \
    --sphere_act_conc 126.457 \
    --background_act_conc 12.6457

echo "Job finished at $(date)"