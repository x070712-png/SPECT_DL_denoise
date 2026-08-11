#!/bin/bash -l
# scripts/submit_generate_earl_phantom_v3_bg0.sh
#
# v4 FIXED-mode EARL phantom generation (8/10, Stathis's finding at T3
# meeting10): tests whether the v2 joint calibration's background activity
# (84.4954, only a ~1.5:1 sphere:background ratio) -- rather than a real
# EARL/NEMA IQ protocol, which has NO background at all (or ~10:1 if
# non-zero) -- is what's actually causing the alpha-independent CNN-output
# overestimation seen in quant_earl_v2_{unet,swin}_output.
#
# No calibration/solving here -- sphere_act_conc is kept at the v2 value
# (126.457), background_act_conc is set directly to 0.0. See
# generate_earl_phantom.py's "FIXED MODE (v4, 8/10...)" docstring section.
#
# Submit with: qsub scripts/submit_generate_earl_phantom_v3_bg0.sh
# Watch with:  qstat -u $(whoami)

#$ -l h_rt=3:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -N spect_earl_phantom_v3bg0
#$ -pe smp 1

#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

echo "Job started on $(hostname) at $(date)"

# Load SIRF environment (same as submit_generate_earl_phantom_v2.sh)
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python3 -u src/spect/baseline/generate_earl_phantom.py \
    --out_dir data/earl_phantom_v3_bg0 \
    --calibration_mode fixed \
    --sphere_act_conc 126.457 \
    --background_act_conc 0.0

echo "Job finished at $(date)"