#!/bin/bash -l
# scripts/hpc/generate_data/submit_generate_xcat_dataset.sh
#
# Mirrors submit_dataset.sh exactly (same SIRF env load, same PYTHONPATH
# convention, same SGE array-job structure) -- just points at
# generate_xcat_dataset.py instead of generate_dataset.py.
#
#$ -l h_rt=4:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -t 1-100
#$ -tc 50
#$ -N spect_xcat_dataset
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -pe smp 1

# NOTE: -t 1-100 assumes 500 XCAT phantoms at 5 per task (same as
# submit_dataset.sh's ellipsoid run). If generate_xcat_parfiles.py was run
# with a different --n_phantoms, recompute n_tasks = ceil(n_phantoms / 5)
# and edit the -t line above BEFORE submitting -- SGE won't warn you if a
# task's slice is empty, generate_xcat_dataset.py just prints a note and
# exits early for that task (see its main()).

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

# Load SIRF environment (identical to submit_dataset.sh)
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

# Set Python path (repo root, same as submit_dataset.sh -- NOT src/)
cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Run a batch of 5 phantoms per task (SGE_TASK_ID goes from 1 to 100)
TASK_IDX=$((SGE_TASK_ID - 1))
echo "Starting task ${TASK_IDX} at $(date)"
python3 -u src/spect/baseline/generate_xcat_dataset.py ${TASK_IDX} \
    --manifest xcat_dataset_manifest.csv \
    --out_dir data/xcat_dataset
echo "Finished task ${TASK_IDX} at $(date)"