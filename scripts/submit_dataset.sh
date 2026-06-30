#!/bin/bash -l
#$ -l h_rt=4:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -t 1-100
#$ -tc 50
#$ -N spect_dataset
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -pe smp 1

# Create logs directory
mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

# Load SIRF environment
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

# Set Python path
cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Run a batch of 5 phantoms per task (SGE_TASK_ID goes from 1 to 100)
TASK_IDX=$((SGE_TASK_ID - 1))
echo "Starting task ${TASK_IDX} at $(date)"
python3 -u src/spect/baseline/generate_dataset.py ${TASK_IDX}
echo "Finished task ${TASK_IDX} at $(date)"