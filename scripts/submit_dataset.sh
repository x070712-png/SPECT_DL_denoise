#!/bin/bash -l
#$ -l h_rt=4:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -t 1-500
#$ -tc 50
#$ -N spect_dataset
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/job_${JOB_ID}_${SGE_TASK_ID}.out
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/job_${JOB_ID}_${SGE_TASK_ID}.err
#$ -pe smp 1

# Create logs directory
mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

# Load SIRF environment
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

# Set Python path
cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Run single phantom (SGE_TASK_ID goes from 1 to 500, convert to 0-499)
PHANTOM_IDX=$((SGE_TASK_ID - 1))
echo "Starting phantom ${PHANTOM_IDX} at $(date)"
python3 src/spect/baseline/generate_dataset.py ${PHANTOM_IDX}
echo "Finished phantom ${PHANTOM_IDX} at $(date)"