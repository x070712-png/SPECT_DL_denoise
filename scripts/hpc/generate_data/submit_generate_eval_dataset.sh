#!/bin/bash -l
# scripts/hpc/generate_data/submit_generate_eval_dataset.sh
#
# Runs generate_eval_phantom_dataset.py for NEMA or EARL -- single fixed
# phantom, all 5 count levels, NOT an array job (unlike XCAT's 100-task
# array, there's only one phantom here, so one regular job is enough).
#
# Same SIRF env load + PYTHONPATH convention as submit_dataset.sh /
# submit_generate_xcat_dataset.sh.
#
# SEEDS (NEW): multiple independent noise realizations per alpha, needed
# because NEMA/EARL is a single fixed phantom -- per-sphere RC from one
# noise draw has large sample-to-sample variance (see
# generate_eval_phantom_dataset.py docstring). Pass as a space-separated
# list via -v SEEDS="42 43 44 45 46 47 48 49 50 51"; defaults to 10 seeds
# (42-51) if not given. label.npy is seed-independent and only computed
# once regardless of how many seeds are requested.
#
# h_rt bumped to 24:00:00 -- 10 seeds means 10x the noisy-reconstruction
# work of a single-seed run (each seed needs its own OSEM reconstruction),
# and even a single seed with the real (non-uniform) NEMA/EARL attenuation
# map previously blew through h_rt=1:00:00 before being bumped to 6:00:00.
#
# Submit with:
#   qsub -v DATASET=nema scripts/hpc/generate_data/submit_generate_eval_dataset.sh
#   qsub -v DATASET=earl scripts/hpc/generate_data/submit_generate_eval_dataset.sh
#   qsub -v DATASET=nema,SEEDS="42 43 44 45 46 47 48 49 50 51" scripts/hpc/generate_data/submit_generate_eval_dataset.sh

#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -N spect_eval_dataset
#$ -wd /home/ucapiuw/SPECT_DL_denoise
#$ -o /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -e /home/ucapiuw/SPECT_DL_denoise/logs/
#$ -pe smp 1

mkdir -p /home/ucapiuw/SPECT_DL_denoise/logs

# Load SIRF environment (identical to submit_dataset.sh / submit_generate_xcat_dataset.sh)
source ~/scripts/sirf_build/sirf_requirements.sh
source ~/devel/SIRF/build/INSTALL/bin/env_sirf.sh

cd /home/ucapiuw/SPECT_DL_denoise
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# DATASET must be "nema" or "earl" -- pass via: qsub -v DATASET=nema ...
if [ "$DATASET" == "nema" ]; then
    ACTIVITY="nema/NEMA_activity.npy"
    ATT_MAP="nema/NEMA_att_map.npy"
    OUT_DIR="data/nema_dataset"
elif [ "$DATASET" == "earl" ]; then
    ACTIVITY="earl/EARL_activity.npy"
    ATT_MAP="earl/EARL_att_map.npy"
    OUT_DIR="data/earl_dataset"
else
    echo "ERROR: DATASET must be 'nema' or 'earl' (got '$DATASET') -- submit with -v DATASET=nema|earl"
    exit 1
fi

# SEEDS: space-separated list via -v SEEDS="42 43 ..."; default 10 seeds
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"

echo "Starting $DATASET at $(date) with seeds: $SEEDS"
python3 -u src/spect/baseline/generate_eval_phantom_dataset.py \
    --activity "$ACTIVITY" \
    --att_map "$ATT_MAP" \
    --out_dir "$OUT_DIR" \
    --seeds $SEEDS
echo "Finished $DATASET at $(date)"