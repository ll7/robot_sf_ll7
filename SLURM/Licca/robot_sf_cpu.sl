#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-cpu
#SBATCH --partition=epyc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=robot-sf-cpu-%j.out

# Multi-threaded CPU template for LiCCA. Adjust CPUs/memory/time per experiment.

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
SCRATCH_RESULTS=${SCRATCH_RESULTS:-/hpc/gpfs2/scratch/u/$USER/robot_sf/results}
WORKDIR=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}

# Guard conda wrapper variables when running with set -u.
: "${_CE_M:=}"
: "${_CE_CONDA:=}"

cleanup() {
  echo "[licca] Syncing results to ${SCRATCH_RESULTS}"
  mkdir -p "${SCRATCH_RESULTS}"
  rsync -a --partial --prune-empty-dirs "${WORKDIR}/results/" "${SCRATCH_RESULTS}/" || true
}
trap cleanup EXIT

module purge
module load miniforge
if [[ -n "${CONDA_EXE:-}" ]]; then
  eval "$("${CONDA_EXE}" shell.bash hook)"
fi
conda activate robot-sf

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

mkdir -p "${WORKDIR}" "${SCRATCH_RESULTS}"
LOG_PATH="${WORKDIR}/results"
mkdir -p "${LOG_PATH}"

cd "${PROJECT_ROOT}"

# TODO: copy large datasets to ${WORKDIR} if they cause GPFS contention.

srun python scripts/training_ppo.py \
  --config configs/scenarios/classic_interactions.yaml \
  --output-dir "${LOG_PATH}" \
  --num-envs 1 \
  --total-timesteps 200000
