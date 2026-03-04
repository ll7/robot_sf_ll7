#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-gpu
#SBATCH --partition=epyc-gpu-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=6:00:00
#SBATCH --gpus=a100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=robot-sf-gpu-%j.out

# Single-GPU template for LiCCA A100 nodes. Keep CPU requests ≤32 per GPU to
# stay on one socket and avoid stranding resources.

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
SCRATCH_RESULTS=${SCRATCH_RESULTS:-/hpc/gpfs2/scratch/u/$USER/robot_sf/results}
WORKDIR=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
ENABLE_CUDA_MPS=${ENABLE_CUDA_MPS:-0}

# Guard conda wrapper variables when running with set -u.
: "${_CE_M:=}"
: "${_CE_CONDA:=}"

cleanup() {
  echo "[licca] Syncing results to ${SCRATCH_RESULTS}"
  mkdir -p "${SCRATCH_RESULTS}"
  rsync -a --partial --prune-empty-dirs "${WORKDIR}/results/" "${SCRATCH_RESULTS}/" || true
  if [[ "${ENABLE_CUDA_MPS}" == "1" ]]; then
    module unload cuda-mps || true
  fi
}
trap cleanup EXIT

module purge
module load miniforge
module load cuda/12.8
if [[ -n "${CONDA_EXE:-}" ]]; then
  eval "$("${CONDA_EXE}" shell.bash hook)"
fi
conda activate robot-sf

if [[ "${ENABLE_CUDA_MPS}" == "1" ]]; then
  module load cuda-mps
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

mkdir -p "${WORKDIR}" "${SCRATCH_RESULTS}"
LOG_PATH="${WORKDIR}/results"
mkdir -p "${LOG_PATH}"
export ROBOT_SF_ARTIFACT_ROOT="${LOG_PATH}"

cd "${PROJECT_ROOT}"

# TODO: stage large datasets to ${WORKDIR} if GPFS throughput becomes a bottleneck.

srun python scripts/training/train_ppo.py \
  --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --log-level INFO
