#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-auxme-gpu
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=auxme-gpu-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-/scratch/${USER}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
PY_VERSION=${UV_PYTHON_VERSION:-3.11}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${SCRATCH_ROOT}/robot_sf/results}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
TRAIN_SCRIPT=${AUXME_TRAIN_SCRIPT:-scripts/training_ppo.py}
TRAIN_ARGS=${AUXME_TRAIN_ARGS:-"--config configs/scenarios/classic_interactions.yaml --device cuda --total-timesteps 200000"}
RUN_OUTPUT_DIR=""

if [[ ! -d ${PROJECT_ROOT}/.git ]]; then
  echo "[auxme] Run this script from within the repository." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${RUN_OUTPUT_DIR}" && -d "${RUN_OUTPUT_DIR}" ]]; then
    mkdir -p "${RESULTS_ROOT}/job-${SLURM_JOB_ID}"
    rsync -a --partial --prune-empty-dirs "${RUN_OUTPUT_DIR}/" "${RESULTS_ROOT}/job-${SLURM_JOB_ID}/" || true
  fi
}
trap cleanup EXIT

module purge
for mod in ${MODULE_LIST}; do
  [[ -z "${mod}" ]] && continue
  echo "[auxme] Loading module ${mod}"
  module load "${mod}"
done

echo "[auxme] Loading CUDA module ${CUDA_MODULE}"
module load "${CUDA_MODULE}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[auxme] uv is not available in PATH. Load the appropriate module or contact the cluster admins." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

cd "${PROJECT_ROOT}"

echo "[auxme] Synchronising git submodules"
git submodule sync --recursive
git submodule update --init --recursive

uv sync --all-extras

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

mkdir -p "${WORKDIR}"
RUN_STAGING_DIR="${WORKDIR}/robot_sf"
RUN_OUTPUT_DIR="${RUN_STAGING_DIR}/results"
mkdir -p "${RUN_OUTPUT_DIR}"

srun --kill-on-bad-exit=1 uv run python "${TRAIN_SCRIPT}" \
  --output-dir "${RUN_OUTPUT_DIR}" \
  ${TRAIN_ARGS}
