#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-issue708-ppo
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=issue708-ppo-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue708-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
TRAIN_CONFIG=${ISSUE708_TRAIN_CONFIG:-configs/training/ppo/expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml}
LOG_LEVEL=${ISSUE708_LOG_LEVEL:-INFO}
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue708] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${RUN_OUTPUT_DIR}" && -d "${RUN_OUTPUT_DIR}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    rsync -a --partial --prune-empty-dirs "${RUN_OUTPUT_DIR}/" "${RESULTS_ROOT}/" || true
  fi
}
trap cleanup EXIT

ensure_module_command() {
  if command -v module >/dev/null 2>&1; then
    MODULES_AVAILABLE=1
    return 0
  fi

  local init_script
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -f "${init_script}" ]]; then
      # shellcheck disable=SC1090
      source "${init_script}" >/dev/null 2>&1 || true
    fi
    if command -v module >/dev/null 2>&1; then
      MODULES_AVAILABLE=1
      return 0
    fi
  done

  echo "[issue708] module command is unavailable; continuing without module loads." >&2
  return 0
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue708] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue708] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
else
  echo "[issue708] Skipping module and CUDA module loads."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[issue708] uv is not available in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

cd "${PROJECT_ROOT}"

uv sync --all-extras

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}

mkdir -p "${WORKDIR}"
RUN_STAGING_DIR="${WORKDIR}/robot_sf"
RUN_OUTPUT_DIR="${RUN_STAGING_DIR}/output"
mkdir -p "${RUN_OUTPUT_DIR}"
export ROBOT_SF_ARTIFACT_ROOT="${RUN_OUTPUT_DIR}"

echo "[issue708] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[issue708] TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "[issue708] ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue708] RESULTS_ROOT=${RESULTS_ROOT}"

uv run python scripts/training/train_ppo.py \
  --config "${TRAIN_CONFIG}" \
  --log-level "${LOG_LEVEL}"
