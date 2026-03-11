#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-auxme-gpu
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=auxme-gpu-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
PY_VERSION=${UV_PYTHON_VERSION:-3.11}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
TRAIN_SCRIPT=${AUXME_TRAIN_SCRIPT:-scripts/training/train_ppo.py}
TRAIN_ARGS=${AUXME_TRAIN_ARGS:-"--config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml --log-level INFO"}
AUXME_UV_RUN_ARGS=${AUXME_UV_RUN_ARGS:-""}
RUN_OUTPUT_DIR=""
MODULES_AVAILABLE=0

if [[ ! -d ${PROJECT_ROOT}/.git ]]; then
  echo "[auxme] Run this script from within the repository." >&2
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

  echo "[auxme] module command is unavailable; continuing without module loads." >&2
  return 0
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[auxme] Loading module ${mod}"
    module load "${mod}"
  done

  echo "[auxme] Loading CUDA module ${CUDA_MODULE}"
  if [[ -n "${CUDA_MODULE}" ]]; then
    module load "${CUDA_MODULE}"
  fi
else
  echo "[auxme] Skipping module and CUDA module loads."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[auxme] uv is not available in PATH. Load the appropriate module or contact the cluster admins." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

cd "${PROJECT_ROOT}"

uv sync --all-extras

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

mkdir -p "${WORKDIR}"
RUN_STAGING_DIR="${WORKDIR}/robot_sf"
RUN_OUTPUT_DIR="${RUN_STAGING_DIR}/results"
mkdir -p "${RUN_OUTPUT_DIR}"
export ROBOT_SF_ARTIFACT_ROOT="${RUN_OUTPUT_DIR}"

echo "[auxme] Running: uv run python ${TRAIN_SCRIPT} ${TRAIN_ARGS}"
if [[ -n "${AUXME_UV_RUN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  UV_RUN_ARGS_ARRAY=(${AUXME_UV_RUN_ARGS})
else
  UV_RUN_ARGS_ARRAY=()
fi

echo "[auxme] Running: uv run ${AUXME_UV_RUN_ARGS} python ${TRAIN_SCRIPT} ${TRAIN_ARGS}"
srun --kill-on-bad-exit=1 uv run "${UV_RUN_ARGS_ARRAY[@]}" python "${TRAIN_SCRIPT}" ${TRAIN_ARGS}
