#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-dreamer-wm
#SBATCH --account=mitarbeiter
#SBATCH --partition=pro6000
#SBATCH --qos=pro6000-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-13:00:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=auxme-pro6000-dreamer-wm-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
PY_VERSION=${UV_PYTHON_VERSION:-3.11}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
PIPELINE_MODE=${DREAMER_PIPELINE_MODE:-all}
MODULES_AVAILABLE=0

if [[ ! -d ${PROJECT_ROOT}/.git ]]; then
  echo "[auxme] Run this script from within the repository." >&2
  exit 1
fi

if [[ -z "${DREAMER_WM_TEACHER_MODEL_ID:-}" && -z "${DREAMER_WM_TEACHER_CHECKPOINT:-}" ]]; then
  echo "[auxme] Set DREAMER_WM_TEACHER_MODEL_ID or DREAMER_WM_TEACHER_CHECKPOINT before submit." >&2
  exit 2
fi

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
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[auxme] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[auxme] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
else
  echo "[auxme] Skipping module and CUDA module loads."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[auxme] uv is not available in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

cd "${PROJECT_ROOT}"

echo "[auxme] Using Python ${PY_VERSION} through the existing uv-managed environment"
uv sync --all-extras

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export PYGAME_HIDE_SUPPORT_PROMPT=${PYGAME_HIDE_SUPPORT_PROMPT:-1}
export ROBOT_SF_ARTIFACT_ROOT="${PROJECT_ROOT}/output"

echo "[auxme] Running Dreamer world-model pipeline mode=${PIPELINE_MODE}"
echo "[auxme] Teacher model id=${DREAMER_WM_TEACHER_MODEL_ID:-<unset>}"
echo "[auxme] Teacher checkpoint=${DREAMER_WM_TEACHER_CHECKPOINT:-<unset>}"

uv run --extra rllib bash scripts/training/run_dreamerv3_world_model_pipeline.sh "${PIPELINE_MODE}"
