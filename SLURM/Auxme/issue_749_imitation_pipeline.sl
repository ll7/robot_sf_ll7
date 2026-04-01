#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-issue749
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=issue749-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${ISSUE749_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue749-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
TRAIN_SCRIPT=${ISSUE749_TRAIN_SCRIPT:-scripts/training/collect_expert_trajectories.py}
TRAIN_ARGS=${ISSUE749_TRAIN_ARGS:-"--dataset-id issue_749_v3_expert_traj_200 --policy-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 --training-config configs/training/ppo/expert_ppo_issue_576_br06_v10_predictive_foresight_success_priority_policy_analysis_select.yaml --episodes 200 --seeds 123 231 777 992 1337"}
ISSUE749_SOURCE_POLICY_ID=${ISSUE749_SOURCE_POLICY_ID:-}
ISSUE749_UV_RUN_ARGS=${ISSUE749_UV_RUN_ARGS:-""}
ISSUE749_UV_RUN_ARGS_FILE=${ISSUE749_UV_RUN_ARGS_FILE:-""}
LOG_LEVEL=${ISSUE749_LOG_LEVEL:-INFO}
MODULES_AVAILABLE=0

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue749] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
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

  echo "[issue749] module command is unavailable; continuing without module loads." >&2
  return 0
}

stage_registered_policy() {
  local policy_id="$1"
  local destination="${RESULTS_ROOT}/benchmarks/expert_policies/${policy_id}.zip"

  if [[ -f "${destination}" ]]; then
    echo "[issue749] Source policy already staged: ${destination}"
    return 0
  fi

  mkdir -p "$(dirname "${destination}")"

  local resolved_path
  resolved_path="$(uv run python - "$policy_id" <<'PY' | tail -n 1
from pathlib import Path
import sys

from robot_sf.models import resolve_model_path

path = resolve_model_path(sys.argv[1], allow_download=True)
print(Path(path).resolve())
PY
  )"

  if [[ ! -f "${resolved_path}" ]]; then
    echo "[issue749] Resolved source policy does not exist: ${resolved_path}" >&2
    exit 1
  fi

  cp "${resolved_path}" "${destination}"
  echo "[issue749] Staged source policy ${policy_id} -> ${destination}"
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue749] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue749] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
else
  echo "[issue749] Skipping module and CUDA module loads."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[issue749] uv is not available in PATH." >&2
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
export PYGAME_HIDE_SUPPORT_PROMPT=${PYGAME_HIDE_SUPPORT_PROMPT:-1}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}

mkdir -p "${WORKDIR}"
mkdir -p "${RESULTS_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT="${RESULTS_ROOT}"

if [[ -n "${ISSUE749_SOURCE_POLICY_ID}" ]]; then
  stage_registered_policy "${ISSUE749_SOURCE_POLICY_ID}"
fi

if [[ -n "${ISSUE749_UV_RUN_ARGS_FILE}" ]]; then
  if [[ ! -f "${ISSUE749_UV_RUN_ARGS_FILE}" ]]; then
    echo "[issue749] ISSUE749_UV_RUN_ARGS_FILE does not exist: ${ISSUE749_UV_RUN_ARGS_FILE}" >&2
    exit 1
  fi
  mapfile -t UV_RUN_ARGS_ARRAY < "${ISSUE749_UV_RUN_ARGS_FILE}"
elif [[ -n "${ISSUE749_UV_RUN_ARGS}" ]]; then
  echo "[issue749] ISSUE749_UV_RUN_ARGS uses shell word-splitting and is kept for backwards compatibility." >&2
  echo "[issue749] Prefer ISSUE749_UV_RUN_ARGS_FILE with one uv-run argument per line." >&2
  echo "[issue749] ISSUE749_UV_RUN_ARGS only supports simple space-delimited tokens." >&2
  # shellcheck disable=SC2206
  UV_RUN_ARGS_ARRAY=(${ISSUE749_UV_RUN_ARGS})
else
  UV_RUN_ARGS_ARRAY=()
fi

echo "[issue749] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[issue749] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[issue749] ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue749] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[issue749] Running: uv run ${UV_RUN_ARGS_ARRAY[*]} python ${TRAIN_SCRIPT} ${TRAIN_ARGS}"

uv run "${UV_RUN_ARGS_ARRAY[@]}" python "${TRAIN_SCRIPT}" ${TRAIN_ARGS}