#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-issue791-attention-head
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=issue791-attention-head-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue791-attention-head-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
TRAIN_CONFIG=${ISSUE791_TRAIN_CONFIG:-configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_stage1.yaml}
LOG_LEVEL=${ISSUE791_LOG_LEVEL:-INFO}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue791] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
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

  echo "[issue791] module command is unavailable; continuing without module loads." >&2
  return 0
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue791] Loading module ${mod}"
    module load "${mod}"
  done

  echo "[issue791] Loading CUDA module ${CUDA_MODULE}"
  if [[ -n "${CUDA_MODULE}" ]]; then
    module load "${CUDA_MODULE}"
  fi
fi

mkdir -p "${WORKDIR}"
mkdir -p "${LOCAL_OUTPUT_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[issue791] Expected repo virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}"
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}

cd "${PROJECT_ROOT}"

echo "[issue791] Starting training with config: ${TRAIN_CONFIG}"
echo "[issue791] Output root: ${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue791] Job ID: ${SLURM_JOB_ID}"
echo "[issue791] Log level: ${LOG_LEVEL}"

srun --cpu_bind=cores --gpus-per-node=1 \
  "${PYTHON_BIN}" scripts/training/train_ppo.py \
  --config "${TRAIN_CONFIG}" \
  --log-level "${LOG_LEVEL}"

echo "[issue791] Training completed. Artifacts will be synced to ${RESULTS_ROOT}"
