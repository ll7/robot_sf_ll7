#!/usr/bin/env bash
#SBATCH --job-name=issue1470-oracle-traces
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue1470-oracle-imitation-traces.out
#SBATCH --error=output/slurm/%j-issue1470-oracle-imitation-traces.err

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue1470-oracle-imitation-traces-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/issue1470-${SLURM_JOB_ID:-local}}
PACKET=${ISSUE1470_PACKET:-configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml}
REGISTRY=${ISSUE1470_CANDIDATE_REGISTRY:-docs/context/policy_search/candidate_registry.yaml}
SPLIT=${ISSUE1470_SPLIT:-train}
HORIZON=${ISSUE1470_HORIZON:-500}
WORKERS=${ISSUE1470_WORKERS:-1}
LOG_LEVEL=${ISSUE1470_LOG_LEVEL:-WARNING}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

cleanup() {
  if [[ -n "${RUN_OUTPUT_DIR}" && -d "${RUN_OUTPUT_DIR}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --partial --prune-empty-dirs "${RUN_OUTPUT_DIR}/" "${RESULTS_ROOT}/" || true
    else
      cp -r "${RUN_OUTPUT_DIR}/." "${RESULTS_ROOT}/" || true
    fi
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

  echo "[issue1470] module command is unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if [[ "${ISSUE1470_USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[issue1470] Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores --gpus-per-node=1 "$@"
  else
    echo "[issue1470] Running directly inside the batch allocation; set ISSUE1470_USE_SRUN=1 to opt into srun." >&2
    "$@"
  fi
}

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue1470] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

for path in "${PACKET}" "${REGISTRY}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[issue1470] Required path not found: ${path}" >&2
    exit 1
  fi
done

ensure_module_command
if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue1470] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue1470] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[issue1470] uv is not available in PATH." >&2
  exit 1
fi

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
export UV_PROJECT_ENVIRONMENT=${ISSUE1470_UV_ENV_DIR:-${WORKDIR}/uv_env}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}
TRACE_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}/oracle_imitation/issue1470_${SPLIT}_candidate_traces
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")" "${TRACE_OUTPUT_DIR}"

echo "[issue1470] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[issue1470] ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue1470] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "[issue1470] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[issue1470] PACKET=${PACKET}"
echo "[issue1470] REGISTRY=${REGISTRY}"
echo "[issue1470] SPLIT=${SPLIT}"
echo "[issue1470] HORIZON=${HORIZON}"
echo "[issue1470] WORKERS=${WORKERS}"
echo "[issue1470] commit=$(git rev-parse HEAD)"

uv sync --all-extras

run_in_allocation \
  uv run python scripts/validation/validate_oracle_imitation_launch_packet.py \
  --config "${PACKET}" \
  --json

run_in_allocation \
  uv run python scripts/training/collect_oracle_imitation_candidate_traces.py \
  --config "${PACKET}" \
  --candidate-registry "${REGISTRY}" \
  --split "${SPLIT}" \
  --output-dir "${TRACE_OUTPUT_DIR}" \
  --horizon "${HORIZON}" \
  --workers "${WORKERS}" \
  --json

echo "[issue1470] Oracle candidate trace collection completed. Artifacts will sync to ${RESULTS_ROOT}"
