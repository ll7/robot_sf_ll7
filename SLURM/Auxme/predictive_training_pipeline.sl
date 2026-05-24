#!/usr/bin/env bash
#SBATCH --job-name=rsf-pred-pipe
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-predictive-pipeline.out

# Run one config-driven predictive-planner training/evaluation pipeline on Auxme.
#
# Required env vars:
#   PREDICTIVE_PIPELINE_CONFIG  predictive pipeline YAML, relative to repo root or absolute
#
# Optional env vars:
#   PREDICTIVE_PIPELINE_RUN_ID       run id override for stable artifact roots
#   PREDICTIVE_PIPELINE_LOG_LEVEL    default: INFO
#   PREDICTIVE_PIPELINE_RESULTS_DIR  final sync dir, default:
#                                    output/slurm/predictive-pipeline-job-${SLURM_JOB_ID}
set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
RESULTS_ROOT=${PREDICTIVE_PIPELINE_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/predictive-pipeline-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/predictive-pipeline-${SLURM_JOB_ID:-local}}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python

PIPELINE_CONFIG=${PREDICTIVE_PIPELINE_CONFIG:-}
PIPELINE_RUN_ID=${PREDICTIVE_PIPELINE_RUN_ID:-}
PIPELINE_LOG_LEVEL=${PREDICTIVE_PIPELINE_LOG_LEVEL:-INFO}
RUN_OUTPUT_DIR=""

log() {
  echo "[predictive-pipeline] $*"
}

die() {
  echo "[predictive-pipeline] $*" >&2
  exit 1
}

run_in_allocation() {
  if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && srun --version >/dev/null 2>&1; then
    log "Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores "$@"
  else
    echo "[predictive-pipeline] srun unavailable, broken, or not in a Slurm allocation; running directly." >&2
    "$@"
  fi
}

cleanup() {
  if [[ -n "${RUN_OUTPUT_DIR}" && -d "${RUN_OUTPUT_DIR}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --partial --prune-empty-dirs "${RUN_OUTPUT_DIR}/" "${RESULTS_ROOT}/" \
        || echo "[predictive-pipeline] Warning: rsync failed to sync artifacts to ${RESULTS_ROOT}" >&2
    else
      cp -r "${RUN_OUTPUT_DIR}/." "${RESULTS_ROOT}/" \
        || echo "[predictive-pipeline] Warning: cp failed to sync artifacts to ${RESULTS_ROOT}" >&2
    fi
  fi
}
trap cleanup EXIT

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}"
fi

if [[ -z "${PIPELINE_CONFIG}" ]]; then
  die "PREDICTIVE_PIPELINE_CONFIG is required."
fi

if [[ "${PIPELINE_CONFIG}" = /* ]]; then
  PIPELINE_CONFIG_PATH=${PIPELINE_CONFIG}
else
  PIPELINE_CONFIG_PATH=${PROJECT_ROOT}/${PIPELINE_CONFIG}
fi

if [[ ! -f "${PIPELINE_CONFIG_PATH}" ]]; then
  die "Pipeline config not found: ${PIPELINE_CONFIG_PATH}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  die "Expected repo virtualenv python at ${PYTHON_BIN}"
fi

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${PIPELINE_LOG_LEVEL}}
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}"

cd "${PROJECT_ROOT}"

if [[ -z "${PIPELINE_RUN_ID}" ]]; then
  PIPELINE_RUN_ID="predictive_pipeline_${SLURM_JOB_ID:-local_${BASHPID}}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_ID_ARG=(--run-id "${PIPELINE_RUN_ID}")
RUN_OUTPUT_DIR=${PROJECT_ROOT}/output/tmp/predictive_planner/pipeline/${PIPELINE_RUN_ID}

log "Starting predictive pipeline"
log "Config: ${PIPELINE_CONFIG_PATH}"
log "Run id: ${PIPELINE_RUN_ID}"
log "Run output dir: ${RUN_OUTPUT_DIR}"
log "Results sync dir: ${RESULTS_ROOT}"
log "Node: ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"

run_in_allocation "${PYTHON_BIN}" scripts/training/run_predictive_training_pipeline.py \
  --config "${PIPELINE_CONFIG_PATH}" \
  --log-level "${PIPELINE_LOG_LEVEL}" \
  "${RUN_ID_ARG[@]}"

log "Predictive pipeline completed. Artifacts will be synced to ${RESULTS_ROOT}"
