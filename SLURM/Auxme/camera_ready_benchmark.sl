#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-camera-ready-benchmark
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-camera-ready-benchmark.out

# Run a config-driven camera-ready benchmark campaign on Auxme.
#
# Required env vars:
#   CAMERA_READY_BENCHMARK_CONFIG       config YAML, relative to repo root or absolute
#   CAMERA_READY_BENCHMARK_LABEL        label suffix for a new campaign id, or
#   CAMERA_READY_BENCHMARK_CAMPAIGN_ID  exact campaign id for resumable/reproducible roots
#
# Optional env vars:
#   CAMERA_READY_BENCHMARK_OUTPUT_ROOT  default: output/benchmarks/camera_ready
#   CAMERA_READY_BENCHMARK_MODE         run|preflight (default: run)
#   CAMERA_READY_BENCHMARK_LOG_LEVEL    default: INFO
#   CAMERA_READY_SKIP_PUBLICATION_BUNDLE true/false-like, default: false
#   CAMERA_READY_RESULTS_DIR            final sync dir, default: output/slurm/camera-ready-benchmark-job-${SLURM_JOB_ID}
#   AUXME_MODULES                       default: gcc/13.2.0
#   AUXME_CUDA_MODULE                   default: cuda/12.1
set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${CAMERA_READY_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/camera-ready-benchmark-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/camera-ready-benchmark-${SLURM_JOB_ID:-local}}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python

BENCHMARK_CONFIG=${CAMERA_READY_BENCHMARK_CONFIG:-}
BENCHMARK_LABEL=${CAMERA_READY_BENCHMARK_LABEL:-}
BENCHMARK_CAMPAIGN_ID=${CAMERA_READY_BENCHMARK_CAMPAIGN_ID:-}
BENCHMARK_OUTPUT_ROOT=${CAMERA_READY_BENCHMARK_OUTPUT_ROOT:-${PROJECT_ROOT}/output/benchmarks/camera_ready}
BENCHMARK_MODE=${CAMERA_READY_BENCHMARK_MODE:-run}
BENCHMARK_LOG_LEVEL=${CAMERA_READY_BENCHMARK_LOG_LEVEL:-INFO}
SKIP_PUBLICATION_BUNDLE=${CAMERA_READY_SKIP_PUBLICATION_BUNDLE:-false}
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

log() {
  echo "[camera-ready-bench] $*"
}

die() {
  echo "[camera-ready-bench] $*" >&2
  exit 1
}

usage_error() {
  echo "[camera-ready-bench] $*" >&2
  echo "[camera-ready-bench] Example preflight:" >&2
  echo "  CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/example.yaml \\" >&2
  echo "  CAMERA_READY_BENCHMARK_LABEL=issue999-preflight \\" >&2
  echo "  CAMERA_READY_BENCHMARK_MODE=preflight \\" >&2
  echo "  scripts/dev/sbatch_use_max_time.sh --dry-run SLURM/Auxme/camera_ready_benchmark.sl" >&2
  exit 2
}

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}"
fi

if [[ -z "${BENCHMARK_CONFIG}" ]]; then
  usage_error "CAMERA_READY_BENCHMARK_CONFIG is required."
fi

case "${BENCHMARK_MODE}" in
  run|preflight) ;;
  *) usage_error "CAMERA_READY_BENCHMARK_MODE must be run or preflight, got: ${BENCHMARK_MODE}" ;;
esac

if [[ -z "${BENCHMARK_LABEL}" && -z "${BENCHMARK_CAMPAIGN_ID}" ]]; then
  usage_error "Set CAMERA_READY_BENCHMARK_LABEL or CAMERA_READY_BENCHMARK_CAMPAIGN_ID explicitly."
fi

case "${SKIP_PUBLICATION_BUNDLE,,}" in
  1|true|yes|on) SKIP_PUBLICATION_BUNDLE=true ;;
  0|false|no|off|"") SKIP_PUBLICATION_BUNDLE=false ;;
  *) usage_error "CAMERA_READY_SKIP_PUBLICATION_BUNDLE must be true/false-like, got: ${SKIP_PUBLICATION_BUNDLE}" ;;
esac

if [[ "${BENCHMARK_CONFIG}" = /* ]]; then
  BENCHMARK_CONFIG_PATH=${BENCHMARK_CONFIG}
else
  BENCHMARK_CONFIG_PATH=${PROJECT_ROOT}/${BENCHMARK_CONFIG}
fi

if [[ ! -f "${BENCHMARK_CONFIG_PATH}" ]]; then
  die "Benchmark config not found: ${BENCHMARK_CONFIG_PATH}"
fi

cleanup() {
  if [[ -n "${RUN_OUTPUT_DIR}" && -d "${RUN_OUTPUT_DIR}" ]]; then
    mkdir -p "${RESULTS_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --partial --prune-empty-dirs "${RUN_OUTPUT_DIR}/" "${RESULTS_ROOT}/" \
        || echo "[camera-ready-bench] Warning: rsync failed to sync artifacts to ${RESULTS_ROOT}" >&2
    else
      cp -r "${RUN_OUTPUT_DIR}/." "${RESULTS_ROOT}/" \
        || echo "[camera-ready-bench] Warning: cp failed to sync artifacts to ${RESULTS_ROOT}" >&2
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

  echo "[camera-ready-bench] module command unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if [[ "${CAMERA_READY_BENCH_USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && srun --version >/dev/null 2>&1; then
    log "Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores "$@"
  else
    echo "[camera-ready-bench] running directly inside the batch allocation; set CAMERA_READY_BENCH_USE_SRUN=1 to opt into srun." >&2
    "$@"
  fi
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    log "Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    log "Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  die "Expected repo virtualenv python at ${PYTHON_BIN}"
fi

mkdir -p "${WORKDIR}"
mkdir -p "${LOCAL_OUTPUT_ROOT}"
mkdir -p "${BENCHMARK_OUTPUT_ROOT}"

export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}"
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}

cd "${PROJECT_ROOT}"

LABEL_ARG=()
if [[ -n "${BENCHMARK_LABEL}" ]]; then
  LABEL_ARG=(--label "${BENCHMARK_LABEL}")
fi

CAMPAIGN_ID_ARG=()
if [[ -n "${BENCHMARK_CAMPAIGN_ID}" ]]; then
  CAMPAIGN_ID_ARG=(--campaign-id "${BENCHMARK_CAMPAIGN_ID}")
fi

PUBLICATION_ARG=()
if [[ "${SKIP_PUBLICATION_BUNDLE}" == "true" ]]; then
  PUBLICATION_ARG=(--skip-publication-bundle)
fi

log "Starting camera-ready campaign"
log "Config: ${BENCHMARK_CONFIG_PATH}"
log "Mode: ${BENCHMARK_MODE}"
log "Output root: ${BENCHMARK_OUTPUT_ROOT}"
log "Label: ${BENCHMARK_LABEL:-<none>}"
log "Campaign id: ${BENCHMARK_CAMPAIGN_ID:-<none>}"
log "Skip publication bundle: ${SKIP_PUBLICATION_BUNDLE}"

run_in_allocation \
  "${PYTHON_BIN}" scripts/tools/run_camera_ready_benchmark.py \
  --config "${BENCHMARK_CONFIG_PATH}" \
  --output-root "${BENCHMARK_OUTPUT_ROOT}" \
  --mode "${BENCHMARK_MODE}" \
  --log-level "${BENCHMARK_LOG_LEVEL}" \
  "${LABEL_ARG[@]}" \
  "${CAMPAIGN_ID_ARG[@]}" \
  "${PUBLICATION_ARG[@]}"

log "Benchmark completed. Artifacts will be synced to ${RESULTS_ROOT}"
