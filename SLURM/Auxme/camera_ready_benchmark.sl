#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-camera-ready
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

# Generic Auxme launcher for config-driven camera-ready benchmark campaigns.
#
# Required:
#   CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/<campaign>.yaml
#
# Required for full runs:
#   CAMERA_READY_BENCHMARK_LABEL=<short-campaign-label>
#   or CAMERA_READY_BENCHMARK_CAMPAIGN_ID=<exact-campaign-id>
#
# Optional:
#   CAMERA_READY_BENCHMARK_MODE=run|preflight        default: run
#   CAMERA_READY_BENCHMARK_OUTPUT_ROOT=<path>        default: output/benchmarks/camera_ready
#   CAMERA_READY_BENCHMARK_LOG_LEVEL=<level>         default: INFO
#   CAMERA_READY_BENCHMARK_SKIP_PUBLICATION_BUNDLE=1 default: disabled

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python

BENCHMARK_CONFIG=${CAMERA_READY_BENCHMARK_CONFIG:-}
BENCHMARK_LABEL=${CAMERA_READY_BENCHMARK_LABEL:-}
BENCHMARK_CAMPAIGN_ID=${CAMERA_READY_BENCHMARK_CAMPAIGN_ID:-}
BENCHMARK_OUTPUT_ROOT=${CAMERA_READY_BENCHMARK_OUTPUT_ROOT:-${PROJECT_ROOT}/output/benchmarks/camera_ready}
BENCHMARK_MODE=${CAMERA_READY_BENCHMARK_MODE:-run}
BENCHMARK_LOG_LEVEL=${CAMERA_READY_BENCHMARK_LOG_LEVEL:-INFO}
SKIP_PUBLICATION_BUNDLE=${CAMERA_READY_BENCHMARK_SKIP_PUBLICATION_BUNDLE:-0}
MODULES_AVAILABLE=0

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[camera-ready-bench] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ -z "${BENCHMARK_CONFIG}" ]]; then
  echo "[camera-ready-bench] CAMERA_READY_BENCHMARK_CONFIG is required." >&2
  exit 2
fi

if [[ "${BENCHMARK_MODE}" != "run" && "${BENCHMARK_MODE}" != "preflight" ]]; then
  echo "[camera-ready-bench] CAMERA_READY_BENCHMARK_MODE must be run or preflight: ${BENCHMARK_MODE}" >&2
  exit 2
fi

if [[ "${BENCHMARK_MODE}" == "run" && -z "${BENCHMARK_LABEL}" && -z "${BENCHMARK_CAMPAIGN_ID}" ]]; then
  echo "[camera-ready-bench] Full runs require CAMERA_READY_BENCHMARK_LABEL or CAMERA_READY_BENCHMARK_CAMPAIGN_ID." >&2
  exit 2
fi

if [[ "${BENCHMARK_CONFIG}" = /* ]]; then
  BENCHMARK_CONFIG_PATH=${BENCHMARK_CONFIG}
else
  BENCHMARK_CONFIG_PATH=${PROJECT_ROOT}/${BENCHMARK_CONFIG}
fi

if [[ ! -f "${BENCHMARK_CONFIG_PATH}" ]]; then
  echo "[camera-ready-bench] Benchmark config not found: ${BENCHMARK_CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[camera-ready-bench] Expected repo virtualenv python at ${PYTHON_BIN}" >&2
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

  echo "[camera-ready-bench] module command unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if command -v srun >/dev/null 2>&1; then
    echo "[camera-ready-bench] Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    local srun_stderr="${WORKDIR}/srun-stderr.log"
    set +e
    srun --cpu_bind=cores --gpus-per-node=1 "$@" 2> >(tee "${srun_stderr}" >&2)
    local srun_status=$?
    set -e
    if [[ "${srun_status}" == "0" ]]; then
      return 0
    fi
    if grep -Eq "Unable to confirm allocation|undefined symbol|command not found" "${srun_stderr}"; then
      echo "[camera-ready-bench] srun infrastructure failure (${srun_status}); retrying directly in the batch allocation." >&2
    else
      return "${srun_status}"
    fi
  else
    echo "[camera-ready-bench] srun unavailable on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}; PATH=${PATH}" >&2
  fi
  echo "[camera-ready-bench] Running directly in the batch allocation." >&2
  "$@"
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[camera-ready-bench] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[camera-ready-bench] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}" "${BENCHMARK_OUTPUT_ROOT}"
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
if [[ "${SKIP_PUBLICATION_BUNDLE}" =~ ^(1|true|yes|on)$ ]]; then
  PUBLICATION_ARG=(--skip-publication-bundle)
fi

echo "[camera-ready-bench] Starting camera-ready campaign"
echo "[camera-ready-bench] Config: ${BENCHMARK_CONFIG_PATH}"
echo "[camera-ready-bench] Mode: ${BENCHMARK_MODE}"
echo "[camera-ready-bench] Output root: ${BENCHMARK_OUTPUT_ROOT}"
echo "[camera-ready-bench] Label: ${BENCHMARK_LABEL:-<none>}"
echo "[camera-ready-bench] Campaign id: ${BENCHMARK_CAMPAIGN_ID:-<auto>}"

run_in_allocation \
  "${PYTHON_BIN}" scripts/tools/run_camera_ready_benchmark.py \
  --config "${BENCHMARK_CONFIG_PATH}" \
  --output-root "${BENCHMARK_OUTPUT_ROOT}" \
  --mode "${BENCHMARK_MODE}" \
  --log-level "${BENCHMARK_LOG_LEVEL}" \
  "${LABEL_ARG[@]}" \
  "${CAMPAIGN_ID_ARG[@]}" \
  "${PUBLICATION_ARG[@]}"

echo "[camera-ready-bench] Benchmark completed. Artifacts are under ${BENCHMARK_OUTPUT_ROOT}"
