#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-issue791-benchmark
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=issue791-benchmark-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue791-benchmark-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
BENCHMARK_CONFIG=${ISSUE791_BENCHMARK_CONFIG:-configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml}
BENCHMARK_LABEL=${ISSUE791_BENCHMARK_LABEL:-issue791-eval-aligned-leader}
BENCHMARK_MODE=${ISSUE791_BENCHMARK_MODE:-run}
BENCHMARK_OUTPUT_ROOT=${ISSUE791_BENCHMARK_OUTPUT_ROOT:-}
SKIP_PUBLICATION_BUNDLE=${ISSUE791_SKIP_PUBLICATION_BUNDLE:-0}
LOG_LEVEL=${ISSUE791_LOG_LEVEL:-INFO}
RUN_OUTPUT_DIR=""
MODULES_AVAILABLE=0

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue791-benchmark] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
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

  echo "[issue791-benchmark] module command is unavailable; continuing without module loads." >&2
  return 0
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue791-benchmark] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue791-benchmark] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
else
  echo "[issue791-benchmark] Skipping module and CUDA module loads."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[issue791-benchmark] uv is not available in PATH." >&2
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

cmd=(
  uv run python scripts/tools/run_camera_ready_benchmark.py
  --config "${BENCHMARK_CONFIG}"
  --mode "${BENCHMARK_MODE}"
  --log-level "${LOG_LEVEL}"
)

if [[ -n "${BENCHMARK_LABEL}" ]]; then
  cmd+=(--label "${BENCHMARK_LABEL}")
fi

if [[ -n "${BENCHMARK_OUTPUT_ROOT}" ]]; then
  cmd+=(--output-root "${BENCHMARK_OUTPUT_ROOT}")
fi

if [[ "${SKIP_PUBLICATION_BUNDLE}" == "1" ]]; then
  cmd+=(--skip-publication-bundle)
fi

echo "[issue791-benchmark] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[issue791-benchmark] BENCHMARK_CONFIG=${BENCHMARK_CONFIG}"
echo "[issue791-benchmark] BENCHMARK_MODE=${BENCHMARK_MODE}"
echo "[issue791-benchmark] ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue791-benchmark] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[issue791-benchmark] Running: ${cmd[*]}"

"${cmd[@]}"