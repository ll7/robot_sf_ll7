#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-issue791-benchmark
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue791-benchmark.out

# Run a config-driven camera-ready benchmark campaign for issue-791 promotion candidates.
#
# Usage:
#   ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
#   ISSUE791_BENCHMARK_LABEL=issue791-eval-aligned-compare \
#   sbatch SLURM/Auxme/issue_791_benchmark.sl
#
# Optional env vars:
#   ISSUE791_BENCHMARK_OUTPUT_ROOT  - override base output dir (default: output/benchmarks/issue_791)
#   ISSUE791_BENCHMARK_MODE         - run|preflight (default: run)
#   ISSUE791_BENCHMARK_LOG_LEVEL    - default: INFO
set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue791-benchmark-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python

BENCHMARK_CONFIG=${ISSUE791_BENCHMARK_CONFIG:-}
BENCHMARK_LABEL=${ISSUE791_BENCHMARK_LABEL:-}
BENCHMARK_OUTPUT_ROOT=${ISSUE791_BENCHMARK_OUTPUT_ROOT:-${PROJECT_ROOT}/output/benchmarks/issue_791}
BENCHMARK_MODE=${ISSUE791_BENCHMARK_MODE:-run}
BENCHMARK_LOG_LEVEL=${ISSUE791_BENCHMARK_LOG_LEVEL:-INFO}
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue791-bench] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ -z "${BENCHMARK_CONFIG}" ]]; then
  echo "[issue791-bench] ISSUE791_BENCHMARK_CONFIG is required." >&2
  echo "[issue791-bench] Example:" >&2
  echo "  ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \\" >&2
  echo "  sbatch SLURM/Auxme/issue_791_benchmark.sl" >&2
  exit 2
fi

if [[ "${BENCHMARK_CONFIG}" = /* ]]; then
  BENCHMARK_CONFIG_PATH=${BENCHMARK_CONFIG}
else
  BENCHMARK_CONFIG_PATH=${PROJECT_ROOT}/${BENCHMARK_CONFIG}
fi

if [[ ! -f "${BENCHMARK_CONFIG_PATH}" ]]; then
  echo "[issue791-bench] Benchmark config not found: ${BENCHMARK_CONFIG_PATH}" >&2
  exit 1
fi

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

  echo "[issue791-bench] module command unavailable; continuing without module loads." >&2
  return 0
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue791-bench] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue791-bench] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

mkdir -p "${WORKDIR}"
mkdir -p "${LOCAL_OUTPUT_ROOT}"
mkdir -p "${BENCHMARK_OUTPUT_ROOT}"

# The campaign runner expects model artifacts to be readable at the paths given by
# configs/baselines/*.yaml. The Wave-5 leader artifact must already exist at
# output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip.
EXPECTED_LEADER=${PROJECT_ROOT}/output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip
if [[ ! -f "${EXPECTED_LEADER}" ]]; then
  echo "[issue791-bench] Wave-5 leader artifact missing: ${EXPECTED_LEADER}" >&2
  echo "[issue791-bench] Restore it from output/slurm/issue791-reward-curriculum-job-11724/benchmarks/expert_policies/checkpoints/.../...best.zip before resubmitting." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[issue791-bench] Expected repo virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}"
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}

cd "${PROJECT_ROOT}"

LABEL_ARG=()
if [[ -n "${BENCHMARK_LABEL}" ]]; then
  LABEL_ARG=(--label "${BENCHMARK_LABEL}")
fi

echo "[issue791-bench] Starting camera-ready campaign"
echo "[issue791-bench] Config: ${BENCHMARK_CONFIG_PATH}"
echo "[issue791-bench] Mode: ${BENCHMARK_MODE}"
echo "[issue791-bench] Output root: ${BENCHMARK_OUTPUT_ROOT}"
echo "[issue791-bench] Label: ${BENCHMARK_LABEL:-<none>}"

srun --cpu_bind=cores --gpus-per-node=1 \
  "${PYTHON_BIN}" scripts/tools/run_camera_ready_benchmark.py \
  --config "${BENCHMARK_CONFIG_PATH}" \
  --output-root "${BENCHMARK_OUTPUT_ROOT}" \
  --mode "${BENCHMARK_MODE}" \
  --log-level "${BENCHMARK_LOG_LEVEL}" \
  "${LABEL_ARG[@]}"

echo "[issue791-bench] Benchmark completed. Artifacts will be synced to ${RESULTS_ROOT}"
