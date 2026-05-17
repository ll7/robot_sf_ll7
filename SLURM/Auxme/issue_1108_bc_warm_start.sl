#!/usr/bin/env bash
#SBATCH --job-name=issue1108-bcppo
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue1108-bc-warm-start.out

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue1108-bcppo-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
LOG_LEVEL=${ISSUE1108_LOG_LEVEL:-WARNING}
DATASET_ID=${ISSUE1108_DATASET_ID:-issue_749_b60iopxt_v10_eval_trajectories}
SOURCE_POLICY_ID=${ISSUE1108_SOURCE_POLICY_ID:-ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200}
SCENARIO_CONFIG=${ISSUE1108_SCENARIO_CONFIG:-configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml}
BC_CONFIG=${ISSUE1108_BC_CONFIG:-configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml}
PPO_CONFIG=${ISSUE1108_PPO_CONFIG:-configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml}
EPISODES=${ISSUE1108_EPISODES:-141}
SEEDS=${ISSUE1108_SEEDS:-"111 112 113"}
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue1108] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
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

  echo "[issue1108] module command is unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if command -v srun >/dev/null 2>&1; then
    echo "[issue1108] Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores --gpus-per-node=1 "$@"
  else
    echo "[issue1108] srun unavailable on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}; PATH=${PATH}" >&2
    echo "[issue1108] Running directly in the batch allocation." >&2
    "$@"
  fi
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue1108] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue1108] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[issue1108] uv is not available in PATH." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

for path in "${SCENARIO_CONFIG}" "${BC_CONFIG}" "${PPO_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[issue1108] Required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}
export UV_PROJECT_ENVIRONMENT=${ISSUE1108_UV_ENV_DIR:-${WORKDIR}/uv_env}
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

echo "[issue1108] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[issue1108] ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue1108] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "[issue1108] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[issue1108] DATASET_ID=${DATASET_ID}"
echo "[issue1108] SOURCE_POLICY_ID=${SOURCE_POLICY_ID}"
echo "[issue1108] SCENARIO_CONFIG=${SCENARIO_CONFIG}"
echo "[issue1108] BC_CONFIG=${BC_CONFIG}"
echo "[issue1108] PPO_CONFIG=${PPO_CONFIG}"
echo "[issue1108] EPISODES=${EPISODES}"
echo "[issue1108] SEEDS=${SEEDS}"

uv sync --group imitation

# shellcheck disable=SC2206
SEED_ARGS=(${SEEDS})

run_in_allocation \
  uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id "${DATASET_ID}" \
  --policy-id "${SOURCE_POLICY_ID}" \
  --episodes "${EPISODES}" \
  --scenario-config "${SCENARIO_CONFIG}" \
  --env-config "${PPO_CONFIG}" \
  --seeds "${SEED_ARGS[@]}"

run_in_allocation \
  uv run --group imitation python scripts/training/pretrain_from_expert.py \
  --config "${BC_CONFIG}"

run_in_allocation \
  uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config "${PPO_CONFIG}"

echo "[issue1108] BC warm-start PPO chain completed. Artifacts will sync to ${RESULTS_ROOT}"
