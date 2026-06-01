#!/usr/bin/env bash
#SBATCH --job-name=issue1474-shielded-ppo
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue1474-shielded-ppo-repair.out
#SBATCH --error=output/slurm/%j-issue1474-shielded-ppo-repair.err

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
RESULTS_ROOT=${ISSUE1474_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue1474-shielded-ppo-repair-job-${SLURM_JOB_ID:-local}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/issue1474-shielded-ppo-repair-${SLURM_JOB_ID:-local}}
LAUNCH_PACKET=${ISSUE1474_LAUNCH_PACKET:-configs/training/shielded_ppo_issue_1396_launch_packet.yaml}
TRAIN_CONFIG=${ISSUE1474_TRAIN_CONFIG:-configs/training/ppo/ablations/expert_ppo_issue_1474_shielded_repair_collision20_5m.yaml}
LOG_LEVEL=${ISSUE1474_LOG_LEVEL:-INFO}
RUN_OUTPUT_DIR=""

log() {
  echo "[issue1474-shielded-ppo] $*"
}

die() {
  echo "[issue1474-shielded-ppo] $*" >&2
  exit 1
}

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

run_in_allocation() {
  if [[ "${ISSUE1474_USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log "Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores --gpus-per-node=1 "$@"
  else
    log "Running directly inside the batch allocation; set ISSUE1474_USE_SRUN=1 to opt into srun."
    "$@"
  fi
}

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "PROJECT_ROOT is not inside a git worktree: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"

for path in "${LAUNCH_PACKET}" "${TRAIN_CONFIG}"; do
  [[ -f "${path}" ]] || die "Required path not found: ${path}"
done

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
export UV_PROJECT_ENVIRONMENT=${ISSUE1474_UV_ENV_DIR:-${WORKDIR}/uv_env}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

log "PROJECT_ROOT=${PROJECT_ROOT}"
log "ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
log "RESULTS_ROOT=${RESULTS_ROOT}"
log "LAUNCH_PACKET=${LAUNCH_PACKET}"
log "TRAIN_CONFIG=${TRAIN_CONFIG}"
log "git_branch=$(git branch --show-current)"
log "git_commit=$(git rev-parse HEAD)"

uv sync --group imitation

run_in_allocation \
  uv run python scripts/validation/validate_shielded_ppo_launch_packet.py \
  --config "${LAUNCH_PACKET}" \
  --json

train_args=(
  uv run python scripts/training/train_ppo.py
  --config "${TRAIN_CONFIG}"
  --log-level "${LOG_LEVEL}"
  --log-file "${ROBOT_SF_ARTIFACT_ROOT}/issue1474-shielded-ppo-training.log"
)

if [[ "${ISSUE1474_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  train_args+=(--dry-run)
fi

run_in_allocation "${train_args[@]}"

log "Shielded PPO repair training completed. Artifacts will sync to ${RESULTS_ROOT}"
