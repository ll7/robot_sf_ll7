#!/usr/bin/env bash
#SBATCH --job-name=gse-1977-bcppo
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue1977-bc-warm-start-ppo.out

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue1977-bc-warm-start-ppo-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
LOG_LEVEL=${ISSUE1977_LOG_LEVEL:-WARNING}
WANDB_ARTIFACT=${ISSUE1977_WANDB_ARTIFACT:-ll7/robot_sf/issue_1108_bc_warm_start_job12472:v0}
PPO_CONFIG=${ISSUE1977_PPO_CONFIG:-configs/training/ppo_imitation/ppo_finetune_issue_1977_bc_warm_start_rerun.yaml}
RUN_POLICY_ANALYSIS=${ISSUE1977_RUN_POLICY_ANALYSIS:-1}
USE_SRUN=${ISSUE1977_USE_SRUN:-0}
MODULES_AVAILABLE=0
RUN_OUTPUT_DIR=""

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[issue1977] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
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

  echo "[issue1977] module command is unavailable; continuing without module loads." >&2
  return 0
}

run_in_allocation() {
  if [[ "${USE_SRUN}" == "1" ]] && command -v srun >/dev/null 2>&1; then
    echo "[issue1977] Launching with srun on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}."
    srun --cpu_bind=cores --gpus-per-node=1 "$@"
  else
    echo "[issue1977] Running directly in the batch allocation on node ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}." >&2
    echo "[issue1977] Set ISSUE1977_USE_SRUN=1 to opt into srun; PATH=${PATH}" >&2
    "$@"
  fi
}

ensure_module_command

if [[ "${MODULES_AVAILABLE}" == "1" ]]; then
  module purge
  for mod in ${MODULE_LIST}; do
    [[ -z "${mod}" ]] && continue
    echo "[issue1977] Loading module ${mod}"
    module load "${mod}"
  done
  if [[ -n "${CUDA_MODULE}" ]]; then
    echo "[issue1977] Loading CUDA module ${CUDA_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[issue1977] uv is not available in PATH." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "${PPO_CONFIG}" ]]; then
  echo "[issue1977] PPO config not found: ${PPO_CONFIG}" >&2
  exit 1
fi

mkdir -p "${WORKDIR}" "${LOCAL_OUTPUT_ROOT}"
export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
export ISSUE1977_ARTIFACT_DOWNLOAD_DIR=${WORKDIR}/wandb_artifact
export ISSUE1977_WANDB_ARTIFACT=${WANDB_ARTIFACT}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MPLBACKEND=${MPLBACKEND:-Agg}
export LOGURU_LEVEL=${LOGURU_LEVEL:-${LOG_LEVEL}}
export UV_PROJECT_ENVIRONMENT=${ISSUE1977_UV_ENV_DIR:-${WORKDIR}/uv_env}
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

echo "[issue1977] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[issue1977] ROBOT_SF_ARTIFACT_ROOT=${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue1977] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "[issue1977] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[issue1977] WANDB_ARTIFACT=${WANDB_ARTIFACT}"
echo "[issue1977] PPO_CONFIG=${PPO_CONFIG}"
echo "[issue1977] RUN_POLICY_ANALYSIS=${RUN_POLICY_ANALYSIS}"
echo "[issue1977] USE_SRUN=${USE_SRUN}"
echo "[issue1977] git_commit=$(git rev-parse HEAD)"
echo "[issue1977] git_branch=$(git branch --show-current || true)"

uv sync

uv run python - <<'PY'
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import wandb


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


artifact_name = os.environ["ISSUE1977_WANDB_ARTIFACT"]
download_root = Path(os.environ["ISSUE1977_ARTIFACT_DOWNLOAD_DIR"])
artifact_root = Path(os.environ["ROBOT_SF_ARTIFACT_ROOT"])

api = wandb.Api()
artifact = api.artifact(artifact_name)
source_root = Path(artifact.download(root=str(download_root)))

mappings = {
    "expert_policies/issue_749_bc_preinit_v10_policy.zip": artifact_root
    / "benchmarks/expert_policies/issue_749_bc_preinit_v10_policy.zip",
    "expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.npz": artifact_root
    / "benchmarks/expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.npz",
    "expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.json": artifact_root
    / "benchmarks/expert_trajectories/issue_749_b60iopxt_v10_eval_trajectories.json",
    "ppo_imitation/runs/issue_749_bc_pretrain_v10_warm_start.json": artifact_root
    / "benchmarks/ppo_imitation/runs/issue_749_bc_pretrain_v10_warm_start.json",
}

for relative, destination in mappings.items():
    source = source_root / relative
    if not source.is_file():
        raise FileNotFoundError(f"artifact file missing after download: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[issue1977] hydrated {relative} -> {destination} sha256={sha256(destination)}")

print(f"[issue1977] artifact_digest={artifact.digest}")
PY

run_in_allocation \
  uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config "${PPO_CONFIG}"

if [[ "${RUN_POLICY_ANALYSIS}" == "1" ]]; then
  POLICY_PATH="${ROBOT_SF_ARTIFACT_ROOT}/benchmarks/expert_policies/issue_1977_ppo_finetune_v10_warm_start_rerun_finetuned.zip"
  run_in_allocation \
    uv run python scripts/tools/policy_analysis_run.py \
    --training-config "${PPO_CONFIG}" \
    --policy ppo \
    --model-path "${POLICY_PATH}" \
    --seed-set eval \
    --max-seeds 3 \
    --output "${ROBOT_SF_ARTIFACT_ROOT}/benchmarks/issue1977_policy_analysis" \
    --video-output "${ROBOT_SF_ARTIFACT_ROOT}/recordings/issue1977_policy_analysis" \
    --all
fi

echo "[issue1977] BC warm-start PPO rerun completed. Artifacts will sync to ${RESULTS_ROOT}"
