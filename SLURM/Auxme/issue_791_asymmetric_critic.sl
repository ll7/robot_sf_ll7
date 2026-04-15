#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-issue791-asymmetric-critic
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm/%j-issue791-asymmetric-critic.out

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
LOCAL_OUTPUT_ROOT=${PROJECT_ROOT}/output/slurm
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-${LOCAL_OUTPUT_ROOT}}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
RESULTS_ROOT=${AUXME_RESULTS_DIR:-${LOCAL_OUTPUT_ROOT}/issue791-asymmetric-critic-job-${SLURM_JOB_ID}}
WORKDIR=${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}
TRAIN_CONFIG=${ISSUE791_TRAIN_CONFIG:-}
LOG_LEVEL=${ISSUE791_LOG_LEVEL:-INFO}
ISSUE791_WANDB_POLICY=${ISSUE791_WANDB_POLICY:-auto}
ISSUE791_REQUIRE_WANDB=${ISSUE791_REQUIRE_WANDB:-}
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

if [[ -z "${TRAIN_CONFIG}" ]]; then
  echo "[issue791] ISSUE791_TRAIN_CONFIG is required for this wrapper." >&2
  echo "[issue791] Example: ISSUE791_TRAIN_CONFIG=configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22.yaml sbatch $0" >&2
  exit 2
fi

# Resolve to an absolute config path so policy checks and launch use the same file.
if [[ "${TRAIN_CONFIG}" = /* ]]; then
  TRAIN_CONFIG_PATH=${TRAIN_CONFIG}
else
  TRAIN_CONFIG_PATH=${PROJECT_ROOT}/${TRAIN_CONFIG}
fi

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "[issue791] Training config not found: ${TRAIN_CONFIG_PATH}" >&2
  exit 1
fi

detect_wandb_enabled() {
  local config_path="$1"
  awk '
    BEGIN { in_wandb = 0 }
    {
      line = $0
      if (line ~ /^[[:space:]]*wandb:[[:space:]]*$/) {
        in_wandb = 1
        next
      }
      if (in_wandb && line ~ /^[^[:space:]]/) {
        in_wandb = 0
      }
      if (in_wandb && line ~ /^[[:space:]]*enabled:[[:space:]]*/) {
        sub(/^[[:space:]]*enabled:[[:space:]]*/, "", line)
        gsub(/[[:space:]]+/, "", line)
        print tolower(line)
        exit
      }
    }
  ' "${config_path}"
}

resolve_wandb_requirement() {
  if [[ -n "${ISSUE791_REQUIRE_WANDB}" ]]; then
    case "${ISSUE791_REQUIRE_WANDB,,}" in
      1|true|yes|on) echo "true" ;;
      0|false|no|off) echo "false" ;;
      *)
        echo "[issue791] ISSUE791_REQUIRE_WANDB must be true/false-like, got: ${ISSUE791_REQUIRE_WANDB}" >&2
        exit 1
        ;;
    esac
    return
  fi

  case "${ISSUE791_WANDB_POLICY}" in
    auto)
      if [[ "${TRAIN_CONFIG_PATH}" == *followup* || "${TRAIN_CONFIG_PATH}" == *promotion* ]]; then
        echo "true"
      else
        echo "false"
      fi
      ;;
    require)
      echo "true"
      ;;
    allow-off)
      echo "false"
      ;;
    *)
      echo "[issue791] Unsupported ISSUE791_WANDB_POLICY='${ISSUE791_WANDB_POLICY}'. Use auto|require|allow-off." >&2
      exit 1
      ;;
  esac
}

WAND_ENABLED=$(detect_wandb_enabled "${TRAIN_CONFIG_PATH}")
if [[ -z "${WAND_ENABLED}" ]]; then
  echo "[issue791] Could not resolve tracking.wandb.enabled from ${TRAIN_CONFIG_PATH}" >&2
  exit 1
fi
WAND_REQUIRED=$(resolve_wandb_requirement)

echo "[issue791] WandB policy: ${ISSUE791_WANDB_POLICY}"
echo "[issue791] WandB required: ${WAND_REQUIRED}"
echo "[issue791] WandB enabled in config: ${WAND_ENABLED}"

if [[ "${WAND_REQUIRED}" == "true" && "${WAND_ENABLED}" != "true" ]]; then
  echo "[issue791] Refusing to launch: promotion/follow-up runs require tracking.wandb.enabled: true." >&2
  echo "[issue791] Set ISSUE791_WANDB_POLICY=allow-off only for explicit stage-gate debugging." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[issue791] Expected repo virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

export ROBOT_SF_ARTIFACT_ROOT=${WORKDIR}/results
mkdir -p "${ROBOT_SF_ARTIFACT_ROOT}"
RUN_OUTPUT_DIR=${ROBOT_SF_ARTIFACT_ROOT}

cd "${PROJECT_ROOT}"

echo "[issue791] Starting training with config: ${TRAIN_CONFIG_PATH}"
echo "[issue791] Output root: ${ROBOT_SF_ARTIFACT_ROOT}"
echo "[issue791] Job ID: ${SLURM_JOB_ID}"
echo "[issue791] Log level: ${LOG_LEVEL}"

# Reusable intent: stage gates can opt out of WandB, but follow-up/promotion runs should be tracked.
# Override with ISSUE791_WANDB_POLICY=require|allow-off or ISSUE791_REQUIRE_WANDB=true|false.

srun --cpu_bind=cores --gpus-per-node=1 \
  "${PYTHON_BIN}" scripts/training/train_ppo.py \
  --config "${TRAIN_CONFIG_PATH}" \
  --log-level "${LOG_LEVEL}"

echo "[issue791] Training completed. Artifacts will be synced to ${RESULTS_ROOT}"
