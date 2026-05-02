#!/usr/bin/env bash
#SBATCH --job-name=d3-br08-full
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=output/slurm/%j-dreamer-br08-full.out

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
CONFIG_REL=${DREAMER_BR08_CONFIG:-configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml}
LOG_LEVEL=${DREAMER_BR08_LOG_LEVEL:-WARNING}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[dreamer-br08-full] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[dreamer-br08-full] Expected repo virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG_PATH=${CONFIG_REL}
else
  CONFIG_PATH=${PROJECT_ROOT}/${CONFIG_REL}
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[dreamer-br08-full] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/output/slurm"

cd "${PROJECT_ROOT}"
source .venv/bin/activate

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONWARNINGS=ignore

echo "[dreamer-br08-full] Starting DreamerV3 BR-08 full run"
echo "[dreamer-br08-full] Config: ${CONFIG_PATH}"
echo "[dreamer-br08-full] Job ID: ${SLURM_JOB_ID:-local}"
echo "[dreamer-br08-full] Log level: ${LOG_LEVEL}"

uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config "${CONFIG_PATH}" \
  --log-level "${LOG_LEVEL}"