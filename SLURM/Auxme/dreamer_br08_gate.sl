#!/usr/bin/env bash
#SBATCH --job-name=d3-br08-gate
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=output/slurm/%j-dreamer-br08-gate.out

set -euo pipefail

PROJECT_ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$(pwd)}")
CONFIG_REL=${DREAMER_BR08_CONFIG:-configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml}
LOG_LEVEL=${DREAMER_BR08_LOG_LEVEL:-WARNING}
PYTHON_BIN=${PROJECT_ROOT}/.venv/bin/python

if ! git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[dreamer-br08-gate] PROJECT_ROOT is not inside a git work tree: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[dreamer-br08-gate] Expected repo virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG_PATH=${CONFIG_REL}
else
  CONFIG_PATH=${PROJECT_ROOT}/${CONFIG_REL}
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[dreamer-br08-gate] Config not found: ${CONFIG_PATH}" >&2
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

echo "[dreamer-br08-gate] Starting DreamerV3 BR-08 gate"
echo "[dreamer-br08-gate] Config: ${CONFIG_PATH}"
echo "[dreamer-br08-gate] Job ID: ${SLURM_JOB_ID:-local}"
echo "[dreamer-br08-gate] Log level: ${LOG_LEVEL}"

uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config "${CONFIG_PATH}" \
  --log-level "${LOG_LEVEL}"