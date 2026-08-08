#!/usr/bin/env bash
# SBATCH submission script for issue #5579 MPC tuning sensitivity campaign (#6699)
#
# Implements the canary-first stop rule before production compute:
# 1. Runs preflight config check.
# 2. Runs canary check (seed 101, 6/6 target arm eligibility).
# 3. Stops immediately if canary eligibility fails.
# 4. Runs the bounded tuning phase and freezes its selected target candidates.
# 5. Runs only the frozen held-out production phase after tuning succeeds.
#
# Submit through scripts/dev/sbatch_use_max_time.sh. That wrapper creates output/slurm/ before
# sbatch opens this script's log path; Slurm opens #SBATCH --output before this body runs.
#
#SBATCH --job-name=mpc-tuning-sensitivity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm/%j-mpc-tuning-sensitivity.out

set -euo pipefail

PROJECT_ROOT="$(
  git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null \
    || echo "${SLURM_SUBMIT_DIR:-$(pwd)}"
)"
cd "${PROJECT_ROOT}"

# Keep the conventional artifact directory available for allocations launched through an
# existing wrapper or allocation. The submission wrapper creates it earlier for sbatch itself.
mkdir -p output/slurm

CONFIG_PATH="${CONFIG_PATH:-configs/analysis/issue_5579_mpc_tuning_sensitivity_v2.yaml}"
OUT_DIR="${OUT_DIR:-output/benchmarks/issue_5579_mpc_tuning_sensitivity_v2}"
CANARY_DIR="${OUT_DIR}/canary"

echo "== [issue-5579/6699] MPC Tuning Sensitivity Campaign =="
echo "   config=${CONFIG_PATH}"
echo "   out_dir=${OUT_DIR}"
date

# Threading & headless rendering guards
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"

# Activate environment
if command -v module >/dev/null 2>&1; then
  module purge || true
fi
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "   using UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "   using repo-local .venv"
fi

# --- Phase 1: Preflight Config Check ---
echo "== Phase 1: Config Check =="
uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py \
  --config "${CONFIG_PATH}" \
  --check

# --- Phase 2: Canary Eligibility Gate (6/6 at Seed 101) ---
echo "== Phase 2: Canary Eligibility Check (Seed 101) =="
set +e
CANARY_JSON="$(uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py \
  --config "${CONFIG_PATH}" \
  --out-dir "${CANARY_DIR}" \
  --canary 2>&1)"
CANARY_EXIT=$?
set -e

echo "${CANARY_JSON}"

if [[ ${CANARY_EXIT} -ne 0 ]]; then
  echo "ERROR: Canary eligibility check failed with exit code ${CANARY_EXIT}. Aborting production compute." >&2
  exit "${CANARY_EXIT}"
fi

# --- Phase 3: Tuning Selection ---
echo "== Phase 3: Running Frozen Tuning Scope and Freezing Selection =="
uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py \
  --config "${CONFIG_PATH}" \
  --phase tuning \
  --out-dir "${OUT_DIR}"

# --- Phase 4: Held-Out Production Compute Campaign ---
echo "== Phase 4: Launching Frozen Held-Out Production Campaign =="
uv run python scripts/benchmark/run_mpc_tuning_sensitivity_issue_5579.py \
  --config "${CONFIG_PATH}" \
  --phase held_out \
  --selection-artifact "${OUT_DIR}/tuning_selection.json" \
  --out-dir "${OUT_DIR}"

echo "== Campaign Completed Successfully =="
