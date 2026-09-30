#!/bin/bash
#SBATCH --job-name=ppoabl
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# sbatch ~/ppoabl.sbatch <a1|a2|a3|a4|a5> <full_sha>
# Diagnostic only. This script never submits a job or publishes release evidence.
set -euo pipefail
VARIANT=${1:?variant required}
SHA=${2:?full frozen SHA required}
case "$VARIANT" in a1|a2|a3|a4|a5) ;; *) exit 2 ;; esac
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=disabled
WORK="$HOME/ppoabl_jobs/${VARIANT}-${SLURM_JOB_ID}"
OUT="$HOME/ppoabl_results"
mkdir -p "$HOME/ppoabl_jobs" "$OUT"
test ! -e "$WORK"
git clone --quiet --no-checkout https://github.com/ll7/robot_sf_ll7.git "$WORK"
cd "$WORK"
git fetch --quiet origin "diag/ppo-abl-$VARIANT"
test "$(git rev-parse "origin/diag/ppo-abl-$VARIANT")" = "$SHA"
git checkout --quiet --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"
git submodule update --init --quiet
uv sync --all-extras --quiet
# Require the two checked-in arm profiles to match the requested variant.
.venv/bin/python - "$VARIANT" <<'PY'
import sys
from pathlib import Path
import yaml
for arm in ("ppo", "guarded"):
    profile = Path(f"configs/benchmarks/diagnostics/ppoeval_{arm}_training_plant.yaml")
    assert yaml.safe_load(profile.read_text())["diagnostic_plant_variant"] == sys.argv[1]
PY
UV_NO_SYNC=1 .venv/bin/python scripts/benchmark/run_ppoeval.py --head-sha "$SHA" --output-root "$OUT" --campaign-id "ppoabl-$VARIANT-${SLURM_JOB_ID}" --check-only
UV_NO_SYNC=1 .venv/bin/python scripts/benchmark/run_ppoeval.py --head-sha "$SHA" --output-root "$OUT" --campaign-id "ppoabl-$VARIANT-${SLURM_JOB_ID}"
