#!/bin/bash
#SBATCH --job-name=ppoeval
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# sbatch ppoeval.sbatch <v1|v2|v3> <full_sha>
# Diagnostic only. No sbatch invocation inside this script. Dev seeds only.
set -euo pipefail
VARIANT=${1:?variant required}
SHA=${2:?full frozen SHA required}
case "$VARIANT" in v1|v2|v3) ;; *) exit 2 ;; esac
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=disabled
# Each job owns a fresh clone; concurrent variants never change a shared checkout.
WORK="$HOME/ppoeval_jobs/${VARIANT}-${SLURM_JOB_ID}"
OUT="$HOME/ppoeval_results"
mkdir -p "$HOME/ppoeval_jobs" "$OUT"
test ! -e "$WORK"
git clone --quiet --no-checkout https://github.com/ll7/robot_sf_ll7.git "$WORK"
cd "$WORK"
git fetch --quiet origin "diag/ppo-sem-$VARIANT"
git checkout --quiet --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"
git submodule update --init --quiet
uv sync --all-extras --quiet
# Hydrate and checksum registry assets before campaign preflight, without credentials in logs.
UV_NO_SYNC=1 .venv/bin/python scripts/benchmark/run_ppoeval.py --head-sha "$SHA" --output-root "$OUT" --campaign-id "$VARIANT-${SLURM_JOB_ID}" --check-only
UV_NO_SYNC=1 .venv/bin/python scripts/benchmark/run_ppoeval.py --head-sha "$SHA" --output-root "$OUT" --campaign-id "$VARIANT-${SLURM_JOB_ID}"
