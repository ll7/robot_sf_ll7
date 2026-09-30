#!/bin/bash
#SBATCH --job-name=hzev-dynamic-window
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Usage: sbatch scripts/validation/hzev_dynamic_window.sbatch <full_sha>
# Modelled on the existing sweep.sbatch; submit only via the orchestrator.
set -euo pipefail
HZEV_SHA=${1:?pass the full diagnostic branch SHA}
[[ $HZEV_SHA =~ ^[0-9a-f]{40}$ ]]
HZEV_WORK=${HZEV_WORK:-"$HOME/hzev"}
HZEV_OUT=${HZEV_OUT:-"$HOME/hzev_data/at_${HZEV_SHA}_${SLURM_JOB_ID}"}
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg
if [[ ! -d $HZEV_WORK/.git ]]; then
    git clone --branch diag/scenario-dynamic-window --single-branch https://github.com/ll7/robot_sf_ll7.git "$HZEV_WORK"
fi
cd "$HZEV_WORK"
[[ -z $(git status --porcelain) ]] || { echo 'HZEV checkout is dirty; use HZEV_WORK for a fresh clone' >&2; exit 2; }
git fetch -q origin "$HZEV_SHA"
git checkout -q --detach "$HZEV_SHA"
test "$(git rev-parse HEAD)" = "$HZEV_SHA"
# Three cheap arms only; no GPU/TensorFlow/Ray installation or inference needed.
uv sync --frozen --extra benchmark --extra viz --extra training -q
export PYTHONPATH="$PWD:$PWD/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HZEV_OUT"
uv pip freeze --python .venv/bin/python > "$HZEV_OUT/environment.txt"
git rev-parse HEAD > "$HZEV_OUT/source_sha.txt"
cp configs/benchmarks/hzev_dynamic_window*.yaml configs/scenarios/hzev_dynamic_window.yaml "$HZEV_OUT/"
# Parent + three workers, respecting this lane's four-process acquisition cap.
.venv/bin/python scripts/validation/run_hzev_dynamic_window.py \
    --head-sha "$HZEV_SHA" --workers 3 --output-dir "$HZEV_OUT" --check-only
.venv/bin/python scripts/validation/run_hzev_dynamic_window.py \
    --head-sha "$HZEV_SHA" --workers 3 --output-dir "$HZEV_OUT"
.venv/bin/python scripts/validation/analyze_hzev_dynamic_window.py \
    --input-dir "$HZEV_OUT" --output "$HZEV_OUT/analysis.json"
# Data lives outside the checkout and survives checkout/worktree cleanup.
tar -czf "${HZEV_OUT}.tar.gz.partial" -C "$(dirname "$HZEV_OUT")" "$(basename "$HZEV_OUT")"
mv "${HZEV_OUT}.tar.gz.partial" "${HZEV_OUT}.tar.gz"
sha256sum "${HZEV_OUT}.tar.gz" > "${HZEV_OUT}.tar.gz.sha256"
echo "HZEV ACQUIRED $HZEV_SHA $HZEV_OUT ${HZEV_OUT}.tar.gz"
