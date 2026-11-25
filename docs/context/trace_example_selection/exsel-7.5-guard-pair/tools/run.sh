#!/bin/bash
# usage: EXSEL_WT=<execution worktree> run.sh <config-rel> <output-dir> <label>
set -euo pipefail
WT="${EXSEL_WT:?set EXSEL_WT to the execution worktree}"
cd "$WT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
export PYTHONPATH="$WT:$WT/fast-pysf"
exec "$WT/.venv/bin/python" scripts/tools/run_camera_ready_benchmark.py --config "$1" --output-root "$2" --label "$3" --log-level INFO
