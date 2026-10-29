#!/usr/bin/env bash
# Run one (cell, arms, seed) campaign at the execution commit; one campaign per call.
# usage: run_one.sh <wt> <out_root> <campaign_id> <cell> <seed> [--trace] <arm> [<arm> ...]
set -euo pipefail
WT="$1"; OUT="$2"; CID="$3"; CELL="$4"; SEED="$5"; shift 5
TRACE=0
if [[ "${1:-}" == "--trace" ]]; then TRACE=1; shift; fi
LANE="$(cd "$(dirname "$0")" && pwd)"
RELEASE_CONFIG=configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml
mkdir -p "$OUT/$CID"
EXTRA=()
if ((TRACE)); then EXTRA=(--trace); fi
cd "$WT"
.venv/bin/python "$LANE/make_config.py" --release-config "$RELEASE_CONFIG" --cell "$CELL" \
  --arms "$@" --seed "$SEED" --name "$CID" "${EXTRA[@]}" --out "$OUT/$CID/config.yaml"
export LOGURU_LEVEL=WARNING OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$OUT/$CID/mpl" TMPDIR="$OUT/$CID/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
start=$(date +%s.%N)
set +e
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py --config "$OUT/$CID/config.yaml" \
  --output-root "$OUT/$CID/out" --campaign-id "$CID" --skip-publication-bundle \
  > "$OUT/$CID/run.log" 2>&1
rc=$?
set -e
end=$(date +%s.%N)
printf '{"campaign_id": "%s", "rc": %d, "wall_s": %.2f}\n' "$CID" "$rc" "$(echo "$end - $start" | bc)" > "$OUT/$CID/receipt.json"
rm -rf "$TMPDIR"
exit "$rc"
