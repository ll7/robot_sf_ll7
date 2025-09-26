#!/usr/bin/env bash
# Fast validation smoke for Full Classic Interaction Benchmark (T054)
# Purpose: ensure benchmark orchestration runs end-to-end in smoke mode and produces core artifacts quickly.
# Usage: ./scripts/validation/test_classic_benchmark_full.sh

set -euo pipefail

SCENARIOS=${SCENARIOS:-configs/scenarios/classic_interactions.yaml}
OUT_DIR=${OUT_DIR:-results/validation_classic_smoke}
SEED=${SEED:-123}

echo "[classic-benchmark-smoke] scenarios=${SCENARIOS} out=${OUT_DIR} seed=${SEED}" >&2

uv run python scripts/classic_benchmark_full.py \
  --scenarios "${SCENARIOS}" \
  --output "${OUT_DIR}" \
  --seed "${SEED}" \
  --smoke \
  --initial-episodes 1 \
  --max-episodes 1 \
  --batch-size 1 \
  --target-collision-half-width 0.5 \
  --target-success-half-width 0.5 \
  --target-snqi-half-width 0.5 \
  --disable-videos || { echo "Benchmark script failed" >&2; exit 1; }

REQUIRED=(
  "${OUT_DIR}/episodes/episodes.jsonl"
  "${OUT_DIR}/aggregates/summary.json"
  "${OUT_DIR}/reports/effect_sizes.json"
  "${OUT_DIR}/reports/statistical_sufficiency.json"
  "${OUT_DIR}/manifest.json"
)

missing=0
for f in "${REQUIRED[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "Missing or empty required artifact: $f" >&2
    missing=1
  fi
done

if [[ $missing -ne 0 ]]; then
  echo "[classic-benchmark-smoke] FAILED" >&2
  exit 2
fi

echo "[classic-benchmark-smoke] PASS" >&2
