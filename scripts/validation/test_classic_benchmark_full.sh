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

EP_PATH="${OUT_DIR}/episodes/episodes.jsonl"
SUMMARY_PATH="${OUT_DIR}/aggregates/summary.json"

EP_PATH="$EP_PATH" SUMMARY_PATH="$SUMMARY_PATH" uv run python - <<'PY'
import json
import os
from pathlib import Path

episodes_path = Path(os.environ["EP_PATH"])
summary_path = Path(os.environ["SUMMARY_PATH"])

with episodes_path.open("r", encoding="utf-8") as fh:
    first_record = None
    for line in fh:
        line = line.strip()
        if not line:
            continue
        first_record = json.loads(line)
        break

if not first_record:
    raise SystemExit("No episode records found for metadata validation")

scenario_params = first_record.get("scenario_params")
if not isinstance(scenario_params, dict):
    raise SystemExit("scenario_params missing or not a mapping in episode record")

algo_top = first_record.get("algo")
algo_nested = scenario_params.get("algo")
if not algo_top or algo_top != algo_nested:
    raise SystemExit("Algorithm metadata not mirrored into scenario_params.algo")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
meta = summary.get("_meta")
if not isinstance(meta, dict):
    raise SystemExit("Aggregation summary missing _meta diagnostics")

if meta.get("group_by") != "scenario_params.algo":
    raise SystemExit("Unexpected aggregation group_by metadata: %r" % (meta.get("group_by"),))

if "effective_group_key" not in meta:
    raise SystemExit("effective_group_key missing from aggregation metadata")

print("Aggregation metadata validation OK")
PY

echo "[classic-benchmark-smoke] PASS" >&2
