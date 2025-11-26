#!/usr/bin/env bash
# Fast validation smoke for Full Classic Interaction Benchmark (T054)
# Purpose: ensure benchmark orchestration runs end-to-end in smoke mode and produces core artifacts quickly.
# Usage: ./scripts/validation/test_classic_benchmark_full.sh

set -euo pipefail

SCENARIOS=${SCENARIOS:-configs/scenarios/classic_interactions.yaml}
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR=${OUT_DIR:-output/results/validation_classic_smoke/${STAMP}}
SEED=${SEED:-123}
SMOKE=${SMOKE:-0}
export LOGURU_LEVEL=INFO
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}

echo "[classic-benchmark] scenarios=${SCENARIOS} out=${OUT_DIR} seed=${SEED} smoke=${SMOKE}" >&2

ARGS=(
  --scenarios "${SCENARIOS}"
  --output "${OUT_DIR}"
  --seed "${SEED}"
  --initial-episodes 1
  --max-episodes 1
  --batch-size 1
  --horizon 0
  --target-collision-half-width 0.25
  --target-success-half-width 0.25
  --target-snqi-half-width 0.25
  --video-renderer sim-view
  --max-videos 1
)

if [[ "${SMOKE}" != "0" ]]; then
  ARGS+=(--smoke --disable-videos)
fi

uv run python scripts/classic_benchmark_full.py "${ARGS[@]}" || { echo "Benchmark script failed" >&2; exit 1; }

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
  echo "[classic-benchmark] FAILED" >&2
  exit 2
fi

if [[ "${SMOKE}" == "0" && -d "${OUT_DIR}/videos" ]]; then
  video_count=$(find "${OUT_DIR}/videos" -name "*.mp4" -size +10k | wc -l | tr -d ' ')
  if [[ "${video_count}" -lt 1 ]]; then
    echo "Expected at least one rendered video when capture_replay is enabled" >&2
    exit 3
  fi
fi

EP_PATH="${OUT_DIR}/episodes/episodes.jsonl"

EP_PATH="$EP_PATH" SMOKE="$SMOKE" uv run python - <<'PY'
import json
import os
from pathlib import Path

episodes_path = Path(os.environ["EP_PATH"])
smoke = os.environ.get("SMOKE", "0") != "0"

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

metrics = first_record.get("metrics") or {}
for required in ("success_rate", "collision_rate", "path_efficiency"):
    if required not in metrics:
        raise SystemExit(f"Missing metric '{required}' in episode record")

if not smoke and scenario_params.get("map_file"):
    replay_steps = first_record.get("replay_steps") or []
    if len(replay_steps) < 2:
        raise SystemExit("Replay capture missing or too short for full run")

print("Episode metadata validation OK")
PY

echo "[classic-benchmark] PASS" >&2
