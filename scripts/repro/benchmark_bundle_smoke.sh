#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export DISPLAY="${DISPLAY:-}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-WARNING}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYGAME_HIDE_SUPPORT_PROMPT="${PYGAME_HIDE_SUPPORT_PROMPT:-1}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

OUTPUT_ROOT="${ROBOT_SF_REPRO_OUTPUT_ROOT:-output/docker_repro/benchmark_bundle_smoke}"
MATRIX="${ROBOT_SF_REPRO_MATRIX:-configs/scenarios/planner_sanity_matrix_v1.yaml}"
ALGO="${ROBOT_SF_REPRO_ALGO:-goal}"
HORIZON="${ROBOT_SF_REPRO_HORIZON:-300}"
REPEATS="${ROBOT_SF_REPRO_REPEATS:-1}"
WORKERS="${ROBOT_SF_REPRO_WORKERS:-1}"

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

uv run robot_sf_bench validate-config \
  --matrix "$MATRIX" \
  > "$OUTPUT_ROOT/validate_config.json" \
  2> "$OUTPUT_ROOT/validate_config.log"

uv run robot_sf_bench preview-scenarios \
  --matrix "$MATRIX" \
  > "$OUTPUT_ROOT/preview_scenarios.json" \
  2> "$OUTPUT_ROOT/preview_scenarios.log"

uv run robot_sf_bench --quiet run \
  --matrix "$MATRIX" \
  --out "$OUTPUT_ROOT/episodes.jsonl" \
  --algo "$ALGO" \
  --repeats "$REPEATS" \
  --horizon "$HORIZON" \
  --workers "$WORKERS" \
  --no-video \
  --no-resume \
  --fail-fast \
  --benchmark-profile baseline-safe \
  --external-log-noise suppress \
  --structured-output json \
  > "$OUTPUT_ROOT/run_summary.json" \
  2> "$OUTPUT_ROOT/run.log"

uv run robot_sf_bench aggregate \
  --in "$OUTPUT_ROOT/episodes.jsonl" \
  --out "$OUTPUT_ROOT/summary.json" \
  > "$OUTPUT_ROOT/aggregate.log" \
  2>&1

python - "$OUTPUT_ROOT" "$MATRIX" "$ALGO" "$HORIZON" "$REPEATS" "$WORKERS" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

output_root = Path(sys.argv[1])
matrix = sys.argv[2]
algo = sys.argv[3]
horizon = int(sys.argv[4])
repeats = int(sys.argv[5])
workers = int(sys.argv[6])

required = [
    "validate_config.json",
    "preview_scenarios.json",
    "run_summary.json",
    "episodes.jsonl",
    "summary.json",
]

artifacts: dict[str, dict[str, object]] = {}
for name in required:
    path = output_root / name
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"required artifact missing or empty: {path}")
    artifacts[name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

episode_count = sum(1 for line in (output_root / "episodes.jsonl").read_text().splitlines() if line)
if episode_count < 1:
    raise SystemExit("episodes.jsonl did not contain any episode records")

try:
    import os

    commit = os.environ.get("ROBOT_SF_REPRO_GIT_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    commit = "unknown"

manifest = {
    "schema": "robot-sf-benchmark-docker-repro-smoke.v1",
    "created_at_utc": datetime.now(UTC).isoformat(),
    "git_commit": commit,
    "matrix": matrix,
    "algo": algo,
    "horizon": horizon,
    "repeats": repeats,
    "workers": workers,
    "episode_count": episode_count,
    "artifacts": artifacts,
    "limitations": [
        "CPU/headless smoke slice only; it is not a full benchmark campaign.",
        "No GPU determinism claim is made.",
        "Floating-point and physics-library behavior may vary across CPU architectures.",
    ],
}
(output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "Benchmark Docker reproduction smoke artifacts: $OUTPUT_ROOT"
