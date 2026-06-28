#!/usr/bin/env bash
#
# Issue #3425 empirical vertical-slice runner.
#
# Plain-language summary: prepare and optionally run the small baseline-safe
# planner comparison that exercises the SLURM-to-claim finalizer path.
#
# This script contains no SBATCH directives or private cluster mechanics. A
# private operations wrapper may call it from an approved SLURM-capable public
# worktree. By default it only generates the research packet and camera-ready
# preflight artifacts; set RUN_CAMPAIGN=1 inside the approved execution route to
# roll benchmark episodes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGN_CONFIG="${CAMPAIGN_CONFIG:-configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml}"
RESEARCH_MANIFEST="${RESEARCH_MANIFEST:-configs/benchmarks/issue_3425_empirical_vertical_slice_manifest.yaml}"
CAMPAIGN_ID="${CAMPAIGN_ID:-issue_3425_empirical_vertical_slice}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/benchmarks/issue_3425_empirical_vertical_slice_smoke}"
PACKET_DIR="${PACKET_DIR:-output/research_campaign_packets/issue_3425_empirical_vertical_slice}"

echo "== [#3425] generate research campaign packet =="
uv run python scripts/validation/run_research_campaign_manifest.py \
  "$RESEARCH_MANIFEST" \
  --output-dir "$PACKET_DIR"

echo "== [#3425] camera-ready campaign preflight =="
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config "$CAMPAIGN_CONFIG" \
  --campaign-id "$CAMPAIGN_ID" \
  --output-root "$OUTPUT_ROOT" \
  --mode preflight

if [[ "${RUN_CAMPAIGN:-0}" != "1" ]]; then
  echo "== [#3425] readiness only; set RUN_CAMPAIGN=1 on the approved execution route to run episodes =="
  exit 0
fi

echo "== [#3425] run bounded benchmark campaign =="
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config "$CAMPAIGN_CONFIG" \
  --campaign-id "$CAMPAIGN_ID" \
  --output-root "$OUTPUT_ROOT"
