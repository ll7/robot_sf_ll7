#!/usr/bin/env bash
#
# Public pre-``sbatch`` checkpoint provisioning gate for camera-ready campaigns
# (issue #4613, follow-up #4663).
#
# Plain-language summary: this script materializes + checksum-verifies every
# policy checkpoint a camera-ready campaign needs and writes a per-arm staging
# report next to the submit packet. It runs in seconds on the submit node and
# fails closed *before* sbatch so the S30-style missing-cache failure never
# reaches the compute node. It does not submit Slurm, does not run episodes,
# and is not benchmark evidence -- it is a provisioning-only gate.
#
# The private ops sbatch wrapper must run this gate (or the equivalent
# ``preflight_campaign_checkpoints.py --stage`` command directly) before
# requeueing, and must pass ``--report-path`` into the requeue packet root so
# the staging record travels with the submission metadata.
#
# Usage:
#   scripts/benchmark/submit_camera_ready_checkpoint_gate.sh \
#     --config configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml \
#     --report-path output/benchmarks/camera_ready/<campaign_id>/preflight/checkpoint_staging.json
#
# Env overrides:
#   CONFIG            campaign config YAML (required if --config is omitted)
#   REPORT_PATH       path to write the staging JSON report (optional)
#   CACHE_DIR         optional model-cache override
#   REGISTRY_PATH     optional model-registry path override
#
# Exit codes mirror scripts/benchmark/preflight_campaign_checkpoints.py:
#   0 -- all arm checkpoints staged and checksum-verified; submit-safe=true.
#   2 -- campaign config missing/unreadable.
#   3 -- one or more arm checkpoints are unresolvable (do not submit).
#
# See docs/context/issue_4613_camera_ready_checkpoint_provisioning.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-}"
REPORT_PATH="${REPORT_PATH:-}"
CACHE_DIR="${CACHE_DIR:-}"
REGISTRY_PATH="${REGISTRY_PATH:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --report-path) REPORT_PATH="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --registry-path) REGISTRY_PATH="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo "error: --config (or CONFIG env var) is required" >&2
  exit 2
fi

cmd=(uv run python scripts/benchmark/preflight_campaign_checkpoints.py
  --config "$CONFIG"
  --stage
  --json)
if [ -n "$REPORT_PATH" ]; then
  cmd+=(--report-path "$REPORT_PATH")
fi
if [ -n "$CACHE_DIR" ]; then
  cmd+=(--cache-dir "$CACHE_DIR")
fi
if [ -n "$REGISTRY_PATH" ]; then
  cmd+=(--registry-path "$REGISTRY_PATH")
fi

echo "== [#4613] submit-time checkpoint provisioning gate (stage=True) =="
echo " config=$CONFIG"
if [ -n "$REPORT_PATH" ]; then
  echo " report_path=$REPORT_PATH"
fi

# Catch nonzero exits; do NOT let `set -e` swallow the staging failure signal.
status=0
"${cmd[@]}" || status=$?

if [ "$status" -ne 0 ]; then
  echo "error: checkpoint provisioning gate FAILED with exit $status; do NOT submit sbatch." >&2
  echo "       Stage or promote the missing checkpoint(s) before requeueing." >&2
  exit "$status"
fi

echo "== [#4613] checkpoint gate passed: submit-safe=true =="
exit 0
