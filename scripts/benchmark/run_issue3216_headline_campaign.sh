#!/usr/bin/env bash
#
# Issue #3216 - paper-grade headline 7x7 CI + rank-stability campaign payload.
#
# Plain-language summary: run the increased-seed-budget headline planner
# comparison (7 planners x 7 scenario families), then compute per-cell
# confidence intervals and rank-stability so headline rankings are reported
# with statistical support instead of bare point estimates.
#
# This portable experiment payload carries no cluster directives
# (#SBATCH/partition/QoS/node names). Cluster execution policy lives in the
# private ops overlay; a private `.sl` wrapper may call this script via
# srun/bash, while this script stays reproducible on any host.
#
# Honesty boundary (docs/maintainer_values.md): this payload cannot
# self-certify paper-grade. The report harness classifies the bundle
# (paper_grade | nominal | diagnostic | blocked_until_run), but emitted
# numbers must not be presented as paper-grade without claim-card review.
#
# Usage:
# scripts/benchmark/run_issue3216_headline_campaign.sh [--preflight-only] [--campaign-id ID] [--output-root DIR]
#
# Env overrides:
# CAMPAIGN_ID campaign label (default: issue3216_s20_headline_ci)
# OUTPUT_ROOT campaign output base dir (default: output/benchmarks/camera_ready)
# REPORT_DIR durable report dir (default: docs/context/evidence/issue_3216_headline_ci_rank_stability)
# RANK_METRIC rank metric (default: snqi)
# CONFIG benchmark config used for planner-row coverage preflight

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGN_ID="${CAMPAIGN_ID:-issue3216_s20_headline_ci}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/benchmarks/camera_ready}"
REPORT_DIR="${REPORT_DIR:-docs/context/evidence/issue_3216_headline_ci_rank_stability}"
RANK_METRIC="${RANK_METRIC:-snqi}"
PREFLIGHT_ONLY=0
CONFIG="configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "== [#3216] preflight: sha256 drift guard =="
# Re-verify canonical configs referenced by the launch packet.
# Keys/values mirror configs/benchmarks/headline_ci_rank_stability_issue_3216_launch_packet.yaml.
drift=0
check_sha() {
  local path="$1" expected="$2" actual
  if [ ! -f "$path" ]; then
    echo " MISSING $path"
    drift=1
    return
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" = "$expected" ]; then
    echo " OK $path"
  else
    echo " DRIFT $path"
    echo " expected $expected"
    echo " actual $actual"
    drift=1
  fi
}
check_sha "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml" "249968670830c3b710e1407aecb2256d140cd1ba76294766421447bd89357f15"
check_sha "configs/scenarios/classic_interactions_francis2023.yaml" "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
check_sha "configs/policy_search/scenario_horizons_h500.yaml" "4da9f6eb78c98d1afdf9ac43fde1cbfcac9447edb8491520e8c51d85d3b14fce"
check_sha "configs/benchmarks/seed_sets_v1.yaml" "3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6"
check_sha "configs/benchmarks/paper_experiment_matrix_7planners_v1.yaml" "43eec70a34366239d18e2b5e3cb92f8deb5031990acb8ae1f7981b3aa3fed2f7"
if [ "$drift" -ne 0 ]; then
  echo "ERROR: drift detected; re-verify launch packet before submitting." >&2
  exit 3
fi

echo "== [#3216] preflight: seed budget resolves (paper_eval_s20 == 20 seeds) =="
uv run python - <<'PY'
import sys

import yaml

with open("configs/benchmarks/seed_sets_v1.yaml", encoding="utf-8") as handle:
    data = yaml.safe_load(handle)


def find(node, key):
    if isinstance(node, dict):
        for item_key, value in node.items():
            if item_key == key:
                return value
            result = find(value, key)
            if result is not None:
                return result
    if isinstance(node, list):
        for value in node:
            result = find(value, key)
            if result is not None:
                return result
    return None


s20 = find(data, "paper_eval_s20")
seeds = s20.get("seeds") if isinstance(s20, dict) else s20
seed_count = len(seeds) if isinstance(seeds, list) else 0
print(f" paper_eval_s20 -> {seed_count} seeds")
if seed_count != 20:
    print(" ERROR: paper_eval_s20 must resolve 20 seeds", file=sys.stderr)
    sys.exit(4)
PY

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
  echo "== [#3216] local preflight only: deterministic CI/rank-stability dry-run =="
  mkdir -p "$REPORT_DIR"
  status=0
  uv run python scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py \
    --dry-run \
    --rank-metric "$RANK_METRIC" \
    --expected-planners-from-config "$CONFIG" \
    --output-dir "$REPORT_DIR" \
    --fail-on-decision-blocker || status=$?
  if [ "$status" -ne 0 ] && [ "$status" -ne 4 ]; then
    exit "$status"
  fi
  echo " preflight_report=$REPORT_DIR/result.json"
  echo " preflight_status=$status"
  echo " NOTE: exit 4 from the report builder is expected when the local dry-run"
  echo " still blocks manuscript/S30 decisions; no campaign or Slurm submission ran."
  exit 0
fi

echo "== [#3216] campaign: increased-seed-budget headline 7x7 (S20) =="
echo " config=$CONFIG campaign_id=$CAMPAIGN_ID output_root=$OUTPUT_ROOT"
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config "$CONFIG" \
  --campaign-id "$CAMPAIGN_ID" \
  --output-root "$OUTPUT_ROOT"

echo "== [#3216] locate headline rows =="
ROWS="$(find "$OUTPUT_ROOT" -path '*/reports/headline_rows.json' -newermt '-1 day' 2>/dev/null | sort | tail -1 || true)"
if [ -z "$ROWS" ] || [ ! -f "$ROWS" ]; then
  echo "ERROR: headline_rows.json not found under $OUTPUT_ROOT after campaign." >&2
  echo " Inspect campaign output layout and run the rows report harness manually." >&2
  exit 5
fi
echo " rows=$ROWS"

echo "== [#3216] report: per-cell CI + rank-stability (fail-closed; never self-certifies paper-grade) =="
mkdir -p "$REPORT_DIR"
uv run python scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py \
  --rows "$ROWS" \
  --rank-metric "$RANK_METRIC" \
  --expected-planners-from-config "$CONFIG" \
  --output-dir "$REPORT_DIR"

echo "== [#3216] done =="
echo " report: $REPORT_DIR/result.json"
echo " markdown: $REPORT_DIR/report.md"
echo " NOTE: paper-grade promotion requires claim-card review; harness"
echo " classifies the bundle but does not promote it. See"
echo " docs/context/issue_3216_headline_ci_rank_stability.md."
