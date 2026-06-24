#!/usr/bin/env bash
#
# Issue #3216 — paper-grade headline 7x7 CI + rank-stability campaign payload.
#
# Plain-language summary: run the increased-seed-budget headline planner
# comparison (7 planners x 7 scenario families) and then attach per-cell
# confidence intervals + rank-stability so the headline ranking can be reported
# with statistical support instead of bare point estimates.
#
# This is the PORTABLE experiment payload (public contract). It carries NO
# cluster directives (#SBATCH/partition/QoS/node names) on purpose: per
# SLURM/Auxme/README.md, cluster execution policy lives in the private ops
# overlay. A private `.sl` wrapper calls this script via srun/bash; this script
# stays reproducible on any host.
#
# Honesty boundary (docs/maintainer_values.md): this payload does NOT and cannot
# self-certify paper-grade. The report harness classifies the bundle
# (paper_grade | nominal | diagnostic | blocked_until_run) and STILL emits
# blocked_until_run until a claim-card review promotes the predeclared S20/S30
# run. Do not present the emitted numbers as paper-grade without that review.
#
# Usage:
#   scripts/benchmark/run_issue3216_headline_campaign.sh [--campaign-id ID] [--output-root DIR]
#
# Env overrides:
#   CAMPAIGN_ID   campaign label (default: issue3216_s20_headline_ci)
#   OUTPUT_ROOT   campaign output base dir (default: output/benchmarks/camera_ready)
#   REPORT_DIR    durable report dir (default: docs/context/evidence/issue_3216_headline_ci_rank_stability)
#   RANK_METRIC   rank metric (default: snqi)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGN_ID="${CAMPAIGN_ID:-issue3216_s20_headline_ci}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/benchmarks/camera_ready}"
REPORT_DIR="${REPORT_DIR:-docs/context/evidence/issue_3216_headline_ci_rank_stability}"
RANK_METRIC="${RANK_METRIC:-snqi}"

CONFIG="configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"

# Parse minimal CLI overrides.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "== [#3216] preflight: config sha256 drift guard =="
# Re-verify the canonical configs referenced by the launch packet (drift guard).
# Keys/values mirror configs/benchmarks/headline_ci_rank_stability_issue_3216_launch_packet.yaml.
drift=0
check_sha() {
  local path="$1" expected="$2" actual
  if [ ! -f "$path" ]; then echo "  MISSING  $path"; drift=1; return; fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" = "$expected" ]; then
    echo "  OK       $path"
  else
    echo "  DRIFT    $path"; echo "    expected $expected"; echo "    actual   $actual"; drift=1
  fi
}
check_sha "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml" "3bf75ebb05cc5523c33271e956b21b14749fc427fc9d986438c83d09eee8b75c"
check_sha "configs/scenarios/classic_interactions_francis2023.yaml"                       "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
check_sha "configs/policy_search/scenario_horizons_h500.yaml"                             "4da9f6eb78c98d1afdf9ac43fde1cbfcac9447edb8491520e8c51d85d3b14fce"
check_sha "configs/benchmarks/seed_sets_v1.yaml"                                          "3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6"
check_sha "configs/benchmarks/paper_experiment_matrix_7planners_v1.yaml"                  "43eec70a34366239d18e2b5e3cb92f8deb5031990acb8ae1f7981b3aa3fed2f7"
if [ "$drift" -ne 0 ]; then
  echo "ERROR: config drift detected — re-verify the launch packet before submitting." >&2
  exit 3
fi

echo "== [#3216] preflight: seed budget resolves (paper_eval_s20 == 20 seeds) =="
uv run python - <<'PY'
import sys, yaml
d = yaml.safe_load(open("configs/benchmarks/seed_sets_v1.yaml"))
def find(node, key):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key: return v
            r = find(v, key)
            if r is not None: return r
    elif isinstance(node, list):
        for v in node:
            r = find(v, key)
            if r is not None: return r
    return None
s20 = find(d, "paper_eval_s20")
seeds = s20.get("seeds") if isinstance(s20, dict) else s20
n = len(seeds) if isinstance(seeds, list) else 0
print(f"  paper_eval_s20 -> {n} seeds")
if n != 20:
    print("  ERROR: paper_eval_s20 must resolve to 20 seeds", file=sys.stderr); sys.exit(4)
PY

echo "== [#3216] campaign: increased-seed-budget headline 7x7 (S20) =="
echo "   config=$CONFIG campaign_id=$CAMPAIGN_ID output_root=$OUTPUT_ROOT"
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config "$CONFIG" \
  --campaign-id "$CAMPAIGN_ID" \
  --output-root "$OUTPUT_ROOT"

echo "== [#3216] locate headline rows =="
ROWS="$(find "$OUTPUT_ROOT" -path '*/reports/headline_rows.json' -newermt '-1 day' 2>/dev/null | sort | tail -1 || true)"
if [ -z "$ROWS" ] || [ ! -f "$ROWS" ]; then
  echo "ERROR: headline_rows.json not found under $OUTPUT_ROOT after campaign." >&2
  echo "       Inspect the campaign output layout and pass the rows file to the report harness manually." >&2
  exit 5
fi
echo "   rows=$ROWS"

echo "== [#3216] report: per-cell CI + rank-stability (fail-closed; never self-certifies paper-grade) =="
mkdir -p "$REPORT_DIR"
uv run python scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py \
  --rows "$ROWS" \
  --rank-metric "$RANK_METRIC" \
  --output-dir "$REPORT_DIR"

echo "== [#3216] done =="
echo "   report:   $REPORT_DIR/result.json"
echo "   markdown: $REPORT_DIR/report.md"
echo "   NOTE: paper-grade promotion requires claim-card review — the harness"
echo "         classifies the bundle but does not promote it. See"
echo "         docs/context/issue_3216_headline_ci_rank_stability.md."
