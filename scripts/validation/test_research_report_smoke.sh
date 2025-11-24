#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-output/tmp/smoke_research_report}"
echo "[Smoke] Generating synthetic research report at ${OUT_DIR}..."
mkdir -p "$(dirname "${OUT_DIR}")"
rm -rf "${OUT_DIR}"
export RESEARCH_SMOKE_OUT_DIR="${OUT_DIR}"

uv run python - <<'PY'
from pathlib import Path
import os
from robot_sf.research.orchestrator import ReportOrchestrator

out_dir = Path(os.environ["RESEARCH_SMOKE_OUT_DIR"])
seeds = [1, 2, 3]
metric_records = []
for s in seeds:
    metric_records.append(
        {"seed": s, "policy_type": "baseline", "success_rate": 0.7, "collision_rate": 0.1, "timesteps_to_convergence": 1000 + s}
    )
    metric_records.append(
        {"seed": s, "policy_type": "pretrained", "success_rate": 0.82, "collision_rate": 0.06, "timesteps_to_convergence": 720 + s}
    )

orch = ReportOrchestrator(out_dir)
report_path = orch.generate_report(
    experiment_name="Smoke Test",
    metric_records=metric_records,
    run_id="smoke_run",
    seeds=seeds,
    baseline_timesteps=[r["timesteps_to_convergence"] for r in metric_records if r["policy_type"] == "baseline"],
    pretrained_timesteps=[r["timesteps_to_convergence"] for r in metric_records if r["policy_type"] == "pretrained"],
    telemetry={"smoke": True},
    generate_figures=False,
)
print(f"[Smoke] Report generated at {report_path}")
PY

echo "[Smoke] Listing artifacts:"
ls -1 "${OUT_DIR}/data" || true

test -f "${OUT_DIR}/data/metrics.json"
test -f "${OUT_DIR}/data/hypothesis.json"
test -f "${OUT_DIR}/metadata.json"
test -f "${OUT_DIR}/report.md"

echo "[Smoke] Research report smoke test PASSED"
