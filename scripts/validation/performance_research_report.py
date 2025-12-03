"""Performance smoke test for research report generation.

Generates a synthetic multi-seed research report and asserts runtime budget
(<120s). Intended for Phase 7 polish validation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from robot_sf.research.orchestrator import ReportOrchestrator

TARGET_SECONDS = float(os.environ.get("RESEARCH_PERF_BUDGET", "120.0"))


def main() -> int:
    start = time.time()
    out_dir = Path(os.environ.get("RESEARCH_REPORT_PERF_DIR", "output/tmp/perf_research_report"))
    out_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = ReportOrchestrator(out_dir)
    seeds = [1, 2, 3]
    metric_records = []
    for s in seeds:
        metric_records.append(
            {
                "seed": s,
                "policy_type": "baseline",
                "reward": 50 + s,
                "timesteps": 1000 + s * 10,
            }
        )
        metric_records.append(
            {
                "seed": s,
                "policy_type": "pretrained",
                "reward": 70 + s,
                "timesteps": 700 + s * 5,
            }
        )
    orchestrator.generate_report(
        experiment_name="perf-smoke",
        metric_records=metric_records,
        run_id="perf1",
        seeds=seeds,
        baseline_timesteps=[1000, 1010, 1020],
        pretrained_timesteps=[700, 705, 710],
        baseline_rewards=[[0, 1, 2], [0, 1.1, 2.1], [0, 1.2, 2.2]],
        pretrained_rewards=[[0, 1.5, 2.5], [0, 1.6, 2.6], [0, 1.7, 2.7]],
        telemetry={"cpu_percent": 12.0, "mem_mb": 55.0},
    )
    elapsed = time.time() - start
    print(f"research report performance elapsed={elapsed:.2f}s")
    if elapsed > TARGET_SECONDS:
        print(f"PERF HARD FAIL: elapsed>{TARGET_SECONDS}s")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
