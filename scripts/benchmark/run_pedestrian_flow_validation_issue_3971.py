#!/usr/bin/env python3
"""Run the issue #3971 diagnostic-only pedestrian flow validation harness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.pedestrian_flow_validation import (
    PedFlowRunConfig,
    run_pedestrian_flow_validation,
    write_pedestrian_flow_report,
)


def main() -> int:
    """Parse CLI arguments, run the smoke harness, and write compact evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/issue_3971_pedestrian_flow_validation_smoke.yaml"),
        help="YAML run configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary.json, README.md, and trajectory_quality.csv.",
    )
    parser.add_argument("--log-level", default="WARNING", help="Accepted for script parity.")
    args = parser.parse_args()

    payload = _load_yaml(args.config)
    run_config = PedFlowRunConfig(
        duration_s=float(payload.get("duration_s", 2.0)),
        dt_s=float(payload.get("dt_s", 0.1)),
        pedestrian_counts=tuple(int(v) for v in payload.get("pedestrian_counts", (2, 6))),
        seed=int(payload.get("seed", 3971)),
        speed_mps=float(payload.get("speed_mps", 1.1)),
    )
    scenarios = tuple(payload.get("scenarios", ())) or None
    output_dir = args.output_dir or Path(
        payload.get(
            "output_dir",
            "output/benchmarks/issue3971_pedestrian_flow_validation",
        )
    )

    report = run_pedestrian_flow_validation(config=run_config, scenarios=scenarios)
    written = write_pedestrian_flow_report(report, output_dir)
    print(f"summary_json={written['summary_json']}")
    print(f"summary_md={written['summary_md']}")
    print(f"trajectory_quality_csv={written['trajectory_quality_csv']}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if payload.get("diagnostic_only_not_benchmark_gate") is not True:
        raise ValueError("config must set diagnostic_only_not_benchmark_gate: true")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
