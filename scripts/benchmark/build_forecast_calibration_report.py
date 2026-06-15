#!/usr/bin/env python3
"""Build a ForecastCalibrationReport.v1 artifact from ForecastMetrics.v1 JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.forecast_calibration_report import (
    build_forecast_calibration_report,
    write_forecast_calibration_report,
)


def main() -> None:
    """Run the calibration report CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric_report", nargs="+", type=Path, help="ForecastMetrics.v1 JSON files")
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--coverage-target", type=float, default=0.9)
    parser.add_argument("--coverage-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    metric_reports = []
    for path in args.metric_report:
        try:
            metric_reports.append(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            parser.error(f"could not read metric report {path}: {exc}")
        except json.JSONDecodeError as exc:
            parser.error(f"could not parse metric report {path}: {exc}")
    try:
        report = build_forecast_calibration_report(
            metric_reports,
            report_id=args.report_id,
            coverage_target=args.coverage_target,
            coverage_tolerance=args.coverage_tolerance,
        )
        paths = write_forecast_calibration_report(
            report,
            json_path=args.out_json,
            markdown_path=args.out_md,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "json_path": str(paths["json"]),
                "markdown_path": str(paths.get("markdown", "")),
                "reliability_rows": len(report["reliability_rows"]),
                "limitation_rows": len(report["limitation_rows"]),
                "decision": report["recommendation"]["decision"],
                "claim_status": report["recommendation"]["claim_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
