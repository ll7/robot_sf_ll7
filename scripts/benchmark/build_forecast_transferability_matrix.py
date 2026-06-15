#!/usr/bin/env python3
"""Build a ForecastTransferabilityStressMatrix.v1 report from metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.forecast_transferability_stress_matrix import (
    build_forecast_transferability_stress_matrix,
    write_forecast_transferability_stress_matrix,
)


def main() -> None:
    """Run the transferability matrix CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric_report", nargs="+", type=Path, help="ForecastMetrics.v1 JSON files")
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument(
        "--generated-at-utc",
        help="Optional deterministic ISO-8601 generation timestamp for reviewable artifacts.",
    )
    args = parser.parse_args()

    metric_reports = []
    for path in args.metric_report:
        try:
            metric_reports.append(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            parser.error(f"could not read metric report {path}: {exc}")
        except json.JSONDecodeError as exc:
            parser.error(f"could not parse metric report {path}: {exc}")
    report = build_forecast_transferability_stress_matrix(
        metric_reports,
        report_id=args.report_id,
        generated_at_utc=args.generated_at_utc,
    )
    paths = write_forecast_transferability_stress_matrix(
        report,
        json_path=args.out_json,
        markdown_path=args.out_md,
    )
    print(
        json.dumps(
            {
                "json_path": str(paths["json"]),
                "markdown_path": str(paths.get("markdown", "")),
                "matrix_rows": len(report["matrix_rows"]),
                "limitation_rows": len(report["limitation_rows"]),
                "decision": report["recommendation"]["decision"],
                "claim_status": report["recommendation"]["claim_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
