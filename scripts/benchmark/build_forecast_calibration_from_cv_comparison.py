#!/usr/bin/env python3
"""Build ForecastCalibrationReport.v1 from a CV forecast comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.forecast_calibration_report import (
    build_forecast_calibration_report,
    write_forecast_calibration_report,
)


def main() -> None:
    """Run the comparison-to-calibration report CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_report", type=Path)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--coverage-target", type=float, default=0.9)
    parser.add_argument("--coverage-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--generated-at-utc",
        help="Optional deterministic ISO-8601 generation timestamp for reviewable artifacts.",
    )
    args = parser.parse_args()

    try:
        comparison = json.loads(args.comparison_report.read_text(encoding="utf-8"))
        metric_reports = forecast_metric_reports_from_comparison(comparison)
        report = build_forecast_calibration_report(
            metric_reports,
            report_id=args.report_id,
            coverage_target=args.coverage_target,
            coverage_tolerance=args.coverage_tolerance,
            generated_at_utc=args.generated_at_utc,
        )
        report["source_report"] = str(args.comparison_report)
        paths = write_forecast_calibration_report(
            report,
            json_path=args.out_json,
            markdown_path=args.out_md,
        )
    except OSError as exc:
        parser.error(f"could not read comparison report {args.comparison_report}: {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"could not parse comparison report {args.comparison_report}: {exc}")
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


def forecast_metric_reports_from_comparison(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert comparison rows into ForecastMetrics.v1-shaped reports.

    The comparison report only contains aggregated 1s reliability values. The
    converter preserves that bounded scope and marks unavailable rows with zero
    denominators instead of inventing missing uncertainty evidence.
    """
    rows = comparison.get("comparison_rows")
    if not isinstance(rows, list):
        raise ValueError("comparison report must contain comparison_rows")

    metric_reports = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("comparison_rows entries must be mappings")
        metric_reports.append(_metric_report_from_comparison_row(row))
    return metric_reports


def _metric_report_from_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    baseline = str(_required(row, "baseline"))
    family = str(_required(row, "family"))
    label = str(_required(row, "label"))
    samples = int(float(row.get("evaluable_samples") or 0.0))
    metadata_presence = str(row.get("metadata_presence") or "unknown")
    status = "ok" if str(row.get("status")) == "evaluated" and samples > 0 else "unavailable"
    denominator = samples if status == "ok" else 0
    return {
        "schema_version": "ForecastMetrics.v1",
        "provenance": {
            "predictor_id": baseline,
            "predictor_family": baseline,
            "scenario_id": label,
            "scenario_family": family,
            "observation_tier": "deployable_tracked",
            "dt_s": 0.1,
            "horizons_s": [1.0],
            "semantic_metadata_present": metadata_presence,
            "source_report": "cv_forecast_comparison",
        },
        "aggregate_rows": [
            _metric_row(
                metric="mean_coverage",
                value=row.get("mean_within_95ci_1s"),
                status=status,
                denominator=denominator,
                family=family,
                label=label,
                metadata_presence=metadata_presence,
            ),
            _metric_row(
                metric="mean_likelihood",
                value=_likelihood_from_nll(row.get("mean_negative_log_likelihood_1s")),
                status=status,
                denominator=denominator,
                family=family,
                label=label,
                metadata_presence=metadata_presence,
            ),
            _metric_row(
                metric="mean_expected_ade",
                value=row.get("mean_ade_1s"),
                status=status,
                denominator=denominator,
                family=family,
                label=label,
                metadata_presence=metadata_presence,
            ),
            _metric_row(
                metric="mean_miss_rate",
                value=row.get("mean_miss_rate_1s"),
                status=status,
                denominator=denominator,
                family=family,
                label=label,
                metadata_presence=metadata_presence,
            ),
        ],
    }


def _metric_row(
    *,
    metric: str,
    value: Any,
    status: str,
    denominator: int,
    family: str,
    label: str,
    metadata_presence: str,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "horizon_s": 1.0,
        "value": None if value is None else float(value),
        "status": status if value is not None else "unavailable",
        "denominator": denominator if value is not None else 0,
        "actor_class": "unavailable",
        "scenario_id": label,
        "scenario_family": family,
        "observation_tier": "deployable_tracked",
        "semantic_metadata_present": metadata_presence,
        "dt_s": 0.1,
    }


def _likelihood_from_nll(value: Any) -> float | None:
    return None if value is None else -float(value)


def _required(row: dict[str, Any], key: str) -> Any:
    if key not in row or row[key] is None:
        raise ValueError(f"comparison row missing required field: {key}")
    return row[key]


if __name__ == "__main__":
    main()
