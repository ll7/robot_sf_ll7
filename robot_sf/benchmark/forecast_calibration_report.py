"""Calibration and reliability summaries for ForecastMetrics.v1 artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.forecast_metrics import FORECAST_METRICS_SCHEMA_VERSION

FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION = "ForecastCalibrationReport.v1"


def build_forecast_calibration_report(
    metric_reports: list[dict[str, Any]],
    *,
    report_id: str,
    coverage_target: float = 0.9,
    coverage_tolerance: float = 0.05,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a calibration/reliability report from forecast metric reports.

    Args:
        metric_reports: ForecastMetrics.v1 report dictionaries.
        report_id: Stable identifier for the generated report.
        coverage_target: Desired empirical coverage for calibrated probabilistic forecasts.
        coverage_tolerance: Symmetric tolerance around ``coverage_target`` for recommendation.
        generated_at_utc: Optional deterministic timestamp.

    Returns:
        JSON-compatible calibration report.
    """
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must be between 0 and 1")
    if coverage_tolerance < 0.0:
        raise ValueError("coverage_tolerance must be non-negative")
    if not metric_reports:
        return _empty_report(report_id, coverage_target, coverage_tolerance, generated_at_utc)

    groups: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in metric_reports:
        _require_forecast_metrics_report(report)
        provenance = report["provenance"]
        scenario_family = _scenario_family(report)
        for row in report["aggregate_rows"]:
            groups[
                (
                    scenario_family,
                    float(_required_value(row, "horizon_s")),
                    str(_required_value(provenance, "observation_tier")),
                    str(_required_value(provenance, "predictor_family")),
                )
            ].append(row)

    reliability_rows = [
        _build_reliability_row(
            group_key=group_key,
            aggregate_rows=aggregate_rows,
            coverage_target=coverage_target,
            coverage_tolerance=coverage_tolerance,
        )
        for group_key, aggregate_rows in sorted(groups.items(), key=lambda item: item[0])
    ]
    limitation_rows = _limitation_rows(reliability_rows)
    recommendation = _recommendation(reliability_rows, limitation_rows)
    return {
        "schema_version": FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "source_schema_version": FORECAST_METRICS_SCHEMA_VERSION,
        "calibration_parameters": {
            "coverage_target": float(coverage_target),
            "coverage_tolerance": float(coverage_tolerance),
        },
        "reliability_rows": reliability_rows,
        "limitation_rows": limitation_rows,
        "recommendation": recommendation,
        "claim_boundary": (
            "Calibration summaries are analysis-only evidence. They do not prove planner safety "
            "or navigation benefit, and unavailable uncertainty denominators remain limitations."
        ),
    }


def format_forecast_calibration_markdown(report: dict[str, Any]) -> str:
    """Format a compact Markdown calibration/reliability report.

    Args:
        report: Payload returned by :func:`build_forecast_calibration_report`.

    Returns:
        Markdown summary with reliability rows and caveats.
    """
    recommendation = report["recommendation"]
    lines = [
        "# Forecast Calibration Report",
        "",
        f"- Report id: {report['report_id']}",
        f"- Decision: {recommendation['decision']}",
        f"- Claim status: {recommendation['claim_status']}",
        f"- Reliability rows: {len(report['reliability_rows'])}",
        f"- Limitation rows: {len(report['limitation_rows'])}",
        "",
        "| scenario_family | horizon_s | observation_tier | predictor_family | coverage | status | sharpness | recommendation |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in report["reliability_rows"]:
        coverage = row["empirical_coverage"]
        sharpness = row["sharpness_proxy"]
        lines.append(
            "| {scenario_family} | {horizon_s:g} | {observation_tier} | {predictor_family} | "
            "{coverage} | {calibration_status} | {sharpness} | {recommendation} |".format(
                scenario_family=row["scenario_family"],
                horizon_s=float(row["horizon_s"]),
                observation_tier=row["observation_tier"],
                predictor_family=row["predictor_family"],
                coverage="NA" if coverage is None else f"{float(coverage):.6g}",
                calibration_status=row["calibration_status"],
                sharpness="NA" if sharpness is None else f"{float(sharpness):.6g}",
                recommendation=row["recommendation"],
            )
        )
    if report["limitation_rows"]:
        lines.extend(["", "## Limitations", ""])
        for row in report["limitation_rows"]:
            lines.append(
                "- {scenario_family} / {horizon_s:g}s / {observation_tier} / "
                "{predictor_family}: {reason}".format(**row)
            )
    lines.extend(["", report["claim_boundary"]])
    return "\n".join(lines) + "\n"


def write_forecast_calibration_report(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write calibration JSON and optional Markdown artifacts.

    Returns:
        Mapping of artifact kind to written path.
    """
    _require_calibration_report(report)
    paths = {"json": Path(json_path)}
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        paths["markdown"] = Path(markdown_path)
        paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
        paths["markdown"].write_text(format_forecast_calibration_markdown(report), encoding="utf-8")
    return paths


def _build_reliability_row(
    *,
    group_key: tuple[str, float, str, str],
    aggregate_rows: list[dict[str, Any]],
    coverage_target: float,
    coverage_tolerance: float,
) -> dict[str, Any]:
    scenario_family, horizon_s, observation_tier, predictor_family = group_key
    metrics = _metric_rows_by_name(aggregate_rows)
    coverage_rows = metrics.get("coverage", [])
    likelihood_rows = metrics.get("likelihood", [])
    expected_ade_rows = metrics.get("expected_ade", [])
    minade_rows = metrics.get("minade@k", [])
    empirical_coverage = _available_weighted_mean(coverage_rows)
    likelihood_value = _available_weighted_mean(likelihood_rows)
    sharpness_proxy = _available_weighted_mean(expected_ade_rows)
    sharpness_rows = expected_ade_rows
    if sharpness_proxy is None:
        sharpness_proxy = _available_weighted_mean(minade_rows)
        sharpness_rows = minade_rows
    coverage_gap = None if empirical_coverage is None else empirical_coverage - coverage_target
    calibration_status = _calibration_status(
        coverage_gap=coverage_gap,
        tolerance=coverage_tolerance,
    )
    return {
        "scenario_family": scenario_family,
        "horizon_s": float(horizon_s),
        "observation_tier": observation_tier,
        "predictor_family": predictor_family,
        "empirical_coverage": empirical_coverage,
        "coverage_target": float(coverage_target),
        "coverage_gap": coverage_gap,
        "likelihood": likelihood_value,
        "sharpness_proxy": sharpness_proxy,
        "calibration_status": calibration_status,
        "denominator": _available_denominator(coverage_rows),
        "recommendation": _row_recommendation(calibration_status),
        "unavailable_metrics": _unavailable_metrics(
            {
                "coverage": coverage_rows,
                "likelihood": likelihood_rows,
                "sharpness_proxy": sharpness_rows,
            }
        ),
    }


def _empty_report(
    report_id: str,
    coverage_target: float,
    coverage_tolerance: float,
    generated_at_utc: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "source_schema_version": FORECAST_METRICS_SCHEMA_VERSION,
        "calibration_parameters": {
            "coverage_target": float(coverage_target),
            "coverage_tolerance": float(coverage_tolerance),
        },
        "reliability_rows": [],
        "limitation_rows": [
            {
                "scenario_family": "all",
                "horizon_s": 0.0,
                "observation_tier": "all",
                "predictor_family": "all",
                "reason": "no ForecastMetrics.v1 reports supplied",
            }
        ],
        "recommendation": {
            "decision": "wait",
            "claim_status": "blocked",
            "reason": "no forecast metric reports were supplied",
        },
        "claim_boundary": "No calibration claim is supported without forecast metric reports.",
    }


def _require_forecast_metrics_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("metric report must be a mapping")
    if report.get("schema_version") != FORECAST_METRICS_SCHEMA_VERSION:
        raise ValueError("metric_reports must use ForecastMetrics.v1")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("metric report provenance is required")
    for key in ("predictor_family", "observation_tier"):
        _required_value(provenance, key)
    aggregate_rows = report.get("aggregate_rows")
    if not isinstance(aggregate_rows, list):
        raise ValueError("metric report aggregate_rows must be a list")
    for index, row in enumerate(aggregate_rows):
        if not isinstance(row, dict):
            raise ValueError(f"metric report aggregate_rows[{index}] must be a mapping")
        for key in ("metric", "horizon_s", "status", "denominator"):
            _required_value(row, key)


def _require_calibration_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a mapping")
    if report.get("schema_version") != FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION:
        raise ValueError("report must use ForecastCalibrationReport.v1")


def _scenario_family(report: dict[str, Any]) -> str:
    provenance = report["provenance"]
    value = provenance.get("scenario_family")
    return "unknown" if value is None else str(value)


def _required_value(payload: dict[str, Any], key: str) -> Any:
    if key not in payload or payload[key] is None:
        raise ValueError(f"required metric report field is missing: {key}")
    return payload[key]


def _metric_name(row: dict[str, Any]) -> str:
    metric = str(_required_value(row, "metric"))
    return metric[5:] if metric.startswith("mean_") else metric


def _metric_rows_by_name(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_metric_name(row)].append(row)
    return grouped


def _available_weighted_mean(rows: list[dict[str, Any]]) -> float | None:
    weighted_total = 0.0
    denominator_total = 0
    for row in rows:
        if row.get("status") != "ok" or row.get("value") is None:
            continue
        denominator = int(row.get("denominator", 0))
        value = float(row["value"])
        if denominator > 0:
            weighted_total += value * denominator
            denominator_total += denominator
    if denominator_total > 0:
        return weighted_total / denominator_total
    return None


def _available_denominator(rows: list[dict[str, Any]]) -> int:
    return sum(
        int(row.get("denominator", 0))
        for row in rows
        if row.get("status") == "ok" and row.get("value") is not None
    )


def _calibration_status(*, coverage_gap: float | None, tolerance: float) -> str:
    if coverage_gap is None:
        return "unavailable"
    if coverage_gap < -tolerance:
        return "over_confident_under_coverage"
    if coverage_gap > tolerance:
        return "under_confident_over_coverage"
    return "calibrated_within_tolerance"


def _row_recommendation(calibration_status: str) -> str:
    if calibration_status == "calibrated_within_tolerance":
        return "continue"
    if calibration_status == "unavailable":
        return "wait"
    return "revise"


def _unavailable_metrics(rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [
        metric
        for metric, metric_rows in rows.items()
        if _available_weighted_mean(metric_rows) is None
    ]


def _limitation_rows(reliability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limitations = []
    for row in reliability_rows:
        if row["unavailable_metrics"]:
            limitations.append(
                {
                    "scenario_family": row["scenario_family"],
                    "horizon_s": row["horizon_s"],
                    "observation_tier": row["observation_tier"],
                    "predictor_family": row["predictor_family"],
                    "reason": "unavailable uncertainty metrics: "
                    + ", ".join(row["unavailable_metrics"]),
                }
            )
    return limitations


def _recommendation(
    reliability_rows: list[dict[str, Any]],
    limitation_rows: list[dict[str, Any]],
) -> dict[str, str]:
    if not reliability_rows:
        return {
            "decision": "wait",
            "claim_status": "blocked",
            "reason": "no reliability rows were produced",
        }
    if limitation_rows:
        return {
            "decision": "wait",
            "claim_status": "diagnostic-only",
            "reason": "one or more rows lack uncertainty denominators",
        }
    if any(row["recommendation"] == "revise" for row in reliability_rows):
        return {
            "decision": "revise",
            "claim_status": "diagnostic-only",
            "reason": "one or more rows are outside coverage tolerance",
        }
    return {
        "decision": "continue",
        "claim_status": "analysis-only",
        "reason": "all rows have available uncertainty metrics within coverage tolerance",
    }


__all__ = [
    "FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION",
    "build_forecast_calibration_report",
    "format_forecast_calibration_markdown",
    "write_forecast_calibration_report",
]
