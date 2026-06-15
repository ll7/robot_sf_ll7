"""Transferability stress matrix reports for forecast metric artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.forecast_metrics import FORECAST_METRICS_SCHEMA_VERSION

FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION = "ForecastTransferabilityStressMatrix.v1"

DEFAULT_TRANSFER_DIMENSIONS = (
    "observation_tier",
    "observation_noise",
    "latency",
    "dropout",
    "occlusion",
    "map_family",
    "density",
    "pedestrian_model_family",
    "actor_type",
)

_METRIC_PROVENANCE_KEYS = {
    "observation_tier": ("observation_tier",),
    "observation_noise": ("observation_noise", "noise", "noise_model"),
    "latency": ("latency", "latency_s", "latency_steps"),
    "dropout": ("dropout", "dropout_rate", "missed_detection_probability"),
    "occlusion": ("occlusion", "occlusion_level", "occlusion_status"),
    "map_family": ("map_family", "scenario_family"),
    "density": ("density", "density_label", "ped_density_bucket"),
    "pedestrian_model_family": ("pedestrian_model_family", "ped_model_family"),
}


def build_forecast_transferability_stress_matrix(
    metric_reports: list[dict[str, Any]],
    *,
    report_id: str,
    required_dimensions: tuple[str, ...] = DEFAULT_TRANSFER_DIMENSIONS,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a transferability matrix from forecast metric reports.

    Args:
        metric_reports: ForecastMetrics.v1 report dictionaries.
        report_id: Stable identifier for the generated report.
        required_dimensions: Transfer dimensions that must be surfaced as available or
            explicitly unavailable.
        generated_at_utc: Optional deterministic timestamp for tests and reproducible artifacts.

    Returns:
        JSON-compatible transferability stress matrix payload.
    """
    if not metric_reports:
        return _empty_report(report_id, required_dimensions, generated_at_utc)
    rows: list[dict[str, Any]] = []
    limitation_rows: list[dict[str, Any]] = []
    for report in metric_reports:
        _require_forecast_metrics_report(report)
        provenance = report["provenance"]
        transfer_dimensions = _transfer_dimensions(report)
        missing_dimensions = [
            dimension
            for dimension in required_dimensions
            if dimension != "actor_type" and transfer_dimensions.get(dimension) is None
        ]
        report_key = {
            "predictor_id": str(_required_value(provenance, "predictor_id")),
            "predictor_family": str(_required_value(provenance, "predictor_family")),
            "scenario_id": str(_required_value(provenance, "scenario_id")),
            "observation_tier": str(_required_value(provenance, "observation_tier")),
        }
        for dimension in missing_dimensions:
            limitation_rows.append(
                {
                    **report_key,
                    "dimension": dimension,
                    "availability_status": "not_available",
                    "reason": "dimension metadata unavailable in ForecastMetrics.v1 report",
                }
            )
        for aggregate_row in report["aggregate_rows"]:
            actor_type_value = aggregate_row.get("actor_class")
            actor_type = "unknown" if actor_type_value is None else str(actor_type_value)
            row_dimensions = dict(transfer_dimensions)
            row_dimensions["actor_type"] = actor_type
            row_missing_dimensions = [
                dimension
                for dimension in required_dimensions
                if row_dimensions.get(dimension) is None
            ]
            rows.append(
                {
                    **report_key,
                    "metric": str(_required_value(aggregate_row, "metric")),
                    "horizon_s": float(_required_value(aggregate_row, "horizon_s")),
                    "actor_type": actor_type,
                    "mean_value": aggregate_row.get("value"),
                    "metric_status": str(_required_value(aggregate_row, "status")),
                    "denominator": int(_required_value(aggregate_row, "denominator")),
                    "transfer_dimensions": {
                        dimension: row_dimensions.get(dimension)
                        for dimension in required_dimensions
                    },
                    "unavailable_dimensions": row_missing_dimensions,
                    "evidence_status": _row_evidence_status(
                        observation_tier=report_key["observation_tier"],
                        unavailable_dimensions=row_missing_dimensions,
                        metric_status=str(_required_value(aggregate_row, "status")),
                    ),
                    "claim_boundary": _row_claim_boundary(
                        observation_tier=report_key["observation_tier"],
                        unavailable_dimensions=row_missing_dimensions,
                    ),
                }
            )
    dimension_coverage = _dimension_coverage(rows, limitation_rows, required_dimensions)
    recommendation = _recommendation(rows, limitation_rows)
    return {
        "schema_version": FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "source_schema_version": FORECAST_METRICS_SCHEMA_VERSION,
        "required_dimensions": list(required_dimensions),
        "matrix_rows": rows,
        "limitation_rows": limitation_rows,
        "dimension_coverage": dimension_coverage,
        "recommendation": recommendation,
        "claim_boundary": (
            "Forecast transferability stress rows are diagnostic unless every required "
            "dimension is available, denominators are non-empty, and observation tiers are "
            "deployable or explicitly separated from oracle-only rows."
        ),
    }


def format_forecast_transferability_stress_markdown(report: dict[str, Any]) -> str:
    """Format a compact Markdown summary of a transferability matrix.

    Args:
        report: Payload returned by :func:`build_forecast_transferability_stress_matrix`.

    Returns:
        Markdown report with dimension coverage, recommendation, and limitation rows.
    """
    recommendation = report["recommendation"]
    lines = [
        "# Forecast Transferability Stress Matrix",
        "",
        f"- Report id: {report['report_id']}",
        f"- Decision: {recommendation['decision']}",
        f"- Claim status: {recommendation['claim_status']}",
        f"- Matrix rows: {len(report['matrix_rows'])}",
        f"- Limitation rows: {len(report['limitation_rows'])}",
        "",
        "| dimension | coverage_status | observed_values | unavailable_reports |",
        "| --- | --- | --- | ---: |",
    ]
    for dimension, payload in report["dimension_coverage"].items():
        values = ", ".join(payload["observed_values"]) or "NA"
        lines.append(
            "| {dimension} | {status} | {values} | {count} |".format(
                dimension=dimension,
                status=payload["coverage_status"],
                values=values,
                count=payload["unavailable_report_count"],
            )
        )
    lines.extend(
        [
            "",
            "| metric | horizon_s | observation_tier | actor_type | denominator | status | value | evidence |",
            "| --- | ---: | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in report["matrix_rows"]:
        value = row["mean_value"]
        rendered_value = "NA" if value is None else f"{float(value):.6g}"
        lines.append(
            "| {metric} | {horizon:g} | {tier} | {actor_type} | {denominator} | {status} | "
            "{value} | {evidence} |".format(
                metric=row["metric"],
                horizon=float(row["horizon_s"]),
                tier=row["observation_tier"],
                actor_type=row["actor_type"],
                denominator=int(row["denominator"]),
                status=row["metric_status"],
                value=rendered_value,
                evidence=row["evidence_status"],
            )
        )
    if report["limitation_rows"]:
        lines.extend(["", "## Limitations", ""])
        for row in report["limitation_rows"]:
            lines.append(
                "- {dimension}: {reason} ({predictor_id}, {scenario_id}, {observation_tier})".format(
                    **row
                )
            )
    lines.extend(["", report["claim_boundary"]])
    return "\n".join(lines) + "\n"


def write_forecast_transferability_stress_matrix(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write transferability matrix JSON and optional Markdown artifacts.

    Returns:
        Mapping of artifact kind to written path.
    """
    _require_transferability_report(report)
    paths = {"json": Path(json_path)}
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path is not None:
        paths["markdown"] = Path(markdown_path)
        paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
        paths["markdown"].write_text(
            format_forecast_transferability_stress_markdown(report),
            encoding="utf-8",
        )
    return paths


def _empty_report(
    report_id: str,
    required_dimensions: tuple[str, ...],
    generated_at_utc: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "source_schema_version": FORECAST_METRICS_SCHEMA_VERSION,
        "required_dimensions": list(required_dimensions),
        "matrix_rows": [],
        "limitation_rows": [
            {
                "dimension": "all",
                "availability_status": "not_available",
                "reason": "no ForecastMetrics.v1 reports supplied",
            }
        ],
        "dimension_coverage": {
            dimension: {
                "observed_values": [],
                "unavailable_report_count": 1,
                "coverage_status": "unavailable",
            }
            for dimension in required_dimensions
        },
        "recommendation": {
            "decision": "stop",
            "claim_status": "blocked",
            "reason": "no forecast metric reports were supplied",
        },
        "claim_boundary": "No transferability claim is supported without forecast metric reports.",
    }


def _require_forecast_metrics_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("metric report must be a mapping")
    if report.get("schema_version") != FORECAST_METRICS_SCHEMA_VERSION:
        raise ValueError("metric_reports must use ForecastMetrics.v1")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("metric report provenance is required")
    for key in ("predictor_id", "predictor_family", "scenario_id", "observation_tier"):
        _required_value(provenance, key)
    aggregate_rows = report.get("aggregate_rows")
    if not isinstance(aggregate_rows, list):
        raise ValueError("metric report aggregate_rows must be a list")
    for index, aggregate_row in enumerate(aggregate_rows):
        if not isinstance(aggregate_row, dict):
            raise ValueError(f"metric report aggregate_rows[{index}] must be a mapping")
        for key in ("metric", "horizon_s", "status", "denominator"):
            _required_value(aggregate_row, key)


def _require_transferability_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a mapping")
    if report.get("schema_version") != FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION:
        raise ValueError("report must use ForecastTransferabilityStressMatrix.v1")


def _transfer_dimensions(report: dict[str, Any]) -> dict[str, str | None]:
    provenance = report.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    metadata = report.get("transfer_dimensions")
    if not isinstance(metadata, dict):
        metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    dimensions: dict[str, str | None] = {}
    for dimension, keys in _METRIC_PROVENANCE_KEYS.items():
        value = _first_present(metadata, keys)
        if value is None:
            value = _first_present(provenance, keys)
        dimensions[dimension] = None if value is None else str(value)
    return dimensions


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _required_value(payload: dict[str, Any], key: str) -> Any:
    if key not in payload or payload[key] is None:
        raise ValueError(f"required metric report field is missing: {key}")
    return payload[key]


def _dimension_coverage(
    rows: list[dict[str, Any]],
    limitation_rows: list[dict[str, Any]],
    required_dimensions: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    values_by_dimension: dict[str, set[str]] = defaultdict(set)
    unavailable_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for dimension, value in row["transfer_dimensions"].items():
            if value is not None:
                values_by_dimension[dimension].add(str(value))
    for row in limitation_rows:
        unavailable_counts[str(row["dimension"])] += 1
    coverage = {}
    for dimension in required_dimensions:
        observed_values = sorted(values_by_dimension[dimension])
        unavailable_count = unavailable_counts[dimension]
        coverage[dimension] = {
            "observed_values": observed_values,
            "unavailable_report_count": unavailable_count,
            "coverage_status": "full"
            if observed_values and unavailable_count == 0
            else "partial"
            if observed_values
            else "unavailable",
        }
    return coverage


def _recommendation(
    rows: list[dict[str, Any]],
    limitation_rows: list[dict[str, Any]],
) -> dict[str, str]:
    if not rows:
        return {
            "decision": "stop",
            "claim_status": "blocked",
            "reason": "no matrix rows were produced",
        }
    if limitation_rows:
        return {
            "decision": "revise",
            "claim_status": "diagnostic-only",
            "reason": "one or more transfer dimensions are unavailable",
        }
    if any(_is_oracle_tier(row["observation_tier"]) for row in rows):
        return {
            "decision": "revise",
            "claim_status": "diagnostic-only",
            "reason": "oracle-state rows must not be promoted as deployable transfer evidence",
        }
    if any(row["denominator"] == 0 or row["metric_status"] != "ok" for row in rows):
        return {
            "decision": "revise",
            "claim_status": "diagnostic-only",
            "reason": "one or more matrix cells have empty denominators or unavailable metrics",
        }
    return {
        "decision": "continue",
        "claim_status": "benchmark-eligible",
        "reason": "all required dimensions are explicit with deployable non-empty metric rows",
    }


def _row_evidence_status(
    *,
    observation_tier: str,
    unavailable_dimensions: list[str],
    metric_status: str,
) -> str:
    if unavailable_dimensions or metric_status != "ok":
        return "diagnostic-only"
    if _is_oracle_tier(observation_tier):
        return "oracle-only"
    return "benchmark-eligible"


def _row_claim_boundary(
    *,
    observation_tier: str,
    unavailable_dimensions: list[str],
) -> str:
    if unavailable_dimensions:
        return "unavailable transfer dimensions prevent benchmark-strength transfer claims"
    if _is_oracle_tier(observation_tier):
        return "oracle-state row; not deployable transfer evidence"
    return "deployable observation row; still simulation-only transfer evidence"


def _is_oracle_tier(observation_tier: str) -> bool:
    return observation_tier.strip().lower().startswith("oracle")


__all__ = [
    "DEFAULT_TRANSFER_DIMENSIONS",
    "FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION",
    "build_forecast_transferability_stress_matrix",
    "format_forecast_transferability_stress_markdown",
    "write_forecast_transferability_stress_matrix",
]
