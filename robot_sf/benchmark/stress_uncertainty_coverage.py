"""Stress/uncertainty coverage report builder and validator."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "stress_uncertainty_coverage.v1"
LEGACY_AGGREGATE_SCHEMA_VERSION = "aggregate.schema.v1"

_SCHEMA_MODES = {"required", "advisory", "diagnostic"}
_AGGREGATE_MODES = {"mean", "median", "descriptive_only"}
_AVAILABILITY_STATUSES = {"available", "partial-failure", "failed", "not_available"}
_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "schema_mode",
    "report_id",
    "generated_at_utc",
    "campaign_config_hash",
    "scenario_matrix_hash",
    "coverage_axes",
    "metric_groups",
    "aggregate_mode",
    "availability_status",
}
_REQUIRED_FAILURE_MODES = ("collision", "near_miss", "timeout_without_progress")
_SCENARIO_PARAMETER_KEYS = (
    "archetype",
    "density_label",
    "ped_density_bucket",
    "flow_type",
    "map_name",
    "kinematics_mode",
    "horizon_steps",
    "observation_level",
    "optional_stress_marker",
    "seed_count_bucket",
)


class StressUncertaintyCoverageError(RobotSfError, ValueError):
    """Raised when a stress/uncertainty coverage report fails closed."""


def load_stress_uncertainty_coverage_payload(path: str | Path) -> dict[str, Any]:
    """Load a v1 or legacy aggregate report payload.

    Legacy ``aggregate.schema.v1`` reports remain parseable but explicitly return a
    coverage-unavailable diagnostic wrapper so callers cannot mistake missing annotations for
    evidence.

    Returns:
        Parsed v1 payload or a diagnostic wrapper for legacy aggregate summaries.
    """
    payload = _load_json_object(Path(path))
    schema_version = payload.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        return validate_stress_uncertainty_coverage_report(payload)
    if (
        schema_version is None
        and payload.get("version") == "v1"
        and isinstance(payload.get("groups"), dict)
    ):
        return {
            "schema_version": LEGACY_AGGREGATE_SCHEMA_VERSION,
            "schema_mode": "diagnostic",
            "availability_status": "not_available",
            "missing_fields": ["stress_uncertainty_coverage"],
            "legacy_payload": payload,
        }
    raise StressUncertaintyCoverageError(
        f"Unsupported stress/uncertainty coverage schema_version: {schema_version!r}"
    )


def build_stress_uncertainty_coverage_report(  # noqa: PLR0913
    records: list[dict[str, Any]],
    *,
    report_id: str,
    campaign_config_hash: str,
    scenario_matrix_hash: str,
    schema_mode: str = "required",
    aggregate_mode: str = "mean",
    availability_status: str = "available",
    bootstrap_samples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a conservative v1 stress/uncertainty coverage report from episode records.

    Returns:
        A validated ``stress_uncertainty_coverage.v1`` payload.
    """
    if not records:
        raise StressUncertaintyCoverageError("Cannot build coverage report from zero records")
    missing_fields: list[str] = []
    metric_groups = _build_metric_groups(records, missing_fields)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_mode": schema_mode,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "campaign_config_hash": campaign_config_hash,
        "scenario_matrix_hash": scenario_matrix_hash,
        "coverage_axes": _build_coverage_axes(records),
        "metric_groups": metric_groups,
        "aggregate_mode": aggregate_mode,
        "availability_status": availability_status,
        "missing_fields": missing_fields,
    }
    if bootstrap_samples > 0:
        payload["bootstrap_ci"] = {
            "samples": int(bootstrap_samples),
            "confidence": float(bootstrap_confidence),
            "seed": bootstrap_seed,
            "lower": None,
            "upper": None,
        }
    elif schema_mode != "required":
        payload["missing_fields"].append("bootstrap_ci")
    return validate_stress_uncertainty_coverage_report(payload)


def build_stress_uncertainty_coverage_report_from_jsonl(
    paths: list[str | Path] | str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a v1 stress/uncertainty coverage report from episode JSONL paths.

    Returns:
        Validated report payload.
    """
    return build_stress_uncertainty_coverage_report(read_jsonl(paths), **kwargs)


def validate_stress_uncertainty_coverage_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a v1 report and return a normalized payload.

    Returns:
        Normalized report payload.

    Raises:
        StressUncertaintyCoverageError: If required-mode validation fails.
    """
    if not isinstance(payload, dict):
        raise StressUncertaintyCoverageError("Stress/uncertainty coverage report must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise StressUncertaintyCoverageError(
            f"Unsupported stress/uncertainty coverage schema_version: {payload.get('schema_version')!r}"
        )
    normalized = dict(payload)
    normalized.setdefault("missing_fields", [])
    missing = sorted(field for field in _REQUIRED_TOP_LEVEL if field not in normalized)
    missing.extend(_missing_required_nested_fields(normalized))
    missing = sorted(set(missing))
    if normalized.get("schema_mode") not in _SCHEMA_MODES:
        missing.append("schema_mode")
    if normalized.get("aggregate_mode") not in _AGGREGATE_MODES:
        missing.append("aggregate_mode")
    if normalized.get("availability_status") not in _AVAILABILITY_STATUSES:
        missing.append("availability_status")
    existing_missing = [str(field) for field in normalized.get("missing_fields") or []]
    all_missing = sorted(set(existing_missing + missing))
    normalized["missing_fields"] = all_missing
    if all_missing and normalized.get("schema_mode") == "required":
        normalized["availability_status"] = "failed"
        raise StressUncertaintyCoverageError(
            "Missing required stress/uncertainty coverage fields: " + ", ".join(all_missing)
        )
    if all_missing and normalized.get("schema_mode") == "advisory":
        normalized["availability_status"] = "partial-failure"
    return normalized


def write_stress_uncertainty_coverage_report(
    payload: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Validate and write a stress/uncertainty coverage report.

    Returns:
        Output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = (
        validate_stress_uncertainty_coverage_report(payload)
        if payload.get("schema_version") == SCHEMA_VERSION
        else payload
    )
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Returns:
        Parsed JSON object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StressUncertaintyCoverageError(f"Expected JSON object at {path}")
    return payload


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    """Read a nested value from a mapping.

    Returns:
        Nested value or None when unavailable.
    """
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _number_or_none(value: Any) -> float | None:
    """Normalize finite numeric values.

    Returns:
        Finite float value, or None when unavailable.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_values(records: list[dict[str, Any]], *metric_names: str) -> list[float]:
    """Collect finite metric values from episode records.

    Returns:
        Finite metric values in record order.
    """
    values: list[float] = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name in metric_names:
            number = _number_or_none(metrics.get(name))
            if number is not None:
                values.append(number)
                break
    return values


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean for finite values."""
    return float(sum(values) / len(values)) if values else None


def _success_values_relative_to_total(records: list[dict[str, Any]]) -> list[float]:
    """Build one success value per episode, treating missing values as failures.

    Returns:
        Per-record success values with missing or non-finite success set to ``0.0``.
    """
    values: list[float] = []
    for record in records:
        success_val = _number_or_none(_nested_value(record, "metrics", "success"))
        values.append(success_val if success_val is not None else 0.0)
    return values


def _count_collisions(records: list[dict[str, Any]]) -> int:
    """Count collision episodes from metrics first, then outcome flags.

    Returns:
        Number of collision episodes.
    """
    count = 0
    for record in records:
        metrics_collision = _number_or_none(_nested_value(record, "metrics", "collisions"))
        if metrics_collision is not None:
            count += int(metrics_collision > 0.0)
            continue
        outcome = record.get("outcome")
        if isinstance(outcome, dict) and bool(
            outcome.get("collision_event", outcome.get("collision"))
        ):
            count += 1
    return count


def _count_near_misses(records: list[dict[str, Any]]) -> int:
    """Count near-miss episodes from known metric fields.

    Returns:
        Number of near-miss episodes.
    """
    values = _metric_values(records, "near_misses", "near_miss", "near_miss_count")
    return int(sum(1 for value in values if value > 0.0))


def _count_timeouts(records: list[dict[str, Any]]) -> int:
    """Count timeout-without-progress episodes conservatively.

    Returns:
        Number of timeout-without-progress episodes.
    """
    count = 0
    for record in records:
        reason = str(record.get("termination_reason", "")).strip().lower()
        outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
        timed_out = bool(outcome.get("timeout")) or reason in {"timeout", "max_steps", "time_limit"}
        route_complete = bool(outcome.get("route_complete") or outcome.get("success"))
        if timed_out and not route_complete:
            count += 1
    return count


def _build_metric_groups(
    records: list[dict[str, Any]],
    missing_fields: list[str],
) -> dict[str, Any]:
    """Build conservative safety, comfort, and efficiency metric groups.

    Returns:
        Metric group payload.
    """
    total = len(records)
    collisions = _count_collisions(records)
    near_misses = _count_near_misses(records)
    min_distance = _mean(_metric_values(records, "min_distance", "min_distance_m"))
    comfort_exposure = _mean(_metric_values(records, "comfort_exposure"))
    success_values = _success_values_relative_to_total(records)
    time_to_goal_norm = _mean(_metric_values(records, "time_to_goal_norm"))

    if min_distance is None:
        missing_fields.append("metric_groups.safety.min_distance")
    if comfort_exposure is None:
        missing_fields.append("metric_groups.comfort.comfort_exposure")
    if time_to_goal_norm is None:
        missing_fields.append("metric_groups.efficiency.time_to_goal_norm")

    return {
        "safety": {
            "collisions": collisions,
            "collision_rate": collisions / total if total else 0.0,
            "near_misses": near_misses,
            "min_distance": min_distance,
        },
        "comfort": {
            "comfort_exposure": comfort_exposure,
        },
        "efficiency": {
            "success": sum(success_values) / total if total else 0.0,
            "time_to_goal_norm": time_to_goal_norm,
        },
    }


def _build_coverage_axes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build scenario-parameter and failure-mode coverage axes.

    Returns:
        Coverage axes payload.
    """
    total = len(records)
    return {
        "scenario_parameters": _scenario_parameter_coverage(records),
        "failure_modes": {
            "collision": _failure_mode_payload(
                records, observed=_count_collisions(records), total=total
            ),
            "near_miss": _failure_mode_payload(
                records, observed=_count_near_misses(records), total=total
            ),
            "timeout_without_progress": _failure_mode_payload(
                records, observed=_count_timeouts(records), total=total
            ),
        },
    }


def _scenario_parameter_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return observed scenario-parameter coverage for canonical keys."""
    observed: dict[str, set[str]] = defaultdict(set)
    for record in records:
        params = record.get("scenario_params")
        if not isinstance(params, dict):
            continue
        for key in _SCENARIO_PARAMETER_KEYS:
            value = params.get(key)
            if value is not None:
                observed[key].add(str(value))
    coverage: dict[str, Any] = {}
    for key in sorted(observed):
        values = sorted(observed[key])
        coverage[key] = {
            "observed_values": values,
            "required_values": values,
            "coverage_status": "full" if values else "missing",
        }
    return coverage


def _failure_mode_payload(
    records: list[dict[str, Any]],
    *,
    observed: int,
    total: int,
) -> dict[str, Any]:
    """Build one failure-mode coverage payload.

    Returns:
        Failure-mode coverage payload.
    """
    scenario_ids = sorted(
        {
            str(record.get("scenario_id"))
            for record in records
            if record.get("scenario_id") is not None
        }
    )
    return {
        "observed_episodes": int(observed),
        "observed_fraction": float(observed / total) if total else 0.0,
        "scenario_ids": scenario_ids,
        "classification_source": "episode_metrics",
    }


def _missing_required_nested_fields(payload: dict[str, Any]) -> list[str]:
    """Return missing nested required-field paths for v1 reports."""
    missing: list[str] = []
    coverage_axes = payload.get("coverage_axes")
    if not isinstance(coverage_axes, dict):
        return ["coverage_axes"]
    failure_modes = coverage_axes.get("failure_modes")
    if not isinstance(failure_modes, dict):
        missing.append("coverage_axes.failure_modes")
    else:
        for mode in _REQUIRED_FAILURE_MODES:
            entry = failure_modes.get(mode)
            if not isinstance(entry, dict):
                missing.append(f"coverage_axes.failure_modes.{mode}")
                continue
            for field in ("observed_episodes", "observed_fraction", "classification_source"):
                if field not in entry:
                    missing.append(f"coverage_axes.failure_modes.{mode}.{field}")
    metric_groups = payload.get("metric_groups")
    if not isinstance(metric_groups, dict):
        missing.append("metric_groups")
    elif not any(group in metric_groups for group in ("safety", "comfort", "efficiency")):
        missing.append("metric_groups.safety|comfort|efficiency")
    return missing


__all__ = [
    "LEGACY_AGGREGATE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "StressUncertaintyCoverageError",
    "build_stress_uncertainty_coverage_report",
    "build_stress_uncertainty_coverage_report_from_jsonl",
    "load_stress_uncertainty_coverage_payload",
    "validate_stress_uncertainty_coverage_report",
    "write_stress_uncertainty_coverage_report",
]
