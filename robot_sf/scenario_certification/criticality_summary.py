"""Criticality summary v1 writer/validator for #1610 perturbation-family pilots.

This module produces and validates `criticality_summary.v1` payloads that record
explicit row-status counts and exclude non-completed rows from effect means.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

if TYPE_CHECKING:
    from collections.abc import Mapping

CRITICALITY_SUMMARY_SCHEMA_VERSION = "criticality_summary.v1"
_CRITICALITY_SUMMARY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark" / "schemas" / "criticality_summary.v1.json"
)


def _sum(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


@dataclass(frozen=True)
class CriticalitySummaryV1:
    """Diagnostic criticality summary for one perturbation-family pilot.

    This never carries benchmark-strength or paper-facing claims. It records
    explicit row status counts and keeps fallback/degraded/invalid/missing/failed
    rows separate from completed-pair effect means.
    """

    schema_version: str
    manifest: str
    manifest_id: str
    planners: list[str]
    horizon: int
    dt: float
    seed_limit: int
    materialization: dict[str, Any]
    planner_runs: dict[str, dict[str, Any]]
    pair_summary: dict[str, Any]
    pair_rows: list[dict[str, Any]] = field(default_factory=list)
    description: str | None = None
    claim_boundary: str = (
        "diagnostic local pilot only; not benchmark-strength or paper-facing evidence"
    )


def _classify_episode_row_status(row: dict[str, Any] | None) -> str:
    """Classify one episode row as completed, invalid, fallback, degraded, missing, or failed.

    Returns:
        str: One of completed, invalid, fallback, degraded, missing, or failed.
    """
    if row is None:
        return "missing"
    if isinstance(row.get("scenario_exclusion"), dict):
        return "invalid"
    algorithm_metadata = row.get("algorithm_metadata")
    if isinstance(algorithm_metadata, dict):
        metadata_strings = _nested_strings(algorithm_metadata)
        if any("fallback" in item for item in metadata_strings):
            return "fallback"
        if any("degraded" in item for item in metadata_strings):
            return "degraded"
    reason = str(row.get("termination_reason") or "").strip().lower()
    if reason == "error":
        return "failed"
    return "completed"


def _nested_strings(value: Any) -> list[str]:
    """Return lower-cased string leaves from a nested JSON-like value."""
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value.strip().lower())
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_nested_strings(item))
    elif isinstance(value, list | tuple):
        for item in value:
            strings.extend(_nested_strings(item))
    return strings


def _delta(after: int | float | None, before: int | float | None) -> float | None:
    """Return numeric delta when both sides are available."""
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _episode_value(row: dict[str, Any] | None, name: str) -> float | None:
    """Read a numeric metric from an episode row.

    Returns:
        float | None: The metric value or None when the row or metric is absent.
    """
    if row is None:
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _episode_outcome(row: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact outcome values for one episode row."""
    reason = str(row.get("termination_reason") or "") if row is not None else ""
    return {
        "success": 1 if reason.lower() == "success" else 0,
        "collision": 1 if reason.lower() == "collision" else 0,
        "timeout": 1 if reason.lower() in {"max_steps", "terminated", "truncated"} else 0,
        "min_distance": _episode_value(row, "min_distance"),
        "termination_reason": reason or None,
    }


_REQUIRED_ROW_STATUSES = frozenset(
    {"completed", "invalid", "fallback", "degraded", "missing", "failed"}
)


def _zero_row_status_counts() -> dict[str, int]:
    """Return a zeroed row-status-counts dict for all required statuses.

    Returns:
        dict[str, int]: Zeroed counts for completed, invalid, fallback,
        degraded, missing, and failed.
    """
    return dict.fromkeys(sorted(_REQUIRED_ROW_STATUSES), 0)


def _aggregate_subset(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one pair-table subset with row status counts and completed-pair means.

    Returns:
        dict[str, Any]: Aggregated subset with pairs count, row_status_counts,
        and mean_deltas_completed_pairs.
    """
    status_counts: dict[str, int] = defaultdict(int)
    delta_values: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        perturbed_status = str(row.get("perturbed_status") or "unknown")
        status_counts[perturbed_status] += 1
        if row.get("pair_status") != "completed":
            continue
        for field_name in (
            "success_delta",
            "collision_delta",
            "timeout_delta",
            "min_distance_delta",
        ):
            value = row.get(field_name)
            if value is not None:
                delta_values[field_name].append(float(value))
    combined_status_counts = dict(_zero_row_status_counts())
    for status, count in status_counts.items():
        if status in combined_status_counts:
            combined_status_counts[status] = count
    return {
        "pairs": len(pair_rows),
        "row_status_counts": combined_status_counts,
        "mean_deltas_completed_pairs": {
            field_name: _sum(values)
            for field_name, values in sorted(delta_values.items())
            if values
        },
    }


def _grouped_summaries(
    pair_rows: list[dict[str, Any]],
    *,
    group_field: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate pair rows by one categorical group field.

    Returns:
        dict[str, dict[str, Any]]: Grouped aggregate summaries keyed by group value.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        key = str(row.get(group_field) or "unknown")
        grouped[key].append(row)
    return {key: _aggregate_subset(rows) for key, rows in sorted(grouped.items())}


def _scenarios_by_source(
    scenario_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Group materialized scenario ids by source scenario and perturbation family.

    Returns:
        dict[str, dict[str, list[str]]]: Source-scenario → family → scenario-id list mapping.
    """
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for scenario_id, meta in scenario_metadata.items():
        source = meta["source_scenario_id"]
        family = meta["family"]
        if source and family:
            grouped[source][family].append(scenario_id)
    return grouped


def build_criticality_summary_from_pilot(  # noqa: PLR0913
    records_by_planner: dict[str, list[dict[str, Any]]],
    scenario_metadata: dict[str, dict[str, Any]],
    *,
    manifest: str,
    manifest_id: str,
    planners: list[str],
    horizon: int,
    dt: float,
    seed_limit: int,
    materialization: dict[str, Any],
    planner_runs: dict[str, dict[str, Any]],
    description: str | None = None,
    claim_boundary: str = (
        "diagnostic local pilot only; not benchmark-strength or paper-facing evidence"
    ),
) -> CriticalitySummaryV1:
    """Build a criticality summary from raw pilot episode records.

    Returns:
        CriticalitySummaryV1: Summary with explicit row status counts and
        completed-pair effect means split from non-completed rows.
    """
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for planner_name, records in records_by_planner.items():
        for row in records:
            scenario_id = str(row.get("scenario_id") or row.get("scenario") or "")
            seed = int(row.get("seed", 0))
            rows_by_key[(planner_name, scenario_id, seed)] = row

    grouped_scenarios = _scenarios_by_source(scenario_metadata)
    pair_rows: list[dict[str, Any]] = []
    for planner_name in sorted(records_by_planner):
        for source_scenario_id, families in sorted(grouped_scenarios.items()):
            noop_ids = sorted(families.get("noop", []))
            perturbed_ids = sorted(
                variant_id
                for family, variant_ids in families.items()
                if family != "noop"
                for variant_id in variant_ids
            )
            if not noop_ids:
                for variant_id in perturbed_ids:
                    pair_rows.append(
                        {
                            "planner": planner_name,
                            "source_scenario_id": source_scenario_id,
                            "noop_variant_id": None,
                            "perturbed_variant_id": variant_id,
                            "perturbed_family": "unknown",
                            "seed": None,
                            "pair_status": "missing_noop",
                            "noop_status": "missing",
                            "perturbed_status": "missing",
                        }
                    )
                continue
            noop_id = noop_ids[0]
            for variant_id in perturbed_ids:
                seeds = sorted(
                    {
                        seed
                        for (kp, sid, seed) in rows_by_key
                        if kp == planner_name and sid in {noop_id, variant_id}
                    }
                )
                for seed in seeds:
                    noop_row = rows_by_key.get((planner_name, noop_id, seed))
                    perturbed_row = rows_by_key.get((planner_name, variant_id, seed))
                    noop_status = _classify_episode_row_status(noop_row)
                    perturbed_status = _classify_episode_row_status(perturbed_row)
                    pair_status = (
                        "completed"
                        if noop_status == "completed" and perturbed_status == "completed"
                        else "excluded"
                    )
                    noop_outcome = _episode_outcome(noop_row)
                    perturbed_outcome = _episode_outcome(perturbed_row)
                    perturbed_meta = scenario_metadata.get(variant_id, {})
                    pair_rows.append(
                        {
                            "planner": planner_name,
                            "source_scenario_id": source_scenario_id,
                            "noop_variant_id": noop_id,
                            "perturbed_variant_id": variant_id,
                            "perturbed_family": str(perturbed_meta.get("family") or "unknown"),
                            "seed": seed,
                            "pair_status": pair_status,
                            "noop_status": noop_status,
                            "perturbed_status": perturbed_status,
                            "success_delta": _delta(
                                perturbed_outcome["success"], noop_outcome["success"]
                            ),
                            "collision_delta": _delta(
                                perturbed_outcome["collision"], noop_outcome["collision"]
                            ),
                            "timeout_delta": _delta(
                                perturbed_outcome["timeout"], noop_outcome["timeout"]
                            ),
                            "min_distance_delta": _delta(
                                perturbed_outcome["min_distance"], noop_outcome["min_distance"]
                            ),
                            "noop": noop_outcome,
                            "perturbed": perturbed_outcome,
                        }
                    )
    pair_summary = _aggregate_subset(pair_rows)
    pair_summary["by_planner"] = _grouped_summaries(pair_rows, group_field="planner")
    pair_summary["by_source_scenario"] = _grouped_summaries(
        pair_rows, group_field="source_scenario_id"
    )
    pair_summary["by_perturbation_family"] = _grouped_summaries(
        pair_rows, group_field="perturbed_family"
    )
    return CriticalitySummaryV1(
        schema_version=CRITICALITY_SUMMARY_SCHEMA_VERSION,
        manifest=manifest,
        manifest_id=manifest_id,
        planners=planners,
        horizon=horizon,
        dt=dt,
        seed_limit=seed_limit,
        materialization=materialization,
        planner_runs=planner_runs,
        pair_summary=pair_summary,
        pair_rows=pair_rows,
        description=description,
        claim_boundary=claim_boundary,
    )


def build_criticality_summary_from_compact_evidence(
    compact_evidence: Mapping[str, Any],
) -> CriticalitySummaryV1:
    """Wrap an existing compact #1610 evidence payload as a CriticalitySummaryV1.

    At least one existing compact summary must be representable without relying
    on raw output paths. This constructor converts the existing format into the
    v1 criticality summary, normalizing row status counts.

    Returns:
        CriticalitySummaryV1: Enriched summary with explicit row status counts.
    """
    evidence = dict(compact_evidence)
    raw_pair_summary = dict(evidence.get("pair_summary", {}))
    pair_rows = list(evidence.get("pair_rows", []))
    status_counts = _extract_row_statuses(pair_rows)
    pair_summary = build_pair_summary_with_statuses(pair_rows, status_counts)
    for group_field in (
        "by_planner",
        "by_source_scenario",
        "by_perturbation_family",
    ):
        if group_field in raw_pair_summary and isinstance(raw_pair_summary[group_field], dict):
            mapped: dict[str, dict[str, Any]] = {}
            for key, group_payload in raw_pair_summary[group_field].items():
                if isinstance(group_payload, dict):
                    mapped[key] = _upgrade_to_row_status_counts(group_payload)
                else:
                    mapped[key] = dict(group_payload) if group_payload else {}
            pair_summary[group_field] = mapped
    return CriticalitySummaryV1(
        schema_version=CRITICALITY_SUMMARY_SCHEMA_VERSION,
        manifest=str(evidence.get("manifest", "")),
        manifest_id=str(evidence.get("materialization", {}).get("manifest_id", "")),
        planners=list(evidence.get("planners", [])),
        horizon=int(evidence.get("horizon", 0)),
        dt=float(evidence.get("dt", 0.0)),
        seed_limit=int(evidence.get("seed_limit", 0)),
        materialization=dict(evidence.get("materialization", {})),
        planner_runs=dict(evidence.get("planner_runs", {})),
        pair_summary=pair_summary,
        pair_rows=pair_rows,
        description=str(evidence.get("description") or "") or None,
        claim_boundary=str(
            evidence.get("claim_boundary")
            or "diagnostic local pilot only; not benchmark-strength or paper-facing evidence"
        ),
    )


def _extract_row_statuses(pair_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Extract explicit perturbed-row status counts from pair rows.

    Returns:
        dict[str, int]: Status key → count mapping.
    """
    status_counts: dict[str, int] = defaultdict(int)
    for row in pair_rows:
        status = str(row.get("perturbed_status") or "unknown")
        status_counts[status] += 1
    return dict(status_counts)


def _upgrade_to_row_status_counts(
    group_payload: dict[str, Any],
) -> dict[str, Any]:
    """Upgrade a grouped summary payload to explicit row_status_counts format.

    Returns:
        dict[str, Any]: Grouped summary with row_status_counts and mean_deltas_completed_pairs.
    """
    status_counts = group_payload.get("status_counts")
    if isinstance(status_counts, dict) and "completed" in status_counts:
        combined = dict(_zero_row_status_counts())
        for key, count in status_counts.items():
            if key in combined:
                combined[key] = int(count)
        return {
            "pairs": int(group_payload.get("pairs", 0)),
            "row_status_counts": combined,
            "mean_deltas_completed_pairs": dict(
                group_payload.get("mean_deltas_completed_pairs", {})
            ),
        }
    combined = dict(_zero_row_status_counts())
    if isinstance(status_counts, dict):
        for key, count in status_counts.items():
            if key in combined:
                combined[key] = int(count)
    return {
        "pairs": int(group_payload.get("pairs", 0)),
        "row_status_counts": combined,
        "mean_deltas_completed_pairs": dict(group_payload.get("mean_deltas_completed_pairs", {})),
    }


def build_pair_summary_with_statuses(
    pair_rows: list[dict[str, Any]],
    row_status_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a pair summary payload with explicit row status counts.

    Fallback/degraded/invalid/missing/failed rows are tracked separately
    and excluded from completed-pair effect means.

    Returns:
        dict[str, Any]: Summary payload with pairs count, row_status_counts,
        and mean_deltas_completed_pairs.
    """
    if row_status_counts is None:
        row_status_counts = _extract_row_statuses(pair_rows)
    combined = dict(_zero_row_status_counts())
    for status, count in row_status_counts.items():
        if status in combined:
            combined[status] = count
    delta_values: dict[str, list[float]] = defaultdict(list)
    completed_count = 0
    for row in pair_rows:
        if row.get("pair_status") == "completed":
            completed_count += 1
            for field_name in (
                "success_delta",
                "collision_delta",
                "timeout_delta",
                "min_distance_delta",
            ):
                value = row.get(field_name)
                if value is not None:
                    delta_values[field_name].append(float(value))
    return {
        "pairs": len(pair_rows),
        "row_status_counts": combined,
        "mean_deltas_completed_pairs": {
            field_name: _sum(values)
            for field_name, values in sorted(delta_values.items())
            if values
        },
    }


def validate_criticality_summary(summary: dict[str, Any]) -> None:
    """Validate a criticality summary payload against the public v1 schema.

    Raises:
        ValueError: When validation fails, describing the first violation.
    """
    schema = json.loads(_CRITICALITY_SUMMARY_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(summary), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ValueError(f"criticality_summary schema violation at {location}: {error.message}")


def criticality_summary_to_dict(summary: CriticalitySummaryV1) -> dict[str, Any]:
    """Serialize a CriticalitySummaryV1 to JSON-safe primitives.

    Returns:
        dict[str, Any]: JSON-safe payload suitable for schema validation and
        compact evidence tracking.
    """
    payload: dict[str, Any] = {
        "schema_version": summary.schema_version,
        "manifest": summary.manifest,
        "manifest_id": summary.manifest_id,
        "planners": summary.planners,
        "horizon": summary.horizon,
        "dt": summary.dt,
        "seed_limit": summary.seed_limit,
        "materialization": summary.materialization,
        "planner_runs": summary.planner_runs,
        "pair_summary": summary.pair_summary,
        "pair_rows": summary.pair_rows,
        "claim_boundary": summary.claim_boundary,
    }
    if summary.description is not None:
        payload["description"] = summary.description
    return payload
