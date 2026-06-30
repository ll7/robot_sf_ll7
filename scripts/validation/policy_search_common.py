"""Shared helpers for policy-search evaluation and reporting.

The policy-search notes under ``docs/context/policy_search`` require a
stable scenario-family split and a lightweight failure taxonomy that can be
computed from benchmark episode JSONL files without introducing a second
benchmark contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST


def _as_float(value: Any) -> float | None:
    """Coerce a value to float while preserving missing/invalid as None."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the metrics mapping from an episode row."""
    metrics = row.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _has_safety_near_miss(metrics: Mapping[str, Any]) -> bool:
    """Return true for benchmark safety near-miss semantics."""
    near_misses = _as_float(metrics.get("near_misses")) or 0.0
    if near_misses > 0.0:
        return True
    min_clearance = _as_float(metrics.get("min_clearance"))
    return min_clearance is not None and 0.0 <= min_clearance < NEAR_MISS_DIST


def _has_center_distance_near_miss_diagnostic(metrics: Mapping[str, Any]) -> bool:
    """Return true for legacy geometric center-distance near-miss diagnostics."""
    diagnostic = _as_float(metrics.get("center_distance_near_miss_diagnostic")) or 0.0
    if diagnostic > 0.0:
        return True
    min_distance = _as_float(metrics.get("min_distance"))
    return min_distance is not None and COLLISION_DIST <= min_distance < NEAR_MISS_DIST


def normalize_scenario_exclusion(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a validated invalid/impossible-scenario exclusion, if explicitly present.

    The policy-search reporting contract intentionally does not infer exclusions from scenario IDs
    or termination reasons. A record must carry explicit exclusion metadata with a status, reason,
    and non-empty evidence list before reporting treats it separately from policy failure.
    """
    raw = row.get("scenario_exclusion")
    if not isinstance(raw, Mapping):
        return None
    status = str(raw.get("status", "")).strip().lower()
    if status not in {"invalid", "impossible", "excluded"}:
        return None
    reason = str(raw.get("reason", "")).strip()
    evidence_raw = raw.get("evidence")
    if isinstance(evidence_raw, str):
        evidence = [evidence_raw.strip()] if evidence_raw.strip() else []
    elif isinstance(evidence_raw, list):
        evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]
    else:
        evidence = []
    if not reason or not evidence:
        return None
    return {
        "scenario_id": str(row.get("scenario_id", "unknown")),
        "seed": row.get("seed"),
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }


def _count_configured_pedestrians(row: Mapping[str, Any]) -> int:
    """Count configured pedestrians from scenario params when present."""
    scenario_params = row.get("scenario_params")
    if not isinstance(scenario_params, Mapping):
        return 0
    for key in ("humans", "pedestrians", "agents"):
        raw = scenario_params.get(key)
        if isinstance(raw, list):
            return len(raw)
        if isinstance(raw, Mapping):
            positions = raw.get("positions")
            if isinstance(positions, list):
                return len(positions)
            count = _as_float(raw.get("count"))
            if count is not None:
                return max(int(count), 0)
    count = _as_float(scenario_params.get("pedestrian_count"))
    if count is not None:
        return max(int(count), 0)
    return 0


def infer_scenario_family(row: Mapping[str, Any]) -> str:
    """Infer a compact scenario family label from an episode record."""
    for field in ("scenario_family", "archetype", "family"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    scenario_params = row.get("scenario_params")
    if isinstance(scenario_params, Mapping):
        for field in ("scenario_family", "archetype", "family"):
            value = scenario_params.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    scenario_id = (
        str(row.get("scenario_id") or row.get("name") or row.get("id") or "").strip().lower()
    )
    if scenario_id.startswith("classic_"):
        return "classic"
    if scenario_id.startswith("francis2023_") or "francis2023" in scenario_id:
        return "francis2023"
    if scenario_id.startswith("planner_sanity"):
        return "nominal"
    return "unknown"


def classify_failure_mode(  # noqa: C901
    row: Mapping[str, Any],
    *,
    low_speed_threshold: float = 0.15,
) -> str | None:
    """Classify an episode into the dominant policy-search failure mode.

    The taxonomy is intentionally heuristic: it only uses fields already
    present in benchmark episode records so it can be applied to historical
    JSONL outputs and new campaign runs alike.
    """

    if normalize_scenario_exclusion(row) is not None:
        return None

    reason = str(row.get("termination_reason", "")).strip().lower()
    metrics = _metrics(row)
    scenario_id = str(row.get("scenario_id", "")).strip().lower()
    avg_speed = _as_float(metrics.get("avg_speed"))
    jerky_turns = _as_float(metrics.get("angular_speed_sign_changes"))
    progress = _as_float(metrics.get("goal_progress"))

    if reason == "success":
        return None

    if reason == "collision":
        ped_count = _count_configured_pedestrians(row)
        return "static_collision" if ped_count <= 0 else "pedestrian_collision"

    if _has_safety_near_miss(metrics):
        return "near_miss_intrusive"
    if _has_center_distance_near_miss_diagnostic(metrics):
        return "center_distance_near_miss_diagnostic"

    if jerky_turns is not None and jerky_turns >= 6.0:
        return "oscillation"

    if reason in {"max_steps", "terminated", "truncated"}:
        if avg_speed is not None and avg_speed <= low_speed_threshold:
            if "doorway" in scenario_id or "bottleneck" in scenario_id:
                return "bottleneck_yield_failure"
            return "overconservative_stop"
        if progress is not None and progress <= 0.05:
            return "deadlock"
        return "timeout_low_progress"

    if reason == "error":
        return "wrong_waypoint_behavior"

    return "timeout_low_progress"


def summarize_policy_search_records(  # noqa: C901
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate policy-search metrics, family splits, and failure counts."""
    total = len(records)
    termination_counts = Counter(
        str(row.get("termination_reason", "unknown")).strip().lower() or "unknown"
        for row in records
    )
    failure_counts = Counter()
    family_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    near_miss_count = 0
    center_distance_near_miss_diagnostic_count = 0
    min_distance_values: list[float] = []
    avg_speed_values: list[float] = []
    actuation_metric_values: dict[str, list[float]] = {
        "command_clip_fraction": [],
        "yaw_rate_saturation_fraction": [],
        "signed_braking_peak_m_s2": [],
    }
    exclusions: list[dict[str, Any]] = []

    for row in records:
        family_groups[infer_scenario_family(row)].append(row)
        exclusion = normalize_scenario_exclusion(row)
        if exclusion is not None:
            exclusions.append(exclusion)
        else:
            failure_mode = classify_failure_mode(row)
            if failure_mode is not None:
                failure_counts[failure_mode] += 1

        metrics = _metrics(row)
        min_distance = _as_float(metrics.get("min_distance"))
        avg_speed = _as_float(metrics.get("avg_speed"))
        for metric_name, values in actuation_metric_values.items():
            metric_value = _as_float(metrics.get(metric_name))
            if metric_value is not None:
                values.append(metric_value)

        if _has_safety_near_miss(metrics):
            near_miss_count += 1
        if _has_center_distance_near_miss_diagnostic(metrics):
            center_distance_near_miss_diagnostic_count += 1

        if min_distance is not None:
            min_distance_values.append(min_distance)
        if avg_speed is not None:
            avg_speed_values.append(avg_speed)

    def _suite_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Summarize outcome rates for one row group.

        Returns:
            dict[str, Any]: Episode count and success/collision/near-miss rates.
        """
        episodes = len(rows)
        collisions = sum(
            1
            for row in rows
            if str(row.get("termination_reason", "")).strip().lower() == "collision"
        )
        successes = sum(
            1 for row in rows if str(row.get("termination_reason", "")).strip().lower() == "success"
        )
        near_misses_local = 0
        for row in rows:
            metrics = _metrics(row)
            if _has_safety_near_miss(metrics):
                near_misses_local += 1
        denom = float(episodes) if episodes > 0 else 1.0
        return {
            "episodes": episodes,
            "success_rate": successes / denom if episodes > 0 else 0.0,
            "collision_rate": collisions / denom if episodes > 0 else 0.0,
            "near_miss_rate": near_misses_local / denom if episodes > 0 else 0.0,
        }

    family_summary = {
        family: _suite_summary(rows) for family, rows in sorted(family_groups.items())
    }

    denom = float(total) if total > 0 else 1.0
    adjusted_rows = [row for row in records if normalize_scenario_exclusion(row) is None]
    adjusted_summary = _suite_summary(adjusted_rows)
    return {
        "episodes": total,
        "success_rate": termination_counts.get("success", 0) / denom if total > 0 else 0.0,
        "collision_rate": termination_counts.get("collision", 0) / denom if total > 0 else 0.0,
        "near_miss_rate": near_miss_count / denom if total > 0 else 0.0,
        "near_miss_semantics": "surface_clearance_safety_metric",
        "center_distance_near_miss_diagnostic_rate": (
            center_distance_near_miss_diagnostic_count / denom if total > 0 else 0.0
        ),
        "termination_reason_counts": dict(termination_counts),
        "failure_mode_counts": dict(failure_counts),
        "scenario_exclusions": {
            "count": len(exclusions),
            "by_status": dict(Counter(str(item["status"]) for item in exclusions)),
            "by_reason": dict(Counter(str(item["reason"]) for item in exclusions)),
            "records": exclusions,
        },
        "evidence_adjusted": {
            "episodes": adjusted_summary["episodes"],
            "excluded_episodes": len(exclusions),
            "success_rate": adjusted_summary["success_rate"],
            "collision_rate": adjusted_summary["collision_rate"],
            "near_miss_rate": adjusted_summary["near_miss_rate"],
        },
        "scenario_family": family_summary,
        "mean_min_distance": (
            sum(min_distance_values) / len(min_distance_values) if min_distance_values else None
        ),
        "mean_avg_speed": sum(avg_speed_values) / len(avg_speed_values)
        if avg_speed_values
        else None,
        "synthetic_actuation": {
            f"{metric_name}_mean": sum(values) / len(values) if values else None
            for metric_name, values in actuation_metric_values.items()
        },
    }


__all__ = [
    "classify_failure_mode",
    "infer_scenario_family",
    "normalize_scenario_exclusion",
    "summarize_policy_search_records",
]
