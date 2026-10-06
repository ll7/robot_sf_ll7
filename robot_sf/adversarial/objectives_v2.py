"""Additive, fail-closed constraints-first objective used by new falsification runs.

The original objective registry is source-pinned by historical preregistrations.
Keep this corrected objective in a separate module so those byte-level identities
continue to resolve to the code that was actually frozen.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.objectives import (
    _consistent_boolean_alias,
    _success_metric_matches_route_complete,
    _valid_constraints_metric,
    register_objective,
)

if TYPE_CHECKING:
    from robot_sf.adversarial.config import CandidateEvaluation


def _component_from_outcome(outcome: dict[str, Any], *names: str) -> tuple[bool | None, bool]:
    """Return one alias value and whether present outcome evidence is malformed/conflicting."""
    present = any(name in outcome for name in names)
    value = _consistent_boolean_alias(outcome, *names)
    return value, present and value is None


def _collision_component(outcome: dict[str, Any], metrics: dict[str, Any]) -> bool | None:
    outcome_value, outcome_conflict = _component_from_outcome(
        outcome, "collision", "collision_event"
    )
    if outcome_conflict:
        return None
    raw_metric = metrics.get("collisions")
    metric_value: bool | None = None
    if raw_metric is not None:
        if not _valid_constraints_metric("collisions", raw_metric):
            return None
        metric_value = raw_metric > 0

    if outcome_value is not None and metric_value is not None and outcome_value != metric_value:
        return None
    return outcome_value if outcome_value is not None else metric_value


def _intrusion_component(outcome: dict[str, Any], metrics: dict[str, Any]) -> bool | None:
    names = ("severe_intrusion", "severe_intrusion_event")
    outcome_value, outcome_conflict = _component_from_outcome(outcome, *names)
    if outcome_conflict:
        return None
    metric_values = [
        metrics[name] for name in names if name in metrics and metrics[name] is not None
    ]
    if metric_values and (
        not all(isinstance(value, bool) for value in metric_values)
        or any(value != metric_values[0] for value in metric_values[1:])
    ):
        return None
    metric_value = metric_values[0] if metric_values else None

    if outcome_value is not None and metric_value is not None and outcome_value != metric_value:
        return None
    return outcome_value if outcome_value is not None else metric_value


def _safety_evidence(outcome: dict[str, Any], metrics: dict[str, Any]) -> bool | None:
    collision = _collision_component(outcome, metrics)
    intrusion = _intrusion_component(outcome, metrics)
    if collision is True or intrusion is True:
        return True
    if collision is False and intrusion is False:
        return False
    return None


def _unavailable_projection() -> dict[str, Any]:
    return {
        "status": "not_available",
        "collision_or_severe_intrusion": None,
        "liveness_or_goal_completion": None,
        "comfort_and_efficiency": None,
    }


def constraints_first_outcome_projection_v2(record: dict[str, Any]) -> dict[str, Any]:
    """Project an episode while retaining incomplete safety evidence as unknown."""
    if not isinstance(record, dict):
        return _unavailable_projection()
    outcome = record.get("outcome")
    metrics = record.get("metrics")
    if not isinstance(outcome, dict) or not isinstance(metrics, dict):
        return _unavailable_projection()

    route_complete, route_complete_conflict = _component_from_outcome(outcome, "route_complete")
    timeout_names = ("timeout", "timeout_event")
    timeout, timeout_conflict = _component_from_outcome(outcome, *timeout_names)
    if (
        route_complete is None
        or route_complete_conflict
        or timeout_conflict
        or not _success_metric_matches_route_complete(metrics, route_complete)
    ):
        return _unavailable_projection()

    for name in ("success", "near_misses", "snqi", "path_efficiency"):
        if not _valid_constraints_metric(name, metrics.get(name)):
            return _unavailable_projection()

    collision_or_intrusion = _safety_evidence(outcome, metrics)
    if collision_or_intrusion is None:
        return _unavailable_projection()

    return {
        "status": "observed",
        "collision_or_severe_intrusion": collision_or_intrusion,
        "liveness_or_goal_completion": bool(timeout) or not route_complete,
        "comfort_and_efficiency": {
            "snqi": metrics.get("snqi"),
            "near_misses": metrics.get("near_misses"),
            "path_efficiency": metrics.get("path_efficiency"),
        },
    }


def _optional_numeric_metric(
    value: Any, *, minimum: float | None = None, maximum: float | None = None
) -> tuple[float | None, bool]:
    """Return a finite optional metric and whether its supplied value is valid."""
    if value is None:
        return None, True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, False
    parsed = float(value)
    if not math.isfinite(parsed):
        return None, False
    if minimum is not None and parsed < minimum:
        return None, False
    if maximum is not None and parsed > maximum:
        return None, False
    return parsed, True


def _soft_degradation_components(comfort: dict[str, Any]) -> list[float] | None:
    """Convert available soft metrics to bounded adversarial-score components."""
    components: list[float] = []
    near_misses, valid = _optional_numeric_metric(comfort.get("near_misses"), minimum=0.0)
    if not valid:
        return None
    if near_misses is not None:
        components.append(near_misses / (1.0 + near_misses))

    snqi, valid = _optional_numeric_metric(comfort.get("snqi"))
    if not valid:
        return None
    if snqi is not None:
        components.append(1.0 / (1.0 + max(0.0, snqi)))

    path_efficiency, valid = _optional_numeric_metric(
        comfort.get("path_efficiency"), minimum=0.0, maximum=1.0
    )
    if not valid:
        return None
    if path_efficiency is not None:
        components.append(1.0 - path_efficiency)
    return components


def constraints_first_lexicographic_score_v2(outcome: dict[str, Any]) -> float | None:
    """Score v2 outcomes with disjoint safety/liveness tiers and efficiency degradation."""
    if outcome.get("status") != "observed":
        return None
    safety_failure = outcome.get("collision_or_severe_intrusion")
    liveness_failure = outcome.get("liveness_or_goal_completion")
    if not isinstance(safety_failure, bool) or not isinstance(liveness_failure, bool):
        return None
    comfort = outcome.get("comfort_and_efficiency")
    if not isinstance(comfort, dict):
        return None
    components = _soft_degradation_components(comfort)
    if components is None:
        return None

    soft_component = min(0.999, max(components, default=0.0))
    if safety_failure:
        return float(4.0 + soft_component)
    if liveness_failure:
        return float(2.0 + soft_component)
    return float(soft_component)


def constraints_first_lexicographic_v2(evaluation: CandidateEvaluation) -> float | None:
    """Score available planner executions by safety, liveness, then soft degradation.

    The persisted episode outcome alone is not enough to attribute an event to
    the requested planner: fallback, degraded, unavailable, or unbound execution
    must not steer an optimizer. Preserve those attempts as unscored candidates.
    """
    if evaluation.error is not None:
        return None
    attribution = evaluation.failure_attribution
    if attribution is None or attribution.status != "attributed":
        return None
    details = attribution.details
    if not isinstance(details, dict):
        return None

    def normalized_status(name: str) -> str | None:
        value = details.get(name)
        return value.strip().lower() if isinstance(value, str) and value.strip() else None

    execution_mode = normalized_status("execution_mode")
    readiness_status = normalized_status("readiness_status")
    availability_status = normalized_status("availability_status")
    if (
        execution_mode not in {"native", "adapter", "mixed"}
        or readiness_status not in {"native", "adapter"}
        or availability_status != "available"
    ):
        return None

    record = read_first_jsonl_record(evaluation.episode_record_path)
    if record is None:
        return None
    projection = constraints_first_outcome_projection_v2(record)
    if projection["status"] != "observed":
        return None
    return constraints_first_lexicographic_score_v2(projection)


def register_constraints_first_lexicographic_v2() -> None:
    """Register the additive v2 scorer without changing the historical registry source."""
    register_objective("constraints_first_lexicographic_v2", constraints_first_lexicographic_v2)
