"""Objective registry for adversarial scenario search."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.robustness import temporal_robustness_objective

if TYPE_CHECKING:
    from robot_sf.adversarial.config import CandidateEvaluation

ObjectiveFn = Callable[["CandidateEvaluation"], float | None]


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a finite metric scalar with a default."""
    value = metrics.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def worst_case_snqi(evaluation: CandidateEvaluation) -> float | None:
    """Score lower-SNQI or failure-like records higher for maximization.

    If a benchmark record already contains ``metrics.snqi``, this returns
    ``-snqi`` so the search maximizes worse social-navigation quality. Some
    smoke paths do not compute calibrated SNQI; for those records, a conservative
    fallback failure score keeps the runner usable without pretending the value
    is camera-ready SNQI evidence.
    """
    record = read_first_jsonl_record(evaluation.episode_record_path)
    if record is None:
        return None
    raw_metrics = record.get("metrics")
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
    if "snqi" in metrics:
        return -_metric(metrics, "snqi")
    raw_outcome = record.get("outcome")
    outcome: dict[str, Any] = raw_outcome if isinstance(raw_outcome, dict) else {}
    collision = 1.0 if bool(outcome.get("collision") or outcome.get("collision_event")) else 0.0
    timeout = 1.0 if bool(outcome.get("timeout") or outcome.get("timeout_event")) else 0.0
    success = _metric(metrics, "success", 1.0 if bool(outcome.get("route_complete")) else 0.0)
    near = _metric(metrics, "near_misses", 0.0)
    return float(collision * 10.0 + timeout * 3.0 + near - success)


def minimize_episode_min_robot_distance(evaluation: CandidateEvaluation) -> float | None:
    """Score a completed episode by its global minimum robot-pedestrian distance.

    The open-loop scenario-search arm receives completed episode records rather than the
    reactive arm's one-step state snapshot. Keep this projection separately named and
    fail closed when the canonical ``metrics.min_distance`` value is missing, malformed,
    negative, or non-finite. The search API maximizes scores, so the finite distance is
    negated to prefer closer approaches.
    """
    record = read_first_jsonl_record(evaluation.episode_record_path)
    if record is None:
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("min_distance")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    distance = float(value)
    if not math.isfinite(distance) or distance < 0.0:
        return None
    return -distance


def _valid_constraints_metric(name: str, value: Any) -> bool:
    """Return whether one scalar metric can support the constraints-first projection."""
    if value is None:
        return True
    if name == "success" and isinstance(value, bool):
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    parsed = float(value)
    if not math.isfinite(parsed):
        return False
    if name == "success":
        return parsed in {0.0, 1.0}
    if name in {"collisions", "near_misses"}:
        return parsed >= 0.0
    if name == "path_efficiency":
        return 0.0 <= parsed <= 1.0
    return True


def _success_metric_matches_route_complete(metrics: dict[str, Any], route_complete: bool) -> bool:
    """Return whether an optional canonical success metric agrees with the outcome flag."""
    metric_success = metrics.get("success")
    return metric_success is None or bool(metric_success) == route_complete


def _consistent_boolean_alias(outcome: dict[str, Any], *names: str) -> bool | None:
    """Read aliases only when all present values are booleans with one value."""
    present = [outcome[name] for name in names if name in outcome]
    if (
        not present
        or not all(isinstance(value, bool) for value in present)
        or any(value != present[0] for value in present[1:])
    ):
        return None
    return present[0]


def _safety_evidence(outcome: dict[str, Any], metrics: dict[str, Any]) -> bool | None:
    """Combine collision and intrusion evidence without hiding contradictions."""
    collision_names = ("collision", "collision_event")
    intrusion_names = ("severe_intrusion", "severe_intrusion_event")
    collision = _consistent_boolean_alias(outcome, *collision_names)
    intrusion = _consistent_boolean_alias(outcome, *intrusion_names)
    if (collision is None and any(name in outcome for name in collision_names)) or (
        intrusion is None and any(name in outcome for name in intrusion_names)
    ):
        return None

    metric_collision = metrics["collisions"] > 0 if metrics.get("collisions") is not None else None
    metric_intrusion_values = [
        metrics[name]
        for name in ("severe_intrusion", "severe_intrusion_event")
        if metrics.get(name) is not None
    ]
    if metric_intrusion_values and any(
        value != metric_intrusion_values[0] for value in metric_intrusion_values[1:]
    ):
        return None
    metric_intrusion = metric_intrusion_values[0] if metric_intrusion_values else None
    if (
        collision is not None and metric_collision is not None and collision != metric_collision
    ) or (intrusion is not None and metric_intrusion is not None and intrusion != metric_intrusion):
        return None

    evidence = (collision, intrusion, metric_collision, metric_intrusion)
    present = [value for value in evidence if value is not None]
    return any(present) if present else None


def constraints_first_outcome_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Project one episode record into the strict constraints-first outcome vector.

    Missing containers or non-boolean outcome flags are unavailable rather than
    being coerced into a liveness failure.  This keeps the search objective and
    the diagnostic row writer aligned when an episode record is malformed.
    """
    if not isinstance(record, dict):
        return _unavailable_constraints_first_outcome()
    outcome = record.get("outcome")
    metrics = record.get("metrics")
    if not isinstance(outcome, dict) or not isinstance(metrics, dict):
        return _unavailable_constraints_first_outcome()

    route_complete = _consistent_boolean_alias(outcome, "route_complete")
    timeout_names = ("timeout", "timeout_event")
    timeout = _consistent_boolean_alias(outcome, *timeout_names)
    if (
        route_complete is None
        or (timeout is None and any(name in outcome for name in timeout_names))
        or not _success_metric_matches_route_complete(metrics, route_complete)
    ):
        return _unavailable_constraints_first_outcome()

    for name in ("collisions", "success", "near_misses", "snqi", "path_efficiency"):
        value = metrics.get(name)
        # ``post_process_metrics`` emits the canonical success metric as a bool;
        # the other scalar metrics must remain numeric so malformed records do
        # not get coerced into a clean outcome.
        if not _valid_constraints_metric(name, value):
            return _unavailable_constraints_first_outcome()
    for name in ("severe_intrusion", "severe_intrusion_event"):
        value = metrics.get(name)
        if value is not None and not isinstance(value, bool):
            return _unavailable_constraints_first_outcome()

    collision_or_intrusion = _safety_evidence(outcome, metrics)
    if collision_or_intrusion is None:
        return _unavailable_constraints_first_outcome()

    goal_complete = route_complete
    liveness_failure = bool(timeout) or not goal_complete
    return {
        "status": "observed",
        "collision_or_severe_intrusion": collision_or_intrusion,
        "liveness_or_goal_completion": liveness_failure,
        "comfort_and_efficiency": {
            "snqi": metrics.get("snqi"),
            "near_misses": metrics.get("near_misses"),
            "path_efficiency": metrics.get("path_efficiency"),
        },
    }


def _unavailable_constraints_first_outcome() -> dict[str, Any]:
    """Return the schema-shaped unavailable constraints-first projection."""
    return {
        "status": "not_available",
        "collision_or_severe_intrusion": None,
        "liveness_or_goal_completion": None,
        "comfort_and_efficiency": None,
    }


def constraints_first_lexicographic_score(outcome: dict[str, Any]) -> float | None:
    """Return the frozen scalar score for one observed constraints-first outcome.

    This pure projection-level form is shared by the search objective and the
    diagnostic verifier so an outcome row cannot self-hash a score that merely
    occupies the right tier while disagreeing with its recorded soft metrics.
    """
    if outcome.get("status") != "observed":
        return None
    comfort = outcome.get("comfort_and_efficiency")
    if not isinstance(comfort, dict):
        return None
    try:
        near_miss_count = max(0.0, float(comfort.get("near_misses", 0.0)))
    except (TypeError, ValueError):
        near_miss_count = 0.0
    try:
        parsed_snqi = float(comfort.get("snqi"))
    except (TypeError, ValueError):
        parsed_snqi = math.nan
    if not math.isfinite(near_miss_count):
        near_miss_count = 0.0
    near_miss_component = near_miss_count / (1.0 + near_miss_count)
    snqi_component = 1.0 / (1.0 + max(0.0, parsed_snqi)) if math.isfinite(parsed_snqi) else 0.0
    soft_component = min(0.999, max(near_miss_component, snqi_component))
    if outcome.get("collision_or_severe_intrusion") is True:
        return float(4.0 + soft_component)
    if outcome.get("liveness_or_goal_completion") is True:
        return float(2.0 + soft_component)
    if (
        outcome.get("collision_or_severe_intrusion") is False
        and outcome.get("liveness_or_goal_completion") is False
    ):
        return float(soft_component)
    return None


def constraints_first_lexicographic_v1(evaluation: CandidateEvaluation) -> float | None:
    """Score adversarial outcomes with bounded, constraints-first tiers.

    The search API accepts a scalar objective, so this encodes the frozen
    lexicographic ordering in disjoint score bands: collision/severe intrusion
    (``[4, 5)``), liveness failure (``[2, 3)``), then bounded
    comfort/efficiency degradation (``[0, 1)``).  A lower tier can therefore
    never compensate for a failed higher-priority safety or liveness condition.
    """
    record = read_first_jsonl_record(evaluation.episode_record_path)
    if record is None:
        return None

    projection = constraints_first_outcome_projection(record)
    if projection["status"] != "observed":
        return None
    return constraints_first_lexicographic_score(projection)


_OBJECTIVES: dict[str, ObjectiveFn] = {
    "constraints_first_lexicographic_v1": constraints_first_lexicographic_v1,
    "minimize_episode_min_robot_distance": minimize_episode_min_robot_distance,
    "worst_case_snqi": worst_case_snqi,
    "temporal_robustness": temporal_robustness_objective,
}


def register_objective(name: str, objective: ObjectiveFn) -> None:
    """Register an objective function by name."""
    key = name.strip()
    if not key:
        raise ValueError("objective name must be non-empty")
    _OBJECTIVES[key] = objective


def unregister_objective(name: str) -> None:
    """Remove a registered objective function if present."""
    _OBJECTIVES.pop(name.strip(), None)


def get_objective(name: str) -> ObjectiveFn:
    """Return a registered objective function."""
    try:
        return _OBJECTIVES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_OBJECTIVES))
        raise ValueError(f"Unknown adversarial objective '{name}'. Available: {available}") from exc


def list_objectives() -> tuple[str, ...]:
    """Return registered objective names."""
    return tuple(sorted(_OBJECTIVES))
