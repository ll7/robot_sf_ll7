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
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    if "snqi" in metrics:
        return -_metric(metrics, "snqi")
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    collision = 1.0 if bool(outcome.get("collision") or outcome.get("collision_event")) else 0.0
    timeout = 1.0 if bool(outcome.get("timeout") or outcome.get("timeout_event")) else 0.0
    success = _metric(metrics, "success", 1.0 if bool(outcome.get("route_complete")) else 0.0)
    near = _metric(metrics, "near_misses", 0.0)
    return float(collision * 10.0 + timeout * 3.0 + near - success)


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

    def _flag(*names: str) -> bool | None:
        """Read one required boolean flag, accepting the canonical aliases."""
        present = [outcome[name] for name in names if name in outcome]
        if not present or not all(isinstance(value, bool) for value in present):
            return None
        return any(present)

    route_complete = _flag("route_complete")
    collision_names = ("collision", "collision_event", "severe_intrusion")
    timeout_names = ("timeout", "timeout_event")
    collision_or_intrusion = _flag(*collision_names)
    timeout = _flag(*timeout_names)
    metric_collision_names = ("collisions", "severe_intrusion", "severe_intrusion_event")
    if (
        route_complete is None
        or (collision_or_intrusion is None and any(name in outcome for name in collision_names))
        or (timeout is None and any(name in outcome for name in timeout_names))
        or (
            collision_or_intrusion is None
            and not any(metrics.get(name) is not None for name in metric_collision_names)
        )
    ):
        return _unavailable_constraints_first_outcome()

    for name in ("collisions", "success", "near_misses", "snqi", "path_efficiency"):
        value = metrics.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return _unavailable_constraints_first_outcome()
    for name in ("severe_intrusion", "severe_intrusion_event"):
        value = metrics.get(name)
        if value is not None and not isinstance(value, bool):
            return _unavailable_constraints_first_outcome()

    collision_or_intrusion = bool(
        collision_or_intrusion
        or metrics.get("severe_intrusion") is True
        or metrics.get("severe_intrusion_event") is True
        or _metric(metrics, "collisions") > 0.0
    )
    goal_complete = bool(route_complete) or _metric(metrics, "success") >= 1.0
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
    metrics = record["metrics"]
    collision_or_intrusion = projection["collision_or_severe_intrusion"]
    liveness_failure = projection["liveness_or_goal_completion"]

    near_miss_count = max(0.0, _metric(metrics, "near_misses"))
    near_miss_component = near_miss_count / (1.0 + near_miss_count)
    snqi = metrics.get("snqi")
    try:
        parsed_snqi = float(snqi)
    except (TypeError, ValueError):
        parsed_snqi = math.nan
    snqi_component = 1.0 / (1.0 + max(0.0, parsed_snqi)) if math.isfinite(parsed_snqi) else 0.0
    soft_component = min(0.999, max(near_miss_component, snqi_component))

    if collision_or_intrusion:
        return float(4.0 + soft_component)
    if liveness_failure:
        return float(2.0 + soft_component)
    return float(soft_component)


_OBJECTIVES: dict[str, ObjectiveFn] = {
    "constraints_first_lexicographic_v1": constraints_first_lexicographic_v1,
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
