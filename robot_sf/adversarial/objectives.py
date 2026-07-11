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


_OBJECTIVES: dict[str, ObjectiveFn] = {
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
