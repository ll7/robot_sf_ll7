"""Scenario criticality objective for classical optimization baseline.

This module defines a criticality scoring function for scenario parameter optimization.
It uses existing benchmark metrics to compute an interpretable scalar objective that
increases with collision risk, near-misses, low clearance, and progress failures.

Public API:
    - compute_criticality_score(): scalar score for optimization
    - compute_criticality_decomposed(): detailed metric breakdown
    - CriticalityObjectiveConfig: configuration dataclass
    - apply_criticality_parameters(): scenario parameter patching

Claim boundary: exploratory/diagnostic-only; not a validated benchmark method.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.benchmark.map_runner_episode import EpisodeData


@dataclass
class CriticalityObjectiveConfig:
    """Configuration for criticality objective computation.

    Attributes:
        collision_weight: Weight for collision events (default: 10.0)
        near_miss_weight: Weight for near-miss events (default: 2.0)
        clearance_margin: Clearance threshold for penalty in meters (default: 0.5)
        clearance_weight: Weight for clearance violation (default: 1.0)
        progress_failure_weight: Weight for failure-to-progress (default: 5.0)
        stalled_time_weight: Weight for stalled time in seconds (default: 0.5)
        fail_closed: If True, missing required metrics produce not_evaluable (default: True)
    """

    collision_weight: float = 10.0
    near_miss_weight: float = 2.0
    clearance_margin: float = 0.5
    clearance_weight: float = 1.0
    progress_failure_weight: float = 5.0
    stalled_time_weight: float = 0.5
    fail_closed: bool = True


@dataclass
class CriticalityResult:
    """Result of criticality objective computation.

    Attributes:
        criticality_score: Scalar score for optimization (higher = more critical)
        collision_term: Contribution from collisions
        near_miss_term: Contribution from near-misses
        clearance_term: Contribution from low clearance
        progress_failure_term: Contribution from progress failure
        stalled_time_term: Contribution from stalled time
        status: "evaluated", "not_evaluable", or "invalid_candidate"
        reason: Optional explanation for non-evaluated status
        raw_metrics: Dict of raw metric values used
    """

    criticality_score: float
    collision_term: float
    near_miss_term: float
    clearance_term: float
    progress_failure_term: float
    stalled_time_term: float
    status: str
    reason: str | None = None
    raw_metrics: dict[str, float] | None = None


def _safe_get_metric(
    metrics: dict[str, Any],
    key: str,
    default: float | None = None,
    fail_closed: bool = True,
) -> float | None:
    """Safely extract a metric value, respecting fail-closed policy.

    Returns:
        The metric value as float, default if missing and not fail_closed,
        or None if missing and fail_closed.
    """
    if key not in metrics:
        if fail_closed:
            return None
        return default if default is not None else float("nan")

    value = metrics[key]
    if value is None:
        if fail_closed:
            return None
        return default if default is not None else float("nan")

    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value):  # noqa: PLR0124
            if fail_closed:
                return None
        return float(value)

    if fail_closed:
        return None
    return default if default is not None else float("nan")


def compute_criticality_score(
    episode_data: EpisodeData,
    config: CriticalityObjectiveConfig | None = None,
) -> CriticalityResult:
    """Compute criticality score for an episode.

    The criticality score increases with:
    - Collision events (highest penalty)
    - Near-miss events
    - Low clearance (below clearance_margin)
    - Failure to progress
    - Stalled time

    Args:
        episode_data: Episode data with metrics
        config: Objective configuration (uses defaults if None)

    Returns:
        CriticalityResult with score and decomposition
    """
    if config is None:
        config = CriticalityObjectiveConfig()

    metrics = episode_data.metrics if hasattr(episode_data, "metrics") else {}

    fail_closed = config.fail_closed

    collision_count = _safe_get_metric(metrics, "collision_count", 0.0, fail_closed)
    near_misses = _safe_get_metric(metrics, "near_misses", 0.0, fail_closed)
    min_clearance = _safe_get_metric(metrics, "min_clearance", float("nan"), fail_closed)
    failure_to_progress = _safe_get_metric(metrics, "failure_to_progress", 0.0, fail_closed)
    stalled_time = _safe_get_metric(metrics, "stalled_time", 0.0, fail_closed)

    required_metrics = [collision_count, near_misses, min_clearance, failure_to_progress, stalled_time]
    if fail_closed and any(m is None for m in required_metrics):
        return CriticalityResult(
            criticality_score=float("nan"),
            collision_term=0.0,
            near_miss_term=0.0,
            clearance_term=0.0,
            progress_failure_term=0.0,
            stalled_time_term=0.0,
            status="not_evaluable",
            reason="Missing required metrics with fail_closed=True",
            raw_metrics=metrics,
        )

    collision_count = collision_count if collision_count is not None else 0.0
    near_misses = near_misses if near_misses is not None else 0.0
    min_clearance = min_clearance if min_clearance is not None else float("nan")
    failure_to_progress = failure_to_progress if failure_to_progress is not None else 0.0
    stalled_time = stalled_time if stalled_time is not None else 0.0

    collision_term = config.collision_weight * collision_count

    near_miss_term = config.near_miss_weight * near_misses

    if min_clearance == min_clearance:  # noqa: PLR0124
        clearance_violation = max(0.0, config.clearance_margin - min_clearance)
    else:
        clearance_violation = 0.0
    clearance_term = config.clearance_weight * clearance_violation

    progress_failure_term = config.progress_failure_weight * failure_to_progress

    stalled_time_term = config.stalled_time_weight * stalled_time

    criticality_score = (
        collision_term
        + near_miss_term
        + clearance_term
        + progress_failure_term
        + stalled_time_term
    )

    return CriticalityResult(
        criticality_score=criticality_score,
        collision_term=collision_term,
        near_miss_term=near_miss_term,
        clearance_term=clearance_term,
        progress_failure_term=progress_failure_term,
        stalled_time_term=stalled_time_term,
        status="evaluated",
        raw_metrics={
            "collision_count": collision_count,
            "near_misses": near_misses,
            "min_clearance": min_clearance,
            "failure_to_progress": failure_to_progress,
            "stalled_time": stalled_time,
        },
    )


def _apply_speed_scale(ped: dict[str, Any], value: float) -> None:
    if "speed_multiplier" in ped:
        ped["speed_multiplier"] = float(value)
    elif "speed_mps" in ped:
        ped["speed_mps"] = ped["speed_mps"] * float(value)


def _apply_start_delay(ped: dict[str, Any], value: float) -> None:
    ped["start_delay_s"] = float(value)


def _apply_waypoint_offset(ped: dict[str, Any], value: float) -> None:
    if "waypoints" not in ped:
        return
    waypoints = ped["waypoints"]
    if not isinstance(waypoints, list) or len(waypoints) == 0:
        return
    wp = waypoints[0]
    if isinstance(wp, dict) and "y" in wp:
        wp["y"] = wp.get("y", 0.0) + float(value)
    elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
        waypoints[0] = list(wp)
        waypoints[0][1] = waypoints[0][1] + float(value)


def _apply_robot_offset(patched: dict[str, Any], value: float) -> None:
    if "robot" not in patched:
        return
    robot = patched["robot"]
    if not isinstance(robot, dict):
        return
    if "start_x" in robot:
        robot["start_x"] = robot.get("start_x", 0.0) + float(value)
    elif "start" in robot:
        start = robot["start"]
        if isinstance(start, (list, tuple)) and len(start) >= 1:
            robot["start"] = [start[0] + float(value)] + list(start[1:])


_PARAM_APPLIERS: dict[str, Any] = {
    "pedestrian_speed_scale": _apply_speed_scale,
    "pedestrian_start_delay_s": _apply_start_delay,
    "crossing_waypoint_y_offset_m": _apply_waypoint_offset,
}


def _apply_single_param(
    patched: dict[str, Any],
    key: str,
    value: float,
) -> None:
    if key == "robot_start_offset_m":
        _apply_robot_offset(patched, value)
    elif key in _PARAM_APPLIERS and "pedestrians" in patched:
        for ped in patched["pedestrians"]:
            if isinstance(ped, dict):
                _PARAM_APPLIERS[key](ped, value)


def _validate_params(params: dict[str, float]) -> None:
    for key, value in params.items():
        if not isinstance(key, str):
            raise ValueError(f"parameter key must be string, got {key!r}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"parameter value must be numeric, got {key}={value!r}")


def apply_criticality_parameters(
    scenario: dict[str, Any],
    params: dict[str, float],
) -> dict[str, Any]:
    """Apply criticality parameters to a scenario without mutation.

    Creates a deep copy of the scenario and applies parameter perturbations.
    Attaches parameter metadata to the scenario.

    Args:
        scenario: Original scenario dict
        params: Parameter values to apply

    Returns:
        New scenario dict with parameters applied and metadata attached

    Raises:
        ValueError: If scenario is invalid or parameters cannot be applied
    """
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a dict")
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _validate_params(params)

    patched = copy.deepcopy(scenario)

    for key, value in params.items():
        _apply_single_param(patched, key, float(value))

    if "metadata" not in patched:
        patched["metadata"] = {}
    patched["metadata"]["issue_4362_criticality_parameters"] = params

    original_id = patched.get("id", patched.get("scenario_id", "unknown"))
    param_hash = "_".join(f"{k}{v:.3f}" for k, v in sorted(params.items()))
    patched["id"] = f"{original_id}_crit_{param_hash}"
    patched["metadata"]["issue_4362_candidate_id"] = patched["id"]

    return patched


__all__ = [
    "CriticalityObjectiveConfig",
    "CriticalityResult",
    "apply_criticality_parameters",
    "compute_criticality_score",
]
