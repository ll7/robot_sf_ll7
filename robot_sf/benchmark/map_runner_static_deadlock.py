"""Static-deadlock mechanism trace helpers for map-runner episode rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

_STATIC_DEADLOCK_SUITE_ID = "static_deadlock_recovery"
_STATIC_DEADLOCK_LOW_PROGRESS_WINDOW_STEPS = 10
_STATIC_DEADLOCK_LOW_PROGRESS_THRESHOLD_M = 0.05


def _is_static_deadlock_suite(scenario: Mapping[str, Any]) -> bool:
    """Return whether a scenario row belongs to the static-deadlock mechanism suite."""
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return str(metadata.get("mechanism_aware_suite_id", "")) == _STATIC_DEADLOCK_SUITE_ID


def _finite_or_none(value: float) -> float | None:
    """Return a JSON-friendly float when finite."""
    value = float(value)
    return value if math.isfinite(value) else None


def _mechanism_row_status(termination_reason: str) -> str:
    """Classify an episode row for mechanism-suite reportability accounting.

    Returns:
        ``"completed"`` for valid terminal rows, or ``"failed"`` for error rows.
    """
    return "failed" if str(termination_reason).strip().lower() == "error" else "completed"


def _recenter_activation_count(planner_decision_trace: list[dict[str, Any]]) -> int:
    """Count planner-decision steps whose static-recenter term was active.

    Returns:
        Number of recorded decision steps with a positive static-recenter term.
    """
    count = 0
    for step in planner_decision_trace:
        value = step.get("static_recenter")
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) > 0.0:
            count += 1
    return count


def static_deadlock_trace_fields(
    scenario: Mapping[str, Any],
    *,
    robot_pos_arr: np.ndarray,
    goal_vec: np.ndarray,
    initial_goal_distance: float,
    termination_reason: str,
    outcome: Mapping[str, bool],
    planner_decision_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build static-deadlock mechanism trace fields for suite-tagged episode rows.

    Returns:
        Top-level episode row fields required by the static-deadlock suite, or an empty payload for
        unrelated scenarios.
    """
    if not _is_static_deadlock_suite(scenario):
        return {}

    if robot_pos_arr.size:
        distances = np.linalg.norm(robot_pos_arr - goal_vec, axis=1)
        distance_series = [float(initial_goal_distance), *[float(value) for value in distances]]
    else:
        distance_series = [float(initial_goal_distance)]

    sample_count = max(0, len(distance_series) - 1)
    window_steps = min(_STATIC_DEADLOCK_LOW_PROGRESS_WINDOW_STEPS, sample_count)
    window_start_idx = max(0, len(distance_series) - 1 - window_steps)
    window_start_distance = distance_series[window_start_idx]
    final_distance = distance_series[-1]
    window_progress_delta = window_start_distance - final_distance
    total_progress_delta = float(initial_goal_distance) - final_distance
    route_complete = bool(outcome.get("route_complete", False))
    collision_event = bool(outcome.get("collision_event", False))
    timeout_event = bool(outcome.get("timeout_event", False))
    low_progress_active = (
        sample_count > 0
        and not route_complete
        and window_progress_delta <= _STATIC_DEADLOCK_LOW_PROGRESS_THRESHOLD_M
    )
    local_minimum = bool(low_progress_active and timeout_event and not collision_event)

    return {
        "low_progress_window": {
            "schema_version": "static-deadlock-low-progress-window.v1",
            "window_steps": int(window_steps),
            "sample_count": int(sample_count),
            "start_distance_to_goal_m": _finite_or_none(window_start_distance),
            "end_distance_to_goal_m": _finite_or_none(final_distance),
            "progress_delta_m": _finite_or_none(window_progress_delta),
            "threshold_m": float(_STATIC_DEADLOCK_LOW_PROGRESS_THRESHOLD_M),
            "active": bool(low_progress_active),
        },
        "recenter_activation_count": int(_recenter_activation_count(planner_decision_trace)),
        "distance_to_goal_delta": {
            "schema_version": "static-deadlock-distance-to-goal-delta.v1",
            "initial_distance_to_goal_m": _finite_or_none(initial_goal_distance),
            "final_distance_to_goal_m": _finite_or_none(final_distance),
            "delta_m": _finite_or_none(total_progress_delta),
            "interpretation": "positive values indicate progress toward the goal",
        },
        "local_minimum_indicator": {
            "schema_version": "static-deadlock-local-minimum-indicator.v1",
            "is_local_minimum": local_minimum,
            "status": "detected" if local_minimum else "not_detected",
            "reason": (
                "timeout with low progress and no collision"
                if local_minimum
                else "route completed, collision occurred, or low-progress timeout was not observed"
            ),
        },
        "row_status": _mechanism_row_status(termination_reason),
    }
