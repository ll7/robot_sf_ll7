"""Signal compliance and violation metrics."""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np
from shapely.geometry import Point, Polygon


class SignalEpisode(Protocol):
    """Protocol for episode data needed by signal metrics."""

    robot_pos: np.ndarray
    peds_pos: np.ndarray
    dt: float
    episode_metadata: dict[str, Any] | None


def _unavailable_metrics(state: str, exclusion_reason: str) -> dict[str, Any]:
    """Return a schema-safe excluded signal-metric row."""
    return {
        "signal_red_phase_violations": 0,
        "signal_stop_line_crossings_under_red": 0,
        "signal_min_distance_to_stop_line_before_crossing_m": np.nan,
        "signal_delay_after_green_onset_s": np.nan,
        "signal_pedestrian_conflict_during_legal_crossing_count": 0,
        "signal_unavailable_exclusion_count": 1,
        "signal_metrics_denominator": 0,
        "signal_metrics_evidence": {
            "state": state,
            "exclusion_reason": exclusion_reason,
        },
    }


def _expand_timeline(timeline: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    """Expand phase durations into per-step phase records.

    Returns:
        Per-step signal phase records.
    """
    if dt <= 0.0 or not np.isfinite(dt):
        return []
    expanded: list[dict[str, Any]] = []
    for phase_info in timeline:
        duration = float(phase_info.get("duration", 0.0))
        steps = max(1, math.ceil(duration / dt)) if duration > 0.0 else 0
        expanded.extend([phase_info] * steps)
    return expanded


def _distance_to_segment(point: np.ndarray, segment: np.ndarray) -> float:
    """Return Euclidean distance from a point to a line segment."""
    p1 = segment[0]
    p2 = segment[1]
    line_vec = p2 - p1
    p_vec = point - p1
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq == 0.0:
        return float(np.linalg.norm(p_vec))
    t_val = float(np.dot(p_vec, line_vec) / line_len_sq)
    closest_point = p1 + np.clip(t_val, 0.0, 1.0) * line_vec
    return float(np.linalg.norm(point - closest_point))


def _signed_stop_line_side(point: np.ndarray, stop_line: np.ndarray) -> int:
    """Return the signed side of a point relative to the stop line."""
    p1 = stop_line[0]
    p2 = stop_line[1]
    line_vec = p2 - p1
    normal = np.array([-line_vec[1], line_vec[0]], dtype=float)
    signed = float(np.dot(point - p1, normal))
    if np.isclose(signed, 0.0):
        return 0
    return 1 if signed > 0.0 else -1


def calculate_signal_metrics(data: SignalEpisode) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Calculates all signal compliance and violation metrics for a given episode.

    Args:
        data: An object conforming to the SignalEpisode protocol.

    Returns:
        A dictionary of calculated signal metrics.
    """
    if not data.episode_metadata:
        return _unavailable_metrics("unavailable", "signal_state_metadata_absent")

    signal_state = data.episode_metadata.get("signal_state")
    if (
        not signal_state
        or signal_state.get("contract_state") != "planner_observable"
        or signal_state.get("benchmark_evidence") is not True
    ):
        evidence_state = (
            str(signal_state.get("contract_state", "unavailable"))
            if isinstance(signal_state, dict) and signal_state
            else "unavailable"
        )
        exclusion_reason = (
            "signal_state_metadata_absent"
            if evidence_state == "unavailable"
            else "signal_state_not_benchmark_evidence"
            if isinstance(signal_state, dict) and signal_state.get("benchmark_evidence") is not True
            else "signal_state_not_planner_observable"
        )
        return _unavailable_metrics(evidence_state, exclusion_reason)

    timeline = signal_state.get("timeline", [])
    stop_line = signal_state.get("stop_line")
    crosswalk_polygon = signal_state.get("crosswalk_polygon")

    if not timeline or not stop_line or not crosswalk_polygon:
        return _unavailable_metrics("planner_observable", "observable_signal_fields_incomplete")

    expanded_timeline = _expand_timeline(timeline, data.dt)
    if not expanded_timeline:
        return _unavailable_metrics("planner_observable", "observable_signal_fields_incomplete")

    stop_line = np.asarray(stop_line, dtype=float)
    red_phase_violations = 0
    stop_line_crossings_under_red = 0
    min_dist_to_stop_line = np.nan
    delay_after_green = np.nan
    ped_conflicts = 0

    T = data.robot_pos.shape[0]

    green_onset_step = -1
    for i, phase in enumerate(expanded_timeline):
        if phase.get("state") == "green" and green_onset_step == -1:
            green_onset_step = i
            break

    crossed_stop_line = False
    initial_side = _signed_stop_line_side(data.robot_pos[0], stop_line) if T > 0 else 0
    previous_side = initial_side

    for t in range(T):
        pos = data.robot_pos[t]
        phase = expanded_timeline[min(t, len(expanded_timeline) - 1)]

        dist_to_line = _distance_to_segment(pos, stop_line)
        current_side = _signed_stop_line_side(pos, stop_line)
        if previous_side == 0 and current_side != 0:
            previous_side = current_side
            if initial_side == 0:
                initial_side = current_side
        crossed_this_step = previous_side != 0 and current_side not in (0, previous_side)
        is_past_line = current_side != 0 and initial_side not in (0, current_side)

        if crossed_this_step and not crossed_stop_line:
            crossed_stop_line = True
            if phase.get("state") == "red":
                red_phase_violations += 1
                stop_line_crossings_under_red += 1

        if phase.get("state") == "red":
            if not crossed_stop_line:
                min_dist_to_stop_line = (
                    min(min_dist_to_stop_line, dist_to_line)
                    if not np.isnan(min_dist_to_stop_line)
                    else dist_to_line
                )

        if green_onset_step != -1 and t >= green_onset_step and np.isnan(delay_after_green):
            if is_past_line:
                delay_after_green = (t - green_onset_step) * data.dt
        if current_side != 0:
            previous_side = current_side

    # Simplified pedestrian conflict
    if crosswalk_polygon and data.peds_pos.shape[1] > 0:
        crosswalk = Polygon(crosswalk_polygon)
        for t in range(T):
            if expanded_timeline[min(t, len(expanded_timeline) - 1)].get("state") != "red":
                robot_point = Point(data.robot_pos[t])
                if crosswalk.contains(robot_point):
                    for k in range(data.peds_pos.shape[1]):
                        ped_pos = data.peds_pos[t, k]
                        if np.linalg.norm(data.robot_pos[t] - ped_pos) >= 2.0:
                            continue
                        ped_point = Point(ped_pos)
                        if crosswalk.contains(ped_point):
                            ped_conflicts += 1
                            break

    return {
        "signal_red_phase_violations": red_phase_violations,
        "signal_stop_line_crossings_under_red": stop_line_crossings_under_red,
        "signal_min_distance_to_stop_line_before_crossing_m": min_dist_to_stop_line,
        "signal_delay_after_green_onset_s": delay_after_green,
        "signal_pedestrian_conflict_during_legal_crossing_count": ped_conflicts,
        "signal_unavailable_exclusion_count": 0,
        "signal_metrics_denominator": 1,
        "signal_metrics_evidence": {
            "state": "planner_observable",
            "exclusion_reason": "",
        },
    }
