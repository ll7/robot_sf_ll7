"""Signal compliance and violation metrics."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from shapely.geometry import Point, Polygon


class SignalEpisode(Protocol):
    """Protocol for episode data needed by signal metrics."""

    robot_pos: np.ndarray
    peds_pos: np.ndarray
    dt: float
    episode_metadata: dict[str, Any] | None


def calculate_signal_metrics(data: SignalEpisode) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Calculates all signal compliance and violation metrics for a given episode.

    Args:
        data: An object conforming to the SignalEpisode protocol.

    Returns:
        A dictionary of calculated signal metrics.
    """
    if not data.episode_metadata:
        return {
            "signal_red_phase_violations": 0,
            "signal_stop_line_crossings_under_red": 0,
            "signal_min_distance_to_stop_line_before_crossing_m": np.nan,
            "signal_delay_after_green_onset_s": np.nan,
            "signal_pedestrian_conflict_during_legal_crossing_count": 0,
            "signal_unavailable_exclusion_count": 1,
            "signal_metrics_denominator": 0,
            "signal_metrics_evidence": {
                "state": "unavailable",
                "exclusion_reason": "signal_state_metadata_absent",
            },
        }

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
        return {
            "signal_red_phase_violations": 0,
            "signal_stop_line_crossings_under_red": 0,
            "signal_min_distance_to_stop_line_before_crossing_m": np.nan,
            "signal_delay_after_green_onset_s": np.nan,
            "signal_pedestrian_conflict_during_legal_crossing_count": 0,
            "signal_unavailable_exclusion_count": 1,
            "signal_metrics_denominator": 0,
            "signal_metrics_evidence": {
                "state": evidence_state,
                "exclusion_reason": exclusion_reason,
            },
        }

    timeline = signal_state.get("timeline", [])
    stop_line = signal_state.get("stop_line")
    crosswalk_polygon = signal_state.get("crosswalk_polygon")

    if not timeline or not stop_line:
        return {
            "signal_red_phase_violations": 0,
            "signal_stop_line_crossings_under_red": 0,
            "signal_min_distance_to_stop_line_before_crossing_m": np.nan,
            "signal_delay_after_green_onset_s": np.nan,
            "signal_pedestrian_conflict_during_legal_crossing_count": 0,
            "signal_unavailable_exclusion_count": 1,
            "signal_metrics_denominator": 0,
            "signal_metrics_evidence": {
                "state": "planner_observable",
                "exclusion_reason": "observable_signal_fields_incomplete",
            },
        }

    # Expand the timeline based on duration and dt
    expanded_timeline = []
    for phase_info in timeline:
        steps = int(phase_info["duration"] / data.dt)
        for _ in range(steps):
            expanded_timeline.append(phase_info)

    stop_line = np.array(stop_line)
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

    for t in range(T):
        pos = data.robot_pos[t]
        phase = expanded_timeline[min(t, len(expanded_timeline) - 1)]

        # Distance to stop line (assuming stop line is a line segment)
        p1 = stop_line[0]
        p2 = stop_line[1]
        line_vec = p2 - p1
        p_vec = pos - p1
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:  # Should not happen with valid stop line
            dist_to_line = np.linalg.norm(p_vec)
        else:
            t_val = np.dot(p_vec, line_vec) / line_len_sq
            t_val = np.clip(t_val, 0, 1)
            closest_point = p1 + t_val * line_vec
            dist_to_line = np.linalg.norm(pos - closest_point)

        # Check if robot is past the stop line
        is_past_line = pos[0] > stop_line[0][0]

        if phase.get("state") == "red":
            if is_past_line:
                red_phase_violations += 1
                if not crossed_stop_line:
                    stop_line_crossings_under_red += 1
            elif not crossed_stop_line:
                min_dist_to_stop_line = (
                    min(min_dist_to_stop_line, dist_to_line)
                    if not np.isnan(min_dist_to_stop_line)
                    else dist_to_line
                )

        if green_onset_step != -1 and t >= green_onset_step and np.isnan(delay_after_green):
            if is_past_line:
                delay_after_green = (t - green_onset_step) * data.dt

        if is_past_line:
            crossed_stop_line = True

    # Simplified pedestrian conflict
    if crosswalk_polygon and data.peds_pos.shape[1] > 0:
        crosswalk = Polygon(crosswalk_polygon)
        for t in range(T):
            if not expanded_timeline[min(t, len(expanded_timeline) - 1)].get("state") == "red":
                robot_point = Point(data.robot_pos[t])
                if crosswalk.contains(robot_point):
                    for k in range(data.peds_pos.shape[1]):
                        ped_point = Point(data.peds_pos[t, k])
                        if crosswalk.contains(ped_point) and robot_point.distance(ped_point) < 2.0:
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
