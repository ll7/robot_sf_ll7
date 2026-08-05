"""Pair compatibility records for renderer-neutral worked-example traces."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from robot_sf.analysis_workbench.simulation_trace_export import SimulationTraceExport

PAIR_COMPATIBILITY_PROFILE_VERSION = "pair_compatibility.deterministic.v1"


def unavailable_pair_compatibility(reason: str = "no_pair_trace_declared") -> dict[str, Any]:
    """Return an explicit unavailable pair-compatibility record."""

    return {
        "profile_version": PAIR_COMPATIBILITY_PROFILE_VERSION,
        "status": "unavailable",
        "reason": reason,
        "initial_state_equivalence": {"status": "unavailable", "reason": reason},
        "route_spawn_separation": {"status": "unavailable", "reason": reason},
        "shared_prefix": {"status": "unavailable", "reason": reason},
        "valid_common_event_anchors": [],
        "divergence_interpretation": {
            "allowed": False,
            "reason": reason,
        },
    }


def build_pair_compatibility_record(
    left: SimulationTraceExport,
    right: SimulationTraceExport,
    *,
    left_events: Sequence[Mapping[str, Any]],
    right_events: Sequence[Mapping[str, Any]],
    position_tolerance_m: float = 1e-6,
    heading_tolerance_rad: float = 1e-6,
    shared_prefix_steps: int = 3,
) -> dict[str, Any]:
    """Build deterministic pair-compatibility diagnostics without normalizing durations.

    Returns:
        JSON-safe pair compatibility record.
    """

    if not left.frames or not right.frames:
        return unavailable_pair_compatibility("empty_pair_trace")
    initial = _initial_equivalence(
        left,
        right,
        position_tolerance_m=position_tolerance_m,
        heading_tolerance_rad=heading_tolerance_rad,
    )
    shared_prefix = _shared_prefix(
        left,
        right,
        position_tolerance_m=position_tolerance_m,
        max_steps=shared_prefix_steps,
    )
    route_spawn = {
        "status": "available",
        "scenario_id_equal": left.source.scenario_id == right.source.scenario_id,
        "planner_id_left": left.source.planner_id,
        "planner_id_right": right.source.planner_id,
        "seed_left": left.source.seed,
        "seed_right": right.source.seed,
    }
    common_event_types = sorted(
        {str(event["event_type"]) for event in left_events if event.get("status") == "available"}
        & {str(event["event_type"]) for event in right_events if event.get("status") == "available"}
    )
    shared_prefix_status = shared_prefix["status"] == "available" and bool(
        shared_prefix["shared_prefix"]
    )
    return {
        "profile_version": PAIR_COMPATIBILITY_PROFILE_VERSION,
        "status": "available",
        "initial_state_equivalence": initial,
        "route_spawn_separation": route_spawn,
        "shared_prefix": shared_prefix,
        "valid_common_event_anchors": common_event_types if initial["equivalent"] else [],
        "duration_normalization": {"applied": False},
        "divergence_interpretation": {
            "allowed": shared_prefix_status,
            "reason": (
                "shared_prefix_available"
                if shared_prefix_status
                else "no_shared_prefix_reject_divergence_output"
            ),
        },
    }


def _initial_equivalence(
    left: SimulationTraceExport,
    right: SimulationTraceExport,
    *,
    position_tolerance_m: float,
    heading_tolerance_rad: float,
) -> dict[str, Any]:
    left_robot = left.frames[0].robot
    right_robot = right.frames[0].robot
    left_pos = _vector2(left_robot.get("position"))
    right_pos = _vector2(right_robot.get("position"))
    if left_pos is None or right_pos is None:
        return {"status": "unavailable", "reason": "missing_initial_robot_position"}
    heading_delta = abs(
        float(left_robot.get("heading", 0.0)) - float(right_robot.get("heading", 0.0))
    )
    position_delta = _distance(left_pos, right_pos)
    equivalent = position_delta <= position_tolerance_m and heading_delta <= heading_tolerance_rad
    return {
        "status": "available",
        "equivalent": bool(equivalent),
        "position_delta_m": position_delta,
        "heading_delta_rad": heading_delta,
        "position_tolerance_m": position_tolerance_m,
        "heading_tolerance_rad": heading_tolerance_rad,
    }


def _shared_prefix(
    left: SimulationTraceExport,
    right: SimulationTraceExport,
    *,
    position_tolerance_m: float,
    max_steps: int,
) -> dict[str, Any]:
    compared = min(len(left.frames), len(right.frames), max(max_steps, 1))
    for index in range(compared):
        left_pos = _vector2(left.frames[index].robot.get("position"))
        right_pos = _vector2(right.frames[index].robot.get("position"))
        if left_pos is None or right_pos is None:
            return {"status": "unavailable", "reason": "missing_prefix_robot_position"}
        if _distance(left_pos, right_pos) > position_tolerance_m:
            return {
                "status": "available",
                "shared_prefix": False,
                "matched_steps": index,
                "required_steps": compared,
                "position_tolerance_m": position_tolerance_m,
            }
    return {
        "status": "available",
        "shared_prefix": True,
        "matched_steps": compared,
        "required_steps": compared,
        "position_tolerance_m": position_tolerance_m,
    }


def _vector2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])
