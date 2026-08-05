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
        "duration_normalization": {"applied": False},
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
    provenance = _provenance_gate(left, right)
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
        "scenario_provenance_compatible": provenance["compatible"],
        "scenario_id_equal": left.source.scenario_id == right.source.scenario_id,
        "planner_id_left": left.source.planner_id,
        "planner_id_right": right.source.planner_id,
        "seed_left": left.source.seed,
        "seed_right": right.source.seed,
        "initial_robot_separation_m": initial.get("robot_position_delta_m"),
        "max_initial_actor_separation_m": initial.get("max_actor_position_delta_m"),
    }
    common_event_types = sorted(
        {str(event["event_type"]) for event in left_events if event.get("status") == "available"}
        & {str(event["event_type"]) for event in right_events if event.get("status") == "available"}
    )
    compatible = bool(provenance["compatible"] and initial.get("equivalent"))
    shared_prefix_status = (
        compatible
        and shared_prefix["status"] == "available"
        and bool(shared_prefix["shared_prefix"])
    )
    return {
        "profile_version": PAIR_COMPATIBILITY_PROFILE_VERSION,
        "status": "available" if compatible else "incompatible",
        "provenance_gate": provenance,
        "initial_state_equivalence": initial,
        "route_spawn_separation": route_spawn,
        "shared_prefix": shared_prefix,
        "valid_common_event_anchors": common_event_types if compatible else [],
        "duration_normalization": {"applied": False},
        "divergence_interpretation": {
            "allowed": shared_prefix_status,
            "reason": (
                "shared_prefix_available"
                if shared_prefix_status
                else "scenario_or_initial_state_incompatible"
                if not compatible
                else "no_shared_prefix_reject_divergence_output"
            ),
        },
    }


def _provenance_gate(left: SimulationTraceExport, right: SimulationTraceExport) -> dict[str, Any]:
    checks = {
        "scenario_id_equal": left.source.scenario_id == right.source.scenario_id,
        "coordinate_frame_equal": left.coordinate_frame == right.coordinate_frame,
        "units_equal": left.units == right.units,
    }
    return {
        "status": "available",
        "compatible": all(checks.values()),
        "checks": checks,
        "left_trace_id": left.trace_id,
        "right_trace_id": right.trace_id,
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
    left_vel = _vector2(left_robot.get("velocity"))
    right_vel = _vector2(right_robot.get("velocity"))
    left_actors = _actor_state(left.frames[0].pedestrians)
    right_actors = _actor_state(right.frames[0].pedestrians)
    if left_pos is None or right_pos is None or left_vel is None or right_vel is None:
        return {"status": "unavailable", "reason": "missing_initial_robot_pose_or_velocity"}
    if left_actors is None or right_actors is None:
        return {"status": "unavailable", "reason": "missing_initial_actor_state"}
    heading_delta = abs(
        float(left_robot.get("heading", 0.0)) - float(right_robot.get("heading", 0.0))
    )
    position_delta = _distance(left_pos, right_pos)
    velocity_delta = _distance(left_vel, right_vel)
    actor_ids_equal = set(left_actors) == set(right_actors)
    actor_position_deltas = {
        actor_id: _distance(left_actors[actor_id]["position"], right_actors[actor_id]["position"])
        for actor_id in sorted(set(left_actors) & set(right_actors))
    }
    actor_velocity_deltas = {
        actor_id: _distance(left_actors[actor_id]["velocity"], right_actors[actor_id]["velocity"])
        for actor_id in sorted(set(left_actors) & set(right_actors))
    }
    max_actor_position_delta = max(actor_position_deltas.values(), default=None)
    max_actor_velocity_delta = max(actor_velocity_deltas.values(), default=None)
    equivalent = (
        position_delta <= position_tolerance_m
        and velocity_delta <= position_tolerance_m
        and heading_delta <= heading_tolerance_rad
        and actor_ids_equal
        and (
            max_actor_position_delta is not None
            and max_actor_position_delta <= position_tolerance_m
        )
        and (
            max_actor_velocity_delta is not None
            and max_actor_velocity_delta <= position_tolerance_m
        )
    )
    return {
        "status": "available",
        "equivalent": bool(equivalent),
        "robot_position_delta_m": position_delta,
        "robot_velocity_delta_mps": velocity_delta,
        "heading_delta_rad": heading_delta,
        "actor_id_sets_equal": actor_ids_equal,
        "actor_position_delta_m": actor_position_deltas,
        "actor_velocity_delta_mps": actor_velocity_deltas,
        "max_actor_position_delta_m": max_actor_position_delta,
        "max_actor_velocity_delta_mps": max_actor_velocity_delta,
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
        left_vel = _vector2(left.frames[index].robot.get("velocity"))
        right_vel = _vector2(right.frames[index].robot.get("velocity"))
        left_actors = _actor_state(left.frames[index].pedestrians)
        right_actors = _actor_state(right.frames[index].pedestrians)
        if (
            left_pos is None
            or right_pos is None
            or left_vel is None
            or right_vel is None
            or left_actors is None
            or right_actors is None
        ):
            return {"status": "unavailable", "reason": "missing_prefix_full_state"}
        equal, reason = _full_state_equal(
            left_pos,
            right_pos,
            left_vel,
            right_vel,
            left_actors,
            right_actors,
            tolerance_m=position_tolerance_m,
        )
        if not equal:
            return {
                "status": "available",
                "shared_prefix": False,
                "matched_steps": index,
                "required_steps": compared,
                "position_tolerance_m": position_tolerance_m,
                "reason": reason,
            }
    return {
        "status": "available",
        "shared_prefix": True,
        "matched_steps": compared,
        "required_steps": compared,
        "position_tolerance_m": position_tolerance_m,
    }


def _full_state_equal(
    left_pos: tuple[float, float],
    right_pos: tuple[float, float],
    left_vel: tuple[float, float],
    right_vel: tuple[float, float],
    left_actors: dict[str, dict[str, tuple[float, float]]],
    right_actors: dict[str, dict[str, tuple[float, float]]],
    *,
    tolerance_m: float,
) -> tuple[bool, str]:
    if _distance(left_pos, right_pos) > tolerance_m:
        return False, "robot_position_diverged"
    if _distance(left_vel, right_vel) > tolerance_m:
        return False, "robot_velocity_diverged"
    if set(left_actors) != set(right_actors):
        return False, "actor_id_set_diverged"
    for actor_id in sorted(left_actors):
        if (
            _distance(left_actors[actor_id]["position"], right_actors[actor_id]["position"])
            > tolerance_m
        ):
            return False, f"actor_position_diverged:{actor_id}"
        if (
            _distance(left_actors[actor_id]["velocity"], right_actors[actor_id]["velocity"])
            > tolerance_m
        ):
            return False, f"actor_velocity_diverged:{actor_id}"
    return True, "full_state_prefix_matches"


def _actor_state(
    pedestrians: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[float, float]]] | None:
    state: dict[str, dict[str, tuple[float, float]]] = {}
    for pedestrian in pedestrians:
        actor_id = pedestrian.get("id")
        position = _vector2(pedestrian.get("position"))
        velocity = _vector2(pedestrian.get("velocity"))
        if actor_id is None or position is None or velocity is None:
            return None
        state[str(actor_id)] = {"position": position, "velocity": velocity}
    return state


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
