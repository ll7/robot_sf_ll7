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
        "comparison_grain": _comparison_grain(),
        "provenance_gate": {"status": "unavailable", "compatible": False, "reason": reason},
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
        heading_tolerance_rad=heading_tolerance_rad,
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
    compatible = bool(provenance["compatible"])
    common_event_anchors = _common_event_anchors(left_events, right_events) if compatible else []
    shared_prefix_status = (
        compatible
        and shared_prefix["status"] == "available"
        and bool(shared_prefix["shared_prefix"])
    )
    return {
        "profile_version": PAIR_COMPATIBILITY_PROFILE_VERSION,
        "status": "available" if compatible else "incompatible",
        "comparison_grain": _comparison_grain(),
        "provenance_gate": provenance,
        "initial_state_equivalence": initial,
        "route_spawn_separation": route_spawn,
        "shared_prefix": shared_prefix,
        "valid_common_event_anchors": common_event_anchors,
        "duration_normalization": {"applied": False},
        "divergence_interpretation": {
            "allowed": shared_prefix_status,
            "reason": (
                "shared_prefix_available"
                if shared_prefix_status
                else "scenario_or_configuration_incompatible"
                if not compatible
                else "no_shared_prefix_reject_divergence_output"
            ),
        },
    }


def _comparison_grain() -> dict[str, Any]:
    return {
        "grain_id": "matched_planner_realization.v1",
        "left_role": "primary_trace",
        "right_role": "comparison_trace",
        "required_gate_fields": ["scenario_id", "coordinate_frame", "units"],
        "conditional_gate_fields": ["map_id", "horizon", "config_digest"],
        "seed_handling": "reported_not_required_for_sensitivity_pairs",
        "divergence_quantity": "per-frame difference curves require shared_prefix true",
        "anchor_alignment": "deterministic_common_event_identity",
    }


def _provenance_gate(left: SimulationTraceExport, right: SimulationTraceExport) -> dict[str, Any]:
    optional = {
        "map_id": (_meta(left, "map_id"), _meta(right, "map_id")),
        "horizon": (_meta(left, "horizon"), _meta(right, "horizon")),
        "config_digest": (_meta(left, "config_digest"), _meta(right, "config_digest")),
    }
    checks = {
        "scenario_id_equal": left.source.scenario_id == right.source.scenario_id,
        "coordinate_frame_equal": left.coordinate_frame == right.coordinate_frame,
        "units_equal": left.units == right.units,
        "seed_equal": left.source.seed == right.source.seed,
        "map_id_equal": _optional_equal(*optional["map_id"]),
        "horizon_equal": _optional_equal(*optional["horizon"]),
        "config_digest_equal": _optional_equal(*optional["config_digest"]),
    }
    return {
        "status": "available",
        "compatible": bool(
            checks["scenario_id_equal"]
            and checks["coordinate_frame_equal"]
            and checks["units_equal"]
            and checks["map_id_equal"]
            and checks["horizon_equal"]
            and checks["config_digest_equal"]
        ),
        "checks": checks,
        "availability": {
            key: {
                "left": left_value,
                "right": right_value,
                "status": "available"
                if left_value is not None and right_value is not None
                else "unavailable",
            }
            for key, (left_value, right_value) in optional.items()
        },
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
        _wrapped_angle_delta(
            float(left_robot.get("heading", 0.0)),
            float(right_robot.get("heading", 0.0)),
        )
    )
    position_delta = _distance(left_pos, right_pos)
    velocity_delta = _distance(left_vel, right_vel)
    robot_radius_delta = _nullable_delta(left_robot.get("radius"), right_robot.get("radius"))
    actor_ids_equal = set(left_actors) == set(right_actors)
    actor_position_deltas = {
        actor_id: _distance(left_actors[actor_id]["position"], right_actors[actor_id]["position"])
        for actor_id in sorted(set(left_actors) & set(right_actors))
    }
    actor_velocity_deltas = {
        actor_id: _distance(left_actors[actor_id]["velocity"], right_actors[actor_id]["velocity"])
        for actor_id in sorted(set(left_actors) & set(right_actors))
    }
    actor_radius_deltas = {
        actor_id: _nullable_delta(left_actors[actor_id]["radius"], right_actors[actor_id]["radius"])
        for actor_id in sorted(set(left_actors) & set(right_actors))
    }
    max_actor_position_delta = max(actor_position_deltas.values(), default=None)
    max_actor_velocity_delta = max(actor_velocity_deltas.values(), default=None)
    max_actor_radius_delta = max(
        (delta for delta in actor_radius_deltas.values() if delta is not None),
        default=None,
    )
    equivalent = (
        position_delta <= position_tolerance_m
        and velocity_delta <= position_tolerance_m
        and heading_delta <= heading_tolerance_rad
        and robot_radius_delta == 0.0
        and actor_ids_equal
        and (
            max_actor_position_delta is not None
            and max_actor_position_delta <= position_tolerance_m
        )
        and (
            max_actor_velocity_delta is not None
            and max_actor_velocity_delta <= position_tolerance_m
        )
        and max_actor_radius_delta == 0.0
    )
    return {
        "status": "available",
        "equivalent": bool(equivalent),
        "robot_position_delta_m": position_delta,
        "robot_velocity_delta_mps": velocity_delta,
        "robot_heading_delta_rad": heading_delta,
        "robot_radius_delta_m": robot_radius_delta,
        "actor_id_sets_equal": actor_ids_equal,
        "actor_position_delta_m": actor_position_deltas,
        "actor_velocity_delta_mps": actor_velocity_deltas,
        "actor_radius_delta_m": actor_radius_deltas,
        "max_actor_position_delta_m": max_actor_position_delta,
        "max_actor_velocity_delta_mps": max_actor_velocity_delta,
        "max_actor_radius_delta_m": max_actor_radius_delta,
        "position_tolerance_m": position_tolerance_m,
        "heading_tolerance_rad": heading_tolerance_rad,
    }


def _shared_prefix(
    left: SimulationTraceExport,
    right: SimulationTraceExport,
    *,
    position_tolerance_m: float,
    heading_tolerance_rad: float,
    max_steps: int,
) -> dict[str, Any]:
    compared = min(len(left.frames), len(right.frames), max(max_steps, 1))
    for index in range(compared):
        equal, reason = _full_state_equal(
            left.frames[index],
            right.frames[index],
            position_tolerance_m=position_tolerance_m,
            heading_tolerance_rad=heading_tolerance_rad,
        )
        if not equal:
            return {
                "status": "available",
                "shared_prefix": False,
                "matched_steps": index,
                "required_steps": compared,
                "position_tolerance_m": position_tolerance_m,
                "heading_tolerance_rad": heading_tolerance_rad,
                "reason": reason,
            }
    return {
        "status": "available",
        "shared_prefix": True,
        "matched_steps": compared,
        "required_steps": compared,
        "position_tolerance_m": position_tolerance_m,
        "heading_tolerance_rad": heading_tolerance_rad,
    }


def _full_state_equal(  # noqa: C901
    left_frame: Any,
    right_frame: Any,
    *,
    position_tolerance_m: float,
    heading_tolerance_rad: float,
) -> tuple[bool, str]:
    if left_frame.step != right_frame.step:
        return False, "step_diverged"
    if float(left_frame.time_s) != float(right_frame.time_s):
        return False, "time_diverged"
    left_robot = left_frame.robot
    right_robot = right_frame.robot
    left_pos = _vector2(left_robot.get("position"))
    right_pos = _vector2(right_robot.get("position"))
    left_vel = _vector2(left_robot.get("velocity"))
    right_vel = _vector2(right_robot.get("velocity"))
    left_actors = _actor_state(left_frame.pedestrians)
    right_actors = _actor_state(right_frame.pedestrians)
    if left_pos is None or right_pos is None or left_vel is None or right_vel is None:
        return False, "missing_robot_state"
    if left_actors is None or right_actors is None:
        return False, "missing_actor_state"
    if _distance(left_pos, right_pos) > position_tolerance_m:
        return False, "robot_position_diverged"
    if _distance(left_vel, right_vel) > position_tolerance_m:
        return False, "robot_velocity_diverged"
    if (
        abs(
            _wrapped_angle_delta(
                float(left_robot.get("heading", 0.0)),
                float(right_robot.get("heading", 0.0)),
            )
        )
        > heading_tolerance_rad
    ):
        return False, "robot_heading_diverged"
    if _nullable_delta(left_robot.get("radius"), right_robot.get("radius")) != 0.0:
        return False, "robot_radius_diverged"
    if set(left_actors) != set(right_actors):
        return False, "actor_id_set_diverged"
    for actor_id in sorted(left_actors):
        if (
            _distance(left_actors[actor_id]["position"], right_actors[actor_id]["position"])
            > position_tolerance_m
        ):
            return False, "actor_position_diverged"
        if (
            _distance(left_actors[actor_id]["velocity"], right_actors[actor_id]["velocity"])
            > position_tolerance_m
        ):
            return False, "actor_velocity_diverged"
        if (
            _nullable_delta(left_actors[actor_id]["radius"], right_actors[actor_id]["radius"])
            != 0.0
        ):
            return False, "actor_radius_diverged"
    return True, "shared_prefix_equal"


def _actor_state(pedestrians: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]] | None:
    result: dict[str, dict[str, Any]] = {}
    for pedestrian in pedestrians:
        if "id" not in pedestrian:
            return None
        pos = _vector2(pedestrian.get("position"))
        vel = _vector2(pedestrian.get("velocity"))
        if pos is None or vel is None:
            return None
        result[str(pedestrian["id"])] = {
            "position": pos,
            "velocity": vel,
            "radius": pedestrian.get("radius"),
        }
    return result


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


def _wrapped_angle_delta(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _nullable_delta(left: object, right: object) -> float | None:
    if left is None and right is None:
        return 0.0
    if isinstance(left, int | float) and isinstance(right, int | float):
        return abs(float(left) - float(right))
    return None


def _optional_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return True
    return left == right


def _meta(trace: SimulationTraceExport, key: str) -> object:
    for frame in trace.frames:
        value = frame.planner.get(key)
        if value is not None:
            return value
        run_config = frame.planner.get("run_config")
        if isinstance(run_config, dict) and key in run_config:
            return run_config[key]
    return None


def _common_event_anchors(
    left_events: Sequence[Mapping[str, Any]],
    right_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    right_by_identity = {
        _event_identity(event): event
        for event in right_events
        if event.get("status") == "available"
    }
    matches: list[dict[str, Any]] = []
    for left in left_events:
        if left.get("status") != "available":
            continue
        identity = _event_identity(left)
        right = right_by_identity.get(identity)
        if right is None:
            continue
        matches.append(
            {
                "event_type": identity[0],
                "detector_profile_version": identity[1],
                "actor_id": identity[2],
                "zone_id": identity[3],
                "source_fields": list(identity[4]),
                "status": "available",
                "left_event_id": left["event_id"],
                "right_event_id": right["event_id"],
            }
        )
    return sorted(
        matches,
        key=lambda item: (
            item["event_type"],
            item["actor_id"] or "",
            item["zone_id"] or "",
            item["left_event_id"],
        ),
    )


def _event_identity(event: Mapping[str, Any]) -> tuple[str, str, object, object, tuple[str, ...]]:
    return (
        str(event.get("event_type")),
        str(event.get("detector_profile_version")),
        event.get("actor_id"),
        event.get("zone_id"),
        tuple(str(field) for field in event.get("source_fields", [])),
    )
