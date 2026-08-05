"""Pair compatibility records for renderer-neutral worked-example traces."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.process_trace_receipt import (
    build_simulation_trace_receipt,
    simulation_trace_receipt_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from robot_sf.analysis_workbench.simulation_trace_export import SimulationTraceExport

PAIR_COMPATIBILITY_PROFILE_VERSION = "pair_compatibility.deterministic.v1"
PAIR_COMPARISON_GRAINS = frozenset({"matched_planner_pair", "matched_realization_pair"})
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def unavailable_pair_compatibility(
    reason: str = "no_pair_trace_declared",
    *,
    comparison_grain: str | None = None,
) -> dict[str, Any]:
    """Return an explicit unavailable pair-compatibility record."""

    return {
        "profile_version": PAIR_COMPATIBILITY_PROFILE_VERSION,
        "status": "unavailable",
        "reason": reason,
        "comparison_grain": _comparison_grain(comparison_grain),
        "provenance_gate": {"status": "unavailable", "compatible": False, "reason": reason},
        "right_source_trace": {"status": "unavailable", "reason": reason},
        "initial_state_equivalence": {"status": "unavailable", "reason": reason},
        "route_spawn_separation": {"status": "unavailable", "reason": reason},
        "shared_prefix": {"status": "unavailable", "reason": reason},
        "valid_common_event_anchors": [],
        "right_event_anchors": [],
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
    comparison_grain: str,
    position_tolerance_m: float = 1e-6,
    heading_tolerance_rad: float = 1e-6,
    shared_prefix_steps: int = 3,
) -> dict[str, Any]:
    """Build deterministic pair-compatibility diagnostics without normalizing durations.

    Returns:
        JSON-safe pair compatibility record.
    """

    if comparison_grain not in PAIR_COMPARISON_GRAINS:
        return unavailable_pair_compatibility(
            "unsupported_pair_comparison_grain",
            comparison_grain=comparison_grain,
        )
    if not left.frames or not right.frames:
        return unavailable_pair_compatibility("empty_pair_trace", comparison_grain=comparison_grain)
    provenance = _provenance_gate(left, right, comparison_grain=comparison_grain)
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
    requires_initial_equivalence = comparison_grain == "matched_planner_pair"
    compatible = bool(
        provenance["compatible"]
        and (not requires_initial_equivalence or initial.get("equivalent") is True)
    )
    common_event_anchors = _common_event_anchors(left_events, right_events) if compatible else []
    shared_prefix_status = (
        compatible
        and shared_prefix["status"] == "available"
        and bool(shared_prefix["shared_prefix"])
    )
    return {
        "profile_version": PAIR_COMPATIBILITY_PROFILE_VERSION,
        "status": "available" if compatible else "incompatible",
        "comparison_grain": _comparison_grain(comparison_grain),
        "provenance_gate": provenance,
        "right_source_trace": {
            "status": "available",
            "schema_version": "simulation_trace_export.v1",
            "trace_id": right.trace_id,
            "coordinate_frame": right.coordinate_frame,
            "units": right.units,
            "content_sha256": _trace_content_sha256(right),
            "content_receipt": build_simulation_trace_receipt(right),
            "source": {
                "scenario_id": right.source.scenario_id,
                "seed": right.source.seed,
                "planner_id": right.source.planner_id,
                "episode_id": right.source.episode_id,
                "generated_by": right.source.generated_by,
            },
        },
        "initial_state_equivalence": initial,
        "route_spawn_separation": route_spawn,
        "shared_prefix": shared_prefix,
        "valid_common_event_anchors": common_event_anchors,
        "right_event_anchors": _event_receipts(right_events),
        "duration_normalization": {"applied": False},
        "divergence_interpretation": {
            "allowed": shared_prefix_status,
            "reason": (
                "shared_prefix_available"
                if shared_prefix_status
                else "grain_provenance_or_initial_state_incompatible"
                if not compatible
                else "no_shared_prefix_reject_divergence_output"
            ),
        },
    }


def _comparison_grain(comparison_grain: str | None) -> dict[str, Any]:
    required_gate_fields = [
        "scenario_id",
        "coordinate_frame",
        "units",
        "map_id",
        "horizon",
        "time_step_s",
    ]
    if comparison_grain == "matched_planner_pair":
        required_gate_fields.append("initial_state")
    if comparison_grain == "matched_realization_pair":
        required_gate_fields.append("config_digest")
    return {
        "grain_id": comparison_grain or "undeclared",
        "left_role": "primary_trace",
        "right_role": "comparison_trace",
        "required_gate_fields": required_gate_fields,
        "planner_seed_rule": _planner_seed_rule(comparison_grain),
        "divergence_quantity": "per-frame difference curves require shared_prefix true",
        "anchor_alignment": "deterministic_common_event_identity",
    }


def _planner_seed_rule(comparison_grain: str | None) -> str:
    if comparison_grain == "matched_planner_pair":
        return "planner_id_different_seed_equal_initial_state_equal_config_may_differ"
    if comparison_grain == "matched_realization_pair":
        return "planner_id_equal_seed_different_start_spawn_may_differ"
    return "unsupported_grain"


def _provenance_gate(
    left: SimulationTraceExport,
    right: SimulationTraceExport,
    *,
    comparison_grain: str,
) -> dict[str, Any]:
    left_time_step_contract = build_trace_run_config_contract(left)
    right_time_step_contract = build_trace_run_config_contract(right)
    required_meta = {
        "map_id": (_meta(left, "map_id"), _meta(right, "map_id")),
        "horizon": (_meta(left, "horizon"), _meta(right, "horizon")),
        "config_digest": (_meta(left, "config_digest"), _meta(right, "config_digest")),
        "time_step_s": (
            left_time_step_contract.get("time_step_s"),
            right_time_step_contract.get("time_step_s"),
        ),
    }
    availability = {
        key: {
            "left": left_value,
            "right": right_value,
            "status": "available"
            if left_value is not None and right_value is not None
            else "unavailable",
        }
        for key, (left_value, right_value) in required_meta.items()
    }
    checks = {
        "scenario_id_equal": left.source.scenario_id == right.source.scenario_id,
        "coordinate_frame_equal": left.coordinate_frame == right.coordinate_frame,
        "units_equal": left.units == right.units,
        "seed_equal": left.source.seed == right.source.seed,
        "planner_id_equal": left.source.planner_id == right.source.planner_id,
        "planner_id_different": left.source.planner_id != right.source.planner_id,
        "seed_different": left.source.seed != right.source.seed,
        "map_id_present": availability["map_id"]["status"] == "available",
        "horizon_present": availability["horizon"]["status"] == "available",
        "config_digest_present": availability["config_digest"]["status"] == "available",
        "time_step_s_present": bool(
            left_time_step_contract.get("status") == "available"
            and right_time_step_contract.get("status") == "available"
        ),
        "map_id_equal": _required_equal(*required_meta["map_id"]),
        "horizon_equal": _required_equal(*required_meta["horizon"]),
        "config_digest_equal": _required_equal(*required_meta["config_digest"]),
        "time_step_s_equal": bool(
            left_time_step_contract.get("status") == "available"
            and right_time_step_contract.get("status") == "available"
            and _required_equal(*required_meta["time_step_s"])
        ),
    }
    if comparison_grain == "matched_planner_pair":
        grain_specific = checks["planner_id_different"] and checks["seed_equal"]
        config_compatible = True
    elif comparison_grain == "matched_realization_pair":
        grain_specific = checks["planner_id_equal"] and checks["seed_different"]
        config_compatible = checks["config_digest_equal"]
    else:
        grain_specific = False
        config_compatible = False
    return {
        "status": "available",
        "compatible": bool(
            checks["scenario_id_equal"]
            and checks["coordinate_frame_equal"]
            and checks["units_equal"]
            and checks["map_id_equal"]
            and checks["horizon_equal"]
            and checks["time_step_s_equal"]
            and config_compatible
            and grain_specific
        ),
        "checks": checks,
        "availability": availability,
        "time_step_contracts": {
            "left": left_time_step_contract,
            "right": right_time_step_contract,
        },
        "comparison_grain": comparison_grain,
        "left_trace_id": left.trace_id,
        "right_trace_id": right.trace_id,
        "left_content_sha256": _trace_content_sha256(left),
        "right_content_sha256": _trace_content_sha256(right),
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
    left_heading = _finite_float(left_robot.get("heading"))
    right_heading = _finite_float(right_robot.get("heading"))
    left_actors = _actor_state(left.frames[0].pedestrians)
    right_actors = _actor_state(right.frames[0].pedestrians)
    if left_pos is None or right_pos is None or left_vel is None or right_vel is None:
        return {"status": "unavailable", "reason": "missing_initial_robot_pose_or_velocity"}
    if left_heading is None or right_heading is None:
        return {"status": "unavailable", "reason": "missing_initial_robot_heading"}
    if left_actors is None or right_actors is None:
        return {"status": "unavailable", "reason": "missing_initial_actor_state"}
    heading_delta = abs(
        _wrapped_angle_delta(
            left_heading,
            right_heading,
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
    actor_positions_equivalent = not actor_position_deltas or (
        max_actor_position_delta is not None and max_actor_position_delta <= position_tolerance_m
    )
    actor_velocities_equivalent = not actor_velocity_deltas or (
        max_actor_velocity_delta is not None and max_actor_velocity_delta <= position_tolerance_m
    )
    actor_radii_equivalent = not actor_radius_deltas or max_actor_radius_delta == 0.0
    equivalent = (
        position_delta <= position_tolerance_m
        and velocity_delta <= position_tolerance_m
        and heading_delta <= heading_tolerance_rad
        and robot_radius_delta == 0.0
        and actor_ids_equal
        and actor_positions_equivalent
        and actor_velocities_equivalent
        and actor_radii_equivalent
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
    left_heading = _finite_float(left_robot.get("heading"))
    right_heading = _finite_float(right_robot.get("heading"))
    left_actors = _actor_state(left_frame.pedestrians)
    right_actors = _actor_state(right_frame.pedestrians)
    if left_pos is None or right_pos is None or left_vel is None or right_vel is None:
        return False, "missing_robot_state"
    if left_actors is None or right_actors is None:
        return False, "missing_actor_state"
    if left_heading is None or right_heading is None:
        return False, "missing_robot_heading"
    if _distance(left_pos, right_pos) > position_tolerance_m:
        return False, "robot_position_diverged"
    if _distance(left_vel, right_vel) > position_tolerance_m:
        return False, "robot_velocity_diverged"
    if (
        abs(
            _wrapped_angle_delta(
                left_heading,
                right_heading,
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
    if _finite_number(left) and _finite_number(right):
        return abs(float(left) - float(right))
    return None


def _required_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    return left == right


def _meta(trace: SimulationTraceExport, key: str) -> object:
    values: list[object] = []
    for frame in trace.frames:
        value = frame.planner.get(key)
        if value is None:
            run_config = frame.planner.get("run_config")
            value = run_config.get(key) if isinstance(run_config, dict) else None
        if value is None:
            return None
        values.append(value)
    if not values or any(value != values[0] for value in values[1:]):
        return None
    return values[0]


def build_trace_run_config_contract(trace: SimulationTraceExport) -> dict[str, Any]:
    """Replay the declared run configuration against every source frame and sample interval.

    Returns:
        Available declared/observed time-step receipt, or a fail-closed reason.
    """

    run_configs = [frame.planner.get("run_config") for frame in trace.frames]
    if not run_configs or any(not isinstance(run_config, dict) for run_config in run_configs):
        return {"status": "unavailable", "reason": "run_config_unavailable"}
    time_steps = [run_config.get("time_step_s") for run_config in run_configs]
    if any(
        isinstance(time_step, bool) or not (_finite_number(time_step) and float(time_step) > 0.0)
        for time_step in time_steps
    ):
        return {"status": "unavailable", "reason": "run_config_time_step_unavailable"}
    if len({float(time_step) for time_step in time_steps}) != 1:
        return {"status": "unavailable", "reason": "run_config_time_step_inconsistent"}
    digests = [run_config.get("config_digest") for run_config in run_configs]
    if any(not (isinstance(digest, str) and SHA256_HEX_RE.fullmatch(digest)) for digest in digests):
        return {"status": "unavailable", "reason": "run_config_digest_unavailable"}
    if len(set(digests)) != 1:
        return {"status": "unavailable", "reason": "run_config_digest_inconsistent"}
    declared_time_step = float(time_steps[0])
    configured = {
        "time_step_s": declared_time_step,
        "config_digest": str(digests[0]),
        "source": "planner.run_config",
    }
    observed_time_step, observed_consistent = _observed_trace_time_step(trace)
    if not observed_consistent:
        return {
            "status": "unavailable",
            "reason": "source_time_step_inconsistent",
            **configured,
        }
    if observed_time_step is not None and not math.isclose(
        declared_time_step,
        observed_time_step,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return {
            "status": "unavailable",
            "reason": "run_config_time_step_trace_mismatch",
            **configured,
            "observed_time_step_s": observed_time_step,
        }
    return {"status": "available", **configured}


def _observed_trace_time_step(
    trace: SimulationTraceExport,
) -> tuple[float | None, bool]:
    if len(trace.frames) < 2:
        return None, True
    observed: list[float] = []
    for left, right in zip(trace.frames, trace.frames[1:], strict=False):
        step_delta = right.step - left.step
        time_delta = float(right.time_s) - float(left.time_s)
        if step_delta <= 0 or not math.isfinite(time_delta) or time_delta <= 0.0:
            return None, False
        observed.append(time_delta / step_delta)
    first = observed[0]
    return first, all(
        math.isclose(value, first, rel_tol=1e-9, abs_tol=1e-12) for value in observed[1:]
    )


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _finite_float(value: object) -> float | None:
    return float(value) if _finite_number(value) else None


def _trace_content_sha256(trace: SimulationTraceExport) -> str:
    receipt = build_simulation_trace_receipt(trace)
    return simulation_trace_receipt_sha256(receipt)


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
        match = {
            "event_type": identity[0],
            "detector_profile_version": identity[1],
            "actor_id": identity[2],
            "zone_id": identity[3],
            "source_fields": list(identity[4]),
            "status": "available",
            "left_event_id": left["event_id"],
            "right_event_id": right["event_id"],
        }
        if identity[0] == "exact_collision_event":
            match["collision_partner_type"] = identity[5]
            match["collision_partner_id"] = identity[6]
        matches.append(match)
    return sorted(
        matches,
        key=lambda item: (
            item["event_type"],
            item["actor_id"] or "",
            item["zone_id"] or "",
            item["left_event_id"],
        ),
    )


def _event_receipts(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    receipts = []
    for event in events:
        if event.get("status") != "available":
            continue
        identity = _event_identity(event)
        receipt = {
            "event_id": event["event_id"],
            "event_type": identity[0],
            "detector_profile_version": identity[1],
            "time_s": float(event["time_s"]),
            "step": int(event["step"]),
            "confidence": str(event["confidence"]),
            "actor_id": identity[2],
            "zone_id": identity[3],
            "source_fields": list(identity[4]),
            "status": "available",
            "event_relative_time": dict(event["event_relative_time"]),
            "visual_anchor_eligibility": dict(event["visual_anchor_eligibility"]),
        }
        if identity[0] == "exact_collision_event":
            receipt["collision_partner_type"] = identity[5]
            receipt["collision_partner_id"] = identity[6]
        receipts.append(receipt)
    return sorted(receipts, key=lambda item: str(item["event_id"]))


def _event_identity(
    event: Mapping[str, Any],
) -> tuple[str, str, object, object, tuple[str, ...], object, object]:
    return (
        str(event.get("event_type")),
        str(event.get("detector_profile_version")),
        event.get("actor_id"),
        event.get("zone_id"),
        tuple(str(field) for field in event.get("source_fields", [])),
        event.get("collision_partner_type"),
        event.get("collision_partner_id"),
    )
