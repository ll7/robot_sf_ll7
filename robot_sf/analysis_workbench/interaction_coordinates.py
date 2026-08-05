"""Build ``worked_example_process_trace.v1`` diagnostics from admitted trace exports."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.episode_phases import (
    PHASE_PROFILE_VERSION,
    REVERSAL_PROFILE_VERSION,
    duration_where,
    first_recovery_frame,
    first_sustained_stall_frame,
    summarize_reversals,
    summarize_stall,
)
from robot_sf.analysis_workbench.event_alignment import (
    build_pair_compatibility_record,
    unavailable_pair_compatibility,
)
from robot_sf.analysis_workbench.safety_surrogates import (
    SAFETY_SURROGATE_PROFILE_VERSION,
    constant_velocity_closest_approach,
    proxy_surface_clearance_m,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExport,
    SimulationTraceFrame,
    load_simulation_trace_export,
)
from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION = "worked_example_process_trace.v1"
WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "worked_example_process_trace.v1.json"
)
EVENT_PROFILE_VERSION = "worked_example_event_detectors.v1"
THRESHOLD_PROFILE_VERSION = "worked_example_threshold_profile.diagnostic.v1"


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Registered straight route used for route-frame diagnostics."""

    route_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    provenance_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConflictZoneSpec:
    """Registered circular conflict zone used for conflict-frame diagnostics."""

    zone_id: str
    center: tuple[float, float]
    radius_m: float
    provenance_id: str | None = None


class WorkedExampleProcessTraceValidationError(RobotSfError, ValueError):
    """Raised when a process trace fails JSON Schema validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_worked_example_process_trace_schema() -> dict[str, Any]:
    """Load the public ``worked_example_process_trace.v1`` JSON schema.

    Returns:
        Parsed JSON Schema document.
    """

    return json.loads(WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_worked_example_process_trace(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> None:
    """Validate a process trace payload against its versioned schema."""

    validator = Draft202012Validator(load_worked_example_process_trace_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda err: list(err.absolute_path),
        )
    ]
    if errors:
        raise WorkedExampleProcessTraceValidationError(errors, source=source)


def build_worked_example_process_trace(
    input_path: Path,
    *,
    route: RouteSpec | None = None,
    conflict_zone: ConflictZoneSpec | None = None,
    focal_actor_id: str | None = None,
    pair_input_path: Path | None = None,
) -> dict[str, Any]:
    """Build a renderer-neutral process trace from one admitted trace export.

    Returns:
        Schema-valid process trace payload.
    """

    trace = load_simulation_trace_export(input_path)
    pair_trace = load_simulation_trace_export(pair_input_path) if pair_input_path else None
    payload = build_worked_example_process_trace_from_export(
        trace,
        route=route,
        conflict_zone=conflict_zone,
        focal_actor_id=focal_actor_id,
        pair_trace=pair_trace,
    )
    validate_worked_example_process_trace(payload, source=input_path)
    return payload


def build_worked_example_process_trace_from_export(
    trace: SimulationTraceExport,
    *,
    route: RouteSpec | None = None,
    conflict_zone: ConflictZoneSpec | None = None,
    focal_actor_id: str | None = None,
    pair_trace: SimulationTraceExport | None = None,
) -> dict[str, Any]:
    """Build a schema-valid process trace from a typed trace export.

    Returns:
        Schema-valid process trace payload.
    """

    focal = _resolve_focal_actor(trace, requested_actor_id=focal_actor_id)
    route_availability = _route_availability(route)
    conflict_availability = _conflict_availability(conflict_zone)
    relative_availability = _relative_availability(focal)
    world_availability = _world_availability(trace)
    frames = [
        _process_frame(
            frame,
            frame_index=index,
            focal_actor_id=focal.get("actor_id"),
            route=route,
            conflict_zone=conflict_zone,
            source_coordinate_frame=trace.coordinate_frame,
        )
        for index, frame in enumerate(trace.frames)
    ]
    focal = dict(focal)
    focal["actor_contiguity"] = _actor_contiguity(
        frames,
        focal.get("actor_id"),
        declared=focal.get("declared_encounter"),
    )
    events = _event_anchors(trace, frames=frames, focal_actor_id=focal.get("actor_id"))
    pair = (
        build_pair_compatibility_record(
            trace,
            pair_trace,
            left_events=events,
            right_events=_event_anchors(
                pair_trace,
                frames=[
                    _process_frame(
                        frame,
                        frame_index=index,
                        focal_actor_id=_resolve_focal_actor(pair_trace).get("actor_id"),
                        route=route,
                        conflict_zone=conflict_zone,
                        source_coordinate_frame=pair_trace.coordinate_frame,
                    )
                    for index, frame in enumerate(pair_trace.frames)
                ],
                focal_actor_id=_resolve_focal_actor(pair_trace).get("actor_id"),
            ),
        )
        if pair_trace is not None
        else unavailable_pair_compatibility()
    )
    payload: dict[str, Any] = {
        "schema_version": WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION,
        "process_trace_id": f"{trace.trace_id}-process-trace",
        "source_trace": _source_trace(trace),
        "evidence_boundary": "analysis_workbench_only",
        "source_coordinate_frame": trace.coordinate_frame,
        "units": trace.units,
        "claim_boundary": (
            "Diagnostic renderer-neutral process quantities derived from admitted trace fields. "
            "Not calibrated AMMV safety thresholds, collision probabilities, causal attribution, "
            "or replacement benchmark metrics."
        ),
        "profiles": _profiles(),
        "coordinate_frames": {
            "world": world_availability,
            "route": route_availability,
            "conflict": conflict_availability,
            "relative_interaction": relative_availability,
        },
        "encounters": {
            "focal": focal,
            "global_minimum_over_all_actors": _global_minimum_series(frames),
            "actor_switch_events": _actor_switch_events(frames),
        },
        "frames": frames,
        "diagnostics": _diagnostics(frames, route_available=route is not None),
        "event_anchors": events,
        "pair_compatibility": pair,
    }
    validate_worked_example_process_trace(payload)
    return payload


def write_worked_example_process_trace(input_path: Path, output_path: Path, **kwargs: Any) -> Path:
    """Write a deterministic process trace JSON file.

    Returns:
        Output path that received the process trace.
    """

    payload = build_worked_example_process_trace(input_path, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _source_trace(trace: SimulationTraceExport) -> dict[str, Any]:
    return {
        "schema_version": SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
        "trace_id": trace.trace_id,
        "coordinate_frame": trace.coordinate_frame,
        "units": trace.units,
        "source": {
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
            "generated_by": trace.source.generated_by,
        },
    }


def _profiles() -> dict[str, Any]:
    return {
        "event_anchor_profile": {
            "profile_version": EVENT_PROFILE_VERSION,
            "material_deceleration_drop_mps": 0.2,
            "material_turn_response_rad_s": 0.25,
        },
        "threshold_profile": {
            "profile_version": THRESHOLD_PROFILE_VERSION,
            "proxy_surface_clearance_threshold_m": 0.4,
            "radius_policy": "declared_proxy_radii_required",
        },
        "phase_profile": {
            "profile_version": PHASE_PROFILE_VERSION,
            "stall_speed_threshold_mps": 0.05,
            "stall_min_duration_s": 0.2,
            "recovery_speed_threshold_mps": 0.1,
        },
        "reversal_profile": {
            "profile_version": REVERSAL_PROFILE_VERSION,
            "heading_delta_threshold_rad": math.pi / 2.0,
            "velocity_projection_sign_epsilon_mps": 1e-6,
        },
        "safety_surrogate_profile": {"profile_version": SAFETY_SURROGATE_PROFILE_VERSION},
    }


def _route_availability(route: RouteSpec | None) -> dict[str, Any]:
    if route is None:
        return {"status": "unavailable", "reason": "registered_route_unavailable"}
    if not route.provenance_id:
        return {"status": "unavailable", "reason": "registered_route_provenance_unavailable"}
    if _distance(route.start, route.end) <= 1e-12:
        return {"status": "unavailable", "reason": "registered_route_degenerate"}
    return {
        "status": "available",
        "reason": "registered_straight_route",
        "route_id": route.route_id,
        "provenance_id": route.provenance_id,
    }


def _conflict_availability(conflict_zone: ConflictZoneSpec | None) -> dict[str, Any]:
    if conflict_zone is None:
        return {"status": "unavailable", "reason": "registered_conflict_zone_unavailable"}
    if not conflict_zone.provenance_id:
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_provenance_unavailable",
        }
    if not math.isfinite(conflict_zone.radius_m) or conflict_zone.radius_m < 0:
        return {"status": "unavailable", "reason": "registered_conflict_zone_invalid"}
    return {
        "status": "available",
        "reason": "registered_circular_conflict_zone",
        "zone_id": conflict_zone.zone_id,
        "provenance_id": conflict_zone.provenance_id,
    }


def _world_availability(trace: SimulationTraceExport) -> dict[str, Any]:
    if trace.coordinate_frame != "world":
        return {
            "status": "unavailable",
            "reason": "source_coordinate_frame_not_world",
            "source_coordinate_frame": trace.coordinate_frame,
        }
    return {
        "status": "available",
        "reason": "source_trace_world_frame",
        "source_coordinate_frame": trace.coordinate_frame,
    }


def _relative_availability(focal: Mapping[str, Any]) -> dict[str, Any]:
    if focal.get("status") == "available":
        return {
            "status": "available",
            "reason": str(focal.get("source", "focal_actor_resolved")),
            "actor_id": str(focal["actor_id"]),
            "encounter_id": focal.get("encounter_id"),
        }
    return {"status": "unavailable", "reason": str(focal.get("reason", "no_focal_actor"))}


def _resolve_focal_actor(
    trace: SimulationTraceExport,
    *,
    requested_actor_id: str | None = None,
) -> dict[str, Any]:
    declared = _declared_encounter(trace)
    actor_ids = sorted(
        {
            str(pedestrian["id"])
            for frame in trace.frames
            for pedestrian in frame.pedestrians
            if "id" in pedestrian
        }
    )
    if requested_actor_id:
        requested = str(requested_actor_id)
        if requested not in actor_ids:
            return {
                "status": "unavailable",
                "reason": "requested_focal_actor_missing",
                "requested_actor_id": requested,
                "declared_encounter": declared,
            }
        if declared.get("actor_id") and str(declared["actor_id"]) != requested:
            return {
                "status": "unavailable",
                "reason": "requested_focal_actor_conflicts_with_declared_encounter",
                "requested_actor_id": requested,
                "declared_encounter": declared,
            }
        return {
            "status": "available",
            "source": "requested_actor_id",
            "actor_id": requested,
            "encounter_id": declared.get("encounter_id"),
            "declared_encounter": declared,
        }
    if declared.get("actor_id"):
        if str(declared["actor_id"]) not in actor_ids:
            return {
                "status": "unavailable",
                "reason": "declared_encounter_actor_missing",
                "declared_encounter": declared,
            }
        return {
            "status": "available",
            "source": "declared_encounter_record",
            "actor_id": str(declared["actor_id"]),
            "encounter_id": declared.get("encounter_id"),
            "declared_encounter": declared,
        }
    if len(actor_ids) == 1:
        return {
            "status": "available",
            "source": "single_actor_trace",
            "actor_id": actor_ids[0],
            "encounter_id": f"{actor_ids[0]}:trace-wide",
        }
    if not actor_ids:
        return {"status": "unavailable", "reason": "no_pedestrians_in_trace"}
    return {"status": "unavailable", "reason": "multiple_actors_without_encounter_binding"}


def _declared_encounter(trace: SimulationTraceExport) -> dict[str, Any]:
    for frame in trace.frames:
        for key in ("focal_encounter", "encounter"):
            value = frame.planner.get(key)
            if isinstance(value, Mapping):
                actor_id = value.get("actor_id") or value.get("pedestrian_id")
                if actor_id is not None:
                    return {
                        "actor_id": str(actor_id),
                        "encounter_id": value.get("encounter_id"),
                        "metadata": _encounter_metadata(value),
                    }
        encounters = frame.planner.get("encounters")
        if isinstance(encounters, Sequence) and not isinstance(encounters, str | bytes):
            for value in encounters:
                if isinstance(value, Mapping):
                    actor_id = value.get("actor_id") or value.get("pedestrian_id")
                    if actor_id is not None:
                        return {
                            "actor_id": str(actor_id),
                            "encounter_id": value.get("encounter_id"),
                            "metadata": _encounter_metadata(value),
                        }
    return {}


def _encounter_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    wanted = (
        "profile_version",
        "start_step",
        "end_step",
        "start_time_s",
        "end_time_s",
        "available_duration_s",
        "min_clearance_m",
        "min_ttc_s",
        "min_pet_s",
        "contact",
        "termination_reason",
    )
    return {key: value[key] for key in wanted if key in value}


def _process_frame(
    frame: SimulationTraceFrame,
    *,
    frame_index: int,
    focal_actor_id: object,
    route: RouteSpec | None,
    conflict_zone: ConflictZoneSpec | None,
    source_coordinate_frame: str,
) -> dict[str, Any]:
    robot_pos = _vector2(frame.robot.get("position"))
    robot_vel = _vector2(frame.robot.get("velocity"))
    nearest = _nearest_actor(frame, robot_pos=robot_pos)
    focal = _pedestrian_by_id(frame, focal_actor_id)
    focal_state = _relative_state(frame, focal=focal, robot_pos=robot_pos, robot_vel=robot_vel)
    return {
        "frame_index": frame_index,
        "step": frame.step,
        "time_s": frame.time_s,
        "source_coordinates": {
            "coordinate_frame": source_coordinate_frame,
            "robot": _world_actor(frame.robot),
            "focal_actor": _world_actor(focal) if focal is not None else None,
        },
        "world": {
            "status": "available"
            if source_coordinate_frame == "world" and robot_pos is not None
            else "unavailable",
            "reason": _world_frame_reason(source_coordinate_frame, robot_pos),
            "robot": _world_actor(frame.robot),
            "focal_actor": _world_actor(focal) if focal is not None else None,
        },
        "route": _route_frame(robot_pos, robot_vel, route),
        "conflict": _conflict_frame(robot_pos, focal, conflict_zone),
        "relative_interaction": focal_state,
        "global_minimum_actor": nearest,
        "commands": _command_state(frame),
    }


def _world_actor(actor: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if actor is None:
        return None
    return {
        "position": list(_vector2(actor.get("position")) or ()),
        "heading": actor.get("heading"),
        "velocity": list(_vector2(actor.get("velocity")) or ()),
        "radius_m": _radius(actor),
    }


def _world_frame_reason(
    source_coordinate_frame: str,
    robot_pos: tuple[float, float] | None,
) -> str:
    if robot_pos is None:
        return "missing_robot_position"
    if source_coordinate_frame != "world":
        return "source_coordinate_frame_not_world"
    return "source_trace_world_frame"


def _route_frame(
    robot_pos: tuple[float, float] | None,
    robot_vel: tuple[float, float] | None,
    route: RouteSpec | None,
) -> dict[str, Any]:
    if route is None:
        return {"status": "unavailable", "reason": "registered_route_unavailable"}
    route_availability = _route_availability(route)
    if route_availability["status"] != "available":
        return route_availability
    if robot_pos is None:
        return {"status": "unavailable", "reason": "missing_robot_position"}
    axis = (route.end[0] - route.start[0], route.end[1] - route.start[1])
    length = _norm(axis)
    if length <= 1e-12:
        return {"status": "unavailable", "reason": "registered_route_degenerate"}
    unit = (axis[0] / length, axis[1] / length)
    rel = (robot_pos[0] - route.start[0], robot_pos[1] - route.start[1])
    s_m = _dot(rel, unit)
    n_m = _cross(unit, rel)
    progress_rate = _dot(robot_vel, unit) if robot_vel is not None else None
    return {
        "status": "available",
        "route_id": route.route_id,
        "provenance_id": route.provenance_id,
        "s_m": s_m,
        "n_m": n_m,
        "progress_rate_mps": progress_rate,
    }


def _conflict_frame(
    robot_pos: tuple[float, float] | None,
    focal: Mapping[str, Any] | None,
    conflict_zone: ConflictZoneSpec | None,
) -> dict[str, Any]:
    if conflict_zone is None:
        return {"status": "unavailable", "reason": "registered_conflict_zone_unavailable"}
    conflict_availability = _conflict_availability(conflict_zone)
    if conflict_availability["status"] != "available":
        return conflict_availability
    if robot_pos is None:
        return {"status": "unavailable", "reason": "missing_robot_position"}
    focal_pos = _vector2(focal.get("position")) if focal is not None else None
    robot_distance = _distance(robot_pos, conflict_zone.center)
    result: dict[str, Any] = {
        "status": "available",
        "zone_id": conflict_zone.zone_id,
        "robot_signed_distance_to_zone_m": robot_distance - conflict_zone.radius_m,
    }
    if focal_pos is None:
        result["focal_actor_signed_distance_to_zone_m"] = None
        result["focal_actor_status"] = "unavailable"
        result["focal_actor_reason"] = "missing_focal_actor_position"
    else:
        result["focal_actor_signed_distance_to_zone_m"] = (
            _distance(focal_pos, conflict_zone.center) - conflict_zone.radius_m
        )
        result["focal_actor_status"] = "available"
    return result


def _relative_state(
    frame: SimulationTraceFrame,
    *,
    focal: Mapping[str, Any] | None,
    robot_pos: tuple[float, float] | None,
    robot_vel: tuple[float, float] | None,
) -> dict[str, Any]:
    if focal is None:
        return {"status": "unavailable", "reason": "focal_actor_missing_at_step"}
    focal_pos = _vector2(focal.get("position"))
    focal_vel = _vector2(focal.get("velocity"))
    if robot_pos is None or focal_pos is None:
        return {"status": "unavailable", "reason": "missing_position"}
    rel_pos = (focal_pos[0] - robot_pos[0], focal_pos[1] - robot_pos[1])
    heading = float(frame.robot.get("heading", 0.0))
    forward = (math.cos(heading), math.sin(heading))
    left = (-forward[1], forward[0])
    center_distance = _norm(rel_pos)
    robot_radius = _radius(frame.robot)
    actor_radius = _radius(focal)
    clearance = proxy_surface_clearance_m(
        center_distance,
        robot_radius_m=robot_radius,
        actor_radius_m=actor_radius,
    )
    radius_sum = (
        robot_radius + actor_radius
        if robot_radius is not None and actor_radius is not None
        else None
    )
    payload: dict[str, Any] = {
        "status": "available",
        "actor_id": str(focal.get("id")),
        "relative_longitudinal_m": _dot(rel_pos, forward),
        "relative_lateral_m": _dot(rel_pos, left),
        "center_distance_m": center_distance,
        "proxy_surface_clearance_m": clearance["value_m"],
        "proxy_surface_clearance_status": clearance["status"],
        "proxy_surface_clearance_reason": clearance["reason"],
        "clearance_semantics": "proxy_envelope_surface_clearance",
        "center_distance_semantics": "center_to_center_distance",
        "geometric_body_clearance_status": "unavailable",
        "geometric_body_clearance_reason": "trace_provides_proxy_radii_not_body_geometry",
        "robot_radius_m": robot_radius,
        "focal_actor_radius_m": actor_radius,
    }
    if robot_vel is None or focal_vel is None:
        payload["relative_velocity_status"] = "unavailable"
        payload["relative_velocity_reason"] = "missing_velocity"
        payload["radial_closing_speed_mps"] = None
        payload["closest_approach"] = {
            "status": "unavailable",
            "reason": "missing_velocity",
        }
        return payload
    rel_vel = (focal_vel[0] - robot_vel[0], focal_vel[1] - robot_vel[1])
    payload["relative_velocity_longitudinal_mps"] = _dot(rel_vel, forward)
    payload["relative_velocity_lateral_mps"] = _dot(rel_vel, left)
    payload["radial_closing_speed_mps"] = (
        -_dot(rel_vel, rel_pos) / center_distance if center_distance > 1e-12 else None
    )
    payload["relative_velocity_convention"] = "actor_minus_robot"
    payload["closing_speed_convention"] = "negative_radial_distance_derivative"
    payload["closest_approach"] = constant_velocity_closest_approach(
        rel_pos,
        rel_vel,
        radius_sum_m=radius_sum,
    )
    return payload


def _command_state(frame: SimulationTraceFrame) -> dict[str, Any]:
    selected = frame.planner.get("selected_action")
    if not isinstance(selected, Mapping):
        return {"status": "unavailable", "reason": "selected_action_unavailable"}
    executed = frame.planner.get("executed_action")
    return {
        "status": "available",
        "commanded": dict(selected),
        "executed": dict(executed) if isinstance(executed, Mapping) else None,
        "executed_status": "available" if isinstance(executed, Mapping) else "unavailable",
    }


def _diagnostics(frames: Sequence[Mapping[str, Any]], *, route_available: bool) -> dict[str, Any]:
    clearances = [
        frame["relative_interaction"].get("proxy_surface_clearance_m")
        for frame in frames
        if frame["relative_interaction"].get("status") == "available"
        and frame["relative_interaction"].get("proxy_surface_clearance_status") == "available"
    ]
    threshold = _profiles()["threshold_profile"]["proxy_surface_clearance_threshold_m"]
    if clearances:
        exposure = duration_where(
            frames,
            lambda frame: (
                frame["relative_interaction"].get("status") == "available"
                and frame["relative_interaction"].get("proxy_surface_clearance_status")
                == "available"
                and frame["relative_interaction"]["proxy_surface_clearance_m"] < threshold
            ),
        )
        deficit = 0.0
        for left, right in pairwise(frames):
            value = left["relative_interaction"].get("proxy_surface_clearance_m")
            if isinstance(value, int | float) and value < threshold:
                deficit += (threshold - float(value)) * (
                    float(right["time_s"]) - float(left["time_s"])
                )
    else:
        exposure = None
        deficit = None
    return {
        "minimum_proxy_surface_clearance_m": min(clearances) if clearances else None,
        "threshold_exposure": {
            "profile_version": THRESHOLD_PROFILE_VERSION,
            "threshold_m": threshold,
            "duration_s": exposure,
            "integrated_clearance_deficit_m_s": deficit,
            "status": "available" if clearances else "unavailable",
        },
        "route_progress": _route_progress_summary(frames)
        if route_available
        else {
            "status": "unavailable",
            "reason": "registered_route_unavailable",
        },
        "stall": summarize_stall(
            frames,
            speed_getter=_speed_from_frame,
            stall_speed_threshold_mps=_profiles()["phase_profile"]["stall_speed_threshold_mps"],
            stall_min_duration_s=_profiles()["phase_profile"]["stall_min_duration_s"],
        ),
        "conflict_zone_occupancy": _conflict_occupancy(frames),
        "reversal_counts": summarize_reversals(
            frames,
            speed_getter=_speed_from_frame,
            heading_delta_threshold_rad=_profiles()["reversal_profile"][
                "heading_delta_threshold_rad"
            ],
        ),
    }


def _route_progress_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    route_frames = [
        frame["route"] for frame in frames if frame["route"].get("status") == "available"
    ]
    if not route_frames:
        return {"status": "unavailable", "reason": "route_frame_unavailable"}
    return {
        "status": "available",
        "start_s_m": route_frames[0]["s_m"],
        "end_s_m": route_frames[-1]["s_m"],
        "delta_s_m": route_frames[-1]["s_m"] - route_frames[0]["s_m"],
    }


def _conflict_occupancy(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not any(frame["conflict"].get("status") == "available" for frame in frames):
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_unavailable",
            "robot_duration_s": None,
            "focal_actor_duration_s": None,
        }
    return {
        "status": "available",
        "robot_duration_s": duration_where(
            frames,
            lambda frame: (
                frame["conflict"].get("status") == "available"
                and frame["conflict"]["robot_signed_distance_to_zone_m"] <= 0
            ),
        ),
        "focal_actor_duration_s": duration_where(
            frames,
            lambda frame: (
                frame["conflict"].get("focal_actor_status") == "available"
                and frame["conflict"]["focal_actor_signed_distance_to_zone_m"] <= 0
            ),
        ),
    }


def _event_anchors(
    trace: SimulationTraceExport,
    *,
    frames: Sequence[Mapping[str, Any]],
    focal_actor_id: object,
) -> list[dict[str, Any]]:
    events = [
        _event_from_condition(
            "minimum_clearance",
            _minimum_clearance_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["relative_interaction.proxy_surface_clearance_m"],
            absent_status="unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "first_material_deceleration",
            _first_deceleration_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["commands.commanded.linear_velocity"],
            absent_status="not_observed"
            if _has_command_signal(frames, "linear_velocity")
            else "unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "first_material_turn_response",
            _first_turn_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["commands.commanded.angular_velocity"],
            absent_status="not_observed"
            if _has_command_signal(frames, "angular_velocity")
            else "unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "conflict_zone_entry",
            _first_conflict_entry(frames),
            actor_id=focal_actor_id,
            source_fields=["conflict.robot_signed_distance_to_zone_m"],
            absent_status="not_observed" if _has_conflict_signal(frames) else "unavailable",
            zone_id=_first_zone_id(frames),
        ),
        _event_from_condition(
            "exact_collision_event",
            _first_collision_frame(trace, frames),
            actor_id=focal_actor_id,
            source_fields=["planner.collision", "planner.collision_state"],
            absent_status="not_observed" if _has_collision_signal(trace) else "unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "proxy_overlap_event",
            _first_proxy_overlap_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["relative_interaction.proxy_surface_clearance_m"],
            absent_status="not_observed" if _has_proxy_clearance_signal(frames) else "unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "sustained_stall_onset",
            _first_stall_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["robot.velocity"],
            absent_status="not_observed" if _has_robot_velocity_signal(frames) else "unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "recovery_onset",
            _first_recovery_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["robot.velocity"],
            absent_status="not_observed" if _has_robot_velocity_signal(frames) else "unavailable",
            zone_id=None,
        ),
        _event_from_condition(
            "terminal_event",
            frames[-1] if frames else None,
            actor_id=focal_actor_id,
            source_fields=["trace.frames[-1]"],
            absent_status="unavailable",
            zone_id=None,
        ),
    ]
    return events


def _event_from_condition(
    event_type: str,
    frame: Mapping[str, Any] | None,
    *,
    actor_id: object,
    source_fields: list[str],
    absent_status: str,
    zone_id: object,
) -> dict[str, Any]:
    if frame is None:
        return {
            "event_id": f"{event_type}-{absent_status}",
            "event_type": event_type,
            "detector_profile_version": EVENT_PROFILE_VERSION,
            "status": absent_status,
            "confidence": "not_available",
            "actor_id": str(actor_id) if actor_id is not None else None,
            "zone_id": str(zone_id) if zone_id is not None else None,
            "reason": "event_not_observed"
            if absent_status == "not_observed"
            else "required_signal_unavailable",
            "source_fields": source_fields,
            "visual_anchor_eligibility": {
                "eligible": False,
                "reason": f"event_{absent_status}",
            },
        }
    return {
        "event_id": f"step-{int(frame['step']):04d}-{_slug(event_type)}",
        "event_type": event_type,
        "detector_profile_version": EVENT_PROFILE_VERSION,
        "status": "available",
        "confidence": "deterministic_trace_rule",
        "time_s": float(frame["time_s"]),
        "step": int(frame["step"]),
        "actor_id": str(actor_id) if actor_id is not None else None,
        "zone_id": str(zone_id) if zone_id is not None else None,
        "source_fields": source_fields,
        "visual_anchor_eligibility": {
            "eligible": True,
            "reason": "deterministic_trace_event",
        },
    }


def _minimum_clearance_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        frame
        for frame in frames
        if isinstance(frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float)
    ]
    return (
        min(
            candidates,
            key=lambda frame: (
                float(frame["relative_interaction"]["proxy_surface_clearance_m"]),
                int(frame["step"]),
            ),
        )
        if candidates
        else None
    )


def _first_deceleration_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    previous: float | None = None
    threshold = _profiles()["event_anchor_profile"]["material_deceleration_drop_mps"]
    for frame in frames:
        command = frame["commands"].get("commanded")
        value = command.get("linear_velocity") if isinstance(command, Mapping) else None
        if isinstance(value, int | float):
            current = float(value)
            if previous is not None and previous - current >= threshold:
                return frame
            previous = current
    return None


def _first_turn_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    threshold = _profiles()["event_anchor_profile"]["material_turn_response_rad_s"]
    for frame in frames:
        command = frame["commands"].get("commanded")
        value = command.get("angular_velocity") if isinstance(command, Mapping) else None
        if isinstance(value, int | float) and abs(float(value)) >= threshold:
            return frame
    return None


def _first_conflict_entry(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next(
        (
            frame
            for frame in frames
            if frame["conflict"].get("status") == "available"
            and frame["conflict"]["robot_signed_distance_to_zone_m"] <= 0
        ),
        None,
    )


def _has_command_signal(frames: Sequence[Mapping[str, Any]], key: str) -> bool:
    return any(
        isinstance(command := frame["commands"].get("commanded"), Mapping)
        and isinstance(command.get(key), int | float)
        for frame in frames
    )


def _has_conflict_signal(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(frame["conflict"].get("status") == "available" for frame in frames)


def _first_zone_id(frames: Sequence[Mapping[str, Any]]) -> object:
    return next(
        (
            frame["conflict"].get("zone_id")
            for frame in frames
            if frame["conflict"].get("zone_id") is not None
        ),
        None,
    )


def _has_collision_signal(trace: SimulationTraceExport) -> bool:
    return any(
        "collision" in frame.planner or "collision_state" in frame.planner for frame in trace.frames
    )


def _has_proxy_clearance_signal(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float)
        for frame in frames
    )


def _has_robot_velocity_signal(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(_speed_from_frame(frame) is not None for frame in frames)


def _first_collision_frame(
    trace: SimulationTraceExport,
    frames: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for trace_frame, process_frame in zip(trace.frames, frames, strict=False):
        collision = trace_frame.planner.get("collision") or trace_frame.planner.get(
            "collision_state"
        )
        if collision is True:
            return process_frame
        if isinstance(collision, Mapping) and collision.get("value") is True:
            return process_frame
    return None


def _first_proxy_overlap_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next(
        (
            frame
            for frame in frames
            if isinstance(
                frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float
            )
            and frame["relative_interaction"]["proxy_surface_clearance_m"] <= 0
        ),
        None,
    )


def _first_stall_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    profile = _profiles()["phase_profile"]
    return first_sustained_stall_frame(
        frames,
        speed_getter=_speed_from_frame,
        stall_speed_threshold_mps=profile["stall_speed_threshold_mps"],
        stall_min_duration_s=profile["stall_min_duration_s"],
    )


def _first_recovery_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    profile = _profiles()["phase_profile"]
    return first_recovery_frame(
        frames,
        speed_getter=_speed_from_frame,
        stall_speed_threshold_mps=profile["stall_speed_threshold_mps"],
        stall_min_duration_s=profile["stall_min_duration_s"],
        recovery_speed_threshold_mps=profile["recovery_speed_threshold_mps"],
    )


def _global_minimum_series(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "step": frame["step"],
            "time_s": frame["time_s"],
            "actor_id": frame["global_minimum_actor"].get("actor_id"),
            "center_distance_m": frame["global_minimum_actor"].get("center_distance_m"),
        }
        for frame in frames
        if frame["global_minimum_actor"].get("status") == "available"
    ]
    return {
        "status": "available" if rows else "unavailable",
        "reason": "nearest_actor_by_center_distance" if rows else "no_pedestrians_in_trace",
        "series": rows,
    }


def _actor_switch_events(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_actor: str | None = None
    for frame in frames:
        actor_id = frame["global_minimum_actor"].get("actor_id")
        if actor_id is None:
            continue
        actor = str(actor_id)
        if previous_actor is not None and actor != previous_actor:
            events.append(
                {
                    "event_type": "global_minimum_actor_switch",
                    "step": frame["step"],
                    "time_s": frame["time_s"],
                    "previous_actor_id": previous_actor,
                    "new_actor_id": actor,
                    "status": "available",
                }
            )
        previous_actor = actor
    return events


def _actor_contiguity(
    frames: Sequence[Mapping[str, Any]],
    actor_id: object,
    *,
    declared: object,
) -> dict[str, Any]:
    if actor_id is None:
        return {"status": "unavailable", "reason": "focal_actor_unavailable"}
    missing_steps = [
        int(frame["step"])
        for frame in frames
        if frame["relative_interaction"].get("status") == "unavailable"
        and frame["relative_interaction"].get("reason") == "focal_actor_missing_at_step"
    ]
    available_frames = [
        frame
        for frame in frames
        if frame["relative_interaction"].get("status") == "available"
        and frame["relative_interaction"].get("actor_id") == str(actor_id)
    ]
    clearances = [
        frame["relative_interaction"].get("proxy_surface_clearance_m")
        for frame in available_frames
        if isinstance(frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float)
    ]
    metadata = declared.get("metadata", {}) if isinstance(declared, Mapping) else {}
    required_metadata = {
        key: (
            {"status": "available", "value": metadata[key]}
            if key in metadata
            else {"status": "unavailable", "reason": "declared_encounter_field_missing"}
        )
        for key in (
            "profile_version",
            "start_step",
            "end_step",
            "start_time_s",
            "end_time_s",
            "available_duration_s",
            "min_clearance_m",
            "min_ttc_s",
            "min_pet_s",
            "contact",
        )
    }
    return {
        "status": "available",
        "actor_id": str(actor_id),
        "contiguous": not missing_steps,
        "missing_steps": missing_steps,
        "reason": "actor_present_all_frames" if not missing_steps else "actor_missing_within_trace",
        "computed_available_duration_s": _available_duration(available_frames),
        "computed_min_proxy_surface_clearance_m": min(clearances) if clearances else None,
        "declared_metadata": required_metadata,
    }


def _available_duration(frames: Sequence[Mapping[str, Any]]) -> float | None:
    if len(frames) < 2:
        return None
    return float(frames[-1]["time_s"]) - float(frames[0]["time_s"])


def _nearest_actor(
    frame: SimulationTraceFrame,
    *,
    robot_pos: tuple[float, float] | None,
) -> dict[str, Any]:
    if robot_pos is None or not frame.pedestrians:
        return {"status": "unavailable", "reason": "missing_robot_or_pedestrian_position"}
    candidates: list[tuple[float, str]] = []
    for pedestrian in frame.pedestrians:
        ped_pos = _vector2(pedestrian.get("position"))
        if ped_pos is None or "id" not in pedestrian:
            continue
        candidates.append((_distance(robot_pos, ped_pos), str(pedestrian["id"])))
    if not candidates:
        return {"status": "unavailable", "reason": "missing_pedestrian_position"}
    center_distance, actor_id = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "status": "available",
        "actor_id": actor_id,
        "center_distance_m": center_distance,
    }


def _pedestrian_by_id(frame: SimulationTraceFrame, actor_id: object) -> Mapping[str, Any] | None:
    if actor_id is None:
        return None
    target = str(actor_id)
    for pedestrian in frame.pedestrians:
        if str(pedestrian.get("id")) == target:
            return pedestrian
    return None


def _speed_from_frame(frame: Mapping[str, Any]) -> float | None:
    robot = frame["world"].get("robot")
    velocity = robot.get("velocity") if isinstance(robot, Mapping) else None
    if not isinstance(velocity, list | tuple) or len(velocity) != 2:
        return None
    return math.hypot(float(velocity[0]), float(velocity[1]))


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


def _radius(actor: Mapping[str, Any]) -> float | None:
    value = actor.get("radius")
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _norm(value: tuple[float, float]) -> float:
    return math.hypot(value[0], value[1])


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "event"
