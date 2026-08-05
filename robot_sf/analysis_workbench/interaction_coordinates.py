"""Build ``worked_example_process_trace.v1`` diagnostics from admitted trace exports."""

from __future__ import annotations

import hashlib
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
NEAR_MISS_ENCOUNTER_SCHEMA_FILE = (
    Path(__file__).parents[1] / "benchmark" / "schemas" / "near_miss_encounter.v1.json"
)
EVENT_PROFILE_VERSION = "worked_example_event_detectors.v1"
THRESHOLD_PROFILE_VERSION = "worked_example_threshold_profile.diagnostic.v1"
CANONICAL_ENCOUNTER_SCHEMA_VERSION = "near_miss_encounter.v1"
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
COLLISION_PARTNER_TYPES = frozenset({"pedestrian", "static_geometry", "boundary", "goal_artifact"})


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Registered straight route used for route-frame diagnostics."""

    route_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    provenance_id: str | None = None
    registry_checksum: str | None = None


@dataclass(frozen=True, slots=True)
class ConflictZoneSpec:
    """Registered circular conflict zone used for conflict-frame diagnostics."""

    zone_id: str
    center: tuple[float, float]
    radius_m: float
    provenance_id: str | None = None
    registry_checksum: str | None = None


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


@lru_cache(maxsize=1)
def load_near_miss_encounter_schema() -> dict[str, Any]:
    """Load the canonical ``near_miss_encounter.v1`` report schema.

    Returns:
        Parsed JSON Schema document.
    """

    return json.loads(NEAR_MISS_ENCOUNTER_SCHEMA_FILE.read_text(encoding="utf-8"))


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
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise WorkedExampleProcessTraceValidationError(errors, source=source)


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:  # noqa: C901, PLR0912
    errors: list[str] = []
    frames = payload.get("frames")
    if isinstance(frames, list):
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping) or not frame:
                errors.append(f"/frames/{index}: expected non-empty frame record")
                continue
            for key in ("source_coordinates", "world", "route", "conflict", "relative_interaction"):
                value = frame.get(key)
                if isinstance(value, Mapping) and not value:
                    errors.append(f"/frames/{index}/{key}: expected non-empty record")
            errors.extend(_validate_frame_record(frame, index))
            relative = frame.get("relative_interaction")
            if isinstance(relative, Mapping) and relative.get("status") == "available":
                for key in (
                    "relative_longitudinal_m",
                    "relative_lateral_m",
                    "center_distance_m",
                ):
                    if not _finite_json_number(relative.get(key)):
                        errors.append(
                            f"/frames/{index}/relative_interaction/{key}: expected finite number"
                        )
    events = payload.get("event_anchors")
    if isinstance(events, list):
        for index, event in enumerate(events):
            if not isinstance(event, Mapping) or not event:
                errors.append(f"/event_anchors/{index}: expected non-empty event record")
                continue
            status = event.get("status")
            if status == "available":
                if not isinstance(event.get("time_s"), int | float):
                    errors.append(
                        f"/event_anchors/{index}/time_s: required when status is available"
                    )
                if not isinstance(event.get("step"), int):
                    errors.append(f"/event_anchors/{index}/step: required when status is available")
    hierarchy = payload.get("event_anchor_hierarchy")
    if isinstance(hierarchy, Mapping):
        errors.extend(_validate_event_anchor_hierarchy(hierarchy, events, frames))
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        errors.extend(_validate_diagnostics_record(diagnostics))
    encounters = payload.get("encounters")
    if isinstance(encounters, Mapping):
        errors.extend(
            _validate_global_minimum_series(encounters.get("global_minimum_over_all_actors"))
        )
    focal = (
        payload.get("encounters", {}).get("focal")
        if isinstance(payload.get("encounters"), Mapping)
        else None
    )
    if isinstance(focal, Mapping):
        status = focal.get("status")
        if status not in {"available", "unavailable"}:
            errors.append("/encounters/focal/status: expected available or unavailable")
        if status == "available" and not focal.get("actor_id"):
            errors.append("/encounters/focal/actor_id: required when status is available")
        errors.extend(_validate_focal_actor_binding(focal, frames))
        declared = focal.get("declared_encounter")
        if (
            status == "available"
            and isinstance(declared, Mapping)
            and declared.get("schema_version") == CANONICAL_ENCOUNTER_SCHEMA_VERSION
        ):
            errors.extend(_validate_canonical_declared_encounter(declared))
    pair = payload.get("pair_compatibility")
    if isinstance(pair, Mapping):
        if pair.get("status") not in {"available", "unavailable", "incompatible"}:
            errors.append("/pair_compatibility/status: invalid status")
        errors.extend(_validate_pair_semantics(pair))
        divergence = pair.get("divergence_interpretation")
        if isinstance(divergence, Mapping) and divergence.get("allowed") is True:
            shared_prefix = pair.get("shared_prefix")
            if not (
                isinstance(shared_prefix, Mapping) and shared_prefix.get("shared_prefix") is True
            ):
                errors.append(
                    "/pair_compatibility/divergence_interpretation/allowed: requires shared_prefix true"
                )
    return errors


def _validate_frame_record(frame: Mapping[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _require_keys(
            frame.get("source_coordinates"),
            f"/frames/{index}/source_coordinates",
            required={"coordinate_frame", "robot", "focal_actor"},
            allowed={"coordinate_frame", "robot", "focal_actor"},
        )
    )
    errors.extend(
        _validate_actor_state(
            frame.get("source_coordinates", {}).get("robot")
            if isinstance(frame.get("source_coordinates"), Mapping)
            else None,
            f"/frames/{index}/source_coordinates/robot",
            nullable=False,
        )
    )
    errors.extend(
        _validate_actor_state(
            frame.get("source_coordinates", {}).get("focal_actor")
            if isinstance(frame.get("source_coordinates"), Mapping)
            else None,
            f"/frames/{index}/source_coordinates/focal_actor",
            nullable=True,
        )
    )
    errors.extend(
        _require_keys(
            frame.get("world"),
            f"/frames/{index}/world",
            required={"status", "reason", "robot", "focal_actor"},
            allowed={"status", "reason", "robot", "focal_actor"},
        )
    )
    errors.extend(
        _validate_actor_state(
            frame.get("world", {}).get("robot")
            if isinstance(frame.get("world"), Mapping)
            else None,
            f"/frames/{index}/world/robot",
            nullable=False,
        )
    )
    errors.extend(
        _validate_actor_state(
            frame.get("world", {}).get("focal_actor")
            if isinstance(frame.get("world"), Mapping)
            else None,
            f"/frames/{index}/world/focal_actor",
            nullable=True,
        )
    )
    errors.extend(_validate_route_record(frame.get("route"), f"/frames/{index}/route"))
    errors.extend(_validate_conflict_record(frame.get("conflict"), f"/frames/{index}/conflict"))
    errors.extend(
        _validate_relative_record(
            frame.get("relative_interaction"), f"/frames/{index}/relative_interaction"
        )
    )
    errors.extend(
        _require_keys(
            frame.get("global_minimum_actor"),
            f"/frames/{index}/global_minimum_actor",
            required={"status", "reason"}
            if frame.get("global_minimum_actor", {}).get("status") == "unavailable"
            else {"status", "actor_id", "center_distance_m"},
            allowed={"status", "reason", "actor_id", "center_distance_m"},
        )
    )
    errors.extend(_validate_commands_record(frame.get("commands"), f"/frames/{index}/commands"))
    return errors


def _validate_route_record(value: object, path: str) -> list[str]:
    required = (
        {"status", "reason"}
        if _status(value) == "unavailable"
        else {
            "status",
            "route_id",
            "provenance_id",
            "registry_checksum",
            "geometry",
            "s_m",
            "n_m",
            "progress_rate_mps",
        }
    )
    allowed = required | {"reason", "source_coordinate_frame", "geometry_checksum"}
    errors = _require_keys(value, path, required=required, allowed=allowed)
    if not isinstance(value, Mapping) or value.get("status") != "available":
        return errors
    for key in ("s_m", "n_m"):
        if not _finite_json_number(value.get(key)):
            errors.append(f"{path}/{key}: expected finite number")
    if not _finite_or_null(value.get("progress_rate_mps")):
        errors.append(f"{path}/progress_rate_mps: expected finite number or null")
    errors.extend(_validate_geometry(value.get("geometry"), f"{path}/geometry"))
    checksum = value.get("geometry_checksum", value.get("registry_checksum"))
    if (
        isinstance(checksum, str)
        and isinstance(value.get("geometry"), Mapping)
        and checksum != _geometry_checksum(value["geometry"])
    ):
        errors.append(f"{path}/geometry_checksum: must match geometry")
    return errors


def _validate_conflict_record(value: object, path: str) -> list[str]:
    required = (
        {"status", "reason"}
        if _status(value) == "unavailable"
        else {
            "status",
            "zone_id",
            "provenance_id",
            "registry_checksum",
            "geometry",
            "robot_signed_distance_to_zone_m",
            "focal_actor_signed_distance_to_zone_m",
            "focal_actor_status",
        }
    )
    allowed = required | {
        "reason",
        "source_coordinate_frame",
        "geometry_checksum",
        "focal_actor_reason",
    }
    errors = _require_keys(value, path, required=required, allowed=allowed)
    if not isinstance(value, Mapping) or value.get("status") != "available":
        return errors
    if not _finite_json_number(value.get("robot_signed_distance_to_zone_m")):
        errors.append(f"{path}/robot_signed_distance_to_zone_m: expected finite number")
    if value.get("focal_actor_status") not in {"available", "unavailable"}:
        errors.append(f"{path}/focal_actor_status: expected available or unavailable")
    if not _finite_or_null(value.get("focal_actor_signed_distance_to_zone_m")):
        errors.append(
            f"{path}/focal_actor_signed_distance_to_zone_m: expected finite number or null"
        )
    errors.extend(_validate_geometry(value.get("geometry"), f"{path}/geometry"))
    checksum = value.get("geometry_checksum", value.get("registry_checksum"))
    if (
        isinstance(checksum, str)
        and isinstance(value.get("geometry"), Mapping)
        and checksum != _geometry_checksum(value["geometry"])
    ):
        errors.append(f"{path}/geometry_checksum: must match geometry")
    return errors


def _validate_relative_record(value: object, path: str) -> list[str]:  # noqa: C901
    required = (
        {"status", "reason"}
        if _status(value) == "unavailable"
        else {
            "status",
            "actor_id",
            "relative_longitudinal_m",
            "relative_lateral_m",
            "center_distance_m",
            "proxy_surface_clearance_m",
            "proxy_surface_clearance_status",
            "proxy_surface_clearance_reason",
            "clearance_semantics",
            "center_distance_semantics",
            "geometric_body_clearance_status",
            "geometric_body_clearance_reason",
            "robot_radius_m",
            "focal_actor_radius_m",
        }
    )
    allowed = required | {
        "relative_velocity_status",
        "relative_velocity_reason",
        "radial_closing_speed_mps",
        "closest_approach",
        "relative_velocity_longitudinal_mps",
        "relative_velocity_lateral_mps",
        "relative_velocity_convention",
        "closing_speed_convention",
    }
    errors = _require_keys(value, path, required=required, allowed=allowed)
    if not isinstance(value, Mapping) or value.get("status") != "available":
        return errors
    for key in (
        "relative_longitudinal_m",
        "relative_lateral_m",
        "center_distance_m",
    ):
        if not _finite_json_number(value.get(key)):
            errors.append(f"{path}/{key}: expected finite number")
    if not _finite_or_null(value.get("proxy_surface_clearance_m")):
        errors.append(f"{path}/proxy_surface_clearance_m: expected finite number or null")
    for key in ("robot_radius_m", "focal_actor_radius_m", "radial_closing_speed_mps"):
        if key in value and not _finite_or_null(value.get(key)):
            errors.append(f"{path}/{key}: expected finite number or null")
    if value.get("proxy_surface_clearance_status") not in {"available", "unavailable"}:
        errors.append(f"{path}/proxy_surface_clearance_status: expected available or unavailable")
    if value.get("proxy_surface_clearance_status") == "available" and not _finite_json_number(
        value.get("proxy_surface_clearance_m")
    ):
        errors.append(f"{path}/proxy_surface_clearance_m: required when clearance is available")
    if (
        value.get("proxy_surface_clearance_status") == "unavailable"
        and value.get("proxy_surface_clearance_m") is not None
    ):
        errors.append(
            f"{path}/proxy_surface_clearance_m: must be null when clearance is unavailable"
        )
    if value.get("relative_velocity_status") not in {"available", "unavailable"}:
        errors.append(f"{path}/relative_velocity_status: expected available or unavailable")
    errors.extend(
        _validate_closest_approach(value.get("closest_approach"), f"{path}/closest_approach")
    )
    return errors


def _validate_geometry(value: object, path: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path}: expected object"]
    geometry_type = value.get("type")
    if geometry_type == "line_segment":
        errors = _require_keys(
            value,
            path,
            required={"type", "start", "end"},
            allowed={"type", "start", "end"},
        )
        for key in ("start", "end"):
            if not _finite_vector2(value.get(key)):
                errors.append(f"{path}/{key}: expected finite 2-vector")
        return errors
    if geometry_type == "circle":
        errors = _require_keys(
            value,
            path,
            required={"type", "center", "radius_m"},
            allowed={"type", "center", "radius_m"},
        )
        if not _finite_vector2(value.get("center")):
            errors.append(f"{path}/center: expected finite 2-vector")
        if not _finite_json_number(value.get("radius_m")):
            errors.append(f"{path}/radius_m: expected finite number")
        return errors
    return [f"{path}/type: unexpected geometry type"]


def _validate_commands_record(value: object, path: str) -> list[str]:
    errors = _require_keys(
        value,
        path,
        required={"status", "reason"}
        if _status(value) == "unavailable"
        else {"status", "commanded", "executed", "executed_status"},
        allowed={"status", "reason", "commanded", "executed", "executed_status"},
    )
    if not isinstance(value, Mapping) or value.get("status") != "available":
        return errors
    if not isinstance(value.get("commanded"), Mapping):
        errors.append(f"{path}/commanded: expected command mapping")
    elif not all(_json_scalar(item) for item in value["commanded"].values()):
        errors.append(f"{path}/commanded: expected JSON scalar command values")
    executed = value.get("executed")
    if executed is not None and not isinstance(executed, Mapping):
        errors.append(f"{path}/executed: expected command mapping or null")
    elif isinstance(executed, Mapping) and not all(
        _json_scalar(item) for item in executed.values()
    ):
        errors.append(f"{path}/executed: expected JSON scalar command values")
    if value.get("executed_status") not in {"available", "unavailable"}:
        errors.append(f"{path}/executed_status: expected available or unavailable")
    if value.get("executed_status") == "available" and executed is None:
        errors.append(f"{path}/executed: required when executed_status is available")
    if value.get("executed_status") == "unavailable" and executed is not None:
        errors.append(f"{path}/executed: must be null when executed_status is unavailable")
    return errors


def _validate_closest_approach(value: object, path: str) -> list[str]:
    errors = _require_keys(
        value,
        path,
        required={"status", "reason"}
        if _status(value) == "unavailable"
        else {
            "status",
            "time_to_closest_approach_s",
            "center_distance_at_closest_approach_m",
            "proxy_surface_clearance_at_closest_approach_m",
            "proxy_surface_clearance_status",
            "proxy_surface_clearance_reason",
            "model",
            "profile_version",
        },
        allowed={
            "status",
            "reason",
            "time_to_closest_approach_s",
            "center_distance_at_closest_approach_m",
            "proxy_surface_clearance_at_closest_approach_m",
            "proxy_surface_clearance_status",
            "proxy_surface_clearance_reason",
            "model",
            "profile_version",
        },
    )
    if not isinstance(value, Mapping) or value.get("status") != "available":
        return errors
    for key in ("time_to_closest_approach_s", "center_distance_at_closest_approach_m"):
        if not _finite_json_number(value.get(key)):
            errors.append(f"{path}/{key}: expected finite number")
    if value.get("proxy_surface_clearance_status") not in {"available", "unavailable"}:
        errors.append(f"{path}/proxy_surface_clearance_status: expected available or unavailable")
    if not _finite_or_null(value.get("proxy_surface_clearance_at_closest_approach_m")):
        errors.append(
            f"{path}/proxy_surface_clearance_at_closest_approach_m: expected finite number or null"
        )
    if value.get("proxy_surface_clearance_status") == "available" and not _finite_json_number(
        value.get("proxy_surface_clearance_at_closest_approach_m")
    ):
        errors.append(
            f"{path}/proxy_surface_clearance_at_closest_approach_m: required when clearance is available"
        )
    return errors


def _validate_actor_state(value: object, path: str, *, nullable: bool) -> list[str]:
    if value is None and nullable:
        return []
    errors = _require_keys(
        value,
        path,
        required={"position", "heading", "velocity", "radius_m"},
        allowed={"position", "heading", "velocity", "radius_m"},
    )
    if not isinstance(value, Mapping):
        return errors
    for key in ("position", "velocity"):
        item = value.get(key)
        if not (
            isinstance(item, list)
            and len(item) in {0, 2}
            and all(_finite_json_number(number) for number in item)
        ):
            errors.append(f"{path}/{key}: expected empty or finite 2-vector")
    for key in ("heading", "radius_m"):
        if value.get(key) is not None and not _finite_json_number(value.get(key)):
            errors.append(f"{path}/{key}: expected finite number or null")
    return errors


def _validate_diagnostics_record(diagnostics: Mapping[str, Any]) -> list[str]:  # noqa: C901
    errors: list[str] = []
    errors.extend(
        _require_keys(
            diagnostics.get("route_progress"),
            "/diagnostics/route_progress",
            required={"status", "reason"}
            if _status(diagnostics.get("route_progress")) == "unavailable"
            else {"status", "start_s_m", "end_s_m", "delta_s_m"},
            allowed={"status", "reason", "start_s_m", "end_s_m", "delta_s_m"},
        )
    )
    errors.extend(
        _require_keys(
            diagnostics.get("stall"),
            "/diagnostics/stall",
            required={
                "profile_version",
                "status",
                "reason",
                "stall_min_duration_s",
                "sustained_stall_duration_s",
                "speed_coverage",
                "sustained_stall_onset_step",
            },
            allowed={
                "profile_version",
                "status",
                "reason",
                "stall_min_duration_s",
                "sustained_stall_duration_s",
                "speed_coverage",
                "sustained_stall_onset_step",
            },
        )
    )
    errors.extend(
        _require_keys(
            diagnostics.get("reversal_counts"),
            "/diagnostics/reversal_counts",
            required={
                "profile_version",
                "direction_semantics",
                "heading_reversal_count",
                "velocity_reversal_count",
            },
            allowed={
                "profile_version",
                "direction_semantics",
                "heading_reversal_count",
                "velocity_reversal_count",
            },
        )
    )
    coverage = diagnostics.get("coverage")
    errors.extend(
        _require_keys(
            coverage,
            "/diagnostics/coverage",
            required={"frame_count", "relative_interaction", "proxy_surface_clearance"},
            allowed={"frame_count", "relative_interaction", "proxy_surface_clearance"},
        )
    )
    if isinstance(coverage, Mapping):
        if not isinstance(coverage.get("frame_count"), int) or coverage["frame_count"] < 0:
            errors.append("/diagnostics/coverage/frame_count: expected nonnegative integer")
        for key in ("relative_interaction", "proxy_surface_clearance"):
            errors.extend(
                _validate_coverage_record(coverage.get(key), f"/diagnostics/coverage/{key}")
            )
    threshold = diagnostics.get("threshold_exposure")
    if isinstance(threshold, Mapping):
        for key in ("threshold_m",):
            if not _finite_json_number(threshold.get(key)):
                errors.append(f"/diagnostics/threshold_exposure/{key}: expected finite number")
        for key in ("duration_s", "integrated_clearance_deficit_m_s"):
            if not _finite_or_null(threshold.get(key)):
                errors.append(
                    f"/diagnostics/threshold_exposure/{key}: expected finite number or null"
                )
    reversal = diagnostics.get("reversal_counts")
    if isinstance(reversal, Mapping):
        for key in ("heading_reversal_count", "velocity_reversal_count"):
            if not isinstance(reversal.get(key), int) or reversal[key] < 0:
                errors.append(f"/diagnostics/reversal_counts/{key}: expected nonnegative integer")
    errors.extend(_validate_diagnostic_statuses(diagnostics))
    return errors


def _validate_coverage_record(value: object, path: str) -> list[str]:
    errors = _require_keys(
        value,
        path,
        required={"status", "available_frame_count", "missing_frame_count"},
        allowed={
            "status",
            "frame_count",
            "available_frame_count",
            "missing_frame_count",
            "missing_radius_frame_count",
            "missing_actor_interval_frame_count",
            "reason",
        },
    )
    if not isinstance(value, Mapping):
        return errors
    if value.get("status") not in {"complete", "partial"}:
        errors.append(f"{path}/status: expected complete or partial")
    frame_count = value.get("frame_count")
    available_count = value.get("available_frame_count")
    missing_count = value.get("missing_frame_count")
    if (
        isinstance(frame_count, int)
        and isinstance(available_count, int)
        and isinstance(missing_count, int)
        and available_count + missing_count != frame_count
    ):
        errors.append(f"{path}/missing_frame_count: counts must add to frame_count")
    if value.get("status") == "complete" and value.get("missing_frame_count") != 0:
        errors.append(f"{path}/status: complete requires zero missing frames")
    if value.get("status") == "partial" and value.get("missing_frame_count") == 0:
        errors.append(f"{path}/status: partial requires missing frames")
    for key, item in value.items():
        if key.endswith("_count") and (not isinstance(item, int) or item < 0):
            errors.append(f"{path}/{key}: expected nonnegative integer")
    return errors


def _validate_diagnostic_statuses(diagnostics: Mapping[str, Any]) -> list[str]:  # noqa: C901
    errors: list[str] = []
    threshold = diagnostics.get("threshold_exposure")
    if isinstance(threshold, Mapping):
        duration = threshold.get("duration_s")
        deficit = threshold.get("integrated_clearance_deficit_m_s")
        if threshold.get("status") == "available" and not (
            _finite_json_number(duration) and _finite_json_number(deficit)
        ):
            errors.append("/diagnostics/threshold_exposure/status: available requires durations")
        if threshold.get("status") == "unavailable" and (
            duration is not None or deficit is not None
        ):
            errors.append(
                "/diagnostics/threshold_exposure/status: unavailable requires null durations"
            )
    stall = diagnostics.get("stall")
    if isinstance(stall, Mapping):
        if stall.get("status") == "unavailable" and (
            stall.get("sustained_stall_duration_s") is not None
            or stall.get("sustained_stall_onset_step") is not None
        ):
            errors.append("/diagnostics/stall/status: unavailable requires null stall evidence")
        if stall.get("status") == "available" and not _finite_json_number(
            stall.get("sustained_stall_duration_s")
        ):
            errors.append("/diagnostics/stall/sustained_stall_duration_s: expected finite number")
    conflict = diagnostics.get("conflict_zone_occupancy")
    if isinstance(conflict, Mapping):
        if conflict.get("status") == "available":
            for key in ("robot_duration_s", "focal_actor_duration_s"):
                if not _finite_json_number(conflict.get(key)):
                    errors.append(
                        f"/diagnostics/conflict_zone_occupancy/{key}: expected finite number"
                    )
        if conflict.get("status") == "unavailable" and (
            conflict.get("robot_duration_s") is not None
            or conflict.get("focal_actor_duration_s") is not None
        ):
            errors.append(
                "/diagnostics/conflict_zone_occupancy/status: unavailable requires null durations"
            )
    return errors


def _validate_global_minimum_series(value: object) -> list[str]:
    errors = _require_keys(
        value,
        "/encounters/global_minimum_over_all_actors",
        required={"status", "reason", "series"},
        allowed={"status", "reason", "series"},
    )
    if not isinstance(value, Mapping):
        return errors
    series = value.get("series")
    if not isinstance(series, list):
        return [*errors, "/encounters/global_minimum_over_all_actors/series: expected array"]
    for index, row in enumerate(series):
        row_path = f"/encounters/global_minimum_over_all_actors/series/{index}"
        errors.extend(
            _require_keys(
                row,
                row_path,
                required={"step", "time_s", "actor_id", "center_distance_m"},
                allowed={"step", "time_s", "actor_id", "center_distance_m"},
            )
        )
        if isinstance(row, Mapping):
            if not isinstance(row.get("step"), int):
                errors.append(f"{row_path}/step: expected integer")
            if not _finite_json_number(row.get("time_s")):
                errors.append(f"{row_path}/time_s: expected finite number")
            if not isinstance(row.get("actor_id"), str):
                errors.append(f"{row_path}/actor_id: expected string")
            if not _finite_json_number(row.get("center_distance_m")):
                errors.append(f"{row_path}/center_distance_m: expected finite number")
    return errors


def _require_keys(
    value: object,
    path: str,
    *,
    required: set[str],
    allowed: set[str],
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path}: expected object"]
    errors = [f"{path}/{key}: required" for key in sorted(required - set(value))]
    errors.extend(f"{path}/{key}: unexpected field" for key in sorted(set(value) - allowed))
    return errors


def _status(value: object) -> object:
    return value.get("status") if isinstance(value, Mapping) else None


def _json_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | bool) or _finite_json_number(value)


def _validate_event_anchor_hierarchy(  # noqa: C901, PLR0912
    hierarchy: Mapping[str, Any],
    events: object,
    frames: object,
) -> list[str]:
    errors: list[str] = []
    expected_fallback_order = [
        "exact_collision_event",
        "minimum_clearance",
        "first_safety_predicate_breach",
        "sustained_stall_onset",
        "terminal_event",
    ]
    if hierarchy.get("fallback_order") != expected_fallback_order:
        errors.append("/event_anchor_hierarchy/fallback_order: unexpected fallback order")
    expected_available = _expected_hierarchy_anchors(events, expected_fallback_order)
    if hierarchy.get("available_anchors") != expected_available:
        errors.append(
            "/event_anchor_hierarchy/available_anchors: must match canonical event anchors"
        )
    expected_selected = expected_available[0] if expected_available else None
    expected_status = "available" if expected_selected is not None else "unavailable"
    if hierarchy.get("status") != expected_status:
        errors.append("/event_anchor_hierarchy/status: must match canonical event anchors")
    if hierarchy.get("selected_anchor") != expected_selected:
        errors.append("/event_anchor_hierarchy/selected_anchor: must match canonical event anchors")
    expected_anchor_time = expected_selected["time_s"] if expected_selected is not None else None
    if hierarchy.get("anchor_time_s") != expected_anchor_time:
        errors.append("/event_anchor_hierarchy/anchor_time_s: must match canonical event anchors")
    if hierarchy.get("status") == "available":
        selected = hierarchy.get("selected_anchor")
        if not isinstance(selected, Mapping):
            errors.append("/event_anchor_hierarchy/selected_anchor: required when available")
            return errors
        if selected not in hierarchy.get("available_anchors", []):
            errors.append("/event_anchor_hierarchy/selected_anchor: must be one available anchor")
        available_anchors = hierarchy.get("available_anchors")
        if isinstance(available_anchors, list) and available_anchors:
            best = min(
                (
                    anchor
                    for anchor in available_anchors
                    if isinstance(anchor, Mapping) and isinstance(anchor.get("rank"), int)
                ),
                key=lambda anchor: int(anchor["rank"]),
                default=None,
            )
            if best is not None and selected != best:
                errors.append("/event_anchor_hierarchy/selected_anchor: must be lowest-rank anchor")
        anchor_time = hierarchy.get("anchor_time_s")
        if not _finite_json_number(anchor_time) or anchor_time != selected.get("time_s"):
            errors.append("/event_anchor_hierarchy/anchor_time_s: must match selected anchor time")
        if isinstance(events, list):
            by_id = {
                event.get("event_id"): event
                for event in events
                if isinstance(event, Mapping) and event.get("status") == "available"
            }
            if selected.get("event_id") not in by_id:
                errors.append(
                    "/event_anchor_hierarchy/selected_anchor/event_id: unavailable anchor selected"
                )
        if isinstance(frames, list) and _finite_json_number(anchor_time):
            for index, frame in enumerate(frames):
                if not isinstance(frame, Mapping):
                    continue
                alignment = frame.get("event_alignment")
                if not isinstance(alignment, Mapping) or alignment.get("status") != "available":
                    errors.append(
                        f"/frames/{index}/event_alignment: required for available hierarchy"
                    )
                    continue
                expected_tau = float(frame["time_s"]) - float(anchor_time)
                if alignment.get("anchor_event_id") != selected.get("event_id"):
                    errors.append(
                        f"/frames/{index}/event_alignment/anchor_event_id: must match selected anchor"
                    )
                if alignment.get("anchor_event_type") != selected.get("event_type"):
                    errors.append(
                        f"/frames/{index}/event_alignment/anchor_event_type: must match selected anchor"
                    )
                if alignment.get("anchor_time_s") != selected.get("time_s"):
                    errors.append(
                        f"/frames/{index}/event_alignment/anchor_time_s: must match selected anchor"
                    )
                if not math.isclose(
                    float(alignment.get("tau_s", math.nan)), expected_tau, abs_tol=1e-12
                ):
                    errors.append(
                        f"/frames/{index}/event_alignment/tau_s: inconsistent with anchor"
                    )
    return errors


def _validate_focal_actor_binding(focal: Mapping[str, Any], frames: object) -> list[str]:
    actor_id = focal.get("actor_id")
    if not isinstance(actor_id, str) or not isinstance(frames, list):
        return []
    errors: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        relative = frame.get("relative_interaction")
        if (
            isinstance(relative, Mapping)
            and relative.get("status") == "available"
            and relative.get("actor_id") != actor_id
        ):
            errors.append(f"/frames/{index}/relative_interaction/actor_id: must match focal actor")
    return errors


def _expected_hierarchy_anchors(
    events: object, fallback_order: Sequence[str]
) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    available = {
        str(event["event_type"]): event
        for event in events
        if isinstance(event, Mapping)
        and event.get("status") == "available"
        and event.get("event_type") in fallback_order
        and _finite_json_number(event.get("time_s"))
        and isinstance(event.get("event_id"), str)
    }
    return [
        {
            "rank": rank,
            "event_type": event_type,
            "event_id": str(available[event_type]["event_id"]),
            "time_s": float(available[event_type]["time_s"]),
            "selection_role": "first_safety_predicate_breach"
            if event_type == "first_safety_predicate_breach"
            else "fallback_anchor",
        }
        for rank, event_type in enumerate(fallback_order)
        if event_type in available
    ]


def _validate_pair_semantics(pair: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    grain = pair.get("comparison_grain")
    grain_id = grain.get("grain_id") if isinstance(grain, Mapping) else None
    if pair.get("status") != "unavailable" and grain_id not in {
        "matched_planner_pair",
        "matched_realization_pair",
    }:
        errors.append("/pair_compatibility/comparison_grain/grain_id: invalid or undeclared grain")
    provenance = pair.get("provenance_gate")
    initial = pair.get("initial_state_equivalence")
    if pair.get("status") == "available":
        if not (isinstance(provenance, Mapping) and provenance.get("compatible") is True):
            errors.append(
                "/pair_compatibility/provenance_gate/compatible: required for available pair"
            )
        if grain_id == "matched_planner_pair" and not (
            isinstance(initial, Mapping) and initial.get("equivalent") is True
        ):
            errors.append("/pair_compatibility/initial_state_equivalence/equivalent: required")
    if isinstance(provenance, Mapping):
        checks = provenance.get("checks")
        if isinstance(checks, Mapping):
            required = ["map_id_present", "horizon_present"]
            if pair.get("comparison_grain", {}).get("grain_id") == "matched_realization_pair":
                required.append("config_digest_present")
            for key in required:
                if pair.get("status") == "available" and checks.get(key) is not True:
                    errors.append(f"/pair_compatibility/provenance_gate/checks/{key}: required")
    return errors


def _validate_canonical_declared_encounter(declared: Mapping[str, Any]) -> list[str]:
    record = declared.get("canonical_record")
    if not isinstance(record, Mapping):
        return ["/encounters/focal/declared_encounter/canonical_record: required"]
    required = {
        "schema_version",
        "encounter_id",
        "actor_id",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "minimum_clearance_m",
        "minimum_ttc_s",
        "maximum_closing_speed_mps",
        "minimum_pet_s",
        "sample_count",
        "valid_exposure_duration_s",
        "termination_reason",
        "contact_terminated",
        "contact_status",
        "contact_time_s",
        "unavailable_fields",
        "evidence_status",
    }
    extra = set(record) - required
    missing = required - set(record)
    errors = [
        f"/encounters/focal/declared_encounter/canonical_record/{key}: unexpected field"
        for key in sorted(extra)
    ]
    errors.extend(
        f"/encounters/focal/declared_encounter/canonical_record/{key}: required"
        for key in sorted(missing)
    )
    if record.get("schema_version") != CANONICAL_ENCOUNTER_SCHEMA_VERSION:
        errors.append(
            "/encounters/focal/declared_encounter/canonical_record/schema_version: invalid"
        )
    return errors


def _finite_json_number(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _finite_or_null(value: object) -> bool:
    return value is None or _finite_json_number(value)


def _finite_vector2(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_finite_json_number(item) for item in value)
    )


def load_near_miss_encounter_report(path: Path) -> dict[str, Any]:
    """Load and validate a canonical near-miss encounter report.

    Returns:
        Schema-valid report payload.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise WorkedExampleProcessTraceValidationError(
            ["expected a near_miss_encounter.v1 mapping payload"],
            source=path,
        )
    validator = Draft202012Validator(load_near_miss_encounter_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda err: list(err.absolute_path),
        )
    ]
    if errors:
        raise WorkedExampleProcessTraceValidationError(errors, source=path)
    return dict(payload)


def build_worked_example_process_trace(
    input_path: Path,
    *,
    route: RouteSpec | None = None,
    conflict_zone: ConflictZoneSpec | None = None,
    focal_actor_id: str | None = None,
    pair_input_path: Path | None = None,
    encounter_report_path: Path | None = None,
    pair_comparison_grain: str | None = None,
) -> dict[str, Any]:
    """Build a renderer-neutral process trace from one admitted trace export.

    Returns:
        Schema-valid process trace payload.
    """

    trace = load_simulation_trace_export(input_path)
    input_checksum = _sha256_file(input_path)
    pair_trace = load_simulation_trace_export(pair_input_path) if pair_input_path else None
    encounter_report = (
        load_near_miss_encounter_report(encounter_report_path)
        if encounter_report_path is not None
        else None
    )
    payload = build_worked_example_process_trace_from_export(
        trace,
        route=route,
        conflict_zone=conflict_zone,
        focal_actor_id=focal_actor_id,
        pair_trace=pair_trace,
        encounter_report=encounter_report,
        encounter_report_input_checksum=input_checksum if encounter_report is not None else None,
        pair_comparison_grain=pair_comparison_grain,
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
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
    pair_comparison_grain: str | None = None,
) -> dict[str, Any]:
    """Build a schema-valid process trace from a typed trace export.

    Returns:
        Schema-valid process trace payload.
    """

    focal = _resolve_focal_actor(
        trace,
        requested_actor_id=focal_actor_id,
        encounter_report=encounter_report,
        encounter_report_input_checksum=encounter_report_input_checksum,
    )
    route_availability = _route_availability(route)
    conflict_availability = _conflict_availability(conflict_zone)
    relative_availability = _relative_availability(focal)
    world_availability = _world_availability(trace)
    pair_focal = _resolve_focal_actor(pair_trace) if pair_trace is not None else None
    frames = [
        _process_frame(
            frame,
            frame_index=index,
            focal_actor_id=focal.get("actor_id"),
            focal_encounter=focal,
            route=route,
            conflict_zone=conflict_zone,
            source_coordinate_frame=trace.coordinate_frame,
        )
        for index, frame in enumerate(trace.frames)
    ]
    focal = dict(focal)
    focal["actor_contiguity"] = _actor_contiguity(
        _diagnostic_frames(frames),
        focal.get("actor_id"),
        declared=focal.get("declared_encounter"),
    )
    event_frames = _diagnostic_frames(frames)
    events = _event_anchors(trace, frames=event_frames, focal_actor_id=focal.get("actor_id"))
    event_anchor_hierarchy = _event_anchor_hierarchy(events)
    frames = _frames_with_event_alignment(frames, event_anchor_hierarchy)
    pair = (
        build_pair_compatibility_record(
            trace,
            pair_trace,
            left_events=events,
            comparison_grain=pair_comparison_grain or "undeclared",
            right_events=_event_anchors(
                pair_trace,
                frames=[
                    _process_frame(
                        frame,
                        frame_index=index,
                        focal_actor_id=pair_focal.get("actor_id") if pair_focal else None,
                        focal_encounter=pair_focal or {},
                        route=route,
                        conflict_zone=conflict_zone,
                        source_coordinate_frame=pair_trace.coordinate_frame,
                    )
                    for index, frame in enumerate(pair_trace.frames)
                ],
                focal_actor_id=pair_focal.get("actor_id") if pair_focal else None,
            ),
        )
        if pair_trace is not None
        else unavailable_pair_compatibility(comparison_grain=pair_comparison_grain)
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
        "diagnostics": _diagnostics(
            event_frames,
            route_available=route_availability["status"] == "available",
        ),
        "event_anchors": events,
        "event_anchor_hierarchy": event_anchor_hierarchy,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not route.registry_checksum:
        return {"status": "unavailable", "reason": "registered_route_checksum_unavailable"}
    if SHA256_HEX_RE.fullmatch(str(route.registry_checksum)) is None:
        return {"status": "unavailable", "reason": "registered_route_checksum_invalid"}
    if _vector2(route.start) is None or _vector2(route.end) is None:
        return {"status": "unavailable", "reason": "registered_route_invalid_geometry"}
    if _distance(route.start, route.end) <= 1e-12:
        return {"status": "unavailable", "reason": "registered_route_degenerate"}
    geometry = {"type": "line_segment", "start": list(route.start), "end": list(route.end)}
    geometry_checksum = _geometry_checksum(geometry)
    if route.registry_checksum != geometry_checksum:
        return {
            "status": "unavailable",
            "reason": "registered_route_checksum_geometry_mismatch",
            "geometry_checksum": geometry_checksum,
        }
    return {
        "status": "available",
        "reason": "registered_straight_route",
        "route_id": route.route_id,
        "provenance_id": route.provenance_id,
        "registry_checksum": route.registry_checksum,
        "coordinate_frame": "world",
        "geometry": geometry,
    }


def _conflict_availability(conflict_zone: ConflictZoneSpec | None) -> dict[str, Any]:
    if conflict_zone is None:
        return {"status": "unavailable", "reason": "registered_conflict_zone_unavailable"}
    if not conflict_zone.provenance_id:
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_provenance_unavailable",
        }
    if not conflict_zone.registry_checksum:
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_checksum_unavailable",
        }
    if SHA256_HEX_RE.fullmatch(str(conflict_zone.registry_checksum)) is None:
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_checksum_invalid",
        }
    if _vector2(conflict_zone.center) is None:
        return {"status": "unavailable", "reason": "registered_conflict_zone_invalid"}
    if not math.isfinite(conflict_zone.radius_m) or conflict_zone.radius_m < 0:
        return {"status": "unavailable", "reason": "registered_conflict_zone_invalid"}
    geometry = {
        "type": "circle",
        "center": list(conflict_zone.center),
        "radius_m": conflict_zone.radius_m,
    }
    geometry_checksum = _geometry_checksum(geometry)
    if conflict_zone.registry_checksum != geometry_checksum:
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_checksum_geometry_mismatch",
            "geometry_checksum": geometry_checksum,
        }
    return {
        "status": "available",
        "reason": "registered_circular_conflict_zone",
        "zone_id": conflict_zone.zone_id,
        "provenance_id": conflict_zone.provenance_id,
        "registry_checksum": conflict_zone.registry_checksum,
        "coordinate_frame": "world",
        "geometry": geometry,
    }


def _geometry_checksum(geometry: Mapping[str, Any]) -> str:
    return _json_sha256_digest(geometry)


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
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
) -> dict[str, Any]:
    declared = _declared_encounter(
        trace,
        encounter_report=encounter_report,
        encounter_report_input_checksum=encounter_report_input_checksum,
    )
    if encounter_report is not None and declared.get("status") == "unavailable":
        return {
            "status": "unavailable",
            "reason": str(declared.get("reason", "canonical_encounter_unavailable")),
            "declared_encounter": declared,
        }
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
            "source": "canonical_near_miss_encounter_report"
            if declared.get("schema_version") == CANONICAL_ENCOUNTER_SCHEMA_VERSION
            else "planner_actor_hint",
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


def _declared_encounter(
    trace: SimulationTraceExport,
    *,
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
) -> dict[str, Any]:
    if encounter_report is not None:
        return _select_canonical_encounter(
            trace,
            encounter_report,
            expected_input_checksum=encounter_report_input_checksum,
        )
    for frame in trace.frames:
        for key in ("focal_encounter", "encounter"):
            value = frame.planner.get(key)
            if isinstance(value, Mapping):
                actor_id = value.get("actor_id") or value.get("pedestrian_id")
                if actor_id is not None:
                    return {
                        "actor_id": str(actor_id),
                        "encounter_id": value.get("encounter_id"),
                        "schema_version": "planner_actor_hint.v1",
                        "source": f"planner.{key}",
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
                            "schema_version": "planner_actor_hint.v1",
                            "source": "planner.encounters",
                        }
    return {}


def _select_canonical_encounter(
    trace: SimulationTraceExport,
    encounter_report: Mapping[str, Any],
    *,
    expected_input_checksum: str | None,
) -> dict[str, Any]:
    checksum_status = _encounter_report_checksum_status(
        encounter_report,
        expected_input_checksum=expected_input_checksum,
    )
    if checksum_status["status"] != "available":
        return {
            "status": "unavailable",
            "reason": checksum_status["reason"],
            "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
            "checksum_binding": checksum_status,
        }
    encounters = encounter_report.get("encounters")
    if not isinstance(encounters, Sequence) or isinstance(encounters, str | bytes):
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_report_missing_encounters",
        }
    valid: list[Mapping[str, Any]] = [
        encounter
        for encounter in encounters
        if isinstance(encounter, Mapping)
        and encounter.get("schema_version") == CANONICAL_ENCOUNTER_SCHEMA_VERSION
    ]
    if not valid:
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_report_has_no_encounters",
            "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
        }
    actor_ids = {
        str(pedestrian["id"])
        for frame in trace.frames
        for pedestrian in frame.pedestrians
        if "id" in pedestrian
    }
    candidates = [encounter for encounter in valid if str(encounter["actor_id"]) in actor_ids]
    if not candidates:
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_actor_missing_from_trace",
            "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
        }
    selected = min(
        candidates,
        key=lambda encounter: (
            float(encounter["start_time_s"]),
            str(encounter["actor_id"]),
            str(encounter["encounter_id"]),
        ),
    )
    return {
        "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
        "actor_id": str(selected["actor_id"]),
        "encounter_id": str(selected["encounter_id"]),
        "canonical_record": {key: selected[key] for key in selected},
        "report_profile": dict(encounter_report["profile"]),
        "report_provenance": dict(encounter_report["provenance"]),
        "checksum_binding": checksum_status,
    }


def _encounter_report_checksum_status(
    encounter_report: Mapping[str, Any],
    *,
    expected_input_checksum: str | None,
) -> dict[str, Any]:
    if expected_input_checksum is None:
        return {
            "status": "unavailable",
            "reason": "input_trace_checksum_unavailable",
        }
    provenance = encounter_report.get("provenance")
    input_checksums = provenance.get("input_checksums") if isinstance(provenance, Mapping) else None
    if not isinstance(input_checksums, Mapping):
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_input_checksums_unavailable",
        }
    checksums: dict[str, str] = {}
    for name, checksum in sorted(input_checksums.items()):
        name_text = str(name).strip()
        checksum_text = str(checksum).strip()
        if not name_text or SHA256_HEX_RE.fullmatch(checksum_text) is None:
            return {
                "status": "unavailable",
                "reason": "canonical_encounter_input_checksum_invalid",
            }
        checksums[name_text] = checksum_text
    declared_digest = (
        provenance.get("input_checksum_digest") if isinstance(provenance, Mapping) else None
    )
    expected_digest = _json_sha256_digest(checksums)
    if declared_digest != expected_digest:
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_input_checksum_digest_mismatch",
            "expected_input_checksum_digest": expected_digest,
        }
    checksum_values = set(checksums.values())
    if expected_input_checksum not in checksum_values:
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_input_checksum_mismatch",
            "expected_input_checksum": expected_input_checksum,
        }
    return {
        "status": "available",
        "reason": "canonical_encounter_input_checksum_matched",
        "input_checksum": expected_input_checksum,
        "input_checksum_digest": expected_digest,
    }


def _json_sha256_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _process_frame(
    frame: SimulationTraceFrame,
    *,
    frame_index: int,
    focal_actor_id: object,
    focal_encounter: Mapping[str, Any],
    route: RouteSpec | None,
    conflict_zone: ConflictZoneSpec | None,
    source_coordinate_frame: str,
) -> dict[str, Any]:
    robot_pos = _vector2(frame.robot.get("position"))
    robot_vel = _vector2(frame.robot.get("velocity"))
    nearest = _nearest_actor(frame, robot_pos=robot_pos)
    in_focal_interval = _frame_in_focal_encounter(frame, focal_encounter)
    focal = _pedestrian_by_id(frame, focal_actor_id) if in_focal_interval else None
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
        "encounter_interval": {
            "status": "in_interval" if in_focal_interval else "outside_interval",
            "reason": "canonical_encounter_interval"
            if _has_canonical_encounter_interval(focal_encounter)
            else "trace_wide_interval",
        },
        "world": {
            "status": "available"
            if source_coordinate_frame == "world" and robot_pos is not None
            else "unavailable",
            "reason": _world_frame_reason(source_coordinate_frame, robot_pos),
            "robot": _world_actor(frame.robot),
            "focal_actor": _world_actor(focal) if focal is not None else None,
        },
        "route": _route_frame(robot_pos, robot_vel, route, source_coordinate_frame),
        "conflict": _conflict_frame(robot_pos, focal, conflict_zone, source_coordinate_frame),
        "relative_interaction": focal_state,
        "global_minimum_actor": nearest,
        "commands": _command_state(frame),
    }


def _frame_in_focal_encounter(
    frame: SimulationTraceFrame,
    focal_encounter: Mapping[str, Any],
) -> bool:
    declared = focal_encounter.get("declared_encounter")
    record = focal_encounter.get("canonical_record")
    if record is None and isinstance(declared, Mapping):
        record = declared.get("canonical_record")
    if not isinstance(record, Mapping):
        return True
    return float(record["start_time_s"]) <= float(frame.time_s) <= float(record["end_time_s"])


def _has_canonical_encounter_interval(focal_encounter: Mapping[str, Any]) -> bool:
    declared = focal_encounter.get("declared_encounter")
    record = focal_encounter.get("canonical_record")
    if record is None and isinstance(declared, Mapping):
        record = declared.get("canonical_record")
    return isinstance(record, Mapping)


def _diagnostic_frames(frames: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    interval_frames = [
        frame
        for frame in frames
        if frame.get("encounter_interval", {}).get("status") != "outside_interval"
    ]
    return interval_frames


def _world_actor(actor: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if actor is None:
        return None
    heading = _finite_float(actor.get("heading"))
    return {
        "position": list(_vector2(actor.get("position")) or ()),
        "heading": heading,
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
    source_coordinate_frame: str,
) -> dict[str, Any]:
    if route is None:
        return {"status": "unavailable", "reason": "registered_route_unavailable"}
    if source_coordinate_frame != "world":
        return {
            "status": "unavailable",
            "reason": "source_coordinate_frame_not_world",
            "source_coordinate_frame": source_coordinate_frame,
        }
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
        "registry_checksum": route.registry_checksum,
        "geometry": {
            "type": "line_segment",
            "start": list(route.start),
            "end": list(route.end),
        },
        "s_m": s_m,
        "n_m": n_m,
        "progress_rate_mps": progress_rate,
    }


def _conflict_frame(
    robot_pos: tuple[float, float] | None,
    focal: Mapping[str, Any] | None,
    conflict_zone: ConflictZoneSpec | None,
    source_coordinate_frame: str,
) -> dict[str, Any]:
    if conflict_zone is None:
        return {"status": "unavailable", "reason": "registered_conflict_zone_unavailable"}
    if source_coordinate_frame != "world":
        return {
            "status": "unavailable",
            "reason": "source_coordinate_frame_not_world",
            "source_coordinate_frame": source_coordinate_frame,
        }
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
        "provenance_id": conflict_zone.provenance_id,
        "registry_checksum": conflict_zone.registry_checksum,
        "geometry": {
            "type": "circle",
            "center": list(conflict_zone.center),
            "radius_m": conflict_zone.radius_m,
        },
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
    heading = _finite_float(frame.robot.get("heading"))
    if heading is None:
        return {"status": "unavailable", "reason": "missing_or_nonfinite_robot_heading"}
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
    payload["relative_velocity_status"] = "available"
    payload["relative_velocity_reason"] = "relative_velocity_from_trace"
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
    commanded = _finite_mapping(selected)
    if commanded is None:
        return {"status": "unavailable", "reason": "selected_action_nonfinite"}
    executed = frame.planner.get("executed_action")
    executed_payload = _finite_mapping(executed) if isinstance(executed, Mapping) else None
    return {
        "status": "available",
        "commanded": commanded,
        "executed": executed_payload,
        "executed_status": "available" if executed_payload is not None else "unavailable",
    }


def _finite_mapping(value: Mapping[str, Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            result[str(key)] = item
        elif isinstance(item, int | float):
            if not math.isfinite(float(item)):
                return None
            result[str(key)] = float(item)
        elif isinstance(item, str) or item is None:
            result[str(key)] = item
        else:
            return None
    return result


def _diagnostics(frames: Sequence[Mapping[str, Any]], *, route_available: bool) -> dict[str, Any]:
    coverage = _coverage_summary(frames)
    clearances = [
        frame["relative_interaction"].get("proxy_surface_clearance_m")
        for frame in frames
        if frame["relative_interaction"].get("status") == "available"
        and frame["relative_interaction"].get("proxy_surface_clearance_status") == "available"
    ]
    threshold = _profiles()["threshold_profile"]["proxy_surface_clearance_threshold_m"]
    if coverage["proxy_surface_clearance"]["status"] == "partial":
        exposure = None
        deficit = None
    elif clearances:
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
        "minimum_proxy_surface_clearance_m": min(clearances)
        if clearances and coverage["proxy_surface_clearance"]["status"] == "complete"
        else None,
        "threshold_exposure": {
            "profile_version": THRESHOLD_PROFILE_VERSION,
            "threshold_m": threshold,
            "duration_s": exposure,
            "integrated_clearance_deficit_m_s": deficit,
            "status": "available" if clearances and exposure is not None else "unavailable",
            "reason": coverage["proxy_surface_clearance"].get("reason", "coverage_complete"),
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
        "coverage": coverage,
    }


def _coverage_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(frames)
    relative_available = [
        frame for frame in frames if frame["relative_interaction"].get("status") == "available"
    ]
    clearance_available = [
        frame
        for frame in relative_available
        if frame["relative_interaction"].get("proxy_surface_clearance_status") == "available"
    ]
    interval_missing = total - len(relative_available)
    radius_missing = len(relative_available) - len(clearance_available)
    complete = interval_missing == 0 and radius_missing == 0
    return {
        "frame_count": total,
        "relative_interaction": {
            "status": "complete" if interval_missing == 0 else "partial",
            "frame_count": total,
            "available_frame_count": len(relative_available),
            "missing_frame_count": interval_missing,
            "reason": "coverage_complete"
            if interval_missing == 0
            else "focal_actor_interval_missing",
        },
        "proxy_surface_clearance": {
            "status": "complete" if complete else "partial",
            "frame_count": total,
            "available_frame_count": len(clearance_available),
            "missing_frame_count": total - len(clearance_available),
            "missing_radius_frame_count": radius_missing,
            "missing_actor_interval_frame_count": interval_missing,
            "reason": "coverage_complete" if complete else "missing_radius_or_actor_interval",
        },
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
    if any(frame["conflict"].get("focal_actor_status") != "available" for frame in frames):
        return {
            "status": "unavailable",
            "reason": "focal_actor_conflict_interval_partial",
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
            absent_status=_absent_status_for_command_signal(frames, "linear_velocity"),
            zone_id=None,
        ),
        _event_from_condition(
            "first_material_turn_response",
            _first_turn_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["commands.commanded.angular_velocity"],
            absent_status=_absent_status_for_command_signal(frames, "angular_velocity"),
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
        _collision_event_anchor(trace, frames=frames, focal_actor_id=focal_actor_id),
        _event_from_condition(
            "first_safety_predicate_breach",
            _first_safety_predicate_breach_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["relative_interaction.proxy_surface_clearance_m"],
            absent_status=_absent_status_for_proxy_clearance(frames),
            zone_id=None,
        ),
        _event_from_condition(
            "proxy_overlap_event",
            _first_proxy_overlap_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["relative_interaction.proxy_surface_clearance_m"],
            absent_status=_absent_status_for_proxy_clearance(frames),
            zone_id=None,
        ),
        _event_from_condition(
            "sustained_stall_onset",
            _first_stall_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["robot.velocity"],
            absent_status=_absent_status_for_robot_velocity(frames),
            zone_id=None,
        ),
        _event_from_condition(
            "recovery_onset",
            _first_recovery_frame(frames),
            actor_id=focal_actor_id,
            source_fields=["robot.velocity"],
            absent_status=_absent_status_for_robot_velocity(frames),
            zone_id=None,
        ),
        _terminal_event_anchor(trace, frames=frames, focal_actor_id=focal_actor_id),
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
            "event_relative_time": {
                "status": "unavailable",
                "reason": f"event_{absent_status}",
            },
            "visual_anchor_eligibility": {
                "eligible": False,
                "reason": f"event_{absent_status}",
            },
        }
    event_relative = _event_relative_time(float(frame["time_s"]), float(frame["time_s"]))
    visual_eligible = event_type != "terminal_event"
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
        "event_relative_time": event_relative,
        "visual_anchor_eligibility": {
            "eligible": visual_eligible,
            "reason": "deterministic_trace_event"
            if visual_eligible
            else "terminal_event_requires_provenance",
        },
    }


def _event_relative_time(time_s: float, anchor_time_s: float) -> dict[str, Any]:
    return {
        "status": "available",
        "anchor_time_s": anchor_time_s,
        "tau_s": time_s - anchor_time_s,
    }


def _event_anchor_hierarchy(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fallback_order = [
        "exact_collision_event",
        "minimum_clearance",
        "first_safety_predicate_breach",
        "sustained_stall_onset",
        "terminal_event",
    ]
    available = {
        str(event["event_type"]): event for event in events if event.get("status") == "available"
    }
    ranked = [
        {
            "rank": rank,
            "event_type": event_type,
            "event_id": str(available[event_type]["event_id"]),
            "time_s": float(available[event_type]["time_s"]),
            "selection_role": "first_safety_predicate_breach"
            if event_type == "first_safety_predicate_breach"
            else "fallback_anchor",
        }
        for rank, event_type in enumerate(fallback_order)
        if event_type in available
    ]
    selected = ranked[0] if ranked else None
    return {
        "status": "available" if selected is not None else "unavailable",
        "fallback_order": fallback_order,
        "available_anchors": ranked,
        "selected_anchor": selected,
        "anchor_time_s": selected["time_s"] if selected is not None else None,
    }


def _frames_with_event_alignment(
    frames: Sequence[Mapping[str, Any]],
    hierarchy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = hierarchy.get("selected_anchor")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("time_s"), int | float):
        return [
            {
                **frame,
                "event_alignment": {"status": "unavailable", "reason": "no_available_anchor"},
            }
            for frame in frames
        ]
    anchor_time = float(selected["time_s"])
    return [
        {
            **frame,
            "event_alignment": {
                "status": "available",
                "anchor_event_id": selected["event_id"],
                "anchor_event_type": selected["event_type"],
                "anchor_time_s": anchor_time,
                "tau_s": float(frame["time_s"]) - anchor_time,
            },
        }
        for frame in frames
    ]


def _minimum_clearance_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if any(
        frame["relative_interaction"].get("status") != "available"
        or frame["relative_interaction"].get("proxy_surface_clearance_status") != "available"
        for frame in frames
    ):
        return None
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
    if _has_command_gap(frames, "linear_velocity"):
        return None
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
    if _has_command_gap(frames, "angular_velocity"):
        return None
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


def _has_command_gap(frames: Sequence[Mapping[str, Any]], key: str) -> bool:
    return any(
        not (
            isinstance(command := frame["commands"].get("commanded"), Mapping)
            and isinstance(command.get(key), int | float)
            and math.isfinite(float(command[key]))
        )
        for frame in frames
    )


def _absent_status_for_command_signal(frames: Sequence[Mapping[str, Any]], key: str) -> str:
    if _has_command_gap(frames, key):
        return "unavailable"
    return "not_observed" if _has_command_signal(frames, key) else "unavailable"


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
    return any(_canonical_collision_signal(frame.planner)["observed"] for frame in trace.frames)


def _has_proxy_clearance_signal(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float)
        for frame in frames
    )


def _has_proxy_clearance_gap(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        not (
            isinstance(frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float)
            and math.isfinite(float(frame["relative_interaction"]["proxy_surface_clearance_m"]))
            and frame["relative_interaction"].get("status") == "available"
            and frame["relative_interaction"].get("proxy_surface_clearance_status") == "available"
        )
        for frame in frames
    )


def _absent_status_for_proxy_clearance(frames: Sequence[Mapping[str, Any]]) -> str:
    if _has_proxy_clearance_gap(frames):
        return "unavailable"
    return "not_observed" if _has_proxy_clearance_signal(frames) else "unavailable"


def _has_robot_velocity_signal(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(_speed_from_frame(frame) is not None for frame in frames)


def _has_robot_velocity_gap(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(_speed_from_frame(frame) is None for frame in frames)


def _absent_status_for_robot_velocity(frames: Sequence[Mapping[str, Any]]) -> str:
    if _has_robot_velocity_gap(frames):
        return "unavailable"
    return "not_observed" if _has_robot_velocity_signal(frames) else "unavailable"


def _collision_event_anchor(
    trace: SimulationTraceExport,
    *,
    frames: Sequence[Mapping[str, Any]],
    focal_actor_id: object,
) -> dict[str, Any]:
    state = _collision_anchor_state(trace, frames, focal_actor_id=focal_actor_id)
    if state["status"] == "available":
        event = _event_from_condition(
            "exact_collision_event",
            state["frame"],
            actor_id=focal_actor_id,
            source_fields=["planner.event_ledger.collision_events"],
            absent_status="unavailable",
            zone_id=None,
        )
        event["time_s"] = float(state["collision_time"])
        event["collision_partner_id"] = state.get("collision_partner_id")
        event["collision_partner_type"] = state.get("collision_partner_type")
        event["event_relative_time"] = _event_relative_time(
            float(state["collision_time"]),
            float(state["collision_time"]),
        )
        return event
    event = _event_from_condition(
        "exact_collision_event",
        None,
        actor_id=focal_actor_id,
        source_fields=["planner.outcome.collision_event", "planner.event_ledger.collision_events"],
        absent_status="unavailable",
        zone_id=None,
    )
    if state["observed"]:
        event["reason"] = str(state.get("reason", "collision_observed_time_unavailable"))
        event["collision_observed"] = True
    return event


def _terminal_event_anchor(
    trace: SimulationTraceExport,
    *,
    frames: Sequence[Mapping[str, Any]],
    focal_actor_id: object,
) -> dict[str, Any]:
    return _event_from_condition(
        "terminal_event",
        None,
        actor_id=focal_actor_id,
        source_fields=["terminal_event_contract_unavailable"],
        absent_status="unavailable",
        zone_id=None,
    )


def _collision_anchor_state(
    trace: SimulationTraceExport,
    frames: Sequence[Mapping[str, Any]],
    *,
    focal_actor_id: object,
) -> dict[str, Any]:
    boolean_observed = False
    for trace_frame in trace.frames:
        signal = _canonical_collision_signal(trace_frame.planner)
        boolean_observed = boolean_observed or signal["observed"]
        collision_time = signal.get("collision_time")
        if isinstance(collision_time, int | float) and math.isfinite(float(collision_time)):
            if not _collision_binds_focal(signal, focal_actor_id):
                return {
                    "status": "unavailable",
                    "observed": True,
                    "reason": "collision_not_bound_to_focal_encounter",
                }
            frame = _frame_at_or_after_time(frames, float(collision_time))
            if frame is not None:
                return {
                    "status": "available",
                    "observed": True,
                    "frame": frame,
                    "collision_time": float(collision_time),
                    "collision_partner_id": signal.get("collision_partner_id"),
                    "collision_partner_type": signal.get("collision_partner_type"),
                }
            return {
                "status": "unavailable",
                "observed": True,
                "reason": "collision_time_outside_encounter_interval",
            }
    return {"status": "unavailable", "observed": boolean_observed}


def _collision_binds_focal(signal: Mapping[str, Any], focal_actor_id: object) -> bool:
    return (
        focal_actor_id is not None
        and signal.get("collision_partner_type") == "pedestrian"
        and signal.get("collision_partner_id") == str(focal_actor_id)
    )


def _frame_at_or_after_time(
    frames: Sequence[Mapping[str, Any]],
    collision_time: float,
) -> Mapping[str, Any] | None:
    if not frames:
        return None
    if collision_time < float(frames[0]["time_s"]) or collision_time > float(frames[-1]["time_s"]):
        return None
    return next(
        (frame for frame in frames if float(frame["time_s"]) >= collision_time),
        None,
    )


def _first_collision_frame(
    trace: SimulationTraceExport,
    frames: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for trace_frame, process_frame in zip(trace.frames, frames, strict=False):
        signal = _canonical_collision_signal(trace_frame.planner)
        if signal.get("collision_time") is not None:
            return process_frame
    return None


def _canonical_collision_signal(planner: Mapping[str, Any]) -> dict[str, Any]:
    ledger_signal = _ledger_collision_signal(planner.get("event_ledger"))
    if ledger_signal.get("collision_time") is not None or ledger_signal.get("source") == (
        "invalid_collision_event_record_shape"
    ):
        return ledger_signal
    outcome = planner.get("outcome")
    if isinstance(outcome, Mapping):
        collision = outcome.get("collision_event")
        if collision is True:
            return {"observed": True, "source": "outcome.collision_event"}
        if collision is False or collision is None:
            pass
        else:
            return {"observed": False, "source": "invalid_outcome_collision_shape"}
    if ledger_signal["observed"]:
        return ledger_signal
    return {"observed": False, "source": "no_canonical_collision_signal"}


def _ledger_collision_signal(ledger: object) -> dict[str, Any]:
    if not (
        isinstance(ledger, Mapping) and ledger.get("schema_version") == "EpisodeEventLedger.v2"
    ):
        return {"observed": False, "source": "event_ledger_unavailable"}
    records = ledger.get("collision_events")
    if not (isinstance(records, Sequence) and not isinstance(records, str | bytes)):
        return {"observed": False, "source": "event_ledger_collision_events_unavailable"}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if not _valid_collision_record(record):
            return {"observed": False, "source": "invalid_collision_event_record_shape"}
        collision_time = record.get("collision_time")
        return {
            "observed": True,
            "source": "event_ledger.collision_events",
            "collision_time": float(collision_time),
            "collision_partner_id": record["collision_partner_id"],
            "collision_partner_type": str(record["collision_partner_type"]),
        }
    return {"observed": bool(records), "source": "event_ledger.collision_events"}


def _valid_collision_record(record: Mapping[str, Any]) -> bool:
    allowed = {
        "collision_partner_type",
        "collision_partner_id",
        "collision_time",
        "relative_speed_at_contact",
        "clearance_series_source",
        "exact_event_source",
    }
    if set(record) != allowed:
        return False
    partner_id = record.get("collision_partner_id")
    relative_speed = record.get("relative_speed_at_contact")
    return (
        _finite_json_number(record.get("collision_time"))
        and isinstance(record.get("collision_partner_type"), str)
        and record["collision_partner_type"] in COLLISION_PARTNER_TYPES
        and (partner_id is None or isinstance(partner_id, str))
        and (partner_id is None or bool(partner_id.strip()))
        and (relative_speed is None or _finite_json_number(relative_speed))
        and isinstance(record.get("clearance_series_source"), str)
        and bool(record["clearance_series_source"].strip())
        and isinstance(record.get("exact_event_source"), str)
        and bool(record["exact_event_source"].strip())
    )


def _first_proxy_overlap_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if _has_proxy_clearance_gap(frames):
        return None
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


def _first_safety_predicate_breach_frame(
    frames: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if _has_proxy_clearance_gap(frames):
        return None
    threshold = _profiles()["threshold_profile"]["proxy_surface_clearance_threshold_m"]
    return next(
        (
            frame
            for frame in frames
            if isinstance(
                frame["relative_interaction"].get("proxy_surface_clearance_m"), int | float
            )
            and frame["relative_interaction"]["proxy_surface_clearance_m"] < threshold
        ),
        None,
    )


def _first_stall_frame(frames: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if _has_robot_velocity_gap(frames):
        return None
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
    canonical_record = declared.get("canonical_record") if isinstance(declared, Mapping) else None
    return {
        "status": "available",
        "actor_id": str(actor_id),
        "contiguous": not missing_steps,
        "missing_steps": missing_steps,
        "reason": "actor_present_all_frames" if not missing_steps else "actor_missing_within_trace",
        "computed_available_duration_s": _available_duration(available_frames)
        if not missing_steps
        else None,
        "computed_min_proxy_surface_clearance_m": min(clearances) if clearances else None,
        "canonical_encounter_record_status": "available"
        if isinstance(canonical_record, Mapping)
        else "unavailable",
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
    try:
        speed = math.hypot(float(velocity[0]), float(velocity[1]))
    except (TypeError, ValueError):
        return None
    return speed if math.isfinite(speed) else None


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


def _finite_float(value: Any) -> float | None:
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
