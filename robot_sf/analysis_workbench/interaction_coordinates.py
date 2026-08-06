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
    build_trace_run_config_contract,
    unavailable_pair_compatibility,
)
from robot_sf.analysis_workbench.process_trace_receipt import (
    build_simulation_trace_receipt,
    decode_simulation_trace_receipt,
    simulation_trace_receipt_sha256,
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
    simulation_trace_export_from_dict,
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
GEOMETRY_REGISTRY_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "process_trace_geometry_registry.v1.json"
)
GEOMETRY_OWNER_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "process_trace_geometry_owner.v1.json"
)
EVENT_PROFILE_VERSION = "worked_example_event_detectors.v1"
THRESHOLD_PROFILE_VERSION = "worked_example_threshold_profile.diagnostic.v1"
CANONICAL_ENCOUNTER_SCHEMA_VERSION = "near_miss_encounter.v1"
GEOMETRY_REGISTRY_SCHEMA_VERSION = "process_trace_geometry_registry.v1"
GEOMETRY_OWNER_SCHEMA_VERSION = "process_trace_geometry_owner.v1"
ENCOUNTER_REPORT_INPUT_SCHEMA_VERSION = "near_miss_encounter_report_input.v1"
ANALYSIS_INPUT_SCHEMA_VERSION = "worked_example_process_trace_analysis_input.v1"
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
COLLISION_PARTNER_TYPES = frozenset({"pedestrian", "static_geometry", "boundary", "goal_artifact"})
EXPECTED_EVENT_TYPES = [
    "minimum_clearance",
    "first_material_deceleration",
    "first_material_turn_response",
    "conflict_zone_entry",
    "exact_collision_event",
    "first_safety_predicate_breach",
    "proxy_overlap_event",
    "sustained_stall_onset",
    "recovery_onset",
    "terminal_event",
]
CLAIM_BOUNDARY = (
    "Diagnostic renderer-neutral process quantities derived from admitted trace fields. "
    "Not calibrated AMMV safety thresholds, collision probabilities, causal attribution, "
    "or replacement benchmark metrics."
)


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Route geometry resolved from a versioned external registry entry."""

    route_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    provenance_id: str | None = None
    registry_checksum: str | None = None
    geometry: Mapping[str, Any] | None = None
    registry_artifact_ref: str | None = None
    registry_path: str | None = None
    registry_content_sha256: str | None = None
    registry_entry_id: str | None = None
    registry_entry_sha256: str | None = None
    owner_artifact_ref: str | None = None
    owner_artifact_path: str | None = None


@dataclass(frozen=True, slots=True)
class ConflictZoneSpec:
    """Conflict zone resolved from a versioned external registry entry."""

    zone_id: str
    center: tuple[float, float]
    radius_m: float
    provenance_id: str | None = None
    registry_checksum: str | None = None
    geometry: Mapping[str, Any] | None = None
    registry_artifact_ref: str | None = None
    registry_path: str | None = None
    registry_content_sha256: str | None = None
    registry_entry_id: str | None = None
    registry_entry_sha256: str | None = None
    owner_artifact_ref: str | None = None
    owner_artifact_path: str | None = None


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


@lru_cache(maxsize=1)
def load_process_trace_geometry_registry_schema() -> dict[str, Any]:
    """Load the analysis-workbench-owned external geometry-registry schema.

    Returns:
        Parsed JSON Schema document.
    """

    return json.loads(GEOMETRY_REGISTRY_SCHEMA_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_process_trace_geometry_owner_schema() -> dict[str, Any]:
    """Load the public ``process_trace_geometry_owner.v1`` JSON schema.

    Returns:
        Parsed JSON Schema document.
    """

    return json.loads(GEOMETRY_OWNER_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_worked_example_process_trace(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
    geometry_registry_paths: Mapping[str, str | Path] | None = None,
    expected_artifact_sha256: str | None = None,
) -> None:
    """Validate a process trace payload against its versioned schema.

    ``geometry_registry_paths`` resolves stable registry and canonical owner
    artifact references to machine-local files. Absolute paths are deliberately
    validation context, never part of the public process-trace payload. Admission callers can pass
    an independently obtained ``expected_artifact_sha256`` over official writer bytes;
    a digest stored inside the payload would not authenticate a coherent rewrite.
    """

    errors: list[str] = []
    try:
        validator = Draft202012Validator(load_worked_example_process_trace_schema())
        errors.extend(
            f"{json_pointer(error.absolute_path)}: {error.message}"
            for error in sorted(
                validator.iter_errors(payload),
                key=lambda err: [str(part) for part in err.absolute_path],
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        errors.append("/: malformed JSON payload")
    if expected_artifact_sha256 is not None:
        if (
            not isinstance(expected_artifact_sha256, str)
            or SHA256_HEX_RE.fullmatch(expected_artifact_sha256) is None
        ):
            errors.append("/artifact_sha256: expected external sha256 hex digest")
        else:
            try:
                actual_artifact_sha256 = worked_example_process_trace_artifact_sha256(payload)
            except (TypeError, ValueError):
                actual_artifact_sha256 = None
            if actual_artifact_sha256 != expected_artifact_sha256:
                errors.append("/artifact_sha256: does not match external admission digest")
    try:
        errors.extend(
            _semantic_validation_errors(
                payload,
                geometry_registry_paths=geometry_registry_paths,
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        errors.append("/: malformed JSON payload")
    if errors:
        raise WorkedExampleProcessTraceValidationError(errors, source=source)


def _validate_analysis_input_identity(  # noqa: C901, PLR0912
    payload: Mapping[str, Any],
) -> list[str]:
    """Validate the content-addressed analysis-input receipt and public identity.

    Returns:
        Semantic validation errors for the construction receipt and identity.
    """

    contract = payload.get("analysis_input_contract")
    if not isinstance(contract, Mapping):
        return ["/analysis_input_contract: required"]
    required = {
        "schema_version",
        "source_trace_content_sha256",
        "route",
        "conflict",
        "pair_trace",
        "encounter_report",
        "focal_actor_id",
        "focal_encounter_id",
        "pair_comparison_grain",
    }
    errors = _require_keys(
        contract,
        "/analysis_input_contract",
        required=required,
        allowed=required,
    )
    if contract.get("schema_version") != ANALYSIS_INPUT_SCHEMA_VERSION:
        errors.append("/analysis_input_contract/schema_version: invalid")
    try:
        expected_digest = _json_sha256_digest(contract)
    except (TypeError, ValueError):
        errors.append("/analysis_input_contract: must be canonical strict JSON")
        return errors
    if payload.get("analysis_input_sha256") != expected_digest:
        errors.append("/analysis_input_sha256: must match analysis_input_contract digest")
    source_trace = payload.get("source_trace")
    trace_id = source_trace.get("trace_id") if isinstance(source_trace, Mapping) else None
    expected_id = (
        f"{trace_id}-process-trace-{expected_digest}" if isinstance(trace_id, str) else None
    )
    if payload.get("process_trace_id") != expected_id:
        errors.append("/process_trace_id: must bind the full analysis input digest")
    if isinstance(source_trace, Mapping) and contract.get(
        "source_trace_content_sha256"
    ) != source_trace.get("content_sha256"):
        errors.append(
            "/analysis_input_contract/source_trace_content_sha256: must match source_trace"
        )
    coordinate_frames = payload.get("coordinate_frames")
    if isinstance(coordinate_frames, Mapping):
        for contract_key, frame_key in (("route", "route"), ("conflict", "conflict")):
            frame = coordinate_frames.get(frame_key)
            receipt = frame.get("input_contract") if isinstance(frame, Mapping) else None
            if contract.get(contract_key) != receipt:
                errors.append(
                    f"/analysis_input_contract/{contract_key}: must match coordinate input receipt"
                )
    pair_receipt = contract.get("pair_trace")
    pair = payload.get("pair_compatibility")
    right_source = pair.get("right_source_trace") if isinstance(pair, Mapping) else None
    if isinstance(pair_receipt, Mapping):
        if pair_receipt.get("status") == "supplied":
            if not (
                isinstance(right_source, Mapping)
                and right_source.get("status") == "available"
                and pair_receipt.get("content_sha256") == right_source.get("content_sha256")
                and pair_receipt.get("content_receipt") == right_source.get("content_receipt")
            ):
                errors.append(
                    "/analysis_input_contract/pair_trace: must match embedded right source receipt"
                )
            pair_content = pair_receipt.get("content_receipt")
            try:
                pair_content_digest = simulation_trace_receipt_sha256(pair_content)
            except (TypeError, ValueError):
                pair_content_digest = None
            if pair_receipt.get("content_sha256") != pair_content_digest:
                errors.append(
                    "/analysis_input_contract/pair_trace/content_sha256: must match content"
                )
        elif pair_receipt != {"status": "not_supplied"}:
            errors.append("/analysis_input_contract/pair_trace: invalid absence receipt")
    report_receipt = contract.get("encounter_report")
    if isinstance(report_receipt, Mapping):
        if report_receipt.get("status") == "supplied":
            content = report_receipt.get("content_contract")
            try:
                content_digest = _json_sha256_digest(content)
            except (TypeError, ValueError):
                content_digest = None
            if report_receipt.get("content_sha256") != content_digest:
                errors.append(
                    "/analysis_input_contract/encounter_report/content_sha256: must match content"
                )
            if isinstance(content, Mapping):
                report_validator = Draft202012Validator(load_near_miss_encounter_schema())
                errors.extend(
                    "/analysis_input_contract/encounter_report/content_contract"
                    f"{json_pointer(error.absolute_path)}: {error.message}"
                    for error in sorted(
                        report_validator.iter_errors(content),
                        key=lambda item: list(item.absolute_path),
                    )
                )
        elif report_receipt != {"status": "not_supplied"}:
            errors.append("/analysis_input_contract/encounter_report: invalid absence receipt")
    return errors


def _validate_full_artifact_replay(
    payload: Mapping[str, Any],
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    """Rebuild the complete public artifact from source and bound input receipts.

    Returns:
        Field-level replay errors for any surface that differs from reconstruction.
    """

    contract = payload.get("analysis_input_contract")
    if not isinstance(contract, Mapping):
        return []
    trace = _trace_from_source_contract(payload.get("source_trace"))
    if trace is None:
        return []
    pair_receipt = contract.get("pair_trace")
    pair_trace: SimulationTraceExport | None = None
    if isinstance(pair_receipt, Mapping) and pair_receipt.get("status") == "supplied":
        pair_content = pair_receipt.get("content_receipt")
        if not isinstance(pair_content, Mapping):
            return ["/analysis_input_contract/pair_trace/content_receipt: required"]
        pair_trace = _trace_from_content_receipt(pair_content)
        if pair_trace is None:
            return [
                "/analysis_input_contract/pair_trace/content_receipt: invalid simulation trace receipt"
            ]
    report_receipt = contract.get("encounter_report")
    encounter_report: Mapping[str, Any] | None = None
    encounter_checksum: str | None = None
    if isinstance(report_receipt, Mapping) and report_receipt.get("status") == "supplied":
        report_content = report_receipt.get("content_contract")
        if not isinstance(report_content, Mapping):
            return ["/analysis_input_contract/encounter_report/content_contract: required"]
        encounter_report = report_content
        checksum_value = report_receipt.get("expected_input_checksum")
        encounter_checksum = checksum_value if isinstance(checksum_value, str) else None
    route = _route_spec_from_input_contract(
        contract.get("route"),
        geometry_registry_paths=geometry_registry_paths,
    )
    conflict_zone = _conflict_spec_from_input_contract(
        contract.get("conflict"),
        geometry_registry_paths=geometry_registry_paths,
    )
    try:
        expected = _build_worked_example_process_trace_from_export(
            trace,
            route=route,
            conflict_zone=conflict_zone,
            focal_actor_id=_optional_string(contract.get("focal_actor_id")),
            focal_encounter_id=_optional_string(contract.get("focal_encounter_id")),
            pair_trace=pair_trace,
            encounter_report=encounter_report,
            encounter_report_input_checksum=encounter_checksum,
            pair_comparison_grain=_optional_string(contract.get("pair_comparison_grain")),
        )
    except (KeyError, TypeError, ValueError, WorkedExampleProcessTraceValidationError):
        return ["/analysis_input_contract: cannot reconstruct canonical artifact"]
    return _replay_mismatch_errors(payload, expected, "")


def _semantic_validation_errors(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_analysis_input_identity(payload))
    if payload.get("profiles") != _profiles():
        errors.append("/profiles: must match exact versioned diagnostic profiles")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("/claim_boundary: must match conservative diagnostic claim boundary")
    source_trace = payload.get("source_trace")
    focal = (
        payload.get("encounters", {}).get("focal")
        if isinstance(payload.get("encounters"), Mapping)
        else None
    )
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
        errors.extend(
            _validate_coordinate_frame_contracts(
                payload.get("coordinate_frames"),
                frames,
                source_trace=source_trace,
                focal=focal,
                geometry_registry_paths=geometry_registry_paths,
            )
        )
        errors.extend(
            _validate_frame_replays(
                frames,
                geometry_registry_paths=geometry_registry_paths,
            )
        )
    if isinstance(source_trace, Mapping):
        errors.extend(_validate_source_trace_semantics(source_trace))
    if isinstance(source_trace, Mapping) and isinstance(frames, list):
        errors.extend(_validate_source_contract_frame_replays(source_trace, frames, focal))
    events = payload.get("event_anchors")
    if isinstance(events, list):
        errors.extend(
            _validate_event_inventory(events, frames, focal=focal, source_trace=source_trace)
        )
        for index, event in enumerate(events):
            if not isinstance(event, Mapping) or not event:
                errors.append(f"/event_anchors/{index}: expected non-empty event record")
                continue
            status = event.get("status")
            if status == "available":
                if not _finite_json_number(event.get("time_s")):
                    errors.append(
                        f"/event_anchors/{index}/time_s: required when status is available"
                    )
                if not _json_integer(event.get("step")):
                    errors.append(f"/event_anchors/{index}/step: required when status is available")
    hierarchy = payload.get("event_anchor_hierarchy")
    if isinstance(hierarchy, Mapping):
        errors.extend(_validate_event_anchor_hierarchy(hierarchy, events, frames))
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        errors.extend(_validate_diagnostics_record(diagnostics))
        errors.extend(_validate_diagnostics_replay(diagnostics, frames))
    encounters = payload.get("encounters")
    if isinstance(encounters, Mapping):
        errors.extend(
            _validate_global_minimum_series(encounters.get("global_minimum_over_all_actors"))
        )
        errors.extend(_validate_encounter_replays(encounters, frames))
    if isinstance(focal, Mapping):
        status = focal.get("status")
        if status not in {"available", "unavailable"}:
            errors.append("/encounters/focal/status: expected available or unavailable")
        if status == "available" and not focal.get("actor_id"):
            errors.append("/encounters/focal/actor_id: required when status is available")
        errors.extend(_validate_focal_actor_binding(focal, frames))
        declared = focal.get("declared_encounter")
        canonical_source = focal.get("source") == "canonical_near_miss_encounter_report"
        canonical_declared = (
            isinstance(declared, Mapping)
            and declared.get("schema_version") == CANONICAL_ENCOUNTER_SCHEMA_VERSION
        )
        if status == "available" and (canonical_source or canonical_declared):
            if not canonical_declared:
                errors.append(
                    "/encounters/focal/declared_encounter: canonical report contract required"
                )
            else:
                errors.extend(_validate_canonical_declared_encounter(declared, focal=focal))
    pair = payload.get("pair_compatibility")
    if isinstance(pair, Mapping):
        if pair.get("status") not in {"available", "unavailable", "incompatible"}:
            errors.append("/pair_compatibility/status: invalid status")
        errors.extend(
            _validate_pair_semantics(
                pair,
                events,
                source_trace,
                coordinate_frames=payload.get("coordinate_frames"),
                focal=focal,
                geometry_registry_paths=geometry_registry_paths,
            )
        )
        divergence = pair.get("divergence_interpretation")
        if isinstance(divergence, Mapping) and divergence.get("allowed") is True:
            shared_prefix = pair.get("shared_prefix")
            if not (
                isinstance(shared_prefix, Mapping) and shared_prefix.get("shared_prefix") is True
            ):
                errors.append(
                    "/pair_compatibility/divergence_interpretation/allowed: requires shared_prefix true"
                )
    errors.extend(
        _validate_full_artifact_replay(
            payload,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    return errors


def _validate_frame_record(frame: Mapping[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _require_keys(
            frame.get("source_coordinates"),
            f"/frames/{index}/source_coordinates",
            required={
                "coordinate_frame",
                "robot",
                "focal_actor",
                "focal_actor_id",
                "contextual_actors",
            },
            allowed={
                "coordinate_frame",
                "robot",
                "focal_actor",
                "focal_actor_id",
                "contextual_actors",
            },
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
    source = frame.get("source_coordinates")
    if isinstance(source, Mapping):
        focal_actor_id = source.get("focal_actor_id")
        if focal_actor_id is not None and not isinstance(focal_actor_id, str):
            errors.append(
                f"/frames/{index}/source_coordinates/focal_actor_id: expected string or null"
            )
        contextual_actors = source.get("contextual_actors")
        if not isinstance(contextual_actors, list):
            errors.append(f"/frames/{index}/source_coordinates/contextual_actors: expected array")
        else:
            seen_actor_ids: set[str] = set()
            for actor_index, actor in enumerate(contextual_actors):
                errors.extend(
                    _validate_source_actor_state(
                        actor,
                        f"/frames/{index}/source_coordinates/contextual_actors/{actor_index}",
                    )
                )
                actor_id = actor.get("actor_id") if isinstance(actor, Mapping) else None
                if isinstance(actor_id, str):
                    if actor_id in seen_actor_ids:
                        errors.append(
                            f"/frames/{index}/source_coordinates/contextual_actors/{actor_index}/actor_id: duplicate actor_id"
                        )
                    seen_actor_ids.add(actor_id)
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
    world = frame.get("world")
    world_robot = world.get("robot") if isinstance(world, Mapping) else None
    if (
        isinstance(world, Mapping)
        and world.get("status") == "available"
        and isinstance(world_robot, Mapping)
        and world_robot.get("position") == []
    ):
        errors.append(f"/frames/{index}/world/robot/position: required when world is available")
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


def _validate_source_actor_state(value: object, path: str) -> list[str]:
    errors = _require_keys(
        value,
        path,
        required={"actor_id", "position", "heading", "velocity", "radius_m"},
        allowed={"actor_id", "position", "heading", "velocity", "radius_m"},
    )
    if not isinstance(value, Mapping):
        return errors
    if not isinstance(value.get("actor_id"), str):
        errors.append(f"{path}/actor_id: expected string")
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


def _validate_coordinate_frame_contracts(  # noqa: C901
    coordinate_frames: object,
    frames: Sequence[object],
    *,
    source_trace: object,
    focal: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    if not isinstance(coordinate_frames, Mapping):
        return []
    errors: list[str] = []
    contracts = {
        "world": coordinate_frames.get("world"),
        "route": coordinate_frames.get("route"),
        "conflict": coordinate_frames.get("conflict"),
        "relative_interaction": coordinate_frames.get("relative_interaction"),
    }
    route_contract = contracts["route"]
    conflict_contract = contracts["conflict"]
    route_input = (
        route_contract.get("input_contract") if isinstance(route_contract, Mapping) else None
    )
    conflict_input = (
        conflict_contract.get("input_contract") if isinstance(conflict_contract, Mapping) else None
    )
    errors.extend(
        _validate_geometry_input_contract(
            route_input,
            kind="route",
            availability_contract=route_contract,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    errors.extend(
        _validate_geometry_input_contract(
            conflict_input,
            kind="conflict",
            availability_contract=conflict_contract,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        for key, contract in contracts.items():
            record = frame.get(key)
            if not isinstance(contract, Mapping) or not isinstance(record, Mapping):
                continue
            contract_status = contract.get("status")
            record_status = record.get("status")
            if contract_status == "unavailable" and record_status == "available":
                errors.append(
                    f"/coordinate_frames/{key}/status: unavailable contract cannot have available frames"
                )
        route = frame.get("route")
        if (
            isinstance(route_contract, Mapping)
            and route_contract.get("status") == "available"
            and isinstance(route, Mapping)
            and route.get("status") == "available"
        ):
            for key in ("route_id", "provenance_id", "registry_checksum", "registry", "geometry"):
                if route.get(key) != route_contract.get(key):
                    errors.append(
                        f"/frames/{index}/route/{key}: must match coordinate_frames.route"
                    )
        conflict = frame.get("conflict")
        if (
            isinstance(conflict_contract, Mapping)
            and conflict_contract.get("status") == "available"
            and isinstance(conflict, Mapping)
            and conflict.get("status") == "available"
        ):
            for key in ("zone_id", "provenance_id", "registry_checksum", "registry", "geometry"):
                if conflict.get(key) != conflict_contract.get(key):
                    errors.append(
                        f"/frames/{index}/conflict/{key}: must match coordinate_frames.conflict"
                    )
    trace = _trace_from_source_contract(source_trace)
    if trace is not None:
        expected_world = _world_availability(trace)
        errors.extend(
            _replay_mismatch_errors(contracts["world"], expected_world, "/coordinate_frames/world")
        )
        expected_relative = _relative_availability(focal if isinstance(focal, Mapping) else {})
        errors.extend(
            _replay_mismatch_errors(
                contracts["relative_interaction"],
                expected_relative,
                "/coordinate_frames/relative_interaction",
            )
        )
        route = _route_spec_from_input_contract(
            route_input,
            geometry_registry_paths=geometry_registry_paths,
        )
        expected_route = _route_availability(
            route,
            source_coordinate_frame=trace.coordinate_frame,
        )
        expected_route["input_contract"] = route_input
        if route_contract != expected_route:
            errors.append(
                "/coordinate_frames/route: must replay external geometry registry receipt and source frame"
            )
        errors.extend(
            _replay_mismatch_errors(
                route_contract,
                expected_route,
                "/coordinate_frames/route",
            )
        )
        conflict = _conflict_spec_from_input_contract(
            conflict_input,
            geometry_registry_paths=geometry_registry_paths,
        )
        expected_conflict = _conflict_availability(
            conflict,
            source_coordinate_frame=trace.coordinate_frame,
        )
        expected_conflict["input_contract"] = conflict_input
        if conflict_contract != expected_conflict:
            errors.append(
                "/coordinate_frames/conflict: must replay external geometry registry receipt and source frame"
            )
        errors.extend(
            _replay_mismatch_errors(
                conflict_contract,
                expected_conflict,
                "/coordinate_frames/conflict",
            )
        )
    errors.extend(
        _validate_frame_availability_replays(
            coordinate_frames,
            frames,
            source_trace=source_trace,
            focal=focal,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    return errors


def _validate_geometry_input_contract(
    value: object,
    *,
    kind: str,
    availability_contract: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    path = f"/coordinate_frames/{kind}/input_contract"
    if not isinstance(value, Mapping):
        return [f"{path}: required"]
    status = value.get("status")
    if status == "not_supplied":
        return _require_keys(value, path, required={"status"}, allowed={"status"})
    identity_key = "route_id" if kind == "route" else "zone_id"
    errors = _require_keys(
        value,
        path,
        required={
            "status",
            identity_key,
            "provenance_id",
            "registry_checksum",
            "geometry",
            "registry",
        },
        allowed={
            "status",
            identity_key,
            "provenance_id",
            "registry_checksum",
            "geometry",
            "registry",
        },
    )
    if status not in {"supplied", "supplied_unregistered"}:
        errors.append(f"{path}/status: expected supplied, supplied_unregistered, or not_supplied")
    registry = value.get("registry")
    errors.extend(
        _require_keys(
            registry,
            f"{path}/registry",
            required={"artifact_ref", "content_sha256", "entry_id", "entry_sha256"},
            allowed={"artifact_ref", "content_sha256", "entry_id", "entry_sha256"},
        )
    )
    if status == "supplied" and isinstance(registry, Mapping):
        for key in ("artifact_ref", "content_sha256", "entry_id", "entry_sha256"):
            if not isinstance(registry.get(key), str) or not registry[key]:
                errors.append(f"{path}/registry/{key}: required for supplied registry input")
    errors.extend(
        _validate_supplied_geometry_registry_receipt(
            value,
            kind=kind,
            availability_contract=availability_contract,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    return errors


def _validate_supplied_geometry_registry_receipt(
    value: Mapping[str, Any],
    *,
    kind: str,
    availability_contract: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    if value.get("status") != "supplied":
        return []
    registry = value.get("registry")
    geometry = value.get("geometry")
    if not isinstance(registry, Mapping) or not isinstance(geometry, Mapping):
        return []
    receipt_fields = (
        registry.get("artifact_ref"),
        registry.get("content_sha256"),
        registry.get("entry_id"),
        registry.get("entry_sha256"),
    )
    if not all(isinstance(field, str) and field for field in receipt_fields):
        return []
    artifact_ref = str(receipt_fields[0])
    resolved_path = _resolve_geometry_registry_path(
        artifact_ref,
        geometry_registry_paths=geometry_registry_paths,
    )
    identity_key = "route_id" if kind == "route" else "zone_id"
    collection = "routes" if kind == "route" else "conflict_zones"
    reason_prefix = "registered_route" if kind == "route" else "registered_conflict_zone"
    binding = _registry_binding(
        path=str(resolved_path) if resolved_path is not None else None,
        artifact_ref=artifact_ref,
        content_sha256=str(receipt_fields[1]),
        entry_id=str(receipt_fields[2]),
        entry_sha256=str(receipt_fields[3]),
        collection=collection,
        identity_key=identity_key,
        identity_value=str(value.get(identity_key, "")),
        provenance_id=_optional_string(value.get("provenance_id")),
        geometry=geometry,
        reason_prefix=reason_prefix,
        artifact_paths=geometry_registry_paths,
    )
    if binding.get("status") != "available":
        if (
            isinstance(availability_contract, Mapping)
            and availability_contract.get("status") == "unavailable"
            and availability_contract.get("reason") == binding.get("reason")
        ):
            return []
        if "_owner_" in str(binding.get("reason", "")):
            return [
                f"/coordinate_frames/{kind}/input_contract: canonical owner artifact must resolve and replay exact geometry"
            ]
        return [
            f"/coordinate_frames/{kind}/input_contract: must replay external geometry registry receipt"
        ]
    return []


def _validate_frame_availability_replays(
    coordinate_frames: Mapping[str, Any],
    frames: Sequence[object],
    *,
    source_trace: object,
    focal: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    """Replay coordinate availability from canonical source and registered contracts.

    Returns:
        Semantic validation errors for non-replayable frame availability.
    """

    trace = _trace_from_source_contract(source_trace)
    if trace is None:
        return []
    focal_record = focal if isinstance(focal, Mapping) else {}
    route_contract = coordinate_frames.get("route")
    route = _route_spec_from_input_contract(
        route_contract.get("input_contract") if isinstance(route_contract, Mapping) else None,
        geometry_registry_paths=geometry_registry_paths,
    )
    conflict_contract = coordinate_frames.get("conflict")
    conflict_zone = _conflict_spec_from_input_contract(
        conflict_contract.get("input_contract") if isinstance(conflict_contract, Mapping) else None,
        geometry_registry_paths=geometry_registry_paths,
    )
    errors: list[str] = []
    for index, (source_frame, frame) in enumerate(zip(trace.frames, frames, strict=False)):
        if not isinstance(frame, Mapping):
            continue
        expected = _process_frame(
            source_frame,
            frame_index=index,
            focal_actor_id=focal_record.get("actor_id"),
            focal_encounter=focal_record,
            route=route,
            conflict_zone=conflict_zone,
            source_coordinate_frame=trace.coordinate_frame,
        )
        for key in ("world", "route", "conflict", "relative_interaction"):
            actual_record = frame.get(key)
            expected_record = expected[key]
            if not isinstance(actual_record, Mapping):
                continue
            if actual_record.get("status") != expected_record.get("status"):
                errors.append(
                    f"/frames/{index}/{key}/status: must replay source content and coordinate contract"
                )
    return errors


def _validate_frame_replays(
    frames: Sequence[object],
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    errors: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        source = frame.get("source_coordinates")
        if not isinstance(source, Mapping):
            continue
        robot = source.get("robot")
        focal = source.get("focal_actor")
        robot_pos = _vector2(robot.get("position")) if isinstance(robot, Mapping) else None
        robot_vel = _vector2(robot.get("velocity")) if isinstance(robot, Mapping) else None
        focal_actor = _source_focal_actor_for_replay(frame, focal)
        errors.extend(_validate_world_replay(frame, index, source, robot, focal))
        errors.extend(
            _validate_route_replay(
                frame,
                index,
                source,
                robot_pos,
                robot_vel,
                focal_actor,
                geometry_registry_paths=geometry_registry_paths,
            )
        )
        errors.extend(
            _validate_conflict_replay(
                frame,
                index,
                source,
                robot_pos,
                focal_actor,
                geometry_registry_paths=geometry_registry_paths,
            )
        )
        errors.extend(
            _validate_relative_replay(frame, index, source, robot_pos, robot_vel, focal_actor)
        )
        errors.extend(_validate_global_minimum_actor_replay(frame, index))
    return errors


def _source_focal_actor_for_replay(
    frame: Mapping[str, Any],
    focal: object,
) -> dict[str, Any] | None:
    if not isinstance(focal, Mapping):
        return None
    source = frame.get("source_coordinates")
    actor_id = source.get("focal_actor_id") if isinstance(source, Mapping) else None
    return {
        "id": actor_id,
        "position": focal.get("position"),
        "heading": focal.get("heading"),
        "velocity": focal.get("velocity"),
        "radius": focal.get("radius_m"),
    }


def _validate_world_replay(
    frame: Mapping[str, Any],
    index: int,
    source: Mapping[str, Any],
    robot: object,
    focal: object,
) -> list[str]:
    world = frame.get("world")
    if not isinstance(world, Mapping) or world.get("status") != "available":
        return []
    if source.get("coordinate_frame") != "world":
        return [f"/frames/{index}/world/status: source coordinate frame is not world"]
    if world.get("robot") != robot:
        return [f"/frames/{index}/world/robot: must replay source robot"]
    if world.get("focal_actor") != focal:
        return [f"/frames/{index}/world/focal_actor: must replay source focal actor"]
    return []


def _validate_route_replay(
    frame: Mapping[str, Any],
    index: int,
    source: Mapping[str, Any],
    robot_pos: tuple[float, float] | None,
    robot_vel: tuple[float, float] | None,
    focal_actor: Mapping[str, Any] | None,
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    route_record = frame.get("route")
    if not isinstance(route_record, Mapping) or route_record.get("status") != "available":
        return []
    route = _route_spec_from_frame(
        route_record,
        geometry_registry_paths=geometry_registry_paths,
    )
    if route is None:
        return [f"/frames/{index}/route: cannot replay route geometry"]
    expected = _route_frame(
        robot_pos,
        robot_vel,
        focal_actor,
        route,
        str(source.get("coordinate_frame")),
    )
    errors: list[str] = []
    for key in ("s_m", "n_m", "progress_rate_mps"):
        if route_record.get(key) != expected.get(key):
            errors.append(f"/frames/{index}/route/{key}: must replay source coordinates")
    for key in (
        "focal_actor_status",
        "focal_actor_s_m",
        "focal_actor_n_m",
        "focal_actor_progress_rate_mps",
    ):
        if route_record.get(key) != expected.get(key):
            errors.append(f"/frames/{index}/route/{key}: must replay source focal actor")
    return errors


def _route_spec_from_frame(
    route_record: Mapping[str, Any],
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> RouteSpec | None:
    geometry = route_record.get("geometry")
    registry = route_record.get("registry")
    if not isinstance(geometry, Mapping) or not isinstance(registry, Mapping):
        return None
    points = _route_geometry_points(geometry)
    if len(points) < 2:
        return None
    artifact_ref = registry.get("artifact_ref")
    if not isinstance(artifact_ref, str):
        return None
    registry_path = _resolve_geometry_registry_path(
        artifact_ref,
        geometry_registry_paths=geometry_registry_paths,
    )
    entry_id = str(registry.get("entry_id"))
    owner_ref, owner_path = _registry_entry_owner_from_path(
        registry_path,
        collection="routes",
        entry_id=entry_id,
        artifact_paths=geometry_registry_paths,
    )
    return RouteSpec(
        str(route_record.get("route_id")),
        points[0],
        points[-1],
        str(route_record.get("provenance_id")),
        str(route_record.get("registry_checksum")),
        geometry=dict(geometry),
        registry_artifact_ref=artifact_ref,
        registry_path=str(registry_path) if registry_path is not None else None,
        registry_content_sha256=str(registry.get("content_sha256")),
        registry_entry_id=entry_id,
        registry_entry_sha256=str(registry.get("entry_sha256")),
        owner_artifact_ref=owner_ref,
        owner_artifact_path=str(owner_path) if owner_path is not None else None,
    )


def _validate_conflict_replay(
    frame: Mapping[str, Any],
    index: int,
    source: Mapping[str, Any],
    robot_pos: tuple[float, float] | None,
    focal_actor: Mapping[str, Any] | None,
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    conflict_record = frame.get("conflict")
    if not isinstance(conflict_record, Mapping) or conflict_record.get("status") != "available":
        return []
    conflict = _conflict_spec_from_frame(
        conflict_record,
        geometry_registry_paths=geometry_registry_paths,
    )
    if conflict is None:
        return [f"/frames/{index}/conflict: cannot replay conflict geometry"]
    expected = _conflict_frame(
        robot_pos,
        focal_actor,
        conflict,
        str(source.get("coordinate_frame")),
    )
    errors: list[str] = []
    for key in (
        "robot_signed_distance_to_zone_m",
        "focal_actor_status",
        "focal_actor_signed_distance_to_zone_m",
    ):
        if conflict_record.get(key) != expected.get(key):
            errors.append(f"/frames/{index}/conflict/{key}: must replay source coordinates")
    return errors


def _conflict_spec_from_frame(
    conflict_record: Mapping[str, Any],
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> ConflictZoneSpec | None:
    geometry = conflict_record.get("geometry")
    registry = conflict_record.get("registry")
    if not isinstance(geometry, Mapping) or not isinstance(registry, Mapping):
        return None
    center = _vector2(geometry.get("center"))
    radius = geometry.get("radius_m")
    if center is None or not _finite_json_number(radius):
        return None
    artifact_ref = registry.get("artifact_ref")
    if not isinstance(artifact_ref, str):
        return None
    registry_path = _resolve_geometry_registry_path(
        artifact_ref,
        geometry_registry_paths=geometry_registry_paths,
    )
    entry_id = str(registry.get("entry_id"))
    owner_ref, owner_path = _registry_entry_owner_from_path(
        registry_path,
        collection="conflict_zones",
        entry_id=entry_id,
        artifact_paths=geometry_registry_paths,
    )
    return ConflictZoneSpec(
        str(conflict_record.get("zone_id")),
        center,
        float(radius),
        str(conflict_record.get("provenance_id")),
        str(conflict_record.get("registry_checksum")),
        geometry=dict(geometry),
        registry_artifact_ref=artifact_ref,
        registry_path=str(registry_path) if registry_path is not None else None,
        registry_content_sha256=str(registry.get("content_sha256")),
        registry_entry_id=entry_id,
        registry_entry_sha256=str(registry.get("entry_sha256")),
        owner_artifact_ref=owner_ref,
        owner_artifact_path=str(owner_path) if owner_path is not None else None,
    )


def _validate_relative_replay(
    frame: Mapping[str, Any],
    index: int,
    source: Mapping[str, Any],
    robot_pos: tuple[float, float] | None,
    robot_vel: tuple[float, float] | None,
    focal_actor: Mapping[str, Any] | None,
) -> list[str]:
    relative = frame.get("relative_interaction")
    if not isinstance(relative, Mapping) or relative.get("status") != "available":
        return []
    expected = _relative_state(
        _replay_frame_robot(source),
        focal=focal_actor,
        robot_pos=robot_pos,
        robot_vel=robot_vel,
    )
    errors: list[str] = []
    for key in (
        "relative_longitudinal_m",
        "relative_lateral_m",
        "actor_id",
        "center_distance_m",
        "proxy_surface_clearance_m",
        "proxy_surface_clearance_status",
        "closest_approach",
    ):
        if relative.get(key) != expected.get(key):
            errors.append(
                f"/frames/{index}/relative_interaction/{key}: must replay source coordinates"
            )
    return errors


def _validate_global_minimum_actor_replay(frame: Mapping[str, Any], index: int) -> list[str]:
    expected = _global_minimum_actor_from_source(frame)
    if frame.get("global_minimum_actor") != expected:
        return [f"/frames/{index}/global_minimum_actor: must replay source actor inventory"]
    return []


def _global_minimum_actor_from_source(frame: Mapping[str, Any]) -> dict[str, Any]:
    source = frame.get("source_coordinates")
    robot = source.get("robot") if isinstance(source, Mapping) else None
    robot_pos = _vector2(robot.get("position")) if isinstance(robot, Mapping) else None
    if robot_pos is None:
        return {"status": "unavailable", "reason": "missing_robot_position"}
    actors = source.get("contextual_actors") if isinstance(source, Mapping) else None
    if not isinstance(actors, list) or not actors:
        return {"status": "unavailable", "reason": "no_pedestrians_in_frame"}
    candidates: list[tuple[float, str]] = []
    for actor in actors:
        if not isinstance(actor, Mapping) or not isinstance(actor.get("actor_id"), str):
            continue
        actor_pos = _vector2(actor.get("position"))
        if actor_pos is not None:
            candidates.append((_distance(robot_pos, actor_pos), str(actor["actor_id"])))
    if not candidates:
        return {"status": "unavailable", "reason": "missing_pedestrian_position"}
    center_distance, actor_id = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "status": "available",
        "actor_id": actor_id,
        "center_distance_m": center_distance,
    }


def _replay_frame_robot(source: Mapping[str, Any]) -> SimulationTraceFrame:
    robot = dict(source.get("robot")) if isinstance(source.get("robot"), Mapping) else {}
    if "radius_m" in robot:
        robot["radius"] = robot["radius_m"]
    return SimulationTraceFrame(step=0, time_s=0.0, robot=dict(robot), pedestrians=[], planner={})


def _validate_route_record(value: object, path: str) -> list[str]:  # noqa: C901
    required = (
        {"status", "reason"}
        if _status(value) == "unavailable"
        else {
            "status",
            "route_id",
            "provenance_id",
            "registry_checksum",
            "registry",
            "geometry",
            "s_m",
            "n_m",
            "progress_rate_mps",
            "focal_actor_status",
            "focal_actor_s_m",
            "focal_actor_n_m",
            "focal_actor_progress_rate_mps",
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
    for key in ("s_m", "n_m"):
        if not _finite_json_number(value.get(key)):
            errors.append(f"{path}/{key}: expected finite number")
    if not _finite_or_null(value.get("progress_rate_mps")):
        errors.append(f"{path}/progress_rate_mps: expected finite number or null")
    if value.get("focal_actor_status") not in {"available", "unavailable"}:
        errors.append(f"{path}/focal_actor_status: expected available or unavailable")
    if value.get("focal_actor_status") == "available":
        for key in ("focal_actor_s_m", "focal_actor_n_m"):
            if not _finite_json_number(value.get(key)):
                errors.append(f"{path}/{key}: expected finite number")
        if not _finite_or_null(value.get("focal_actor_progress_rate_mps")):
            errors.append(f"{path}/focal_actor_progress_rate_mps: expected finite number or null")
    else:
        if not isinstance(value.get("focal_actor_reason"), str):
            errors.append(f"{path}/focal_actor_reason: required when focal actor unavailable")
        for key in ("focal_actor_s_m", "focal_actor_n_m", "focal_actor_progress_rate_mps"):
            if value.get(key) is not None:
                errors.append(f"{path}/{key}: unavailable focal actor route requires null")
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
            "registry",
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


def _validate_geometry(value: object, path: str) -> list[str]:  # noqa: C901
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
    if geometry_type == "polyline":
        errors = _require_keys(
            value,
            path,
            required={"type", "points"},
            allowed={"type", "points"},
        )
        points = value.get("points")
        if not isinstance(points, list) or len(points) < 2:
            errors.append(f"{path}/points: expected at least two finite 2-vectors")
        elif any(not _finite_vector2(point) for point in points):
            errors.append(f"{path}/points: expected finite 2-vectors")
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


def _validate_source_trace_semantics(source_trace: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not (
        isinstance(source_trace.get("content_sha256"), str)
        and SHA256_HEX_RE.fullmatch(source_trace["content_sha256"])
    ):
        errors.append("/source_trace/content_sha256: expected sha256 hex digest")
    run_config = source_trace.get("run_config_contract")
    errors.extend(
        _require_keys(
            run_config,
            "/source_trace/run_config_contract",
            required={"status", "reason"}
            if _status(run_config) == "unavailable"
            else {"status", "time_step_s", "config_digest", "source"},
            allowed={
                "status",
                "reason",
                "time_step_s",
                "observed_time_step_s",
                "config_digest",
                "source",
            },
        )
    )
    if isinstance(run_config, Mapping) and run_config.get("status") == "available":
        if not (
            _finite_json_number(run_config.get("time_step_s"))
            and float(run_config["time_step_s"]) > 0.0
        ):
            errors.append("/source_trace/run_config_contract/time_step_s: expected positive finite")
        if not (
            isinstance(run_config.get("config_digest"), str)
            and SHA256_HEX_RE.fullmatch(run_config["config_digest"])
        ):
            errors.append("/source_trace/run_config_contract/config_digest: expected sha256 hex")
        if run_config.get("source") != "planner.run_config":
            errors.append("/source_trace/run_config_contract/source: expected planner.run_config")
    errors.extend(_validate_source_trace_content_receipt(source_trace, "/source_trace"))
    return errors


def _validate_source_trace_content_receipt(
    source_trace: Mapping[str, Any],
    path: str,
) -> list[str]:
    receipt = source_trace.get("content_receipt")
    if not isinstance(receipt, Mapping):
        return [f"{path}/content_receipt: required"]
    trace = _trace_from_content_receipt(receipt)
    if trace is None:
        return [f"{path}/content_receipt: invalid simulation trace content receipt"]
    try:
        expected_digest = simulation_trace_receipt_sha256(receipt)
    except (TypeError, ValueError):
        return [f"{path}/content_receipt: must be strict JSON"]
    if source_trace.get("content_sha256") != expected_digest:
        return [f"{path}/content_sha256: must match content_receipt digest"]
    contract = receipt.get("content_contract")
    if not isinstance(contract, Mapping):
        return [f"{path}/content_receipt/content_contract: required"]
    errors: list[str] = []
    errors.extend(
        error.replace(
            "source trace frame ", f"{path}/content_receipt/content_contract/frames/"
        ).replace(": duplicate pedestrian id ", "/pedestrians: duplicate pedestrian id ")
        for error in _duplicate_pedestrian_identity_errors(trace, label="source trace")
    )
    for key in ("schema_version", "trace_id", "coordinate_frame", "units", "source"):
        if source_trace.get(key) != contract.get(key):
            errors.append(f"{path}/{key}: must match content_receipt")
    if "run_config_contract" in source_trace and source_trace.get(
        "run_config_contract"
    ) != _run_config_contract(trace):
        errors.append(f"{path}/run_config_contract: must replay content_receipt")
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
                "status",
                "reason",
                "heading_reversal_count",
                "velocity_reversal_count",
            },
            allowed={
                "profile_version",
                "direction_semantics",
                "status",
                "reason",
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
        if not _json_integer(coverage.get("frame_count")) or coverage["frame_count"] < 0:
            errors.append("/diagnostics/coverage/frame_count: expected nonnegative integer")
        for key in ("relative_interaction", "proxy_surface_clearance"):
            errors.extend(
                _validate_coverage_record(coverage.get(key), f"/diagnostics/coverage/{key}")
            )
    threshold = diagnostics.get("threshold_exposure")
    if isinstance(threshold, Mapping):
        for key in ("threshold_m",):
            if not (_finite_json_number(threshold.get(key)) and float(threshold[key]) > 0.0):
                errors.append(
                    f"/diagnostics/threshold_exposure/{key}: expected positive finite number"
                )
        for key in ("duration_s", "integrated_clearance_deficit_m_s"):
            if not _finite_or_null(threshold.get(key)):
                errors.append(
                    f"/diagnostics/threshold_exposure/{key}: expected finite number or null"
                )
    reversal = diagnostics.get("reversal_counts")
    if isinstance(reversal, Mapping):
        if reversal.get("status") not in {"available", "unavailable"}:
            errors.append("/diagnostics/reversal_counts/status: expected available or unavailable")
        for key in ("heading_reversal_count", "velocity_reversal_count"):
            if reversal.get("status") == "available":
                if not _json_integer(reversal.get(key)) or reversal[key] < 0:
                    errors.append(
                        f"/diagnostics/reversal_counts/{key}: expected nonnegative integer"
                    )
            elif reversal.get(key) is not None:
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
    if value.get("status") not in {"complete", "partial", "unavailable"}:
        errors.append(f"{path}/status: expected complete, partial, or unavailable")
    frame_count = value.get("frame_count")
    available_count = value.get("available_frame_count")
    missing_count = value.get("missing_frame_count")
    if (
        _json_integer(frame_count)
        and _json_integer(available_count)
        and _json_integer(missing_count)
        and available_count + missing_count != frame_count
    ):
        errors.append(f"{path}/missing_frame_count: counts must add to frame_count")
    if value.get("status") == "complete" and value.get("missing_frame_count") != 0:
        errors.append(f"{path}/status: complete requires zero missing frames")
    if value.get("status") == "partial" and value.get("missing_frame_count") == 0:
        errors.append(f"{path}/status: partial requires missing frames")
    if value.get("status") == "unavailable" and not isinstance(value.get("reason"), str):
        errors.append(f"{path}/reason: required when unavailable")
    for key, item in value.items():
        if key.endswith("_count") and (not _json_integer(item) or item < 0):
            errors.append(f"{path}/{key}: expected nonnegative integer")
    return errors


def _validate_diagnostic_statuses(  # noqa: C901, PLR0912
    diagnostics: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    threshold = diagnostics.get("threshold_exposure")
    if isinstance(threshold, Mapping):
        duration = threshold.get("duration_s")
        deficit = threshold.get("integrated_clearance_deficit_m_s")
        if threshold.get("status") not in {"available", "unavailable"}:
            errors.append("/diagnostics/threshold_exposure/status: invalid status")
        if threshold.get("status") == "available" and not (
            _finite_json_number(duration) and _finite_json_number(deficit)
        ):
            errors.append("/diagnostics/threshold_exposure/status: available requires durations")
        if (
            threshold.get("status") == "available"
            and _finite_json_number(duration)
            and _finite_json_number(deficit)
            and (float(duration) < 0.0 or float(deficit) < 0.0)
        ):
            errors.append("/diagnostics/threshold_exposure/status: durations must be nonnegative")
        if threshold.get("status") == "unavailable" and (
            duration is not None or deficit is not None
        ):
            errors.append(
                "/diagnostics/threshold_exposure/status: unavailable requires null durations"
            )
    stall = diagnostics.get("stall")
    if isinstance(stall, Mapping):
        if not (
            _finite_json_number(stall.get("stall_min_duration_s"))
            and float(stall["stall_min_duration_s"]) > 0.0
        ):
            errors.append("/diagnostics/stall/stall_min_duration_s: expected positive finite")
        if stall.get("status") == "unavailable" and (
            stall.get("sustained_stall_duration_s") is not None
            or stall.get("sustained_stall_onset_step") is not None
        ):
            errors.append("/diagnostics/stall/status: unavailable requires null stall evidence")
        stall_duration = stall.get("sustained_stall_duration_s")
        if stall.get("status") == "available" and not _finite_json_number(stall_duration):
            errors.append("/diagnostics/stall/sustained_stall_duration_s: expected finite number")
        if (
            stall.get("status") == "available"
            and _finite_json_number(stall_duration)
            and float(stall_duration) < 0.0
        ):
            errors.append("/diagnostics/stall/sustained_stall_duration_s: expected nonnegative")
    conflict = diagnostics.get("conflict_zone_occupancy")
    if isinstance(conflict, Mapping):
        if conflict.get("status") == "available":
            for key in ("robot_duration_s", "focal_actor_duration_s"):
                if not _finite_json_number(conflict.get(key)):
                    errors.append(
                        f"/diagnostics/conflict_zone_occupancy/{key}: expected finite number"
                    )
                elif float(conflict[key]) < 0.0:
                    errors.append(
                        f"/diagnostics/conflict_zone_occupancy/{key}: expected nonnegative"
                    )
        if conflict.get("status") == "unavailable" and (
            conflict.get("robot_duration_s") is not None
            or conflict.get("focal_actor_duration_s") is not None
        ):
            errors.append(
                "/diagnostics/conflict_zone_occupancy/status: unavailable requires null durations"
            )
    route = diagnostics.get("route_progress")
    if isinstance(route, Mapping) and route.get("status") == "available":
        start = route.get("start_s_m")
        end = route.get("end_s_m")
        delta = route.get("delta_s_m")
        if (
            _finite_json_number(start)
            and _finite_json_number(end)
            and _finite_json_number(delta)
            and float(delta) != float(end) - float(start)
        ):
            errors.append("/diagnostics/route_progress/delta_s_m: must equal end minus start")
    return errors


def _validate_diagnostics_replay(diagnostics: Mapping[str, Any], frames: object) -> list[str]:
    if not isinstance(frames, list):
        return []
    process_frames = [frame for frame in frames if isinstance(frame, Mapping)]
    route_available = any(
        frame.get("route", {}).get("status") == "available" for frame in process_frames
    )
    try:
        expected = _diagnostics(_diagnostic_frames(process_frames), route_available=route_available)
    except (KeyError, TypeError, ValueError):
        return ["/diagnostics: cannot replay malformed frames"]
    errors: list[str] = []
    for key in (
        "minimum_proxy_surface_clearance_m",
        "threshold_exposure",
        "route_progress",
        "stall",
        "conflict_zone_occupancy",
        "reversal_counts",
        "coverage",
    ):
        if diagnostics.get(key) != expected.get(key):
            errors.append(f"/diagnostics/{key}: must replay frames")
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
            if not _json_integer(row.get("step")):
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
                    if isinstance(anchor, Mapping) and _json_integer(anchor.get("rank"))
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


def _validate_event_inventory(  # noqa: C901
    events: list[object],
    frames: object,
    *,
    focal: object,
    source_trace: object,
) -> list[str]:
    errors: list[str] = []
    actual_types = [
        event.get("event_type") if isinstance(event, Mapping) else None for event in events
    ]
    if actual_types != EXPECTED_EVENT_TYPES:
        errors.append("/event_anchors: must contain exact canonical event inventory")
    ids = [event.get("event_id") for event in events if isinstance(event, Mapping)]
    if len(ids) != len(set(ids)):
        errors.append("/event_anchors: event_id values must be unique")
    by_type = {event.get("event_type"): event for event in events if isinstance(event, Mapping)}
    focal_actor_id = focal.get("actor_id") if isinstance(focal, Mapping) else None
    frame_by_step = (
        {frame.get("step"): frame for frame in frames if isinstance(frame, Mapping)}
        if isinstance(frames, list)
        else {}
    )
    for event_type in EXPECTED_EVENT_TYPES:
        event = by_type.get(event_type)
        if not isinstance(event, Mapping):
            continue
        path = f"/event_anchors/{EXPECTED_EVENT_TYPES.index(event_type)}"
        errors.extend(_validate_event_record_semantics(event, path, frame_by_step, focal_actor_id))
    if isinstance(frames, list):
        errors.extend(
            _validate_event_replays(
                events,
                frames,
                focal_actor_id,
                source_trace=source_trace,
                focal_interval=_focal_interval_bounds(focal)
                if isinstance(focal, Mapping)
                else None,
            )
        )
    terminal = by_type.get("terminal_event")
    if isinstance(terminal, Mapping):
        if terminal.get("status") != "unavailable":
            errors.append("/event_anchors/9/status: terminal_event must remain unavailable")
        for key in ("time_s", "step"):
            if key in terminal:
                errors.append(f"/event_anchors/9/{key}: terminal_event must not carry time")
        if terminal.get("source_fields") != ["terminal_event_contract_unavailable"]:
            errors.append("/event_anchors/9/source_fields: terminal_event contract unavailable")
    return errors


def _validate_event_replays(
    events: Sequence[object],
    frames: Sequence[object],
    focal_actor_id: object,
    *,
    source_trace: object,
    focal_interval: tuple[float, float] | None,
) -> list[str]:
    process_frames = [frame for frame in frames if isinstance(frame, Mapping)]
    event_frames = _diagnostic_frames(process_frames)
    trace = _trace_from_source_contract(source_trace)
    try:
        if trace is None:
            return ["/source_trace/content_receipt: required for event replay"]
        expected = _event_anchors(
            trace,
            frames=event_frames,
            episode_frames=process_frames,
            focal_actor_id=focal_actor_id,
            focal_interval=focal_interval,
        )
    except (KeyError, TypeError, ValueError):
        return ["/event_anchors: cannot replay malformed frames"]
    errors: list[str] = []
    for index, expected_event in enumerate(expected):
        if expected_event is None:
            continue
        actual = events[index] if index < len(events) else None
        if isinstance(actual, Mapping) and dict(actual) != expected_event:
            errors.append(f"/event_anchors/{index}: must replay detector output")
    return errors


def _trace_from_source_contract(source_trace: object) -> SimulationTraceExport | None:
    if not isinstance(source_trace, Mapping):
        return None
    receipt = source_trace.get("content_receipt")
    if not isinstance(receipt, Mapping):
        return None
    return _trace_from_content_receipt(receipt)


def _trace_from_content_receipt(receipt: Mapping[str, Any]) -> SimulationTraceExport | None:
    try:
        restored = decode_simulation_trace_receipt(receipt)
        return simulation_trace_export_from_dict(restored)
    except (RobotSfError, TypeError, ValueError):
        return None


def _validate_source_contract_frame_replays(
    source_trace: Mapping[str, Any],
    frames: Sequence[object],
    focal: object,
) -> list[str]:
    trace = _trace_from_source_contract(source_trace)
    if trace is None:
        return []
    focal_actor_id = focal.get("actor_id") if isinstance(focal, Mapping) else None
    focal_encounter = focal if isinstance(focal, Mapping) else {}
    errors: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        if index >= len(trace.frames):
            errors.append(f"/frames/{index}: missing source content frame")
            continue
        source_frame = trace.frames[index]
        if frame.get("step") != source_frame.step or frame.get("time_s") != source_frame.time_s:
            errors.append(f"/frames/{index}: step/time must match source content frame")
            continue
        expected = _process_frame(
            source_frame,
            frame_index=index,
            focal_actor_id=focal_actor_id,
            focal_encounter=focal_encounter,
            route=None,
            conflict_zone=None,
            source_coordinate_frame=trace.coordinate_frame,
        )
        if frame.get("source_coordinates") != expected["source_coordinates"]:
            errors.append(f"/frames/{index}/source_coordinates: must replay source content")
        if frame.get("commands") != expected["commands"]:
            errors.append(f"/frames/{index}/commands: must replay source content")
    if len(frames) != len(trace.frames):
        errors.append("/frames: frame count must match source content")
    return errors


def _validate_event_record_semantics(
    event: Mapping[str, Any],
    path: str,
    frame_by_step: Mapping[object, Mapping[str, Any]],
    focal_actor_id: object,
) -> list[str]:
    errors: list[str] = []
    event_type = event.get("event_type")
    status = event.get("status")
    if status == "available":
        if (
            event_type != "exact_collision_event"
            and focal_actor_id is not None
            and event.get("actor_id") != str(focal_actor_id)
        ):
            errors.append(f"{path}/actor_id: must match focal actor")
        step = event.get("step")
        frame = frame_by_step.get(step)
        if not _json_integer(step) or frame is None:
            errors.append(f"{path}/step: must identify a process frame")
        elif event_type != "exact_collision_event" and event.get("time_s") != frame.get("time_s"):
            errors.append(f"{path}/time_s: must match event frame time")
        expected_id = (
            f"step-{int(step):04d}-{_slug(str(event_type))}" if _json_integer(step) else None
        )
        if expected_id is not None and event.get("event_id") != expected_id:
            errors.append(f"{path}/event_id: must match event type and step")
    else:
        for key in ("time_s", "step"):
            if key in event:
                errors.append(f"{path}/{key}: unavailable event must not carry time")
    if event_type == "exact_collision_event":
        errors.extend(_validate_collision_event_semantics(event, path))
    return errors


def _validate_collision_event_semantics(  # noqa: C901
    event: Mapping[str, Any], path: str
) -> list[str]:
    errors: list[str] = []
    if event.get("status") == "available":
        partner_type = event.get("collision_partner_type")
        partner_id = event.get("collision_partner_id")
        binding = event.get("focal_binding")
        if partner_type not in COLLISION_PARTNER_TYPES:
            errors.append(f"{path}/collision_partner_type: invalid canonical partner type")
        if not isinstance(binding, Mapping):
            errors.append(f"{path}/focal_binding: required for exact collision")
        elif binding.get("status") == "available":
            actor_id = binding.get("actor_id")
            if not isinstance(actor_id, str):
                errors.append(f"{path}/focal_binding/actor_id: required when available")
            if event.get("actor_id") != actor_id:
                errors.append(f"{path}/actor_id: must match available focal binding")
            if partner_type != "pedestrian" or partner_id != actor_id:
                errors.append(f"{path}/collision_partner_id: must match available focal binding")
        elif binding.get("status") == "unavailable":
            if event.get("actor_id") is not None:
                errors.append(
                    f"{path}/actor_id: unbound episode collision must not claim focal actor"
                )
            if not isinstance(binding.get("reason"), str):
                errors.append(f"{path}/focal_binding/reason: required when unavailable")
        else:
            errors.append(f"{path}/focal_binding/status: invalid status")
        if not _finite_json_number(event.get("time_s")):
            errors.append(f"{path}/time_s: expected finite collision time")
    else:
        for key in ("collision_partner_type", "collision_partner_id"):
            if key in event:
                errors.append(f"{path}/{key}: unavailable collision must not carry partner fields")
    return errors


def _validate_encounter_replays(encounters: Mapping[str, Any], frames: object) -> list[str]:
    if not isinstance(frames, list):
        return []
    errors: list[str] = []
    expected_global = _global_minimum_series(
        [frame for frame in frames if isinstance(frame, Mapping)]
    )
    if encounters.get("global_minimum_over_all_actors") != expected_global:
        errors.append("/encounters/global_minimum_over_all_actors: must replay frames")
    expected_switches = _actor_switch_events(
        [frame for frame in frames if isinstance(frame, Mapping)]
    )
    if encounters.get("actor_switch_events") != expected_switches:
        errors.append("/encounters/actor_switch_events: must replay global-minimum series")
    return errors


def _validate_pair_semantics(  # noqa: C901, PLR0912
    pair: Mapping[str, Any],
    events: object,
    source_trace: object,
    *,
    coordinate_frames: object,
    focal: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    errors: list[str] = []
    expected_right_inputs = _coordinate_input_contracts(coordinate_frames)
    if pair.get("right_coordinate_input_contract") != expected_right_inputs:
        errors.append(
            "/pair_compatibility/right_coordinate_input_contract: must match coordinate input contracts"
        )
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
        if pair.get("status") != "unavailable":
            for key in ("left_content_sha256", "right_content_sha256"):
                if not (
                    isinstance(provenance.get(key), str)
                    and SHA256_HEX_RE.fullmatch(provenance[key])
                ):
                    errors.append(f"/pair_compatibility/provenance_gate/{key}: expected sha256 hex")
            if isinstance(source_trace, Mapping) and provenance.get(
                "left_content_sha256"
            ) != source_trace.get("content_sha256"):
                errors.append(
                    "/pair_compatibility/provenance_gate/left_content_sha256: must match source trace"
                )
            right_source = pair.get("right_source_trace")
            if isinstance(right_source, Mapping) and right_source.get("status") == "available":
                errors.extend(
                    _validate_source_trace_content_receipt(
                        right_source,
                        "/pair_compatibility/right_source_trace",
                    )
                )
                if provenance.get("right_content_sha256") != right_source.get("content_sha256"):
                    errors.append(
                        "/pair_compatibility/provenance_gate/right_content_sha256: must match right source trace"
                    )
            elif pair.get("status") != "unavailable":
                errors.append(
                    "/pair_compatibility/right_source_trace: required for pair verification"
                )
        checks = provenance.get("checks")
        if isinstance(checks, Mapping):
            required = ["map_id_present", "horizon_present", "time_step_s_present"]
            if pair.get("comparison_grain", {}).get("grain_id") == "matched_realization_pair":
                required.append("config_digest_present")
            for key in required:
                if pair.get("status") == "available" and checks.get(key) is not True:
                    errors.append(f"/pair_compatibility/provenance_gate/checks/{key}: required")
    errors.extend(
        _validate_right_event_receipts(
            pair,
            coordinate_frames=coordinate_frames,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    errors.extend(_validate_common_event_anchor_semantics(pair, events))
    errors.extend(
        _validate_pair_replay(
            pair,
            source_trace=source_trace,
            coordinate_frames=coordinate_frames,
            focal=focal,
            geometry_registry_paths=geometry_registry_paths,
        )
    )
    return errors


def _validate_pair_replay(
    pair: Mapping[str, Any],
    *,
    source_trace: object,
    coordinate_frames: object,
    focal: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    """Replay every compatibility-bearing field from the two embedded traces.

    Returns:
        Semantic errors for compatibility fields that do not replay.
    """

    left = _trace_from_source_contract(source_trace)
    right = _trace_from_source_contract(pair.get("right_source_trace"))
    grain = pair.get("comparison_grain")
    grain_id = grain.get("grain_id") if isinstance(grain, Mapping) else None
    if left is None or right is None or not isinstance(grain_id, str):
        return []
    route, conflict_zone = _registered_coordinate_specs(
        coordinate_frames,
        geometry_registry_paths=geometry_registry_paths,
    )
    left_focal = focal if isinstance(focal, Mapping) else _resolve_focal_actor(left)
    right_focal = _resolve_focal_actor(right)
    expected = build_pair_compatibility_record(
        left,
        right,
        left_events=_replayed_trace_events(left, left_focal, route, conflict_zone),
        right_events=_replayed_trace_events(right, right_focal, route, conflict_zone),
        comparison_grain=grain_id,
    )
    expected["right_coordinate_input_contract"] = _coordinate_input_contracts(coordinate_frames)
    errors: list[str] = []
    for key in (
        "profile_version",
        "status",
        "comparison_grain",
        "provenance_gate",
        "right_source_trace",
        "initial_state_equivalence",
        "route_spawn_separation",
        "shared_prefix",
        "valid_common_event_anchors",
        "right_event_anchors",
        "duration_normalization",
        "divergence_interpretation",
        "right_coordinate_input_contract",
    ):
        errors.extend(
            _replay_mismatch_errors(
                pair.get(key),
                expected.get(key),
                f"/pair_compatibility/{key}",
            )
        )
    return errors


def _coordinate_input_contracts(coordinate_frames: object) -> dict[str, Any]:
    if not isinstance(coordinate_frames, Mapping):
        return {"route": None, "conflict": None}
    return {
        key: value.get("input_contract") if isinstance(value, Mapping) else None
        for key in ("route", "conflict")
        for value in (coordinate_frames.get(key),)
    }


def _registered_coordinate_specs(
    coordinate_frames: object,
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> tuple[RouteSpec | None, ConflictZoneSpec | None]:
    if not isinstance(coordinate_frames, Mapping):
        return None, None
    route_contract = coordinate_frames.get("route")
    route = _route_spec_from_input_contract(
        route_contract.get("input_contract") if isinstance(route_contract, Mapping) else None,
        geometry_registry_paths=geometry_registry_paths,
    )
    conflict_contract = coordinate_frames.get("conflict")
    conflict_zone = _conflict_spec_from_input_contract(
        conflict_contract.get("input_contract") if isinstance(conflict_contract, Mapping) else None,
        geometry_registry_paths=geometry_registry_paths,
    )
    return route, conflict_zone


def _replayed_trace_events(
    trace: SimulationTraceExport,
    focal: Mapping[str, Any],
    route: RouteSpec | None,
    conflict_zone: ConflictZoneSpec | None,
) -> list[dict[str, Any]]:
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
    return _event_anchors(
        trace,
        frames=_diagnostic_frames(frames),
        episode_frames=frames,
        focal_actor_id=focal.get("actor_id"),
        focal_interval=_focal_interval_bounds(focal),
    )


def _replay_mismatch_errors(actual: object, expected: object, path: str) -> list[str]:
    if type(actual) is not type(expected):
        return [f"{path}: must replay embedded source content"]
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        errors: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                errors.append(f"{path}/{key}: must replay embedded source content")
                continue
            errors.extend(_replay_mismatch_errors(actual[key], expected[key], f"{path}/{key}"))
        return errors
    if isinstance(actual, list) and isinstance(expected, list):
        errors = []
        for index in range(max(len(actual), len(expected))):
            item_path = f"{path}/{index}"
            if index >= len(actual) or index >= len(expected):
                errors.append(f"{item_path}: must replay embedded source content")
                continue
            errors.extend(_replay_mismatch_errors(actual[index], expected[index], item_path))
        return errors
    if actual != expected:
        return [f"{path}: must replay embedded source content"]
    return []


def _validate_right_event_receipts(  # noqa: C901
    pair: Mapping[str, Any],
    *,
    coordinate_frames: object,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> list[str]:
    receipts = pair.get("right_event_anchors")
    if not isinstance(receipts, list):
        return []
    errors: list[str] = []
    for index, receipt in enumerate(receipts):
        path = f"/pair_compatibility/right_event_anchors/{index}"
        if not isinstance(receipt, Mapping):
            continue
        if receipt.get("status") != "available":
            continue
        step = receipt.get("step")
        event_type = receipt.get("event_type")
        if not _json_integer(step):
            errors.append(f"{path}/step: required")
            continue
        expected_id = f"step-{step:04d}-{_slug(str(event_type))}"
        if receipt.get("event_id") != expected_id:
            errors.append(f"{path}/event_id: must match event type and step")
        expected_relative = (
            _event_relative_time(float(receipt.get("time_s")), float(receipt.get("time_s")))
            if _finite_json_number(receipt.get("time_s"))
            else None
        )
        if (
            expected_relative is not None
            and receipt.get("event_relative_time") != expected_relative
        ):
            errors.append(f"{path}/event_relative_time: must match receipt time")
        if receipt.get("confidence") != "deterministic_trace_rule":
            errors.append(f"{path}/confidence: must be deterministic_trace_rule")
        eligibility = receipt.get("visual_anchor_eligibility")
        if not (isinstance(eligibility, Mapping) and eligibility.get("eligible") is True):
            errors.append(f"{path}/visual_anchor_eligibility: available receipt must be eligible")
    route, conflict_zone = _registered_coordinate_specs(
        coordinate_frames,
        geometry_registry_paths=geometry_registry_paths,
    )
    expected = _right_event_receipts_from_source(
        pair,
        route=route,
        conflict_zone=conflict_zone,
    )
    if expected is not None and receipts != expected:
        errors.append("/pair_compatibility/right_event_anchors: must replay right source content")
    return errors


def _right_event_receipts_from_source(
    pair: Mapping[str, Any],
    *,
    route: RouteSpec | None,
    conflict_zone: ConflictZoneSpec | None,
) -> list[dict[str, Any]] | None:
    right_source = pair.get("right_source_trace")
    if not isinstance(right_source, Mapping) or right_source.get("status") != "available":
        return None
    trace = _trace_from_source_contract(right_source)
    if trace is None:
        return None
    focal = _resolve_focal_actor(trace)
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
    events = _event_anchors(
        trace,
        frames=_diagnostic_frames(frames),
        episode_frames=frames,
        focal_actor_id=focal.get("actor_id"),
        focal_interval=_focal_interval_bounds(focal),
    )
    return _event_receipts_for_validation(events)


def _event_receipts_for_validation(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    receipts = []
    for event in events:
        if event.get("status") != "available":
            continue
        receipt = {
            "event_id": event["event_id"],
            "event_type": str(event["event_type"]),
            "detector_profile_version": str(event["detector_profile_version"]),
            "time_s": float(event["time_s"]),
            "step": int(event["step"]),
            "confidence": str(event["confidence"]),
            "actor_id": event.get("actor_id"),
            "zone_id": event.get("zone_id"),
            "source_fields": [str(field) for field in event.get("source_fields", [])],
            "status": "available",
            "event_relative_time": dict(event["event_relative_time"]),
            "visual_anchor_eligibility": dict(event["visual_anchor_eligibility"]),
        }
        if event.get("event_type") == "exact_collision_event":
            receipt["collision_partner_type"] = event.get("collision_partner_type")
            receipt["collision_partner_id"] = event.get("collision_partner_id")
        receipts.append(receipt)
    return sorted(receipts, key=lambda item: str(item["event_id"]))


def _validate_common_event_anchor_semantics(pair: Mapping[str, Any], events: object) -> list[str]:
    if not isinstance(events, list):
        return []
    left_by_id = {
        event.get("event_id"): event
        for event in events
        if isinstance(event, Mapping) and event.get("status") == "available"
    }
    right_by_id = {
        event.get("event_id"): event
        for event in pair.get("right_event_anchors", [])
        if isinstance(event, Mapping) and event.get("status") == "available"
    }
    errors: list[str] = []
    anchors = pair.get("valid_common_event_anchors")
    if not isinstance(anchors, list):
        return errors
    for index, anchor in enumerate(anchors):
        path = f"/pair_compatibility/valid_common_event_anchors/{index}"
        if not isinstance(anchor, Mapping):
            continue
        left = left_by_id.get(anchor.get("left_event_id"))
        if not isinstance(left, Mapping):
            errors.append(f"{path}/left_event_id: must resolve to available left event")
            continue
        identity = _process_event_identity(left)
        anchor_identity = (
            str(anchor.get("event_type")),
            str(anchor.get("detector_profile_version")),
            anchor.get("actor_id"),
            anchor.get("zone_id"),
            tuple(str(field) for field in anchor.get("source_fields", [])),
            anchor.get("collision_partner_type"),
            anchor.get("collision_partner_id"),
        )
        if anchor_identity != identity:
            errors.append(f"{path}: identity must match resolved left event")
        right = right_by_id.get(anchor.get("right_event_id"))
        if not isinstance(anchor.get("right_event_id"), str) or not anchor["right_event_id"]:
            errors.append(f"{path}/right_event_id: required")
        elif not isinstance(right, Mapping):
            errors.append(f"{path}/right_event_id: must resolve to available right event")
        elif _process_event_identity(right) != identity:
            errors.append(f"{path}: right identity must match resolved left event")
    return errors


def _process_event_identity(
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


def _validate_canonical_declared_encounter(  # noqa: C901, PLR0912
    declared: Mapping[str, Any],
    *,
    focal: Mapping[str, Any],
) -> list[str]:
    base_path = "/encounters/focal/declared_encounter"
    record = declared.get("canonical_record")
    if not isinstance(record, Mapping):
        return [f"{base_path}/canonical_record: required"]
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
    errors = [f"{base_path}/canonical_record/{key}: unexpected field" for key in sorted(extra)]
    errors.extend(f"{base_path}/canonical_record/{key}: required" for key in sorted(missing))
    if record.get("schema_version") != CANONICAL_ENCOUNTER_SCHEMA_VERSION:
        errors.append(f"{base_path}/canonical_record/schema_version: invalid")
    for key in ("actor_id", "encounter_id"):
        expected = focal.get(key)
        if declared.get(key) != expected:
            errors.append(f"{base_path}/{key}: must match focal encounter")
        if record.get(key) != expected:
            errors.append(f"{base_path}/canonical_record/{key}: must match focal encounter")

    report_input = declared.get("report_input_contract")
    report_path = f"{base_path}/report_input_contract"
    errors.extend(
        _require_keys(
            report_input,
            report_path,
            required={
                "schema_version",
                "content_sha256",
                "content_contract",
                "selected_entry_index",
                "selected_entry_sha256",
            },
            allowed={
                "schema_version",
                "content_sha256",
                "content_contract",
                "selected_entry_index",
                "selected_entry_sha256",
            },
        )
    )
    if not isinstance(report_input, Mapping):
        return errors
    if report_input.get("schema_version") != ENCOUNTER_REPORT_INPUT_SCHEMA_VERSION:
        errors.append(f"{report_path}/schema_version: invalid")
    report_content = report_input.get("content_contract")
    if not isinstance(report_content, Mapping):
        errors.append(f"{report_path}/content_contract: required")
        return errors
    try:
        expected_report_digest = _json_sha256_digest(report_content)
    except (TypeError, ValueError):
        errors.append(f"{report_path}/content_contract: must be strict JSON")
        return errors
    if report_input.get("content_sha256") != expected_report_digest:
        errors.append(f"{report_path}/content_sha256: must match content_contract")
    report_validator = Draft202012Validator(load_near_miss_encounter_schema())
    errors.extend(
        f"{report_path}/content_contract{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            report_validator.iter_errors(report_content),
            key=lambda item: list(item.absolute_path),
        )
    )
    encounters = report_content.get("encounters")
    selected_index = report_input.get("selected_entry_index")
    if not (
        isinstance(encounters, list)
        and isinstance(selected_index, int)
        and not isinstance(selected_index, bool)
        and 0 <= selected_index < len(encounters)
        and isinstance(encounters[selected_index], Mapping)
    ):
        errors.append(f"{report_path}/selected_entry_index: must resolve one report encounter")
        return errors
    selected = encounters[selected_index]
    try:
        selected_digest = _json_sha256_digest(selected)
    except (TypeError, ValueError):
        errors.append(f"{report_path}/selected_entry_sha256: selected entry is not strict JSON")
        return errors
    if report_input.get("selected_entry_sha256") != selected_digest:
        errors.append(f"{report_path}/selected_entry_sha256: must match selected report entry")
    if record != selected:
        errors.append(f"{base_path}/canonical_record: must replay selected report entry")
    if declared.get("report_profile") != report_content.get("profile"):
        errors.append(f"{base_path}/report_profile: must replay report content")
    if declared.get("report_provenance") != report_content.get("provenance"):
        errors.append(f"{base_path}/report_provenance: must replay report content")
    checksum_binding = declared.get("checksum_binding")
    checksum = (
        checksum_binding.get("input_checksum") if isinstance(checksum_binding, Mapping) else None
    )
    if not isinstance(checksum, str) or checksum_binding != _encounter_report_checksum_status(
        report_content,
        expected_input_checksum=checksum,
    ):
        errors.append(f"{base_path}/checksum_binding: must replay report provenance")
    return errors


def _finite_json_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def load_registered_route_spec(
    path: Path,
    entry_id: str,
    *,
    geometry_owner_paths: Mapping[str, str | Path] | None = None,
) -> RouteSpec:
    """Resolve one route from a versioned external process-trace geometry registry.

    Returns:
        Route specification bound to the raw registry artifact and unique entry.
    """

    payload, content_sha256, resolved_path = _load_geometry_registry(path)
    entry = _unique_registry_entry(payload, "routes", entry_id, source=resolved_path)
    geometry = entry.get("geometry")
    if not isinstance(geometry, Mapping):
        geometry = {}
    points = _route_geometry_points(geometry)
    start = points[0] if points else (math.nan, math.nan)
    end = points[-1] if points else (math.nan, math.nan)
    owner_ref, owner_path = _resolve_registry_entry_owner(
        entry,
        artifact_paths=geometry_owner_paths,
    )
    return RouteSpec(
        route_id=str(entry.get("route_id", "")),
        start=start,
        end=end,
        provenance_id=str(payload["registry_id"]),
        registry_checksum=_geometry_checksum(geometry),
        geometry=dict(geometry),
        registry_artifact_ref=str(payload["artifact_ref"]),
        registry_path=str(resolved_path),
        registry_content_sha256=content_sha256,
        registry_entry_id=entry_id,
        registry_entry_sha256=_json_sha256_digest(entry),
        owner_artifact_ref=owner_ref,
        owner_artifact_path=str(owner_path) if owner_path is not None else None,
    )


def load_process_trace_geometry_owner(path: Path) -> dict[str, Any]:
    """Load and validate one strict canonical geometry-owner artifact.

    Returns:
        Validated geometry-owner payload.
    """

    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise WorkedExampleProcessTraceValidationError(
            ["expected readable process_trace_geometry_owner.v1 JSON"],
            source=resolved,
        ) from exc
    return _geometry_owner_payload_from_bytes(raw, source=resolved)


def _geometry_owner_payload_from_bytes(raw: bytes, *, source: Path) -> dict[str, Any]:
    try:
        payload = _strict_json_document(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkedExampleProcessTraceValidationError(
            ["/: expected strict process_trace_geometry_owner.v1 JSON"],
            source=source,
        ) from exc
    if not isinstance(payload, Mapping):
        raise WorkedExampleProcessTraceValidationError(
            ["/: expected process-trace geometry owner mapping"],
            source=source,
        )
    validator = Draft202012Validator(load_process_trace_geometry_owner_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
    ]
    if errors:
        raise WorkedExampleProcessTraceValidationError(errors, source=source)
    return dict(payload)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _strict_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number {value}")
    return number


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_document(raw: str | bytes | bytearray) -> Any:
    return json.loads(
        raw,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_strict_json_float,
        object_pairs_hook=_strict_json_object,
    )


def load_registered_conflict_zone_spec(
    path: Path,
    entry_id: str,
    *,
    geometry_owner_paths: Mapping[str, str | Path] | None = None,
) -> ConflictZoneSpec:
    """Resolve one conflict zone from a versioned external process-trace geometry registry.

    Returns:
        Conflict-zone specification bound to the raw registry artifact and unique entry.
    """

    payload, content_sha256, resolved_path = _load_geometry_registry(path)
    entry = _unique_registry_entry(payload, "conflict_zones", entry_id, source=resolved_path)
    geometry = entry.get("geometry")
    geometry_record = geometry if isinstance(geometry, Mapping) else {}
    center = _vector2(geometry_record.get("center")) or (math.nan, math.nan)
    radius = geometry_record.get("radius_m")
    radius_m = float(radius) if _finite_json_number(radius) else math.nan
    owner_ref, owner_path = _resolve_registry_entry_owner(
        entry,
        artifact_paths=geometry_owner_paths,
    )
    return ConflictZoneSpec(
        zone_id=str(entry.get("zone_id", "")),
        center=center,
        radius_m=radius_m,
        provenance_id=str(payload["registry_id"]),
        registry_checksum=_geometry_checksum(geometry_record),
        geometry=dict(geometry_record),
        registry_artifact_ref=str(payload["artifact_ref"]),
        registry_path=str(resolved_path),
        registry_content_sha256=content_sha256,
        registry_entry_id=entry_id,
        registry_entry_sha256=_json_sha256_digest(entry),
        owner_artifact_ref=owner_ref,
        owner_artifact_path=str(owner_path) if owner_path is not None else None,
    )


def _load_geometry_registry(path: Path) -> tuple[dict[str, Any], str, Path]:
    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
        payload = _strict_json_document(raw)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise WorkedExampleProcessTraceValidationError(
            ["expected readable process_trace_geometry_registry.v1 JSON"], source=resolved
        ) from exc
    if not isinstance(payload, Mapping):
        raise WorkedExampleProcessTraceValidationError(
            ["expected process-trace geometry registry mapping"], source=resolved
        )
    required = {
        "schema_version",
        "registry_id",
        "artifact_ref",
        "coordinate_frame",
        "routes",
        "conflict_zones",
    }
    if set(payload) != required:
        raise WorkedExampleProcessTraceValidationError(
            ["geometry registry must contain only the canonical top-level fields"], source=resolved
        )
    if (
        payload.get("schema_version") != GEOMETRY_REGISTRY_SCHEMA_VERSION
        or not isinstance(payload.get("registry_id"), str)
        or not payload.get("registry_id")
        or not _stable_geometry_registry_artifact_ref(payload.get("artifact_ref"))
        or not _registry_upstream_bindings_are_portable(payload)
        or payload.get("coordinate_frame") != "world"
        or not isinstance(payload.get("routes"), list)
        or not isinstance(payload.get("conflict_zones"), list)
    ):
        raise WorkedExampleProcessTraceValidationError(
            ["invalid process-trace geometry registry envelope"], source=resolved
        )
    validator = Draft202012Validator(load_process_trace_geometry_registry_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise WorkedExampleProcessTraceValidationError(errors, source=resolved)
    return dict(payload), hashlib.sha256(raw).hexdigest(), resolved


def _stable_geometry_registry_artifact_ref(value: object) -> bool:
    """Return whether a registry reference is portable public identity."""

    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if re.match(r"^[A-Za-z]:", value):
        return False
    if "\\" in value or "~" in value:
        return False
    uri_match = re.fullmatch(r"([a-z][a-z0-9+.-]*):\S+", value)
    if uri_match is not None:
        uri_segments = [
            segment for segment in value.split(":", maxsplit=1)[1].split("/") if segment
        ]
        return uri_match.group(1) != "file" and all(
            segment not in {".", ".."} for segment in uri_segments
        )
    path = Path(value)
    return not path.is_absolute() and all(
        segment not in {"", ".", ".."} for segment in value.split("/")
    )


def _registry_upstream_bindings_are_portable(payload: Mapping[str, Any]) -> bool:
    """Return whether canonical upstream bindings contain only stable references."""

    for collection in ("routes", "conflict_zones"):
        entries = payload.get(collection)
        if not isinstance(entries, list):
            return False
        for entry in entries:
            binding = entry.get("upstream_binding") if isinstance(entry, Mapping) else None
            if not isinstance(binding, Mapping):
                return False
            if binding.get(
                "kind"
            ) == "canonical_source" and not _stable_geometry_registry_artifact_ref(
                binding.get("source_artifact_ref")
            ):
                return False
    return True


def _resolve_geometry_registry_path(
    artifact_ref: str,
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> Path | None:
    """Resolve portable identity into local validation context without emitting it.

    Returns:
        Machine-local path, or ``None`` when the reference has no resolver.
    """

    if geometry_registry_paths is not None and artifact_ref in geometry_registry_paths:
        return Path(geometry_registry_paths[artifact_ref]).resolve()
    if not _stable_geometry_registry_artifact_ref(artifact_ref) or re.match(
        r"^[a-z][a-z0-9+.-]*:", artifact_ref
    ):
        return None
    return (Path.cwd() / artifact_ref).resolve()


def _resolve_registry_entry_owner(
    entry: Mapping[str, Any],
    *,
    artifact_paths: Mapping[str, str | Path] | None,
) -> tuple[str | None, Path | None]:
    """Resolve a canonical owner reference into private validation context.

    Returns:
        Stable owner reference and its private resolved path, when canonical.
    """

    binding = entry.get("upstream_binding")
    if not isinstance(binding, Mapping) or binding.get("kind") != "canonical_source":
        return None, None
    artifact_ref = binding.get("source_artifact_ref")
    if not isinstance(artifact_ref, str):
        return None, None
    return artifact_ref, _resolve_geometry_registry_path(
        artifact_ref,
        geometry_registry_paths=artifact_paths,
    )


def _registry_entry_owner_from_path(
    registry_path: Path | None,
    *,
    collection: str,
    entry_id: str | None,
    artifact_paths: Mapping[str, str | Path] | None,
) -> tuple[str | None, Path | None]:
    if registry_path is None or entry_id is None:
        return None, None
    try:
        payload = _strict_json_document(registry_path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError):
        return None, None
    entries = payload.get(collection) if isinstance(payload, Mapping) else None
    matches = (
        [
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("entry_id") == entry_id
        ]
        if isinstance(entries, list)
        else []
    )
    if len(matches) != 1:
        return None, None
    return _resolve_registry_entry_owner(matches[0], artifact_paths=artifact_paths)


def _geometry_registry_paths_for_specs(
    route: RouteSpec | None,
    conflict_zone: ConflictZoneSpec | None,
) -> dict[str, Path]:
    """Collect private local registry paths for semantic replay.

    Returns:
        Mapping from stable public identity to machine-local path.
    """

    paths: dict[str, Path] = {}
    for spec in (route, conflict_zone):
        if spec is None or not spec.registry_artifact_ref or not spec.registry_path:
            continue
        paths.setdefault(spec.registry_artifact_ref, Path(spec.registry_path).resolve())
        if spec.owner_artifact_ref and spec.owner_artifact_path:
            paths.setdefault(spec.owner_artifact_ref, Path(spec.owner_artifact_path).resolve())
    return paths


def _unique_registry_entry(
    payload: Mapping[str, Any],
    collection: str,
    entry_id: str,
    *,
    source: Path,
) -> dict[str, Any]:
    entries = payload.get(collection)
    matches = (
        [
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("entry_id") == entry_id
        ]
        if isinstance(entries, list)
        else []
    )
    if len(matches) != 1:
        raise WorkedExampleProcessTraceValidationError(
            [f"/{collection}: entry_id {entry_id!r} must resolve exactly once"], source=source
        )
    return dict(matches[0])


def build_worked_example_process_trace(
    input_path: Path,
    *,
    route: RouteSpec | None = None,
    conflict_zone: ConflictZoneSpec | None = None,
    focal_actor_id: str | None = None,
    focal_encounter_id: str | None = None,
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
        focal_encounter_id=focal_encounter_id,
        pair_trace=pair_trace,
        encounter_report=encounter_report,
        encounter_report_input_checksum=input_checksum if encounter_report is not None else None,
        pair_comparison_grain=pair_comparison_grain,
    )
    validate_worked_example_process_trace(
        payload,
        source=input_path,
        geometry_registry_paths=_geometry_registry_paths_for_specs(route, conflict_zone),
    )
    return payload


def build_worked_example_process_trace_from_export(  # noqa: PLR0913
    trace: SimulationTraceExport,
    *,
    route: RouteSpec | None = None,
    conflict_zone: ConflictZoneSpec | None = None,
    focal_actor_id: str | None = None,
    focal_encounter_id: str | None = None,
    pair_trace: SimulationTraceExport | None = None,
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
    pair_comparison_grain: str | None = None,
) -> dict[str, Any]:
    """Build a schema-valid process trace from a typed trace export.

    Returns:
        Schema-valid process trace payload.
    """

    try:
        payload = _build_worked_example_process_trace_from_export(
            trace,
            route=route,
            conflict_zone=conflict_zone,
            focal_actor_id=focal_actor_id,
            focal_encounter_id=focal_encounter_id,
            pair_trace=pair_trace,
            encounter_report=encounter_report,
            encounter_report_input_checksum=encounter_report_input_checksum,
            pair_comparison_grain=pair_comparison_grain,
        )
    except WorkedExampleProcessTraceValidationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise WorkedExampleProcessTraceValidationError(
            ["/analysis_input_contract: malformed non-JSON input"]
        ) from exc
    validate_worked_example_process_trace(
        payload,
        geometry_registry_paths=_geometry_registry_paths_for_specs(route, conflict_zone),
    )
    return payload


def _build_worked_example_process_trace_from_export(  # noqa: PLR0913
    trace: SimulationTraceExport,
    *,
    route: RouteSpec | None = None,
    conflict_zone: ConflictZoneSpec | None = None,
    focal_actor_id: str | None = None,
    focal_encounter_id: str | None = None,
    pair_trace: SimulationTraceExport | None = None,
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
    pair_comparison_grain: str | None = None,
) -> dict[str, Any]:
    """Construct a process trace without recursively invoking semantic validation.

    Returns:
        Unvalidated deterministic process-trace payload used by build and replay.
    """

    identity_errors = _duplicate_pedestrian_identity_errors(trace, label="source trace")
    if pair_trace is not None:
        identity_errors.extend(
            _duplicate_pedestrian_identity_errors(pair_trace, label="pair trace")
        )
    if identity_errors:
        raise WorkedExampleProcessTraceValidationError(identity_errors)

    focal = _resolve_focal_actor(
        trace,
        requested_actor_id=focal_actor_id,
        requested_encounter_id=focal_encounter_id,
        encounter_report=encounter_report,
        encounter_report_input_checksum=encounter_report_input_checksum,
    )
    route_input_contract = _route_input_contract(route)
    conflict_input_contract = _conflict_input_contract(conflict_zone)
    route_availability = _route_availability(
        route,
        source_coordinate_frame=trace.coordinate_frame,
    )
    route_availability["input_contract"] = _strict_json_value(route_input_contract)
    conflict_availability = _conflict_availability(
        conflict_zone,
        source_coordinate_frame=trace.coordinate_frame,
    )
    conflict_availability["input_contract"] = _strict_json_value(conflict_input_contract)
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
    events = _event_anchors(
        trace,
        frames=event_frames,
        episode_frames=frames,
        focal_actor_id=focal.get("actor_id"),
        focal_interval=_focal_interval_bounds(focal),
    )
    event_anchor_hierarchy = _event_anchor_hierarchy(events)
    frames = _frames_with_event_alignment(frames, event_anchor_hierarchy)
    pair = (
        build_pair_compatibility_record(
            trace,
            pair_trace,
            left_events=events,
            comparison_grain=pair_comparison_grain or "undeclared",
            right_events=_replayed_trace_events(
                pair_trace,
                pair_focal or {},
                route,
                conflict_zone,
            ),
        )
        if pair_trace is not None
        else unavailable_pair_compatibility(comparison_grain=pair_comparison_grain)
    )
    pair["right_coordinate_input_contract"] = {
        "route": _strict_json_value(route_input_contract),
        "conflict": _strict_json_value(conflict_input_contract),
    }
    analysis_input_contract = _analysis_input_contract(
        trace,
        route_input_contract=route_input_contract,
        conflict_input_contract=conflict_input_contract,
        focal_actor_id=focal_actor_id,
        focal_encounter_id=focal_encounter_id,
        pair_trace=pair_trace,
        encounter_report=encounter_report,
        encounter_report_input_checksum=encounter_report_input_checksum,
        pair_comparison_grain=pair_comparison_grain,
    )
    analysis_input_sha256 = _json_sha256_digest(analysis_input_contract)
    payload: dict[str, Any] = {
        "schema_version": WORKED_EXAMPLE_PROCESS_TRACE_SCHEMA_VERSION,
        "process_trace_id": f"{trace.trace_id}-process-trace-{analysis_input_sha256}",
        "analysis_input_contract": analysis_input_contract,
        "analysis_input_sha256": analysis_input_sha256,
        "source_trace": _source_trace(trace),
        "evidence_boundary": "analysis_workbench_only",
        "source_coordinate_frame": trace.coordinate_frame,
        "units": trace.units,
        "claim_boundary": CLAIM_BOUNDARY,
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
            route_available=any(
                frame["route"].get("status") == "available" for frame in event_frames
            ),
        ),
        "event_anchors": events,
        "event_anchor_hierarchy": event_anchor_hierarchy,
        "pair_compatibility": pair,
    }
    return payload


def _analysis_input_contract(  # noqa: PLR0913
    trace: SimulationTraceExport,
    *,
    route_input_contract: Mapping[str, Any],
    conflict_input_contract: Mapping[str, Any],
    focal_actor_id: str | None,
    focal_encounter_id: str | None,
    pair_trace: SimulationTraceExport | None,
    encounter_report: Mapping[str, Any] | None,
    encounter_report_input_checksum: str | None,
    pair_comparison_grain: str | None,
) -> dict[str, Any]:
    """Return the canonical receipt for every analysis-affecting input."""

    pair_receipt: dict[str, Any] = {"status": "not_supplied"}
    if pair_trace is not None:
        pair_content_receipt = build_simulation_trace_receipt(pair_trace)
        pair_receipt = {
            "status": "supplied",
            "content_sha256": simulation_trace_receipt_sha256(pair_content_receipt),
            "content_receipt": pair_content_receipt,
        }
    report_receipt: dict[str, Any] = {"status": "not_supplied"}
    if encounter_report is not None:
        content_contract = _strict_json_value(encounter_report)
        report_receipt = {
            "status": "supplied",
            "content_sha256": _json_sha256_digest(content_contract),
            "content_contract": content_contract,
            "expected_input_checksum": encounter_report_input_checksum,
        }
    return {
        "schema_version": ANALYSIS_INPUT_SCHEMA_VERSION,
        "source_trace_content_sha256": _trace_content_sha256(trace),
        "route": _strict_json_value(route_input_contract),
        "conflict": _strict_json_value(conflict_input_contract),
        "pair_trace": pair_receipt,
        "encounter_report": report_receipt,
        "focal_actor_id": focal_actor_id,
        "focal_encounter_id": focal_encounter_id,
        "pair_comparison_grain": pair_comparison_grain,
    }


def write_worked_example_process_trace(input_path: Path, output_path: Path, **kwargs: Any) -> Path:
    """Write a deterministic process trace JSON file.

    Returns:
        Output path that received the process trace.
    """

    payload = build_worked_example_process_trace(input_path, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialize_worked_example_process_trace(payload))
    return output_path


def serialize_worked_example_process_trace(payload: Mapping[str, Any]) -> bytes:
    """Serialize the exact deterministic bytes emitted by the official writer.

    Returns:
        UTF-8 JSON bytes with deterministic formatting and one trailing newline.
    """

    try:
        _assert_exact_json_value(payload)
        return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise WorkedExampleProcessTraceValidationError(
            ["/: payload must contain only strict JSON values with string object keys"]
        ) from exc


def worked_example_process_trace_artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Digest the exact deterministic bytes emitted by the official writer.

    Returns:
        SHA-256 hex digest of :func:`serialize_worked_example_process_trace`.
    """

    return hashlib.sha256(serialize_worked_example_process_trace(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_trace(trace: SimulationTraceExport) -> dict[str, Any]:
    content_receipt = build_simulation_trace_receipt(trace)
    return {
        "schema_version": SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
        "trace_id": trace.trace_id,
        "coordinate_frame": trace.coordinate_frame,
        "units": trace.units,
        "content_sha256": simulation_trace_receipt_sha256(content_receipt),
        "content_receipt": content_receipt,
        "run_config_contract": _run_config_contract(trace),
        "source": {
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
            "generated_by": trace.source.generated_by,
        },
    }


def _trace_content_sha256(trace: SimulationTraceExport) -> str:
    return simulation_trace_receipt_sha256(build_simulation_trace_receipt(trace))


def _duplicate_pedestrian_identity_errors(
    trace: SimulationTraceExport,
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for frame_index, frame in enumerate(trace.frames):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for pedestrian in frame.pedestrians:
            if "id" not in pedestrian:
                continue
            actor_id = str(pedestrian["id"])
            if actor_id in seen:
                duplicates.add(actor_id)
            seen.add(actor_id)
        errors.extend(
            f"{label} frame {frame_index}: duplicate pedestrian id {actor_id!r}"
            for actor_id in sorted(duplicates)
        )
    return errors


def _strict_json_value(value: Any) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("strict JSON numbers must be finite")
        return value
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("strict JSON object keys must be strings")
        return {key: _strict_json_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"unsupported strict JSON value: {type(value).__name__}")


def _assert_exact_json_value(value: object, *, path: str = "") -> None:
    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise TypeError(f"nonfinite JSON number at {path or '/'}")
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_exact_json_value(item, path=f"{path}/{index}")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"non-string JSON key at {path or '/'}")
            _assert_exact_json_value(item, path=f"{path}/{key}")
        return
    raise TypeError(f"non-JSON value at {path or '/'}: {type(value).__name__}")


def _run_config_contract(trace: SimulationTraceExport) -> dict[str, Any]:
    return build_trace_run_config_contract(trace)


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


def _route_input_contract(route: RouteSpec | None) -> dict[str, Any]:
    if route is None:
        return {"status": "not_supplied"}
    return {
        "status": "supplied" if _has_complete_registry_receipt(route) else "supplied_unregistered",
        "route_id": route.route_id,
        "provenance_id": route.provenance_id,
        "registry_checksum": route.registry_checksum,
        "geometry": _strict_json_value(_route_spec_geometry(route)),
        "registry": {
            "artifact_ref": route.registry_artifact_ref,
            "content_sha256": route.registry_content_sha256,
            "entry_id": route.registry_entry_id,
            "entry_sha256": route.registry_entry_sha256,
        },
    }


def _conflict_input_contract(conflict_zone: ConflictZoneSpec | None) -> dict[str, Any]:
    if conflict_zone is None:
        return {"status": "not_supplied"}
    return {
        "status": "supplied"
        if _has_complete_registry_receipt(conflict_zone)
        else "supplied_unregistered",
        "zone_id": conflict_zone.zone_id,
        "provenance_id": conflict_zone.provenance_id,
        "registry_checksum": conflict_zone.registry_checksum,
        "geometry": _strict_json_value(_conflict_spec_geometry(conflict_zone)),
        "registry": {
            "artifact_ref": conflict_zone.registry_artifact_ref,
            "content_sha256": conflict_zone.registry_content_sha256,
            "entry_id": conflict_zone.registry_entry_id,
            "entry_sha256": conflict_zone.registry_entry_sha256,
        },
    }


def _has_complete_registry_receipt(spec: RouteSpec | ConflictZoneSpec) -> bool:
    return all(
        isinstance(value, str) and bool(value)
        for value in (
            spec.registry_artifact_ref,
            spec.registry_content_sha256,
            spec.registry_entry_id,
            spec.registry_entry_sha256,
        )
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _route_spec_from_input_contract(
    value: object,
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> RouteSpec | None:
    if not isinstance(value, Mapping) or value.get("status") not in {
        "supplied",
        "supplied_unregistered",
    }:
        return None
    geometry = value.get("geometry")
    registry = value.get("registry")
    if not isinstance(geometry, Mapping) or not isinstance(registry, Mapping):
        return None
    points = _route_geometry_points(geometry)
    start = points[0] if points else (math.nan, math.nan)
    end = points[-1] if points else (math.nan, math.nan)
    artifact_ref = _optional_string(registry.get("artifact_ref"))
    registry_path = (
        _resolve_geometry_registry_path(
            artifact_ref,
            geometry_registry_paths=geometry_registry_paths,
        )
        if artifact_ref is not None
        else None
    )
    owner_ref, owner_path = _registry_entry_owner_from_path(
        registry_path,
        collection="routes",
        entry_id=_optional_string(registry.get("entry_id")),
        artifact_paths=geometry_registry_paths,
    )
    return RouteSpec(
        route_id=str(value.get("route_id", "")),
        start=start,
        end=end,
        provenance_id=_optional_string(value.get("provenance_id")),
        registry_checksum=_optional_string(value.get("registry_checksum")),
        geometry=dict(geometry),
        registry_artifact_ref=artifact_ref,
        registry_path=str(registry_path) if registry_path is not None else None,
        registry_content_sha256=_optional_string(registry.get("content_sha256")),
        registry_entry_id=_optional_string(registry.get("entry_id")),
        registry_entry_sha256=_optional_string(registry.get("entry_sha256")),
        owner_artifact_ref=owner_ref,
        owner_artifact_path=str(owner_path) if owner_path is not None else None,
    )


def _conflict_spec_from_input_contract(
    value: object,
    *,
    geometry_registry_paths: Mapping[str, str | Path] | None,
) -> ConflictZoneSpec | None:
    if not isinstance(value, Mapping) or value.get("status") not in {
        "supplied",
        "supplied_unregistered",
    }:
        return None
    geometry = value.get("geometry")
    registry = value.get("registry")
    if not isinstance(geometry, Mapping) or not isinstance(registry, Mapping):
        return None
    center = _vector2(geometry.get("center")) or (math.nan, math.nan)
    radius = geometry.get("radius_m")
    radius_m = float(radius) if _finite_json_number(radius) else math.nan
    artifact_ref = _optional_string(registry.get("artifact_ref"))
    registry_path = (
        _resolve_geometry_registry_path(
            artifact_ref,
            geometry_registry_paths=geometry_registry_paths,
        )
        if artifact_ref is not None
        else None
    )
    owner_ref, owner_path = _registry_entry_owner_from_path(
        registry_path,
        collection="conflict_zones",
        entry_id=_optional_string(registry.get("entry_id")),
        artifact_paths=geometry_registry_paths,
    )
    return ConflictZoneSpec(
        zone_id=str(value.get("zone_id", "")),
        center=center,
        radius_m=radius_m,
        provenance_id=_optional_string(value.get("provenance_id")),
        registry_checksum=_optional_string(value.get("registry_checksum")),
        geometry=dict(geometry),
        registry_artifact_ref=artifact_ref,
        registry_path=str(registry_path) if registry_path is not None else None,
        registry_content_sha256=_optional_string(registry.get("content_sha256")),
        registry_entry_id=_optional_string(registry.get("entry_id")),
        registry_entry_sha256=_optional_string(registry.get("entry_sha256")),
        owner_artifact_ref=owner_ref,
        owner_artifact_path=str(owner_path) if owner_path is not None else None,
    )


def _route_availability(
    route: RouteSpec | None,
    *,
    source_coordinate_frame: str = "world",
) -> dict[str, Any]:
    if route is None:
        return {"status": "unavailable", "reason": "registered_route_unavailable"}
    if source_coordinate_frame != "world":
        return {
            "status": "unavailable",
            "reason": "source_coordinate_frame_not_world",
            "source_coordinate_frame": source_coordinate_frame,
        }
    if not route.provenance_id:
        return {"status": "unavailable", "reason": "registered_route_provenance_unavailable"}
    if not route.registry_checksum:
        return {"status": "unavailable", "reason": "registered_route_checksum_unavailable"}
    if SHA256_HEX_RE.fullmatch(str(route.registry_checksum)) is None:
        return {"status": "unavailable", "reason": "registered_route_checksum_invalid"}
    geometry = _route_spec_geometry(route)
    geometry_reason = _route_geometry_unavailable_reason(geometry)
    if geometry_reason is not None:
        return {"status": "unavailable", "reason": geometry_reason}
    geometry_checksum = _geometry_checksum(geometry)
    if route.registry_checksum != geometry_checksum:
        return {
            "status": "unavailable",
            "reason": "registered_route_checksum_geometry_mismatch",
            "geometry_checksum": geometry_checksum,
        }
    registry = _registry_binding(
        path=route.registry_path,
        artifact_ref=route.registry_artifact_ref,
        content_sha256=route.registry_content_sha256,
        entry_id=route.registry_entry_id,
        entry_sha256=route.registry_entry_sha256,
        collection="routes",
        identity_key="route_id",
        identity_value=route.route_id,
        provenance_id=route.provenance_id,
        geometry=geometry,
        reason_prefix="registered_route",
        owner_artifact_path=route.owner_artifact_path,
    )
    if registry.get("status") != "available":
        return {"status": "unavailable", "reason": str(registry["reason"])}
    return {
        "status": "available",
        "reason": "registered_polyline_route"
        if geometry.get("type") == "polyline"
        else "registered_straight_route",
        "route_id": route.route_id,
        "provenance_id": route.provenance_id,
        "registry_checksum": route.registry_checksum,
        "registry": registry["receipt"],
        "coordinate_frame": "world",
        "geometry": geometry,
    }


def _conflict_availability(
    conflict_zone: ConflictZoneSpec | None,
    *,
    source_coordinate_frame: str = "world",
) -> dict[str, Any]:
    if conflict_zone is None:
        return {"status": "unavailable", "reason": "registered_conflict_zone_unavailable"}
    if source_coordinate_frame != "world":
        return {
            "status": "unavailable",
            "reason": "source_coordinate_frame_not_world",
            "source_coordinate_frame": source_coordinate_frame,
        }
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
    geometry = _conflict_spec_geometry(conflict_zone)
    geometry_kind = geometry.get("type")
    if not _conflict_geometry_matches_spec(conflict_zone, geometry):
        return {"status": "unavailable", "reason": "registered_conflict_zone_invalid"}
    geometry_checksum = _geometry_checksum(geometry)
    if conflict_zone.registry_checksum != geometry_checksum:
        return {
            "status": "unavailable",
            "reason": "registered_conflict_zone_checksum_geometry_mismatch",
            "geometry_checksum": geometry_checksum,
        }
    registry = _registry_binding(
        path=conflict_zone.registry_path,
        artifact_ref=conflict_zone.registry_artifact_ref,
        content_sha256=conflict_zone.registry_content_sha256,
        entry_id=conflict_zone.registry_entry_id,
        entry_sha256=conflict_zone.registry_entry_sha256,
        collection="conflict_zones",
        identity_key="zone_id",
        identity_value=conflict_zone.zone_id,
        provenance_id=conflict_zone.provenance_id,
        geometry=geometry,
        reason_prefix="registered_conflict_zone",
        owner_artifact_path=conflict_zone.owner_artifact_path,
    )
    if registry.get("status") != "available":
        return {"status": "unavailable", "reason": str(registry["reason"])}
    if geometry_kind in {"point", "polygon"}:
        return {
            "status": "unavailable",
            "reason": f"registered_conflict_zone_{geometry_kind}_projection_unavailable",
        }
    return {
        "status": "available",
        "reason": "registered_circular_conflict_zone",
        "zone_id": conflict_zone.zone_id,
        "provenance_id": conflict_zone.provenance_id,
        "registry_checksum": conflict_zone.registry_checksum,
        "registry": registry["receipt"],
        "coordinate_frame": "world",
        "geometry": geometry,
    }


def _geometry_checksum(geometry: Mapping[str, Any]) -> str:
    return _json_sha256_digest(geometry)


def _route_spec_geometry(route: RouteSpec) -> dict[str, Any]:
    if isinstance(route.geometry, Mapping):
        geometry = _strict_json_value(route.geometry)
        if not isinstance(geometry, dict):
            raise TypeError("route geometry must be a JSON object")
        return geometry
    if not (
        len(route.start) == 2
        and len(route.end) == 2
        and all(_finite_json_number(value) for value in (*route.start, *route.end))
    ):
        raise TypeError("route endpoints must be finite two-vectors")
    return {"type": "line_segment", "start": list(route.start), "end": list(route.end)}


def _conflict_spec_geometry(conflict_zone: ConflictZoneSpec) -> dict[str, Any]:
    if isinstance(conflict_zone.geometry, Mapping):
        geometry = _strict_json_value(conflict_zone.geometry)
        if not isinstance(geometry, dict):
            raise TypeError("conflict geometry must be a JSON object")
        return geometry
    if not (
        len(conflict_zone.center) == 2
        and all(_finite_json_number(value) for value in conflict_zone.center)
        and _finite_json_number(conflict_zone.radius_m)
    ):
        raise TypeError("conflict geometry must use finite numeric coordinates")
    return {
        "type": "circle",
        "center": list(conflict_zone.center),
        "radius_m": conflict_zone.radius_m,
    }


def _conflict_geometry_matches_spec(
    conflict_zone: ConflictZoneSpec,
    geometry: Mapping[str, Any],
) -> bool:
    geometry_kind = geometry.get("type")
    if geometry_kind in {"point", "polygon"}:
        return True
    center = _vector2(geometry.get("center"))
    radius = geometry.get("radius_m")
    return bool(
        geometry_kind == "circle"
        and center is not None
        and _finite_json_number(radius)
        and float(radius) >= 0.0
        and center == _vector2(conflict_zone.center)
        and float(radius) == conflict_zone.radius_m
    )


def _route_geometry_points(geometry: Mapping[str, Any]) -> list[tuple[float, float]]:
    if geometry.get("type") == "line_segment":
        start = _vector2(geometry.get("start"))
        end = _vector2(geometry.get("end"))
        return [start, end] if start is not None and end is not None else []
    if geometry.get("type") != "polyline":
        return []
    points = geometry.get("points")
    if not isinstance(points, Sequence) or isinstance(points, str | bytes):
        return []
    parsed = [_vector2(point) for point in points]
    return [point for point in parsed if point is not None] if all(parsed) else []


def _route_geometry_unavailable_reason(geometry: Mapping[str, Any]) -> str | None:
    if geometry.get("type") not in {"line_segment", "polyline"}:
        return "registered_route_branching_or_ambiguous_geometry"
    points = _route_geometry_points(geometry)
    raw_points = geometry.get("points")
    if geometry.get("type") == "line_segment":
        expected_count = 2
    elif not isinstance(raw_points, Sequence) or isinstance(raw_points, str | bytes):
        return "registered_route_invalid_geometry"
    else:
        expected_count = len(raw_points)
    if len(points) < 2 or len(points) != expected_count:
        return "registered_route_invalid_geometry"
    if any(_distance(left, right) <= 1e-12 for left, right in pairwise(points)):
        return "registered_route_degenerate"
    if _polyline_has_adjacent_backtracking(points) or _polyline_has_nonlocal_intersection(points):
        return "registered_route_branching_or_ambiguous_geometry"
    return None


def _polyline_has_adjacent_backtracking(points: Sequence[tuple[float, float]]) -> bool:
    for first, middle, last in zip(points, points[1:], points[2:], strict=False):
        incoming = (middle[0] - first[0], middle[1] - first[1])
        outgoing = (last[0] - middle[0], last[1] - middle[1])
        if (
            abs(_cross(incoming, outgoing)) <= 1e-12
            and (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) < 0.0
        ):
            return True
    return False


def _polyline_has_nonlocal_intersection(points: Sequence[tuple[float, float]]) -> bool:
    for left_index, (left_start, left_end) in enumerate(pairwise(points)):
        for right_index, (right_start, right_end) in enumerate(pairwise(points)):
            if right_index <= left_index + 1:
                continue
            if _segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return False


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return _cross(
            (second[0] - first[0], second[1] - first[1]),
            (third[0] - first[0], third[1] - first[1]),
        )

    values = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    if values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0:
        return True

    def on_segment(
        first: tuple[float, float],
        second: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        return (
            min(first[0], second[0]) - 1e-12 <= point[0] <= max(first[0], second[0]) + 1e-12
            and min(first[1], second[1]) - 1e-12 <= point[1] <= max(first[1], second[1]) + 1e-12
        )

    return any(
        abs(orientation_value) <= 1e-12 and on_segment(first, second, point)
        for orientation_value, first, second, point in (
            (values[0], a, b, c),
            (values[1], a, b, d),
            (values[2], c, d, a),
            (values[3], c, d, b),
        )
    )


def _registry_binding(  # noqa: C901, PLR0913
    *,
    path: str | None,
    artifact_ref: str | None,
    content_sha256: str | None,
    entry_id: str | None,
    entry_sha256: str | None,
    collection: str,
    identity_key: str,
    identity_value: str,
    provenance_id: str | None,
    geometry: Mapping[str, Any],
    reason_prefix: str,
    owner_artifact_path: str | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    if not all((path, artifact_ref, content_sha256, entry_id, entry_sha256)):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_artifact_unavailable"}
    if not _stable_geometry_registry_artifact_ref(artifact_ref):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_receipt_invalid"}
    if (
        SHA256_HEX_RE.fullmatch(str(content_sha256)) is None
        or SHA256_HEX_RE.fullmatch(str(entry_sha256)) is None
    ):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_receipt_invalid"}
    resolved = Path(str(path)).resolve()
    try:
        raw = resolved.read_bytes()
    except OSError:
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_artifact_missing"}
    if hashlib.sha256(raw).hexdigest() != content_sha256:
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_content_mismatch"}
    try:
        payload = _strict_json_document(raw)
    except (UnicodeDecodeError, ValueError):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_invalid"}
    if not isinstance(payload, Mapping) or (
        payload.get("schema_version") != GEOMETRY_REGISTRY_SCHEMA_VERSION
        or payload.get("coordinate_frame") != "world"
        or payload.get("registry_id") != provenance_id
        or payload.get("artifact_ref") != artifact_ref
        or not _registry_upstream_bindings_are_portable(payload)
    ):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_invalid"}
    if not Draft202012Validator(load_process_trace_geometry_registry_schema()).is_valid(payload):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_invalid"}
    entries = payload.get(collection)
    matches = (
        [
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("entry_id") == entry_id
        ]
        if isinstance(entries, list)
        else []
    )
    if not matches:
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_entry_missing"}
    if len(matches) != 1:
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_entry_ambiguous"}
    entry = matches[0]
    if (
        _json_sha256_digest(entry) != entry_sha256
        or entry.get(identity_key) != identity_value
        or entry.get("geometry") != geometry
    ):
        return {"status": "unavailable", "reason": f"{reason_prefix}_registry_entry_mismatch"}
    upstream_binding = entry.get("upstream_binding")
    owner_validation = _geometry_owner_validation(
        upstream_binding,
        geometry=geometry,
        reason_prefix=reason_prefix,
        owner_artifact_path=owner_artifact_path,
        artifact_paths=artifact_paths,
    )
    if owner_validation.get("status") == "unavailable":
        return owner_validation
    return {
        "status": "available",
        "receipt": {
            "schema_version": GEOMETRY_REGISTRY_SCHEMA_VERSION,
            "registry_id": str(payload["registry_id"]),
            "artifact_ref": artifact_ref,
            "content_sha256": str(content_sha256),
            "entry_id": str(entry_id),
            "entry_sha256": str(entry_sha256),
            "coordinate_frame": "world",
            "geometry_kind": str(geometry.get("type")),
            "resolved_geometry": dict(geometry),
            "upstream_binding": dict(entry["upstream_binding"]),
            "owner_validation": owner_validation["receipt"],
        },
    }


def _geometry_owner_validation(  # noqa: C901, PLR0912
    binding: object,
    *,
    geometry: Mapping[str, Any],
    reason_prefix: str,
    owner_artifact_path: str | None,
    artifact_paths: Mapping[str, str | Path] | None,
) -> dict[str, Any]:
    """Verify registry adapter geometry against its declared upstream owner.

    Returns:
        Available owner receipt or an explicit unavailable reason.
    """

    if not isinstance(binding, Mapping):
        return {"status": "unavailable", "reason": f"{reason_prefix}_owner_binding_invalid"}
    if binding.get("kind") == "fixture_only":
        return {
            "status": "available",
            "receipt": {
                "status": "fixture_only",
                "reason": "fixture_binding_not_canonical_owner",
            },
        }
    if binding.get("kind") != "canonical_source":
        return {"status": "unavailable", "reason": f"{reason_prefix}_owner_binding_invalid"}
    artifact_ref = binding.get("source_artifact_ref")
    expected_digest = binding.get("source_content_sha256")
    selector = binding.get("selector")
    if not (
        isinstance(artifact_ref, str)
        and _stable_geometry_registry_artifact_ref(artifact_ref)
        and isinstance(expected_digest, str)
        and SHA256_HEX_RE.fullmatch(expected_digest)
        and isinstance(selector, Mapping)
    ):
        return {"status": "unavailable", "reason": f"{reason_prefix}_owner_binding_invalid"}
    resolved_path = (
        Path(owner_artifact_path).resolve()
        if owner_artifact_path is not None
        else _resolve_geometry_registry_path(
            artifact_ref,
            geometry_registry_paths=artifact_paths,
        )
    )
    if resolved_path is None:
        return {
            "status": "unavailable",
            "reason": f"{reason_prefix}_owner_artifact_unresolved",
        }
    try:
        if not resolved_path.is_file():
            raise OSError
        raw = resolved_path.read_bytes()
    except OSError:
        return {"status": "unavailable", "reason": f"{reason_prefix}_owner_artifact_missing"}
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        return {
            "status": "unavailable",
            "reason": f"{reason_prefix}_owner_content_mismatch",
        }
    try:
        owner = _geometry_owner_payload_from_bytes(raw, source=resolved_path)
    except WorkedExampleProcessTraceValidationError as exc:
        if any("/selector" in error for error in exc.errors):
            suffix = "owner_selector_invalid"
        elif any("/geometry" in error for error in exc.errors):
            suffix = "owner_geometry_invalid"
        elif any("expected strict" in error for error in exc.errors):
            suffix = "owner_artifact_invalid_json"
        else:
            suffix = "owner_artifact_schema_invalid"
        return {"status": "unavailable", "reason": f"{reason_prefix}_{suffix}"}
    try:
        selector_digest = _json_sha256_digest(selector)
        candidates = [
            item
            for item in owner["geometry_bindings"]
            if _json_sha256_digest(item["selector"]) == selector_digest
        ]
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable",
            "reason": f"{reason_prefix}_owner_canonicalization_invalid",
        }
    if len(candidates) != 1:
        suffix = "owner_selector_ambiguous" if len(candidates) > 1 else "owner_selector_unresolved"
        return {"status": "unavailable", "reason": f"{reason_prefix}_{suffix}"}
    try:
        geometry_matches = _json_sha256_digest(candidates[0]["geometry"]) == _json_sha256_digest(
            geometry
        )
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable",
            "reason": f"{reason_prefix}_owner_canonicalization_invalid",
        }
    if not geometry_matches:
        return {"status": "unavailable", "reason": f"{reason_prefix}_owner_geometry_mismatch"}
    return {
        "status": "available",
        "receipt": {
            "status": "verified",
            "source_artifact_ref": artifact_ref,
            "source_content_sha256": expected_digest,
            "selector": dict(selector),
            "geometry_sha256": _json_sha256_digest(geometry),
        },
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


def _resolve_focal_actor(  # noqa: C901
    trace: SimulationTraceExport,
    *,
    requested_actor_id: str | None = None,
    requested_encounter_id: str | None = None,
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
) -> dict[str, Any]:
    declared = _declared_encounter(
        trace,
        requested_actor_id=requested_actor_id,
        requested_encounter_id=requested_encounter_id,
        encounter_report=encounter_report,
        encounter_report_input_checksum=encounter_report_input_checksum,
    )
    if declared.get("status") == "unavailable":
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
    if requested_encounter_id:
        requested_encounter = str(requested_encounter_id)
        if declared.get("encounter_id") != requested_encounter:
            return {
                "status": "unavailable",
                "reason": "requested_focal_encounter_missing",
                "requested_encounter_id": requested_encounter,
                "declared_encounter": declared,
            }
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


def _declared_encounter(  # noqa: C901
    trace: SimulationTraceExport,
    *,
    requested_actor_id: str | None = None,
    requested_encounter_id: str | None = None,
    encounter_report: Mapping[str, Any] | None = None,
    encounter_report_input_checksum: str | None = None,
) -> dict[str, Any]:
    if encounter_report is not None:
        return _select_canonical_encounter(
            trace,
            encounter_report,
            expected_input_checksum=encounter_report_input_checksum,
            requested_actor_id=requested_actor_id,
            requested_encounter_id=requested_encounter_id,
        )
    hints: list[dict[str, Any]] = []
    for frame in trace.frames:
        for key in ("focal_encounter", "encounter"):
            value = frame.planner.get(key)
            if isinstance(value, Mapping):
                actor_id = value.get("actor_id") or value.get("pedestrian_id")
                if actor_id is not None:
                    hints.append(
                        {
                            "actor_id": str(actor_id),
                            "encounter_id": str(value["encounter_id"])
                            if value.get("encounter_id") is not None
                            else None,
                            "source": f"planner.{key}",
                        }
                    )
        encounters = frame.planner.get("encounters")
        if isinstance(encounters, Sequence) and not isinstance(encounters, str | bytes):
            for value in encounters:
                if isinstance(value, Mapping):
                    actor_id = value.get("actor_id") or value.get("pedestrian_id")
                    if actor_id is not None:
                        hints.append(
                            {
                                "actor_id": str(actor_id),
                                "encounter_id": str(value["encounter_id"])
                                if value.get("encounter_id") is not None
                                else None,
                                "source": "planner.encounters",
                            }
                        )
    if not hints:
        return {}
    actor_ids = sorted({hint["actor_id"] for hint in hints})
    if len(actor_ids) != 1:
        return {
            "status": "unavailable",
            "reason": "planner_encounter_actor_hint_ambiguous",
            "schema_version": "planner_actor_hint.v1",
            "hint_actor_ids": actor_ids,
        }
    encounter_ids = sorted(
        {hint["encounter_id"] for hint in hints if hint["encounter_id"] is not None}
    )
    if len(encounter_ids) > 1:
        return {
            "status": "unavailable",
            "reason": "planner_encounter_id_hint_ambiguous",
            "schema_version": "planner_actor_hint.v1",
            "actor_id": actor_ids[0],
            "hint_encounter_ids": encounter_ids,
        }
    if encounter_ids and not _encounter_id_actor_binding_valid(encounter_ids[0], actor_ids[0]):
        return {
            "status": "unavailable",
            "reason": "planner_encounter_id_actor_mismatch",
            "schema_version": "planner_actor_hint.v1",
            "actor_id": actor_ids[0],
            "hint_encounter_ids": encounter_ids,
        }
    sources = sorted({hint["source"] for hint in hints})
    return {
        "actor_id": actor_ids[0],
        "encounter_id": encounter_ids[0] if encounter_ids else None,
        "schema_version": "planner_actor_hint.v1",
        "source": sources[0] if len(sources) == 1 else "planner.multiple_hints",
    }


def _select_canonical_encounter(  # noqa: C901
    trace: SimulationTraceExport,
    encounter_report: Mapping[str, Any],
    *,
    expected_input_checksum: str | None,
    requested_actor_id: str | None,
    requested_encounter_id: str | None,
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
    encounter_id_counts: dict[str, int] = {}
    for encounter in valid:
        encounter_id = str(encounter.get("encounter_id"))
        encounter_id_counts[encounter_id] = encounter_id_counts.get(encounter_id, 0) + 1
    duplicate_ids = sorted(
        encounter_id for encounter_id, count in encounter_id_counts.items() if count != 1
    )
    if duplicate_ids:
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_id_not_unique",
            "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
            "duplicate_encounter_ids": duplicate_ids,
        }
    mismatched_ids = sorted(
        str(encounter.get("encounter_id"))
        for encounter in valid
        if not _encounter_id_actor_binding_valid(
            str(encounter.get("encounter_id")),
            str(encounter.get("actor_id")),
        )
    )
    if mismatched_ids:
        return {
            "status": "unavailable",
            "reason": "canonical_encounter_id_actor_mismatch",
            "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
            "mismatched_encounter_ids": mismatched_ids,
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
    if requested_actor_id is not None:
        requested_actor = str(requested_actor_id)
        candidates = [
            encounter for encounter in candidates if str(encounter["actor_id"]) == requested_actor
        ]
        if not candidates:
            return {
                "status": "unavailable",
                "reason": "requested_focal_actor_has_no_canonical_encounter",
                "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
                "requested_actor_id": requested_actor,
            }
    if requested_encounter_id is not None:
        requested_encounter = str(requested_encounter_id)
        candidates = [
            encounter
            for encounter in candidates
            if str(encounter["encounter_id"]) == requested_encounter
        ]
        if not candidates:
            return {
                "status": "unavailable",
                "reason": "requested_focal_encounter_missing",
                "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
                "requested_encounter_id": requested_encounter,
            }
    selected = min(
        candidates,
        key=lambda encounter: (
            float(encounter["start_time_s"]),
            float(encounter["end_time_s"]),
            str(encounter["actor_id"]),
            str(encounter["encounter_id"]),
        ),
    )
    report_content_contract = _strict_json_value(encounter_report)
    selected_record = {key: selected[key] for key in selected}
    selected_entry_index = next(
        index for index, encounter in enumerate(encounters) if encounter is selected
    )
    return {
        "schema_version": CANONICAL_ENCOUNTER_SCHEMA_VERSION,
        "actor_id": str(selected["actor_id"]),
        "encounter_id": str(selected["encounter_id"]),
        "canonical_record": selected_record,
        "report_profile": dict(encounter_report["profile"]),
        "report_provenance": dict(encounter_report["provenance"]),
        "report_input_contract": {
            "schema_version": ENCOUNTER_REPORT_INPUT_SCHEMA_VERSION,
            "content_sha256": _json_sha256_digest(report_content_contract),
            "content_contract": report_content_contract,
            "selected_entry_index": selected_entry_index,
            "selected_entry_sha256": _json_sha256_digest(selected_record),
        },
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


def _encounter_id_actor_binding_valid(encounter_id: str, actor_id: str) -> bool:
    """Check an optional actor prefix without treating actor-ID colons as delimiters.

    Returns:
        Whether the encounter ID is unprefixed or starts with the complete actor ID.
    """

    return ":" not in encounter_id or encounter_id.startswith(f"{actor_id}:")


def _json_sha256_digest(value: object) -> str:
    _assert_exact_json_value(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
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
            "focal_actor_id": str(focal_actor_id) if focal is not None else None,
            "focal_actor": _world_actor(focal) if focal is not None else None,
            "contextual_actors": _source_actor_inventory(frame.pedestrians),
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
        "route": _route_frame(robot_pos, robot_vel, focal, route, source_coordinate_frame),
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


def _focal_interval_bounds(focal_encounter: Mapping[str, Any]) -> tuple[float, float] | None:
    declared = focal_encounter.get("declared_encounter")
    record = focal_encounter.get("canonical_record")
    if record is None and isinstance(declared, Mapping):
        record = declared.get("canonical_record")
    if not isinstance(record, Mapping):
        return None
    start = record.get("start_time_s")
    end = record.get("end_time_s")
    if not (_finite_json_number(start) and _finite_json_number(end)):
        return None
    return float(start), float(end)


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


def _source_actor_inventory(actors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "actor_id": str(actor["id"]),
            **(_world_actor(actor) or {}),
        }
        for actor in actors
        if "id" in actor
    ]


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
    focal: Mapping[str, Any] | None,
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
    geometry = route_availability["geometry"]
    robot_projection = _project_onto_route(robot_pos, geometry)
    if robot_projection is None:
        return {"status": "unavailable", "reason": "ambiguous_route_projection"}
    progress_rate = _dot(robot_vel, robot_projection["unit"]) if robot_vel is not None else None
    focal_pos = _vector2(focal.get("position")) if focal is not None else None
    focal_vel = _vector2(focal.get("velocity")) if focal is not None else None
    if focal_pos is None:
        focal_payload: dict[str, Any] = {
            "focal_actor_status": "unavailable",
            "focal_actor_reason": "missing_focal_actor_position",
            "focal_actor_s_m": None,
            "focal_actor_n_m": None,
            "focal_actor_progress_rate_mps": None,
        }
    else:
        focal_projection = _project_onto_route(focal_pos, geometry)
        if focal_projection is None:
            focal_payload = {
                "focal_actor_status": "unavailable",
                "focal_actor_reason": "ambiguous_route_projection",
                "focal_actor_s_m": None,
                "focal_actor_n_m": None,
                "focal_actor_progress_rate_mps": None,
            }
        else:
            focal_payload = {
                "focal_actor_status": "available",
                "focal_actor_s_m": focal_projection["s_m"],
                "focal_actor_n_m": focal_projection["n_m"],
                "focal_actor_progress_rate_mps": _dot(focal_vel, focal_projection["unit"])
                if focal_vel is not None
                else None,
            }
    return {
        "status": "available",
        "route_id": route.route_id,
        "provenance_id": route.provenance_id,
        "registry_checksum": route.registry_checksum,
        "registry": route_availability["registry"],
        "geometry": geometry,
        "s_m": robot_projection["s_m"],
        "n_m": robot_projection["n_m"],
        "progress_rate_mps": progress_rate,
        **focal_payload,
    }


def _project_onto_route(
    point: tuple[float, float], geometry: Mapping[str, Any]
) -> dict[str, Any] | None:
    points = _route_geometry_points(geometry)
    candidates: list[dict[str, Any]] = []
    cumulative = 0.0
    for index, (start, end) in enumerate(pairwise(points)):
        axis = (end[0] - start[0], end[1] - start[1])
        length = _norm(axis)
        unit = (axis[0] / length, axis[1] / length)
        relative = (point[0] - start[0], point[1] - start[1])
        along = min(max(_dot(relative, unit), 0.0), length)
        projected = (start[0] + along * unit[0], start[1] + along * unit[1])
        residual = (point[0] - projected[0], point[1] - projected[1])
        candidates.append(
            {
                "index": index,
                "distance": _norm(residual),
                "s_m": cumulative + along,
                "n_m": _cross(unit, residual),
                "unit": unit,
                "projected": projected,
            }
        )
        cumulative += length
    if not candidates:
        return None
    minimum = min(float(candidate["distance"]) for candidate in candidates)
    closest = [
        candidate
        for candidate in candidates
        if math.isclose(float(candidate["distance"]), minimum, rel_tol=1e-9, abs_tol=1e-12)
    ]
    if len(closest) != 1:
        return None
    return closest[0]


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
        "registry": conflict_availability["registry"],
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
        elif _finite_json_number(item):
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
    if coverage["proxy_surface_clearance"]["status"] != "complete":
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
            if _finite_json_number(value) and float(value) < threshold:
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
        "stall": _stall_summary(frames),
        "conflict_zone_occupancy": _conflict_occupancy(frames),
        "reversal_counts": _reversal_counts_summary(frames, route_available=route_available),
        "coverage": coverage,
    }


def _stall_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stall_min_duration_s = _profiles()["phase_profile"]["stall_min_duration_s"]
    if not frames:
        return {
            "profile_version": PHASE_PROFILE_VERSION,
            "status": "unavailable",
            "reason": "no_diagnostic_frames",
            "stall_min_duration_s": stall_min_duration_s,
            "sustained_stall_duration_s": None,
            "speed_coverage": {
                "status": "unavailable",
                "frame_count": 0,
                "available_frame_count": 0,
                "missing_frame_count": 0,
                "reason": "no_diagnostic_frames",
            },
            "sustained_stall_onset_step": None,
        }
    summary = summarize_stall(
        frames,
        speed_getter=_speed_from_frame,
        stall_speed_threshold_mps=_profiles()["phase_profile"]["stall_speed_threshold_mps"],
        stall_min_duration_s=stall_min_duration_s,
    )
    speed_coverage = summary.get("speed_coverage")
    if isinstance(speed_coverage, Mapping):
        enriched = dict(speed_coverage)
        enriched["frame_count"] = len(frames)
        enriched.setdefault(
            "reason",
            "coverage_complete" if enriched.get("missing_frame_count") == 0 else "missing_speed",
        )
        summary["speed_coverage"] = enriched
    return summary


def _reversal_counts_summary(
    frames: Sequence[Mapping[str, Any]], *, route_available: bool
) -> dict[str, Any]:
    base = {
        "profile_version": REVERSAL_PROFILE_VERSION,
        "direction_semantics": "robot_heading_and_velocity_projection",
    }
    if not frames:
        return {
            **base,
            "status": "unavailable",
            "reason": "no_diagnostic_frames",
            "heading_reversal_count": None,
            "velocity_reversal_count": None,
        }
    if any(_speed_from_frame(frame) is None for frame in frames):
        return {
            **base,
            "status": "unavailable",
            "reason": "missing_robot_velocity",
            "heading_reversal_count": None,
            "velocity_reversal_count": None,
        }
    if any(
        not _finite_json_number(
            frame.get("source_coordinates", {}).get("robot", {}).get("heading")
            if isinstance(frame.get("source_coordinates"), Mapping)
            else None
        )
        for frame in frames
    ):
        return {
            **base,
            "status": "unavailable",
            "reason": "missing_robot_heading",
            "heading_reversal_count": None,
            "velocity_reversal_count": None,
        }
    if route_available and any(
        frame.get("route", {}).get("status") != "available"
        or frame.get("route", {}).get("progress_rate_mps") is None
        for frame in frames
    ):
        return {
            **base,
            "status": "unavailable",
            "reason": "route_frame_progress_rate_unavailable",
            "heading_reversal_count": None,
            "velocity_reversal_count": None,
        }
    summary = summarize_reversals(
        frames,
        speed_getter=_speed_from_frame,
        heading_delta_threshold_rad=_profiles()["reversal_profile"]["heading_delta_threshold_rad"],
    )
    return {**summary, "status": "available", "reason": "coverage_complete"}


def _coverage_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(frames)
    if total == 0:
        empty = {
            "status": "unavailable",
            "frame_count": 0,
            "available_frame_count": 0,
            "missing_frame_count": 0,
            "reason": "no_diagnostic_frames",
        }
        return {
            "frame_count": 0,
            "relative_interaction": dict(empty),
            "proxy_surface_clearance": {**empty, "missing_radius_frame_count": 0},
        }
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
    episode_frames: Sequence[Mapping[str, Any]] | None = None,
    focal_actor_id: object,
    focal_interval: tuple[float, float] | None = None,
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
        _collision_event_anchor(
            trace,
            frames=episode_frames if episode_frames is not None else frames,
            focal_actor_id=focal_actor_id,
            focal_interval=focal_interval,
        ),
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
    if not isinstance(selected, Mapping) or not _finite_json_number(selected.get("time_s")):
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
        if _finite_json_number(frame["relative_interaction"].get("proxy_surface_clearance_m"))
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
        if _finite_json_number(value):
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
        if _finite_json_number(value) and abs(float(value)) >= threshold:
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
        and _finite_json_number(command.get(key))
        for frame in frames
    )


def _has_command_gap(frames: Sequence[Mapping[str, Any]], key: str) -> bool:
    return any(
        not (
            isinstance(command := frame["commands"].get("commanded"), Mapping)
            and _finite_json_number(command.get(key))
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
    return any(
        any(signal["observed"] for signal in _canonical_collision_signals(frame.planner))
        for frame in trace.frames
    )


def _has_proxy_clearance_signal(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _finite_json_number(frame["relative_interaction"].get("proxy_surface_clearance_m"))
        for frame in frames
    )


def _has_proxy_clearance_gap(frames: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        not (
            _finite_json_number(frame["relative_interaction"].get("proxy_surface_clearance_m"))
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
    focal_interval: tuple[float, float] | None,
) -> dict[str, Any]:
    state = _collision_anchor_state(
        trace,
        frames,
        focal_actor_id=focal_actor_id,
        focal_interval=focal_interval,
    )
    if state["status"] == "available":
        focal_binding = state["focal_binding"]
        event = _event_from_condition(
            "exact_collision_event",
            state["frame"],
            actor_id=focal_binding.get("actor_id")
            if focal_binding.get("status") == "available"
            else None,
            source_fields=["planner.event_ledger.collision_events"],
            absent_status="unavailable",
            zone_id=None,
        )
        event["time_s"] = float(state["collision_time"])
        event["collision_partner_id"] = state.get("collision_partner_id")
        event["collision_partner_type"] = state.get("collision_partner_type")
        event["focal_binding"] = focal_binding
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
    if state.get("reason") is not None:
        event["reason"] = str(state["reason"])
    elif state["observed"]:
        event["reason"] = "collision_observed_time_unavailable"
    if state["observed"]:
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
    focal_interval: tuple[float, float] | None,
) -> dict[str, Any]:
    boolean_observed = False
    trace_bounds = _trace_time_bounds(trace)
    candidates: list[tuple[float, int, int, SimulationTraceFrame, Mapping[str, Any]]] = []
    for frame_index, trace_frame in enumerate(trace.frames):
        signals = _canonical_collision_signals(trace_frame.planner)
        boolean_observed = boolean_observed or any(signal["observed"] for signal in signals)
        for signal_index, signal in enumerate(signals):
            if signal.get("source") == "invalid_collision_event_record_shape":
                return {
                    "status": "unavailable",
                    "observed": False,
                    "reason": "invalid_collision_event_record_shape",
                }
            collision_time = signal.get("collision_time")
            if not _finite_json_number(collision_time):
                continue
            candidates.append(
                (float(collision_time), frame_index, signal_index, trace_frame, signal)
            )
    if not candidates:
        return {"status": "unavailable", "observed": boolean_observed}
    collision_time, _, _, _, signal = min(candidates, key=lambda item: item[:3])
    if trace_bounds is not None and not (trace_bounds[0] <= collision_time <= trace_bounds[1]):
        return {
            "status": "unavailable",
            "observed": True,
            "reason": "collision_time_outside_trace_sample_bounds",
        }
    frame = _frame_for_collision_time(frames, collision_time)
    if frame is None:
        return {
            "status": "unavailable",
            "observed": True,
            "reason": "collision_frame_unavailable",
        }
    binds_focal = _collision_binds_focal(signal, focal_actor_id)
    if not binds_focal:
        focal_binding = {
            "status": "unavailable",
            "reason": "collision_partner_not_focal_actor",
        }
    elif focal_interval is not None and not (
        focal_interval[0] <= collision_time <= focal_interval[1]
    ):
        focal_binding = {
            "status": "unavailable",
            "reason": "collision_time_outside_encounter_interval",
        }
    else:
        focal_binding = {
            "status": "available",
            "reason": "collision_partner_matches_focal_actor",
            "actor_id": str(focal_actor_id),
        }
    return {
        "status": "available",
        "observed": True,
        "frame": frame,
        "collision_time": collision_time,
        "collision_partner_id": signal.get("collision_partner_id"),
        "collision_partner_type": signal.get("collision_partner_type"),
        "focal_binding": focal_binding,
    }


def _trace_time_bounds(trace: SimulationTraceExport) -> tuple[float, float] | None:
    times = [frame.time_s for frame in trace.frames if math.isfinite(float(frame.time_s))]
    if not times:
        return None
    return min(times), max(times)


def _collision_binds_focal(signal: Mapping[str, Any], focal_actor_id: object) -> bool:
    return (
        focal_actor_id is not None
        and signal.get("collision_partner_type") == "pedestrian"
        and signal.get("collision_partner_id") == str(focal_actor_id)
    )


def _frame_for_collision_time(
    frames: Sequence[Mapping[str, Any]],
    collision_time: float,
) -> Mapping[str, Any] | None:
    if not frames:
        return None
    return next(
        (frame for frame in frames if float(frame["time_s"]) >= collision_time),
        max(
            frames,
            key=lambda frame: (
                float(frame["time_s"]) <= collision_time,
                -abs(float(frame["time_s"]) - collision_time),
            ),
        ),
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
    return _canonical_collision_signals(planner)[0]


def _canonical_collision_signals(planner: Mapping[str, Any]) -> list[dict[str, Any]]:
    ledger_signal = _ledger_collision_signal(planner.get("event_ledger"))
    if isinstance(ledger_signal.get("signals"), list):
        return list(ledger_signal["signals"])
    if ledger_signal.get("collision_time") is not None or ledger_signal.get("source") == (
        "invalid_collision_event_record_shape"
    ):
        return [ledger_signal]
    outcome = planner.get("outcome")
    if isinstance(outcome, Mapping):
        collision = outcome.get("collision_event")
        if collision is True:
            return [{"observed": True, "source": "outcome.collision_event"}]
        if collision is False or collision is None:
            pass
        else:
            return [{"observed": False, "source": "invalid_outcome_collision_shape"}]
    if ledger_signal["observed"]:
        return [ledger_signal]
    return [{"observed": False, "source": "no_canonical_collision_signal"}]


def _ledger_collision_signal(ledger: object) -> dict[str, Any]:
    if not (
        isinstance(ledger, Mapping) and ledger.get("schema_version") == "EpisodeEventLedger.v2"
    ):
        return {"observed": False, "source": "event_ledger_unavailable"}
    records = ledger.get("collision_events")
    if not (isinstance(records, Sequence) and not isinstance(records, str | bytes)):
        return {"observed": False, "source": "event_ledger_collision_events_unavailable"}
    signals: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if not _valid_collision_record(record):
            return {"observed": False, "source": "invalid_collision_event_record_shape"}
        collision_time = record.get("collision_time")
        signals.append(
            {
                "observed": True,
                "source": "event_ledger.collision_events",
                "collision_time": float(collision_time),
                "collision_partner_id": record["collision_partner_id"],
                "collision_partner_type": str(record["collision_partner_type"]),
            }
        )
    if signals:
        return {
            "observed": True,
            "source": "event_ledger.collision_events",
            "signals": signals,
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
    rows = []
    for frame in frames:
        nearest = _nearest_source_actor(frame)
        if nearest is None:
            continue
        rows.append(
            {
                "step": frame["step"],
                "time_s": frame["time_s"],
                "actor_id": nearest["actor_id"],
                "center_distance_m": nearest["center_distance_m"],
            }
        )
    return {
        "status": "available" if rows else "unavailable",
        "reason": "nearest_actor_by_center_distance" if rows else "no_pedestrians_in_trace",
        "series": rows,
    }


def _actor_switch_events(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_actor: str | None = None
    for frame in frames:
        nearest = _nearest_source_actor(frame)
        actor_id = nearest.get("actor_id") if nearest is not None else None
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


def _nearest_source_actor(frame: Mapping[str, Any]) -> dict[str, Any] | None:
    source = frame.get("source_coordinates")
    if not isinstance(source, Mapping):
        return None
    robot = source.get("robot")
    robot_pos = _vector2(robot.get("position")) if isinstance(robot, Mapping) else None
    actors = source.get("contextual_actors")
    if robot_pos is None or not isinstance(actors, list):
        return None
    candidates = []
    for actor in actors:
        if not isinstance(actor, Mapping) or not isinstance(actor.get("actor_id"), str):
            continue
        actor_pos = _vector2(actor.get("position"))
        if actor_pos is None:
            continue
        candidates.append(
            {
                "actor_id": str(actor["actor_id"]),
                "center_distance_m": _distance(robot_pos, actor_pos),
            }
        )
    return (
        min(candidates, key=lambda item: (item["center_distance_m"], item["actor_id"]))
        if candidates
        else None
    )


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
        if _finite_json_number(frame["relative_interaction"].get("proxy_surface_clearance_m"))
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
    if robot_pos is None:
        return {"status": "unavailable", "reason": "missing_robot_position"}
    if not frame.pedestrians:
        return {"status": "unavailable", "reason": "no_pedestrians_in_frame"}
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
    matches = [
        pedestrian for pedestrian in frame.pedestrians if str(pedestrian.get("id")) == target
    ]
    return matches[0] if len(matches) == 1 else None


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
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or not all(_finite_json_number(item) for item in value)
    ):
        return None
    return float(value[0]), float(value[1])


def _radius(actor: Mapping[str, Any]) -> float | None:
    value = actor.get("radius")
    if _finite_json_number(value):
        return float(value)
    return None


def _finite_float(value: Any) -> float | None:
    if _finite_json_number(value):
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
