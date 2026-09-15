"""SREV-03 video-sync component: map capture frames to simulation time.

This module owns the SREV-03 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts and builds an explicit
frame-to-simulation presentation map from fixture-provided capture metadata. It
never advances or mutates simulation state, never guesses frame alignment, and
never generates media: all timestamps come from explicit inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
)

COMPONENT_ID = "srev03-video-sync"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = (
    "capture-frames",
    "sim-stamps",
)
OPTIONAL_CAPABILITIES = ("camera-calibration",)

OUTPUT_MAPPING_FILENAME = "media-mapping.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
ADMISSION_NOT_EVALUATED = "not_evaluated"
EVIDENCE_STATUS = "diagnostic_only"

DEFAULT_PRESENTATION = {"width": 1920, "height": 1080, "fps": 30.0, "speed": 1.0}
DEFAULT_TIMESTAMP_TOLERANCE_S = 0.0
IDENTITY_NAMESPACE = "shared-episode-reset-v1"

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("media-mapping.v1", "missing-capability-report.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)

_DESCRIPTOR_DOCUMENT = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": list(_DESCRIPTOR.output_types),
}

# Keep the public descriptor checked against the shared contract at import time.
component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)

# Presentation preset for tests (mirrors the SREV-01 test preset shape).
_TEST_PRESET = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}


@dataclass
class _ImportOutcome:
    """Per-source load result before mapping."""

    payload: dict[str, Any] | None = None
    file_sha256: str | None = None
    diagnostics: list[str] = field(default_factory=list)
    ref: SourceRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    declared_sha256: str | None = None
    integrity: str = "unverifiable"
    availability: str = "unavailable"
    selected: bool = False


def descriptor() -> dict[str, Any]:
    """Return this component's versioned capability descriptor."""
    return descriptor_document()


def descriptor_document() -> dict[str, Any]:
    """Return a JSON-safe ``component-descriptor.v1`` document."""
    return json.loads(json.dumps(_DESCRIPTOR_DOCUMENT, allow_nan=False))


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write strict-JSON.

    Returns:
        Hex digest of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _reject_nonfinite_json(value: str) -> Any:
    """Reject JSON extensions such as ``NaN`` and ``Infinity``."""
    raise ValueError(f"non-strict JSON constant: {value}")


def _load_strict_json(text: str | bytes) -> Any:
    """Parse JSON without accepting non-standard non-finite constants.

    Returns:
        Parsed JSON value.
    """
    return json.loads(text, parse_constant=_reject_nonfinite_json)


def _safe_identity(value: Any, *, default: str) -> str:
    """Return a non-empty string identity suitable for a failure envelope."""
    return value if isinstance(value, str) and value else default


def _request_identity(request: Any) -> tuple[str, str]:
    """Extract safe request/component identities from typed or malformed input.

    Returns:
        Non-empty request and component identifiers.
    """
    if isinstance(request, ComponentRequest):
        return (
            _safe_identity(request.request_id, default="unknown"),
            _safe_identity(request.component_id, default=COMPONENT_ID),
        )
    if isinstance(request, Mapping):
        return (
            _safe_identity(request.get("request_id"), default="unknown"),
            _safe_identity(request.get("component_id"), default=COMPONENT_ID),
        )
    return "unknown", COMPONENT_ID


def _result(
    request_id: str,
    component_id: str,
    status: str,
    *,
    reason: str = "",
    artifacts: tuple[dict[str, Any], ...] = (),
    diagnostics: tuple[dict[str, Any], ...] = (),
    provenance: dict[str, Any] | None = None,
) -> ComponentResult:
    """Build a result with the diagnostic-only provenance boundary present.

    Returns:
        Component result with shared status and provenance fields.
    """
    result_provenance: dict[str, Any] = {
        "component_version": COMPONENT_VERSION,
        "admission": ADMISSION_NOT_EVALUATED,
        "evidence_status": EVIDENCE_STATUS,
    }
    if provenance:
        result_provenance.update(provenance)
    return ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=status,
        artifacts=artifacts,
        diagnostics=diagnostics,
        provenance=result_provenance,
        reason=reason,
    )


def _failed_result(request: Any, reason: str) -> ComponentResult:
    """Return a stable failed result for malformed or unexpected input.

    Returns:
        Contract-compatible failed component result.
    """
    request_id, component_id = _request_identity(request)
    return _result(request_id, component_id, STATUS_FAILED, reason=reason)


def _validate_source_ref(ref: Any, index: int, seen_artifacts: set[str]) -> list[str]:
    """Validate one typed source reference.

    Returns:
        Stable validation reason codes for the source.
    """
    if not isinstance(ref, SourceRef):
        return [f"invalid_request: source_{index} must be a SourceRef"]
    errors: list[str] = []
    if not isinstance(ref.artifact_id, str) or not ref.artifact_id:
        errors.append(f"invalid_request: source_{index} artifact_id is invalid")
    elif ref.artifact_id in seen_artifacts:
        errors.append(f"invalid_request: duplicate artifact_id: {ref.artifact_id}")
    if isinstance(ref.artifact_id, str) and ref.artifact_id:
        seen_artifacts.add(ref.artifact_id)
    if not isinstance(ref.uri, str) or not ref.uri:
        errors.append(f"invalid_request: source_{index} uri is invalid")
    elif _unsafe_relative_path(ref.uri):
        errors.append(f"invalid_request: source_{index} uri traversal is rejected")
    if not isinstance(ref.format, str) or not ref.format:
        errors.append(f"invalid_request: source_{index} format is invalid")
    return errors


def _unsafe_relative_path(value: str) -> bool:
    """Return whether a path is absolute or contains parent traversal.

    The Windows check keeps requests portable when a Windows-style URI is
    validated on a POSIX worker.
    """
    candidate = Path(value)
    windows = PureWindowsPath(value)
    return (
        candidate.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in candidate.parts
        or ".." in windows.parts
    )


def _validate_request_config(config: Any) -> list[str]:
    """Validate the request config object and its strict-JSON safety.

    Returns:
        Stable validation reason codes for the config.
    """
    if not isinstance(config, dict):
        return ["invalid_request: config must be an object"]
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return ["invalid_request: config must be strict-JSON safe"]
    return []


def _validate_request_shape(request: Any) -> list[str]:
    """Validate the runtime shape expected by ``run`` before field access.

    Returns:
        Stable validation reason codes.
    """
    if not isinstance(request, ComponentRequest):
        return ["invalid_request: expected ComponentRequest"]
    errors: list[str] = []
    if not isinstance(request.request_id, str) or not request.request_id:
        errors.append("invalid_request: request_id must be a non-empty string")
    if not isinstance(request.component_id, str) or not request.component_id:
        errors.append("invalid_request: component_id must be a non-empty string")
    if not isinstance(request.output_directory, str) or not request.output_directory:
        errors.append("invalid_request: output_directory must be a non-empty string")
    elif _unsafe_relative_path(request.output_directory):
        errors.append("invalid_request: output_directory path traversal is rejected")
    errors.extend(_validate_request_config(request.config))
    if not isinstance(request.sources, (tuple, list)) or not request.sources:
        errors.append("invalid_request: sources must be a non-empty sequence")
    else:
        seen_artifacts: set[str] = set()
        for index, ref in enumerate(request.sources):
            errors.extend(_validate_source_ref(ref, index, seen_artifacts))
    if not isinstance(request.required_capabilities, (tuple, list)) or any(
        not isinstance(value, str) or not value for value in request.required_capabilities
    ):
        errors.append("invalid_request: required_capabilities must contain non-empty strings")
    return errors


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies it.
    """
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    if not isinstance(minimum, str) or not minimum:
        return "incompatible_component_version: malformed min_component_version"
    try:
        wanted = int(minimum.split(".", maxsplit=1)[0])
        ours = int(COMPONENT_VERSION.split(".", maxsplit=1)[0])
    except ValueError:
        return f"incompatible_component_version: malformed min_component_version: {minimum!r}"
    if wanted > ours:
        return f"incompatible_component_version: request needs v{wanted}, component is v{ours}"
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject requests this component cannot serve.

    Returns:
        An unavailable/failed result, or None when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return _result(
            request.request_id,
            request.component_id,
            STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return _result(
            request.request_id,
            request.component_id,
            STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return _result(
            request.request_id,
            request.component_id,
            STATUS_FAILED,
            reason=version_error,
        )
    return None


def _resolve_path_under_base(path_value: str, base: Path, *, kind: str) -> Path:
    """Resolve a path under ``base`` after checking its realpath containment.

    Returns:
        Resolved path contained by the resolved base directory.
    """
    if _unsafe_relative_path(path_value):
        raise ReviewContractsValidationError(
            [f"unsafe_{kind}_path: absolute or traversal path is rejected: {path_value}"]
        )
    try:
        resolved_base = base.resolve(strict=False)
        resolved = (resolved_base / Path(path_value)).resolve(strict=False)
        resolved.relative_to(resolved_base)
    except (OSError, RuntimeError, ValueError) as error:
        raise ReviewContractsValidationError(
            [f"unsafe_{kind}_path: path resolves outside base: {path_value}"]
        ) from error
    return resolved


def _resolve_source(path_value: str, base: Path) -> Path:
    """Resolve a source URI under the base directory, including symlinks.

    Returns:
        Resolved source path contained by the realpath of ``base``.
    """
    return _resolve_path_under_base(path_value, base, kind="source")


def _resolve_output_directory(root: Path, output_directory: str) -> Path:
    """Resolve an output directory under the base, including symlinks.

    Returns:
        Resolved output directory contained by the realpath of ``root``.
    """
    return _resolve_path_under_base(output_directory, root, kind="output")


def _valid_sha256(value: Any) -> bool:
    """Return whether a value is a 64-character hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _source_metadata(config: dict[str, Any], ref: SourceRef) -> dict[str, Any]:
    """Return validated per-source identity metadata from the request config.

    Returns:
        A copy of the source identity metadata.
    """
    raw_metadata = config.get("source_metadata", {})
    if raw_metadata is None or not isinstance(raw_metadata, dict):
        raise ReviewContractsValidationError(["invalid_config: source_metadata must be an object"])
    declared = raw_metadata.get(ref.artifact_id, {})
    if declared is None or not isinstance(declared, dict):
        raise ReviewContractsValidationError(
            [f"invalid_config: source_metadata.{ref.artifact_id} must be an object"]
        )
    metadata = {
        field_name: getattr(ref, field_name)
        for field_name in ("schema", "sha256", "source_commit", "config_identity")
        if getattr(ref, field_name)
    }
    metadata.update(declared)
    errors: list[str] = []
    for field_name in ("schema", "source_commit", "config_identity"):
        if field_name in metadata and not isinstance(metadata[field_name], str):
            errors.append(
                f"invalid_config: source_metadata.{ref.artifact_id}.{field_name} must be a string"
            )
    if "sha256" in metadata and not isinstance(metadata["sha256"], str):
        errors.append(f"invalid_config: source_metadata.{ref.artifact_id}.sha256 must be a string")
    if errors:
        raise ReviewContractsValidationError(errors)
    return dict(metadata)


def _source_record(outcome: _ImportOutcome) -> dict[str, Any]:
    """Project one source outcome into stable, source-faithful provenance.

    Returns:
        JSON-safe source identity and integrity record.
    """
    ref = outcome.ref
    if ref is None:  # pragma: no cover - defensive invariant guard
        raise ReviewContractsValidationError(["internal_error: source outcome has no reference"])
    metadata = outcome.metadata
    return {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "format": ref.format,
        "schema": metadata.get("schema"),
        "source_commit": metadata.get("source_commit"),
        "config_identity": metadata.get("config_identity"),
        "declared_sha256": outcome.declared_sha256,
        "sha256": outcome.file_sha256,
        "source_sha256": outcome.file_sha256,
        "computed_sha256": outcome.file_sha256,
        "integrity": outcome.integrity,
        "integrity_status": outcome.integrity,
        "availability": outcome.availability,
        "selected": outcome.selected,
        "admission": ADMISSION_NOT_EVALUATED,
    }


def _load_document(ref: SourceRef, metadata: dict[str, Any], root: Path) -> _ImportOutcome:
    """Read and parse one JSON source without modifying it.

    Returns:
        Import outcome with the parsed payload or a diagnostic.
    """
    expected_sha256 = metadata.get("sha256")
    outcome = _ImportOutcome(
        ref=ref,
        metadata=metadata,
        declared_sha256=expected_sha256,
        integrity=(
            "not_declared"
            if expected_sha256 is None
            else ("unverifiable" if _valid_sha256(expected_sha256) else "invalid_declared_sha256")
        ),
    )
    if expected_sha256 is not None and not _valid_sha256(expected_sha256):
        outcome.diagnostics.append(f"{ref.artifact_id}: declared_sha256_invalid")
    try:
        raw = _resolve_source(ref.uri, root).read_bytes()
    except ReviewContractsValidationError:
        outcome.diagnostics.append(f"{ref.artifact_id}: source_uri_rejected")
        outcome.availability = "source_uri_rejected"
        return outcome
    except (OSError, TypeError):
        outcome.diagnostics.append(f"{ref.artifact_id}: source_unreadable")
        outcome.availability = "source_unreadable"
        return outcome
    outcome.file_sha256 = _sha256_bytes(raw)
    if expected_sha256 is None:
        outcome.integrity = "not_declared"
    elif _valid_sha256(expected_sha256):
        outcome.integrity = (
            "match" if expected_sha256.lower() == outcome.file_sha256 else "mismatch"
        )
        if outcome.integrity == "mismatch":
            outcome.diagnostics.append(f"{ref.artifact_id}: stale_digest")
    try:
        payload = _load_strict_json(raw.decode("utf-8"))
    except UnicodeDecodeError:
        outcome.diagnostics.append(f"{ref.artifact_id}: source_not_utf8")
        outcome.availability = "source_not_utf8"
        return outcome
    except (ValueError, RecursionError):
        outcome.diagnostics.append(f"{ref.artifact_id}: source_not_strict_json")
        outcome.availability = "source_not_strict_json"
        return outcome
    if not isinstance(payload, dict):
        outcome.diagnostics.append(f"{ref.artifact_id}: source_not_json_object")
        outcome.availability = "source_not_json_object"
        return outcome
    outcome.payload = payload
    outcome.availability = "ok"
    return outcome


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or None for malformed input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _parse_frame_row(item: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one capture frame row.

    Returns:
        The normalized frame and an optional stable malformed-row diagnostic.
    """
    if not isinstance(item, dict):
        return None, f"frame_row_{index}_malformed"
    frame_index = item.get("frame_index")
    pts = _finite_number(item.get("pts_s"))
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        or pts is None
    ):
        return None, f"frame_row_{index}_malformed"
    frame = {"frame_index": frame_index, "pts_s": pts}
    for identity in ("episode_id", "reset_id"):
        if identity in item:
            value = item[identity]
            if not isinstance(value, str) or not value:
                return None, f"frame_row_{index}_malformed"
            frame[identity] = value
    return frame, None


def _parse_frame_rows(raw_frames: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and validate capture frame rows.

    Returns:
        Tuple of (valid frames, diagnostic codes).
    """
    if not isinstance(raw_frames, list) or not raw_frames:
        return [], ["capture_frames_missing_or_empty"]
    frames: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for index, item in enumerate(raw_frames):
        frame, diagnostic = _parse_frame_row(item, index)
        if frame is None:
            diagnostics.append(diagnostic or f"frame_row_{index}_malformed")
        else:
            frames.append(frame)
    frames.sort(key=lambda row: row["frame_index"])
    seen: set[int] = set()
    duplicates: set[int] = set()
    for row in frames:
        frame_index = row["frame_index"]
        if frame_index in seen:
            duplicates.add(frame_index)
        seen.add(frame_index)
    diagnostics.extend(f"duplicate_frame_index:{index}" for index in sorted(duplicates))
    if duplicates:
        return [], diagnostics
    return frames, diagnostics


def _parse_step_rows(raw_steps: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and validate simulation stamp rows.

    Returns:
        Tuple of (valid steps sorted by time, diagnostic codes).
    """
    steps: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    if not isinstance(raw_steps, list) or not raw_steps:
        return [], ["sim_stamps_missing_or_empty"]
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            diagnostics.append(f"step_row_{index}_malformed")
            continue
        step = item.get("step")
        when = _finite_number(item.get("time_s"))
        episode_id = item.get("episode_id")
        reset_id = item.get("reset_id")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or when is None
            or not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(reset_id, str)
            or not reset_id
        ):
            diagnostics.append(f"step_row_{index}_malformed")
            continue
        steps.append(
            {
                "step": step,
                "time_s": when,
                "episode_id": episode_id,
                "reset_id": reset_id,
            }
        )
    return steps, diagnostics


def _detect_sampling_gaps(
    frames: list[dict[str, Any]], fps_nominal: float
) -> tuple[list[int], list[str]]:
    """Detect skipped indexes and nonuniform presentation deltas.

    Returns:
        Tuple of (skipped frame indexes, diagnostic codes).
    """
    seen = [row["frame_index"] for row in frames]
    skipped = [i for i in range(seen[0], seen[-1] + 1) if i not in set(seen)]
    diagnostics = []
    if skipped:
        diagnostics.append(f"skipped_frames:{','.join(str(i) for i in skipped)}")
    expected_delta = 1.0 / fps_nominal
    for first, second in itertools.pairwise(frames):
        if abs(second["pts_s"] - first["pts_s"] - expected_delta) > 0.5 * expected_delta:
            diagnostics.append(
                f"nonuniform_sampling:frames_{first['frame_index']}_{second['frame_index']}"
            )
            break
    return skipped, diagnostics


def _boundary_records(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record source-order episode/reset transitions without renumbering them.

    Returns:
        Boundary records, including the initial source identity.
    """
    boundaries: list[dict[str, Any]] = []
    previous: tuple[str, str] | None = None
    for index, row in enumerate(steps):
        identity = (row["episode_id"], row["reset_id"])
        if previous != identity:
            boundaries.append(
                {
                    "source_index": index,
                    "time_s": row["time_s"],
                    "step": row["step"],
                    "episode_id": row["episode_id"],
                    "reset_id": row["reset_id"],
                    "kind": "initial" if previous is None else "episode_or_reset_change",
                }
            )
            previous = identity
    return boundaries


def _timeline_diagnostics(steps: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """Find repeated or regressing simulation times while preserving source order.

    Returns:
        Diagnostic codes and whether the timeline cannot be mapped safely.
    """
    diagnostics: list[str] = []
    invalid = False
    for previous, current in itertools.pairwise(steps):
        if current["time_s"] == previous["time_s"]:
            diagnostics.append(f"ambiguous_repeated_sim_time:{current['time_s']}")
            invalid = True
        elif current["time_s"] < previous["time_s"]:
            diagnostics.append(f"nonmonotonic_sim_time:{previous['time_s']}_{current['time_s']}")
            invalid = True
    if invalid:
        diagnostics.append("ambiguous_sim_stamp_timeline")
    return sorted(set(diagnostics)), invalid


def _within_timestamp_tolerance(error: float, tolerance: float) -> bool:
    """Return whether a timestamp error satisfies the inclusive policy.

    The zero-tolerance policy remains exact. For a positive tolerance, a small
    representational boundary allowance avoids rejecting a decimal boundary
    solely because subtraction rounded a few bits above the supplied tolerance.
    """
    if tolerance == 0.0:
        return error == 0.0
    return error <= tolerance or math.isclose(
        error,
        tolerance,
        rel_tol=0.0,
        abs_tol=2.0 * math.ulp(max(abs(error), abs(tolerance))),
    )


def _select_sim_anchor(
    frame: dict[str, Any],
    steps: list[dict[str, Any]],
    sim_time: float,
    timestamp_tolerance_s: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Select one simulation stamp or return an explicit unavailable reason.

    Returns:
        The selected anchor, an unavailable reason, and the nearest temporal
        error in seconds.
    """
    candidates = [
        step
        for step in steps
        if _within_timestamp_tolerance(abs(step["time_s"] - sim_time), timestamp_tolerance_s)
    ]
    nearest_error = min(abs(step["time_s"] - sim_time) for step in steps)
    if not candidates:
        return None, "no_sim_stamp_within_tolerance", nearest_error
    if len(candidates) > 1:
        return None, "ambiguous_sim_stamp_match", nearest_error
    anchor = candidates[0]
    temporal_error = abs(anchor["time_s"] - sim_time)
    if any(
        identity in frame and frame[identity] != anchor[identity]
        for identity in ("episode_id", "reset_id")
    ):
        return None, "capture_sim_identity_mismatch", temporal_error
    return anchor, None, temporal_error


def _unavailable_frame(
    frame: dict[str, Any],
    sim_time: float,
    reason: str,
    camera: dict[str, Any],
    *,
    temporal_error_s: float | None = None,
) -> dict[str, Any]:
    """Represent a captured frame that has no trustworthy simulation anchor.

    Returns:
        An unavailable frame record without simulation identity fields.
    """
    result: dict[str, Any] = {
        "frame_index": frame["frame_index"],
        "pts_s": frame["pts_s"],
        "sim_time_s": sim_time,
        "status": STATUS_UNAVAILABLE,
        "reason": reason,
        "camera": camera,
    }
    if temporal_error_s is not None:
        result["temporal_error_s"] = temporal_error_s
    for identity in ("episode_id", "reset_id"):
        if identity in frame:
            result[f"capture_{identity}"] = frame[identity]
    return result


def _map_frames(
    frames: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    time_origin: float,
    speed: float,
    camera: dict[str, Any],
    timestamp_tolerance_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Anchor frames only to one simulation stamp within explicit tolerance.

    Returns:
        Tuple of (mapped entries, unavailable frame records, diagnostic codes).
    """
    diagnostics: list[str] = []
    unavailable: list[dict[str, Any]] = []
    first_time = time_origin + frames[0]["pts_s"] / speed
    last_time = time_origin + frames[-1]["pts_s"] / speed
    outside_range = first_time < steps[0]["time_s"] or last_time > steps[-1]["time_s"]
    if outside_range:
        diagnostics.append("frames_outside_stamp_range")
    timeline_diagnostics, timeline_invalid = _timeline_diagnostics(steps)
    diagnostics.extend(timeline_diagnostics)
    mapping: list[dict[str, Any]] = []
    for frame in frames:
        sim_time = time_origin + frame["pts_s"] / speed
        if timeline_invalid:
            reason = "ambiguous_sim_stamp_timeline"
            unavailable.append(_unavailable_frame(frame, sim_time, reason, camera))
            diagnostics.append(f"frame_{frame['frame_index']}_{reason}")
            continue
        if sim_time < steps[0]["time_s"]:
            reason = "before_first_sim_stamp"
            unavailable.append(
                _unavailable_frame(
                    frame,
                    sim_time,
                    reason,
                    camera,
                    temporal_error_s=steps[0]["time_s"] - sim_time,
                )
            )
            diagnostics.append(f"frame_{frame['frame_index']}_{reason}")
            continue
        if sim_time > steps[-1]["time_s"]:
            reason = "after_last_sim_stamp"
            unavailable.append(
                _unavailable_frame(
                    frame,
                    sim_time,
                    reason,
                    camera,
                    temporal_error_s=sim_time - steps[-1]["time_s"],
                )
            )
            diagnostics.append(f"frame_{frame['frame_index']}_{reason}")
            continue
        anchor, reason, temporal_error = _select_sim_anchor(
            frame, steps, sim_time, timestamp_tolerance_s
        )
        if reason is not None:
            unavailable.append(
                _unavailable_frame(
                    frame,
                    sim_time,
                    reason,
                    camera,
                    temporal_error_s=temporal_error,
                )
            )
            diagnostics.append(f"frame_{frame['frame_index']}_{reason}")
            continue
        if anchor is None:  # pragma: no cover - invariant guard
            raise RuntimeError("timestamp anchor missing without an unavailable reason")
        entry: dict[str, Any] = {
            "frame_index": frame["frame_index"],
            "pts_s": frame["pts_s"],
            "sim_time_s": sim_time,
            "sim_stamp_time_s": anchor["time_s"],
            "temporal_error_s": temporal_error,
            "sim_step": anchor["step"],
            "episode_id": anchor["episode_id"],
            "reset_id": anchor["reset_id"],
            "camera": camera,
            "status": "mapped",
        }
        for identity in ("episode_id", "reset_id"):
            if identity in frame:
                entry[f"capture_{identity}"] = frame[identity]
        mapping.append(entry)
    return mapping, unavailable, sorted(set(diagnostics))


def _validate_presentation_dimension(
    raw: Mapping[str, Any], name: str
) -> tuple[int | None, str | None]:
    """Validate one positive integer presentation dimension.

    Returns:
        The dimension or a stable invalid-field reason.
    """
    if name not in raw:
        return None, None
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, f"presentation_{name}_invalid"
    return value, None


def _validate_presentation_rate(
    raw: Mapping[str, Any], name: str
) -> tuple[float | None, str | None]:
    """Validate one finite positive presentation rate.

    Returns:
        The rate or a stable invalid-field reason.
    """
    if name not in raw:
        return None, None
    value = _finite_number(raw[name])
    if value is None or value <= 0:
        return None, f"presentation_{name}_invalid"
    return value, None


def _presentation_base(
    config: dict[str, Any], raw: Any
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Resolve the presentation object, preset, and defaults.

    Returns:
        Raw presentation object, effective defaults, and an optional error.
    """
    if not isinstance(raw, dict):
        return None, None, "presentation_invalid: expected an object"
    preset = config.get("preset", "default")
    if not isinstance(preset, str) or preset not in {"default", "test"}:
        return None, None, "presentation_preset_invalid"
    unknown = sorted(set(raw) - {"width", "height", "fps", "speed", "preset"})
    if unknown:
        return None, None, f"presentation_unknown_field:{unknown[0]}"
    if "preset" in raw and (
        not isinstance(raw["preset"], str)
        or raw["preset"] not in {"default", "test"}
        or raw["preset"] != preset
    ):
        return None, None, "presentation_preset_invalid"
    defaults = _TEST_PRESET if preset == "test" else DEFAULT_PRESENTATION
    return raw, {**defaults, "preset": preset}, None


def _validate_presentation(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate presentation settings without coercing malformed values.

    Returns:
        Effective presentation settings and stable validation reason codes.
    """
    raw, effective, error = _presentation_base(config, config.get("presentation", {}))
    if error:
        return {}, [error]
    if raw is None or effective is None:  # pragma: no cover - invariant guard
        return {}, ["presentation_invalid: missing validation context"]
    for name in ("width", "height"):
        value, error = _validate_presentation_dimension(raw, name)
        if error:
            return {}, [error]
        if value is not None:
            effective[name] = value
    for name in ("fps", "speed"):
        value, error = _validate_presentation_rate(raw, name)
        if error:
            return {}, [error]
        if value is not None:
            effective[name] = value
    return effective, []


def _build_mapping(
    frames_doc: dict[str, Any],
    stamps_doc: dict[str, Any],
    config: dict[str, Any],
    selected_sources: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Build the explicit frame-to-simulation presentation map.

    Returns:
        Tuple of (mapping document, diagnostic codes). Skipped frames,
        nonuniform sampling, and reset boundaries are recorded explicitly;
        alignment is never guessed.
    """
    diagnostics: list[str] = []
    raw_frames = frames_doc.get("frames")
    raw_steps = stamps_doc.get("steps")
    fps_nominal = _finite_number(frames_doc.get("fps_nominal"))
    if fps_nominal is None or fps_nominal <= 0:
        return {}, ["fps_nominal_missing_or_invalid"]
    presentation, presentation_errors = _validate_presentation(config)
    if presentation_errors:
        return {}, presentation_errors
    speed = presentation["speed"]
    time_origin = _finite_number(config.get("time_origin_s", 0.0))
    if time_origin is None:
        return {}, ["time_origin_invalid"]
    timestamp_tolerance = _finite_number(
        config.get("timestamp_tolerance_s", DEFAULT_TIMESTAMP_TOLERANCE_S)
    )
    if timestamp_tolerance is None or timestamp_tolerance < 0:
        return {}, ["timestamp_tolerance_invalid"]
    frames, diagnostics = _parse_frame_rows(raw_frames)
    steps, step_diagnostics = _parse_step_rows(raw_steps)
    diagnostics.extend(step_diagnostics)
    if not frames or not steps:
        return {}, diagnostics or ["no_mappable_rows"]
    skipped, gap_diagnostics = _detect_sampling_gaps(frames, fps_nominal)
    diagnostics.extend(gap_diagnostics)
    camera = frames_doc.get("camera", {})
    if not isinstance(camera, dict):
        return {}, [*diagnostics, "camera_invalid: expected an object"]
    mapping, unavailable, map_diagnostics = _map_frames(
        frames,
        steps,
        time_origin,
        speed,
        camera,
        timestamp_tolerance,
    )
    diagnostics.extend(map_diagnostics)
    all_frame_records = sorted([*mapping, *unavailable], key=lambda row: row["frame_index"])
    boundaries = _boundary_records(steps)
    episodes = list(dict.fromkeys(row["episode_id"] for row in steps))
    resets = list(dict.fromkeys(row["reset_id"] for row in steps))
    provenance = {
        "sources": selected_sources,
        "source_integrity": {
            source["artifact_id"]: source["integrity"] for source in selected_sources
        },
        "admission": ADMISSION_NOT_EVALUATED,
        "evidence_status": EVIDENCE_STATUS,
    }
    document = {
        "schema_version": "media-mapping.v1",
        "fps_nominal": fps_nominal,
        "speed": speed,
        "time_origin_s": time_origin,
        "timestamp_alignment": {
            "policy": "unique_sim_stamp_within_tolerance",
            "tolerance_s": timestamp_tolerance,
        },
        "identity_namespace": {
            "capture": IDENTITY_NAMESPACE,
            "simulation": IDENTITY_NAMESPACE,
            "rule": "capture episode_id/reset_id must match the selected simulation anchor",
        },
        "presentation": presentation,
        "skipped_frame_indexes": skipped,
        "unavailable_frame_indexes": [row["frame_index"] for row in unavailable],
        "unavailable_frames": unavailable,
        "episode_ids": episodes,
        "reset_ids": resets,
        "episode_reset_boundaries": boundaries,
        "first_frame": all_frame_records[0] if all_frame_records else None,
        "last_frame": all_frame_records[-1] if all_frame_records else None,
        "entries": mapping,
        "sources": selected_sources,
        "selected_sources": selected_sources,
        "provenance": provenance,
    }
    return document, sorted(set(diagnostics))


def _load_sources(
    request: ComponentRequest, root: Path
) -> tuple[dict[str, _ImportOutcome], list[_ImportOutcome], list[str]]:
    """Load the first declared source for each supported format.

    Returns:
        Selected sources by format, all source outcomes, and diagnostics.
    """
    by_format: dict[str, _ImportOutcome] = {}
    outcomes: list[_ImportOutcome] = []
    diagnostics: list[str] = []
    supported_formats = REQUIRED_CAPABILITIES + OPTIONAL_CAPABILITIES
    for ref in request.sources:
        metadata = _source_metadata(request.config, ref)
        outcome = _ImportOutcome(
            ref=ref,
            metadata=metadata,
            declared_sha256=metadata.get("sha256"),
            availability="not_evaluated",
            integrity="not_evaluated",
        )
        if ref.format not in supported_formats:
            diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
            outcome.availability = "unsupported_format"
        elif (
            ref.format in OPTIONAL_CAPABILITIES and ref.format not in request.required_capabilities
        ):
            diagnostics.append(f"{ref.artifact_id}: optional_stream_skipped:{ref.format}")
            outcome.availability = "skipped"
        elif ref.format in by_format:
            diagnostics.append(f"{ref.artifact_id}: duplicate_source_format:{ref.format}")
            outcome.availability = "duplicate_not_selected"
        else:
            outcome = _load_document(ref, metadata, root)
            outcome.selected = True
            by_format[ref.format] = outcome
            diagnostics.extend(outcome.diagnostics)
        outcomes.append(outcome)
    return by_format, outcomes, diagnostics


def _missing_required_capabilities(
    request: ComponentRequest, selected_by_format: dict[str, _ImportOutcome]
) -> list[str]:
    """Return requested capabilities without a usable selected source family.

    Returns:
        Sorted known capabilities that are absent or unreadable.
    """
    missing = {
        capability
        for capability in request.required_capabilities
        if capability in _DESCRIPTOR.required_capabilities + _DESCRIPTOR.optional_capabilities
        and (capability not in selected_by_format or selected_by_format[capability].payload is None)
    }
    return sorted(missing)


def _source_provenance(
    outcomes: list[_ImportOutcome],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Build all-source, selected-source, and integrity provenance views.

    Returns:
        All source records, selected source records, and integrity by artifact ID.
    """
    records = [_source_record(outcome) for outcome in outcomes]
    selected = [record for record in records if record["selected"]]
    integrity = {record["artifact_id"]: record["integrity"] for record in records}
    return records, selected, integrity


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Map capture frames to simulation time with an explicit presentation map.

    Args:
        request: Validated component request with capture-frames and sim-stamps sources.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when every frame maps cleanly,
        ``partial`` on skipped/out-of-range/malformed rows, ``unavailable`` when
        the component or a required capability does not apply, ``failed`` on
        validation, version, collision, or internal errors.
    """
    request_id, component_id = _request_identity(request)
    try:
        shape_errors = _validate_request_shape(request)
        if shape_errors:
            return _result(
                request_id,
                component_id,
                STATUS_FAILED,
                reason="; ".join(shape_errors),
            )
        root = (base if base is not None else Path.cwd()).resolve(strict=False)
        rejected = _reject_not_applicable(request)
        if rejected is not None:
            return rejected
        output_dir = _resolve_output_directory(root, request.output_directory)
        if output_dir.exists():
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=f"output_collision: already exists: {request.output_directory}",
            )
        by_format, all_outcomes, diagnostics = _load_sources(request, root)
        source_records, selected_sources, source_integrity = _source_provenance(all_outcomes)
        provenance = {
            "sources": source_records,
            "selected_sources": selected_sources,
            "source_integrity": source_integrity,
        }
        missing_capabilities = _missing_required_capabilities(request, by_format)
        if missing_capabilities:
            diagnostics.extend(
                f"missing_required_capability:{capability}" for capability in missing_capabilities
            )
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE,
                reason="missing_required_capabilities: " + ", ".join(missing_capabilities),
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance=provenance,
            )
        missing_source_families = sorted(
            capability
            for capability in REQUIRED_CAPABILITIES
            if capability not in by_format or by_format[capability].payload is None
        )
        if missing_source_families:
            diagnostics.append("required_source_family_missing")
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=(
                    "required_source_family_missing: "
                    + ", ".join(missing_source_families)
                    + "; "
                    + "; ".join(sorted(set(diagnostics))[:5])
                ),
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance=provenance,
            )
        frames_doc = by_format["capture-frames"].payload or {}
        stamps_doc = by_format["sim-stamps"].payload or {}
        document, mapping_diagnostics = _build_mapping(
            frames_doc, stamps_doc, request.config, selected_sources
        )
        diagnostics.extend(mapping_diagnostics)
        if not document:
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason="no_mapping_document: " + "; ".join(sorted(set(diagnostics))[:5]),
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance=provenance,
            )
        capability_payload = {
            "missing_capabilities": [],
            "diagnostics": sorted(set(diagnostics)),
            "sources": selected_sources,
            "admission": ADMISSION_NOT_EVALUATED,
        }
        mapping_digest = _write_json(output_dir / OUTPUT_MAPPING_FILENAME, document)
        _write_json(output_dir / OUTPUT_CAPABILITY_FILENAME, capability_payload)
        informational = {"optional_stream_skipped", "nonuniform_sampling"}
        blocking = [item for item in diagnostics if not any(tag in item for tag in informational)]
        partial = bool(blocking) or bool(document.get("unavailable_frames"))
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = (
            (
                {
                    "artifact_id": OUTPUT_MAPPING_FILENAME,
                    "uri": str(Path(request.output_directory) / OUTPUT_MAPPING_FILENAME),
                    "sha256": mapping_digest,
                },
            )
            if status == STATUS_COMPLETE
            else ()
        )
        provenance.update(
            {
                "output_directory": request.output_directory,
                "entries": len(document["entries"]),
                "unavailable_frames": len(document.get("unavailable_frames", [])),
            }
        )
        return _result(
            request.request_id,
            request.component_id,
            status,
            artifacts=artifacts,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            provenance=provenance,
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        return _result(
            request_id,
            component_id,
            STATUS_FAILED,
            reason="; ".join(error.errors),
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _failed_result(request, "internal_error: video-sync execution failed safely")


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the video-sync component."""
    parser = argparse.ArgumentParser(description="Map capture frames to simulation time.")
    parser.add_argument(
        "--input", required=False, default=None, help="Component request JSON file."
    )
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument(
        "--output", required=False, default=None, help="Output directory (must not exist)."
    )
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    parser.add_argument(
        "--descriptor",
        action="store_true",
        help="Print the component-descriptor.v1 document instead of running a request.",
    )
    return parser


def _cli_identity(payload: Any) -> tuple[str, str]:
    """Return safe identities for a failure emitted before request validation."""
    if not isinstance(payload, Mapping):
        return "unknown", COMPONENT_ID
    return (
        _safe_identity(payload.get("request_id"), default="unknown"),
        _safe_identity(payload.get("component_id"), default=COMPONENT_ID),
    )


def _print_cli_failure(reason: str, *, payload: Any = None) -> int:
    """Print a contract-valid failed result for a CLI boundary failure.

    Returns:
        Non-zero CLI exit code.
    """
    request_id, component_id = _cli_identity(payload)
    result = _result(request_id, component_id, STATUS_FAILED, reason=reason)
    print(json.dumps(_result_document(result), sort_keys=True, indent=2))  # noqa: T201
    return 1


def _result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize a result with the shared ``component-result.v1`` envelope.

    Returns:
        JSON-safe shared result envelope.
    """
    return json.loads(
        json.dumps(
            {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)},
            allow_nan=False,
        )
    )


def _prepare_cli_payload(args: argparse.Namespace) -> tuple[Any, str | None]:
    """Read and merge strict-JSON CLI inputs.

    Returns:
        Prepared request payload and an optional stable input-failure reason.
    """
    try:
        payload = _load_strict_json(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None, "invalid_input: request JSON cannot be parsed safely"
    if not isinstance(payload, dict):
        return payload, "invalid_input: request must be a JSON object"
    if args.config is not None:
        try:
            config = _load_strict_json(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return payload, "invalid_input: config JSON cannot be parsed safely"
        if not isinstance(config, dict):
            return payload, "invalid_input: config must be an object"
        request_config = payload.get("config", {})
        if not isinstance(request_config, dict):
            return payload, "invalid_input: request config must be an object"
        payload = {**payload, "config": {**request_config, **config}}
    return {**payload, "output_directory": args.output}, None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the video-sync component.

    Returns:
        Process exit code (0 only when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    if args.descriptor:
        print(json.dumps(descriptor_document(), sort_keys=True, indent=2))  # noqa: T201
        return 0
    if args.input is None or args.output is None:
        return _print_cli_failure("invalid_input: --input and --output are required")
    payload, input_error = _prepare_cli_payload(args)
    if input_error:
        return _print_cli_failure(input_error, payload=payload)
    try:
        request = component_request_from_dict(payload, source=args.input)
    except (ReviewContractsValidationError, TypeError, ValueError, RecursionError):
        return _print_cli_failure(
            "invalid_input: request does not satisfy component-request.v1", payload=payload
        )
    try:
        result = run(request, base=Path(args.base) if args.base is not None else None)
        document = _result_document(result)
        output = json.dumps(document, sort_keys=True, indent=2, allow_nan=False)
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _print_cli_failure(
            "internal_error: video-sync result could not be serialized safely", payload=payload
        )
    print(output)  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
