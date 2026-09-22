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
import os
import shutil
import stat
import tempfile
from bisect import bisect_left, bisect_right
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
    component_result_from_dict,
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

# This component only maps small, already-produced diagnostic documents. Keep
# every read, parse, and derived index finite so a valid-looking request cannot
# turn the mapper into an unbounded file or memory consumer.
MAX_SOURCES = 16
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FRAME_ROWS = 100_000
MAX_STAMP_ROWS = 100_000
MAX_FRAME_INDEX = 1_000_000
MAX_FRAME_INDEX_SPAN = 100_000
MAX_IDENTITY_LENGTH = 512
MAX_ABS_TIME_S = 1_000_000_000.0
MAX_TIMESTAMP_TOLERANCE_S = 86_400.0
MAX_PRESENTATION_WIDTH = 16_384
MAX_PRESENTATION_HEIGHT = 16_384
MAX_PRESENTATION_RATE = 1_000.0
MIN_PRESENTATION_RATE = 1.0e-6
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_SOURCE_METADATA_FIELDS = (
    "schema",
    "sha256",
    "source_commit",
    "config_identity",
    "units",
    "coordinate_frame",
)

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


class _BoundedSourceError(ValueError):
    """Describe a regular-file or byte-budget violation at an input boundary."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


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
    if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise _BoundedSourceError("resource_limit:output_bytes")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _read_bounded_regular_file(path: Path) -> bytes:
    """Read one regular file without blocking on special files.

    The path has already crossed the caller's containment boundary. The
    descriptor and post-open stat checks still close the FIFO/device and
    replacement-file cases before a read can block or exceed the byte budget.

    Returns:
        At most ``MAX_JSON_BYTES`` bytes from ``path``.

    Raises:
        _BoundedSourceError: If the path is not a regular file or exceeds the
            component's finite input budget.
    """
    initial_stat = path.stat()
    if not stat.S_ISREG(initial_stat.st_mode):
        raise _BoundedSourceError("source_not_regular_file")
    if initial_stat.st_size > MAX_JSON_BYTES:
        raise _BoundedSourceError("resource_limit:source_bytes")

    flags = os.O_RDONLY | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(path, flags)
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise _BoundedSourceError("source_not_regular_file")
        if opened_stat.st_size > MAX_JSON_BYTES:
            raise _BoundedSourceError("resource_limit:source_bytes")
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = None
            payload = stream.read(MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            raise _BoundedSourceError("resource_limit:source_bytes")
        return payload
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)


def _reject_nonfinite_json(value: str) -> Any:
    """Reject JSON extensions such as ``NaN`` and ``Infinity``."""
    raise ValueError(f"non-strict JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys so source identity cannot be ambiguous.

    Returns:
        Parsed object with unique keys.
    """
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _load_strict_json(text: str | bytes) -> Any:
    """Parse JSON without accepting non-standard non-finite constants.

    Returns:
        Parsed JSON value.
    """
    try:
        byte_length = len(text) if isinstance(text, bytes) else len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("input is not valid UTF-8") from error
    if byte_length > MAX_JSON_BYTES:
        raise _BoundedSourceError("resource_limit:json_bytes")
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json,
    )


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


def _validate_source_ref(  # noqa: C901 - one fail-closed check per source field
    ref: Any, index: int, seen_artifacts: set[str]
) -> list[str]:
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
    elif _unsafe_artifact_id(ref.artifact_id):
        errors.append(f"invalid_request: source_{index} artifact_id is unsafe")
    if isinstance(ref.artifact_id, str) and ref.artifact_id:
        seen_artifacts.add(ref.artifact_id)
    if not isinstance(ref.uri, str) or not ref.uri:
        errors.append(f"invalid_request: source_{index} uri is invalid")
    elif _unsafe_relative_path(ref.uri):
        errors.append(f"invalid_request: source_{index} uri traversal is rejected")
    if not isinstance(ref.format, str) or not ref.format:
        errors.append(f"invalid_request: source_{index} format is invalid")
    for field_name in _SOURCE_METADATA_FIELDS:
        value = getattr(ref, field_name, "")
        if value not in (None, "") and not isinstance(value, str):
            errors.append(f"invalid_request: source_{index} {field_name} must be a string")
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
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _unsafe_artifact_id(value: str) -> bool:
    """Return whether an artifact identity could become a path component."""
    windows = PureWindowsPath(value)
    return (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or windows.is_absolute()
        or bool(windows.drive)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _validate_source_metadata_bindings(
    config: dict[str, Any], sources: tuple[SourceRef, ...] | list[SourceRef]
) -> list[str]:
    """Reject metadata entries that are not bound to one declared source.

    Returns:
        Stable validation reason codes.
    """
    if "source_metadata" not in config:
        return []
    raw_metadata = config["source_metadata"]
    if raw_metadata is None or not isinstance(raw_metadata, dict):
        return ["invalid_request: source_metadata must be an object"]
    source_ids = {
        ref.artifact_id
        for ref in sources
        if isinstance(ref, SourceRef) and isinstance(ref.artifact_id, str)
    }
    errors: list[str] = []
    for artifact_id, metadata in raw_metadata.items():
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append("invalid_request: source_metadata keys must be non-empty strings")
            continue
        if artifact_id not in source_ids:
            errors.append(f"invalid_request: source_metadata.{artifact_id} has no declared source")
        if not isinstance(metadata, dict):
            errors.append(f"invalid_request: source_metadata.{artifact_id} must be an object")
    return errors


def _validate_request_config(config: Any) -> list[str]:
    """Validate the request config object and its strict-JSON safety.

    Returns:
        Stable validation reason codes for the config.
    """
    if not isinstance(config, dict):
        return ["invalid_request: config must be an object"]
    try:
        encoded = json.dumps(config, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        return ["invalid_request: config must be strict-JSON safe"]
    if len(encoded) > MAX_JSON_BYTES:
        return [f"invalid_request: config exceeds resource limit of {MAX_JSON_BYTES} bytes"]
    return []


def _validate_request_shape(request: Any) -> list[str]:  # noqa: C901 - bounded request gate
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
        if len(request.sources) > MAX_SOURCES:
            errors.append(f"invalid_request: sources exceed resource limit of {MAX_SOURCES}")
    if isinstance(request.config, dict) and isinstance(request.sources, (tuple, list)):
        errors.extend(_validate_source_metadata_bindings(request.config, request.sources))
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
    declared_versions = [
        (field_name, config[field_name])
        for field_name in ("min_component_version", "required_component_version")
        if field_name in config
    ]
    if len(declared_versions) > 1 and not _metadata_values_equal(
        "component_version", declared_versions[0][1], declared_versions[1][1]
    ):
        return "incompatible_component_version: conflicting version requirements"
    if not declared_versions:
        return None
    field_name, minimum = declared_versions[0]
    if not isinstance(minimum, str) or not minimum:
        return f"incompatible_component_version: malformed {field_name}"
    wanted = _parse_version(minimum)
    ours = _parse_version(COMPONENT_VERSION)
    if wanted is None or ours is None:
        return f"incompatible_component_version: malformed {field_name}: {minimum!r}"
    if wanted > ours:
        return (
            f"incompatible_component_version: request needs v{minimum}, "
            f"component is v{COMPONENT_VERSION}"
        )
    return None


def _parse_version(value: str) -> tuple[int, int, int] | None:
    """Parse the component's bounded three-part semantic version form.

    Returns:
        Parsed major, minor, and patch numbers, or ``None``.
    """
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        return None
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError:
        return None
    return major, minor, patch


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


def _resolve_base(base: Path | str | None) -> Path:
    """Resolve the invocation base without assuming a concrete path type.

    Returns:
        An absolute, non-strictly resolved base path.
    """
    try:
        candidate = Path.cwd() if base is None else Path(base)
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReviewContractsValidationError(
            ["invalid_base: cannot resolve base directory"]
        ) from error


def _path_exists_any(path: Path) -> bool:
    """Return whether a path entry exists, including a dangling symlink."""
    try:
        return os.path.lexists(path)
    except OSError:
        return True


def _commit_outputs(  # noqa: C901 - staged publication has explicit cleanup branches
    output_dir: Path, mapping_document: dict[str, Any], capability_document: dict[str, Any]
) -> str:
    """Stage both output documents and publish them without overwriting a target.

    Returns:
        SHA-256 digest of the published mapping document.
    """
    stage_dir: Path | None = None
    reserved_output = False
    published = False
    linked_outputs: list[tuple[Path, Path]] = []
    filenames = (OUTPUT_MAPPING_FILENAME, OUTPUT_CAPABILITY_FILENAME)
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = Path(tempfile.mkdtemp(prefix=".srev03-video-sync-", dir=str(output_dir.parent)))
        mapping_digest = _write_json(stage_dir / OUTPUT_MAPPING_FILENAME, mapping_document)
        _write_json(stage_dir / OUTPUT_CAPABILITY_FILENAME, capability_document)
        try:
            output_dir.mkdir()
        except FileExistsError as error:
            raise ReviewContractsValidationError(
                [f"output_collision: already exists: {output_dir}"]
            ) from error
        reserved_output = True
        for filename in filenames:
            staged = stage_dir / filename
            destination = output_dir / filename
            os.link(staged, destination)
            linked_outputs.append((destination, staged))
        published = True
        return mapping_digest
    except _BoundedSourceError as error:
        raise ReviewContractsValidationError([error.code]) from error
    except ReviewContractsValidationError:
        raise
    except FileExistsError as error:
        raise ReviewContractsValidationError(
            [f"output_collision: already exists: {output_dir}"]
        ) from error
    except (OSError, TypeError, ValueError) as error:
        raise ReviewContractsValidationError(
            [f"output_write_failed: {type(error).__name__}: {error}"]
        ) from error
    finally:
        if reserved_output and not published:
            for destination, staged in linked_outputs:
                try:
                    if (
                        staged.exists()
                        and destination.exists()
                        and os.path.samefile(destination, staged)
                    ):
                        destination.unlink()
                except OSError:
                    pass
            try:
                output_dir.rmdir()
            except OSError:
                pass
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _valid_sha256(value: Any) -> bool:
    """Return whether a value is a 64-character hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _source_metadata(  # noqa: C901 - bounded provenance field validation
    config: dict[str, Any], ref: SourceRef
) -> dict[str, Any]:
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
        for field_name in _SOURCE_METADATA_FIELDS
        if getattr(ref, field_name, "") not in (None, "")
    }
    errors: list[str] = []
    for field_name, value in declared.items():
        if field_name not in _SOURCE_METADATA_FIELDS:
            errors.append(
                f"invalid_config: source_metadata.{ref.artifact_id}.{field_name} "
                "is not a supported provenance field"
            )
            continue
        if field_name in metadata and not _metadata_values_equal(
            field_name, metadata[field_name], value
        ):
            errors.append(
                f"invalid_config: source_metadata.{ref.artifact_id}.{field_name} "
                "conflicts with the source reference"
            )
        else:
            metadata[field_name] = value
    for field_name in _SOURCE_METADATA_FIELDS:
        if (
            field_name in metadata
            and field_name != "sha256"
            and not isinstance(metadata[field_name], str)
        ):
            errors.append(
                f"invalid_config: source_metadata.{ref.artifact_id}.{field_name} must be a string"
            )
    if "sha256" in metadata:
        if not isinstance(metadata["sha256"], str):
            errors.append(
                f"invalid_config: source_metadata.{ref.artifact_id}.sha256 must be a string"
            )
        elif not _valid_sha256(metadata["sha256"]):
            errors.append(
                f"invalid_config: source_metadata.{ref.artifact_id}.sha256 must be 64-hex"
            )
    if "source_commit" in metadata and (
        not isinstance(metadata["source_commit"], str)
        or not _valid_sha40(metadata["source_commit"])
    ):
        errors.append(
            f"invalid_config: source_metadata.{ref.artifact_id}.source_commit must be 40-hex"
        )
    if errors:
        raise ReviewContractsValidationError(errors)
    return dict(metadata)


def _metadata_values_equal(field_name: str, left: Any, right: Any) -> bool:
    """Compare overlapping source declarations without losing digest casing.

    Returns:
        Whether both declarations represent the same value.
    """
    if field_name == "sha256" and isinstance(left, str) and isinstance(right, str):
        return left.lower() == right.lower()
    return left == right


def _valid_sha40(value: Any) -> bool:
    """Return whether a value is a 40-character hexadecimal commit digest."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


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
        "units": metadata.get("units"),
        "coordinate_frame": metadata.get("coordinate_frame"),
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


def _load_document(  # noqa: C901 - source boundary maps each failure explicitly
    ref: SourceRef, metadata: dict[str, Any], root: Path
) -> _ImportOutcome:
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
        raw = _read_bounded_regular_file(_resolve_source(ref.uri, root))
    except _BoundedSourceError as error:
        outcome.diagnostics.append(f"{ref.artifact_id}: {error.code}")
        outcome.availability = error.code
        return outcome
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
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_identity(value: Any) -> str | None:
    """Return a finite-size non-empty identity string, or ``None``."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_IDENTITY_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return None
    return value


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
        or frame_index > MAX_FRAME_INDEX
        or pts is None
        or abs(pts) > MAX_ABS_TIME_S
    ):
        return None, f"frame_row_{index}_malformed"
    frame = {"frame_index": frame_index, "pts_s": pts}
    for identity in ("episode_id", "reset_id"):
        if identity in item:
            value = _bounded_identity(item[identity])
            if value is None:
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
    if len(raw_frames) > MAX_FRAME_ROWS:
        return [], [f"resource_limit:capture_frame_rows:{MAX_FRAME_ROWS}"]
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
    if len(raw_steps) > MAX_STAMP_ROWS:
        return [], [f"resource_limit:sim_stamp_rows:{MAX_STAMP_ROWS}"]
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            diagnostics.append(f"step_row_{index}_malformed")
            continue
        step = item.get("step")
        when = _finite_number(item.get("time_s"))
        episode_id = _bounded_identity(item.get("episode_id"))
        reset_id = _bounded_identity(item.get("reset_id"))
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step > MAX_FRAME_INDEX
            or when is None
            or abs(when) > MAX_ABS_TIME_S
            or episode_id is None
            or reset_id is None
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
    frame_span = seen[-1] - seen[0]
    if frame_span > MAX_FRAME_INDEX_SPAN:
        return [], [f"resource_limit:frame_index_span:{MAX_FRAME_INDEX_SPAN}"]
    seen_set = set(seen)
    skipped = [i for i in range(seen[0], seen[-1] + 1) if i not in seen_set]
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
    *,
    step_times: list[float] | None = None,
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Select one simulation stamp or return an explicit unavailable reason.

    Returns:
        The selected anchor, an unavailable reason, and the nearest temporal
        error in seconds.
    """
    times = step_times if step_times is not None else [step["time_s"] for step in steps]
    insertion = bisect_left(times, sim_time)
    neighbors = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
    nearest_error = min((abs(times[index] - sim_time) for index in neighbors), default=math.inf)
    allowance = (
        0.0
        if timestamp_tolerance_s == 0.0
        else 2.0 * math.ulp(max(1.0, abs(sim_time), abs(timestamp_tolerance_s)))
    )
    left = bisect_left(times, sim_time - timestamp_tolerance_s - allowance)
    right = bisect_right(times, sim_time + timestamp_tolerance_s + allowance)
    candidate_count = right - left
    if candidate_count == 0:
        return None, "no_sim_stamp_within_tolerance", nearest_error
    if candidate_count > 1:
        return None, "ambiguous_sim_stamp_match", nearest_error
    candidate_index = left
    if not _within_timestamp_tolerance(
        abs(times[candidate_index] - sim_time), timestamp_tolerance_s
    ):
        return None, "no_sim_stamp_within_tolerance", nearest_error
    anchor = steps[candidate_index]
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


def _map_frames(  # noqa: C901 - each mapping boundary has an explicit reason
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
    sim_times: list[float] = []
    for frame in frames:
        try:
            sim_time = time_origin + frame["pts_s"] / speed
        except (OverflowError, ZeroDivisionError):
            return [], [], ["resource_limit:derived_sim_time"]
        if not math.isfinite(sim_time) or abs(sim_time) > MAX_ABS_TIME_S:
            return [], [], ["resource_limit:derived_sim_time"]
        sim_times.append(sim_time)
    first_time = sim_times[0]
    last_time = sim_times[-1]
    outside_range = first_time < steps[0]["time_s"] or last_time > steps[-1]["time_s"]
    if outside_range:
        diagnostics.append("frames_outside_stamp_range")
    timeline_diagnostics, timeline_invalid = _timeline_diagnostics(steps)
    diagnostics.extend(timeline_diagnostics)
    mapping: list[dict[str, Any]] = []
    step_times = [step["time_s"] for step in steps]
    for frame, sim_time in zip(frames, sim_times, strict=True):
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
            frame,
            steps,
            sim_time,
            timestamp_tolerance_s,
            step_times=step_times,
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
    maximum = {
        "width": MAX_PRESENTATION_WIDTH,
        "height": MAX_PRESENTATION_HEIGHT,
    }[name]
    if value > maximum:
        return None, f"resource_limit:presentation_{name}:{maximum}"
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
    if value < MIN_PRESENTATION_RATE:
        return None, f"presentation_{name}_invalid"
    if value > MAX_PRESENTATION_RATE:
        return None, f"resource_limit:presentation_{name}:{MAX_PRESENTATION_RATE}"
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


def _build_mapping(  # noqa: C901 - fail-closed mapping stages are intentionally linear
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
    if fps_nominal < MIN_PRESENTATION_RATE or fps_nominal > MAX_PRESENTATION_RATE:
        return {}, [f"resource_limit:fps_nominal:{MAX_PRESENTATION_RATE}"]
    presentation, presentation_errors = _validate_presentation(config)
    if presentation_errors:
        return {}, presentation_errors
    speed = presentation["speed"]
    time_origin = _finite_number(config.get("time_origin_s", 0.0))
    if time_origin is None:
        return {}, ["time_origin_invalid"]
    if abs(time_origin) > MAX_ABS_TIME_S:
        return {}, [f"resource_limit:time_origin_s:{MAX_ABS_TIME_S}"]
    timestamp_tolerance = _finite_number(
        config.get("timestamp_tolerance_s", DEFAULT_TIMESTAMP_TOLERANCE_S)
    )
    if timestamp_tolerance is None or timestamp_tolerance < 0:
        return {}, ["timestamp_tolerance_invalid"]
    if timestamp_tolerance > MAX_TIMESTAMP_TOLERANCE_S:
        return {}, [f"resource_limit:timestamp_tolerance_s:{MAX_TIMESTAMP_TOLERANCE_S}"]
    frames, diagnostics = _parse_frame_rows(raw_frames)
    steps, step_diagnostics = _parse_step_rows(raw_steps)
    diagnostics.extend(step_diagnostics)
    if not frames or not steps:
        return {}, diagnostics or ["no_mappable_rows"]
    skipped, gap_diagnostics = _detect_sampling_gaps(frames, fps_nominal)
    diagnostics.extend(gap_diagnostics)
    if any(item.startswith("resource_limit:") for item in diagnostics):
        return {}, sorted(set(diagnostics))
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
    if any(item.startswith("resource_limit:") for item in diagnostics):
        return {}, sorted(set(diagnostics))
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


def run(request: ComponentRequest, *, base: Path | str | None = None) -> ComponentResult:
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
        root = _resolve_base(base)
        rejected = _reject_not_applicable(request)
        if rejected is not None:
            return rejected
        output_dir = _resolve_output_directory(root, request.output_directory)
        if _path_exists_any(root / request.output_directory) or _path_exists_any(output_dir):
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
        try:
            mapping_digest = _commit_outputs(output_dir, document, capability_payload)
        except ReviewContractsValidationError as error:
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason="; ".join(error.errors),
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance=provenance,
            )
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
    document = json.loads(
        json.dumps(
            {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)},
            allow_nan=False,
        )
    )
    component_result_from_dict(document)
    return document


def _read_cli_json(path_value: str) -> Any:
    """Read one bounded regular UTF-8 JSON file for the CLI boundary.

    Returns:
        Parsed strict-JSON payload.
    """
    raw = _read_bounded_regular_file(Path(path_value))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("input is not valid UTF-8") from error
    return _load_strict_json(text)


def _prepare_cli_payload(args: argparse.Namespace) -> tuple[Any, str | None]:
    """Read and merge strict-JSON CLI inputs.

    Returns:
        Prepared request payload and an optional stable input-failure reason.
    """
    try:
        payload = _read_cli_json(args.input)
    except (OSError, ValueError, RecursionError):
        return None, "invalid_input: request JSON cannot be parsed safely"
    if not isinstance(payload, dict):
        return payload, "invalid_input: request must be a JSON object"
    if args.config is not None:
        try:
            config = _read_cli_json(args.config)
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
