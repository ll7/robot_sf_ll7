"""SREV-10 review-encode component for bounded, source-backed video edits.

The leaf consumes a versioned frame-sequence manifest or an actual source clip,
applies a finite edit plan, and emits a playable MP4 plus a receipt and a
piecewise source-time/presentation-time map.  Source bytes are integrity-bound
before decoding.  The component is deliberately diagnostic-only: its output
does not establish benchmark, safety, causal, planner, simulator, training, or
scientific evidence.

Audio output is silent in v1.  The shared SREV-01 envelope schemas remain the
owner of request, descriptor, and result contracts; this module owns only the
SREV-10 leaf payload validation and publication boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)

COMPONENT_ID = "srev10-review-encode"
COMPONENT_VERSION = "1.0.0"

FRAME_MANIFEST_FORMAT = "frame-sequence-manifest.v1"
SOURCE_CLIP_FORMAT = "source-clip"

REQUIRED_CAPABILITIES: tuple[str, ...] = ()
OPTIONAL_CAPABILITIES = ("frame-sequence", "source-clip")

OUTPUT_VIDEO_FILENAME = "edit.mp4"
OUTPUT_RECEIPT_FILENAME = "encode-receipt.json"
OUTPUT_TIMEMAP_FILENAME = "time-map.json"

RECEIPT_SCHEMA_VERSION = "encode-receipt.v1"
TIMEMAP_SCHEMA_VERSION = "presentation-time-map.v1"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

SOURCE_FPS = "source_fps"
SOURCE_FRAMES = "source_frames"

ENCODER_NAME = "imageio-ffmpeg"
ENCODER_CODEC = "libx264"
AUDIO_POLICY_SILENT = "silent"

DEFAULT_PRESET = {"width": 1920, "height": 1080, "fps": 30.0}
TINY_PRESET = {"width": 160, "height": 120, "fps": 10.0}

# These are hard admission limits, not tuning hints.  They keep malformed or
# adversarial requests from allocating unbounded frame lists or pixel buffers.
MAX_SOURCE_FRAMES = 3000
MAX_SOURCE_DURATION_S = 300.0
MAX_SOURCE_FPS = 240.0
MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024
MAX_SOURCE_BUFFER_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_FRAMES = 6000
MAX_OUTPUT_WIDTH = 1920
MAX_OUTPUT_HEIGHT = 1080
MAX_OUTPUT_FPS = 120.0
MAX_OUTPUT_PIXELS = MAX_OUTPUT_WIDTH * MAX_OUTPUT_HEIGHT
MAX_OUTPUT_BUFFER_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_FILE_BYTES = 512 * 1024 * 1024
MAX_ENCODER_SECONDS = 30.0
MAX_DECODER_SECONDS = 30.0
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_CONFIG_BYTES = 1 * 1024 * 1024
MAX_EDIT_COUNT = 128
MIN_SPEED_FACTOR = 0.125
MAX_SPEED_FACTOR = 8.0
MAX_PAUSE_DURATION_S = 300.0
MAX_CROP_COORDINATE = 1_000_000.0
MAX_CROP_SIZE = 1_000_000.0

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_CONFIG_KEYS = frozenset({"audio_policy", "edits", "min_component_version", "preset"})
_PRESET_KEYS = frozenset({"width", "height", "fps"})
_EDIT_KEYS = {
    "cut": frozenset({"op", "start_s", "end_s"}),
    "pause": frozenset({"op", "at_s", "duration_s"}),
    "speed": frozenset({"op", "start_s", "end_s", "factor"}),
    "crop": frozenset({"op", "x", "y", "width", "height"}),
}
_CROP_PROVENANCE_KEYS = frozenset({"operation", "box"})
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "component",
        "component_version",
        "encoder",
        "source",
        "config_sha256",
        "edit_plan_digest",
        "output",
        "presentation_duration_s",
        "crop",
        "audio_policy",
        "evidence_status",
        "evidence_boundary",
        "benchmark_success",
        "scientific_claim_allowed",
    }
)
_RECEIPT_ENCODER_KEYS = frozenset(
    {
        "encoder",
        "codec",
        "ffmpeg_binary",
        "imageio_version",
        "pillow_version",
        "numpy_version",
        "platform",
        "audio_policy",
        "encoder_timeout_s",
        "decoder_timeout_s",
        "output_fps",
        "output_frames",
        "output_bytes",
    }
)
_RECEIPT_SOURCE_KEYS = frozenset(
    {
        "artifact_id",
        "uri",
        "format",
        "schema",
        "source_commit",
        "config_identity",
        "units",
        "coordinate_frame",
        "sha256",
        "declared_sha256",
        "source_frames",
        "source_fps",
        "source_duration_s",
        "frame_sha256",
        "integrity",
    }
)
_RECEIPT_SOURCE_REQUIRED_KEYS = _RECEIPT_SOURCE_KEYS - frozenset(
    {"schema", "source_commit", "config_identity", "units", "coordinate_frame"}
)
_RECEIPT_OUTPUT_KEYS = frozenset(
    {
        "video",
        "video_sha256",
        "time_map",
        "time_map_sha256",
        "frames",
        "width",
        "height",
        "fps",
    }
)
_TIMEMAP_KEYS = frozenset(
    {
        "schema_version",
        "component",
        "source_fps",
        "presentation_fps",
        "source_frames",
        "source_duration_s",
        "presentation_duration_s",
        "segments",
        "frame_order_source_indices",
        "first_source_frame",
        "terminal_source_frame",
        "crop",
        "evidence_status",
        "evidence_boundary",
        "benchmark_success",
        "scientific_claim_allowed",
    }
)
_TIMEMAP_SEGMENT_KEYS = frozenset(
    {
        "segment",
        "operation",
        "factor",
        "presentation_start_s",
        "presentation_end_s",
        "source_start_s",
        "source_end_s",
        "frame_count",
        "source_frame_start",
        "source_frame_end",
    }
)

_DESCRIPTOR_DOC: dict[str, Any] = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": [
        "review-edit-mp4.v1",
        "encode-receipt.v1",
        "presentation-time-map.v1",
    ],
}

# Validate the descriptor at import time so an accidental envelope drift cannot
# reach the CLI or a registry consumer.
_DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOC)


class _OperationTimeout(RuntimeError):
    """Raised when a bounded decoder or encoder exceeds its wall-time limit."""


class _SourceLoadError(RuntimeError):
    """Carry a stable source admission failure and optional structured details."""

    def __init__(
        self,
        reason: str,
        *,
        status: str = STATUS_FAILED,
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.diagnostics = diagnostics


class _OutputCollisionError(RuntimeError):
    """Raised when the requested output directory is already reserved."""


@dataclass(frozen=True)
class _Source:
    """Decoded source frames and the verified identity of their source artifact."""

    frames: tuple[Any, ...]
    source_fps: float
    source_ref: SourceRef
    source_kind: str
    source_document: dict[str, Any]
    observed_sha256: str
    frame_digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Prepared:
    """Validated source, output preset, and encoder environment."""

    root: Path
    out_dir: Path
    config: dict[str, Any]
    env: dict[str, Any]
    source: _Source
    width: int
    height: int
    out_fps: float


@dataclass(frozen=True)
class _TimelineSegment:
    """One source interval after cut/speed processing."""

    segment_id: int
    operation: str
    factor: float
    source_start_s: float
    source_end_s: float
    source_indices: tuple[int, ...]
    presentation_indices: tuple[int, ...]


@dataclass(frozen=True)
class _PresentationFrame:
    """One output frame plus the map segment that produced it."""

    source_index: int
    operation: str
    segment_id: int
    factor: float
    source_start_s: float
    source_end_s: float


@dataclass(frozen=True)
class _Planned:
    """Complete presentation order and its validated timing metadata."""

    frames: tuple[_PresentationFrame, ...]
    segments: tuple[_TimelineSegment, ...]
    diagnostics: tuple[str, ...]
    crop_box: tuple[int, int, int, int] | None
    source_duration_s: float


def descriptor() -> dict[str, Any]:
    """Return the schema-valid component-descriptor.v1 document."""

    return json.loads(json.dumps(_DESCRIPTOR_DOC, sort_keys=True))


def result_payload(result: ComponentResult) -> dict[str, Any]:
    """Serialize and validate a component-result.v1 envelope.

    The shared dataclass intentionally stores the common fields only; the
    schema-version discriminator is added at the JSON boundary.

    Returns:
        A JSON-safe, schema-validated result envelope.
    """

    payload = json.loads(
        json.dumps(
            {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)},
            allow_nan=False,
        )
    )
    component_result_from_dict(payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(payload: Any) -> str:
    """Return a location-independent digest for strict JSON content."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    """Return the stable identity fields used for source read race checks."""

    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _sha256_file(path: Path, *, max_bytes: int) -> tuple[str, int]:
    """Hash one regular file with a byte ceiling and post-read stability check.

    Returns:
        Tuple of the lowercase digest and bytes read.
    """

    try:
        before = _file_signature(path)
    except OSError as error:
        raise OSError(f"source file is not readable: {path.name}") from error
    if before[2] > max_bytes:
        raise _SourceLoadError(
            f"resource_limit: source_bytes>{max_bytes}",
            diagnostics=(
                {
                    "code": "source_bytes_limit",
                    "path": path.name,
                    "limit_bytes": max_bytes,
                },
            ),
        )
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise _SourceLoadError(
                        f"resource_limit: source_bytes>{max_bytes}",
                        diagnostics=(
                            {
                                "code": "source_bytes_limit",
                                "path": path.name,
                                "limit_bytes": max_bytes,
                            },
                        ),
                    )
                digest.update(chunk)
        after = _file_signature(path)
    except _SourceLoadError:
        raise
    except OSError as error:
        raise OSError(f"source file read failed: {path.name}") from error
    if before != after:
        raise _SourceLoadError(
            f"source_changed_during_read: {path.name}",
            diagnostics=({"code": "source_changed_during_read", "path": path.name},),
        )
    return digest.hexdigest(), total


def _write_json(path: Path, payload: Any) -> str:
    """Write strict JSON atomically within a private staging directory.

    Returns:
        Lowercase SHA-256 digest of the written bytes.
    """

    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _finite_number(value: Any) -> bool:
    """Return whether a value is a non-boolean finite JSON number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _safe_relative(value: Any) -> bool:
    """Reject absolute, traversal, Windows-drive, separator, and NUL paths.

    Returns:
        Whether the value is a safe relative path.
    """

    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    path = Path(value)
    pure = PureWindowsPath(value)
    return not (path.is_absolute() or pure.is_absolute() or bool(pure.drive) or ".." in path.parts)


def _safe_artifact_id(value: Any) -> bool:
    """Return whether an artifact identifier is safe as a scoped name."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    pure = PureWindowsPath(value)
    return not (
        "/" in value
        or "\\" in value
        or pure.is_absolute()
        or bool(pure.drive)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _source_reference_identity(ref: SourceRef) -> dict[str, str]:
    """Return the validated source-reference identity for provenance outputs."""

    identity = {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "format": ref.format,
    }
    if ref.sha256:
        identity["sha256"] = ref.sha256.lower()
    for field in ("schema", "source_commit", "config_identity", "units", "coordinate_frame"):
        value = getattr(ref, field)
        if value:
            identity[field] = value
    return identity


def _request_field_errors(request: ComponentRequest) -> list[str]:
    """Validate scalar request fields.

    Returns:
        Stable scalar-field admission errors.
    """

    errors: list[str] = []
    if not isinstance(request.request_id, str) or not request.request_id:
        errors.append("request_invalid: request_id must be a non-empty string")
    if not isinstance(request.component_id, str) or not request.component_id:
        errors.append("request_invalid: component_id must be a non-empty string")
    if not _safe_relative(request.output_directory):
        errors.append("unsafe_output_path: output directory must be relative and contained")
    if not isinstance(request.sources, tuple) or not request.sources:
        errors.append("request_invalid: sources must be a non-empty tuple")
    if not isinstance(request.config, dict):
        errors.append("request_invalid: config must be an object")
    if not isinstance(request.required_capabilities, tuple):
        errors.append("request_invalid: required_capabilities must be a tuple")
    return errors


def _source_ref_errors(index: int, ref: Any, seen: set[str]) -> list[str]:
    """Validate one source reference and update scoped identity state.

    Returns:
        Stable source-reference admission errors.
    """

    if not isinstance(ref, SourceRef):
        return [f"request_invalid: sources[{index}] must be a SourceRef"]
    errors: list[str] = []
    if not _safe_artifact_id(ref.artifact_id):
        errors.append(f"request_invalid: sources[{index}].artifact_id is unsafe")
    elif ref.artifact_id in seen:
        errors.append(f"request_invalid: duplicate source artifact_id: {ref.artifact_id}")
    else:
        seen.add(ref.artifact_id)
    if not isinstance(ref.uri, str) or not _safe_relative(ref.uri):
        errors.append(f"request_invalid: sources[{index}].uri is unsafe")
    if not isinstance(ref.format, str) or not ref.format:
        errors.append(f"request_invalid: sources[{index}].format is required")
    errors.extend(_source_ref_identity_errors(index, ref))
    return errors


def _source_ref_identity_errors(index: int, ref: SourceRef) -> list[str]:
    """Validate optional direct source identity fields without coercion.

    Returns:
        Stable validation errors, or an empty list.
    """

    errors: list[str] = []
    if not isinstance(ref.schema, str):
        errors.append(f"request_invalid: sources[{index}].schema must be a string")
    if not isinstance(ref.sha256, str):
        errors.append(f"request_invalid: sources[{index}].sha256 must be a string")
    elif ref.sha256 and _SHA256_RE.fullmatch(ref.sha256) is None:
        errors.append(f"request_invalid: sources[{index}].sha256 is not SHA-256")
    if not isinstance(ref.source_commit, str):
        errors.append(f"request_invalid: sources[{index}].source_commit must be a string")
    elif ref.source_commit and _SHA40_RE.fullmatch(ref.source_commit) is None:
        errors.append(f"request_invalid: sources[{index}].source_commit is not a commit SHA")
    for field in ("config_identity", "units", "coordinate_frame"):
        if not isinstance(getattr(ref, field), str):
            errors.append(f"request_invalid: sources[{index}].{field} must be a string")
    return errors


def _request_source_errors(request: ComponentRequest) -> list[str]:
    """Validate source references and requested capability names.

    Returns:
        Stable source and capability admission errors.
    """

    if not isinstance(request.sources, tuple) or not isinstance(
        request.required_capabilities, tuple
    ):
        return ["request_invalid: sources and required_capabilities must be tuples"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, ref in enumerate(request.sources):
        errors.extend(_source_ref_errors(index, ref, seen))
    for index, capability in enumerate(request.required_capabilities):
        if not isinstance(capability, str) or not capability:
            errors.append(f"request_invalid: required_capabilities[{index}] is invalid")
    return errors


def _request_integrity_errors(request: Any) -> list[str]:
    """Validate direct API callers that bypass component_request_from_dict.

    Returns:
        Stable admission errors, or an empty list when the request is safe.
    """

    if not isinstance(request, ComponentRequest):
        return ["request_invalid: expected ComponentRequest"]
    errors = _request_field_errors(request)
    if not isinstance(request.sources, tuple) or not isinstance(
        request.required_capabilities, tuple
    ):
        return errors + ["request_invalid: sources and required_capabilities must be tuples"]
    return errors + _request_source_errors(request)


def _build_result(
    request: ComponentRequest,
    status: str,
    reason: str,
    *,
    artifacts: tuple[dict[str, Any], ...] = (),
    diagnostics: tuple[dict[str, Any], ...] = (),
    provenance: dict[str, Any] | None = None,
) -> ComponentResult:
    """Build a stable result carrying the diagnostic-only evidence boundary.

    Returns:
        A component result whose JSON form satisfies component-result.v1.
    """

    request_id = (
        request.request_id
        if isinstance(request.request_id, str) and request.request_id
        else "invalid-request"
    )
    component_id = (
        request.component_id
        if isinstance(request.component_id, str) and request.component_id
        else COMPONENT_ID
    )
    output_directory = (
        request.output_directory
        if isinstance(request.output_directory, str)
        else "<invalid-output-directory>"
    )
    result_provenance: dict[str, Any] = {
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "output_directory": output_directory,
        "evidence_status": "diagnostic-only",
        "evidence_boundary": "diagnostic-only; not benchmark evidence",
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "admission": "not_evaluated",
    }
    if provenance:
        result_provenance.update(provenance)
    result = ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=status,
        artifacts=artifacts,
        diagnostics=diagnostics,
        provenance=result_provenance,
        reason=reason,
    )
    # Validate the envelope even for early failures.  This catches accidental
    # additions of non-schema fields or malformed artifact records immediately.
    result_payload(result)
    return result


def _failure(
    request: ComponentRequest,
    reason: str,
    *,
    diagnostics: tuple[dict[str, Any], ...] = (),
    provenance: dict[str, Any] | None = None,
) -> ComponentResult:
    """Return a failed result with no complete artifact references."""

    return _build_result(
        request,
        STATUS_FAILED,
        reason,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _unavailable(request: ComponentRequest, reason: str) -> ComponentResult:
    """Return an unavailable result with no artifact references."""

    return _build_result(request, STATUS_UNAVAILABLE, reason)


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a stable error when a caller requires a newer component."""

    if "min_component_version" not in config:
        return None
    minimum = config["min_component_version"]
    if (
        not isinstance(minimum, str)
        or re.fullmatch(r"[vV]?[0-9]+(?:\.[0-9]+){0,2}", minimum) is None
    ):
        return "config_invalid: min_component_version must be a dotted version string"
    def _version_key(value: str) -> tuple[tuple[int, str], ...]:
        """Normalize decimal components without converting untrusted strings to ints.

        Returns:
            Length-first pairs that compare arbitrary-size decimal components.
        """

        parts = tuple(part.lstrip("0") or "0" for part in value.lstrip("vV").split("."))
        parts += ("0",) * (3 - len(parts))
        # Length-first pairs compare arbitrary-size decimal strings numerically.
        return tuple((len(part), part) for part in parts)

    wanted = _version_key(minimum)
    ours = _version_key(COMPONENT_VERSION)
    if wanted > ours:
        if len(minimum) > 64:
            return "incompatible_component_version: requested minimum is newer than component"
        return (
            "incompatible_component_version: "
            f"request needs v{minimum.lstrip('vV')}, "
            f"component is v{COMPONENT_VERSION.lstrip('vV')}"
        )
    return None


def _validate_preset_config(preset: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and normalize the closed output-preset object.

    Returns:
        Normalized preset and no error, or (None, reason).
    """

    if not isinstance(preset, dict):
        return None, "config_invalid: preset must be an object"
    unknown = sorted((key for key in preset if key not in _PRESET_KEYS), key=str)
    if unknown:
        return None, "config_invalid: preset unknown keys: " + ", ".join(map(str, unknown))
    for key in _PRESET_KEYS:
        if key not in preset or not _finite_number(preset[key]):
            return None, f"config_invalid: preset.{key} must be finite"
    for key in ("width", "height"):
        if not float(preset[key]).is_integer():
            return None, f"config_invalid: preset.{key} must be an integer"
    return {
        "width": int(preset["width"]),
        "height": int(preset["height"]),
        "fps": float(preset["fps"]),
    }, None


def _validate_edit_config(raw: Any) -> str | None:
    """Validate the edit-list container before operation-level validation.

    Returns:
        Error reason, or None when the list is bounded.
    """

    if not isinstance(raw, list):
        return "config_invalid: edits must be a list"
    if len(raw) > MAX_EDIT_COUNT:
        return f"resource_limit: edit_count>{MAX_EDIT_COUNT}"
    return None


def _validate_config(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the closed SREV-10 configuration surface.

    Returns:
        Normalized config and no error, or (None, reason).
    """

    unknown = sorted((key for key in config if key not in _CONFIG_KEYS), key=str)
    if unknown:
        return None, f"config_invalid: unknown keys: {', '.join(map(str, unknown))}"
    audio = config.get("audio_policy", AUDIO_POLICY_SILENT)
    if not isinstance(audio, str):
        return None, "config_invalid: audio_policy must be a string"
    preset, preset_error = _validate_preset_config(config.get("preset", DEFAULT_PRESET))
    if preset is None or preset_error is not None:
        return None, preset_error or "config_invalid: preset"
    edit_error = _validate_edit_config(config.get("edits", []))
    if edit_error is not None:
        return None, edit_error
    normalized = dict(config)
    normalized["audio_policy"] = audio
    normalized["preset"] = preset
    return normalized, None


def _resolve_root(base: Path | str) -> tuple[Path | None, str | None]:
    """Resolve an existing directory used as the source/output root.

    Returns:
        Canonical root and no error, or (None, reason).
    """

    try:
        root = Path(base).resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        return None, "unsafe_output_path: base directory is not readable"
    if not root.is_dir():
        return None, "unsafe_output_path: base directory is not a directory"
    return root, None


def _resolve_output_directory(root: Path, output_directory: str) -> tuple[Path | None, str | None]:
    """Resolve output under root and reject existing or escaping symlink parts.

    Returns:
        Canonical output directory and no error, or (None, reason).
    """

    if not _safe_relative(output_directory):
        return None, "unsafe_output_path: output directory must be relative and contained"
    relative = Path(output_directory)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError:
            return None, "unsafe_output_path: output path cannot be inspected"
        if stat.S_ISLNK(info.st_mode):
            return None, "unsafe_output_path: output path contains a symlink"
        if not stat.S_ISDIR(info.st_mode):
            return None, "unsafe_output_path: output path contains a non-directory"
    try:
        resolved = (root / relative).resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe_output_path: output directory escapes base"
    return resolved, None


def _ensure_output_parent(root: Path, output_directory: Path) -> None:
    """Create missing output parents one directory at a time without symlinks."""

    relative_parent = output_directory.parent.relative_to(root)
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current)
            except FileExistsError:
                info = os.lstat(current)
            else:
                continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError("unsafe output parent")


def _check_capabilities(request: ComponentRequest) -> str | None:
    """Return an unavailable reason for an unsupported requested capability."""

    supported = set(REQUIRED_CAPABILITIES) | set(OPTIONAL_CAPABILITIES)
    missing = sorted(set(request.required_capabilities) - supported)
    if missing:
        return "unsupported_required_capability: " + ",".join(missing)
    return None


def _encoder_probe() -> tuple[dict[str, Any] | None, str | None]:
    """Probe the bounded MP4 backend without importing simulation modules.

    Returns:
        Encoder environment record and no error, or (None, reason).
    """

    from robot_sf.common.optional_import import try_import  # noqa: PLC0415

    optional_modules = {
        "imageio": try_import("imageio"),
        "imageio_ffmpeg": try_import("imageio_ffmpeg"),
        "numpy": try_import("numpy"),
        "PIL": try_import("PIL"),
    }
    missing = [name for name, module in optional_modules.items() if module is None]
    if missing:
        return None, "missing_encoder_backend: " + ",".join(sorted(missing))
    imageio = optional_modules["imageio"]
    imageio_ffmpeg = optional_modules["imageio_ffmpeg"]
    numpy = optional_modules["numpy"]
    PIL = optional_modules["PIL"]
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        return None, "missing_encoder_backend: ffmpeg-binary"
    return (
        {
            "encoder": ENCODER_NAME,
            "codec": ENCODER_CODEC,
            "ffmpeg_binary": Path(executable).name,
            "imageio_version": getattr(imageio, "__version__", "unknown"),
            "pillow_version": getattr(PIL, "__version__", "unknown"),
            "numpy_version": getattr(numpy, "__version__", "unknown"),
            "platform": platform.system(),
            "audio_policy": AUDIO_POLICY_SILENT,
            "encoder_timeout_s": MAX_ENCODER_SECONDS,
            "decoder_timeout_s": MAX_DECODER_SECONDS,
        },
        None,
    )


def _require_imageio() -> Any:
    """Import imageio for MP4 operations and fail closed when unavailable.

    Returns:
        The imageio v2 module.

    Raises:
        ImportError: If the configured encoder dependency is unavailable.
    """

    from robot_sf.common.optional_import import try_import  # noqa: PLC0415

    imageio = try_import("imageio.v2")
    if imageio is None:
        raise ImportError("review encode requires imageio")
    return imageio


@contextmanager
def _operation_deadline(seconds: float) -> Iterator[None]:
    """Bound an ffmpeg-backed operation or fail closed on worker threads."""

    if threading_is_not_main():
        raise _OperationTimeout("bounded deadline is unavailable on a non-main thread")
    if signal.getsignal(signal.SIGALRM) is None:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _alarm_handler(_signum: int, _frame: Any) -> None:
        raise _OperationTimeout

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def threading_is_not_main() -> bool:
    """Return whether the current caller cannot safely install SIGALRM."""

    import threading  # noqa: PLC0415

    return threading.current_thread() is not threading.main_thread()


def _read_json(source: Path | Any, *, name: str | None = None) -> Any:
    """Read a UTF-8 JSON document or raise a stable source failure.

    Returns:
        Parsed JSON value.
    """

    source_name = name or (source.name if isinstance(source, Path) else "source")
    try:
        if isinstance(source, Path):
            payload = source.read_bytes()
        else:
            source.seek(0)
            payload = source.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return json.loads(payload.decode("utf-8"))
    except FileNotFoundError as error:
        raise _SourceLoadError(f"{source_name}: source_missing") from error
    except (OSError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _SourceLoadError(f"{source_name}: source_unreadable") from error


def _resolve_under(path_value: str, root: Path) -> Path | None:
    """Resolve a relative URI below a canonical root.

    Returns:
        Canonical path below root, or None for an unsafe URI.
    """

    if not _safe_relative(path_value):
        return None
    try:
        resolved_root = root.resolve(strict=True)
        resolved = (resolved_root / path_value).resolve(strict=False)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _validate_manifest_header(doc: dict[str, Any]) -> tuple[float | None, int | None, str | None]:
    """Validate manifest identity, source rate, count, and duration.

    Returns:
        Source fps/count and no error, or an error reason.
    """

    allowed = {"schema_version", SOURCE_FPS, SOURCE_FRAMES, "frame_paths", "frame_digests"}
    unknown = sorted((key for key in doc if key not in allowed), key=str)
    if unknown:
        return None, None, f"manifest_invalid: unknown keys: {', '.join(map(str, unknown))}"
    if doc.get("schema_version") != FRAME_MANIFEST_FORMAT:
        return None, None, "manifest_invalid: unsupported schema_version"
    fps = doc.get(SOURCE_FPS)
    frames = doc.get(SOURCE_FRAMES)
    if not _finite_number(fps) or float(fps) <= 0 or float(fps) > MAX_SOURCE_FPS:
        return None, None, "manifest_invalid: source_fps is outside the finite bound"
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        return None, None, "manifest_invalid: source_frames must be a positive integer"
    if frames > MAX_SOURCE_FRAMES:
        return None, None, f"resource_limit: source_frames>{MAX_SOURCE_FRAMES}"
    if frames / float(fps) > MAX_SOURCE_DURATION_S:
        return None, None, f"resource_limit: source_duration_s>{MAX_SOURCE_DURATION_S:g}"
    return float(fps), frames, None


def _validate_manifest_frames(
    doc: dict[str, Any], frame_count: int
) -> tuple[list[str] | None, list[str] | None, str | None]:
    """Validate manifest frame paths and per-frame integrity digests.

    Returns:
        Normalized paths/digests and no error, or an error reason.
    """

    paths = doc.get("frame_paths")
    digests = doc.get("frame_digests")
    if not isinstance(paths, list) or len(paths) != frame_count:
        return None, None, "manifest_invalid: frame_paths must list every source frame"
    if not isinstance(digests, list) or len(digests) != frame_count:
        return None, None, "manifest_invalid: frame_digests must list every source frame"
    for index, path in enumerate(paths):
        if not _safe_relative(path):
            return None, None, f"manifest_invalid: frame_paths[{index}] is unsafe"
    for index, digest in enumerate(digests):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            return None, None, f"manifest_invalid: frame_digests[{index}] is not SHA-256"
    return list(paths), [str(value).lower() for value in digests], None


def _validate_manifest(doc: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the leaf frame-sequence-manifest.v1 input document.

    Returns:
        Normalized manifest and no error, or (None, reason).
    """

    if not isinstance(doc, dict):
        return None, "manifest_invalid: document must be an object"
    fps, frames, error = _validate_manifest_header(doc)
    if error is not None or fps is None or frames is None:
        return None, error or "manifest_invalid"
    paths, digests, error = _validate_manifest_frames(doc, frames)
    if error is not None or paths is None or digests is None:
        return None, error or "manifest_invalid"
    return {
        "schema_version": FRAME_MANIFEST_FORMAT,
        SOURCE_FPS: fps,
        SOURCE_FRAMES: frames,
        "frame_paths": paths,
        "frame_digests": digests,
    }, None


def _normalise_frame_array(frame: Any) -> Any:
    """Convert one decoded image to a bounded uint8 RGB array.

    Returns:
        A copied HxWx3 uint8 RGB array.
    """

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    array = np.asarray(frame)
    if array.ndim == 2:
        image = Image.fromarray(array)
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        image = Image.fromarray(array)
    else:
        raise _SourceLoadError("source_frame_invalid: expected a 2D or RGB/RGBA image")
    if image.width < 1 or image.height < 1 or image.width * image.height > MAX_OUTPUT_PIXELS:
        raise _SourceLoadError("resource_limit: source_frame_pixels")
    rgb = image.convert("RGB")
    if rgb.width * rgb.height * 3 > MAX_SOURCE_BUFFER_BYTES:
        raise _SourceLoadError("resource_limit: source_frame_buffer_bytes")
    return np.asarray(rgb, dtype=np.uint8).copy()


@contextmanager
def _verified_file_snapshot(path: Path, *, max_bytes: int) -> Iterator[tuple[Any, str, int]]:
    """Expose a bounded private snapshot for hash-and-decode operations.

    The source path is read once into a private spooled file.  Consumers hash
    and decode that same snapshot, so replacing the named path between those
    operations cannot make the digest describe different bytes from the
    decoded image.  The source signature is checked both before and after the
    consumer uses the snapshot to reject a concurrent source mutation.

    Yields:
        Snapshot file object, its SHA-256 digest, and the encoded byte count.
    """

    try:
        before = _file_signature(path)
    except OSError as error:
        raise OSError(f"source file is not readable: {path.name}") from error
    if max_bytes < 1 or before[2] > max_bytes:
        raise _SourceLoadError(f"resource_limit: source_bytes>{max_bytes}")

    with (
        path.open("rb") as source,
        tempfile.SpooledTemporaryFile(
            max_size=min(max_bytes, 8 * 1024 * 1024), mode="w+b"
        ) as snapshot,
    ):
        digest, total = _copy_to_snapshot(source, snapshot, path.name, max_bytes)
        _require_unchanged_file(path, before)

        snapshot.seek(0)
        try:
            yield snapshot, digest.hexdigest(), total
        finally:
            _require_unchanged_file(path, before)


@contextmanager
def _verified_named_file_snapshot(path: Path, *, max_bytes: int) -> Iterator[tuple[Path, str, int]]:
    """Expose a bounded named snapshot for decoders that require a path.

    The decoder receives the private snapshot path, never the mutable source
    path.  The source signature checks remain defense in depth: the snapshot
    is the integrity binding, while the checks still reject ordinary source
    mutation when the filesystem exposes a changed signature.

    Yields:
        Snapshot path, its SHA-256 digest, and the encoded byte count.
    """

    try:
        before = _file_signature(path)
    except OSError as error:
        raise OSError(f"source file is not readable: {path.name}") from error
    if max_bytes < 1 or before[2] > max_bytes:
        raise _SourceLoadError(f"resource_limit: source_bytes>{max_bytes}")

    snapshot_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="review-encode-", suffix=".source", mode="w+b", delete=False
        ) as snapshot:
            snapshot_path = Path(snapshot.name)
            with path.open("rb") as source:
                digest, total = _copy_to_snapshot(
                    source=source,
                    snapshot=snapshot,
                    name=path.name,
                    max_bytes=max_bytes,
                )
            snapshot.flush()
        _require_unchanged_file(path, before)
        try:
            yield snapshot_path, digest.hexdigest(), total
        finally:
            _require_unchanged_file(path, before)
    finally:
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except FileNotFoundError:
                pass


def _copy_to_snapshot(source: Any, snapshot: Any, name: str, max_bytes: int) -> tuple[Any, int]:
    """Copy one source into a bounded snapshot while computing its digest.

    Returns:
        SHA-256 hash object and encoded byte count.
    """

    digest = hashlib.sha256()
    total = 0
    try:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(chunk)
            if total > max_bytes:
                raise _SourceLoadError(f"resource_limit: source_bytes>{max_bytes}")
            digest.update(chunk)
            snapshot.write(chunk)
    except _SourceLoadError:
        raise
    except OSError as error:
        raise OSError(f"source file read failed: {name}") from error
    return digest, total


def _require_unchanged_file(path: Path, before: tuple[int, int, int, int]) -> None:
    """Raise a stable failure if a source path changed during snapshot use."""

    try:
        after = _file_signature(path)
    except OSError as error:
        raise _SourceLoadError(f"source_changed_during_read: {path.name}") from error
    if before != after:
        raise _SourceLoadError(f"source_changed_during_read: {path.name}")


def _decode_image_snapshot(snapshot: Any, name: str) -> Any:
    """Decode one bounded private image snapshot into an RGB array.

    Returns:
        A copied HxWx3 uint8 RGB array.
    """

    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(snapshot) as image:
            if image.width * image.height > MAX_OUTPUT_PIXELS:
                raise _SourceLoadError("resource_limit: source_frame_pixels")
            image.load()
            return _normalise_frame_array(image)
    except _SourceLoadError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise _SourceLoadError(f"source_frame_decode_failed: {name}") from error


def _decode_image(path: Path) -> Any:
    """Decode one source frame with a pixel ceiling and stability check.

    Returns:
        A copied HxWx3 uint8 RGB array.
    """

    try:
        before = _file_signature(path)
        if before[2] > MAX_SOURCE_FILE_BYTES:
            raise _SourceLoadError(f"resource_limit: source_bytes>{MAX_SOURCE_FILE_BYTES}")
        with path.open("rb") as source:
            result = _decode_image_snapshot(source, path.name)
        after = _file_signature(path)
    except _SourceLoadError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise _SourceLoadError(f"source_frame_decode_failed: {path.name}") from error
    if before != after:
        raise _SourceLoadError(f"source_changed_during_read: {path.name}")
    return result


def _load_manifest_frame(
    frame_path: Path | None,
    index: int,
    declared_digest: str,
    total_bytes: int,
    decoded_bytes: int,
) -> tuple[Any, str, int, int]:
    """Verify and decode one manifest frame within both memory ceilings.

    Returns:
        Decoded frame, observed digest, encoded-byte total, and decoded-byte total.
    """

    if frame_path is None or not frame_path.is_file():
        raise _SourceLoadError(f"source_frame_missing: frame_paths[{index}]")
    remaining_bytes = MAX_SOURCE_BUFFER_BYTES - total_bytes
    if remaining_bytes < 1:
        raise _SourceLoadError(f"resource_limit: source_buffer_bytes>{MAX_SOURCE_BUFFER_BYTES}")
    with _verified_file_snapshot(
        frame_path,
        max_bytes=min(MAX_SOURCE_FILE_BYTES, remaining_bytes),
    ) as (snapshot, observed, size):
        total_bytes += size
        if total_bytes > MAX_SOURCE_BUFFER_BYTES:
            raise _SourceLoadError(f"resource_limit: source_buffer_bytes>{MAX_SOURCE_BUFFER_BYTES}")
        if observed.lower() != declared_digest.lower():
            raise _SourceLoadError(
                f"source_frame_digest_mismatch: frame_paths[{index}]",
                diagnostics=(
                    {
                        "code": "source_frame_digest_mismatch",
                        "frame_index": index,
                        "declared_sha256": str(declared_digest).lower(),
                        "observed_sha256": observed,
                    },
                ),
            )
        decoded = _decode_image_snapshot(snapshot, frame_path.name)
    decoded_bytes += int(decoded.nbytes)
    if decoded_bytes > MAX_SOURCE_BUFFER_BYTES:
        raise _SourceLoadError(
            f"resource_limit: decoded_source_buffer_bytes>{MAX_SOURCE_BUFFER_BYTES}"
        )
    return decoded, observed, total_bytes, decoded_bytes


def _load_manifest_source(ref: SourceRef, root: Path) -> _Source:
    """Load and integrity-check every frame named by a frame manifest.

    Returns:
        Decoded, source-digest-bound frame sequence.
    """

    manifest_path = _resolve_under(ref.uri, root)
    if manifest_path is None:
        raise _SourceLoadError(f"{ref.artifact_id}: source_uri_escapes_root")
    if not ref.sha256:
        raise _SourceLoadError(f"source_digest_missing: {ref.artifact_id}")
    with _verified_file_snapshot(
        manifest_path,
        max_bytes=MAX_SOURCE_FILE_BYTES,
    ) as (snapshot, manifest_digest, manifest_bytes):
        if manifest_digest.lower() != ref.sha256.lower():
            raise _SourceLoadError(
                f"source_digest_mismatch: {ref.artifact_id}",
                diagnostics=(
                    {
                        "code": "source_digest_mismatch",
                        "artifact_id": ref.artifact_id,
                        "declared_sha256": ref.sha256.lower(),
                        "observed_sha256": manifest_digest,
                    },
                ),
            )
        document = _read_json(snapshot, name=manifest_path.name)
    manifest, error = _validate_manifest(document)
    if manifest is None or error is not None:
        raise _SourceLoadError(error or "manifest_invalid")

    frames: list[Any] = []
    frame_digests: list[str] = []
    total_bytes = manifest_bytes
    decoded_bytes = 0
    for index, (frame_uri, declared_digest) in enumerate(
        zip(manifest["frame_paths"], manifest["frame_digests"], strict=True)
    ):
        frame_path = _resolve_under(str(frame_uri), manifest_path.parent)
        decoded, observed, total_bytes, decoded_bytes = _load_manifest_frame(
            frame_path,
            index,
            declared_digest,
            total_bytes,
            decoded_bytes,
        )
        frames.append(decoded)
        frame_digests.append(observed)
    return _Source(
        frames=tuple(frames),
        source_fps=float(manifest[SOURCE_FPS]),
        source_ref=ref,
        source_kind="frame-sequence",
        source_document=manifest,
        observed_sha256=manifest_digest,
        frame_digests=tuple(frame_digests),
    )


def _positive_infinite(value: Any) -> bool:
    """Return whether metadata explicitly marks an unknown positive bound."""

    return isinstance(value, float) and math.isinf(value) and value > 0


def _clip_metadata_count_error(nframes: Any) -> str | None:
    """Validate a decoder-reported frame count when it is finite.

    Returns:
        Stable admission error, or None for an absent/valid count.
    """

    if nframes is None or _positive_infinite(nframes):
        return None
    if not _finite_number(nframes) or float(nframes) < 1 or not float(nframes).is_integer():
        return "source_clip_invalid: decoder frame count is invalid"
    if float(nframes) > MAX_SOURCE_FRAMES:
        return f"resource_limit: source_frames>{MAX_SOURCE_FRAMES}"
    return None


def _clip_metadata_duration_error(duration: Any, fps: float) -> str | None:
    """Validate a decoder-reported duration and its implied frame count.

    Returns:
        Stable admission error, or None for an absent/valid duration.
    """

    if duration is None or _positive_infinite(duration):
        return None
    if not _finite_number(duration) or float(duration) < 0:
        return "source_clip_invalid: decoder duration is invalid"
    if float(duration) > MAX_SOURCE_DURATION_S:
        return f"resource_limit: source_duration_s>{MAX_SOURCE_DURATION_S:g}"
    if float(duration) * fps > MAX_SOURCE_FRAMES + 1e-9:
        return f"resource_limit: source_frames>{MAX_SOURCE_FRAMES}"
    return None


def _clip_metadata_preflight(metadata: Any) -> tuple[float | None, int | None, str | None]:
    """Validate decoder metadata before the first source frame is allocated.

    Returns:
        Decoder fps, bounded RGB24 frame bytes, and no error, or an admission
        error when the decoder does not expose a safe geometry contract.
    """

    if not isinstance(metadata, dict):
        return None, None, "source_clip_invalid: decoder metadata is not an object"
    fps = metadata.get("fps")
    if not _finite_number(fps) or float(fps) <= 0 or float(fps) > MAX_SOURCE_FPS:
        return None, None, "source_clip_invalid: decoder did not provide bounded fps"
    size = metadata.get("size")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(not _finite_number(value) or not float(value).is_integer() for value in size)
    ):
        return None, None, "source_clip_invalid: decoder did not provide bounded dimensions"
    width, height = (int(value) for value in size)
    if width < 1 or height < 1:
        return None, None, "source_clip_invalid: decoder dimensions are invalid"
    pixels = width * height
    if pixels > MAX_OUTPUT_PIXELS:
        return None, None, "resource_limit: source_frame_pixels"
    frame_bytes = pixels * 3  # imageio-ffmpeg's file reader yields uint8 RGB24 by default.
    if frame_bytes > MAX_SOURCE_BUFFER_BYTES:
        return None, None, "resource_limit: source_frame_buffer_bytes"
    count_error = _clip_metadata_count_error(metadata.get("nframes"))
    if count_error is not None:
        return None, None, count_error
    duration_error = _clip_metadata_duration_error(metadata.get("duration"), float(fps))
    if duration_error is not None:
        return None, None, duration_error
    return float(fps), frame_bytes, None


def _normalise_bounded_clip_frame(
    frame: Any, frame_bytes: int, width: int, height: int
) -> tuple[Any, int]:
    """Reject and copy one raw decoder frame within the preflight contract.

    Returns:
        Normalized RGB frame and its retained byte count.
    """

    import numpy as np  # noqa: PLC0415

    try:
        raw = memoryview(frame)
    except TypeError as error:
        raise _SourceLoadError("source_clip_invalid: decoder frame is not raw bytes") from error
    if raw.nbytes > frame_bytes:
        raise _SourceLoadError("resource_limit: decoded_source_frame_bytes exceeds metadata bound")
    if raw.nbytes != frame_bytes:
        raise _SourceLoadError("source_clip_invalid: decoder frame byte count is invalid")
    try:
        normalised = (
            np.frombuffer(raw, dtype=np.uint8, count=frame_bytes).reshape((height, width, 3)).copy()
        )
    except (TypeError, ValueError) as error:
        raise _SourceLoadError("source_clip_invalid: decoder frame geometry is invalid") from error
    normalised_bytes = int(normalised.nbytes)
    if normalised_bytes > frame_bytes:
        raise _SourceLoadError(
            "resource_limit: normalized_source_frame_bytes exceeds metadata bound"
        )
    return normalised, normalised_bytes


def _read_bounded_clip_frames(reader: Any, frame_bytes: int, width: int, height: int) -> list[Any]:
    """Read raw RGB24 frames within preflighted allocation and buffer bounds.

    ``imageio_ffmpeg.read_frames`` yields raw bytes after it has received the
    metadata header.  Its read loop requests at most the exact RGB24 frame
    size, so no decoded-array allocation occurs before this boundary checks
    the configured per-frame and cumulative ceilings.  A bytes-length check
    remains defense in depth for alternate readers and malformed decoder
    output.

    Returns:
        Decoded, normalized frames within the declared source limits.
    """

    if frame_bytes != width * height * 3:
        raise _SourceLoadError("source_clip_invalid: decoder frame geometry is inconsistent")
    frames: list[Any] = []
    decoded_bytes = 0
    frame_iterator = iter(reader)
    for _frame_index in range(MAX_SOURCE_FRAMES):
        if decoded_bytes + frame_bytes > MAX_SOURCE_BUFFER_BYTES:
            raise _SourceLoadError(f"resource_limit: source_buffer_bytes>{MAX_SOURCE_BUFFER_BYTES}")
        try:
            frame = next(frame_iterator)
        except StopIteration:
            return frames
        normalised, normalised_bytes = _normalise_bounded_clip_frame(
            frame, frame_bytes, width, height
        )
        decoded_bytes += normalised_bytes
        frames.append(normalised)

    if decoded_bytes + frame_bytes > MAX_SOURCE_BUFFER_BYTES:
        raise _SourceLoadError(f"resource_limit: source_buffer_bytes>{MAX_SOURCE_BUFFER_BYTES}")
    try:
        next(frame_iterator)
    except StopIteration:
        return frames
    raise _SourceLoadError(f"resource_limit: source_frames>{MAX_SOURCE_FRAMES}")


def _decode_clip_frames(path: Path) -> tuple[list[Any], float]:
    """Decode bounded clip frames and require an explicit decoder frame rate.

    Returns:
        Decoded RGB frames and the source fps.
    """

    import imageio_ffmpeg  # noqa: PLC0415

    frame_iterator = None
    try:
        with _operation_deadline(MAX_DECODER_SECONDS):
            frame_iterator = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", bpp=3)
            metadata = next(frame_iterator)
            fps, frame_bytes, metadata_error = _clip_metadata_preflight(metadata)
            if metadata_error is not None or fps is None or frame_bytes is None:
                raise _SourceLoadError(metadata_error or "source_clip_invalid")
            size = metadata.get("size") if isinstance(metadata, dict) else None
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise _SourceLoadError("source_clip_invalid: decoder dimensions are invalid")
            width, height = (int(value) for value in size)
            frames = _read_bounded_clip_frames(frame_iterator, frame_bytes, width, height)
    except _SourceLoadError:
        raise
    except _OperationTimeout as error:
        raise _SourceLoadError("decoder_timeout") from error
    except Exception as error:
        raise _SourceLoadError(f"decoder_failed: {type(error).__name__}") from error
    finally:
        if frame_iterator is not None:
            frame_iterator.close()
    if not frames:
        raise _SourceLoadError("source_clip_invalid: clip contains no decodable frames")
    return frames, float(fps)


def _load_clip_source(ref: SourceRef, root: Path) -> _Source:
    """Decode a source clip through imageio with hard count/time limits.

    Returns:
        Decoded, source-digest-bound clip frames.
    """

    clip_path = _resolve_under(ref.uri, root)
    if clip_path is None:
        raise _SourceLoadError(f"{ref.artifact_id}: source_uri_escapes_root")
    if not ref.sha256:
        raise _SourceLoadError(f"source_digest_missing: {ref.artifact_id}")
    with _verified_named_file_snapshot(
        clip_path,
        max_bytes=MAX_SOURCE_FILE_BYTES,
    ) as (snapshot_path, observed, _size):
        if observed.lower() != ref.sha256.lower():
            raise _SourceLoadError(
                f"source_digest_mismatch: {ref.artifact_id}",
                diagnostics=(
                    {
                        "code": "source_digest_mismatch",
                        "artifact_id": ref.artifact_id,
                        "declared_sha256": ref.sha256.lower(),
                        "observed_sha256": observed,
                    },
                ),
            )
        frames, source_fps = _decode_clip_frames(snapshot_path)
    duration = len(frames) / source_fps
    if duration > MAX_SOURCE_DURATION_S:
        raise _SourceLoadError(f"resource_limit: source_duration_s>{MAX_SOURCE_DURATION_S:g}")
    return _Source(
        frames=tuple(frames),
        source_fps=source_fps,
        source_ref=ref,
        source_kind="source-clip",
        source_document={
            "schema_version": SOURCE_CLIP_FORMAT,
            SOURCE_FPS: source_fps,
            SOURCE_FRAMES: len(frames),
        },
        observed_sha256=observed,
        frame_digests=tuple(_sha256_bytes(frame.tobytes()) for frame in frames),
    )


def _validate_preset(config: dict[str, Any]) -> tuple[int, int, float, str | None]:
    """Return bounded output geometry or a stable preset error."""

    preset = config["preset"]
    width, height, fps = int(preset["width"]), int(preset["height"]), float(preset["fps"])
    if width < 16 or width > MAX_OUTPUT_WIDTH:
        return width, height, fps, "config_invalid: preset.width is outside bounds"
    if height < 16 or height > MAX_OUTPUT_HEIGHT:
        return width, height, fps, "config_invalid: preset.height is outside bounds"
    if width * height > MAX_OUTPUT_PIXELS:
        return width, height, fps, "resource_limit: output_pixels"
    if fps <= 0 or fps > MAX_OUTPUT_FPS:
        return width, height, fps, "config_invalid: preset.fps is outside bounds"
    return width, height, fps, None


def _select_source_ref(
    request: ComponentRequest,
) -> tuple[SourceRef | None, str | None, str | None]:
    """Select one source family with an explicit required/optional policy.

    Exactly one required source family wins over any optional source family.
    Multiple required families and multiple unqualified families are rejected
    rather than silently choosing an arbitrary source.

    Returns:
        Selected source, optional reason, and optional unavailable marker.
    """

    frame_refs = [ref for ref in request.sources if ref.format == FRAME_MANIFEST_FORMAT]
    clip_refs = [ref for ref in request.sources if ref.format == SOURCE_CLIP_FORMAT]
    required_families = [
        ("frame-sequence", frame_refs),
        ("source-clip", clip_refs),
    ]
    required = [item for item in required_families if item[0] in request.required_capabilities]
    if len(required) > 1:
        return (
            None,
            "source_invalid: multiple required source families are unsupported",
            STATUS_FAILED,
        )
    if required:
        family, refs = required[0]
        if not refs:
            return None, f"required_source_family_missing: {family}", STATUS_UNAVAILABLE
    elif not frame_refs and not clip_refs:
        return (
            None,
            "required_source_family_missing: frame-sequence or source-clip",
            STATUS_UNAVAILABLE,
        )
    elif frame_refs and clip_refs:
        return (
            None,
            "source_invalid: multiple source families require one required capability",
            STATUS_FAILED,
        )
    else:
        refs = frame_refs or clip_refs
    if len(refs) > 1:
        return None, "source_invalid: multiple source artifacts in one family", STATUS_FAILED
    return refs[0], None, None


def _load_selected_source(ref: SourceRef, root: Path) -> _Source:
    """Load the selected frame-sequence or source-clip family.

    Returns:
        Source frames decoded from the selected artifact.
    """

    if ref.format == FRAME_MANIFEST_FORMAT:
        return _load_manifest_source(ref, root)
    return _load_clip_source(ref, root)


def _prepare_source(
    request: ComponentRequest, root: Path
) -> tuple[_Source | None, ComponentResult | None]:
    """Select and decode the one source family admitted by the request.

    Returns:
        Decoded source and no early result, or (None, terminal result).
    """

    selected, selection_error, selection_status = _select_source_ref(request)
    if selected is None or selection_error is not None:
        if selection_status == STATUS_UNAVAILABLE:
            return None, _unavailable(request, selection_error or "source_unavailable")
        return None, _failure(request, selection_error or "source_invalid")
    try:
        return _load_selected_source(selected, root), None
    except _SourceLoadError as error:
        if error.status == STATUS_UNAVAILABLE:
            return None, _unavailable(request, error.reason)
        return None, _failure(request, error.reason, diagnostics=error.diagnostics)
    except OSError as error:
        return None, _failure(request, f"source_unreadable: {type(error).__name__}")


def _prepare(
    request: ComponentRequest,
    root: Path,
    out_dir: Path,
) -> tuple[_Prepared | None, ComponentResult | None]:
    """Validate configuration, capabilities, source bytes, and output preset.

    Returns:
        Prepared inputs and no early result, or (None, result).
    """

    config, config_error = _validate_config(request.config)
    if config is None or config_error is not None:
        return None, _failure(request, config_error or "config_invalid")
    version_error = _check_version_compatible(config)
    if version_error is not None:
        return None, _failure(request, version_error)
    capability_error = _check_capabilities(request)
    if capability_error is not None:
        return None, _unavailable(request, capability_error)
    if config["audio_policy"] != AUDIO_POLICY_SILENT:
        return None, _failure(
            request,
            f"unsupported_audio_policy: v1 output is silent, got {config['audio_policy']!r}",
        )
    env, encoder_error = _encoder_probe()
    if env is None or encoder_error is not None:
        return None, _unavailable(request, encoder_error or "missing_encoder_backend")

    source, early = _prepare_source(request, root)
    if source is None or early is not None:
        return None, early
    width, height, out_fps, preset_error = _validate_preset(config)
    if preset_error is not None:
        return None, _failure(request, preset_error)
    return (
        _Prepared(
            root=root,
            out_dir=out_dir,
            config=config,
            env=env,
            source=source,
            width=width,
            height=height,
            out_fps=out_fps,
        ),
        None,
    )


def _number(
    item: dict[str, Any], key: str, operation: str, position: int
) -> tuple[float | None, str | None]:
    """Read one finite edit number without leaking conversion exceptions.

    Returns:
        Parsed finite value and no error, or (None, reason).
    """

    value = item.get(key)
    if not _finite_number(value):
        return None, f"edit_plan_invalid: {operation}[{position}].{key} must be finite"
    return float(value), None


def _validate_edit_values(
    operation: str,
    values: dict[str, float],
    position: int,
    duration_s: float,
) -> str | None:
    """Validate operation-specific finite values and bounds.

    Returns:
        Error reason, or None for valid values.
    """

    if operation == "cut":
        start, end = values["start_s"], values["end_s"]
        if not 0 <= start < end <= duration_s:
            return f"edit_plan_invalid: cut[{position}] out of range"
    elif operation == "pause":
        at_s, held = values["at_s"], values["duration_s"]
        if not 0 <= at_s <= duration_s or held <= 0 or held > MAX_PAUSE_DURATION_S:
            return f"edit_plan_invalid: pause[{position}] out of range"
    elif operation == "speed":
        start, end, factor = values["start_s"], values["end_s"], values["factor"]
        if (
            not 0 <= start < end <= duration_s
            or factor < MIN_SPEED_FACTOR
            or factor > MAX_SPEED_FACTOR
        ):
            return f"edit_plan_invalid: speed[{position}] out of range"
    elif (
        abs(values["x"]) > MAX_CROP_COORDINATE
        or abs(values["y"]) > MAX_CROP_COORDINATE
        or values["width"] <= 0
        or values["height"] <= 0
        or values["width"] > MAX_CROP_SIZE
        or values["height"] > MAX_CROP_SIZE
    ):
        return f"edit_plan_invalid: crop[{position}] out of range"
    return None


def _validate_edit_item(
    item: Any, position: int, duration_s: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and normalize one closed-vocabulary edit item.

    Returns:
        Normalized edit and no error, or (None, reason).
    """

    if not isinstance(item, dict):
        return None, f"edit_plan_invalid: edits[{position}] must be an object"
    operation = item.get("op")
    if not isinstance(operation, str) or operation not in _EDIT_KEYS:
        return None, f"edit_plan_invalid: unknown op {operation!r}"
    unknown = sorted((key for key in item if key not in _EDIT_KEYS[operation]), key=str)
    if unknown:
        return None, (
            f"edit_plan_invalid: {operation}[{position}] unknown keys: "
            + ", ".join(map(str, unknown))
        )
    values: dict[str, float] = {}
    for key in _EDIT_KEYS[operation] - {"op"}:
        value, error = _number(item, key, operation, position)
        if error is not None or value is None:
            return None, error or "edit_plan_invalid: non-finite value"
        values[key] = value
    error = _validate_edit_values(operation, values, position, duration_s)
    if error is not None:
        return None, error
    return {"op": operation, **values}, None


def _validate_edits(raw: Any, duration_s: float) -> tuple[list[dict[str, Any]], str | None]:
    """Validate the closed edit vocabulary and all finite/bounded values.

    Returns:
        Normalized edits and no error, or (empty, reason).
    """

    if not isinstance(raw, list):
        return [], "edit_plan_invalid: edits must be a list"
    if len(raw) > MAX_EDIT_COUNT:
        return [], f"resource_limit: edit_count>{MAX_EDIT_COUNT}"
    edits: list[dict[str, Any]] = []
    for position, item in enumerate(raw):
        normalized, error = _validate_edit_item(item, position, duration_s)
        if error is not None or normalized is None:
            return [], error or "edit_plan_invalid"
        edits.append(normalized)
    if sum(item["op"] == "crop" for item in edits) > 1:
        return [], "edit_plan_invalid: only one crop edit is supported"
    conflict = _find_time_conflict(edits)
    if conflict is not None:
        return [], "conflicting_time_map: " + conflict
    return edits, None


def _find_time_conflict(edits: list[dict[str, Any]]) -> str | None:
    """Detect overlapping speed edits with different factors.

    Returns:
        Conflict description, or None when the map is consistent.
    """

    speeds = [item for item in edits if item["op"] == "speed"]
    for left_position, left in enumerate(speeds):
        for right in speeds[left_position + 1 :]:
            overlap = min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"])
            if overlap > 0 and not math.isclose(
                left["factor"], right["factor"], rel_tol=0.0, abs_tol=1e-12
            ):
                return (
                    f"overlapping speed {left['factor']:g}x "
                    f"[{left['start_s']:g},{left['end_s']:g}] vs "
                    f"{right['factor']:g}x [{right['start_s']:g},{right['end_s']:g}]"
                )
    return None


def _apply_cuts(cuts: list[dict[str, Any]], duration_s: float) -> list[tuple[float, float]]:
    """Subtract half-open cut intervals from the source timeline.

    Returns:
        Ordered retained source intervals.
    """

    spans = [(0.0, duration_s)]
    for cut in sorted(cuts, key=lambda item: (item["start_s"], item["end_s"])):
        updated: list[tuple[float, float]] = []
        for start, end in spans:
            if end <= cut["start_s"] or start >= cut["end_s"]:
                updated.append((start, end))
                continue
            if start < cut["start_s"]:
                updated.append((start, cut["start_s"]))
            if end > cut["end_s"]:
                updated.append((cut["end_s"], end))
        spans = updated
    return spans


def _frame_indices(start_s: float, end_s: float, fps: float, frame_count: int) -> tuple[int, ...]:
    """Select source timestamps in a half-open interval with deterministic rounding.

    Returns:
        Source frame indices whose timestamps fall in the interval.
    """

    epsilon = 1e-9
    first = max(0, math.ceil(start_s * fps - epsilon))
    stop = min(frame_count, math.ceil(end_s * fps - epsilon))
    if stop <= first:
        return ()
    return tuple(range(first, stop))


def _source_segments(
    kept: list[tuple[float, float]],
    speeds: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    fps: float,
    frame_count: int,
) -> tuple[tuple[_TimelineSegment, ...], str | None]:
    """Split kept spans at speed boundaries and resample each actual span.

    Returns:
        Ordered timeline segments with source and presentation indices, or a
        stable resource-limit error before presentation indices are allocated.
    """

    segments: list[_TimelineSegment] = []
    segment_id = 0
    presentation_frame_count = 0
    for start, end in kept:
        boundaries = {start, end}
        for speed in speeds:
            if speed["start_s"] < end and speed["end_s"] > start:
                boundaries.add(max(start, speed["start_s"]))
                boundaries.add(min(end, speed["end_s"]))
        ordered = sorted(boundaries)
        for left, right in pairwise(ordered):
            indices = _frame_indices(left, right, fps, frame_count)
            if not indices:
                continue
            midpoint = (left + right) / 2.0
            active = [
                speed["factor"] for speed in speeds if speed["start_s"] <= midpoint < speed["end_s"]
            ]
            factor = active[0] if active else 1.0
            operation = (
                "speed"
                if not math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-12)
                else "cut"
                if cuts
                else "passthrough"
            )
            presentation_count = max(1, math.ceil(len(indices) / factor - 1e-12))
            presentation_frame_count += presentation_count
            if presentation_frame_count > MAX_OUTPUT_FRAMES:
                return (), f"resource_limit: output_frames>{MAX_OUTPUT_FRAMES}"
            presentation_indices = tuple(
                indices[min(len(indices) - 1, math.floor(position * factor + 1e-12))]
                for position in range(presentation_count)
            )
            segments.append(
                _TimelineSegment(
                    segment_id=segment_id,
                    operation=operation,
                    factor=float(factor),
                    source_start_s=float(left),
                    source_end_s=float(right),
                    source_indices=indices,
                    presentation_indices=presentation_indices,
                )
            )
            segment_id += 1
    return tuple(segments), None


def _pause_anchor(
    records: list[_PresentationFrame],
    kept: list[tuple[float, float]],
    at_s: float,
    duration_s: float,
    source_fps: float,
) -> int | None:
    """Find an endpoint-safe retained frame for a pause edit.

    Returns:
        Position in the retained presentation order, or None.
    """

    if not records:
        return None
    if math.isclose(at_s, duration_s, rel_tol=0.0, abs_tol=1e-9):
        if not any(math.isclose(end, duration_s, rel_tol=0.0, abs_tol=1e-9) for _, end in kept):
            return None
        # A fast resample can legitimately represent the retained source
        # endpoint with the last sampled frame before the raw terminal index.
        # The kept-span boundary, not the pre-resampling index, determines
        # whether this endpoint remains in the presentation timeline.
        return len(records) - 1
    active = any(start <= at_s < end for start, end in kept)
    if not active:
        at_boundary = any(math.isclose(end, at_s, rel_tol=0.0, abs_tol=1e-9) for _, end in kept)
        if not at_boundary:
            return None
        candidates = [
            (position, record)
            for position, record in enumerate(records)
            if record.source_index / source_fps <= at_s
        ]
        if not candidates:
            return None
    else:
        candidates = list(enumerate(records))
    target = min(
        len(records) - 1,
        max(0, math.floor(at_s * source_fps + 0.5)),
    )
    return min(
        candidates,
        key=lambda pair: (
            abs(pair[1].source_index - target),
            pair[1].source_index > target,
            pair[0],
        ),
    )[0]


def _apply_pauses(
    segments: tuple[_TimelineSegment, ...],
    pauses: list[dict[str, Any]],
    kept: list[tuple[float, float]],
    source_fps: float,
    out_fps: float,
    duration_s: float,
) -> tuple[tuple[_PresentationFrame, ...], tuple[str, ...], str | None]:
    """Insert rounded pauses after retained anchor frames, including endpoints.

    Returns:
        Presentation frames, stable diagnostics for dropped pauses, and an
        optional resource-limit error.
    """

    base: list[_PresentationFrame] = []
    for segment in segments:
        base.extend(
            _PresentationFrame(
                source_index=index,
                operation=segment.operation,
                segment_id=segment.segment_id,
                factor=segment.factor,
                source_start_s=segment.source_start_s,
                source_end_s=segment.source_end_s,
            )
            for index in segment.presentation_indices
        )
    diagnostics: list[str] = []
    insertions: list[tuple[int, int, int, float]] = []
    for order, pause in sorted(enumerate(pauses), key=lambda item: (item[1]["at_s"], item[0])):
        position = _pause_anchor(
            base,
            kept,
            pause["at_s"],
            duration_s,
            source_fps,
        )
        if position is None:
            diagnostics.append(
                f"dropped_pause: at_s {pause['at_s']:g} lies outside the retained timeline"
            )
            continue
        hold = max(1, math.ceil(pause["duration_s"] * out_fps - 1e-12))
        anchor_time = base[position].source_index / source_fps
        insertions.append((position, order, hold, anchor_time))
    projected_count = len(base) + sum(hold for _, _, hold, _ in insertions)
    if projected_count > MAX_OUTPUT_FRAMES:
        return (), tuple(diagnostics), f"resource_limit: output_frames>{MAX_OUTPUT_FRAMES}"
    final: list[_PresentationFrame] = []
    by_position: dict[int, list[tuple[int, int, float]]] = {}
    for position, order, hold, anchor_time in insertions:
        by_position.setdefault(position, []).append((order, hold, anchor_time))
    for position, record in enumerate(base):
        final.append(record)
        for order, hold, anchor_time in sorted(by_position.get(position, [])):
            pause_segment = len(segments) + order
            final.extend(
                _PresentationFrame(
                    source_index=record.source_index,
                    operation="pause",
                    segment_id=pause_segment,
                    factor=1.0,
                    source_start_s=anchor_time,
                    source_end_s=anchor_time,
                )
                for _ in range(hold)
            )
    return tuple(final), tuple(diagnostics), None


def _resolve_crop(
    crops: list[dict[str, Any]], width: int, height: int
) -> tuple[tuple[int, int, int, int] | None, str | None, str | None]:
    """Resolve a crop box, returning a partial diagnostic for clamping.

    Returns:
        Crop box, optional diagnostic, and optional failure reason.
    """

    if not crops:
        return None, None, None
    crop = crops[0]
    left = math.floor(crop["x"])
    top = math.floor(crop["y"])
    right = math.ceil(crop["x"] + crop["width"])
    bottom = math.ceil(crop["y"] + crop["height"])
    clamped = left < 0 or top < 0 or right > width or bottom > height
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        return None, None, "crop_invalid: crop lies outside the frame"
    diagnostic = "crop_clamped: box clipped to frame bounds" if clamped else None
    return (left, top, right, bottom), diagnostic, None


def _crop_provenance(crop_box: tuple[int, int, int, int] | None) -> dict[str, Any] | None:
    """Return the normalized crop operation carried by both output artifacts."""

    if crop_box is None:
        return None
    return {"operation": "crop", "box": list(crop_box)}


def _build_timeline(
    source: _Source,
    kept: list[tuple[float, float]],
    speeds: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    pauses: list[dict[str, Any]],
    duration_s: float,
    out_fps: float,
) -> tuple[
    tuple[_TimelineSegment, ...], tuple[_PresentationFrame, ...], tuple[str, ...], str | None
]:
    """Build the source and pause timeline before crop and output checks.

    Returns:
        Timeline segments, presentation records, pause diagnostics, and an
        optional stable planning error.
    """

    segments, segment_error = _source_segments(
        kept,
        speeds,
        cuts,
        source.source_fps,
        len(source.frames),
    )
    if segment_error is not None:
        return (), (), (), segment_error
    if not segments:
        return (), (), (), "empty_timeline: no source frames remain"
    records, pause_diagnostics, pause_error = _apply_pauses(
        segments,
        pauses,
        kept,
        source.source_fps,
        out_fps,
        duration_s,
    )
    if pause_error is not None:
        return (), (), pause_diagnostics, pause_error
    if not records:
        return (), (), pause_diagnostics, "empty_timeline: no presentation frames remain"
    return segments, records, pause_diagnostics, None


def _plan_order(prepared: _Prepared) -> tuple[_Planned | None, str | None]:
    """Build a split, resource-bounded presentation timeline.

    Returns:
        Validated plan and no error, or (None, reason).
    """

    source = prepared.source
    duration_s = len(source.frames) / source.source_fps
    edits, edit_error = _validate_edits(prepared.config.get("edits", []), duration_s)
    if edit_error is not None:
        return None, edit_error
    cuts = [item for item in edits if item["op"] == "cut"]
    speeds = [item for item in edits if item["op"] == "speed"]
    pauses = [item for item in edits if item["op"] == "pause"]
    crops = [item for item in edits if item["op"] == "crop"]
    kept = _apply_cuts(cuts, duration_s)
    if not kept:
        return None, "empty_timeline: cuts remove the whole source"
    segments, records, pause_diagnostics, timeline_error = _build_timeline(
        source,
        kept,
        speeds,
        cuts,
        pauses,
        duration_s,
        prepared.out_fps,
    )
    if timeline_error is not None:
        return None, timeline_error
    crop_box, crop_diagnostic, crop_error = _resolve_crop(crops, prepared.width, prepared.height)
    if crop_error is not None:
        return None, crop_error
    diagnostics = list(pause_diagnostics)
    if crop_diagnostic is not None:
        diagnostics.append(crop_diagnostic)
    crop_width = crop_box[2] - crop_box[0] if crop_box else prepared.width
    crop_height = crop_box[3] - crop_box[1] if crop_box else prepared.height
    if len(records) > MAX_OUTPUT_FRAMES:
        return None, f"resource_limit: output_frames>{MAX_OUTPUT_FRAMES}"
    estimated_bytes = len(records) * crop_width * crop_height * 3
    if estimated_bytes > MAX_OUTPUT_BUFFER_BYTES:
        return None, f"resource_limit: output_buffer_bytes>{MAX_OUTPUT_BUFFER_BYTES}"
    return (
        _Planned(
            frames=records,
            segments=segments,
            diagnostics=tuple(diagnostics),
            crop_box=crop_box,
            source_duration_s=duration_s,
        ),
        None,
    )


def _build_time_map(prepared: _Prepared, plan: _Planned) -> dict[str, Any]:
    """Build and locally validate the presentation-time-map.v1 payload.

    Returns:
        Schema-valid time-map payload.
    """

    records = plan.frames
    segments: list[dict[str, Any]] = []
    start_position = 0
    while start_position < len(records):
        first = records[start_position]
        end_position = start_position + 1
        while end_position < len(records) and records[end_position].segment_id == first.segment_id:
            end_position += 1
        block = records[start_position:end_position]
        segments.append(
            {
                "segment": len(segments),
                "operation": first.operation,
                "factor": first.factor,
                "presentation_start_s": start_position / prepared.out_fps,
                "presentation_end_s": end_position / prepared.out_fps,
                "source_start_s": first.source_start_s,
                "source_end_s": first.source_end_s,
                "frame_count": len(block),
                "source_frame_start": block[0].source_index,
                "source_frame_end": block[-1].source_index,
            }
        )
        start_position = end_position
    payload = {
        "schema_version": TIMEMAP_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "source_fps": prepared.source.source_fps,
        "presentation_fps": prepared.out_fps,
        "source_frames": len(prepared.source.frames),
        "source_duration_s": plan.source_duration_s,
        "presentation_duration_s": len(records) / prepared.out_fps,
        "segments": segments,
        "frame_order_source_indices": [record.source_index for record in records],
        "first_source_frame": records[0].source_index,
        "terminal_source_frame": records[-1].source_index,
        "crop": _crop_provenance(plan.crop_box),
        "evidence_status": "diagnostic-only",
        "evidence_boundary": "diagnostic-only; not benchmark evidence",
        "benchmark_success": False,
        "scientific_claim_allowed": False,
    }
    _validate_time_map(payload)
    return payload


def _render_source_frame(
    frame: Any,
    width: int,
    height: int,
    crop_box: tuple[int, int, int, int] | None,
) -> Any:
    """Resize and crop one actual decoded source frame.

    Returns:
        A copied uint8 RGB output frame.
    """

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    image = Image.fromarray(frame)
    if image.size != (width, height):
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((width, height), resampling)
    if crop_box is not None:
        image = image.crop(crop_box)
    return np.asarray(image, dtype=np.uint8).copy()


def _encode_mp4(
    frames: list[Any], path: Path, fps: float, env: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    """Encode presentation frames with a bounded imageio-ffmpeg call.

    Returns:
        Failure reason and encoder receipt record.
    """

    record = dict(env)
    try:
        imageio = _require_imageio()
        with _operation_deadline(MAX_ENCODER_SECONDS):
            imageio.mimsave(
                str(path),
                frames,
                fps=fps,
                codec=ENCODER_CODEC,
                macro_block_size=None,
            )
    except _OperationTimeout:
        return "encoder_timeout", record
    except Exception as error:  # noqa: BLE001 - encoder failure is a stable component result
        return f"encoder_failed: {type(error).__name__}", record
    if not path.is_file():
        return "encoder_failed: output_missing", record
    try:
        size = path.stat().st_size
    except OSError:
        return "encoder_failed: output_unreadable", record
    if size <= 0:
        return "encoder_failed: output_empty", record
    if size > MAX_OUTPUT_FILE_BYTES:
        return f"resource_limit: encoded_bytes>{MAX_OUTPUT_FILE_BYTES}", record
    record["output_fps"] = fps
    record["output_frames"] = len(frames)
    record["output_bytes"] = size
    return None, record


def _decode_frame_count(path: Path) -> tuple[int | None, str | None]:
    """Count output frames with a decoder timeout and frame ceiling.

    Returns:
        Decoded count and no error, or (None, reason).
    """

    try:
        imageio = _require_imageio()
        count = 0
        with _operation_deadline(MAX_DECODER_SECONDS):
            with imageio.get_reader(str(path)) as reader:
                for _frame in reader:
                    count += 1
                    if count > MAX_OUTPUT_FRAMES:
                        return None, f"decoder_frame_limit>{MAX_OUTPUT_FRAMES}"
        return count, None
    except _OperationTimeout:
        return None, "decoder_timeout"
    except Exception as error:  # noqa: BLE001 - decoder failure is reported, not raised
        return None, f"decoder_failed: {type(error).__name__}"


def _validate_time_map_segment(segment: Any, previous_end: float) -> float:  # noqa: C901
    """Validate one leaf time-map segment.

    Returns:
        The segment presentation end used by the next interval.
    """

    if not isinstance(segment, dict):
        raise ValueError("time-map segment is not an object")
    if set(segment) != _TIMEMAP_SEGMENT_KEYS:
        raise ValueError("time-map segment keys are invalid")
    if segment.get("operation") not in {"cut", "speed", "pause", "passthrough"}:
        raise ValueError("time-map operation is invalid")
    numeric = (
        "factor",
        "presentation_start_s",
        "presentation_end_s",
        "source_start_s",
        "source_end_s",
    )
    if any(not _finite_number(segment.get(key)) for key in numeric):
        raise ValueError("time-map segment contains non-finite values")
    if not math.isclose(segment["presentation_start_s"], previous_end, abs_tol=1e-9):
        raise ValueError("time-map presentation intervals are not contiguous")
    if segment["presentation_end_s"] <= segment["presentation_start_s"]:
        raise ValueError("time-map presentation intervals are not ordered")
    if segment["source_end_s"] < segment["source_start_s"]:
        raise ValueError("time-map source interval is reversed")
    if (
        isinstance(segment.get("frame_count"), bool)
        or not isinstance(segment.get("frame_count"), int)
        or segment["frame_count"] < 1
    ):
        raise ValueError("time-map frame_count is invalid")
    for key in ("segment", "source_frame_start", "source_frame_end"):
        if isinstance(segment.get(key), bool) or not isinstance(segment.get(key), int):
            raise ValueError("time-map frame identity is invalid")
    if segment["segment"] < 0 or segment["source_frame_start"] < 0:
        raise ValueError("time-map frame identity is invalid")
    if segment["source_frame_end"] < segment["source_frame_start"]:
        raise ValueError("time-map source frame interval is reversed")
    if segment["operation"] == "speed" and not (
        MIN_SPEED_FACTOR <= segment["factor"] <= MAX_SPEED_FACTOR
    ):
        raise ValueError("time-map speed factor is outside bounds")
    if segment["operation"] != "speed" and not math.isclose(
        segment["factor"], 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("time-map non-speed factor is invalid")
    return float(segment["presentation_end_s"])


def _validate_crop_provenance(value: Any) -> None:
    """Validate normalized crop coordinates stored in a leaf artifact."""

    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _CROP_PROVENANCE_KEYS:
        raise ValueError("crop provenance is invalid")
    if value["operation"] != "crop":
        raise ValueError("crop provenance operation is invalid")
    box = value["box"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(
            isinstance(coordinate, bool) or not isinstance(coordinate, int) for coordinate in box
        )
    ):
        raise ValueError("crop provenance box is invalid")
    left, top, right, bottom = box
    if (
        left < 0
        or top < 0
        or right <= left
        or bottom <= top
        or right > MAX_OUTPUT_WIDTH
        or bottom > MAX_OUTPUT_HEIGHT
    ):
        raise ValueError("crop provenance box is out of bounds")


def _validate_time_map(payload: Any) -> None:  # noqa: C901, PLR0912
    """Validate the SREV-10 leaf time-map shape and finite interval semantics."""

    if not isinstance(payload, dict) or payload.get("schema_version") != TIMEMAP_SCHEMA_VERSION:
        raise ValueError("time-map schema_version is invalid")
    if set(payload) != _TIMEMAP_KEYS:
        raise ValueError("time-map keys are invalid")
    required = {
        "component",
        "source_fps",
        "presentation_fps",
        "source_frames",
        "source_duration_s",
        "presentation_duration_s",
        "segments",
        "frame_order_source_indices",
        "first_source_frame",
        "terminal_source_frame",
    }
    if not required <= set(payload) or payload["component"] != COMPONENT_ID:
        raise ValueError("time-map envelope is invalid")
    if (
        isinstance(payload["source_frames"], bool)
        or not isinstance(payload["source_frames"], int)
        or payload["source_frames"] < 1
        or payload["source_frames"] > MAX_SOURCE_FRAMES
        or not _finite_number(payload["source_fps"])
        or not _finite_number(payload["presentation_fps"])
        or payload["source_fps"] <= 0
        or payload["source_fps"] > MAX_SOURCE_FPS
        or payload["presentation_fps"] <= 0
        or payload["presentation_fps"] > MAX_OUTPUT_FPS
    ):
        raise ValueError("time-map source geometry is invalid")
    for key in ("source_duration_s", "presentation_duration_s"):
        if not _finite_number(payload[key]) or payload[key] < 0:
            raise ValueError("time-map duration is invalid")
    if payload["source_duration_s"] > MAX_SOURCE_DURATION_S:
        raise ValueError("time-map source duration exceeds bounds")
    if payload["evidence_status"] != "diagnostic-only":
        raise ValueError("time-map evidence status is invalid")
    if payload["evidence_boundary"] != "diagnostic-only; not benchmark evidence":
        raise ValueError("time-map evidence boundary is invalid")
    if (
        payload["benchmark_success"] is not False
        or payload["scientific_claim_allowed"] is not False
    ):
        raise ValueError("time-map evidence flags are invalid")
    _validate_crop_provenance(payload["crop"])
    order = payload["frame_order_source_indices"]
    if (
        not isinstance(order, list)
        or not order
        or any(isinstance(index, bool) or not isinstance(index, int) for index in order)
        or any(index < 0 or index >= payload["source_frames"] for index in order)
    ):
        raise ValueError("time-map frame order is invalid")
    if payload["first_source_frame"] != order[0] or payload["terminal_source_frame"] != order[-1]:
        raise ValueError("time-map endpoint frames disagree with frame order")
    if not isinstance(payload["segments"], list) or not payload["segments"]:
        raise ValueError("time-map segments are invalid")
    previous_end = 0.0
    frame_count = 0
    seen_segments: set[int] = set()
    for segment in payload["segments"]:
        segment_id = segment.get("segment") if isinstance(segment, dict) else None
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise ValueError("time-map segment identity is invalid")
        if segment_id in seen_segments:
            raise ValueError("time-map segment identity is duplicated")
        seen_segments.add(segment_id)
        if segment.get("source_frame_end", 0) >= payload["source_frames"]:
            raise ValueError("time-map source frame identity is out of range")
        previous_end = _validate_time_map_segment(segment, previous_end)
        frame_count += segment["frame_count"]
    if frame_count != len(order):
        raise ValueError("time-map segment counts disagree with frame order")
    if not math.isclose(previous_end, payload["presentation_duration_s"], abs_tol=1e-9):
        raise ValueError("time-map duration disagrees with segments")
    if not math.isclose(
        payload["source_duration_s"], payload["source_frames"] / payload["source_fps"], abs_tol=1e-9
    ):
        raise ValueError("time-map source duration disagrees with geometry")


def _validate_receipt(payload: Any) -> None:  # noqa: C901, PLR0912
    """Validate the SREV-10 encode-receipt.v1 leaf payload."""

    if not isinstance(payload, dict) or payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("receipt schema_version is invalid")
    if set(payload) != _RECEIPT_KEYS:
        raise ValueError("receipt keys are invalid")
    required = {
        "component",
        "component_version",
        "encoder",
        "source",
        "config_sha256",
        "edit_plan_digest",
        "output",
        "presentation_duration_s",
        "evidence_boundary",
    }
    if not required <= set(payload):
        raise ValueError("receipt required fields are missing")
    if payload["component"] != COMPONENT_ID or not isinstance(payload["component_version"], str):
        raise ValueError("receipt component is invalid")
    if (
        not isinstance(payload["source"], dict)
        or not _RECEIPT_SOURCE_REQUIRED_KEYS <= set(payload["source"])
        or not set(payload["source"]) <= _RECEIPT_SOURCE_KEYS
    ):
        raise ValueError("receipt leaf keys are invalid")
    for value, allowed in (
        (payload["encoder"], _RECEIPT_ENCODER_KEYS),
        (payload["output"], _RECEIPT_OUTPUT_KEYS),
    ):
        if not isinstance(value, dict) or set(value) != allowed:
            raise ValueError("receipt leaf keys are invalid")
    for digest in (
        payload["source"].get("sha256"),
        payload["source"].get("declared_sha256"),
        payload["config_sha256"],
        payload["edit_plan_digest"],
        payload["output"].get("video_sha256"),
        payload["output"].get("time_map_sha256"),
    ):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("receipt digest is invalid")
    if payload["source"]["sha256"].lower() != payload["source"]["declared_sha256"].lower():
        raise ValueError("receipt source digest binding is invalid")
    source = payload["source"]
    for field in ("schema", "source_commit", "config_identity", "units", "coordinate_frame"):
        if field in source and not isinstance(source[field], str):
            raise ValueError("receipt source identity is invalid")
    if source.get("source_commit"):
        if _SHA40_RE.fullmatch(source["source_commit"]) is None:
            raise ValueError("receipt source identity is invalid")
    if payload["evidence_status"] != "diagnostic-only":
        raise ValueError("receipt evidence status is invalid")
    if payload["evidence_boundary"] != "diagnostic-only; not benchmark evidence":
        raise ValueError("receipt evidence boundary is invalid")
    if (
        payload["benchmark_success"] is not False
        or payload["scientific_claim_allowed"] is not False
    ):
        raise ValueError("receipt evidence flags are invalid")
    _validate_crop_provenance(payload["crop"])
    if payload["audio_policy"] != AUDIO_POLICY_SILENT:
        raise ValueError("receipt audio policy is invalid")
    output = payload["output"]
    if (
        isinstance(output.get("frames"), bool)
        or not isinstance(output.get("frames"), int)
        or output["frames"] < 1
        or output["frames"] > MAX_OUTPUT_FRAMES
        or not _finite_number(output.get("width"))
        or not _finite_number(output.get("height"))
        or not _finite_number(output.get("fps"))
        or output["width"] < 1
        or output["height"] < 1
        or output["width"] > MAX_OUTPUT_WIDTH
        or output["height"] > MAX_OUTPUT_HEIGHT
        or output["width"] * output["height"] > MAX_OUTPUT_PIXELS
        or output["fps"] <= 0
        or output["fps"] > MAX_OUTPUT_FPS
    ):
        raise ValueError("receipt output geometry is invalid")
    if payload["crop"] is not None:
        box = payload["crop"]["box"]
        if output["width"] != box[2] - box[0] or output["height"] != box[3] - box[1]:
            raise ValueError("receipt crop provenance disagrees with output geometry")
    if (
        isinstance(payload["source"].get("source_frames"), bool)
        or not isinstance(payload["source"].get("source_frames"), int)
        or payload["source"]["source_frames"] < 1
        or payload["source"]["source_frames"] > MAX_SOURCE_FRAMES
    ):
        raise ValueError("receipt source frame count is invalid")
    frame_digests = payload["source"].get("frame_sha256")
    if (
        not isinstance(frame_digests, list)
        or len(frame_digests) != payload["source"]["source_frames"]
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in frame_digests
        )
    ):
        raise ValueError("receipt frame digests are invalid")
    if (
        not _finite_number(payload["source"].get("source_fps"))
        or payload["source"]["source_fps"] <= 0
        or payload["source"]["source_fps"] > MAX_SOURCE_FPS
    ):
        raise ValueError("receipt source fps is invalid")
    if (
        not _finite_number(payload["source"].get("source_duration_s"))
        or payload["source"]["source_duration_s"] < 0
        or payload["source"]["source_duration_s"] > MAX_SOURCE_DURATION_S
    ):
        raise ValueError("receipt source duration is invalid")
    if not _finite_number(payload["presentation_duration_s"]):
        raise ValueError("receipt duration is invalid")


def _artifact_records(out_dir: Path) -> tuple[dict[str, Any], ...]:
    """Digest the three published files into result artifact records.

    Returns:
        Artifact records with stable identities and byte digests.
    """

    records = []
    for artifact_id, filename in (
        ("review-edit", OUTPUT_VIDEO_FILENAME),
        ("encode-receipt", OUTPUT_RECEIPT_FILENAME),
        ("presentation-time-map", OUTPUT_TIMEMAP_FILENAME),
    ):
        digest, _size = _sha256_file(
            out_dir / filename,
            max_bytes=MAX_OUTPUT_FILE_BYTES,
        )
        records.append(
            {
                "artifact_id": artifact_id,
                "uri": filename,
                "sha256": digest,
            }
        )
    return tuple(records)


def _validate_published_artifacts(out_dir: Path, artifacts: tuple[dict[str, Any], ...]) -> None:
    """Validate artifact identity, containment, regular-file type, and digests."""

    expected = {
        "review-edit": OUTPUT_VIDEO_FILENAME,
        "encode-receipt": OUTPUT_RECEIPT_FILENAME,
        "presentation-time-map": OUTPUT_TIMEMAP_FILENAME,
    }
    seen: set[str] = set()
    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id")
        uri = artifact.get("uri")
        expected_digest = artifact.get("sha256")
        if artifact_id in seen or artifact_id not in expected:
            raise ValueError("published artifact identity is invalid")
        seen.add(artifact_id)
        if uri != expected[artifact_id] or not isinstance(uri, str) or not _safe_relative(uri):
            raise ValueError("published artifact URI is invalid")
        path = out_dir / uri
        if path.is_symlink() or not path.is_file():
            raise ValueError("published artifact is not a regular file")
        if not isinstance(expected_digest, str) or _SHA256_RE.fullmatch(expected_digest) is None:
            raise ValueError("published artifact digest is invalid")
        observed, _size = _sha256_file(path, max_bytes=MAX_OUTPUT_FILE_BYTES)
        if observed != expected_digest.lower():
            raise ValueError("published artifact digest mismatch")
    if seen != set(expected):
        raise ValueError("published artifact set is incomplete")


def _try_renameat2_noreplace(staging: Path, destination: Path) -> bool:
    """Try Linux's atomic directory move and report whether it was used.

    Returns:
        True when renameat2 published the directory, otherwise False.
    """

    if os.name != "posix":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
    except OSError:
        renameat2 = None
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise _OutputCollisionError
    if error_number not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise OSError(error_number, os.strerror(error_number))
    return False


def _fallback_publish_noreplace(staging: Path, destination: Path) -> None:
    """Publish staged regular files without any replace-capable operation."""

    # A plain rename is replace-capable on POSIX.  Claim the destination with
    # mkdir and hard-link each private staging file instead, so the fallback
    # never overwrites a racing destination.  Linux takes the atomic directory
    # rename path above; this branch is a conservative no-replace fallback.
    try:
        os.mkdir(destination)
    except FileExistsError as error:
        raise _OutputCollisionError from error
    linked: list[Path] = []
    try:
        for child in sorted(staging.iterdir(), key=lambda path: path.name):
            if child.is_symlink() or not child.is_file():
                raise OSError("staging contains a non-regular file")
            target = destination / child.name
            os.link(child, target, follow_symlinks=False)
            linked.append(target)
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
    except Exception:
        for target in reversed(linked):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


def _atomic_publish_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""

    if _try_renameat2_noreplace(staging, destination):
        return
    _fallback_publish_noreplace(staging, destination)


def _build_receipt(
    prepared: _Prepared,
    plan: _Planned,
    encoder_record: dict[str, Any],
    video_sha256: str,
    time_map_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic encode-receipt.v1 payload.

    Returns:
        Strict-JSON receipt payload.
    """

    source = prepared.source
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "encoder": encoder_record,
        "source": {
            **_source_reference_identity(source.source_ref),
            "sha256": source.observed_sha256,
            "declared_sha256": source.source_ref.sha256.lower(),
            "source_frames": len(source.frames),
            "source_fps": source.source_fps,
            "source_duration_s": plan.source_duration_s,
            "frame_sha256": list(source.frame_digests),
            "integrity": "verified",
        },
        "config_sha256": _canonical_digest(prepared.config),
        "edit_plan_digest": _canonical_digest(prepared.config.get("edits", [])),
        "output": {
            "video": OUTPUT_VIDEO_FILENAME,
            "video_sha256": video_sha256,
            "time_map": OUTPUT_TIMEMAP_FILENAME,
            "time_map_sha256": time_map_sha256,
            "frames": len(plan.frames),
            "width": (plan.crop_box[2] - plan.crop_box[0] if plan.crop_box else prepared.width),
            "height": (plan.crop_box[3] - plan.crop_box[1] if plan.crop_box else prepared.height),
            "fps": prepared.out_fps,
        },
        "crop": _crop_provenance(plan.crop_box),
        "presentation_duration_s": len(plan.frames) / prepared.out_fps,
        "audio_policy": AUDIO_POLICY_SILENT,
        "evidence_status": "diagnostic-only",
        "evidence_boundary": "diagnostic-only; not benchmark evidence",
        "benchmark_success": False,
        "scientific_claim_allowed": False,
    }


def _publish_outputs(
    request: ComponentRequest,
    prepared: _Prepared,
    plan: _Planned,
) -> ComponentResult:
    """Encode into a private staging directory and publish atomically.

    Returns:
        Complete result on publication, or a failed result without references.
    """

    staging: Path | None = None
    try:
        _ensure_output_parent(prepared.root, prepared.out_dir)
        if os.path.lexists(prepared.out_dir):
            raise _OutputCollisionError
        staging = Path(
            tempfile.mkdtemp(prefix=f".{prepared.out_dir.name}-", dir=prepared.out_dir.parent)
        )
        frames = [
            _render_source_frame(
                prepared.source.frames[record.source_index],
                prepared.width,
                prepared.height,
                plan.crop_box,
            )
            for record in plan.frames
        ]
        video_path = staging / OUTPUT_VIDEO_FILENAME
        encode_error, encoder_record = _encode_mp4(
            frames,
            video_path,
            prepared.out_fps,
            prepared.env,
        )
        if encode_error is not None:
            return _failure(request, encode_error)
        decoded_count, decode_error = _decode_frame_count(video_path)
        if decode_error is not None or decoded_count != len(frames):
            return _failure(
                request,
                f"encoder_verification_failed: {decode_error or f'decoded {decoded_count}, wrote {len(frames)}'}",
            )
        video_digest, _video_size = _sha256_file(
            video_path,
            max_bytes=MAX_OUTPUT_FILE_BYTES,
        )
        time_map = _build_time_map(prepared, plan)
        time_map_digest = _write_json(staging / OUTPUT_TIMEMAP_FILENAME, time_map)
        receipt = _build_receipt(
            prepared,
            plan,
            encoder_record,
            video_digest,
            time_map_digest,
        )
        _validate_receipt(receipt)
        _write_json(staging / OUTPUT_RECEIPT_FILENAME, receipt)
        artifacts = _artifact_records(staging)
        _validate_published_artifacts(staging, artifacts)
        result = _build_result(
            request,
            STATUS_COMPLETE,
            "encoded",
            artifacts=artifacts,
            provenance={
                "source_kind": prepared.source.source_kind,
                "source_sha256": prepared.source.observed_sha256,
                "source_identity": _source_reference_identity(prepared.source.source_ref),
                "config_sha256": _canonical_digest(prepared.config),
                "presentation_frames": len(plan.frames),
                "artifact_digests": {
                    artifact["artifact_id"]: artifact["sha256"] for artifact in artifacts
                },
            },
        )
        # Validate the final JSON envelope before any visible publication.
        result_payload(result)
        _atomic_publish_noreplace(staging, prepared.out_dir)
        staging = None
        return result
    except _OutputCollisionError:
        return _failure(request, "output_collision: output directory is already reserved")
    except _OperationTimeout:
        return _failure(request, "encoder_timeout")
    except (OSError, ValueError, RuntimeError) as error:
        return _failure(request, f"output_write_failed: {type(error).__name__}")
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _fallback_request(request: Any) -> ComponentRequest:
    """Create a valid result identity when the API argument is not a request.

    Returns:
        Minimal request suitable for a stable failure envelope.
    """

    if isinstance(request, ComponentRequest):
        return request
    return ComponentRequest(
        request_id="invalid-request",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory="invalid",
    )


def _run_admission(
    request: ComponentRequest, root: Path
) -> tuple[Path | None, ComponentResult | None]:
    """Validate request identity, capabilities, output path, and collision state.

    Returns:
        Safe output directory and no result, or an early terminal result.
    """

    integrity_errors = _request_integrity_errors(request)
    if integrity_errors:
        return None, _failure(request, "; ".join(integrity_errors))
    if request.component_id != COMPONENT_ID:
        return None, _unavailable(request, f"unsupported_component: {request.component_id}")
    capability_error = _check_capabilities(request)
    if capability_error is not None:
        return None, _unavailable(request, capability_error)
    out_dir, path_error = _resolve_output_directory(root, request.output_directory)
    if out_dir is None or path_error is not None:
        return None, _failure(request, path_error or "unsafe_output_path")
    if os.path.lexists(out_dir):
        return None, _failure(request, "output_collision: output directory already exists")
    return out_dir, None


def run(request: ComponentRequest, base: Path | str = Path(".")) -> ComponentResult:
    """Execute one bounded source-backed review encode request.

    Returns:
        Complete, partial, unavailable, or failed component result.
    """

    safe_request = _fallback_request(request)
    root, root_error = _resolve_root(base)
    if root is None or root_error is not None:
        return _failure(safe_request, root_error or "unsafe_output_path: invalid base")
    if not isinstance(request, ComponentRequest):
        return _failure(safe_request, "request_invalid: expected ComponentRequest")
    out_dir, admission_result = _run_admission(request, root)
    if out_dir is None or admission_result is not None:
        return admission_result or _failure(request, "request_admission_failed")
    prepared, early = _prepare(request, root, out_dir)
    if prepared is None or early is not None:
        return early if early is not None else _failure(request, "prepare_failed")
    plan, plan_error = _plan_order(prepared)
    if plan is None or plan_error is not None:
        return _failure(request, plan_error or "plan_failed")
    if plan.diagnostics:
        diagnostics = sorted(set(plan.diagnostics))
        return _build_result(
            request,
            STATUS_PARTIAL,
            "; ".join(diagnostics[:8]),
            diagnostics=tuple({"code": diagnostic} for diagnostic in diagnostics),
            provenance={"source_kind": prepared.source.source_kind},
        )
    return _publish_outputs(request, prepared, plan)


def _load_request(input_path: Path) -> ComponentRequest:
    """Load a component-request.v1 document from disk.

    Returns:
        Validated component request.
    """

    try:
        if input_path.stat().st_size > MAX_REQUEST_BYTES:
            raise ReviewContractsValidationError(
                [f"request exceeds {MAX_REQUEST_BYTES} bytes"], source=input_path
            )
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError(
            [f"cannot read request: {type(error).__name__}"],
            source=input_path,
        ) from error
    if not isinstance(payload, dict):
        raise ReviewContractsValidationError(["request must be an object"], source=input_path)
    return component_request_from_dict(payload, source=str(input_path))


def _load_config(config_path: Path | None) -> dict[str, Any]:
    """Load a JSON config override object.

    Returns:
        Parsed config mapping.
    """

    if config_path is None:
        return {}
    try:
        if config_path.stat().st_size > MAX_CONFIG_BYTES:
            raise ReviewContractsValidationError(
                [f"config exceeds {MAX_CONFIG_BYTES} bytes"], source=config_path
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError(
            [f"cannot read config: {type(error).__name__}"],
            source=config_path,
        ) from error
    if not isinstance(payload, dict):
        raise ReviewContractsValidationError(["config must be an object"], source=config_path)
    return payload


def _cli_output_directory(value: str, base: Path) -> str:
    """Convert an absolute CLI output into a safe path relative to base.

    Returns:
        Relative output path suitable for component-request.v1.
    """

    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        relative = candidate.resolve(strict=False).relative_to(base.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ReviewContractsValidationError(
            ["output must resolve below --base; absolute output escapes the output root"]
        ) from error
    return relative.as_posix()


def _cli_failure(reason: str) -> dict[str, Any]:
    """Build a schema-valid result for malformed CLI input without a request.

    Returns:
        component-result.v1 failure payload.
    """

    result = ComponentResult(
        request_id="invalid-request",
        component_id=COMPONENT_ID,
        status=STATUS_FAILED,
        provenance={
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "output_directory": "<invalid>",
            "evidence_status": "diagnostic-only",
            "evidence_boundary": "diagnostic-only; not benchmark evidence",
            "benchmark_success": False,
            "scientific_claim_allowed": False,
            "admission": "not_evaluated",
        },
        reason=reason,
    )
    return result_payload(result)


def main(argv: list[str] | None = None) -> int:
    """Run the standalone CLI and return zero only for complete output.

    Returns:
        Zero for complete output, one for a handled non-complete result, or two
        for invalid CLI input.
    """

    parser = argparse.ArgumentParser(
        description="Encode bounded source-backed review edits from a frame sequence or clip."
    )
    parser.add_argument("--input", required=True, help="component-request.v1 JSON path")
    parser.add_argument("--config", help="optional JSON config override path")
    parser.add_argument("--output", required=True, help="new output directory below --base")
    parser.add_argument("--base", default=".", help="source and output root")
    args = parser.parse_args(argv)
    try:
        base = Path(args.base).resolve(strict=True)
        request = _load_request(Path(args.input))
        overrides = _load_config(Path(args.config) if args.config else None)
        merged_config = {**request.config, **overrides}
        payload = {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": request.request_id,
            "component_id": request.component_id,
            "sources": [
                {key: value for key, value in asdict(source).items() if value}
                for source in request.sources
            ],
            "required_capabilities": list(request.required_capabilities),
            "config": merged_config,
            "output_directory": _cli_output_directory(args.output, base),
        }
        request = component_request_from_dict(payload, source=str(args.input))
        result = run(request, base=base)
        print(json.dumps(result_payload(result), sort_keys=True))  # noqa: T201 - CLI output
        return 0 if result.status == STATUS_COMPLETE else 1
    except (OSError, ValueError, ReviewContractsValidationError) as error:
        print(json.dumps(_cli_failure(f"request_invalid: {error}"), sort_keys=True))  # noqa: T201 - CLI output
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
