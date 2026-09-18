"""Offline planner and pedestrian diagnostic panels (SREV-18, issue #9288).

This component is a read-only consumer of the two trace inventories already
owned by Robot SF: ``simulation_timeline.v1`` and ``analysis-trace.v1`` (the
``simulation_trace_export.v1`` source envelope is accepted as their canonical
producer form).  It deliberately does not instrument a planner, infer private
observations, execute a simulator, or calculate a benchmark metric.  Every
displayed value is either copied from a recorded row or is a plainly labelled
post-hoc subtraction of two recorded control values.

The Python and browser surfaces share the SREV-16 source-time cursor and
selection revision.  Source identities, units, coordinate frames, actor
selection, and missing reasons stay in the emitted model and evidence-reference
export so SREV-17 and the future BA-05 service can consume references without
silently rebinding them.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import resources
from itertools import pairwise
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
from robot_sf.analysis_workbench.simulation_timeline import validate_simulation_timeline
from robot_sf.analysis_workbench.simulation_trace_export import simulation_trace_export_from_dict
from robot_sf.benchmark.analysis_trace import trace_coverage
from robot_sf.benchmark.failure_diagnosis import validate_failure_diagnosis_payload
from robot_sf.render.review_panels import ReviewContext, SourceTimeCursor

COMPONENT_ID = "srev18-review-diagnostics"
COMPONENT_VERSION = "1.0.0"
DIAGNOSTICS_MODEL_SCHEMA_VERSION = "review-diagnostics.v1"
EVIDENCE_REFERENCE_SCHEMA_VERSION = "evidence-reference-export.v1"
MISSING_CAPABILITY_SCHEMA_VERSION = "missing-capability-report.v1"

SIMULATION_TIMELINE_SCHEMA_VERSION = "simulation_timeline.v1"
SIMULATION_TRACE_EXPORT_SCHEMA_VERSION = "simulation_trace_export.v1"
ANALYSIS_TRACE_RECORD_SCHEMA_VERSION = "analysis-trace.v1"
FAILURE_DIAGNOSIS_SCHEMA_VERSION = "failure_diagnosis.v1"

STATUS_AVAILABLE = "available"
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

VALUE_ORIGINS = (
    "planner_visible",
    "simulator_ground_truth",
    "post_hoc",
    "unavailable",
)
INVENTORY_KINDS = {
    SIMULATION_TIMELINE_SCHEMA_VERSION: "simulation_timeline",
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION: "simulation_trace_export",
    ANALYSIS_TRACE_RECORD_SCHEMA_VERSION: "analysis_trace",
}
TRACE_FORMATS = frozenset(
    {
        "simulation-timeline",
        "simulation-timeline.v1",
        "simulation_timeline.v1",
        "simulation-trace",
        "simulation_trace_export.v1",
        "simulation-trace-export.v1",
        "analysis-trace",
        "analysis-trace.v1",
        "analysis_trace.v1",
    }
)
DIAGNOSIS_FORMATS = frozenset(
    {"failure-diagnosis", "failure-diagnosis.v1", FAILURE_DIAGNOSIS_SCHEMA_VERSION}
)
SUPPORTED_CAPABILITIES = frozenset(
    {
        "source-time-cursor",
        "diagnostic-panels",
        "planner-diagnostics",
        "pedestrian-diagnostics",
        "control-comparison",
        "evidence-references",
        "offline-browser",
        "recorded-trace",
        "failure-diagnosis",
    }
)
REQUIRED_CAPABILITIES = ("source-time-cursor",)
OPTIONAL_CAPABILITIES = (
    "planner-diagnostics",
    "pedestrian-diagnostics",
    "control-comparison",
    "failure-diagnosis",
    "evidence-references",
    "offline-browser",
)

OUTPUT_MODEL_FILENAME = f"{DIAGNOSTICS_MODEL_SCHEMA_VERSION}.json"
OUTPUT_HTML_FILENAME = f"{DIAGNOSTICS_MODEL_SCHEMA_VERSION}.html"
OUTPUT_REFERENCE_FILENAME = f"{EVIDENCE_REFERENCE_SCHEMA_VERSION}.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SOURCES = 32
MAX_FRAMES = 100_000
MAX_RECORDS = 100_000
MAX_CANDIDATES = 10_000
MAX_PED_DIAGNOSTICS = 512
MAX_IDENTITY_LENGTH = 512
MAX_REASON_LENGTH = 200

_DESCRIPTOR_DOCUMENT = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": [
        DIAGNOSTICS_MODEL_SCHEMA_VERSION,
        EVIDENCE_REFERENCE_SCHEMA_VERSION,
        MISSING_CAPABILITY_SCHEMA_VERSION,
    ],
}
component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)
DESCRIPTOR: ComponentDescriptor = component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)


@dataclass(frozen=True, slots=True)
class DiagnosticSample:
    """One normalized source-time trace row."""

    time_s: float
    source_index: int
    step: int | None
    planner: dict[str, Any]
    robot: dict[str, Any]
    pedestrians: list[dict[str, Any]]
    controls: dict[str, Any]
    pointer: str


@dataclass
class _LoadedSource:
    ref: SourceRef
    payload: Any | None
    digest: str | None
    integrity: str
    status: str
    reason: str
    kind: str
    schema_version: str
    identity: dict[str, Any] = field(default_factory=dict)
    units: Any = None
    coordinate_frame: str | None = None
    samples: list[DiagnosticSample] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    source_error: str | None = None


@dataclass
class _DirectoryHandle:
    """Descriptor-rooted directory identity used for all source/output I/O."""

    path: Path
    fd: int

    def close(self) -> None:
        """Close the owned descriptor once, ignoring already-closed handles."""

        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class _InputError(ValueError):
    """Bounded component input error with a stable leading reason code."""


def _reason(value: Any, default: str = "unavailable") -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return (text or default)[:MAX_REASON_LENGTH]


def _finite(value: Any, label: str = "value") -> float:
    if isinstance(value, bool):
        raise _InputError(f"invalid_input: {label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _InputError(f"invalid_input: {label} must be finite numeric") from error
    if not math.isfinite(number):
        raise _InputError(f"invalid_input: {label} must be finite numeric")
    return number


def _optional_finite(value: Any) -> float | None:
    try:
        return _finite(value)
    except _InputError:
        return None


def _strict_loads(raw: bytes) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise _InputError("resource_limit: source JSON is too large")

    def parse_finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number: {value}")
        return number

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_float=parse_finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise _InputError("source_corrupt: source is not strict UTF-8 JSON") from error


def _unsafe_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return True
    candidate = Path(value)
    windows = PureWindowsPath(value)
    return (
        candidate.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in candidate.parts
        or ".." in windows.parts
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _path_parts(value: str, *, kind: str = "source") -> tuple[str, ...]:
    if _unsafe_path(value):
        raise _InputError(f"unsafe_{kind}_path: absolute, traversal, or control path rejected")
    parts = tuple(part for part in Path(value).parts if part not in {"", "."})
    if not parts:
        raise _InputError(f"unsafe_{kind}_path: empty relative path")
    return parts


def _absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _open_directory(
    root: Path,
    *,
    error_prefix: str = "source",
    missing_reason: str | None = None,
) -> _DirectoryHandle:
    """Open a directory by components, retaining its identity across renames.

    Returns:
        An owned descriptor handle for the directory.
    """

    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
    ):
        raise _InputError("source_protection_unavailable")
    path = _absolute_path(root)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    descriptor: int | None = None
    succeeded = False
    try:
        descriptor = os.open(os.sep, flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        succeeded = True
        return _DirectoryHandle(path, descriptor)
    except FileNotFoundError as error:
        raise _InputError(missing_reason or f"{error_prefix}_root_missing") from error
    except NotADirectoryError as error:
        raise _InputError(f"{error_prefix}_root_not_directory") from error
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise _InputError(f"unsafe_{error_prefix}_path: symlink component rejected") from error
        raise _InputError(f"{error_prefix}_root_unreadable") from error
    finally:
        if descriptor is not None and not succeeded:
            os.close(descriptor)


def _dup_directory_fd(root: Path, root_fd: int | None) -> tuple[int, Path]:
    """Duplicate a caller-owned root descriptor, or open one safely.

    Returns:
        A duplicated descriptor and its lexical display path.
    """

    if root_fd is None:
        handle = _open_directory(root)
        try:
            return os.dup(handle.fd), handle.path
        finally:
            handle.close()
    try:
        metadata = os.fstat(root_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise _InputError("source_root_not_directory")
        return os.dup(root_fd), _absolute_path(root)
    except _InputError:
        raise
    except OSError as error:
        raise _InputError("source_root_unreadable") from error


def _read_under(  # noqa: C901
    root: Path, uri: str, *, root_fd: int | None = None
) -> bytes:
    """Read a regular relative file through a retained descriptor root.

    Returns:
        The exact bytes read from the protected regular file.
    """

    parts = _path_parts(uri)
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise _InputError("source_protection_unavailable")
    descriptor, _root_path = _dup_directory_fd(root, root_fd)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_flags = flags | os.O_DIRECTORY
    source_fd: int | None = None
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        source_fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=descriptor)
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise _InputError("source_not_regular_file")
        if metadata.st_size > MAX_SOURCE_BYTES:
            raise _InputError("resource_limit: source bytes")
        with os.fdopen(source_fd, "rb") as handle:
            source_fd = None
            payload = handle.read(MAX_SOURCE_BYTES + 1)
        if len(payload) > MAX_SOURCE_BYTES:
            raise _InputError("resource_limit: source bytes")
        return payload
    except _InputError:
        raise
    except FileNotFoundError as error:
        raise _InputError("source_missing") from error
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise _InputError("unsafe_source_path: symlink component rejected") from error
        raise _InputError("source_unreadable") from error
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(descriptor)


def _reserve_output(root: Path, value: str, *, root_fd: int | None = None) -> _DirectoryHandle:
    """Reserve a fresh output directory and retain its descriptor identity.

    Returns:
        An owned descriptor handle for the new output directory.
    """

    parts = _path_parts(value, kind="output")
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
    ):
        raise _InputError("source_protection_unavailable")
    descriptor, root_path = _dup_directory_fd(root, root_fd)
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    try:
        for part in parts[:-1]:
            try:
                next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        os.mkdir(parts[-1], mode=0o700, dir_fd=descriptor)
        output_fd = os.open(parts[-1], directory_flags, dir_fd=descriptor)
        return _DirectoryHandle(root_path.joinpath(*parts), output_fd)
    except FileExistsError as error:
        raise _InputError("output_collision: output directory already exists") from error
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise _InputError("unsafe_output_path: symlink component rejected") from error
        raise _InputError("unsafe_output_path: output directory cannot be protected") from error
    finally:
        os.close(descriptor)


def _ensure_directory(root: _DirectoryHandle, parts: Sequence[str]) -> _DirectoryHandle:
    """Open/create fixed output subdirectories from a retained descriptor.

    Returns:
        An owned descriptor handle for the requested subdirectory.
    """

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise _InputError("source_protection_unavailable")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    descriptor = os.dup(root.fd)
    path = root.path
    try:
        for part in parts:
            if not part or part in {".", ".."} or Path(part).name != part:
                raise _InputError("unsafe_output_path: invalid output component")
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            path = path / part
        return _DirectoryHandle(path, descriptor)
    except _InputError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        if error.errno == errno.ELOOP:
            raise _InputError("unsafe_output_path: symlink component rejected") from error
        raise _InputError("unsafe_output_path: output subdirectory cannot be protected") from error


def _write_text(directory: _DirectoryHandle, name: str, text: str) -> str:
    payload = text.encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise _InputError("resource_limit: output bytes")
    name_parts = _path_parts(name, kind="output")
    if len(name_parts) != 1 or name_parts[0] != name:
        raise _InputError("unsafe_output_path: artifact name must be one relative component")
    temporary = f".{name}.tmp"
    if (
        not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.link not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or os.link not in os.supports_follow_symlinks
    ):
        raise _InputError("source_protection_unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=directory.fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory.fd,
                dst_dir_fd=directory.fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise _InputError("output_collision: artifact already exists") from error
        return hashlib.sha256(payload).hexdigest()
    except OSError as error:
        if isinstance(error, _InputError):
            raise
        raise _InputError("output_unwritable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory.fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _write_json(directory: _DirectoryHandle, name: str, payload: Any) -> str:
    try:
        text = (
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise _InputError("invalid_output: model is not strict JSON") from error
    return _write_text(directory, name, text)


def _copy_web_component(output_dir: _DirectoryHandle) -> str:
    destination = _ensure_directory(output_dir, ("components", "review_diagnostics"))
    asset = resources.files("robot_sf.render.web_assets").joinpath(
        "components", "review_diagnostics", "review_diagnostics.js"
    )
    try:
        with resources.as_file(asset) as asset_path:
            return _write_text(
                destination,
                "review_diagnostics.js",
                asset_path.read_text(encoding="utf-8"),
            )
    finally:
        destination.close()


def _format_token(ref: SourceRef, payload: Any = None) -> str:
    value = (ref.schema or ref.format or "").strip().lower()
    payload_schema = payload.get("schema_version") if isinstance(payload, Mapping) else None
    return str(payload_schema or value).strip().lower()


def _source_kind(ref: SourceRef, payload: Any) -> tuple[str, str]:
    token = _format_token(ref, payload)
    payload_schema = (
        str(payload.get("schema_version", "")).strip().lower()
        if isinstance(payload, Mapping)
        else ""
    )
    declared = (ref.schema or ref.format or "").strip().lower()
    if declared and declared not in {payload_schema, "json"}:
        # A schema declaration is authoritative.  A generic JSON format may
        # accompany a payload-owned schema, but a declared trace family must
        # not silently bind a different inventory from the one requested.
        declared_aliases = {
            "simulation_timeline.v1": {
                "simulation_timeline.v1",
                "simulation-timeline.v1",
                "simulation-timeline",
            },
            "simulation_trace_export.v1": {
                "simulation_trace_export.v1",
                "simulation-trace-export.v1",
                "simulation-trace",
            },
            "analysis-trace.v1": {"analysis-trace.v1", "analysis_trace.v1", "analysis-trace"},
            "failure_diagnosis.v1": {
                "failure_diagnosis.v1",
                "failure-diagnosis.v1",
                "failure-diagnosis",
            },
        }
        matches = next(
            (aliases for schema, aliases in declared_aliases.items() if schema == payload_schema),
            set(),
        )
        if declared not in matches:
            raise _InputError(
                f"incompatible_schema: declared {declared} does not match {payload_schema or token}"
            )
    if token in {"simulation_timeline.v1", "simulation-timeline.v1"}:
        return "simulation_timeline", SIMULATION_TIMELINE_SCHEMA_VERSION
    if token in {"simulation_trace_export.v1", "simulation-trace-export.v1"}:
        return "simulation_trace_export", SIMULATION_TRACE_EXPORT_SCHEMA_VERSION
    if token in {"analysis-trace.v1", "analysis_trace.v1"}:
        return "analysis_trace", ANALYSIS_TRACE_RECORD_SCHEMA_VERSION
    if token in {"failure_diagnosis.v1", "failure-diagnosis.v1"}:
        return "failure_diagnosis", FAILURE_DIAGNOSIS_SCHEMA_VERSION
    # A format-only declaration is accepted when the payload itself identifies
    # one of the documented inventories.  Unknown formats remain unavailable.
    if payload_schema in INVENTORY_KINDS:
        return INVENTORY_KINDS[payload_schema], payload_schema
    if payload_schema in {FAILURE_DIAGNOSIS_SCHEMA_VERSION}:
        return "failure_diagnosis", FAILURE_DIAGNOSIS_SCHEMA_VERSION
    raise _InputError(f"incompatible_schema: unsupported trace schema {token or payload_schema}")


def _validate_source_payload(kind: str, payload: Mapping[str, Any]) -> None:
    """Validate an admitted inventory with its owning producer contract."""

    try:
        if kind == "simulation_timeline":
            validate_simulation_timeline(payload)
        elif kind == "simulation_trace_export":
            simulation_trace_export_from_dict(payload)
        elif kind == "analysis_trace":
            coverage = trace_coverage(
                {
                    "scenario_id": payload.get("scenario_id"),
                    "algo": payload.get("planner"),
                    "algorithm_metadata": {"analysis_trace": dict(payload)},
                }
            )
            if coverage.get("status") != "complete":
                reasons = ",".join(str(item) for item in coverage.get("reasons", ()))
                raise _InputError(
                    "source_schema_invalid: analysis trace coverage incomplete"
                    + (f" ({reasons})" if reasons else "")
                )
        elif kind == "failure_diagnosis":
            validate_failure_diagnosis_payload(payload)
    except (TypeError, ValueError) as error:
        if isinstance(error, _InputError):
            raise
        raise _InputError("source_schema_invalid: producer contract rejected source") from error


def _embedded_identity(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    for key in ("source_identity", "source"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    source_trace = payload.get("source_trace")
    if isinstance(source_trace, Mapping):
        source = source_trace.get("source")
        return source if isinstance(source, Mapping) else source_trace
    # ``analysis-trace.v1`` keeps provenance fields at the payload root.
    return payload


def _identity(ref: SourceRef, payload: Any, digest: str | None) -> dict[str, Any]:
    embedded = _embedded_identity(payload)
    result: dict[str, Any] = {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "format": ref.format,
        "schema": ref.schema
        or (str(payload.get("schema_version", "")) if isinstance(payload, Mapping) else ""),
        "declared_sha256": ref.sha256 or None,
        "sha256": digest,
        "source_commit": ref.source_commit or None,
        "config_identity": ref.config_identity or None,
        "units": ref.units or None,
        "coordinate_frame": ref.coordinate_frame or None,
    }
    aliases = {
        "campaign_id": ("campaign_id", "campaign"),
        "execution_id": ("execution_id", "run_id", "trace_id"),
        "episode_id": ("episode_id", "episode"),
        "scenario_id": ("scenario_id", "scenario"),
        "planner_id": ("planner_id", "planner"),
        "seed": ("seed",),
        "source_commit": ("source_commit", "planner_commit", "git_hash", "commit"),
        "config_identity": ("config_identity", "config_digest", "config_hash"),
        "coordinate_frame": ("coordinate_frame",),
        "units": ("units",),
    }
    for output_key, keys in aliases.items():
        if result.get(output_key) is not None:
            continue
        for key in keys:
            value = embedded.get(key)
            if (
                value is not None
                and isinstance(value, (str, int, float))
                and not isinstance(value, bool)
            ):
                result[output_key] = value
                break
    # Timeline identity is nested below source_trace.
    if isinstance(payload, Mapping) and isinstance(payload.get("source_trace"), Mapping):
        trace_id = payload["source_trace"].get("trace_id")
        if trace_id is not None:
            result.setdefault("trace_id", trace_id)
    return {key: value for key, value in result.items() if value is not None}


def _units_token(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return ""


def _units_compatible(declared: str, observed: Any) -> bool:
    if not declared:
        return True
    if observed is None:
        return False
    observed_token = _units_token(observed)
    if declared == observed_token:
        return True
    # Existing SREV fixture refs use this compact label for the canonical
    # seconds/metres/radians inventory, while trace payloads use a field map.
    if declared == "seconds/metres/radians" and isinstance(observed, Mapping):
        return (
            observed.get("position") == "m"
            and observed.get("velocity") == "m/s"
            and observed.get("heading") == "rad"
            and observed.get("time") == "s"
        )
    return False


def _coordinate_compatible(declared: str, observed: Any) -> bool:
    return not declared or (observed is not None and declared == str(observed))


def _source_status(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        return "unavailable", "source_shape_invalid"
    for key in ("execution_status", "row_status", "status"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"fallback", "degraded", "failed", "unavailable"}:
                return "unavailable", f"source_execution_{normalized}"
            if normalized in {"partial", "incomplete"}:
                return "partial", f"source_execution_{normalized}"
    return "available", ""


def _row_collection(payload: Mapping[str, Any], kind: str) -> tuple[list[Any], str]:
    if kind == "simulation_timeline":
        rows = payload.get("frames")
        return (rows if isinstance(rows, list) else [], "frames")
    if kind == "simulation_trace_export":
        rows = payload.get("frames")
        return (rows if isinstance(rows, list) else [], "frames")
    if kind == "analysis_trace":
        rows = payload.get("steps")
        return (rows if isinstance(rows, list) else [], "steps")
    return ([], "records")


def _planner_from_row(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "simulation_timeline":
        state = row.get("state")
        if isinstance(state, Mapping):
            planner = state.get("planner_decision", state.get("planner"))
            return dict(planner) if isinstance(planner, Mapping) else {}
    planner = row.get("planner", row.get("planner_decision"))
    return dict(planner) if isinstance(planner, Mapping) else {}


def _robot_from_row(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "simulation_timeline" and isinstance(row.get("state"), Mapping):
        value = row["state"].get("robot")
    else:
        value = row.get("robot")
    return dict(value) if isinstance(value, Mapping) else {}


def _pedestrians_from_row(row: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "simulation_timeline" and isinstance(row.get("state"), Mapping):
        value = row["state"].get("pedestrians")
    else:
        value = row.get("pedestrians")
    return (
        [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, list)
        else []
    )


def _controls_from_row(
    row: Mapping[str, Any], planner: Mapping[str, Any], kind: str
) -> dict[str, Any]:
    value: Any = row.get("controls")
    if kind == "simulation_timeline" and isinstance(row.get("state"), Mapping):
        value = row["state"].get("controls", value)
    if isinstance(value, Mapping):
        return dict(value)
    # ``analysis_trace`` preserves the AMV requested/applied pair in controls;
    # simulation timeline currently retains the selected planner action only.
    amv = planner.get("amv")
    if isinstance(amv, Mapping):
        requested = {
            key: deepcopy(amv[key])
            for key in (
                "requested_linear_m_s",
                "requested_angular_rad_s",
                "requested_turn_rate_rad_s",
                "units",
            )
            if key in amv
        }
        applied = {
            key: deepcopy(amv[key])
            for key in (
                "applied_linear_m_s",
                "applied_angular_rad_s",
                "applied_turn_rate_rad_s",
                "units",
            )
            if key in amv
        }
        controls: dict[str, Any] = {}
        if requested:
            controls["requested"] = requested
        if applied:
            controls["applied"] = applied
        return controls
    return {}


def _normalize_frames(source: _LoadedSource) -> None:
    if not isinstance(source.payload, Mapping):
        source.status, source.reason = "unavailable", "source_shape_invalid"
        return
    rows, row_name = _row_collection(source.payload, source.kind)
    if not rows:
        source.status, source.reason = "unavailable", f"{row_name}_not_recorded"
        return
    if len(rows) > MAX_FRAMES:
        source.status, source.reason = "unavailable", "resource_limit: frame count"
        return
    samples: list[DiagnosticSample] = []
    previous_time: float | None = None
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            source.diagnostics.append({"reason_code": "row_not_mapping", "source_index": index})
            continue
        try:
            time_s = _finite(raw.get("time_s"), f"{row_name}[{index}].time_s")
        except _InputError:
            source.diagnostics.append({"reason_code": "source_time_missing", "source_index": index})
            continue
        if previous_time is not None and time_s < previous_time:
            source.status, source.reason = "unavailable", "source_time_not_nondecreasing"
            source.diagnostics.append(
                {"reason_code": "source_time_decreased", "source_index": index}
            )
            source.samples = []
            return
        previous_time = time_s
        raw_step = raw.get("step", raw.get("frame_index"))
        step = raw_step if isinstance(raw_step, int) and not isinstance(raw_step, bool) else None
        planner = _planner_from_row(raw, source.kind)
        robot = _robot_from_row(raw, source.kind)
        pedestrians = _pedestrians_from_row(raw, source.kind)
        controls = _controls_from_row(raw, planner, source.kind)
        pointer = f"/{row_name}/{index}"
        samples.append(
            DiagnosticSample(time_s, index, step, planner, robot, pedestrians, controls, pointer)
        )
    if not samples:
        source.status, source.reason = "unavailable", "source_time_unavailable"
        return
    source.samples = samples
    if source.diagnostics:
        source.status, source.reason = "partial", "source_rows_skipped"


def _load_source(  # noqa: C901
    ref: SourceRef, root: Path, *, root_fd: int | None = None
) -> _LoadedSource:
    try:
        raw = _read_under(root, ref.uri, root_fd=root_fd)
        digest = hashlib.sha256(raw).hexdigest()
        integrity = (
            "unbound"
            if not ref.sha256
            else ("match" if digest.lower() == ref.sha256.lower() else "mismatch")
        )
        if integrity == "mismatch":
            return _LoadedSource(
                ref,
                None,
                digest,
                integrity,
                STATUS_UNAVAILABLE,
                "source_integrity_mismatch",
                "",
                "",
                source_error="source_integrity_mismatch",
            )
        payload = _strict_loads(raw)
        kind, schema = _source_kind(ref, payload)
        if isinstance(payload, Mapping):
            _validate_source_payload(kind, payload)
    except _InputError as error:
        reason = _reason(error)
        failed_reasons = (
            "source_corrupt",
            "source_schema_invalid",
            "source_unreadable",
            "source_protection_unavailable",
            "unsafe_source_path",
            "incompatible_schema",
        )
        return _LoadedSource(
            ref,
            None,
            digest if isinstance(locals().get("digest"), str) else None,
            "unverifiable",
            STATUS_FAILED if reason.startswith(failed_reasons) else STATUS_UNAVAILABLE,
            reason,
            "",
            "",
            source_error=reason,
        )
    except (TypeError, ValueError) as error:
        reason = _reason(error, "incompatible_schema")
        return _LoadedSource(
            ref,
            None,
            digest if "digest" in locals() else None,
            "unverifiable",
            STATUS_UNAVAILABLE,
            reason,
            "",
            "",
            source_error=reason,
        )

    status, source_reason = _source_status(payload)
    source = _LoadedSource(
        ref,
        payload,
        digest,
        integrity,
        status,
        source_reason,
        kind,
        schema,
        identity=_identity(ref, payload, digest),
        units=payload.get("units") if isinstance(payload, Mapping) else None,
        coordinate_frame=(
            str(payload.get("coordinate_frame"))
            if isinstance(payload, Mapping) and payload.get("coordinate_frame") is not None
            else None
        ),
    )
    if integrity == "unbound":
        source.status, source.reason = STATUS_PARTIAL, "source_integrity_unbound"
        source.source_error = source.reason
    if not _units_compatible(ref.units, source.units):
        source.status, source.reason = STATUS_UNAVAILABLE, "unit_source_mismatch"
        source.source_error = source.reason
        return source
    if not _coordinate_compatible(ref.coordinate_frame, source.coordinate_frame):
        source.status, source.reason = STATUS_UNAVAILABLE, "coordinate_frame_source_mismatch"
        source.source_error = source.reason
        return source
    for ref_key, identity_key in (
        ("source_commit", "source_commit"),
        ("config_identity", "config_identity"),
    ):
        declared = getattr(ref, ref_key, "")
        embedded = _embedded_identity(payload)
        observed = next(
            (
                embedded.get(alias)
                for alias in {
                    "source_commit": ("source_commit", "planner_commit", "git_hash", "commit"),
                    "config_identity": ("config_identity", "config_digest", "config_hash"),
                }[identity_key]
                if embedded.get(alias) is not None
            ),
            None,
        )
        if declared and observed is None:
            source.status, source.reason = STATUS_UNAVAILABLE, f"{identity_key}_source_missing"
            source.source_error = source.reason
            return source
        if declared and str(declared) != str(observed):
            source.status, source.reason = STATUS_UNAVAILABLE, f"{identity_key}_source_mismatch"
            source.source_error = source.reason
            return source
    if source.kind != "failure_diagnosis":
        _normalize_frames(source)
    elif not isinstance(payload.get("records"), list):
        source.status, source.reason = STATUS_UNAVAILABLE, "diagnosis_records_not_recorded"
    return source


def _load_sources(
    request: ComponentRequest, root: Path, *, root_fd: int | None = None
) -> list[_LoadedSource]:
    if len(request.sources) > MAX_SOURCES:
        raise _InputError(f"resource_limit: sources exceed {MAX_SOURCES}")
    seen: set[str] = set()
    result: list[_LoadedSource] = []
    for ref in request.sources:
        if ref.artifact_id in seen:
            raise _InputError(f"invalid_input: duplicate source artifact id {ref.artifact_id}")
        seen.add(ref.artifact_id)
        result.append(_load_source(ref, root, root_fd=root_fd))
    return result


def _resolution(samples: Sequence[DiagnosticSample], config: Mapping[str, Any]) -> float:
    declared = config.get("sample_resolution_s", config.get("nearest_sample_resolution_s"))
    if declared is not None:
        value = _finite(declared, "sample_resolution_s")
        if value < 0:
            raise _InputError("invalid_input: sample_resolution_s must be non-negative")
        return value
    spacings = [
        right.time_s - left.time_s
        for left, right in pairwise(samples)
        if right.time_s > left.time_s
    ]
    return min(spacings) / 2.0 if spacings else 0.0


def _nearest(
    samples: Sequence[DiagnosticSample], target: float, resolution: float
) -> tuple[DiagnosticSample | None, str, float | None]:
    if not samples:
        return None, "source_time_unavailable", None
    selected = min(samples, key=lambda sample: (abs(sample.time_s - target), sample.time_s))
    error = abs(selected.time_s - target)
    if target < samples[0].time_s or target > samples[-1].time_s:
        return selected, "outside_declared_resolution", error
    if error > resolution + 1e-12:
        return selected, "outside_declared_resolution", error
    return selected, "", error


def _context_from_sources(  # noqa: C901
    sources: Sequence[_LoadedSource],
    config: Mapping[str, Any],
    cursor_time_s: float,
    interval_id: str | None,
) -> ReviewContext:
    raw_context = config.get("context")
    context = raw_context if isinstance(raw_context, Mapping) else {}
    raw_cursor = context.get("cursor") if isinstance(context.get("cursor"), Mapping) else {}
    revision_candidates = [
        candidate
        for candidate in (
            config.get("context_revision"),
            config.get("selection_revision"),
            context.get("context_revision"),
            context.get("selection_revision"),
        )
        if candidate is not None
    ]
    if revision_candidates and any(
        candidate != revision_candidates[0] for candidate in revision_candidates[1:]
    ):
        raise _InputError("stale_context_revision: context and selection revisions differ")
    revision_value = revision_candidates[0] if revision_candidates else 0
    if isinstance(raw_cursor, Mapping) and raw_cursor.get("context_revision") is not None:
        cursor_revision = raw_cursor.get("context_revision")
        if revision_value not in (0, cursor_revision):
            raise _InputError("stale_context_revision: cursor and selection revision differ")
        revision_value = cursor_revision
    if (
        isinstance(revision_value, bool)
        or not isinstance(revision_value, int)
        or revision_value < 0
    ):
        raise _InputError("invalid_input: context_revision must be a non-negative integer")

    def selected(name: str, *aliases: str) -> str | None:
        for container in (context, config):
            for key in (name, *aliases):
                value = container.get(key) if isinstance(container, Mapping) else None
                if value is not None:
                    return str(value)
        for source in sources:
            for key in (name, *aliases):
                value = source.identity.get(key)
                if value is not None:
                    return str(value)
        return None

    return ReviewContext(
        campaign_id=selected("campaign_id"),
        execution_id=selected("execution_id", "run_id", "trace_id"),
        episode_id=selected("episode_id", "episode"),
        interval_id=interval_id,
        actor_id=selected("actor_id", "actor"),
        cursor=SourceTimeCursor(cursor_time_s, revision_value, "initial", interval_id),
        context_revision=revision_value,
    )


def _mapping_copy(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _first_mapping(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> tuple[str, Mapping[str, Any]] | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return key, value
    return None


def _control_values(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}
    aliases = {
        "linear_m_s": (
            "linear_m_s",
            "linear_velocity",
            "linear_velocity_m_s",
            "requested_linear_m_s",
            "applied_linear_m_s",
        ),
        "turn_rate_rad_s": (
            "turn_rate_rad_s",
            "angular_velocity",
            "angular_velocity_rad_s",
            "requested_angular_rad_s",
            "applied_angular_rad_s",
        ),
    }
    for output_key, keys in aliases.items():
        for key in keys:
            if key in value:
                number = _optional_finite(value[key])
                if number is not None:
                    output[output_key] = number
                    break
    if isinstance(value.get("units"), (str, Mapping)):
        output["units"] = deepcopy(value["units"])
    if "raw" in value:
        output["raw"] = deepcopy(value["raw"])
    return output


def _control_snapshot(sample: DiagnosticSample, units: Any) -> dict[str, Any]:
    controls = sample.controls
    command_raw = _first_mapping(controls, ("commanded", "requested", "desired", "selected"))
    executed_raw = _first_mapping(
        controls, ("executed", "applied", "environment_action", "applied_environment_action")
    )
    planner_command = _first_mapping(
        sample.planner, ("selected_action", "commanded_action", "requested_action")
    )
    if command_raw is None and planner_command is not None:
        command_raw = planner_command
    command = _control_values(command_raw[1] if command_raw else None)
    executed = _control_values(executed_raw[1] if executed_raw else None)
    # Keep the original control record for reviewers; do not replace missing
    # dimensions with zero or with robot velocity.
    command_state = {
        "status": "available" if command else "unavailable",
        "value": command or None,
        "units": command.get("units", units),
        "value_origin": "planner_visible" if command else "unavailable",
        "source_field": command_raw[0] if command_raw else None,
        "missing_reason": "commanded_control_not_recorded" if not command else None,
    }
    executed_state = {
        "status": "available" if executed else "unavailable",
        "value": executed or None,
        "units": executed.get("units", units),
        "value_origin": "simulator_ground_truth" if executed else "unavailable",
        "source_field": executed_raw[0] if executed_raw else None,
        "missing_reason": "executed_control_not_recorded" if not executed else None,
    }
    comparison: dict[str, Any] = {
        "status": "unavailable",
        "value": None,
        "units": command.get("units", units),
        "value_origin": "post_hoc",
        "missing_reason": "commanded_or_executed_control_not_recorded",
    }
    command_units = _units_token(command.get("units", units))
    executed_units = _units_token(executed.get("units", units))
    if command and executed and not command_units:
        comparison["missing_reason"] = "control_units_not_recorded"
    elif command and executed and command_units == executed_units:
        delta: dict[str, float] = {}
        for key in ("linear_m_s", "turn_rate_rad_s"):
            if key in command and key in executed:
                delta[key] = executed[key] - command[key]
        if delta:
            comparison.update(
                {
                    "status": "available",
                    "value": delta,
                    "missing_reason": None,
                    "derived_from": ["commanded", "executed"],
                }
            )
        else:
            comparison["missing_reason"] = "control_dimension_missing"
    elif command and executed:
        comparison["missing_reason"] = "control_units_mismatch"
    return {
        "commanded": command_state,
        "executed": executed_state,
        "comparison": comparison,
        "status": "available"
        if command_state["status"] == "available" or executed_state["status"] == "available"
        else "unavailable",
        "source_time_s": sample.time_s,
        "source_index": sample.source_index,
        "step": sample.step,
    }


def _candidate_snapshot(planner: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
    raw_candidates: Any = None
    candidate_field = None
    for key in (
        "candidates",
        "candidate_actions",
        "action_candidates",
        "candidate_controls",
        "trajectory_candidates",
        "options",
    ):
        if key in planner:
            raw_candidates = planner[key]
            candidate_field = key
            break
    candidates: list[dict[str, Any]] = []
    if isinstance(raw_candidates, Mapping):
        raw_candidates = [
            dict(value, candidate_id=str(key))
            if isinstance(value, Mapping)
            else {"candidate_id": str(key), "value": value}
            for key, value in raw_candidates.items()
        ]
    if isinstance(raw_candidates, list):
        for index, raw in enumerate(raw_candidates[:MAX_CANDIDATES]):
            if isinstance(raw, Mapping):
                candidate = deepcopy(dict(raw))
            else:
                candidate = {"value": deepcopy(raw)}
            candidate.setdefault("candidate_id", f"candidate-{index}")
            candidate["identity_kind"] = (
                "recorded_id"
                if isinstance(raw, Mapping)
                and any(key in raw for key in ("candidate_id", "id", "name"))
                else "source_index"
            )
            candidates.append(candidate)
    elif raw_candidates is not None:
        candidates.append(
            {
                "candidate_id": "candidate-data-invalid",
                "value": None,
                "missing_reason": "candidates_not_a_collection",
            }
        )
    costs: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        found = _first_mapping(candidate, ("cost", "costs", "objective_cost", "objective", "score"))
        if found:
            costs.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "value": deepcopy(dict(found[1])),
                    "field": found[0],
                }
            )
        else:
            for key in ("cost", "cost_value", "objective_cost", "objective", "score"):
                if key in candidate:
                    costs.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "value": deepcopy(candidate[key]),
                            "field": key,
                        }
                    )
                    break
    if not costs and isinstance(planner.get("candidate_costs"), Mapping):
        costs = [
            {"candidate_id": str(key), "value": deepcopy(value), "field": "candidate_costs"}
            for key, value in planner["candidate_costs"].items()
        ]
    constraints: Any = None
    constraint_field = None
    for key in (
        "constraints",
        "constraint_checks",
        "feasibility",
        "constraint_status",
        "violations",
    ):
        if key in planner:
            constraints, constraint_field = deepcopy(planner[key]), key
            break
    return {
        "candidates": candidates,
        "candidates_state": {
            "status": "available" if candidates else "unavailable",
            "value_origin": "planner_visible" if candidates else "unavailable",
            "source_field": candidate_field,
            "missing_reason": None if candidates else "candidates_not_recorded",
        },
        "costs": costs,
        "costs_state": {
            "status": "available" if costs else "unavailable",
            "value_origin": "planner_visible" if costs else "unavailable",
            "source_field": "candidate_costs" if costs and not candidate_field else candidate_field,
            "missing_reason": None if costs else "costs_not_recorded",
        },
        "constraints": constraints,
        "constraints_state": {
            "status": "available" if constraints is not None else "unavailable",
            "value_origin": "planner_visible" if constraints is not None else "unavailable",
            "source_field": constraint_field,
            "missing_reason": None if constraints is not None else "constraints_not_recorded",
        },
        "selected": deepcopy(planner.get("selected_action", planner.get("selected_candidate"))),
        "selected_status": {
            "status": "available"
            if planner.get("selected_action", planner.get("selected_candidate")) is not None
            else "unavailable",
            "missing_reason": None
            if planner.get("selected_action", planner.get("selected_candidate")) is not None
            else "selected_action_not_recorded",
        },
    }


def _pedestrian_snapshot(  # noqa: C901
    sample: DiagnosticSample, selected_actor: str | None
) -> dict[str, Any]:
    actors: list[dict[str, Any]] = []
    selected_found = selected_actor is None
    for index, actor in enumerate(sample.pedestrians):
        normalized = deepcopy(actor)
        actor_id = normalized.get("actor_id", normalized.get("id", normalized.get("pedestrian_id")))
        if actor_id is not None:
            actor_id = str(actor_id)
            normalized.setdefault("actor_id", actor_id)
        if (
            selected_actor is not None
            and actor_id is not None
            and str(actor_id) == str(selected_actor)
        ):
            selected_found = True
        actors.append(normalized)
    diagnostics: list[dict[str, Any]] = []
    diagnostic_origins: list[dict[str, Any]] = []
    for key in ("pedestrian_diagnostics", "pedestrian_diagnostic", "pedestrian_metrics"):
        value = sample.planner.get(key)
        if isinstance(value, list):
            for item in value[:MAX_PED_DIAGNOSTICS]:
                if isinstance(item, Mapping):
                    diagnostics.append(deepcopy(dict(item)))
                    diagnostic_origins.append(
                        {
                            "index": len(diagnostics) - 1,
                            "value_origin": "planner_visible",
                            "source_field": key,
                        }
                    )
        elif isinstance(value, Mapping):
            diagnostics.append(deepcopy(dict(value)))
            diagnostic_origins.append(
                {
                    "index": len(diagnostics) - 1,
                    "value_origin": "planner_visible",
                    "source_field": key,
                }
            )
    for actor in actors:
        explicit = actor.get("diagnostics")
        if isinstance(explicit, Mapping):
            diagnostics.append({"actor_id": actor.get("actor_id"), **deepcopy(dict(explicit))})
            diagnostic_origins.append(
                {
                    "index": len(diagnostics) - 1,
                    "value_origin": "simulator_ground_truth",
                    "source_field": "pedestrians[].diagnostics",
                    "actor_id": actor.get("actor_id"),
                }
            )
    if selected_actor is not None and not selected_found:
        return {
            "status": "unavailable",
            "reason": "actor_disappeared",
            "actors": [],
            "diagnostics": [],
            "diagnostic_origins": [],
            "selected_actor_id": selected_actor,
            "missing_reason": "selected_actor_not_present_at_source_time",
        }
    origin_values = {item["value_origin"] for item in diagnostic_origins}
    diagnostics_origin = (
        next(iter(origin_values))
        if len(origin_values) == 1
        else "post_hoc"
        if origin_values
        else "unavailable"
    )
    return {
        "status": "available",
        "reason": "",
        "actors": actors
        if selected_actor is None
        else [
            actor
            for actor in actors
            if actor.get("actor_id") is not None
            and str(actor.get("actor_id")) == str(selected_actor)
        ],
        "diagnostics": diagnostics,
        "diagnostic_origins": diagnostic_origins,
        "selected_actor_id": selected_actor,
        "diagnostics_state": {
            "status": "available" if diagnostics else "unavailable",
            "value_origin": diagnostics_origin,
            "missing_reason": None if diagnostics else "pedestrian_diagnostics_not_recorded",
        },
        "actor_count": len(actors),
    }


def _diagnosis_rows(sources: Sequence[_LoadedSource]) -> list[dict[str, Any]]:  # noqa: C901
    rows: list[dict[str, Any]] = []
    for source in sources:
        if (
            source.kind != "failure_diagnosis"
            or source.status != "available"
            or not isinstance(source.payload, Mapping)
        ):
            continue
        for index, row in enumerate(source.payload.get("records", [])[:MAX_RECORDS]):
            if isinstance(row, Mapping):
                source_time = _optional_finite(row.get("onset_time_s"))
                actor_id: str | None = None
                predicate = row.get("source_predicate")
                if isinstance(predicate, Mapping):
                    actors = predicate.get("involved_actors")
                    if isinstance(actors, list) and actors and actors[0] is not None:
                        actor_id = str(actors[0])
                evidence_items = row.get("causal_evidence")
                if actor_id is None and isinstance(evidence_items, list):
                    for evidence in evidence_items:
                        if not isinstance(evidence, Mapping):
                            continue
                        actors = evidence.get("involved_actors")
                        if isinstance(actors, list) and actors and actors[0] is not None:
                            actor_id = str(actors[0])
                            break
                missing_reasons = []
                if source_time is None:
                    missing_reasons.append("source_time_not_recorded")
                if source.units is None:
                    missing_reasons.append("units_not_recorded")
                if source.coordinate_frame is None:
                    missing_reasons.append("coordinate_frame_not_recorded")
                if actor_id is None:
                    missing_reasons.append("actor_not_selected")
                rows.append(
                    {
                        "source": source.ref.artifact_id,
                        "source_index": index,
                        "value": deepcopy(dict(row)),
                        "source_artifact_id": source.ref.artifact_id,
                        "source_identity": deepcopy(source.identity),
                        "source_ref": {
                            "artifact_id": source.ref.artifact_id,
                            "uri": source.ref.uri,
                            "format": source.ref.format,
                            "schema": source.schema_version or source.ref.schema,
                            "sha256": source.digest,
                            "units": _units_token(source.units),
                            "coordinate_frame": source.coordinate_frame
                            or source.ref.coordinate_frame,
                        },
                        "source_time_s": source_time,
                        "time_s": source_time,
                        "actor_id": actor_id,
                        "units": deepcopy(source.units),
                        "coordinate_frame": source.coordinate_frame,
                        "missing_reasons": missing_reasons,
                    }
                )
    return rows


def _source_document(source: _LoadedSource, resolution: float | None = None) -> dict[str, Any]:
    return {
        "artifact_id": source.ref.artifact_id,
        "uri": source.ref.uri,
        "format": source.ref.format,
        "schema": source.schema_version or source.ref.schema,
        "status": source.status,
        "reason": source.reason,
        "kind": source.kind,
        "integrity": source.integrity,
        "declared_sha256": source.ref.sha256 or None,
        "sha256": source.digest,
        "source_identity": deepcopy(source.identity),
        "units": deepcopy(source.units),
        "coordinate_frame": source.coordinate_frame,
        "resolution_s": resolution,
        "range": {"start_s": source.samples[0].time_s, "end_s": source.samples[-1].time_s}
        if source.samples
        else None,
        "diagnostics": deepcopy(source.diagnostics),
        "admission": "not_evaluated",
    }


_UNSET_SOURCE_TIME = object()


def _reference(  # noqa: PLR0913
    source: _LoadedSource,
    sample: DiagnosticSample | None,
    context: ReviewContext,
    *,
    kind: str,
    pointer: str,
    actor_id: str | None = None,
    units: Any = None,
    source_time_s: float | object | None = _UNSET_SOURCE_TIME,
    value_origin: str,
) -> dict[str, Any]:
    source_revision = source.identity.get("source_commit") or source.digest or ""
    identity = (
        source.identity.get("trace_id")
        or source.identity.get("episode_id")
        or source.ref.artifact_id
    )
    source_time = (
        sample.time_s
        if sample is not None
        else context.cursor.time_s
        if source_time_s is _UNSET_SOURCE_TIME
        else source_time_s
    )
    missing_reasons: list[str] = []
    if source_time is None:
        missing_reasons.append("source_time_not_recorded")
    if actor_id is None:
        missing_reasons.append("actor_not_selected")
    if units is None and source.units is None:
        missing_reasons.append("units_not_recorded")
    if source.coordinate_frame is None and source.ref.coordinate_frame is None:
        missing_reasons.append("coordinate_frame_not_recorded")
    reference_id = f"{source.ref.artifact_id}:{kind}:{sample.source_index if sample else 'none'}:{pointer}:{context.context_revision}"
    return {
        "reference_id": reference_id,
        "kind": kind,
        "source_artifact_id": source.ref.artifact_id,
        "source_identity": identity,
        "source_revision": source_revision,
        "source_ref": {
            "artifact_id": source.ref.artifact_id,
            "uri": source.ref.uri,
            "format": source.ref.format,
            "schema": source.schema_version or source.ref.schema,
            "sha256": source.digest,
            "source_commit": source.identity.get("source_commit", source.ref.source_commit or ""),
            "config_identity": source.identity.get(
                "config_identity", source.ref.config_identity or ""
            ),
            "units": _units_token(units if units is not None else source.units),
            "coordinate_frame": source.coordinate_frame or source.ref.coordinate_frame,
        },
        "json_pointer": pointer,
        "source_time_s": source_time,
        "time_s": source_time,
        "step": sample.step if sample is not None else None,
        "actor_id": actor_id,
        "units": deepcopy(units if units is not None else source.units),
        "coordinate_frame": source.coordinate_frame,
        "context_revision": context.context_revision,
        "selection_revision": context.context_revision,
        "interval_id": context.interval_id,
        "value_origin": value_origin,
        "evidence_status": "recorded",
        "missing_reason": missing_reasons[0] if missing_reasons else None,
        "missing_reasons": missing_reasons,
        "admission": "not_evaluated",
    }


def _panel_state(status: str, reason: str = "") -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "value_origin": "unavailable",
        "missing_reason": reason or None,
    }


def _build_document(  # noqa: C901, PLR0915
    request: ComponentRequest,
    sources: Sequence[_LoadedSource],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    diagnostics: list[dict[str, Any]] = []
    for source in sources:
        if source.source_error or source.status != "available":
            diagnostics.append(
                {
                    "artifact_id": source.ref.artifact_id,
                    "reason_code": source.reason or source.source_error or "source_unavailable",
                    "detail": source.reason or source.source_error or "source unavailable",
                }
            )
        diagnostics.extend(deepcopy(source.diagnostics))
    traces = [
        source
        for source in sources
        if source.kind in {"simulation_timeline", "simulation_trace_export", "analysis_trace"}
        and source.status == STATUS_AVAILABLE
        and source.samples
    ]
    diagnoses = _diagnosis_rows(sources)
    primary = traces[0] if traces else None
    resolution = _resolution(primary.samples, config) if primary else 0.0
    source_start = primary.samples[0].time_s if primary else 0.0
    source_end = primary.samples[-1].time_s if primary else source_start
    interval_id = config.get("interval_id")
    if interval_id is not None:
        interval_id = str(interval_id)
    cursor_value = config.get("cursor_time_s", config.get("initial_time_s", source_start))
    cursor_time = _finite(cursor_value, "cursor_time_s")
    context = _context_from_sources(sources, config, cursor_time, interval_id)
    selected_actor = context.actor_id

    planner_panel: dict[str, Any] = {
        "panel": "planner",
        "visible": bool(
            config.get(
                "planner_visible",
                config.get("panels", {}).get("planner", True)
                if isinstance(config.get("panels"), Mapping)
                else True,
            )
        ),
        "toggleable": True,
        "context_revision": context.context_revision,
        "selection_revision": context.context_revision,
        "status": "unavailable",
        "reason": "trace_not_recorded" if primary is None else "source_time_unavailable",
        "source_artifact_id": primary.ref.artifact_id if primary else None,
        "source_identity": deepcopy(primary.identity) if primary else None,
        "units": deepcopy(primary.units) if primary else None,
        "coordinate_frame": primary.coordinate_frame if primary else None,
        "source_time_s": cursor_time,
        "source_index": None,
        "candidates": [],
        "costs": [],
        "constraints": None,
        "missing": [],
    }
    controls_panel: dict[str, Any] = {
        "panel": "controls",
        "visible": bool(
            config.get(
                "controls_visible",
                config.get("panels", {}).get("controls", True)
                if isinstance(config.get("panels"), Mapping)
                else True,
            )
        ),
        "toggleable": True,
        "context_revision": context.context_revision,
        "selection_revision": context.context_revision,
        "status": "unavailable",
        "reason": "trace_not_recorded" if primary is None else "source_time_unavailable",
        "source_artifact_id": primary.ref.artifact_id if primary else None,
        "source_identity": deepcopy(primary.identity) if primary else None,
        "units": deepcopy(primary.units) if primary else None,
        "coordinate_frame": primary.coordinate_frame if primary else None,
        "commanded": _panel_state("unavailable", "commanded_control_not_recorded"),
        "executed": _panel_state("unavailable", "executed_control_not_recorded"),
        "comparison": _panel_state("unavailable", "commanded_or_executed_control_not_recorded"),
        "source_time_s": cursor_time,
        "source_index": None,
    }
    pedestrian_panel: dict[str, Any] = {
        "panel": "pedestrians",
        "visible": bool(
            config.get(
                "pedestrian_visible",
                config.get("panels", {}).get("pedestrians", True)
                if isinstance(config.get("panels"), Mapping)
                else True,
            )
        ),
        "toggleable": True,
        "context_revision": context.context_revision,
        "selection_revision": context.context_revision,
        "status": "unavailable",
        "reason": "trace_not_recorded" if primary is None else "source_time_unavailable",
        "source_artifact_id": primary.ref.artifact_id if primary else None,
        "source_identity": deepcopy(primary.identity) if primary else None,
        "units": deepcopy(primary.units) if primary else None,
        "coordinate_frame": primary.coordinate_frame if primary else None,
        "actors": [],
        "diagnostics": [],
        "selected_actor_id": selected_actor,
        "source_time_s": cursor_time,
        "source_index": None,
    }
    references: list[dict[str, Any]] = []
    if primary is not None:
        selected, selection_reason, temporal_error = _nearest(
            primary.samples, cursor_time, resolution
        )
        if selected is None or selection_reason:
            reason = selection_reason or "source_time_unavailable"
            planner_panel["reason"] = controls_panel["reason"] = pedestrian_panel["reason"] = reason
            planner_panel["temporal_error_s"] = controls_panel[
                "temporal_error_s"
            ] = pedestrian_panel["temporal_error_s"] = temporal_error
        else:
            candidate_view = _candidate_snapshot(selected.planner)
            planner_panel.update(candidate_view)
            planner_panel.update(
                {
                    "status": "available"
                    if candidate_view["candidates_state"]["status"] == "available"
                    or candidate_view["costs_state"]["status"] == "available"
                    or candidate_view["constraints_state"]["status"] == "available"
                    else "partial",
                    "reason": ""
                    if candidate_view["candidates_state"]["status"] == "available"
                    or candidate_view["costs_state"]["status"] == "available"
                    or candidate_view["constraints_state"]["status"] == "available"
                    else "planner_diagnostics_not_recorded",
                    "source_time_s": selected.time_s,
                    "source_index": selected.source_index,
                    "temporal_error_s": temporal_error,
                    "step": selected.step,
                    "value_origin": "planner_visible",
                }
            )
            planner_panel["missing"] = [
                key
                for key in ("candidates", "costs", "constraints")
                if planner_panel[f"{key}_state"]["status"] != "available"
            ]
            controls_panel.update(_control_snapshot(selected, primary.units))
            controls_panel["temporal_error_s"] = temporal_error
            pedestrian_panel.update(_pedestrian_snapshot(selected, selected_actor))
            pedestrian_panel.update(
                {
                    "source_time_s": selected.time_s,
                    "source_index": selected.source_index,
                    "temporal_error_s": temporal_error,
                    "step": selected.step,
                }
            )
            references.append(
                _reference(
                    primary,
                    selected,
                    context,
                    kind="planner",
                    pointer=selected.pointer + "/planner",
                    units=primary.units,
                    value_origin="planner_visible",
                )
            )
            references.append(
                _reference(
                    primary,
                    selected,
                    context,
                    kind="controls",
                    pointer=selected.pointer + "/controls",
                    units=primary.units,
                    value_origin="post_hoc",
                )
            )
            references.append(
                _reference(
                    primary,
                    selected,
                    context,
                    kind="pedestrians",
                    pointer=selected.pointer + "/pedestrians",
                    units=primary.units,
                    actor_id=selected_actor,
                    value_origin="simulator_ground_truth",
                )
            )
            for index, _candidate in enumerate(planner_panel.get("candidates", [])):
                references.append(
                    _reference(
                        primary,
                        selected,
                        context,
                        kind="planner_candidate",
                        pointer=f"{selected.pointer}/planner/candidates/{index}",
                        units=primary.units,
                        value_origin="planner_visible",
                    )
                )

    diagnosis_sources = [source for source in sources if source.kind == "failure_diagnosis"]
    diagnosis_source_reasons = sorted(
        {
            source.reason
            for source in diagnosis_sources
            if source.status != "available" and source.reason
        }
    )
    single_diagnosis = diagnoses[0] if len(diagnoses) == 1 else None
    diagnosis_missing_reasons = sorted(
        {
            reason
            for item in diagnoses
            for reason in item.get("missing_reasons", [])
            if isinstance(reason, str)
        }
    )
    if not diagnoses:
        diagnosis_missing_reasons.extend(
            diagnosis_source_reasons
            or (
                "failure_diagnosis_not_recorded",
                "source_artifact_not_recorded",
                "source_time_not_recorded",
                "units_not_recorded",
                "coordinate_frame_not_recorded",
                "actor_not_selected",
            )
        )
    diagnosis_reason = (
        diagnosis_source_reasons[0]
        if not diagnoses and diagnosis_source_reasons
        else ""
        if diagnoses
        else "failure_diagnosis_not_recorded"
    )
    diagnosis_panel = {
        "panel": "failure_diagnosis",
        "visible": bool(
            config.get(
                "diagnosis_visible",
                config.get("panels", {}).get("failure_diagnosis", True)
                if isinstance(config.get("panels"), Mapping)
                else True,
            )
        ),
        "toggleable": True,
        "status": "available" if diagnoses else "unavailable",
        "reason": diagnosis_reason,
        "records": diagnoses,
        "value_origin": "post_hoc" if diagnoses else "unavailable",
        "source_artifact_id": (
            single_diagnosis.get("source_artifact_id") if single_diagnosis else None
        ),
        "source_identity": (
            deepcopy(single_diagnosis.get("source_identity")) if single_diagnosis else None
        ),
        "source_time_s": single_diagnosis.get("source_time_s") if single_diagnosis else None,
        "actor_id": single_diagnosis.get("actor_id") if single_diagnosis else None,
        "units": deepcopy(single_diagnosis.get("units")) if single_diagnosis else None,
        "coordinate_frame": (
            single_diagnosis.get("coordinate_frame") if single_diagnosis else None
        ),
        "missing_reasons": diagnosis_missing_reasons,
        "context_revision": context.context_revision,
        "selection_revision": context.context_revision,
        "source_artifacts": [
            {
                "artifact_id": source.ref.artifact_id,
                "source_identity": deepcopy(source.identity),
                "source_ref": {
                    "artifact_id": source.ref.artifact_id,
                    "uri": source.ref.uri,
                    "format": source.ref.format,
                    "schema": source.schema_version or source.ref.schema,
                    "sha256": source.digest,
                },
                "units": deepcopy(source.units),
                "coordinate_frame": source.coordinate_frame,
            }
            for source in sources
            if source.kind == "failure_diagnosis"
        ],
    }
    for item in diagnoses:
        item["context_revision"] = context.context_revision
        item["selection_revision"] = context.context_revision
        source = next(
            (candidate for candidate in sources if candidate.ref.artifact_id == item["source"]),
            None,
        )
        if source is not None:
            references.append(
                _reference(
                    source,
                    None,
                    context,
                    kind="failure_diagnosis",
                    pointer=f"/records/{item['source_index']}",
                    units=source.units,
                    actor_id=item.get("actor_id"),
                    source_time_s=item.get("source_time_s"),
                    value_origin="post_hoc",
                )
            )

    panel_status = {
        "planner": planner_panel["status"],
        "controls": controls_panel["status"],
        "pedestrians": pedestrian_panel["status"],
        "failure_diagnosis": diagnosis_panel["status"],
    }
    required_panels = config.get("required_panels", [])
    if required_panels is None:
        required_panels = []
    if not isinstance(required_panels, list):
        raise _InputError("invalid_input: required_panels must be a list")
    missing_required = [
        str(name) for name in required_panels if panel_status.get(str(name)) != "available"
    ]
    diagnostics.extend(
        {"reason_code": "required_panel_unavailable", "panel": name} for name in missing_required
    )

    source_identity = {
        source.ref.artifact_id: _source_document(source, resolution if source.samples else None)
        for source in sources
    }
    source_bindings = dict(source_identity)
    source_identity_document = {**source_bindings, "sources": source_bindings}
    status = (
        STATUS_COMPLETE
        if primary is not None and not diagnostics and not missing_required
        else STATUS_PARTIAL
    )
    if not traces:
        status = (
            STATUS_FAILED
            if any(source.status == STATUS_FAILED for source in sources)
            else (
                STATUS_PARTIAL
                if any(source.status == STATUS_PARTIAL for source in sources)
                else STATUS_UNAVAILABLE
            )
        )
    if missing_required and not traces:
        status = STATUS_UNAVAILABLE
    document = {
        "schema_version": DIAGNOSTICS_MODEL_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "status": status,
        "evidence_status": "diagnostic_only",
        "evidence_boundary": "analysis_workbench_only",
        "diagnostic_only": True,
        "time": {
            "authority": "simulation_time",
            "origin_s": source_start,
            "terminal_s": source_end,
            "cursor": context.cursor.to_dict(),
            "resolution_s": resolution,
            "rule": "nearest_sample_within_declared_resolution; no_interpolation",
        },
        "context": context.to_dict(),
        "selection_revision": context.context_revision,
        "source_identity": source_identity_document,
        "inventories": [
            _source_document(source, resolution if source.samples else None) for source in sources
        ],
        "panels": {
            "planner": planner_panel,
            "controls": controls_panel,
            "pedestrians": pedestrian_panel,
            "failure_diagnosis": diagnosis_panel,
        },
        "panel_status": panel_status,
        "toggles": {
            name: {"visible": bool(panel.get("visible", True)), "toggleable": True}
            for name, panel in (
                ("planner", planner_panel),
                ("controls", controls_panel),
                ("pedestrians", pedestrian_panel),
                ("failure_diagnosis", diagnosis_panel),
            )
        },
        "evidence_references": references,
        "provenance": {
            "evidence_status": "diagnostic_only",
            "evidence_boundary": "analysis_workbench_only",
            "admission": "not_evaluated",
            "source_mutation": "none",
            "source_identity": source_identity_document,
            "selection_revision": context.context_revision,
            "primary_inventory": primary.ref.artifact_id if primary else None,
            "unsupported_claims": [
                "no planner instrumentation or private observation is implied",
                "no benchmark, scientific, causal, or paper-facing claim is made",
            ],
        },
        "diagnostics": diagnostics,
    }
    return document, diagnostics, status


def descriptor_document() -> dict[str, Any]:
    """Return the component-descriptor.v1 document."""

    return json.loads(json.dumps(_DESCRIPTOR_DOCUMENT, allow_nan=False))


def descriptor() -> dict[str, Any]:
    """Return the component descriptor.

    Returns:
        A JSON-safe ``component-descriptor.v1`` document.
    """

    return descriptor_document()


def build_diagnostic_model(
    request: ComponentRequest, *, base: Path | None = None
) -> dict[str, Any]:
    """Build the renderer-neutral diagnostics model without writing output.

    Returns:
        A strict-JSON-compatible ``review-diagnostics.v1`` mapping.
    """

    if not isinstance(request, ComponentRequest):
        raise TypeError("request must be a ComponentRequest")
    root = _absolute_path(base or Path.cwd())
    root_handle = _open_directory(root)
    try:
        sources = _load_sources(request, root, root_fd=root_handle.fd)
        document, _diagnostics, _status = _build_document(request, sources, request.config)
        return document
    finally:
        root_handle.close()


build_model = build_diagnostic_model


def _render_html(document: Mapping[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Robot SF recorded diagnostics</title></head><body>"
        "<main><h1>Recorded planner and pedestrian diagnostics</h1>"
        '<div id="review-diagnostics-root"></div>'
        '<script type="application/json" id="review-diagnostics-data">'
        + payload
        + '</script><script type="module" src="./components/review_diagnostics/review_diagnostics.js"></script>'
        "</main></body></html>"
    )


def result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize and validate one component-result.v1 envelope.

    Returns:
        A validated JSON-safe result mapping.
    """

    document = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": result.request_id,
        "component_id": result.component_id,
        "status": result.status,
        "artifacts": [dict(item) for item in result.artifacts],
        "diagnostics": [dict(item) for item in result.diagnostics],
        "provenance": dict(result.provenance),
        "reason": result.reason,
    }
    component_result_from_dict(document)
    return document


def _result(
    request: ComponentRequest | Mapping[str, Any],
    status: str,
    *,
    reason: str = "",
    artifacts: Sequence[dict[str, Any]] = (),
    diagnostics: Sequence[dict[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> ComponentResult:
    request_id = (
        request.request_id
        if isinstance(request, ComponentRequest)
        else str(request.get("request_id", "unknown"))
    )
    component_id = (
        request.component_id
        if isinstance(request, ComponentRequest)
        else str(request.get("component_id", COMPONENT_ID))
    )
    result_provenance = {
        "component_version": COMPONENT_VERSION,
        "evidence_status": "diagnostic_only",
        "evidence_boundary": "analysis_workbench_only",
        "admission": "not_evaluated",
    }
    if provenance:
        result_provenance.update(dict(provenance))
    return ComponentResult(
        request_id,
        component_id,
        status,
        tuple(artifacts),
        tuple(diagnostics),
        result_provenance,
        reason,
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:  # noqa: C901
    """Build and emit one offline diagnostics component result.

    Returns:
        A truthful shared component-result envelope.
    """

    if not isinstance(request, ComponentRequest):
        return _result({}, STATUS_FAILED, reason="invalid_request: expected ComponentRequest")
    if request.component_id != COMPONENT_ID:
        return _result(
            request, STATUS_UNAVAILABLE, reason=f"unsupported_component: {request.component_id}"
        )
    unsupported = sorted(set(request.required_capabilities) - SUPPORTED_CAPABILITIES)
    if unsupported:
        return _result(
            request,
            STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(unsupported)}",
        )
    if request.config.get("cancelled") is True:
        return _result(request, STATUS_CANCELLED, reason="cancelled: request cancelled before read")
    minimum = request.config.get(
        "min_component_version", request.config.get("required_component_version")
    )
    if minimum is not None:
        try:
            requested = tuple(int(part) for part in str(minimum).split("."))
            current = tuple(int(part) for part in COMPONENT_VERSION.split("."))
        except (TypeError, ValueError):
            requested, current = (10**9,), (0,)
        if not isinstance(minimum, str) or requested > current:
            return _result(
                request,
                STATUS_FAILED,
                reason=f"incompatible_component_version: request needs {minimum}, component is {COMPONENT_VERSION}",
            )
    root = _absolute_path(base or Path.cwd())
    root_handle: _DirectoryHandle | None = None
    output_dir: _DirectoryHandle | None = None
    try:
        root_handle = _open_directory(root)
        sources = _load_sources(request, root, root_fd=root_handle.fd)
        output_dir = _reserve_output(root, request.output_directory, root_fd=root_handle.fd)
        document, diagnostics, status = _build_document(request, sources, request.config)
        result_reason = ""
        if status != STATUS_COMPLETE:
            result_reason = next(
                (
                    str(item.get("detail"))
                    for item in diagnostics
                    if isinstance(item, Mapping) and item.get("detail")
                ),
                "diagnostic_panels_not_complete",
            )
        emitted: list[dict[str, Any]] = []
        model_digest = _write_json(output_dir, OUTPUT_MODEL_FILENAME, document)
        emitted.append(
            {
                "artifact_id": OUTPUT_MODEL_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_MODEL_FILENAME),
                "sha256": model_digest,
            }
        )
        reference_document = {
            "schema_version": EVIDENCE_REFERENCE_SCHEMA_VERSION,
            "evidence_status": "diagnostic_only",
            "context": document["context"],
            "selection_revision": document["selection_revision"],
            "source_identity": document["source_identity"],
            "references": document["evidence_references"],
            "limitations": document["provenance"]["unsupported_claims"],
        }
        reference_digest = _write_json(output_dir, OUTPUT_REFERENCE_FILENAME, reference_document)
        emitted.append(
            {
                "artifact_id": OUTPUT_REFERENCE_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_REFERENCE_FILENAME),
                "sha256": reference_digest,
            }
        )
        html_digest = _write_text(output_dir, OUTPUT_HTML_FILENAME, _render_html(document))
        emitted.append(
            {
                "artifact_id": OUTPUT_HTML_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_HTML_FILENAME),
                "sha256": html_digest,
            }
        )
        component_digest = _copy_web_component(output_dir)
        component_artifact = "components/review_diagnostics/review_diagnostics.js"
        emitted.append(
            {
                "artifact_id": component_artifact,
                "uri": str(Path(request.output_directory) / component_artifact),
                "sha256": component_digest,
            }
        )
        capability_document = {
            "schema_version": MISSING_CAPABILITY_SCHEMA_VERSION,
            "requested": list(request.required_capabilities),
            "available": [
                name for name, value in document["panel_status"].items() if value == "available"
            ],
            "missing": [
                name for name, value in document["panel_status"].items() if value != "available"
            ],
            "diagnostics": diagnostics,
        }
        capability_digest = _write_json(output_dir, OUTPUT_CAPABILITY_FILENAME, capability_document)
        emitted.append(
            {
                "artifact_id": OUTPUT_CAPABILITY_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_CAPABILITY_FILENAME),
                "sha256": capability_digest,
            }
        )
        return _result(
            request,
            status,
            artifacts=emitted if status == STATUS_COMPLETE else (),
            diagnostics=diagnostics,
            reason=result_reason,
            provenance={
                "output_directory": request.output_directory,
                "model_schema": DIAGNOSTICS_MODEL_SCHEMA_VERSION,
                "reference_schema": EVIDENCE_REFERENCE_SCHEMA_VERSION,
                "emitted_artifacts": emitted,
                "selection_revision": document["selection_revision"],
                "source_identity": document["source_identity"],
            },
        )
    except (
        ReviewContractsValidationError,
        _InputError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        return _result(request, STATUS_FAILED, reason=_reason(error, "failed"))
    finally:
        if output_dir is not None:
            output_dir.close()
        if root_handle is not None:
            root_handle.close()


def _strict_cli_json(path: str) -> Any:
    return _strict_loads(_read_under(Path(path).parent, Path(path).name))


def _cli_failure(reason: str, payload: Any = None) -> int:
    if isinstance(payload, Mapping):
        request_id = str(payload.get("request_id", "unknown"))
        component_id = str(payload.get("component_id", COMPONENT_ID))
    else:
        request_id, component_id = "unknown", COMPONENT_ID
    print(  # noqa: T201
        json.dumps(
            result_document(
                _result(
                    {"request_id": request_id, "component_id": component_id},
                    STATUS_FAILED,
                    reason=reason,
                )
            ),
            sort_keys=True,
            indent=2,
        )
    )
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build offline recorded planner/pedestrian diagnostics."
    )
    parser.add_argument("--input", default=None, help="component-request.v1 JSON")
    parser.add_argument(
        "--config", default=None, help="optional JSON config merged into request config"
    )
    parser.add_argument("--output", default=None, help="new output directory relative to --base")
    parser.add_argument("--base", default=None, help="base directory for source/output resolution")
    parser.add_argument("--descriptor", action="store_true", help="print component descriptor")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; only a complete result returns zero.

    Returns:
        Shell exit code (zero only for a complete result).
    """

    args = _build_parser().parse_args(argv)
    if args.descriptor:
        print(json.dumps(descriptor_document(), sort_keys=True, indent=2))  # noqa: T201
        return 0
    if args.input is None or args.output is None:
        return _cli_failure("invalid_input: --input and --output are required")
    payload: Any = None
    try:
        payload = _strict_cli_json(args.input)
        if not isinstance(payload, dict):
            return _cli_failure("invalid_input: request must be an object", payload)
        config = payload.get("config", {})
        if not isinstance(config, dict):
            return _cli_failure("invalid_input: request config must be an object", payload)
        if args.config is not None:
            external = _strict_cli_json(args.config)
            if not isinstance(external, dict):
                return _cli_failure("invalid_input: config must be an object", payload)
            config = {**config, **external}
        payload = {**payload, "config": config, "output_directory": args.output}
        request = component_request_from_dict(payload, source=args.input)
        result = run(request, base=Path(args.base) if args.base else None)
        print(json.dumps(result_document(result), sort_keys=True, indent=2))  # noqa: T201
        return (
            0
            if result.status == STATUS_COMPLETE
            else 2
            if result.status in {STATUS_PARTIAL, STATUS_UNAVAILABLE, STATUS_CANCELLED}
            else 1
        )
    except (
        ReviewContractsValidationError,
        _InputError,
        OSError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        return _cli_failure(f"invalid_input: {_reason(error)}", payload)


__all__ = [
    "ANALYSIS_TRACE_RECORD_SCHEMA_VERSION",
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "DESCRIPTOR",
    "DIAGNOSTICS_MODEL_SCHEMA_VERSION",
    "EVIDENCE_REFERENCE_SCHEMA_VERSION",
    "OUTPUT_HTML_FILENAME",
    "OUTPUT_MODEL_FILENAME",
    "OUTPUT_REFERENCE_FILENAME",
    "ReviewContext",
    "SourceTimeCursor",
    "build_diagnostic_model",
    "build_model",
    "descriptor",
    "descriptor_document",
    "main",
    "result_document",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
