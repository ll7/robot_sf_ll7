"""SREV-07 review-storyboard component: ranked candidates plus storyboard spec.

This module owns the SREV-07 leaf surface only: a ``run(request)`` adapter plus
a standalone CLI that consumes the SREV-01 shared contracts, ranks bundle
episodes with stable tie-breaking, and emits a deterministic
``visualization-spec.v1`` document with explicit source and override
provenance.  The component is diagnostic tooling: its fixture outputs are not
scientific, benchmark, safety, or paper-facing evidence.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

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
    review_bundle_from_dict,
    visualization_spec_from_dict,
)
from robot_sf.benchmark.trace_exemplar_interest import (
    DEFAULT_WEIGHTS as TRACE_EXEMPLAR_INTEREST_DEFAULT_WEIGHTS,
)

_SUPPORTED_DIR_FD_OPERATIONS = frozenset(
    name
    for name in ("link", "mkdir", "open", "rmdir", "unlink")
    if getattr(os, name, None) in getattr(os, "supports_dir_fd", ())
)

COMPONENT_ID = "srev07-review-storyboard"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = ("review-bundle",)
OPTIONAL_CAPABILITIES = ("exemplar-scores", "event-index")

OUTPUT_SPEC_FILENAME = "storyboard-spec.json"
OUTPUT_OVERRIDES_FILENAME = "override-provenance.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

TIE_BREAK_METHOD = "score-descending-then-episode-id-ascending"
EVIDENCE_BOUNDARY = "diagnostic_only"
MISSING_CAPABILITY_SCHEMA_VERSION = "missing-capability-report.v1"
EXEMPLAR_SCORES_SCHEMA_VERSION = "srev07-exemplar-scores.v1"
CANONICAL_SCORE_ADAPTER_VERSION = "trace-exemplar-interest.report-adapter.v1"
CANONICAL_SCORE_FEATURES = frozenset(TRACE_EXEMPLAR_INTEREST_DEFAULT_WEIGHTS)
CANONICAL_SCORE_REPORT_FIELDS = frozenset({"roots", "weights", "episodes", "comparison_pairs"})
CANONICAL_SCORE_EPISODE_FIELDS = frozenset(
    {
        "episode_dir",
        "episode_id",
        "episode_status",
        "planner",
        "scenario_id",
        "seed",
        "features",
        "composite_score",
    }
)
CANONICAL_SCORE_PAIR_FIELDS = frozenset(
    {
        "scenario_id",
        "seed",
        "left_episode_id",
        "left_planner",
        "right_episode_id",
        "right_planner",
        "outcome_divergence",
        "trajectory_divergence",
        "pair_score",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")

_CANONICAL_NON_ADMISSIBLE_STATUS_TOKENS = frozenset(
    {
        "aborted",
        "blocked",
        "cancelled",
        "canceled",
        "degraded",
        "error",
        "failed",
        "fallback",
        "incomplete",
        "invalid",
        "not_applicable",
        "not_admissible",
        "not_available",
        "not_run",
        "not_usable",
        "non_admissible",
        "non_canonical",
        "noncanonical",
        "partial",
        "partial_failure",
        "provisional",
        "skipped",
        "timeout",
        "timed_out",
        "unavailable",
        "unusable",
        "unknown",
    }
)

# The component is intentionally bounded because it is an offline diagnostic
# consumer.  These limits prevent malformed or accidentally unbounded fixture
# inputs from turning a contract probe into an uncontrolled file/memory read.
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_SOURCES = 256
MAX_BUNDLE_EPISODES = 10_000
MAX_EPISODE_REFERENCES = 64
MAX_SCORE_ENTRIES = 100_000
MAX_INTERVALS = 100_000
MAX_OVERRIDE_IDS = 10_000
# A valid bundle can still contain many references.  These aggregate caps keep
# nested verification bounded independently of the per-episode and per-file
# limits above, including when the same URI is repeated across episodes.
MAX_TOTAL_BUNDLE_REFERENCES = 20_000
MAX_TOTAL_BUNDLE_READS = 20_000
MAX_TOTAL_BUNDLE_BYTES = 64 * 1024 * 1024

_DESCRIPTOR_DOCUMENT: dict[str, Any] = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": [
        "visualization-spec.v1",
        "override-provenance.v1",
        "missing-capability-report.v1",
    ],
}

# Validate the public descriptor at import time so descriptor drift fails
# before a caller can invoke the component.
_DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)


class _SecurePathError(OSError):
    """Identify a path operation that could not preserve the containment contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _OutputCollisionError(FileExistsError):
    """Identify an output-directory collision separately from staging failures."""


@dataclass(frozen=True, slots=True)
class _StoryboardComponentResult(ComponentResult):
    """Local result envelope carrying the shared schema version field."""

    artifacts: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = COMPONENT_RESULT_SCHEMA_VERSION


@dataclass
class _Candidate:
    """One ranked storyboard candidate."""

    episode_id: str
    score: float | None = None
    score_source: str = "unavailable"
    pinned: bool = False
    excluded: bool = False


def _result(**kwargs: Any) -> ComponentResult:
    """Construct a result with the versioned local API envelope.

    Returns:
        A result carrying ``component-result.v1`` as ``schema_version``.
    """
    if "artifacts" in kwargs:
        kwargs["artifacts"] = list(kwargs["artifacts"])
    if "diagnostics" in kwargs:
        kwargs["diagnostics"] = list(kwargs["diagnostics"])
    return _StoryboardComponentResult(**kwargs)


def descriptor() -> dict[str, Any]:
    """Return this component's versioned, self-contained descriptor."""
    return json.loads(json.dumps(_DESCRIPTOR_DOCUMENT, allow_nan=False))


def result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize and validate the versioned result envelope for API/CLI use.

    Returns:
        A JSON-serializable, schema-validated result mapping.
    """
    payload: dict[str, Any] = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": result.request_id,
        "component_id": result.component_id,
        "status": result.status,
        "artifacts": [dict(item) for item in result.artifacts],
        "diagnostics": [dict(item) for item in result.diagnostics],
        "provenance": dict(result.provenance),
        "reason": result.reason,
    }
    component_result_from_dict(payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently keeping one value.

    Returns:
        The object mapping when every key is unique.
    """
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject JSON extensions such as ``NaN`` and infinities."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_loads(raw: bytes) -> Any:
    """Parse UTF-8 JSON without duplicate keys or non-standard constants.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If the bytes are not strict, finite UTF-8 JSON.
    """
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError("invalid strict JSON") from error


def _safe_relative_parts(path_value: str) -> tuple[str, ...] | None:
    """Return safe lexical path components for descriptor-relative access."""
    try:
        if not isinstance(path_value, str) or not path_value:
            return None
        windows_candidate = PureWindowsPath(path_value)
        candidate = Path(path_value)
        if (
            candidate.is_absolute()
            or windows_candidate.is_absolute()
            or windows_candidate.drive
            or "\\" in path_value
            or any(ord(character) < 32 or ord(character) == 127 for character in path_value)
            or ".." in candidate.parts
            or ".." in windows_candidate.parts
        ):
            return None
        parts = tuple(part for part in candidate.parts if part not in {"", "."})
        return parts or None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _supports_dir_fd(*operation_names: str) -> bool:
    """Return whether all requested operations expose descriptor-relative APIs."""
    return all(name in _SUPPORTED_DIR_FD_OPERATIONS for name in operation_names)


def _secure_directory_flags() -> int:
    """Return directory-open flags required for no-follow traversal."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None or not _supports_dir_fd("open"):
        raise _SecurePathError("secure_path_unavailable")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _secure_file_flags() -> int:
    """Return regular-file flags that cannot follow a final symlink or block on a FIFO."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None or not _supports_dir_fd("open"):
        raise _SecurePathError("secure_path_unavailable")
    return os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)


def _open_directory_anchor(root: Path) -> int:
    """Open an absolute directory one component at a time without following symlinks.

    Returns:
        An open descriptor for the pinned directory.
    """
    flags = _secure_directory_flags()
    absolute = root if root.is_absolute() else Path(os.path.abspath(root))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise _SecurePathError("secure_path_unavailable")
    descriptor = os.open(os.sep, flags)
    try:
        for part in parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_relative(
    directory_fd: int, parts: tuple[str, ...], *, create: bool = False
) -> int:
    """Open a directory below ``directory_fd`` with no-follow components.

    Returns:
        An open descriptor for the requested directory.
    """
    flags = _secure_directory_flags()
    if create and not _supports_dir_fd("mkdir"):
        raise _SecurePathError("secure_path_unavailable")
    descriptor = os.dup(directory_fd)
    try:
        for part in parts:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_file(root_fd: int, parts: tuple[str, ...]) -> int:
    """Open one relative file below a pinned root directory.

    Returns:
        An open descriptor for the requested file.
    """
    if not parts:
        raise _SecurePathError("source_uri_unsafe")
    parent_fd = _open_directory_relative(root_fd, parts[:-1])
    try:
        return os.open(parts[-1], _secure_file_flags(), dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _read_regular_file(  # noqa: C901
    path: Path,
    *,
    max_bytes: int | None = None,
    root_fd: int | None = None,
    relative_parts: tuple[str, ...] | None = None,
) -> tuple[bytes | None, str | None]:
    """Read a bounded regular file through a pinned, no-follow descriptor path.

    Returns:
        Raw bytes and no error, or ``None`` and a stable read error code.
    """
    if max_bytes is None:
        max_bytes = MAX_SOURCE_BYTES
    descriptor: int | None = None
    try:
        if root_fd is not None:
            parts = relative_parts or _safe_relative_parts(str(path))
            if parts is None:
                return None, "source_uri_unsafe"
            descriptor = _open_relative_file(root_fd, parts)
        else:
            absolute = Path(os.path.abspath(path))
            parent_fd = _open_directory_anchor(absolute.parent)
            try:
                descriptor = os.open(absolute.name, _secure_file_flags(), dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None, "source_not_regular_file"
        if metadata.st_size > max_bytes:
            return None, "source_too_large"
        chunks: list[bytes] = []
        total = 0
        while total < max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total >= max_bytes and os.fstat(descriptor).st_size > max_bytes:
            return None, "source_too_large"
        return b"".join(chunks), None
    except _SecurePathError as error:
        return (
            None,
            "source_uri_unsafe" if error.reason == "source_uri_unsafe" else "source_unreadable",
        )
    except OSError as error:
        unsafe_errors = {errno.ELOOP, errno.ENOTDIR}
        return None, "source_uri_unsafe" if error.errno in unsafe_errors else "source_unreadable"
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _json_bytes(payload: Any) -> bytes:
    """Serialize one sidecar as deterministic strict JSON bytes.

    Returns:
        Exact UTF-8 bytes for the sidecar payload.
    """
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_json_at(directory_fd: int, filename: str, payload: Any) -> str:
    """Create one sidecar below a pinned directory and return its byte digest.

    Returns:
        SHA-256 digest of the exact bytes created below ``directory_fd``.
    """
    if _safe_relative_parts(filename) != (filename,):
        raise _SecurePathError("unsafe_output_filename")
    if not _supports_dir_fd("open"):
        raise _SecurePathError("secure_path_unavailable")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    encoded = _json_bytes(payload)
    descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("sidecar write made no progress")
            written += count
    finally:
        os.close(descriptor)
    return _sha256_bytes(encoded)


def _remove_known_entries(directory_fd: int, filenames: tuple[str, ...]) -> None:
    """Remove only known sidecar names through a pinned directory descriptor."""
    if not _supports_dir_fd("unlink"):
        return
    for filename in filenames:
        try:
            os.unlink(filename, dir_fd=directory_fd)
        except OSError:
            pass


def _same_directory(left_fd: int, right_fd: int) -> bool:
    """Return whether two descriptors identify the same directory inode."""
    try:
        left = os.fstat(left_fd)
        right = os.fstat(right_fd)
    except OSError:
        return False
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _assert_directory_identity(
    root_fd: int, parts: tuple[str, ...], expected_fd: int, reason: str
) -> None:
    """Fail closed when the visible path no longer names the opened directory."""
    try:
        observed_fd = _open_directory_relative(root_fd, parts)
    except OSError as error:
        raise _SecurePathError(reason) from error
    try:
        if not _same_directory(observed_fd, expected_fd):
            raise _SecurePathError(reason)
    finally:
        os.close(observed_fd)


def _assert_anchor_identity(root: Path, expected_fd: int, reason: str) -> None:
    """Fail closed when the visible root no longer names the pinned root."""
    try:
        observed_fd = _open_directory_anchor(root)
    except OSError as error:
        raise _SecurePathError(reason) from error
    try:
        if not _same_directory(observed_fd, expected_fd):
            raise _SecurePathError(reason)
    finally:
        os.close(observed_fd)


def _create_staging_directory(parent_fd: int, output_name: str) -> tuple[str, int]:
    """Reserve a private staging directory below a pinned output parent.

    Returns:
        The private directory name and its open descriptor.
    """
    flags = _secure_directory_flags()
    for _ in range(128):
        staging_name = f".{output_name}-{secrets.token_hex(8)}"
        try:
            os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            return staging_name, os.open(staging_name, flags, dir_fd=parent_fd)
        except BaseException:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    raise OSError("could not reserve a private staging directory")


def _stage_outputs(  # noqa: C901
    root_fd: int, output_parts: tuple[str, ...], payloads: dict[str, Any]
) -> dict[str, str]:
    """Publish sidecars atomically using only pinned descriptor-relative paths.

    Returns:
        Mapping from sidecar filename to its exact-byte SHA-256 digest.
    """
    if not output_parts or not _supports_dir_fd("link", "mkdir", "rmdir", "unlink"):
        raise _SecurePathError("secure_path_unavailable")
    parent_parts = output_parts[:-1]
    output_name = output_parts[-1]
    parent_fd = _open_directory_relative(root_fd, parent_parts, create=True)
    staging_name: str | None = None
    staging_fd: int | None = None
    output_fd: int | None = None
    output_created = False
    committed = False
    filenames = tuple(sorted(payloads))
    try:
        staging_name, staging_fd = _create_staging_directory(parent_fd, output_name)
        digests = {
            filename: _write_json_at(staging_fd, filename, payloads[filename])
            for filename in filenames
        }
        try:
            os.mkdir(output_name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise _OutputCollisionError(output_name) from error
        output_created = True
        output_fd = os.open(output_name, _secure_directory_flags(), dir_fd=parent_fd)
        for filename in filenames:
            os.link(
                filename,
                filename,
                src_dir_fd=staging_fd,
                dst_dir_fd=output_fd,
                follow_symlinks=False,
            )
        _assert_directory_identity(
            root_fd, parent_parts, parent_fd, "output_parent_changed_during_publication"
        )
        _assert_directory_identity(
            root_fd, output_parts, output_fd, "output_changed_during_publication"
        )
        committed = True
        return digests
    finally:
        if output_fd is not None:
            if not committed:
                _remove_known_entries(output_fd, filenames)
            os.close(output_fd)
        if output_created and not committed:
            try:
                os.rmdir(output_name, dir_fd=parent_fd)
            except OSError:
                pass
        if staging_fd is not None:
            _remove_known_entries(staging_fd, filenames)
            os.close(staging_fd)
        if staging_name is not None:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _canonical_digest(payload: Any) -> str:
    """Return the stable logical digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component."""
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    try:
        wanted = int(str(minimum).split(".", maxsplit=1)[0])
        ours = int(COMPONENT_VERSION.split(".", maxsplit=1)[0])
    except ValueError:
        return f"incompatible_component_version: malformed min_component_version: {minimum!r}"
    if wanted > ours:
        return f"incompatible_component_version: request needs v{wanted}, component is v{ours}"
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject requests this component cannot serve.

    Returns:
        An unavailable or failed result, or ``None`` when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=version_error,
        )
    return None


def _ref_value(ref: Any, key: str, default: Any = "") -> Any:
    """Read a field from either a SourceRef or a validated bundle mapping.

    Returns:
        The requested field value or ``default`` when it is not present.
    """
    if hasattr(ref, key):
        return getattr(ref, key)
    if isinstance(ref, dict):
        return ref.get(key, default)
    return default


def _source_record(
    ref: Any,
    *,
    integrity_status: str,
    observed_sha256: str = "",
    episode_id: str | None = None,
    scope: str = "request",
) -> dict[str, Any]:
    """Build a deterministic, provenance-complete source record.

    Returns:
        A JSON-serializable source identity and integrity record.
    """
    record: dict[str, Any] = {
        "artifact_id": str(_ref_value(ref, "artifact_id")),
        "uri": str(_ref_value(ref, "uri")),
        "format": str(_ref_value(ref, "format")),
        "schema": str(_ref_value(ref, "schema")),
        "sha256": str(_ref_value(ref, "sha256")),
        "source_commit": str(_ref_value(ref, "source_commit")),
        "config_identity": str(_ref_value(ref, "config_identity")),
        "units": str(_ref_value(ref, "units")),
        "coordinate_frame": str(_ref_value(ref, "coordinate_frame")),
        "integrity_status": integrity_status,
        "scope": scope,
    }
    if observed_sha256:
        record["observed_sha256"] = observed_sha256
    if episode_id is not None:
        record["episode_id"] = episode_id
    return record


def _verify_reference(
    ref: Any,
    root: Path,
    *,
    root_fd: int | None = None,
    episode_id: str | None = None,
    scope: str = "request",
    max_bytes: int | None = None,
) -> tuple[bytes | None, dict[str, Any], str | None]:
    """Read bytes, verify the declared digest, and report safe path status.

    Returns:
        Raw bytes when readable, the source record, and an optional diagnostic code.
    """
    artifact_id = str(_ref_value(ref, "artifact_id"))
    uri = str(_ref_value(ref, "uri"))
    relative_parts = _safe_relative_parts(uri)
    if relative_parts is None:
        return (
            None,
            _source_record(ref, integrity_status="unsafe_path", episode_id=episode_id, scope=scope),
            f"{artifact_id}: source_uri_unsafe",
        )
    if max_bytes is not None and max_bytes < 0:
        return (
            None,
            _source_record(ref, integrity_status="unavailable", episode_id=episode_id, scope=scope),
            f"{artifact_id}: source_too_large",
        )
    path = root.joinpath(*relative_parts)
    raw, read_problem = _read_regular_file(
        path,
        max_bytes=max_bytes,
        root_fd=root_fd,
        relative_parts=relative_parts,
    )
    if read_problem is not None:
        integrity_status = "unsafe_path" if read_problem == "source_uri_unsafe" else "unavailable"
        record = _source_record(
            ref,
            integrity_status=integrity_status,
            episode_id=episode_id,
            scope=scope,
        )
        record["read_status"] = read_problem
        return (
            None,
            record,
            f"{artifact_id}: {read_problem}",
        )
    assert raw is not None
    observed = _sha256_bytes(raw)
    declared = str(_ref_value(ref, "sha256")).lower()
    if not declared:
        record = _source_record(
            ref,
            integrity_status="observed_unbound",
            observed_sha256=observed,
            episode_id=episode_id,
            scope=scope,
        )
        return raw, record, f"{artifact_id}: source_digest_missing"
    if declared != observed:
        record = _source_record(
            ref,
            integrity_status="digest_mismatch",
            observed_sha256=observed,
            episode_id=episode_id,
            scope=scope,
        )
        return raw, record, f"{artifact_id}: source_digest_mismatch"
    return (
        raw,
        _source_record(
            ref,
            integrity_status="verified",
            observed_sha256=observed,
            episode_id=episode_id,
            scope=scope,
        ),
        None,
    )


def _read_json(
    ref: Any,
    root: Path,
    *,
    root_fd: int | None = None,
    scope: str = "request",
    episode_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    """Read one JSON object and return its provenance record and diagnostic.

    Returns:
        Parsed object when valid, its source record, and an optional diagnostic code.
    """
    raw, record, problem = _verify_reference(
        ref, root, root_fd=root_fd, episode_id=episode_id, scope=scope
    )
    if raw is None:
        return None, record, problem
    try:
        payload = _strict_json_loads(raw)
    except ValueError:
        record = {**record, "content_status": "invalid_json"}
        return None, record, f"{record['artifact_id']}: source_unreadable"
    if not isinstance(payload, dict):
        record = {**record, "content_status": "not_json_object"}
        return None, record, f"{record['artifact_id']}: source_not_json_object"
    return payload, {**record, "content_status": "json_object"}, problem


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or None for malformed input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rank_candidates(
    episode_ids: list[str],
    scores: dict[str, float],
    pins: list[str],
    excludes: list[str],
    score_format: str = "unavailable",
) -> tuple[list[_Candidate], list[str]]:
    """Rank candidates with stable tie-breaking and explicit overrides.

    Returns:
        Kept candidates in deterministic order and any override diagnostics.
    """
    diagnostics: list[str] = []
    known = set(episode_ids)
    pin_set = set(pins)
    exclude_set = set(excludes)
    for conflict in sorted(pin_set & exclude_set):
        diagnostics.append(f"override_pin_exclude_conflict:{conflict}")
    for pinned in pins:
        if pinned not in known:
            diagnostics.append(f"override_pin_unknown:{pinned}")
    for excluded in excludes:
        if excluded not in known:
            diagnostics.append(f"override_exclude_unknown:{excluded}")
    candidates = [
        _Candidate(
            episode_id=episode_id,
            score=scores.get(episode_id),
            score_source=score_format if episode_id in scores else "unavailable",
            pinned=episode_id in pin_set,
            excluded=episode_id in exclude_set,
        )
        for episode_id in episode_ids
    ]
    kept = [candidate for candidate in candidates if not candidate.excluded]
    if excludes:
        diagnostics.append(f"excluded_episodes:{','.join(sorted(exclude_set & known))}")
    kept.sort(
        key=lambda candidate: (
            not candidate.pinned,
            -(candidate.score if candidate.score is not None else float("-inf")),
            candidate.episode_id,
        )
    )
    return kept, diagnostics


def _parse_overrides(
    config: dict[str, Any],
    bundle_doc: dict[str, Any],
    bundle_source_id: str,
    diagnostics: list[str],
) -> dict[str, Any]:
    """Parse overrides only when they are bound to the current bundle digest.

    Returns:
        Pin/exclude state, freshness, and declared/observed source digests.
    """
    observed = _canonical_bundle_digest(bundle_doc)
    raw = config.get("overrides", {})
    if not isinstance(raw, dict):
        diagnostics.append("overrides_malformed:ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": "",
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    pins_raw = raw.get("pin", [])
    excludes_raw = raw.get("exclude", [])
    if not isinstance(pins_raw, list) or not isinstance(excludes_raw, list):
        diagnostics.append("overrides_malformed:ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": str(raw.get("source_digest", "")),
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    if len(pins_raw) > MAX_OVERRIDE_IDS or len(excludes_raw) > MAX_OVERRIDE_IDS:
        diagnostics.append("overrides_limit_exceeded:ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": str(raw.get("source_digest", "")),
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    if any(not isinstance(item, str) for item in (*pins_raw, *excludes_raw)):
        diagnostics.append("overrides_malformed:ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": str(raw.get("source_digest", "")),
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    pins = list(pins_raw)
    excludes = list(excludes_raw)
    declared = str(raw.get("source_digest", ""))
    has_overrides = bool(pins or excludes)
    if has_overrides and not declared:
        diagnostics.append("override_source_digest_missing:overrides_ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": "",
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    if declared and declared.lower() != observed:
        diagnostics.append("stale_override_source:overrides_ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": declared,
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    return {
        "pin": pins,
        "exclude": excludes,
        "fresh": has_overrides,
        "declared_source_digest": declared,
        "observed_source_digest": observed,
        "source_artifact_id": bundle_source_id,
    }


def _canonical_bundle_digest(bundle_doc: dict[str, Any]) -> str:
    """Return the stable logical digest of a bundle document."""
    return _canonical_digest(bundle_doc)


def _clip_intervals(  # noqa: C901
    intervals: list[Any],
    duration_s: float,
    diagnostics: list[str],
    *,
    episode_ids: set[str],
) -> list[dict[str, Any]]:
    """Clip event rows while preserving episode-scoped interval identity.

    Returns:
        Deterministically clipped interval annotations with source identity.
    """
    if len(intervals) > MAX_INTERVALS:
        diagnostics.append("event_index_limit_exceeded:intervals")
        return []
    clipped: list[dict[str, Any]] = []
    seen_interval_ids: set[tuple[str, str]] = set()
    seen_event_ids: set[tuple[str, str]] = set()
    for position, item in enumerate(intervals):
        if not isinstance(item, dict):
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        start = _finite_number(item.get("start_s"))
        end = _finite_number(item.get("end_s"))
        if start is None or end is None or not end > start:
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        if start >= duration_s or end <= 0.0:
            diagnostics.append("interval_outside_duration")
            continue
        episode_id = item.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            diagnostics.append(f"event_row_{position}_episode_scope_missing")
            continue
        if episode_id not in episode_ids:
            diagnostics.append(f"event_row_{position}_unknown_episode:{episode_id}")
            continue
        interval_id = item.get("interval_id")
        if interval_id is not None and (not isinstance(interval_id, str) or not interval_id):
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        if isinstance(interval_id, str):
            scoped_interval_id = (episode_id, interval_id)
            if scoped_interval_id in seen_interval_ids:
                diagnostics.append(f"duplicate_interval_id:{episode_id}:{interval_id}")
                continue
            seen_interval_ids.add(scoped_interval_id)
        entry: dict[str, Any] = {
            "episode_id": episode_id,
            "start_s": max(start, 0.0),
            "end_s": min(end, duration_s),
            "source_row": position,
        }
        if isinstance(interval_id, str):
            entry["interval_id"] = interval_id
        event_id = item.get("event_id")
        if event_id is not None and (not isinstance(event_id, str) or not event_id):
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        if isinstance(event_id, str):
            scoped_event_id = (episode_id, event_id)
            if scoped_event_id in seen_event_ids:
                diagnostics.append(f"duplicate_event_id:{episode_id}:{event_id}")
                continue
            seen_event_ids.add(scoped_event_id)
            entry["event_id"] = event_id
        clipped.append(entry)
    return clipped


def _spec_intervals(
    index_doc: dict[str, Any] | None,
    duration: float | None,
    diagnostics: list[str],
    *,
    episode_ids: set[str],
) -> list[dict[str, Any]] | None:
    """Derive clipped event rows, retaining identity for annotation consumers.

    Returns:
        Clipped interval annotations, or ``None`` when the optional index is unavailable.
    """
    if index_doc is None or duration is None:
        return None
    raw = index_doc.get("intervals")
    if not isinstance(raw, list):
        diagnostics.append("event_index_intervals_missing")
        return None
    return _clip_intervals(raw, duration, diagnostics, episode_ids=episode_ids)


def _read_family_docs(
    by_format: dict[str, list[Any]],
    root: Path,
    diagnostics: list[str],
    *,
    root_fd: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Read each source family once and retain every source provenance record.

    Returns:
        Parsed family documents and all request-source provenance records.
    """
    docs: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for family in sorted(by_format):
        refs = by_format[family]
        if len(refs) != 1:
            diagnostics.append(f"{family}: source_family_collision")
            records.extend(_source_record(ref, integrity_status="family_collision") for ref in refs)
            continue
        payload, record, problem = _read_json(refs[0], root, root_fd=root_fd)
        records.append(record)
        if problem is not None:
            diagnostics.append(problem)
        if payload is not None:
            docs[family] = payload
    return docs, records


def _load_inputs(
    request: ComponentRequest,
    root: Path,
    diagnostics: list[str],
    *,
    root_fd: int | None = None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    dict[str, list[Any]],
]:
    """Load and validate bundle, scores, and event-index source families.

    Returns:
        Bundle, optional documents, source records, and grouped source references.
    """
    if len(request.sources) > MAX_REQUEST_SOURCES:
        diagnostics.append("request_limit_exceeded:sources")
        return None, None, None, [], {}
    by_format: dict[str, list[Any]] = {}
    records: list[dict[str, Any]] = []
    known_formats = set(REQUIRED_CAPABILITIES) | set(OPTIONAL_CAPABILITIES)
    for ref in request.sources:
        if ref.format not in known_formats:
            records.append(_source_record(ref, integrity_status="unsupported"))
            diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
            continue
        by_format.setdefault(ref.format, []).append(ref)
    for family in REQUIRED_CAPABILITIES:
        if family not in by_format:
            diagnostics.append(f"required_source_family_missing:{family}")
    if any(code.startswith("required_source_family_missing") for code in diagnostics):
        return None, None, None, records, by_format
    docs, loaded_records = _read_family_docs(by_format, root, diagnostics, root_fd=root_fd)
    records.extend(loaded_records)
    bundle_doc = docs.get("review-bundle")
    if bundle_doc is not None:
        try:
            review_bundle_from_dict(bundle_doc)
        except ReviewContractsValidationError as error:
            diagnostics.append("bundle_invalid:" + error.errors[0][:120])
            bundle_doc = None
    return bundle_doc, docs.get("exemplar-scores"), docs.get("event-index"), records, by_format


def _bundle_episode_index(  # noqa: C901, PLR0912, PLR0915
    bundle_doc: dict[str, Any],
    root: Path,
    diagnostics: list[str],
    *,
    root_fd: int | None = None,
) -> tuple[dict[str, tuple[dict[str, Any], ...]] | None, list[dict[str, Any]], bool]:
    """Validate episode identity and verify every referenced bundle artifact.

    Returns:
        Episode-to-reference mapping, bundle-reference records, and hard-failure state.
    """
    episode_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    records: list[dict[str, Any]] = []
    hard_failure = False
    episodes = bundle_doc["episodes"]
    if len(episodes) > MAX_BUNDLE_EPISODES:
        diagnostics.append("bundle_limit_exceeded:episodes")
        return None, records, True
    total_references = sum(len(episode["references"]) for episode in episodes)
    if total_references > MAX_TOTAL_BUNDLE_REFERENCES:
        diagnostics.append("bundle_limit_exceeded:references_total")
        return None, records, True
    total_reads = 0
    total_bytes = 0
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id in episode_by_id:
            diagnostics.append(f"duplicate_episode_id:{episode_id}")
            hard_failure = True
            continue
        raw_refs = episode["references"]
        if len(raw_refs) > MAX_EPISODE_REFERENCES:
            diagnostics.append(f"bundle_limit_exceeded:references:{episode_id}")
            return None, records, True
        refs = tuple(dict(ref) for ref in raw_refs)
        episode_by_id[episode_id] = refs
        verified_refs: list[dict[str, Any]] = []
        for ref in refs:
            if total_reads >= MAX_TOTAL_BUNDLE_READS:
                diagnostics.append("bundle_limit_exceeded:reference_reads_total")
                hard_failure = True
                break
            remaining_bytes = MAX_TOTAL_BUNDLE_BYTES - total_bytes
            if remaining_bytes <= 0:
                diagnostics.append("bundle_limit_exceeded:reference_bytes_total")
                hard_failure = True
                break
            total_reads += 1
            raw, record, problem = _verify_reference(
                ref,
                root,
                root_fd=root_fd,
                episode_id=episode_id,
                scope="bundle-reference",
                max_bytes=min(MAX_SOURCE_BYTES, remaining_bytes),
            )
            records.append(record)
            if raw is not None:
                if len(raw) > remaining_bytes:
                    diagnostics.append("bundle_limit_exceeded:reference_bytes_total")
                    hard_failure = True
                    break
                total_bytes += len(raw)
            elif (
                record.get("read_status") == "source_too_large"
                and remaining_bytes < MAX_SOURCE_BYTES
            ):
                diagnostics.append("bundle_limit_exceeded:reference_bytes_total")
                hard_failure = True
            if problem is not None:
                diagnostics.append(
                    problem.replace("source_uri_unsafe", "source_reference_uri_unsafe")
                )
                if record["integrity_status"] != "verified":
                    hard_failure = True
            if raw is not None and record["integrity_status"] == "verified":
                content_problem = _validate_episode_reference(ref, raw, episode_id)
                if content_problem is not None:
                    diagnostics.append(content_problem)
                    record["content_status"] = "invalid_or_mismatched"
                    hard_failure = True
                else:
                    if ref["format"] == "episode-json":
                        record["content_status"] = "validated_json"
                    verified_refs.append(ref)
        if hard_failure:
            break
        if len(verified_refs) != len(refs):
            hard_failure = True
    if hard_failure:
        return None, records, True
    return episode_by_id, records, False


def _family_records(
    records: list[dict[str, Any]], family: str, *, scope: str = "request"
) -> list[dict[str, Any]]:
    """Select records belonging to one source family and scope.

    Returns:
        Matching source provenance records.
    """
    return [
        record for record in records if record["format"] == family and record.get("scope") == scope
    ]


def _family_integrity_verified(records: list[dict[str, Any]], family: str) -> bool:
    """Return whether exactly one family source has verified bytes.

    Returns:
        Whether one and only one request source for the family is byte-verified.
    """
    family_records = _family_records(records, family)
    return len(family_records) == 1 and family_records[0]["integrity_status"] == "verified"


def _validate_episode_reference(ref: dict[str, Any], raw: bytes, episode_id: str) -> str | None:
    """Validate JSON episode references and bind their optional identity fields.

    Returns:
        A stable diagnostic code, or ``None`` when the content is valid.
    """
    if ref["format"] != "episode-json":
        return None
    try:
        payload = _strict_json_loads(raw)
    except ValueError:
        return f"{ref['artifact_id']}: source_unreadable"
    if not isinstance(payload, dict):
        return f"{ref['artifact_id']}: source_not_json_object"
    if "artifact_id" in payload and payload["artifact_id"] != ref["artifact_id"]:
        return f"{ref['artifact_id']}: source_artifact_id_mismatch"
    if "episode_id" in payload and payload["episode_id"] != episode_id:
        return f"{ref['artifact_id']}: source_episode_id_mismatch"
    return None


def _score_entries(
    entries: list[Any], known: set[str], diagnostics: list[str]
) -> tuple[dict[str, float], bool]:
    """Parse canonical trace-exemplar-interest episode entries.

    Returns:
        Finite scores keyed by known episode ID and whether every row was valid.
    """
    scores: dict[str, float] = {}
    valid = True
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            diagnostics.append(f"exemplar_scores_row_malformed:{position}")
            valid = False
            continue
        episode_id = entry.get("episode_id")
        number = _finite_number(entry.get("composite_score"))
        if not isinstance(episode_id, str) or number is None:
            diagnostics.append(f"exemplar_scores_row_malformed:{position}")
            valid = False
            continue
        if not 0.0 <= number <= 1.0:
            diagnostics.append(f"score_out_of_range:{episode_id}")
            valid = False
            continue
        if episode_id not in known:
            diagnostics.append(f"score_unknown_episode:{episode_id}")
            valid = False
            continue
        if episode_id in scores:
            if scores[episode_id] != number:
                diagnostics.append(f"score_conflicting_duplicate:{episode_id}")
            else:
                diagnostics.append(f"score_duplicate_episode:{episode_id}")
            valid = False
            continue
        scores[episode_id] = number
    return scores, valid


def _score_map(
    raw_map: dict[str, Any], known: set[str], diagnostics: list[str]
) -> tuple[dict[str, float], bool]:
    """Parse an explicitly versioned SREV-07 score map.

    Returns:
        Finite scores keyed by known episode ID and whether every row was valid.
    """
    scores: dict[str, float] = {}
    valid = True
    for episode_id, value in raw_map.items():
        if not isinstance(episode_id, str):
            diagnostics.append("score_malformed:episode_id")
            valid = False
            continue
        number = _finite_number(value)
        if episode_id not in known:
            diagnostics.append(f"score_unknown_episode:{episode_id}")
            valid = False
            continue
        if number is None:
            diagnostics.append(f"score_malformed:{episode_id}")
            valid = False
            continue
        if not 0.0 <= number <= 1.0:
            diagnostics.append(f"score_out_of_range:{episode_id}")
            valid = False
            continue
        scores[episode_id] = number
    return scores, valid


def _record_missing_scores(
    scores: dict[str, float], known: set[str], diagnostics: list[str]
) -> None:
    """Record known bundle episodes omitted by a score document."""
    for episode_id in sorted(known - set(scores)):
        diagnostics.append(f"score_missing:{episode_id}")


def _canonical_non_admissible_status(status: Any) -> str | None:
    """Return a normalized execution status that cannot support canonical scoring."""
    if not isinstance(status, str) or not status.strip():
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", status.strip().lower()).strip("_")
    if normalized in _CANONICAL_NON_ADMISSIBLE_STATUS_TOKENS:
        return normalized
    if set(normalized.split("_")) & _CANONICAL_NON_ADMISSIBLE_STATUS_TOKENS:
        return normalized
    return None


def _canonical_score_document_valid(  # noqa: C901, PLR0912, PLR0915
    scores_doc: dict[str, Any], known: set[str], diagnostics: list[str]
) -> bool:
    """Validate the owner-native trace-exemplar-interest report shape and value semantics.

    The adapter is deliberately strict: a document is canonical only when it
    has the complete owner report envelope, complete episode rows, normalized
    feature values, and bounded composite scores.  This prevents a lookalike
    ``episodes`` list from being advertised as the canonical adapter.

    Returns:
        Whether the document is safe to identify as the canonical owner format.
    """
    if set(scores_doc) != CANONICAL_SCORE_REPORT_FIELDS:
        diagnostics.append("exemplar_scores_canonical_shape_mismatch")
        return False
    roots = scores_doc["roots"]
    weights = scores_doc["weights"]
    episodes = scores_doc["episodes"]
    comparison_pairs = scores_doc["comparison_pairs"]
    valid = True
    if not isinstance(roots, list) or any(not isinstance(root, str) or not root for root in roots):
        diagnostics.append("exemplar_scores_canonical_field_invalid:roots")
        valid = False
    if not isinstance(weights, dict) or set(weights) != CANONICAL_SCORE_FEATURES:
        diagnostics.append("exemplar_scores_canonical_field_invalid:weights")
        valid = False
    elif (
        any(_finite_number(value) is None or float(value) < 0.0 for value in weights.values())
        or sum(float(value) for value in weights.values()) <= 0.0
    ):
        diagnostics.append("exemplar_scores_canonical_field_invalid:weights")
        valid = False
    if not isinstance(episodes, list):
        diagnostics.append("exemplar_scores_canonical_field_invalid:episodes")
        return False
    if len(episodes) > MAX_SCORE_ENTRIES:
        diagnostics.append("exemplar_scores_limit_exceeded:episodes")
        return False
    seen_episode_ids: set[str] = set()
    for position, entry in enumerate(episodes):
        if not isinstance(entry, dict) or set(entry) != CANONICAL_SCORE_EPISODE_FIELDS:
            diagnostics.append(f"exemplar_scores_canonical_row_malformed:{position}")
            valid = False
            continue
        episode_id = entry["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            diagnostics.append(f"exemplar_scores_canonical_row_malformed:{position}")
            valid = False
        elif episode_id in seen_episode_ids:
            diagnostics.append(f"score_duplicate_episode:{episode_id}")
            valid = False
        else:
            seen_episode_ids.add(episode_id)
            if episode_id not in known:
                diagnostics.append(f"score_unknown_episode:{episode_id}")
                valid = False
        for field_name in ("episode_dir", "episode_status", "planner", "scenario_id"):
            if not isinstance(entry[field_name], str) or not entry[field_name]:
                diagnostics.append(f"exemplar_scores_canonical_row_malformed:{position}")
                valid = False
        non_admissible_status = _canonical_non_admissible_status(entry["episode_status"])
        if non_admissible_status is not None:
            diagnostics.append(
                f"exemplar_scores_canonical_row_non_admissible:{position}:{non_admissible_status}"
            )
            valid = False
        seed = entry["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            diagnostics.append(f"exemplar_scores_canonical_row_malformed:{position}")
            valid = False
        features = entry["features"]
        if not isinstance(features, dict) or set(features) != CANONICAL_SCORE_FEATURES:
            diagnostics.append(f"exemplar_scores_canonical_features_malformed:{position}")
            valid = False
        elif any(
            _finite_number(value) is None or not 0.0 <= float(value) <= 1.0
            for value in features.values()
        ):
            diagnostics.append(f"exemplar_scores_canonical_features_out_of_range:{position}")
            valid = False
        composite_score = _finite_number(entry["composite_score"])
        if composite_score is None or not 0.0 <= composite_score <= 1.0:
            diagnostics.append(f"score_out_of_range:{episode_id}")
            valid = False
    if not isinstance(comparison_pairs, list):
        diagnostics.append("exemplar_scores_canonical_field_invalid:comparison_pairs")
        return False
    if len(comparison_pairs) > MAX_SCORE_ENTRIES:
        diagnostics.append("exemplar_scores_limit_exceeded:comparison_pairs")
        return False
    for position, pair in enumerate(comparison_pairs):
        if not isinstance(pair, dict) or set(pair) != CANONICAL_SCORE_PAIR_FIELDS:
            diagnostics.append(f"exemplar_scores_canonical_pair_malformed:{position}")
            valid = False
            continue
        for field_name in (
            "scenario_id",
            "left_episode_id",
            "left_planner",
            "right_episode_id",
            "right_planner",
        ):
            if not isinstance(pair[field_name], str) or not pair[field_name]:
                diagnostics.append(f"exemplar_scores_canonical_pair_malformed:{position}")
                valid = False
        for field_name in ("left_episode_id", "right_episode_id"):
            episode_id = pair[field_name]
            if isinstance(episode_id, str) and episode_id not in known:
                diagnostics.append(
                    "exemplar_scores_canonical_pair_unknown_episode:"
                    f"{position}:{field_name}:{episode_id}"
                )
                valid = False
        seed = pair["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            diagnostics.append(f"exemplar_scores_canonical_pair_malformed:{position}")
            valid = False
        if any(
            _finite_number(pair[field_name]) is None or not 0.0 <= float(pair[field_name]) <= 1.0
            for field_name in ("outcome_divergence", "trajectory_divergence", "pair_score")
        ):
            diagnostics.append(f"exemplar_scores_canonical_pair_out_of_range:{position}")
            valid = False
    return valid


def _score_table(
    scores_doc: dict[str, Any] | None, episode_ids: list[str], diagnostics: list[str]
) -> tuple[dict[str, float], str]:
    """Extract only versioned or canonical-owner exemplar scores.

    Missing scores remain unavailable (``None`` in the output ranking); they
    never become an implicit numeric default.

    Returns:
        Parsed scores and the explicit input/adaptor format identifier.
    """
    if scores_doc is None:
        diagnostics.append("exemplar_scores_unavailable")
        return {}, "unavailable"
    known = set(episode_ids)
    if "episodes" in scores_doc:
        raw_entries = scores_doc["episodes"]
        if not isinstance(raw_entries, list):
            diagnostics.append("exemplar_scores_canonical_shape_mismatch")
            return {}, "unavailable"
        if len(raw_entries) > MAX_SCORE_ENTRIES:
            diagnostics.append("exemplar_scores_limit_exceeded:episodes")
            return {}, "unavailable"
        if not _canonical_score_document_valid(scores_doc, known, diagnostics):
            return {}, "unavailable"
        scores, valid = _score_entries(raw_entries, known, diagnostics)
        _record_missing_scores(scores, known, diagnostics)
        if not valid:
            return {}, "unavailable"
        return scores, CANONICAL_SCORE_ADAPTER_VERSION
    raw_map = scores_doc.get("scores")
    if (
        set(scores_doc) != {"schema_version", "scores"}
        or scores_doc.get("schema_version") != EXEMPLAR_SCORES_SCHEMA_VERSION
        or not isinstance(raw_map, dict)
    ):
        diagnostics.append("exemplar_scores_unversioned_or_malformed")
        return {}, "unavailable"
    if len(raw_map) > MAX_SCORE_ENTRIES:
        diagnostics.append("exemplar_scores_limit_exceeded:scores")
        return {}, "unavailable"
    scores, valid = _score_map(raw_map, known, diagnostics)
    _record_missing_scores(scores, known, diagnostics)
    if not valid:
        return {}, "unavailable"
    return scores, EXEMPLAR_SCORES_SCHEMA_VERSION


def _blocking_diagnostics(
    diagnostics: list[str],
    *,
    requested_capabilities: tuple[str, ...],
    score_source_present: bool,
) -> list[str]:
    """Classify diagnostics without treating an absent optional stream as failure.

    Returns:
        Diagnostics that make the result partial rather than complete.
    """
    blocking: list[str] = []
    for item in diagnostics:
        if item.startswith("excluded_episodes:"):
            continue
        if item == "exemplar_scores_unavailable" and not score_source_present:
            if "exemplar-scores" not in requested_capabilities:
                continue
        blocking.append(item)
    return blocking


def _request_identity(request: Any) -> tuple[str, str]:
    """Return contract-safe identity fields for an API failure result."""
    if isinstance(request, dict):
        request_id = request.get("request_id")
        component_id = request.get("component_id")
    else:
        request_id = getattr(request, "request_id", None)
        component_id = getattr(request, "component_id", None)
    return (
        request_id if isinstance(request_id, str) and request_id else "unknown",
        component_id if isinstance(component_id, str) and component_id else COMPONENT_ID,
    )


def _request_shape_error(request: Any) -> str | None:  # noqa: C901
    """Return a stable error for a manually constructed invalid API request."""
    if not isinstance(request, ComponentRequest):
        return "invalid_request: expected ComponentRequest"
    if not isinstance(request.request_id, str) or not request.request_id:
        return "invalid_request: request_id must be a non-empty string"
    if not isinstance(request.component_id, str) or not request.component_id:
        return "invalid_request: component_id must be a non-empty string"
    if not isinstance(request.output_directory, str) or not request.output_directory:
        return "invalid_request: output_directory must be a non-empty string"
    if not isinstance(request.config, dict):
        return "invalid_request: config must be an object"
    if not isinstance(request.sources, (tuple, list)):
        return "invalid_request: sources must be a sequence"
    if not isinstance(request.required_capabilities, (tuple, list)) or any(
        not isinstance(capability, str) for capability in request.required_capabilities
    ):
        return "invalid_request: required_capabilities must be strings"
    for source in request.sources:
        if not isinstance(source, SourceRef):
            return "invalid_request: sources require SourceRef values"
        if any(
            not isinstance(getattr(source, field_name, None), str)
            for field_name in (
                "artifact_id",
                "uri",
                "format",
                "schema",
                "sha256",
                "source_commit",
                "config_identity",
                "units",
                "coordinate_frame",
            )
        ):
            return "invalid_request: source metadata must be strings"
        if not source.artifact_id or not source.uri or not source.format:
            return "invalid_request: sources require non-empty artifact_id, uri, and format"
        if source.sha256 and _SHA256_RE.fullmatch(source.sha256) is None:
            return "invalid_request: source sha256 must be a 64-hex digest"
        if source.source_commit and _SHA40_RE.fullmatch(source.source_commit) is None:
            return "invalid_request: source_commit must be a 40-hex commit SHA"
    return None


def run(  # noqa: C901, PLR0912, PLR0915
    request: ComponentRequest, *, base: Path | None = None
) -> ComponentResult:
    """Rank storyboard candidates and emit deterministic, provenance-bound sidecars.

    Returns:
        A versioned component result describing status, sidecars, and provenance.
    """
    request_id, component_id = _request_identity(request)
    shape_error = _request_shape_error(request)
    if shape_error is not None:
        return _result(
            request_id=request_id,
            component_id=component_id,
            status=STATUS_FAILED,
            reason=shape_error,
        )
    try:
        root = (base if base is not None else Path.cwd()).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return _result(
            request_id=request_id,
            component_id=component_id,
            status=STATUS_FAILED,
            reason="invalid_request: base path cannot be resolved",
        )
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    output_parts = _safe_relative_parts(request.output_directory)
    if output_parts is None:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=f"unsafe_output_path: {request.output_directory}",
        )
    try:
        root_fd = _open_directory_anchor(root)
    except _SecurePathError as error:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=error.reason,
        )
    except OSError:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="invalid_request: base path cannot be securely opened",
        )
    diagnostics: list[str] = []
    try:
        bundle_doc, scores_doc, index_doc, source_records, refs_by_format = _load_inputs(
            request, root, diagnostics, root_fd=root_fd
        )
        if bundle_doc is None:
            return _result(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                reason="required_source_family_unusable: "
                + "; ".join(sorted(set(diagnostics))[:5]),
            )
        bundle_records = _family_records(source_records, "review-bundle")
        if not _family_integrity_verified(source_records, "review-bundle"):
            diagnostics.append("required_source_integrity_unverified:review-bundle")
            return _result(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                reason="required_source_integrity_unverified: review-bundle; "
                + "; ".join(sorted(set(diagnostics))[:5]),
            )
        episode_by_id, bundle_reference_records, bundle_hard_failure = _bundle_episode_index(
            bundle_doc, root, diagnostics, root_fd=root_fd
        )
        source_records.extend(bundle_reference_records)
        if bundle_hard_failure or episode_by_id is None:
            return _result(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance={
                    "source_provenance": source_records,
                    "evidence_boundary": EVIDENCE_BOUNDARY,
                },
                reason="; ".join(sorted(set(diagnostics))[:8]),
            )

        episode_ids = list(episode_by_id)
        score_source_present = bool(refs_by_format.get("exemplar-scores"))
        event_source_present = bool(refs_by_format.get("event-index"))
        available_capabilities = ["review-bundle"]
        score_usable = scores_doc is not None and _family_integrity_verified(
            source_records, "exemplar-scores"
        )
        event_usable = index_doc is not None and _family_integrity_verified(
            source_records, "event-index"
        )
        if score_usable:
            available_capabilities.append("exemplar-scores")
        elif score_source_present:
            diagnostics.append("source_integrity_unverified:exemplar-scores")
        if event_usable:
            available_capabilities.append("event-index")
        elif event_source_present:
            diagnostics.append("source_integrity_unverified:event-index")
        missing_capabilities = [
            capability
            for capability in request.required_capabilities
            if capability not in available_capabilities
        ]
        for capability in missing_capabilities:
            if f"required_capability_missing:{capability}" not in diagnostics:
                diagnostics.append(f"required_capability_missing:{capability}")

        scores, score_format = _score_table(
            scores_doc if score_usable else None, episode_ids, diagnostics
        )
        bundle_source_id = str(bundle_records[0]["artifact_id"])
        overrides = _parse_overrides(request.config, bundle_doc, bundle_source_id, diagnostics)
        kept, rank_diagnostics = _rank_candidates(
            episode_ids,
            scores,
            overrides["pin"],
            overrides["exclude"],
            score_format,
        )
        diagnostics.extend(rank_diagnostics)
        duration = _finite_number(request.config.get("source_duration_s"))
        if duration is None or duration <= 0:
            diagnostics.append("source_duration_missing_or_invalid")
            duration = None
        event_intervals = _spec_intervals(
            index_doc if event_usable else None,
            duration,
            diagnostics,
            episode_ids=set(episode_ids),
        )
        selected_episode_ids = [candidate.episode_id for candidate in kept]
        selected_source_artifact_ids = [
            str(ref["artifact_id"])
            for episode_id in selected_episode_ids
            for ref in episode_by_id[episode_id]
        ]
        spec_payload: dict[str, Any] = {
            "schema_version": "visualization-spec.v1",
            "spec_id": f"{request.request_id}-storyboard",
            "sources": [
                {"artifact_id": artifact_id} for artifact_id in selected_source_artifact_ids
            ],
            "annotations": [
                {
                    "ranking": [
                        {
                            "episode_id": candidate.episode_id,
                            "source_artifact_ids": [
                                str(ref["artifact_id"])
                                for ref in episode_by_id[candidate.episode_id]
                            ],
                            "score": candidate.score,
                            "score_source": candidate.score_source,
                            "pinned": candidate.pinned,
                        }
                        for candidate in kept
                    ],
                    "tie_break_method": TIE_BREAK_METHOD,
                    "source_binding": {
                        "selected_episode_ids": selected_episode_ids,
                        "selected_source_artifact_ids": selected_source_artifact_ids,
                    },
                }
            ],
        }
        if event_intervals is not None:
            spec_payload["source_intervals"] = [
                {
                    "start_s": interval["start_s"],
                    "end_s": interval["end_s"],
                    "include_terminal_frame": False,
                }
                for interval in event_intervals
            ]
            spec_payload["annotations"][0]["event_intervals"] = event_intervals
        visualization_spec_from_dict(spec_payload)
        _assert_anchor_identity(root, root_fd, "source_root_changed_during_run")

        blocking = _blocking_diagnostics(
            diagnostics,
            requested_capabilities=request.required_capabilities,
            score_source_present=score_source_present,
        )
        status = STATUS_PARTIAL if blocking else STATUS_COMPLETE
        diagnostics_payload = sorted(set(diagnostics))
        override_payload = {
            "schema_version": "override-provenance.v1",
            "result_status": status,
            "source_artifact_id": overrides["source_artifact_id"],
            "declared_source_digest": overrides["declared_source_digest"],
            "observed_source_digest": overrides["observed_source_digest"],
            "pins": sorted(set(overrides["pin"])),
            "excludes": sorted(set(overrides["exclude"])),
            "overrides_applied": overrides["fresh"],
            "diagnostics": diagnostics_payload,
        }
        capability_payload = {
            "schema_version": MISSING_CAPABILITY_SCHEMA_VERSION,
            "result_status": status,
            "requested_capabilities": list(request.required_capabilities),
            "available_capabilities": sorted(available_capabilities),
            "missing_capabilities": sorted(set(missing_capabilities)),
            "optional_capabilities_unavailable": sorted(
                set(OPTIONAL_CAPABILITIES) - set(available_capabilities)
            ),
            "diagnostics": diagnostics_payload,
        }
        payloads = {
            OUTPUT_CAPABILITY_FILENAME: capability_payload,
            OUTPUT_OVERRIDES_FILENAME: override_payload,
            OUTPUT_SPEC_FILENAME: spec_payload,
        }
        file_digests = _stage_outputs(root_fd, output_parts, payloads)
        output_artifacts = tuple(
            {
                "artifact_id": filename,
                "uri": (Path(request.output_directory) / filename).as_posix(),
                "sha256": file_digests[filename],
            }
            for filename in sorted(payloads)
        )
        provenance = {
            "output_directory": request.output_directory,
            "component_version": COMPONENT_VERSION,
            "bundle_artifact_id": bundle_source_id,
            "bundle_logical_digest": _canonical_bundle_digest(bundle_doc),
            "selected_episode_ids": selected_episode_ids,
            "selected_source_artifact_ids": selected_source_artifact_ids,
            "source_provenance": source_records,
            "requested_capabilities": list(request.required_capabilities),
            "available_capabilities": sorted(available_capabilities),
            "missing_capabilities": sorted(set(missing_capabilities)),
            "score_input_format": score_format,
            "score_adapter_version": (
                CANONICAL_SCORE_ADAPTER_VERSION
                if score_format == CANONICAL_SCORE_ADAPTER_VERSION
                else ""
            ),
            "emitted_artifacts": [dict(item) for item in output_artifacts],
            "evidence_boundary": EVIDENCE_BOUNDARY,
        }
        result_artifacts = output_artifacts if status == STATUS_COMPLETE else ()
        reason = "" if status == STATUS_COMPLETE else "; ".join(blocking[:8])
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=result_artifacts,
            diagnostics=tuple({"code": item} for item in diagnostics_payload),
            provenance=provenance,
            reason=reason,
        )
    except _OutputCollisionError:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason=f"output_collision: already exists: {request.output_directory}",
        )
    except _SecurePathError as error:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason=error.reason,
        )
    except FileExistsError:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason=f"output_collision: already exists: {request.output_directory}",
        )
    except ReviewContractsValidationError as error:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason="; ".join(error.errors),
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason=f"internal_failure: {type(error).__name__}",
        )
    finally:
        os.close(root_fd)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-storyboard component."""
    parser = argparse.ArgumentParser(description="Rank storyboard candidates.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def _load_cli_json(path: Path, label: str) -> Any:
    """Read one bounded, regular, strict-JSON CLI input.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If the path cannot be read as strict JSON.
    """
    raw, read_problem = _read_regular_file(path)
    if read_problem is not None or raw is None:
        raise ValueError(f"invalid_input: {label} JSON cannot be parsed safely")
    try:
        return _strict_json_loads(raw)
    except ValueError as error:
        raise ValueError(f"invalid_input: {label} JSON cannot be parsed safely") from error


def _print_cli_failure(reason: str, *, payload: Any = None) -> int:
    """Print a contract-valid failed result for pre-request CLI errors.

    Returns:
        The CLI failure exit code.
    """
    request_id, component_id = _request_identity(payload)
    result = ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=STATUS_FAILED,
        reason=reason,
    )
    print(json.dumps(result_document(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and print a schema-valid result envelope.

    Returns:
        Zero for a complete result, otherwise one.
    """
    args = _build_parser().parse_args(argv)
    payload: Any = None
    try:
        payload = _load_cli_json(Path(args.input), "request")
    except (OSError, TypeError, ValueError, RecursionError):
        return _print_cli_failure("invalid_input: request JSON cannot be parsed safely")
    if not isinstance(payload, dict):
        return _print_cli_failure("invalid_input: request must be a JSON object")
    if args.config is not None:
        try:
            config = _load_cli_json(Path(args.config), "config")
        except (OSError, TypeError, ValueError, RecursionError):
            return _print_cli_failure(
                "invalid_input: config JSON cannot be parsed safely", payload=payload
            )
        if not isinstance(config, dict):
            return _print_cli_failure(
                "invalid_input: config must be a JSON object", payload=payload
            )
        request_config = payload.get("config", {})
        if not isinstance(request_config, dict):
            return _print_cli_failure(
                "invalid_input: request config must be a JSON object", payload=payload
            )
        payload = {**payload, "config": {**request_config, **config}}
    payload = {**payload, "output_directory": args.output}
    try:
        request = component_request_from_dict(payload, source=args.input)
    except (ReviewContractsValidationError, RecursionError, TypeError, ValueError):
        return _print_cli_failure(
            "invalid_input: request does not satisfy component-request.v1", payload=payload
        )
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(result_document(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
