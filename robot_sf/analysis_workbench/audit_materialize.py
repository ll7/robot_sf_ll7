"""Source-first, lazy materialization for the Benchmark Auditor.

This module is deliberately a small BA-05 leaf.  It resolves an explicitly
declared recording, verifies the bytes and available episode identities, and
returns a historical-original reference without copying or replaying it.  If
the original is unavailable, it can render *retained* trace/replay state using
the existing SREV scene or replay figure renderer.  It never constructs or
advances a simulator.  Exact-input diagnostic execution is an explicit
``unavailable`` result in this slice and belongs to a later adapter.

The result is diagnostic-only.  A derived render is not a historical
recording, and an output from a later diagnostic execution must not be
classified as either one by this module.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import imageio_ffmpeg

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ComponentResult,
    SourceRef,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExportValidationError,
    simulation_trace_export_from_dict,
)
from robot_sf.benchmark.analysis_trace import (
    ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
    trace_artifact_sha256,
)
from robot_sf.benchmark.episode_replay_figure import (
    generate_trajectory,
)
from robot_sf.benchmark.full_classic.replay import ReplayEpisode, ReplayStep
from robot_sf.render import review_scene

MATERIALIZATION_SCHEMA_VERSION = "audit-materialization.v1"
MATERIALIZATION_TOOL_VERSION = "audit_materialize.v1"
HISTORICAL_ORIGINAL = "historical_original"
DERIVED_RENDER = "derived_render"
UNAVAILABLE = "unavailable"

FIDELITY_VERIFIED = "verified"
FIDELITY_DIVERGED = "diverged"
FIDELITY_UNVERIFIABLE = "unverifiable"
FIDELITY_UNAVAILABLE = "unavailable"

MATERIALIZATION_STATUSES = ("complete", "partial", "unavailable", "failed")
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_STATES = 4096
MAX_TEXT = 256
MAX_OUTPUT_DIRECTORY = 4096

_SHA256_HEX = frozenset("0123456789abcdefABCDEF")
_IDENTITY_FIELDS = (
    "episode_id",
    "campaign_id",
    "execution_id",
    "scenario_id",
    "source_digest",
    "config_identity",
    "source_commit",
    "checkpoint_digest",
    "environment_digest",
    "initial_state_digest",
)
_IDENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "episode_id": ("episode_id",),
    "campaign_id": ("campaign_id", "campaign", "study_id"),
    "execution_id": ("execution_id", "run_id"),
    "scenario_id": ("scenario_id", "scenario"),
    "source_digest": (
        "recording_digest",
        "recording_sha256",
        "source_digest",
        "sha256",
    ),
    "config_identity": ("config_identity", "config_digest", "config_hash"),
    "source_commit": ("source_commit", "repo_commit", "commit_sha", "commit"),
    "checkpoint_digest": ("checkpoint_digest", "checkpoint_hash"),
    "environment_digest": ("environment_digest", "environment_hash"),
    "initial_state_digest": ("initial_state_digest", "initial_state_hash"),
}


class MaterializationValidationError(ValueError):
    """Raised for a malformed request that cannot be classified safely."""


@dataclass(frozen=True, slots=True)
class MaterializationRequest:
    """Bounded request used by :func:`materialize_episode`.

    ``episode`` is the selected retained audit row.  ``source_root`` and
    ``output_root`` are caller-owned trust boundaries; source URIs and output
    directories are interpreted only relative to them.
    """

    episode: Mapping[str, Any]
    source_root: str | Path | MaterializationSourceRoot
    output_root: str | Path | MaterializationOutputRoot
    output_directory: str | None = None
    render_config: Mapping[str, Any] | None = None


@dataclass(slots=True)
class MaterializationOutputRoot:
    """Descriptor-bound output-root capability issued by a trusted service.

    The capability retains an allowed-root descriptor rather than handing a
    path-only policy decision to the renderer.  A missing output root is
    created later, relative to that retained descriptor, with no-follow
    directory opens.  The service owns the capability and must call
    :meth:`close` after materialization returns.
    """

    root: Path
    allowed_root_fd: int
    allowed_root_identity: tuple[int, int]
    relative_parts: tuple[str, ...]
    root_fd: int = -1
    root_identity: tuple[int, int] | None = None

    def open_root(self) -> tuple[int, tuple[int, int]]:
        """Return a descriptor for the admitted root, creating it safely."""

        allowed_stat = os.fstat(self.allowed_root_fd)
        if not stat.S_ISDIR(allowed_stat.st_mode):
            raise OSError("allowed output root is not a directory")
        if (allowed_stat.st_dev, allowed_stat.st_ino) != self.allowed_root_identity:
            raise OSError("allowed output root changed before materialization")
        if self.root_fd >= 0:
            root_fd = os.dup(self.root_fd)
        else:
            root_fd = _walk_output_from_fd(
                self.allowed_root_fd,
                self.relative_parts,
                create=True,
            )
        try:
            root_stat = os.fstat(root_fd)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise OSError("output root is not a directory")
            identity = (root_stat.st_dev, root_stat.st_ino)
            if self.root_identity is None:
                self.root_identity = identity
            elif self.root_identity != identity:
                raise OSError("output root changed before materialization")
            return root_fd, identity
        except BaseException:
            os.close(root_fd)
            raise

    def close(self) -> None:
        """Close the capability descriptors without masking adapter errors."""

        for attribute in ("root_fd", "allowed_root_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
            finally:
                setattr(self, attribute, -1)


@dataclass(slots=True)
class MaterializationSourceRoot:
    """Descriptor-bound source-root capability issued by a trusted service."""

    root: Path
    root_fd: int
    root_identity: tuple[int, int]

    def open_root(self) -> tuple[int, tuple[int, int]]:
        """Return a duplicate descriptor after verifying the retained identity."""

        root_stat = os.fstat(self.root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise OSError("source root is not a directory")
        identity = (root_stat.st_dev, root_stat.st_ino)
        if identity != self.root_identity:
            raise OSError("source root changed before materialization")
        return os.dup(self.root_fd), identity

    def close(self) -> None:
        """Close the retained source-root descriptor without masking errors."""

        if self.root_fd < 0:
            return
        try:
            os.close(self.root_fd)
        except OSError:
            pass
        finally:
            self.root_fd = -1


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """A source-bound, diagnostic-only materialization result."""

    status: str
    materialization_kind: str
    fidelity: str
    episode_id: str
    cache_key: str
    source_digest: str = ""
    output_directory: str | None = None
    artifacts: tuple[Mapping[str, Any], ...] = ()
    diagnostics: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    simulation_executed: bool = False

    @property
    def classification(self) -> str:
        """Return the explicit user-facing historical/derived/unavailable label."""

        return self.materialization_kind

    @property
    def ok(self) -> bool:
        """Return whether a complete or partial artifact was produced."""

        return self.status in {"complete", "partial"} and self.materialization_kind != UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe result envelope."""

        return {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": self.status,
            "materialization_kind": self.materialization_kind,
            "classification": self.materialization_kind,
            "fidelity": self.fidelity,
            "episode_id": self.episode_id,
            "cache_key": self.cache_key,
            "source_digest": self.source_digest,
            "output_directory": self.output_directory,
            "artifacts": [dict(item) for item in self.artifacts],
            "diagnostics": list(self.diagnostics),
            "provenance": dict(self.provenance),
            "reason": self.reason,
            "simulation_executed": self.simulation_executed,
            "diagnostic_only": True,
        }


@dataclass(frozen=True, slots=True)
class _ReadBytes:
    """Bytes read from one protected regular-file descriptor."""

    path: Path
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _OriginalRecording:
    """Verified original recording and its identity status."""

    read: _ReadBytes
    identity: Mapping[str, Any]
    descriptor: Mapping[str, Any]
    uri: str


def _bounded_text(value: Any, *, name: str, required: bool = False) -> str:
    """Normalize bounded textual request fields without accepting whitespace.

    Returns:
        The trimmed value, or an empty string for an omitted optional field.
    """

    if not isinstance(value, str):
        if required:
            raise MaterializationValidationError(f"{name} must be a string")
        return ""
    value = value.strip()
    if required and not value:
        raise MaterializationValidationError(f"{name} must not be blank")
    if len(value) > MAX_TEXT:
        raise MaterializationValidationError(f"{name} exceeds {MAX_TEXT} characters")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Copy a mapping or raise a bounded validation error.

    Returns:
        A mutable shallow copy of ``value``.
    """

    if not isinstance(value, Mapping):
        raise MaterializationValidationError(f"{name} must be an object")
    return dict(value)


def _canonical_bytes(value: Any) -> bytes:
    """Serialize logical input with strict JSON semantics for identity hashes.

    Returns:
        Canonical UTF-8 JSON bytes.
    """

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise MaterializationValidationError(f"input is not strict JSON: {error}") from error


def _sha256(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: Any) -> bool:
    """Return whether ``value`` is a 64-character hexadecimal digest."""

    return (
        isinstance(value, str) and len(value) == 64 and all(char in _SHA256_HEX for char in value)
    )


def _identity_containers(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return the canonical identity containers without traversing arbitrary data.

    Episode references are serialized with identity split between their top-level
    fields and the nested ``source`` object.  Audit rows may also carry a
    ``source_identity`` or ``provenance`` envelope.  Walking only these named
    containers keeps identity admission bounded while covering each supported
    serialization shape.
    """

    containers: list[Mapping[str, Any]] = []
    pending: list[Mapping[str, Any]] = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        containers.append(current)
        for key in ("source", "source_identity", "provenance", "identity"):
            if key not in current:
                continue
            nested = current[key]
            if not isinstance(nested, Mapping):
                raise MaterializationValidationError(f"{key} identity envelope must be an object")
            pending.append(nested)
    return tuple(containers)


def _identity_value(value: Mapping[str, Any], field_name: str) -> str:
    """Read one identity field through all supported aliases and envelopes.

    Returns:
        The declared identity value, or an empty string when it is omitted.

    Raises:
        MaterializationValidationError: If aliases declare contradictory values.
    """

    candidates: list[tuple[str, str]] = []
    for container in _identity_containers(value):
        for alias in _IDENTITY_ALIASES[field_name]:
            if alias not in container:
                continue
            if not isinstance(container[alias], str):
                raise MaterializationValidationError(
                    f"{field_name}.{alias} identity must be a string"
                )
            candidate = _bounded_text(
                container[alias],
                name=f"{field_name}.{alias}",
                required=False,
            )
            if not candidate:
                continue
            candidates.append((alias, candidate))
    if not candidates:
        return ""
    case_insensitive = field_name in {"source_digest", "source_commit"}
    normalized = {
        candidate.casefold() if case_insensitive else candidate for _, candidate in candidates
    }
    if len(normalized) > 1:
        aliases = ", ".join(alias for alias, _ in candidates)
        raise MaterializationValidationError(f"{field_name} identity aliases conflict ({aliases})")
    return candidates[0][1]


def _episode_values(episode: Mapping[str, Any]) -> dict[str, str]:
    """Collect only identity fields that the selected episode explicitly carries.

    Returns:
        Normalized identity fields, including the required episode identifier.
    """

    values = {field: _identity_value(episode, field) for field in _IDENTITY_FIELDS}
    values["episode_id"] = _bounded_text(values["episode_id"], name="episode_id", required=True)
    return values


def _descriptor_identity(descriptor: Mapping[str, Any]) -> dict[str, str]:
    """Collect identity fields declared alongside an original recording.

    Returns:
        Normalized identity values declared by the recording descriptor.
    """

    return {field: _identity_value(descriptor, field) for field in _IDENTITY_FIELDS}


def _safe_relative_uri(value: Any, *, name: str) -> str:
    """Validate one local relative URI and reject traversal/schemes/separators.

    Returns:
        The validated local relative URI.
    """

    uri = _bounded_text(value, name=name, required=True)
    parsed = urlsplit(uri)
    pure = PurePosixPath(uri)
    windows = PureWindowsPath(uri)
    if (
        parsed.scheme
        or parsed.query
        or parsed.fragment
        or "\\" in uri
        or "\x00" in uri
        or pure.is_absolute()
        or ".." in pure.parts
        or windows.is_absolute()
        or bool(windows.drive)
    ):
        raise MaterializationValidationError(f"{name} must be a local relative path")
    return uri


def _safe_output_directory(value: str | None) -> str | None:
    """Validate a caller-selected output directory without resolving it yet.

    Returns:
        The validated relative path, or ``None`` when the default is requested.
    """

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_OUTPUT_DIRECTORY:
        raise MaterializationValidationError("output_directory must be a bounded non-empty string")
    return _safe_relative_uri(value, name="output_directory")


def _open_read_bounded(  # noqa: C901 - descriptor-safe path walk
    root: Path | MaterializationSourceRoot,
    uri: str,
) -> _ReadBytes:
    """Open, hash, and retain one regular file under ``root`` without path races.

    Every path component is opened relative to no-follow directory descriptors.
    The digest and bytes therefore come from the same descriptor, so a path
    replacement between a preliminary stat and read cannot silently alter the
    identity being verified.

    Returns:
        The source path, bytes, and digest read from one protected descriptor.
    """

    root_path = root.root if isinstance(root, MaterializationSourceRoot) else Path(root).absolute()
    if not isinstance(root, MaterializationSourceRoot) and not root_path.is_dir():
        raise FileNotFoundError(f"source root is unavailable: {root_path}")
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    nonblock_flag = getattr(os, "O_NONBLOCK", None)
    if not all(isinstance(flag, int) for flag in (directory_flag, nofollow_flag, nonblock_flag)):
        raise OSError("protected source opening is unavailable on this platform")

    parts = Path(uri).parts
    if not parts:
        raise MaterializationValidationError("source URI has no path components")
    root_flags = os.O_RDONLY | directory_flag | nofollow_flag | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | nonblock_flag | nofollow_flag | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        parent_fd = (
            root.open_root()[0]
            if isinstance(root, MaterializationSourceRoot)
            else _open_absolute_directory_no_follow(root_path)
        )
        descriptors.append(parent_fd)
        for part in parts[:-1]:
            parent_fd = os.open(part, root_flags, dir_fd=parent_fd)
            descriptors.append(parent_fd)
        source_fd = os.open(parts[-1], file_flags, dir_fd=parent_fd)
        descriptors.append(source_fd)
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("source is not a regular file")
        if source_stat.st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"source exceeds maximum size of {MAX_SOURCE_BYTES} bytes")
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            payload.extend(chunk)
            digest.update(chunk)
            if len(payload) > MAX_SOURCE_BYTES:
                raise ValueError(f"source exceeds maximum size of {MAX_SOURCE_BYTES} bytes")
        return _ReadBytes(root_path.joinpath(*parts), bytes(payload), digest.hexdigest())
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _identity_match(
    episode_values: Mapping[str, str], descriptor_values: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Compare available identity fields and expose missing/bound statuses.

    Returns:
        A per-field status map and the names of mismatched fields.
    """

    statuses: dict[str, str] = {}
    mismatches: list[str] = []
    for field_name in _IDENTITY_FIELDS:
        expected = episode_values.get(field_name, "")
        declared = descriptor_values.get(field_name, "")
        if expected and declared:
            equal = (
                expected.lower() == declared.lower()
                if field_name in {"source_digest", "source_commit"}
                else expected == declared
            )
            statuses[field_name] = "verified" if equal else "mismatch"
            if not equal:
                mismatches.append(field_name)
        elif expected:
            statuses[field_name] = "unbound"
        elif declared:
            statuses[field_name] = "declared_only"
        else:
            statuses[field_name] = "not_provided"
    return statuses, mismatches


def _recording_descriptor(
    episode: Mapping[str, Any], explicit: Mapping[str, Any] | SourceRef | None
) -> Any:
    """Return only an explicitly declared original-recording mapping.

    In particular, fields such as ``video_path`` and ``replay_map_path`` are
    not accepted as recording declarations: selecting those would guess a
    source identity from an unrelated convenience field.
    """

    candidate: Any = explicit
    if candidate is None:
        for key in ("recording", "original_recording", "original_trace"):
            if key in episode:
                candidate = episode[key]
                break
    if isinstance(candidate, SourceRef):
        return {
            "artifact_id": candidate.artifact_id,
            "uri": candidate.uri,
            "format": candidate.format,
            "schema": candidate.schema,
            "sha256": candidate.sha256,
            "source_commit": candidate.source_commit,
            "config_identity": candidate.config_identity,
        }
    if candidate is not None:
        return candidate
    return None


def _recording_uri(descriptor: Mapping[str, Any]) -> str:
    """Read the explicit recording URI/path alias.

    Returns:
        The validated relative URI declared by ``descriptor``.
    """

    for key in ("uri", "path", "recording_path", "source_uri", "trace_path"):
        if key in descriptor:
            return _safe_relative_uri(descriptor[key], name=f"recording.{key}")
    raise MaterializationValidationError("recording declaration requires an explicit URI/path")


def _original_media_reason(payload: bytes, declared_format: Any) -> str | None:
    """Admit only a decodable video or the separately validated trace format.

    Returns:
        A fail-closed reason, or ``None`` for a supported original container.
    """

    if declared_format == ANALYSIS_TRACE_RECORD_SCHEMA_VERSION:
        return "original_recording_native_trace_not_original"
    if declared_format == SIMULATION_TRACE_EXPORT_SCHEMA_VERSION:
        return None
    if declared_format != "video/mp4":
        return "original_recording_format_unsupported"
    try:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "mp4",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "framehash",
            "pipe:1",
        ]
        probe = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return "original_recording_media_decoder_unavailable"
    if probe.returncode != 0 or not any(
        line.startswith(b"0,") for line in probe.stdout.splitlines()
    ):
        return "original_recording_media_type_mismatch"
    return None


def _verify_original(  # noqa: C901, PLR0912 - fail-closed source proof has ordered checks
    episode: Mapping[str, Any],
    descriptor_value: Any,
    *,
    source_root: Path | MaterializationSourceRoot,
) -> tuple[_OriginalRecording | None, str | None]:
    """Locate and verify an original recording, returning a fallback reason.

    Returns:
        The verified recording and no reason, or ``None`` and a safe fallback
        reason when source proof is insufficient.
    """

    if descriptor_value is None:
        return None, "original_recording_not_declared"
    if not isinstance(descriptor_value, Mapping):
        return None, "original_recording_declaration_malformed"
    descriptor = dict(descriptor_value)
    try:
        uri = _recording_uri(descriptor)
        declared_digest = descriptor.get("sha256", descriptor.get("digest"))
        if not _valid_sha256(declared_digest):
            return None, "original_recording_digest_missing_or_malformed"
        read = _open_read_bounded(source_root, uri)
    except (MaterializationValidationError, FileNotFoundError, OSError, ValueError) as error:
        return None, f"original_recording_unavailable:{type(error).__name__}"
    if read.sha256.lower() != str(declared_digest).lower():
        return None, "original_recording_digest_mismatch"
    try:
        episode_values = _episode_values(episode)
        descriptor_values = _descriptor_identity(descriptor)
    except MaterializationValidationError as error:
        return None, f"original_recording_identity_conflict:{error}"
    declared_identity_digest = descriptor_values.get("source_digest", "")
    if (
        declared_identity_digest
        and declared_identity_digest.lower() != str(declared_digest).lower()
    ):
        return None, "original_recording_identity_mismatch:source_digest"
    descriptor_values["source_digest"] = str(declared_digest)

    # For opaque media, the explicit declaration is the only safe identity
    # anchor.  Trace recordings additionally expose an embedded episode ID;
    # use it only as a corroborating check, never as a guessed path/identity.
    embedded_episode = ""
    embedded_scenario = ""
    if descriptor.get("format") == SIMULATION_TRACE_EXPORT_SCHEMA_VERSION:
        try:
            payload = json.loads(read.payload)
            typed_trace = simulation_trace_export_from_dict(payload)
            embedded_episode = typed_trace.source.episode_id
            embedded_scenario = typed_trace.source.scenario_id
        except (
            SimulationTraceExportValidationError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return None, "original_recording_trace_malformed"
        if not embedded_episode:
            return None, "original_recording_episode_identity_missing"
        if embedded_episode != episode_values["episode_id"]:
            return None, "original_recording_episode_identity_mismatch"
        declared_episode = descriptor_values.get("episode_id", "")
        if declared_episode and declared_episode != embedded_episode:
            return None, "original_recording_identity_mismatch:episode_id"
        declared_scenario = descriptor_values.get("scenario_id", "")
        if declared_scenario and declared_scenario != embedded_scenario:
            return None, "original_recording_identity_mismatch:scenario_id"
        descriptor_values["episode_id"] = embedded_episode
        descriptor_values["scenario_id"] = embedded_scenario

    identity_status, mismatches = _identity_match(episode_values, descriptor_values)
    if mismatches:
        return None, "original_recording_identity_mismatch:" + ",".join(mismatches)
    unbound = [
        name
        for name, status in identity_status.items()
        if status == "unbound" and name not in {"source_digest", "episode_id"}
    ]
    if unbound:
        return None, "original_recording_identity_unbound:" + ",".join(unbound)
    try:
        final_read = _open_read_bounded(source_root, uri)
    except (FileNotFoundError, OSError, ValueError, MaterializationValidationError):
        return None, "original_recording_source_changed"
    if final_read.sha256.lower() != read.sha256.lower():
        return None, "original_recording_source_changed"
    media_reason = _original_media_reason(final_read.payload, descriptor.get("format"))
    if media_reason is not None:
        return None, media_reason
    return _OriginalRecording(final_read, identity_status, descriptor, uri), None


def _retained_trace(episode: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return an explicitly retained canonical trace payload, if present."""

    for key in ("retained_trace", "simulation_trace", "trace"):
        value = episode.get(key)
        if isinstance(value, Mapping):
            return value
    retained = episode.get("retained_state")
    if isinstance(retained, Mapping):
        for key in ("trace", "simulation_trace"):
            value = retained.get(key)
            if isinstance(value, Mapping):
                return value
    return None


def _verify_native_trace_identity(  # noqa: C901 - each selected claim fails closed
    episode: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    trusted_scan_identity_defaults: Mapping[str, Any] | None = None,
) -> None:
    """Require a digest-bound trace identity for every selected claim."""
    if trace.get("schema_version") != ANALYSIS_TRACE_RECORD_SCHEMA_VERSION:
        raise MaterializationValidationError("native retained trace schema is unsupported")
    digest = trace.get("artifact_sha256")
    if not _valid_sha256(digest) or digest != trace_artifact_sha256(trace):
        raise MaterializationValidationError("native retained trace digest mismatch")
    selected_id = _episode_values(episode)["episode_id"]
    scan_defaults = trusted_scan_identity_defaults or {}

    def scanner_default(name: str, expected: str) -> bool:
        return bool(expected) and scan_defaults.get(name) == expected

    trace_id = trace.get("episode_id")
    trace_seed = trace.get("seed")
    if not isinstance(trace_id, str) or not trace_id or trace_id != selected_id:
        raise MaterializationValidationError("native retained trace episode_id mismatch or missing")
    if isinstance(trace_seed, bool) or not isinstance(trace_seed, int):
        raise MaterializationValidationError("native retained trace seed is missing or invalid")
    selected_seed = episode.get("seed")
    if selected_seed is not None and selected_seed != trace_seed:
        raise MaterializationValidationError("native retained trace seed does not match selection")
    selected_execution = _identity_value(episode, "execution_id")
    # BA-01 fills an absent execution_id with episode_id in its readable row.
    # Native runner traces do not declare execution_id, so that scanner default
    # is not a historical execution claim. An explicit differing ID still is.
    trace_execution = trace.get("execution_id")
    scanner_default_execution = (
        selected_execution == selected_id
        and trace_execution is None
        and scanner_default("execution_id", selected_execution)
    )
    if (
        selected_execution
        and selected_execution != trace_execution
        and not scanner_default_execution
    ):
        raise MaterializationValidationError(
            "native retained trace execution_id does not match selection"
        )
    for selected, observed in (
        ("campaign_id", "campaign_id"),
        ("checkpoint_digest", "checkpoint_digest"),
        ("environment_digest", "environment_digest"),
        ("initial_state_digest", "initial_state_digest"),
    ):
        expected = _identity_value(episode, selected)
        observed_value = trace.get(observed)
        # A scanner-added campaign wrapper can name a standalone runner trace
        # without the trace declaring that campaign. Explicit row claims bind.
        if (
            expected
            and expected != observed_value
            and not (
                selected == "campaign_id"
                and observed_value is None
                and scanner_default("campaign_id", expected)
            )
        ):
            raise MaterializationValidationError(
                f"native retained trace {selected} does not match selection"
            )
    selected_source_digest = _identity_value(episode, "source_digest")
    declared_trace_source_digest = trace.get("source_digest")
    if selected_source_digest:
        if declared_trace_source_digest is not None:
            source_mismatch = selected_source_digest != declared_trace_source_digest
        else:
            source_mismatch = (
                not scanner_default("source_digest", selected_source_digest)
                and selected_source_digest != digest
            )
        if source_mismatch:
            raise MaterializationValidationError(
                "native retained trace source_digest does not match selection"
            )
    for selected, observed in (
        ("scenario_id", "scenario_id"),
        ("source_commit", "git_hash"),
        ("config_identity", "config_digest"),
    ):
        expected = _identity_value(episode, selected)
        if expected and expected != trace.get(observed):
            raise MaterializationValidationError(
                f"native retained trace {selected} does not match selection"
            )
    planner_claims = [
        value for key in ("algo", "planner_id") if (value := episode.get(key)) not in (None, "")
    ]
    if any(not isinstance(value, str) or value != trace.get("planner") for value in planner_claims):
        raise MaterializationValidationError(
            "native retained trace planner does not match selection"
        )


def _native_trace_states(
    episode: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    trusted_scan_identity_defaults: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project verified native telemetry into retained replay states only.

    The projection preserves recorded time, pose and pedestrian positions. It
    deliberately does not invent a planner command, stable actor ID, or a
    historical recording from incomplete analysis telemetry.

    Returns:
        Bounded retained states suitable for the canonical trajectory renderer.
    """

    _verify_native_trace_identity(
        episode, trace, trusted_scan_identity_defaults=trusted_scan_identity_defaults
    )
    raw_steps = trace.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps or len(raw_steps) > MAX_STATES:
        raise MaterializationValidationError("native retained trace steps are unavailable")
    states: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, Mapping):
            raise MaterializationValidationError(f"native trace step {index} is not an object")
        robot = step.get("robot")
        if not isinstance(robot, Mapping):
            raise MaterializationValidationError(f"native trace step {index} has no robot")
        position = robot.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise MaterializationValidationError(f"native trace step {index} has no robot position")
        pedestrians = step.get("pedestrians")
        if not isinstance(pedestrians, list) or len(pedestrians) > MAX_STATES:
            raise MaterializationValidationError(
                f"native trace step {index} has invalid pedestrians"
            )
        states.append(
            {
                "time_s": _finite(step.get("time_s"), name=f"native.steps[{index}].time_s"),
                "x": _finite(position[0], name=f"native.steps[{index}].robot.x"),
                "y": _finite(position[1], name=f"native.steps[{index}].robot.y"),
                "heading": _finite(robot.get("heading"), name=f"native.steps[{index}].heading"),
                "pedestrians": pedestrians,
            }
        )
    return states


def _retained_trace_path(episode: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return a retained trace path declaration, without treating it as original."""

    for key in ("retained_trace_source", "retained_trace_recording"):
        value = episode.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _retained_steps(episode: Mapping[str, Any]) -> Sequence[Any] | None:
    """Return explicit retained replay states, if present."""

    value = episode.get("retained_states")
    if isinstance(value, list):
        return value
    value = episode.get("replay_steps")
    return value if isinstance(value, list) else None


def _finite(value: Any, *, name: str) -> float:
    """Coerce one finite numeric state value.

    Returns:
        A finite floating-point value.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterializationValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MaterializationValidationError(f"{name} must be finite")
    return number


def _replay_episode(episode: Mapping[str, Any], states: Sequence[Any]) -> ReplayEpisode:
    """Validate retained replay states before passing them to the canonical renderer.

    Returns:
        A typed replay episode accepted by the canonical figure renderer.
    """

    if not states or len(states) > MAX_STATES:
        raise MaterializationValidationError(
            "retained replay states are missing or exceed the limit"
        )
    episode_values = _episode_values(episode)
    steps: list[ReplayStep] = []
    previous_time = -math.inf
    for index, raw in enumerate(states):
        if not isinstance(raw, Mapping):
            raise MaterializationValidationError(f"retained_states[{index}] must be an object")
        time_s = _finite(raw.get("time_s", raw.get("t")), name=f"retained_states[{index}].time_s")
        x = _finite(raw.get("x"), name=f"retained_states[{index}].x")
        y = _finite(raw.get("y"), name=f"retained_states[{index}].y")
        heading = _finite(raw.get("heading", 0.0), name=f"retained_states[{index}].heading")
        if time_s <= previous_time:
            raise MaterializationValidationError(
                "retained replay times must be strictly increasing"
            )
        previous_time = time_s
        ped_positions: list[tuple[float, float]] | None = None
        raw_peds = raw.get("ped_positions", raw.get("pedestrians"))
        if raw_peds is not None:
            if not isinstance(raw_peds, list):
                raise MaterializationValidationError(
                    f"retained_states[{index}].pedestrians must be a list"
                )
            ped_positions = []
            for ped_index, raw_ped in enumerate(raw_peds):
                if isinstance(raw_ped, Mapping):
                    raw_ped = raw_ped.get("position")
                if not isinstance(raw_ped, (list, tuple)) or len(raw_ped) != 2:
                    raise MaterializationValidationError(
                        f"retained_states[{index}].pedestrians[{ped_index}] must be a 2-vector"
                    )
                ped_positions.append(
                    (
                        _finite(raw_ped[0], name="pedestrian.x"),
                        _finite(raw_ped[1], name="pedestrian.y"),
                    )
                )
        steps.append(
            ReplayStep(
                t=time_s,
                x=x,
                y=y,
                heading=heading,
                speed=(
                    _finite(raw["speed"], name=f"retained_states[{index}].speed")
                    if "speed" in raw
                    else None
                ),
                ped_positions=ped_positions,
            )
        )
    return ReplayEpisode(
        episode_id=episode_values["episode_id"],
        scenario_id=_identity_value(episode, "scenario_id"),
        steps=steps,
    )


def _trace_payload(episode: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    """Validate retained trace identity and return an isolated JSON payload.

    Returns:
        A normalized, strict JSON trace export.
    """

    payload = dict(trace)
    try:
        typed = simulation_trace_export_from_dict(payload)
    except (SimulationTraceExportValidationError, TypeError, ValueError) as error:
        raise MaterializationValidationError(f"retained trace is invalid: {error}") from error
    episode_id = _episode_values(episode)["episode_id"]
    if typed.source.episode_id != episode_id:
        raise MaterializationValidationError(
            "retained trace episode identity does not match selection"
        )
    scenario_id = _identity_value(episode, "scenario_id")
    if scenario_id and typed.source.scenario_id != scenario_id:
        raise MaterializationValidationError(
            "retained trace scenario identity does not match selection"
        )
    if not typed.frames:
        raise MaterializationValidationError("retained trace contains no states")
    return typed.to_dict()


def _render_config(config: Mapping[str, Any] | None, frame_count: int) -> dict[str, Any]:
    """Build the small allowlisted SREV scene config used for retained traces.

    Returns:
        A scene component configuration with bounded default output formats.
    """

    default_scene: dict[str, Any] = {
        "frame_indices": list(range(frame_count)),
        "formats": ["png"],
        "figure": {"width_in": 3.2, "height_in": 2.4, "dpi": 80},
    }
    if config is None:
        return {"scene": default_scene}
    candidate = dict(config)
    scene = candidate.get("scene", candidate)
    if not isinstance(scene, Mapping):
        raise MaterializationValidationError("render_config.scene must be an object")
    merged = dict(default_scene)
    merged.update(dict(scene))
    merged["frame_indices"] = list(merged.get("frame_indices", default_scene["frame_indices"]))
    return {"scene": merged}


def build_materialization_cache_key(
    episode: Mapping[str, Any],
    *,
    source_digest: str = "",
    render_config: Mapping[str, Any] | None = None,
    tool_version: str = MATERIALIZATION_TOOL_VERSION,
) -> str:
    """Build an immutable key from source, identity, tool, and render config.

    The output location is intentionally excluded.  A changed source digest,
    selected episode identity, renderer config, or tool version therefore
    cannot reuse a prior materialization key.

    Returns:
        A lowercase SHA-256 cache key.
    """

    if not isinstance(episode, Mapping):
        raise MaterializationValidationError("episode must be an object")
    episode_values = _episode_values(episode)
    if source_digest and not _valid_sha256(source_digest):
        raise MaterializationValidationError("source_digest must be a 64-hex SHA-256")
    payload = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "tool_version": _bounded_text(tool_version, name="tool_version", required=True),
        "renderer": {
            "component_id": review_scene.COMPONENT_ID,
            "component_version": review_scene.COMPONENT_VERSION,
        },
        "episode": episode_values,
        "source_digest": source_digest.lower(),
        "render_config": dict(render_config or {}),
    }
    return _sha256(_canonical_bytes(payload))


def _default_output_directory(episode_id: str, cache_key: str) -> str:
    """Return a deterministic safe output path for a derived render.

    Returns:
        A bounded relative output directory.
    """

    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in episode_id)
    safe_id = safe_id[:80] or "episode"
    return f"audit-materialization/{safe_id}-{cache_key[:12]}"


def _prepare_output_root(
    output_root: str | Path | MaterializationOutputRoot,
) -> Path | MaterializationOutputRoot:
    """Prepare a caller-owned output root without following a root link.

    Returns:
        The absolute output directory or its descriptor-bound capability.
    """

    if isinstance(output_root, MaterializationOutputRoot):
        allowed_stat = os.fstat(output_root.allowed_root_fd)
        if (allowed_stat.st_dev, allowed_stat.st_ino) != output_root.allowed_root_identity:
            raise MaterializationValidationError(
                "allowed output root changed before materialization"
            )
        return output_root
    root = Path(output_root).absolute()
    root_fd = _open_absolute_directory_create_no_follow(root)
    os.close(root_fd)
    return root


def _output_path(root: Path | MaterializationOutputRoot, relative: str) -> Path:
    """Resolve one new output directory beneath the output root.

    Returns:
        A not-yet-created output directory below ``root``.
    """

    root_path = root.root if isinstance(root, MaterializationOutputRoot) else root
    output = (root_path / relative).resolve(strict=False)
    try:
        output.relative_to(root_path.resolve(strict=False))
    except ValueError as error:
        raise MaterializationValidationError("output_directory escapes output_root") from error
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {relative}")
    return output


def _output_directory_flags() -> int:
    """Return flags for opening one output directory without following links."""

    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if (
        not isinstance(directory, int)
        or not isinstance(nofollow, int)
        or directory == 0
        or nofollow == 0
    ):
        raise OSError("descriptor-backed output publication is unavailable")
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _open_absolute_directory_no_follow(path: Path) -> int:
    """Open an absolute directory path by walking every component safely.

    Returns:
        An open descriptor for the admitted directory.
    """

    absolute = path if path.is_absolute() else path.absolute()
    parts = absolute.parts
    if not parts or parts[0] != os.sep or ".." in parts:
        raise MaterializationValidationError("output root must be an absolute directory")
    current_fd = os.open(os.sep, _output_directory_flags())
    try:
        for part in parts[1:]:
            next_fd = os.open(part, _output_directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise MaterializationValidationError("output root must be a directory")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_absolute_directory_create_no_follow(path: Path) -> int:
    """Create and open an absolute directory by walking every component safely.

    Returns:
        An open descriptor for the created or existing directory.
    """

    absolute = path if path.is_absolute() else path.absolute()
    parts = absolute.parts
    if not parts or parts[0] != os.sep or ".." in parts:
        raise MaterializationValidationError("output root must be an absolute directory")
    current_fd = os.open(os.sep, _output_directory_flags())
    try:
        for part in parts[1:]:
            try:
                os.mkdir(part, mode=0o755, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, _output_directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise MaterializationValidationError("output root must be a directory")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _output_parts(relative: str) -> tuple[str, ...]:
    """Return validated POSIX components for one output-relative directory."""

    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise MaterializationValidationError("output_directory must be a relative directory")
    return parts


@dataclass(slots=True)
class _OutputLease:
    """Descriptor-backed output boundary retained across renderer writes."""

    root: Path
    relative: str
    parts: tuple[str, ...]
    root_fd: int
    root_identity: tuple[int, int]
    parent_fd: int = -1
    parent_identity: tuple[int, int] | None = None
    directory_fd: int = -1
    directory_identity: tuple[int, int] | None = None

    @property
    def path(self) -> Path:
        """Return the caller-visible output path used in result envelopes."""

        return self.root.joinpath(*self.parts)

    @property
    def renderer_root(self) -> Path:
        """Return a path anchored to the retained root descriptor."""

        return _descriptor_path(self.root_fd)

    @property
    def renderer_directory(self) -> Path:
        """Return a path anchored to the retained output-directory descriptor."""

        if self.directory_fd < 0:
            raise OSError("output directory has not been bound")
        return _descriptor_path(self.directory_fd)

    def close(self) -> None:
        """Close retained descriptors without masking the materialization result."""

        for attribute in ("directory_fd", "parent_fd", "root_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
            finally:
                setattr(self, attribute, -1)


def _descriptor_path(descriptor: int) -> Path:
    """Return a Linux descriptor-backed path for path-only renderer APIs."""

    proc_path = Path(f"/proc/self/fd/{descriptor}")
    if not proc_path.exists():
        raise OSError("descriptor-backed renderer path is unavailable")
    return proc_path


def _walk_output_from_fd(root_fd: int, parts: Sequence[str], *, create: bool) -> int:
    """Walk relative output components from a retained root descriptor.

    Returns:
        An open descriptor for the final walked directory.
    """

    current_fd = os.dup(root_fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, _output_directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def admit_output_root(  # noqa: C901 - ordered fail-closed admission checks
    output_root: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
) -> MaterializationOutputRoot:
    """Admit an output root beneath one policy root and retain its descriptor.

    The caller should perform its policy decision immediately before this
    admission.  The admission repeats the resolved containment check, opens
    the selected policy root without following links, and retains that
    descriptor through the materializer handoff.  Missing output components
    are intentionally left uncreated until :func:`materialize_episode` needs
    to render; creation then occurs relative to the retained descriptor.

    Raises:
        MaterializationValidationError: If the requested root is not safely
            contained by an allowed root.
        OSError: If the allowed root or an existing output component cannot be
            opened without following a link.

    Returns:
        A descriptor-bound output-root capability owned by the caller.
    """

    try:
        requested = Path(output_root).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise MaterializationValidationError("output_root cannot be resolved safely") from error
    candidates: list[tuple[Path, tuple[str, ...]]] = []
    for configured_root in allowed_roots:
        try:
            allowed = Path(configured_root).resolve(strict=False)
            relative = requested.relative_to(allowed)
        except (OSError, RuntimeError, ValueError):
            continue
        parts = tuple(relative.parts)
        if any(part in {"", ".", ".."} for part in parts):
            continue
        candidates.append((allowed, parts))
    if not candidates:
        raise MaterializationValidationError("output_root is not beneath an allowed root")

    # Prefer the most-specific allowed root when policies contain nested roots.
    allowed, relative_parts = max(candidates, key=lambda item: len(item[0].parts))
    allowed_root_fd = _open_absolute_directory_no_follow(allowed)
    root_fd = -1
    try:
        allowed_stat = os.fstat(allowed_root_fd)
        allowed_identity = (allowed_stat.st_dev, allowed_stat.st_ino)
        if relative_parts:
            try:
                root_fd = _walk_output_from_fd(allowed_root_fd, relative_parts, create=False)
            except FileNotFoundError:
                # The materializer will create a missing root from the retained
                # allowed-root descriptor, never through the mutable pathname.
                root_fd = -1
        else:
            root_fd = os.dup(allowed_root_fd)
        root_identity = None
        if root_fd >= 0:
            root_stat = os.fstat(root_fd)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise MaterializationValidationError("output root must be a directory")
            root_identity = (root_stat.st_dev, root_stat.st_ino)
        return MaterializationOutputRoot(
            root=requested,
            allowed_root_fd=allowed_root_fd,
            allowed_root_identity=allowed_identity,
            relative_parts=relative_parts,
            root_fd=root_fd,
            root_identity=root_identity,
        )
    except BaseException:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(allowed_root_fd)
        raise


def admit_source_root(
    source_root: str | Path,
    *,
    allowed_roots: Sequence[str | Path],
) -> MaterializationSourceRoot:
    """Admit an existing source root beneath one policy root.

    The returned descriptor keeps source reads bound to the root identity that
    was admitted, even if the visible source-root pathname is later replaced.

    Raises:
        MaterializationValidationError: If the requested root is not safely
            contained by an allowed root.
        OSError: If the allowed or source root cannot be opened without
            following a link.

    Returns:
        A descriptor-bound source-root capability owned by the caller.
    """

    try:
        requested = Path(source_root).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise MaterializationValidationError("source_root cannot be resolved safely") from error
    candidates: list[tuple[Path, tuple[str, ...]]] = []
    for configured_root in allowed_roots:
        try:
            allowed = Path(configured_root).resolve(strict=False)
            relative = requested.relative_to(allowed)
        except (OSError, RuntimeError, ValueError):
            continue
        parts = tuple(relative.parts)
        if any(part in {"", ".", ".."} for part in parts):
            continue
        candidates.append((allowed, parts))
    if not candidates:
        raise MaterializationValidationError("source_root is not beneath an allowed root")

    allowed, relative_parts = max(candidates, key=lambda item: len(item[0].parts))
    allowed_root_fd = _open_absolute_directory_no_follow(allowed)
    source_fd = -1
    try:
        if relative_parts:
            source_fd = _walk_output_from_fd(allowed_root_fd, relative_parts, create=False)
        else:
            source_fd = os.dup(allowed_root_fd)
        source_stat = os.fstat(source_fd)
        if not stat.S_ISDIR(source_stat.st_mode):
            raise MaterializationValidationError("source_root must be a directory")
        return MaterializationSourceRoot(
            root=requested,
            root_fd=source_fd,
            root_identity=(source_stat.st_dev, source_stat.st_ino),
        )
    except BaseException:
        if source_fd >= 0:
            os.close(source_fd)
        raise
    finally:
        os.close(allowed_root_fd)


def _open_output_lease(
    root: Path | MaterializationOutputRoot,
    relative: str,
) -> _OutputLease:
    """Admit an output root and retain its descriptor through rendering.

    Returns:
        A lease that keeps the admitted root descriptor open.
    """

    parts = _output_parts(relative)
    root_path = root.root if isinstance(root, MaterializationOutputRoot) else root
    if isinstance(root, MaterializationOutputRoot):
        root_fd, root_identity = root.open_root()
    else:
        root_fd = _open_absolute_directory_no_follow(root_path)
        root_stat = os.fstat(root_fd)
        root_identity = (root_stat.st_dev, root_stat.st_ino)
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise OSError("output root is not a directory")
        return _OutputLease(
            root=root_path,
            relative=relative,
            parts=parts,
            root_fd=root_fd,
            root_identity=root_identity,
        )
    except BaseException:
        os.close(root_fd)
        raise


def _open_private_staging_parent(root_fd: int, root_identity: tuple[int, int]) -> int:
    """Duplicate the retained output-root descriptor for private staging.

    Renderer staging must not select a temporary parent through a mutable
    pathname.  Duplicating the already-admitted root descriptor keeps the
    staging directory on the output filesystem and below the same retained
    directory even if its visible parent is renamed or replaced.

    Returns:
        An open descriptor for the retained output-root staging parent.
    """

    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise OSError("retained output root is not a directory")
    if (root_stat.st_dev, root_stat.st_ino) != root_identity:
        raise OSError("retained output root changed before staging")
    return os.dup(root_fd)


def _bind_output_directory(lease: _OutputLease, *, create: bool) -> None:
    """Create or open the final output directory through the retained root fd."""

    retained_parent = lease.parent_fd >= 0
    parent_fd = (
        os.dup(lease.parent_fd)
        if retained_parent
        else _walk_output_from_fd(lease.root_fd, lease.parts[:-1], create=create)
    )
    directory_fd = -1
    try:
        if create:
            try:
                os.mkdir(lease.parts[-1], mode=0o755, dir_fd=parent_fd)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output directory already exists: {lease.relative}"
                ) from error
        directory_fd = os.open(lease.parts[-1], _output_directory_flags(), dir_fd=parent_fd)
        parent_stat = os.fstat(parent_fd)
        directory_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(parent_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            raise OSError("output path is not a directory")
        if not retained_parent:
            lease.parent_fd = parent_fd
            lease.parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        lease.directory_fd = directory_fd
        lease.directory_identity = (directory_stat.st_dev, directory_stat.st_ino)
        if not retained_parent:
            parent_fd = -1
        directory_fd = -1
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _retain_output_parent(lease: _OutputLease) -> None:
    """Retain the output parent before a renderer receives any path."""

    if lease.parent_fd >= 0:
        return
    parent_fd = _walk_output_from_fd(lease.root_fd, lease.parts[:-1], create=True)
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise OSError("output parent is not a directory")
        lease.parent_fd = parent_fd
        lease.parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        parent_fd = -1
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def _publish_staged_files(lease: _OutputLease, staged_output: int) -> None:
    """Publish private staged files into a reserved output directory.

    Each source is opened no-follow through the retained staging descriptor and
    copied into a fresh private inode before publication.  A libc
    ``renameat2(RENAME_NOREPLACE)`` then makes that complete, fsynced copy
    visible atomically without aliasing a renderer-owned hardlink.  The final
    output directory is reserved before rendering; an actor cannot win a
    separate destination-directory stat/rename race.
    """

    if lease.directory_fd < 0:
        raise OSError("output directory descriptor is not retained")
    staged_stat = os.fstat(staged_output)
    if not stat.S_ISDIR(staged_stat.st_mode):
        raise OSError("private trace staging is not a directory")
    names = os.listdir(staged_output)
    for name in sorted(names):
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise OSError(f"unsafe staged output entry: {name!r}")
        _copy_staged_file(staged_output, name, lease.directory_fd)
        os.unlink(name, dir_fd=staged_output)
    os.fsync(lease.directory_fd)


def _rename_noreplace(
    source_name: str,
    source_directory_fd: int,
    destination_name: str,
    destination_directory_fd: int,
) -> None:
    """Atomically rename one descriptor-relative path without replacing it.

    Python's portable ``os.rename`` API has no no-replace flag.  This module's
    descriptor-backed publication contract therefore uses the Linux libc
    ``renameat2`` primitive and fails closed when that primitive is unavailable;
    an overwrite-prone or partially visible fallback would weaken the output
    boundary.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as error:
        raise OSError("atomic no-replace rename is unavailable") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _open_private_staged_copy(staged_output: int) -> tuple[int, str]:
    """Allocate a fresh private copy inode below the retained staging fd.

    Returns:
        The copy descriptor and its private descriptor-relative name.
    """

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise OSError("no-follow staged-file opening is unavailable")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    for _ in range(8):
        copy_name = f".audit-materialize-copy-{uuid.uuid4().hex}"
        try:
            copy_fd = os.open(
                copy_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow | close_on_exec,
                mode=0o644,
                dir_fd=staged_output,
            )
        except FileExistsError:
            continue
        return copy_fd, copy_name
    raise OSError("could not allocate private staged copy")


def _copy_descriptor_bytes(source_fd: int, destination_fd: int) -> str:
    """Copy bytes between descriptors and return the source-read digest.

    Returns:
        The lowercase SHA-256 digest of bytes read from ``source_fd``.
    """

    digest = hashlib.sha256()
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("staged copy write made no progress")
            view = view[written:]
    return digest.hexdigest()


def _copy_staged_file(staged_output: int, name: str, destination_directory: int) -> None:
    """Copy one no-follow staged file to a fresh fsynced inode and publish it."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise OSError("no-follow staged-file opening is unavailable")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    source_fd = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | nofollow | close_on_exec,
        dir_fd=staged_output,
    )
    copy_fd = -1
    copy_name = ""
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise OSError(f"staged output entry is not a regular file: {name}")
        copy_fd, copy_name = _open_private_staged_copy(staged_output)
        digest = _copy_descriptor_bytes(source_fd, copy_fd)
        os.fsync(copy_fd)
        os.lseek(copy_fd, 0, os.SEEK_SET)
        if _artifact_digest_from_fd(copy_fd) != digest:
            raise OSError(f"staged copy digest changed: {name}")
        os.close(copy_fd)
        copy_fd = -1
        _rename_noreplace(copy_name, staged_output, name, destination_directory)
        copy_name = ""
    finally:
        os.close(source_fd)
        if copy_fd >= 0:
            os.close(copy_fd)
        if copy_name:
            try:
                os.unlink(copy_name, dir_fd=staged_output)
            except FileNotFoundError:
                pass


def _open_output_artifact(directory_fd: int, uri: str) -> int:
    """Open one output artifact by relative URI without following links.

    Returns:
        An open descriptor for the regular artifact file.
    """

    safe_uri = _safe_relative_uri(uri, name="artifact.uri")
    parts = PurePosixPath(safe_uri).parts
    if not parts:
        raise OSError("artifact URI has no path components")
    current_fd = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _output_directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(nofollow, int) or nofollow == 0:
            raise OSError("no-follow artifact opening is unavailable")
        file_fd = os.open(
            parts[-1], os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0), dir_fd=current_fd
        )
        os.close(current_fd)
        current_fd = -1
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(file_fd)
            raise OSError(f"output artifact is not a regular file: {safe_uri}")
        return file_fd
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _artifact_digest_from_fd(file_fd: int) -> str:
    """Hash all bytes from one already-open output artifact descriptor.

    Returns:
        The lowercase SHA-256 digest of the descriptor's bytes.
    """

    digest = hashlib.sha256()
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _verify_output_file(
    directory_fd: int, uri: str, expected_digest: str, *, require_nonempty: bool = False
) -> None:
    """Verify one output file through a no-follow regular-file descriptor."""

    if not _valid_sha256(expected_digest):
        raise OSError("output file identity is invalid")
    file_fd = _open_output_artifact(directory_fd, uri)
    try:
        if require_nonempty and os.fstat(file_fd).st_size <= 0:
            raise OSError(f"output file is empty: {uri}")
        observed_digest = _artifact_digest_from_fd(file_fd)
    finally:
        os.close(file_fd)
    if observed_digest != expected_digest.lower():
        raise OSError(f"output file digest changed: {uri}")


def _verify_output_artifacts(directory_fd: int, artifacts: Sequence[Mapping[str, Any]]) -> None:
    """Verify every declared artifact against a no-follow final-file descriptor."""

    if not artifacts:
        raise OSError("manifest must declare at least one artifact")
    for artifact in artifacts:
        uri = artifact.get("uri")
        expected_digest = artifact.get("sha256")
        if not isinstance(uri, str) or not _valid_sha256(expected_digest):
            raise OSError("manifest artifact identity is invalid")
        _verify_output_file(directory_fd, uri, expected_digest, require_nonempty=True)


def _assert_output_lease_current(lease: _OutputLease) -> None:
    """Fail closed if the visible root, parent, or output entry was replaced."""

    root_stat = os.fstat(lease.root_fd)
    if (root_stat.st_dev, root_stat.st_ino) != lease.root_identity:
        raise OSError("output root descriptor changed during publication")
    if lease.parent_fd < 0 or lease.directory_fd < 0:
        raise OSError("output directory descriptor is not retained")
    parent_stat = os.fstat(lease.parent_fd)
    directory_stat = os.fstat(lease.directory_fd)
    if lease.parent_identity != (parent_stat.st_dev, parent_stat.st_ino):
        raise OSError("output parent descriptor changed during publication")
    if lease.directory_identity != (directory_stat.st_dev, directory_stat.st_ino):
        raise OSError("output directory descriptor changed during publication")

    visible_root_fd = _open_absolute_directory_no_follow(lease.root)
    try:
        visible_root = os.fstat(visible_root_fd)
        if (visible_root.st_dev, visible_root.st_ino) != lease.root_identity:
            raise OSError("output root was replaced during publication")
        visible_parent_fd = _walk_output_from_fd(visible_root_fd, lease.parts[:-1], create=False)
        try:
            visible_parent = os.fstat(visible_parent_fd)
            if lease.parent_identity != (visible_parent.st_dev, visible_parent.st_ino):
                raise OSError("output parent was replaced during publication")
            visible_directory_fd = os.open(
                lease.parts[-1], _output_directory_flags(), dir_fd=visible_parent_fd
            )
            try:
                visible_directory = os.fstat(visible_directory_fd)
                if lease.directory_identity != (
                    visible_directory.st_dev,
                    visible_directory.st_ino,
                ):
                    raise OSError("output directory was replaced during publication")
            finally:
                os.close(visible_directory_fd)
        finally:
            os.close(visible_parent_fd)
    finally:
        os.close(visible_root_fd)


def _ensure_output_parent_no_follow(
    root: Path | MaterializationOutputRoot,
    relative: str,
) -> None:
    """Create output parents through a short-lived descriptor-backed lease."""

    lease = _open_output_lease(root, relative)
    try:
        parent_fd = _walk_output_from_fd(lease.root_fd, lease.parts[:-1], create=True)
        os.close(parent_fd)
    finally:
        lease.close()


def _exclusive_json(
    output_root: Path | MaterializationOutputRoot,
    output_relative: str,
    payload: Mapping[str, Any],
    *,
    expected_identity: tuple[int, int] | None = None,
    directory_fd: int | None = None,
) -> str:
    """Publish one manifest through a no-follow output-directory descriptor.

    Returns:
        The SHA-256 digest of the bytes written.
    """

    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if directory_fd is None:
        lease = _open_output_lease(output_root, output_relative)
        try:
            _bind_output_directory(lease, create=False)
            directory_fd = os.dup(lease.directory_fd)
        finally:
            lease.close()
    else:
        directory_fd = os.dup(directory_fd)
    manifest_fd = -1
    try:
        directory_stat = os.fstat(directory_fd)
        if (
            expected_identity is not None
            and (
                directory_stat.st_dev,
                directory_stat.st_ino,
            )
            != expected_identity
        ):
            raise OSError("output directory identity changed during publication")
        declared_artifacts = payload.get("artifacts", ())
        if not isinstance(declared_artifacts, Sequence) or isinstance(
            declared_artifacts, (str, bytes, bytearray)
        ):
            raise OSError("manifest artifacts must be a sequence")
        _verify_output_artifacts(directory_fd, declared_artifacts)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        manifest_fd = os.open("audit-materialization.v1.json", flags, 0o644, dir_fd=directory_fd)
        with os.fdopen(manifest_fd, "wb") as stream:
            manifest_fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if manifest_fd >= 0:
            os.close(manifest_fd)
        os.close(directory_fd)
    return _sha256(encoded)


def _artifact_uri(path: Path, output: Path) -> str:
    """Return a relative artifact URI for a renderer-produced path.

    Returns:
        A path relative to the materialization output directory.
    """

    try:
        return path.resolve(strict=False).relative_to(output.resolve(strict=False)).as_posix()
    except ValueError:
        return path.name


def _component_artifacts(
    artifacts: Sequence[Mapping[str, Any]], *, output_relative: str
) -> tuple[dict[str, Any], ...]:
    """Normalize renderer artifact URIs relative to the materialization directory.

    Returns:
        Renderer artifact records with safe, output-relative URIs.
    """

    output_prefix = PurePosixPath(output_relative)
    normalized: list[dict[str, Any]] = []
    for raw in artifacts:
        artifact = dict(raw)
        uri = artifact.get("uri")
        if isinstance(uri, str):
            uri_path = PurePosixPath(uri)
            try:
                uri = uri_path.relative_to(output_prefix).as_posix()
            except ValueError:
                uri = uri_path.as_posix()
            artifact["uri"] = _safe_relative_uri(uri, name="artifact.uri")
        normalized.append(artifact)
    return tuple(normalized)


def _validate_renderer_artifacts(artifacts: Any) -> None:
    """Validate renderer artifact declarations before output admission."""

    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)):
        raise MaterializationValidationError("renderer result artifacts are invalid")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise MaterializationValidationError("renderer result artifact is invalid")
        if not isinstance(artifact.get("artifact_id"), str) or not artifact["artifact_id"].strip():
            raise MaterializationValidationError("renderer result artifact ID is invalid")
        if not isinstance(artifact.get("uri"), str):
            raise MaterializationValidationError("renderer result artifact URI is invalid")
        if not _valid_sha256(artifact.get("sha256")):
            raise MaterializationValidationError("renderer result artifact digest is invalid")


def _validate_renderer_diagnostics(diagnostics: Any) -> None:
    """Validate renderer diagnostics before materialization reads their fields."""

    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes, bytearray)):
        raise MaterializationValidationError("renderer result diagnostics are invalid")
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping) or not isinstance(diagnostic.get("reason", ""), str):
            raise MaterializationValidationError("renderer result diagnostic is invalid")


def _validate_renderer_result(
    result: Any,
    *,
    expected_request_id: str,
    expected_component_id: str,
) -> ComponentResult:
    """Validate a renderer result before reading any of its fields.

    Returns:
        The validated component result.
    """

    if not isinstance(result, ComponentResult):
        raise MaterializationValidationError("renderer result must be a ComponentResult")
    if (
        not isinstance(expected_request_id, str)
        or not expected_request_id
        or not isinstance(result.request_id, str)
        or not result.request_id
        or result.request_id != expected_request_id
    ):
        raise MaterializationValidationError("renderer result request ID is not bound")
    if (
        not isinstance(expected_component_id, str)
        or not expected_component_id
        or not isinstance(result.component_id, str)
        or not result.component_id
        or result.component_id != expected_component_id
    ):
        raise MaterializationValidationError("renderer result component ID is not bound")
    if result.status not in {"complete", "partial", "unavailable", "failed", "cancelled"}:
        raise MaterializationValidationError("renderer result status is invalid")
    if not isinstance(result.reason, str):
        raise MaterializationValidationError("renderer result reason is invalid")
    if not isinstance(result.provenance, Mapping):
        raise MaterializationValidationError("renderer result provenance is invalid")
    _validate_renderer_artifacts(result.artifacts)
    _validate_renderer_diagnostics(result.diagnostics)
    if result.status in {"complete", "partial"} and not result.artifacts:
        raise MaterializationValidationError("renderer result has no artifacts")
    return result


def _figure_artifact_digest(figure: Any) -> str:
    """Validate the replay renderer's figure result and return its digest.

    Returns:
        The normalized SHA-256 digest declared by the figure result.
    """

    if not isinstance(getattr(figure, "artifact_type", None), str) or (
        figure.artifact_type != "trajectory"
    ):
        raise MaterializationValidationError("replay renderer artifact type is invalid")
    for field_name in ("path", "format", "stamp_text"):
        if not isinstance(getattr(figure, field_name, None), str):
            raise MaterializationValidationError(
                f"replay renderer artifact {field_name} is invalid"
            )
    digest = getattr(figure, "sha256", None)
    if not _valid_sha256(digest):
        raise MaterializationValidationError("replay renderer artifact digest is invalid")
    return digest.lower()


def _result(  # noqa: PLR0913 - result envelope keeps all diagnostic fields explicit
    *,
    status: str,
    kind: str,
    fidelity: str,
    episode_id: str,
    cache_key: str,
    source_digest: str = "",
    output_directory: Path | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    reason: str = "",
) -> MaterializationResult:
    """Construct a consistently bounded result envelope.

    Returns:
        A JSON-safe diagnostic-only result.
    """

    return MaterializationResult(
        status=status,
        materialization_kind=kind,
        fidelity=fidelity,
        episode_id=episode_id,
        cache_key=cache_key,
        source_digest=source_digest,
        output_directory=str(output_directory) if output_directory is not None else None,
        artifacts=tuple(dict(item) for item in artifacts),
        diagnostics=tuple(str(item)[:MAX_TEXT] for item in diagnostics),
        provenance=dict(provenance or {}),
        reason=str(reason)[:MAX_TEXT],
    )


def _original_result(
    original: _OriginalRecording,
    *,
    episode_values: Mapping[str, str],
    cache_key: str,
    diagnostic: str | None,
) -> MaterializationResult:
    """Build a historical-original result without copying source bytes.

    Returns:
        A verified reference to the original recording bytes.
    """

    descriptor = original.descriptor
    artifact_id = str(descriptor.get("artifact_id", "original-recording"))
    uri = original.uri
    artifact = {
        "artifact_id": artifact_id,
        "uri": uri,
        "sha256": original.read.sha256,
        "materialization_kind": HISTORICAL_ORIGINAL,
    }
    diagnostics = (diagnostic,) if diagnostic else ()
    provenance = {
        "source_path": str(original.read.path),
        "source_uri": uri,
        "source_identity": dict(original.identity),
        "episode_identity": dict(episode_values),
        "source_bytes_preserved": False,
        "source_bytes_verified": True,
        "source_reference_mutable": True,
        "source_reference_binding": "normalized_uri_rechecked",
        "renderer_invoked": False,
        "evidence_boundary": "diagnostic_only",
        "claim_boundary": (
            "original bytes verified at admission; returned URI is mutable and must be "
            "revalidated by consumers; not new benchmark evidence"
        ),
    }
    return _result(
        status="complete",
        kind=HISTORICAL_ORIGINAL,
        fidelity=FIDELITY_VERIFIED,
        episode_id=episode_values["episode_id"],
        cache_key=cache_key,
        source_digest=original.read.sha256,
        artifacts=(artifact,),
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _render_replay_states(  # noqa: PLR0913 - native provenance accompanies existing render inputs
    episode: Mapping[str, Any],
    states: Sequence[Any],
    *,
    output_root: Path | MaterializationOutputRoot,
    output_relative: str,
    cache_key: str,
    diagnostics: Sequence[str],
    render_config: Mapping[str, Any] | None,
    native_trace_digest: str | None = None,
    retained_source_file_digest: str | None = None,
) -> MaterializationResult:
    """Render retained replay states through the existing figure renderer.

    Returns:
        A derived-render result, or an unavailable result when rendering fails.
    """

    episode_values = _episode_values(episode)
    replay = _replay_episode(episode, states)
    state_digest = _sha256(_canonical_bytes(list(states)))
    _output_path(output_root, output_relative)
    lease = _open_output_lease(output_root, output_relative)
    try:
        _bind_output_directory(lease, create=True)
        return _render_replay_states_with_lease(
            episode_values=episode_values,
            replay=replay,
            state_digest=state_digest,
            cache_key=cache_key,
            diagnostics=diagnostics,
            render_config=render_config,
            native_trace_digest=native_trace_digest,
            retained_source_file_digest=retained_source_file_digest,
            lease=lease,
        )
    finally:
        lease.close()


def _render_replay_states_with_lease(  # noqa: PLR0913 - keeps digest identities explicit
    *,
    episode_values: Mapping[str, str],
    replay: ReplayEpisode,
    state_digest: str,
    cache_key: str,
    diagnostics: Sequence[str],
    render_config: Mapping[str, Any] | None,
    native_trace_digest: str | None,
    retained_source_file_digest: str | None,
    lease: _OutputLease,
) -> MaterializationResult:
    """Render replay state through a retained descriptor-backed output lease.

    Returns:
        A derived-render result, or an unavailable result when rendering fails.
    """

    del render_config  # The replay renderer has a deliberately narrow API.
    source_digest = native_trace_digest or state_digest
    output = lease.path
    staging_parent_fd = -1
    figure_digest = ""
    try:
        staging_parent_fd = _open_private_staging_parent(lease.root_fd, lease.root_identity)
        with tempfile.TemporaryDirectory(
            prefix=".audit-materialize-render-",
            dir=str(_descriptor_path(staging_parent_fd)),
        ) as render_root_name:
            render_root = Path(render_root_name)
            render_root_fd = os.open(
                render_root.name,
                _output_directory_flags(),
                dir_fd=staging_parent_fd,
            )
            try:
                artifact_path = _descriptor_path(render_root_fd) / "trajectory.png"
                figure = generate_trajectory(replay, artifact_path, fmt="png")
                figure_digest = _figure_artifact_digest(figure)
                _publish_staged_files(lease, render_root_fd)
            finally:
                os.close(render_root_fd)
    except Exception as error:  # noqa: BLE001 - renderer failures fail closed
        # Preserve renderer-created partial files for diagnosis.  No manifest is
        # published, so a retry must select a fresh output directory.
        return _result(
            status="failed",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            source_digest=source_digest,
            output_directory=output,
            diagnostics=(*diagnostics, f"retained_replay_render_failed:{type(error).__name__}"),
            reason="retained_replay_render_failed",
        )
    finally:
        if staging_parent_fd >= 0:
            os.close(staging_parent_fd)
    artifacts = (
        {
            "artifact_id": "trajectory.png",
            "uri": "trajectory.png",
            "sha256": figure_digest,
            "materialization_kind": DERIVED_RENDER,
        },
    )
    provenance = {
        "source_identity": dict(episode_values),
        "retained_state_digest": state_digest,
        "renderer": "robot_sf.benchmark.episode_replay_figure.generate_trajectory",
        "renderer_version": MATERIALIZATION_TOOL_VERSION,
        "renderer_invoked": True,
        "simulation_advanced": False,
        "evidence_boundary": "diagnostic_only",
        "claim_boundary": "derived render of retained states; not historical original or benchmark evidence",
    }
    if native_trace_digest is not None:
        provenance["native_trace_digest"] = native_trace_digest
    if retained_source_file_digest is not None:
        provenance["retained_source_file_digest"] = retained_source_file_digest
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "materialization_kind": DERIVED_RENDER,
        "fidelity": FIDELITY_UNVERIFIABLE,
        "episode_id": episode_values["episode_id"],
        "cache_key": cache_key,
        "source_digest": source_digest,
        "artifacts": list(artifacts),
        "provenance": provenance,
        "diagnostics": list(diagnostics),
    }
    try:
        _assert_output_lease_current(lease)
        _verify_output_artifacts(lease.directory_fd, artifacts)
        manifest_digest = _exclusive_json(
            lease.root,
            lease.relative,
            manifest,
            expected_identity=lease.directory_identity,
            directory_fd=lease.directory_fd,
        )
        _assert_output_lease_current(lease)
        _verify_output_artifacts(lease.directory_fd, artifacts)
        _verify_output_file(lease.directory_fd, "audit-materialization.v1.json", manifest_digest)
    except Exception as error:  # noqa: BLE001 - publication failures fail closed
        try:
            os.unlink("audit-materialization.v1.json", dir_fd=lease.directory_fd)
        except (FileNotFoundError, OSError):
            pass
        return _result(
            status="failed",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            source_digest=source_digest,
            output_directory=output,
            diagnostics=(*diagnostics, f"manifest_write_failed:{type(error).__name__}"),
            reason="manifest_write_failed",
        )
    return _result(
        status="complete",
        kind=DERIVED_RENDER,
        fidelity=FIDELITY_UNVERIFIABLE,
        episode_id=episode_values["episode_id"],
        cache_key=cache_key,
        source_digest=source_digest,
        output_directory=output,
        artifacts=(
            *artifacts,
            {
                "artifact_id": "audit-materialization.v1.json",
                "uri": "audit-materialization.v1.json",
                "sha256": manifest_digest,
            },
        ),
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _render_trace_payload(
    episode: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    output_root: Path | MaterializationOutputRoot,
    output_relative: str,
    cache_key: str,
    diagnostics: Sequence[str],
    render_config: Mapping[str, Any] | None,
    source_digest: str | None = None,
) -> MaterializationResult:
    """Render retained trace state through a retained output-root descriptor.

    Returns:
        A derived-render result, or an unavailable result when rendering fails.
    """

    _output_path(output_root, output_relative)
    lease = _open_output_lease(output_root, output_relative)
    try:
        _retain_output_parent(lease)
        _bind_output_directory(lease, create=True)
        return _render_trace_payload_with_lease(
            episode,
            payload,
            output_root=output_root,
            output_relative=output_relative,
            cache_key=cache_key,
            diagnostics=diagnostics,
            render_config=render_config,
            source_digest=source_digest,
            lease=lease,
        )
    finally:
        lease.close()


def _render_trace_payload_with_lease(  # noqa: PLR0913, PLR0915
    episode: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    output_root: Path | MaterializationOutputRoot,
    output_relative: str,
    cache_key: str,
    diagnostics: Sequence[str],
    render_config: Mapping[str, Any] | None,
    source_digest: str | None = None,
    lease: _OutputLease,
) -> MaterializationResult:
    """Render one retained canonical trace through SREV-09.

    Returns:
        A derived-render result, or an unavailable result when rendering fails.
    """

    episode_values = _episode_values(episode)
    trace_bytes = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )
    normalized_trace_digest = _sha256(trace_bytes)
    trace_digest = source_digest or normalized_trace_digest
    output = lease.path
    renderer_output_relative = "rendered"
    staging_parent_fd = -1
    normalized_rendered_artifacts: tuple[dict[str, Any], ...] = ()
    renderer_diagnostics: tuple[str, ...] = ()
    try:
        staging_parent_fd = _open_private_staging_parent(lease.root_fd, lease.root_identity)
        with tempfile.TemporaryDirectory(
            prefix=".audit-materialize-source-",
            dir=str(lease.renderer_root),
        ) as source_dir:
            source_base = Path(source_dir)
            source_path = source_base / "retained-trace.json"
            with source_path.open("xb") as stream:
                stream.write(trace_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            source_ref = SourceRef(
                artifact_id="retained-trace",
                uri="retained-trace.json",
                format=SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
                schema=SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
                sha256=normalized_trace_digest,
            )
            typed = simulation_trace_export_from_dict(payload)
            with tempfile.TemporaryDirectory(
                prefix=".audit-materialize-render-",
                dir=str(_descriptor_path(staging_parent_fd)),
            ) as render_root_name:
                render_root = Path(render_root_name)
                render_root_fd = os.open(
                    render_root.name,
                    _output_directory_flags(),
                    dir_fd=staging_parent_fd,
                )
                try:
                    renderer_base = _descriptor_path(render_root_fd)
                    request = ComponentRequest(
                        request_id=f"materialize-{cache_key[:16]}",
                        component_id=review_scene.COMPONENT_ID,
                        sources=(source_ref,),
                        output_directory=renderer_output_relative,
                        config=_render_config(render_config, len(typed.frames)),
                    )
                    rendered = _validate_renderer_result(
                        review_scene.run(
                            request,
                            base=renderer_base,
                            source_base=source_base,
                        ),
                        expected_request_id=request.request_id,
                        expected_component_id=request.component_id,
                    )
                    renderer_diagnostics = tuple(
                        str(item.get("reason", "")) for item in rendered.diagnostics
                    )
                    if rendered.status in {"complete", "partial"}:
                        normalized_rendered_artifacts = _component_artifacts(
                            rendered.artifacts, output_relative=renderer_output_relative
                        )
                        staged_output_fd = _walk_output_from_fd(
                            render_root_fd,
                            (renderer_output_relative,),
                            create=False,
                        )
                        try:
                            _publish_staged_files(lease, staged_output_fd)
                        finally:
                            os.close(staged_output_fd)
                finally:
                    os.close(render_root_fd)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        SimulationTraceExportValidationError,
    ) as error:
        return _result(
            status="unavailable",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            source_digest=trace_digest,
            diagnostics=(*diagnostics, f"retained_trace_render_failed:{type(error).__name__}"),
            reason="retained_trace_render_failed",
        )
    except Exception as error:  # noqa: BLE001 - renderer failures fail closed
        # Preserve renderer-created partial files for diagnosis.  No manifest is
        # published, so a retry must select a fresh output directory.
        return _result(
            status="unavailable",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            source_digest=trace_digest,
            output_directory=output,
            diagnostics=(*diagnostics, f"retained_trace_render_failed:{type(error).__name__}"),
            reason="retained_trace_render_failed",
        )
    finally:
        if staging_parent_fd >= 0:
            os.close(staging_parent_fd)
    if rendered.status not in {"complete", "partial"}:
        return _result(
            status="unavailable",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            source_digest=trace_digest,
            diagnostics=(*diagnostics, rendered.reason or "retained_trace_renderer_unavailable"),
            reason="retained_trace_renderer_unavailable",
        )
    artifacts = tuple(
        {
            **dict(artifact),
            "materialization_kind": DERIVED_RENDER,
        }
        for artifact in normalized_rendered_artifacts
    )
    provenance = {
        "source_identity": dict(episode_values),
        "retained_trace_digest": trace_digest,
        "renderer": "robot_sf.render.review_scene.run",
        "renderer_version": review_scene.COMPONENT_VERSION,
        "renderer_invoked": True,
        "simulation_advanced": False,
        "evidence_boundary": "diagnostic_only",
        "claim_boundary": "derived render of retained states; not historical original or benchmark evidence",
    }
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "materialization_kind": DERIVED_RENDER,
        "fidelity": FIDELITY_UNVERIFIABLE,
        "episode_id": episode_values["episode_id"],
        "cache_key": cache_key,
        "source_digest": trace_digest,
        "artifacts": list(artifacts),
        "provenance": provenance,
        "diagnostics": [*diagnostics, *renderer_diagnostics],
    }
    try:
        _assert_output_lease_current(lease)
        _verify_output_artifacts(lease.directory_fd, artifacts)
        manifest_digest = _exclusive_json(
            lease.root,
            lease.relative,
            manifest,
            expected_identity=lease.directory_identity,
            directory_fd=lease.directory_fd,
        )
        _assert_output_lease_current(lease)
        _verify_output_artifacts(lease.directory_fd, artifacts)
        _verify_output_file(lease.directory_fd, "audit-materialization.v1.json", manifest_digest)
    except Exception as error:  # noqa: BLE001 - publication failures fail closed
        try:
            os.unlink("audit-materialization.v1.json", dir_fd=lease.directory_fd)
        except (FileNotFoundError, OSError):
            pass
        return _result(
            status="failed",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            source_digest=trace_digest,
            output_directory=output,
            diagnostics=(*diagnostics, f"manifest_write_failed:{type(error).__name__}"),
            reason="manifest_write_failed",
        )
    return _result(
        status=rendered.status,
        kind=DERIVED_RENDER,
        fidelity=FIDELITY_UNVERIFIABLE,
        episode_id=episode_values["episode_id"],
        cache_key=cache_key,
        source_digest=trace_digest,
        output_directory=output,
        artifacts=(
            *artifacts,
            {
                "artifact_id": "audit-materialization.v1.json",
                "uri": "audit-materialization.v1.json",
                "sha256": manifest_digest,
            },
        ),
        diagnostics=(*diagnostics, *renderer_diagnostics),
        provenance=provenance,
    )


def materialize_episode(  # noqa: C901, PLR0912, PLR0915 - ordered fail-closed adapter
    episode: Mapping[str, Any] | MaterializationRequest | Any,
    *,
    source_root: str | Path | MaterializationSourceRoot | None = None,
    output_root: str | Path | MaterializationOutputRoot | None = None,
    output_directory: str | None = None,
    render_config: Mapping[str, Any] | None = None,
    recording: Mapping[str, Any] | SourceRef | None = None,
    _trusted_scan_identity_defaults: Mapping[str, Any] | None = None,
) -> MaterializationResult:
    """Materialize one selected episode through the source-first order.

    The order is strict: verified original recording, retained canonical trace,
    retained replay states, then explicit unavailable for missing state/exact
    inputs.  No path is guessed and no simulator/runner is called.

    Args:
        episode: Selected audit row, or a :class:`MaterializationRequest`.
        source_root: Allowed root for explicitly declared source URIs.
        output_root: Allowed root for derived render output.
        output_directory: New relative output directory under ``output_root``.
        render_config: Optional allowlisted SREV scene configuration.
        recording: Explicit original-recording declaration, useful when the
            service stores it separately from the episode row.
        _trusted_scan_identity_defaults: Scanner-derived identity defaults,
            passed only by the service after scanning an untrusted source.
            Marker-looking fields inside ``episode`` are never trusted.

    Returns:
        A diagnostic-only result classified as historical, derived, or unavailable.
    """

    try:
        if _trusted_scan_identity_defaults is not None:
            if not isinstance(_trusted_scan_identity_defaults, Mapping) or any(
                key not in {"campaign_id", "source_digest", "execution_id"}
                or not isinstance(value, str)
                or not value
                for key, value in _trusted_scan_identity_defaults.items()
            ):
                raise MaterializationValidationError("trusted scan identity defaults are malformed")
            _trusted_scan_identity_defaults = dict(_trusted_scan_identity_defaults)
        if isinstance(episode, MaterializationRequest):
            request = episode
            episode_map = _mapping(request.episode, name="episode")
            source_root = request.source_root
            output_root = request.output_root
            output_directory = request.output_directory
            render_config = request.render_config
        else:
            if hasattr(episode, "row") and isinstance(episode.row, Mapping):
                row = dict(episode.row)
                row_episode_id = episode.episode_id if hasattr(episode, "episode_id") else None
                if "episode_id" not in row and isinstance(row_episode_id, str):
                    row["episode_id"] = row_episode_id
                episode = row
            episode_map = _mapping(episode, name="episode")
        episode_map.pop("_audit_scan_identity_defaults", None)
        episode_values = _episode_values(episode_map)
    except MaterializationValidationError as error:
        return _result(
            status="unavailable",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id="",
            cache_key=_sha256(
                _canonical_bytes(
                    {"schema_version": MATERIALIZATION_SCHEMA_VERSION, "invalid": str(error)}
                )
            ),
            reason="invalid_episode",
            diagnostics=(str(error),),
        )
    if source_root is None:
        source_root = Path.cwd()
    if output_root is None:
        output_root = Path.cwd() / "output"
    try:
        source_root_path: Path | MaterializationSourceRoot = (
            source_root
            if isinstance(source_root, MaterializationSourceRoot)
            else Path(source_root).absolute()
        )
        output_root_path: Path | MaterializationOutputRoot = (
            output_root
            if isinstance(output_root, MaterializationOutputRoot)
            else Path(output_root).absolute()
        )
        relative_output = _safe_output_directory(output_directory)
        descriptor_value = _recording_descriptor(episode_map, recording)
        declared_source_digest = ""
        if isinstance(descriptor_value, Mapping):
            candidate_digest = descriptor_value.get("sha256", descriptor_value.get("digest", ""))
            declared_source_digest = candidate_digest if _valid_sha256(candidate_digest) else ""
        cache_key = build_materialization_cache_key(
            episode_map,
            source_digest=declared_source_digest,
            render_config=render_config,
        )
        original, original_reason = _verify_original(
            episode_map,
            descriptor_value,
            source_root=source_root_path,
        )
        if original is not None:
            return _original_result(
                original,
                episode_values=episode_values,
                cache_key=cache_key,
                diagnostic=None,
            )
        fallback_diagnostics = (original_reason,) if original_reason else ()

        trace_payload = _retained_trace(episode_map)
        if (
            trace_payload is not None
            and trace_payload.get("schema_version") == ANALYSIS_TRACE_RECORD_SCHEMA_VERSION
        ):
            try:
                states = _native_trace_states(
                    episode_map,
                    trace_payload,
                    trusted_scan_identity_defaults=_trusted_scan_identity_defaults,
                )
                trace_digest = str(trace_payload["artifact_sha256"])
                cache_key = build_materialization_cache_key(
                    episode_map,
                    source_digest=trace_digest,
                    render_config=render_config,
                )
                if relative_output is None:
                    relative_output = _default_output_directory(
                        episode_values["episode_id"], cache_key
                    )
                output_root_path = _prepare_output_root(output_root_path)
                return _render_replay_states(
                    episode_map,
                    states,
                    output_root=output_root_path,
                    output_relative=relative_output,
                    cache_key=cache_key,
                    diagnostics=(*fallback_diagnostics, "native_retained_trace_projected"),
                    render_config=render_config,
                    native_trace_digest=trace_digest,
                )
            except (MaterializationValidationError, TypeError, ValueError) as error:
                return _result(
                    status="unavailable",
                    kind=UNAVAILABLE,
                    fidelity=FIDELITY_UNAVAILABLE,
                    episode_id=episode_values["episode_id"],
                    cache_key=cache_key,
                    diagnostics=(*fallback_diagnostics, str(error)),
                    reason="native_retained_trace_unavailable",
                )
        if trace_payload is not None:
            try:
                normalized_trace = _trace_payload(episode_map, trace_payload)
                trace_digest = _sha256(
                    json.dumps(normalized_trace, sort_keys=True, indent=2, allow_nan=False).encode(
                        "utf-8"
                    )
                    + b"\n"
                )
                cache_key = build_materialization_cache_key(
                    episode_map,
                    source_digest=trace_digest,
                    render_config=render_config,
                )
                if relative_output is None:
                    relative_output = _default_output_directory(
                        episode_values["episode_id"], cache_key
                    )
                output_root_path = _prepare_output_root(output_root_path)
                return _render_trace_payload(
                    episode_map,
                    normalized_trace,
                    output_root=output_root_path,
                    output_relative=relative_output,
                    cache_key=cache_key,
                    diagnostics=fallback_diagnostics,
                    render_config=render_config,
                )
            except (MaterializationValidationError, TypeError, ValueError) as error:
                return _result(
                    status="unavailable",
                    kind=UNAVAILABLE,
                    fidelity=FIDELITY_UNAVAILABLE,
                    episode_id=episode_values["episode_id"],
                    cache_key=cache_key,
                    diagnostics=(*fallback_diagnostics, str(error)),
                    reason="retained_trace_unavailable",
                )

        path_declaration = _retained_trace_path(episode_map)
        if path_declaration is not None:
            try:
                uri = _recording_uri(path_declaration)
                expected_digest = path_declaration.get("sha256", path_declaration.get("digest"))
                if not _valid_sha256(expected_digest):
                    raise MaterializationValidationError("retained trace source digest is required")
                read = _open_read_bounded(source_root_path, uri)
                if read.sha256.lower() != str(expected_digest).lower():
                    raise MaterializationValidationError("retained trace source digest mismatch")
                payload = json.loads(read.payload)
                if not isinstance(payload, Mapping):
                    raise MaterializationValidationError("retained trace source must be an object")
                if payload.get("schema_version") == ANALYSIS_TRACE_RECORD_SCHEMA_VERSION:
                    states = _native_trace_states(
                        episode_map,
                        payload,
                        trusted_scan_identity_defaults=_trusted_scan_identity_defaults,
                    )
                    trace_digest = str(payload["artifact_sha256"])
                    cache_key = build_materialization_cache_key(
                        episode_map,
                        source_digest=trace_digest,
                        render_config=render_config,
                    )
                    if relative_output is None:
                        relative_output = _default_output_directory(
                            episode_values["episode_id"], cache_key
                        )
                    output_root_path = _prepare_output_root(output_root_path)
                    return _render_replay_states(
                        episode_map,
                        states,
                        output_root=output_root_path,
                        output_relative=relative_output,
                        cache_key=cache_key,
                        diagnostics=(*fallback_diagnostics, "native_retained_trace_projected"),
                        render_config=render_config,
                        native_trace_digest=trace_digest,
                        retained_source_file_digest=read.sha256,
                    )
                normalized_trace = _trace_payload(episode_map, payload)
                cache_key = build_materialization_cache_key(
                    episode_map,
                    source_digest=read.sha256,
                    render_config=render_config,
                )
                if relative_output is None:
                    relative_output = _default_output_directory(
                        episode_values["episode_id"], cache_key
                    )
                output_root_path = _prepare_output_root(output_root_path)
                return _render_trace_payload(
                    episode_map,
                    normalized_trace,
                    output_root=output_root_path,
                    output_relative=relative_output,
                    cache_key=cache_key,
                    diagnostics=(*fallback_diagnostics, "retained_trace_source_verified"),
                    render_config=render_config,
                    source_digest=read.sha256,
                )
            except (
                MaterializationValidationError,
                FileNotFoundError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                fallback_diagnostics = (
                    *fallback_diagnostics,
                    f"retained_trace_source_unavailable:{type(error).__name__}",
                )

        states = _retained_steps(episode_map)
        if states is not None:
            try:
                state_digest = _sha256(_canonical_bytes(list(states)))
                cache_key = build_materialization_cache_key(
                    episode_map,
                    source_digest=state_digest,
                    render_config=render_config,
                )
                if relative_output is None:
                    relative_output = _default_output_directory(
                        episode_values["episode_id"], cache_key
                    )
                output_root_path = _prepare_output_root(output_root_path)
                return _render_replay_states(
                    episode_map,
                    states,
                    output_root=output_root_path,
                    output_relative=relative_output,
                    cache_key=cache_key,
                    diagnostics=fallback_diagnostics,
                    render_config=render_config,
                )
            except (MaterializationValidationError, TypeError, ValueError) as error:
                return _result(
                    status="unavailable",
                    kind=UNAVAILABLE,
                    fidelity=FIDELITY_UNAVAILABLE,
                    episode_id=episode_values["episode_id"],
                    cache_key=cache_key,
                    diagnostics=(*fallback_diagnostics, str(error)),
                    reason="retained_states_unavailable",
                )

        exact_inputs = any(
            _identity_value(episode_map, field_name)
            for field_name in (
                "config_identity",
                "source_commit",
                "checkpoint_digest",
                "environment_digest",
            )
        ) or any(
            key in episode_map for key in ("scenario_params", "planner_config", "initial_state")
        )
        reason = "exact_input_execution_deferred" if exact_inputs else "retained_state_missing"
        return _result(
            status="unavailable",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=cache_key,
            diagnostics=(*fallback_diagnostics, "simulation_not_started"),
            provenance={"simulation_advanced": False, "execution_deferred": exact_inputs},
            reason=reason,
        )
    except (
        MaterializationValidationError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return _result(
            status="unavailable",
            kind=UNAVAILABLE,
            fidelity=FIDELITY_UNAVAILABLE,
            episode_id=episode_values["episode_id"],
            cache_key=_sha256(
                _canonical_bytes({"episode": episode_values, "error": type(error).__name__})
            ),
            diagnostics=(str(error),),
            reason="materialization_unavailable",
        )


__all__ = [
    "DERIVED_RENDER",
    "FIDELITY_DIVERGED",
    "FIDELITY_UNAVAILABLE",
    "FIDELITY_UNVERIFIABLE",
    "FIDELITY_VERIFIED",
    "HISTORICAL_ORIGINAL",
    "MATERIALIZATION_SCHEMA_VERSION",
    "MATERIALIZATION_TOOL_VERSION",
    "UNAVAILABLE",
    "MaterializationOutputRoot",
    "MaterializationRequest",
    "MaterializationResult",
    "MaterializationSourceRoot",
    "MaterializationValidationError",
    "admit_output_root",
    "admit_source_root",
    "build_materialization_cache_key",
    "materialize_episode",
]
