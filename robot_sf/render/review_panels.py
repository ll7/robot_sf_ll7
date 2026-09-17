"""Offline synchronized scene/video/metric/event review panels (SREV-16).

The component is deliberately renderer-neutral at its Python boundary.  It consumes
already-recorded, explicit source-time rows and emits one immutable panel model.  A
small browser component renders that model and owns transient interaction state.  The
simulation timeline is the only clock: video rows must carry an explicit mapping to
source time and are never aligned from a nominal frame rate.

This module does not launch a simulator, planner, AI provider, network request, or
annotation writer.  Source files are read-only and every unavailable field remains
visible in the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
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

# Keep these canonical source tokens local so importing this offline component
# does not pull optional pygame/moviepy dependencies from the legacy viewer.
# The adapters below consume the same versioned payloads owned by these modules.
EVENT_INDEX_FILENAME = "event-index.json"
SIMULATION_TIMELINE_SCHEMA_VERSION = "simulation_timeline.v1"
ANALYSIS_TRACE_RECORD_SCHEMA_VERSION = "analysis-trace.v1"
FAILURE_DIAGNOSIS_SCHEMA_VERSION = "failure_diagnosis.v1"
SCENE_SCHEMA_VERSION = "threejs-viewer.v1"

COMPONENT_ID = "srev16-review-panels"
COMPONENT_VERSION = "1.0.0"
PANEL_MODEL_SCHEMA_VERSION = "review-panels.v1"
CONTEXT_SCHEMA_VERSION = "review-context.v1"
MISSING_CAPABILITY_SCHEMA_VERSION = "missing-capability-report.v1"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

SCENE_FORMATS = frozenset(
    {
        "scene",
        "scene.v1",
        SCENE_SCHEMA_VERSION,
        "simulation-timeline",
        "simulation-timeline.v1",
        SIMULATION_TIMELINE_SCHEMA_VERSION,
        "simulation_trace_export.v1",
        "simulation-trace-export.v1",
        "analysis-trace.v1",
        ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
        "recording-jsonl.v1",
        "jsonl-recording.v1",
    }
)
VIDEO_FORMATS = frozenset(
    {
        "video",
        "video-mp4.v1",
        "video-frames.v1",
        "media-mapping",
        "media-mapping.v1",
        "video-sync",
        "video-sync.v1",
    }
)
METRIC_FORMATS = frozenset(
    {
        "metric",
        "metrics",
        "metric-series",
        "metric-series.v1",
        "telemetry",
        "telemetry.v1",
        "analysis-trace.v1",
        ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
        "simulation-timeline",
        "simulation-timeline.v1",
        SIMULATION_TIMELINE_SCHEMA_VERSION,
    }
)
EVENT_FORMATS = frozenset(
    {
        "event",
        "events",
        "event-list",
        "event-list.v1",
        "event-index",
        "event-index.v1",
        EVENT_INDEX_FILENAME.removesuffix(".json"),
        "phase-list",
        "phase-list.v1",
        "failure-diagnosis",
        FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        "simulation-timeline",
        "simulation-timeline.v1",
        SIMULATION_TIMELINE_SCHEMA_VERSION,
        "analysis-trace.v1",
        ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
        SCENE_SCHEMA_VERSION,
        "simulation_trace_export.v1",
    }
)

REQUIRED_CAPABILITIES = ("source-time-cursor", "synchronized-panels")
OPTIONAL_CAPABILITIES = (
    "scene",
    "video",
    "metrics",
    "events",
    "goal-geometry",
    "actor-identity",
    "interval-selection",
)
OUTPUT_TYPES = (PANEL_MODEL_SCHEMA_VERSION, MISSING_CAPABILITY_SCHEMA_VERSION)

DEFAULT_METRIC_IDS = ("goal_distance", "speed", "angular_velocity", "clearance")
DEFAULT_SAMPLE_RESOLUTION_S = 0.0
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SOURCES = 32
MAX_SAMPLES = 100_000
MAX_EVENTS = 100_000
MAX_METRICS = 256
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_IDENTITY_LENGTH = 512

_DESCRIPTOR_DOCUMENT = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": list(OUTPUT_TYPES),
}
component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)
DESCRIPTOR: ComponentDescriptor = component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)

OUTPUT_MODEL_FILENAME = f"{PANEL_MODEL_SCHEMA_VERSION}.json"
OUTPUT_HTML_FILENAME = f"{PANEL_MODEL_SCHEMA_VERSION}.html"


@dataclass(frozen=True, slots=True)
class Sample:
    """One explicit source-time sample used by nearest-sample alignment."""

    time_s: float
    value: Any
    source_index: int
    missing: bool = False


@dataclass(frozen=True, slots=True)
class SourceTimeCursor:
    """Immutable shared cursor passed to every panel consumer."""

    time_s: float
    context_revision: int = 0
    source: str = "initial"
    interval_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned cursor shape used by the browser and agents."""

        return {
            "time_s": self.time_s,
            "context_revision": self.context_revision,
            "source": self.source,
            "interval_id": self.interval_id,
        }


@dataclass(frozen=True, slots=True)
class ReviewContext:
    """Episode-scoped selection snapshot with stale-consumer protection."""

    campaign_id: str | None = None
    execution_id: str | None = None
    episode_id: str | None = None
    interval_id: str | None = None
    actor_id: str | None = None
    cursor: SourceTimeCursor = field(default_factory=lambda: SourceTimeCursor(0.0))
    context_revision: int = 0

    def seek(self, time_s: float, *, source: str = "seek") -> ReviewContext:
        """Return a new context revision for a source-time seek."""

        _finite_number(time_s, "cursor time_s")
        return ReviewContext(
            campaign_id=self.campaign_id,
            execution_id=self.execution_id,
            episode_id=self.episode_id,
            interval_id=self.interval_id,
            actor_id=self.actor_id,
            cursor=SourceTimeCursor(
                float(time_s), self.context_revision + 1, source, self.interval_id
            ),
            context_revision=self.context_revision + 1,
        )

    def with_interval(self, interval_id: str | None) -> ReviewContext:
        """Return a new revision after selecting an interval."""

        revision = self.context_revision + 1
        return ReviewContext(
            campaign_id=self.campaign_id,
            execution_id=self.execution_id,
            episode_id=self.episode_id,
            interval_id=interval_id,
            actor_id=self.actor_id,
            cursor=SourceTimeCursor(self.cursor.time_s, revision, "interval", interval_id),
            context_revision=revision,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe selection snapshot."""

        return {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "execution_id": self.execution_id,
            "episode_id": self.episode_id,
            "interval_id": self.interval_id,
            "actor_id": self.actor_id,
            "cursor": self.cursor.to_dict(),
            "context_revision": self.context_revision,
        }


@dataclass(frozen=True, slots=True)
class NearestSample:
    """Nearest-sample result with the declared temporal resolution visible."""

    status: str
    target_time_s: float
    sample_time_s: float | None
    temporal_error_s: float | None
    resolution_s: float
    source_index: int | None = None
    value: Any = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the stable mapping used in panel snapshots."""

        return asdict(self)


@dataclass
class _Stream:
    """Internal normalized source stream."""

    name: str
    samples: list[Sample] = field(default_factory=list)
    resolution_s: float = DEFAULT_SAMPLE_RESOLUTION_S
    status: str = "unavailable"
    reason: str = "stream_not_declared"
    source_ids: list[str] = field(default_factory=list)
    source_identity: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, float]] = field(default_factory=list)
    media_uri: str | None = None
    surface: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LoadedSource:
    """Source bytes and integrity metadata used by all adapters."""

    ref: SourceRef
    payload: Any | None
    digest: str | None
    integrity: str
    availability: str
    identity: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


class _InputError(ValueError):
    """Stable component-owned input error."""


def _finite_number(value: Any, label: str = "value") -> float:
    """Return a finite float or raise a bounded input error."""

    if isinstance(value, bool):
        raise _InputError(f"invalid_input: {label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _InputError(f"invalid_input: {label} must be finite numeric") from error
    if not math.isfinite(number):
        raise _InputError(f"invalid_input: {label} must be finite numeric")
    return number


def _bounded_text(value: Any, default: str = "") -> str:
    """Return compact identity text without allowing unbounded diagnostics."""

    if not isinstance(value, str):
        return default
    return " ".join(value.split())[:MAX_IDENTITY_LENGTH]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_loads(raw: bytes) -> Any:
    """Parse bounded strict UTF-8 JSON.

    Returns:
        Parsed JSON value.
    """

    if len(raw) > MAX_JSON_BYTES:
        raise _InputError("resource_limit: source JSON is too large")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise _InputError("invalid_input: source is not strict UTF-8 JSON") from error


def _unsafe_path(value: str) -> bool:
    """Return whether a source/output URI is not a safe relative path."""

    candidate = Path(value)
    windows = PureWindowsPath(value)
    return (
        not value
        or candidate.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in candidate.parts
        or ".." in windows.parts
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _resolve_under(root: Path, value: str, *, kind: str) -> Path:
    """Resolve a path under ``root`` and reject symlink escapes.

    Returns:
        Resolved path below ``root``.
    """

    if _unsafe_path(value):
        raise _InputError(f"unsafe_{kind}_path: absolute, traversal, or control path rejected")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root / Path(value)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise _InputError(f"unsafe_{kind}_path: path escapes base") from error
    return resolved


def _read_regular(path: Path) -> bytes:
    """Read one bounded regular file without following a final symlink.

    Returns:
        Source bytes.
    """

    try:
        initial = path.stat()
    except OSError as error:
        raise _InputError("source_missing") from error
    if not stat.S_ISREG(initial.st_mode):
        raise _InputError("source_not_regular_file")
    if initial.st_size > MAX_SOURCE_BYTES:
        raise _InputError("resource_limit: source bytes")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise _InputError("source_not_regular_file")
        if opened.st_size > MAX_SOURCE_BYTES:
            raise _InputError("resource_limit: source bytes")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(MAX_SOURCE_BYTES + 1)
        if len(payload) > MAX_SOURCE_BYTES:
            raise _InputError("resource_limit: source bytes")
        return payload
    except OSError as error:
        raise _InputError("source_unreadable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _actor_alias_from_row(row: Mapping[str, Any]) -> Any:
    """Find an explicitly recorded actor alias in one bounded scene row.

    Returns:
        Recorded actor identifier, or ``None`` when the row has no alias.
    """

    state = row.get("state")
    state_robot = state.get("robot") if isinstance(state, Mapping) else None
    for candidate in (row, state, row.get("robot"), state_robot):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("actor_id", "actor", "ego_actor_id", "robot_actor_id"):
            value = candidate.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                return value
    actors = row.get("actors")
    if isinstance(actors, Mapping):
        for candidate in actors.values():
            if isinstance(candidate, Mapping):
                value = _actor_alias_from_row(candidate)
                if value is not None:
                    return value
    return None


def _identity(ref: SourceRef, payload: Any) -> dict[str, Any]:  # noqa: C901
    """Collect declared/embedded source identity without inventing values.

    Returns:
        Source identity fields that were explicitly declared or observed.
    """

    embedded: Mapping[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in ("source_identity", "source", "metadata"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                embedded = candidate
                break
        if not embedded and isinstance(payload.get("source_trace"), Mapping):
            trace = payload["source_trace"]
            if isinstance(trace.get("source"), Mapping):
                embedded = trace["source"]
            else:
                embedded = trace
        if not embedded:
            embedded = payload
    declared = {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "format": ref.format,
        "schema": ref.schema or None,
        "sha256": ref.sha256 or None,
        "source_commit": ref.source_commit or None,
        "config_identity": ref.config_identity or None,
        "units": ref.units or None,
        "coordinate_frame": ref.coordinate_frame or None,
    }
    aliases = {
        "episode_id": ("episode_id", "episode", "case_id"),
        "execution_id": ("execution_id", "run_id", "trace_id"),
        "campaign_id": ("campaign_id", "campaign"),
        "scenario_id": ("scenario_id", "scenario"),
        "planner_id": ("planner_id", "planner"),
        "seed": ("seed",),
        "source_commit": ("source_commit", "git_hash", "commit"),
        "config_identity": ("config_identity", "config_digest", "config_hash"),
        "coordinate_frame": ("coordinate_frame",),
        "units": ("units",),
        "actor_id": ("actor_id", "actor", "ego_actor_id", "robot_actor_id"),
        "media_uri": ("media_uri", "media_source_uri", "video_uri"),
    }
    result: dict[str, Any] = {key: value for key, value in declared.items() if value is not None}
    for output_key, keys in aliases.items():
        if output_key in result:
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
    if "actor_id" not in result and isinstance(payload, Mapping):
        for row in _rows(payload, "frames", "steps", "samples")[:1]:
            if not isinstance(row, Mapping):
                continue
            actor = _actor_alias_from_row(row)
            if isinstance(actor, (str, int, float)) and not isinstance(actor, bool):
                result["actor_id"] = actor
                break
    return result


def _load_sources(
    request: ComponentRequest, root: Path
) -> tuple[list[_LoadedSource], list[dict[str, Any]]]:
    """Read declared JSON source documents and retain integrity diagnostics.

    Returns:
        Loaded sources and bounded integrity/read diagnostics.
    """

    sources: list[_LoadedSource] = []
    diagnostics: list[dict[str, Any]] = []
    if len(request.sources) > MAX_SOURCES:
        raise _InputError(f"resource_limit: sources exceed {MAX_SOURCES}")
    for ref in request.sources:
        if not isinstance(ref, SourceRef):
            raise _InputError("invalid_input: source is not a SourceRef")
        try:
            path = _resolve_under(root, ref.uri, kind="source")
            raw = _read_regular(path)
            try:
                payload = _strict_loads(raw)
            except _InputError as error:
                # An opaque local media file is valid input, but remains
                # unavailable until an explicit source-time mapping is supplied.
                if _format_token(ref.format) in VIDEO_FORMATS and not str(error).startswith(
                    "resource_limit:"
                ):
                    payload = {}
                else:
                    raise
        except _InputError as error:
            reason = str(error)
            if reason.startswith("unsafe_source_path"):
                raise
            diagnostics.append(
                {"artifact_id": ref.artifact_id, "reason_code": reason, "detail": reason}
            )
            sources.append(
                _LoadedSource(
                    ref=ref,
                    payload=None,
                    digest=None,
                    integrity="unverifiable",
                    availability=reason,
                    identity={"artifact_id": ref.artifact_id, "uri": ref.uri, "format": ref.format},
                )
            )
            continue
        digest = _sha256(raw)
        integrity = (
            "unverifiable" if not ref.sha256 else ("match" if digest == ref.sha256 else "mismatch")
        )
        source = _LoadedSource(
            ref=ref,
            payload=payload,
            digest=digest,
            integrity=integrity,
            availability="ok",
            identity=_identity(ref, payload),
        )
        if integrity == "mismatch":
            source.diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason_code": "source_integrity_mismatch",
                    "detail": "declared sha256 does not match source bytes",
                }
            )
        sources.append(source)
    return sources, diagnostics


def _format(source: _LoadedSource) -> str:
    """Return a lower-case source format/schema token."""

    return (source.ref.format or source.ref.schema or "").strip().lower()


def _format_token(value: str) -> str:
    """Normalize a source format before a payload wrapper exists.

    Returns:
        Lower-case format token.
    """

    return str(value or "").strip().lower()


def _payload_schema(source: _LoadedSource) -> str:
    if isinstance(source.payload, Mapping):
        return str(source.payload.get("schema_version", "")).strip().lower()
    return ""


def _is_scene(source: _LoadedSource) -> bool:
    return _format(source) in SCENE_FORMATS or _payload_schema(source) in SCENE_FORMATS


def _is_video(source: _LoadedSource) -> bool:
    token = _format(source)
    return token in VIDEO_FORMATS or token.startswith("video-") or token.startswith("media-mapping")


def _is_metric(source: _LoadedSource) -> bool:
    return _format(source) in METRIC_FORMATS or _payload_schema(source) in METRIC_FORMATS


def _is_event(source: _LoadedSource) -> bool:
    token = _format(source)
    return token in EVENT_FORMATS or _payload_schema(source) in EVENT_FORMATS


def _rows(payload: Any, *keys: str) -> list[Any]:
    """Return the first list-valued row container without coercing indices to time."""

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _time_from(row: Mapping[str, Any], *, video: bool = False) -> float | None:
    """Extract an explicit source-time field; FPS/PTS alone is never enough.

    Returns:
        Explicit source time, or ``None`` when it is absent or invalid.
    """

    keys = (
        (
            "source_time_s",
            "source_t_s",
            "sim_time_s",
            "simulation_time_s",
            "time_s",
            "t_s",
            "timestamp_s",
        )
        if not video
        else (
            "source_time_s",
            "source_t_s",
            "sim_time_s",
            "simulation_time_s",
            "mapped_time_s",
        )
    )
    for key in keys:
        if key in row and row[key] is not None:
            try:
                return _finite_number(row[key], key)
            except _InputError:
                return None
    return None


def _sample_resolution(config: Mapping[str, Any], source: _LoadedSource) -> float | None:
    """Resolve a declared resolution, preserving zero for exact matching.

    Returns:
        Non-negative resolution in seconds, or ``None`` when undeclared.
    """

    values: list[Any] = []
    for key in ("sample_resolution_s", "nearest_sample_resolution_s", "max_sample_error_s"):
        if key in config:
            values.append(config[key])
            break
    if isinstance(source.payload, Mapping):
        for key in ("sample_resolution_s", "resolution_s", "nearest_sample_resolution_s"):
            if key in source.payload:
                values.append(source.payload[key])
                break
    if not values:
        return None
    value = _finite_number(values[-1], "sample resolution")
    if value < 0:
        raise _InputError("invalid_input: sample resolution must be non-negative")
    return value


def _derive_resolution(samples: Sequence[Sample], declared: float | None) -> float:
    if declared is not None:
        return declared
    spacings = [
        right.time_s - left.time_s
        for left, right in pairwise(samples)
        if right.time_s > left.time_s
    ]
    if not spacings:
        return DEFAULT_SAMPLE_RESOLUTION_S
    # The half-minimum rule is deterministic and makes a large source gap
    # visibly unavailable rather than silently holding a stale sample.
    return min(spacings) / 2.0


def _normalize_samples(  # noqa: C901
    rows: Sequence[Any],
    *,
    source: _LoadedSource,
    config: Mapping[str, Any],
    video: bool = False,
    value_only: bool = False,
) -> _Stream:
    """Normalize rows and retain missingness/regression diagnostics.

    Returns:
        Normalized stream with explicit status and diagnostics.
    """

    stream_name = "video" if video else "scene"
    stream = _Stream(
        name=stream_name, source_ids=[source.ref.artifact_id], source_identity=source.identity
    )
    samples: list[Sample] = []
    for index, raw in enumerate(rows[:MAX_SAMPLES]):
        if not isinstance(raw, Mapping):
            stream.diagnostics.append({"reason_code": "row_not_mapping", "source_index": index})
            continue
        time_s = _time_from(raw, video=video)
        if time_s is None:
            stream.diagnostics.append({"reason_code": "source_time_missing", "source_index": index})
            continue
        if value_only:
            value = raw.get("value")
        else:
            value = dict(raw)
            value.pop("source_time_s", None)
            value["time_s"] = time_s
        samples.append(
            Sample(
                time_s=time_s,
                value=value,
                source_index=index,
                missing=bool(
                    raw.get("missing", False)
                    or raw.get("available") is False
                    or ("value" in raw and raw.get("value") is None)
                    or (value_only and "value" not in raw)
                ),
            )
        )
    if len(rows) > MAX_SAMPLES:
        stream.diagnostics.append({"reason_code": "resource_limit:samples", "limit": MAX_SAMPLES})
    if not samples:
        stream.reason = "source_time_unavailable"
        stream.status = "unavailable"
        return stream
    invalid_order = False
    for left, right in pairwise(samples):
        if right.time_s < left.time_s:
            invalid_order = True
            stream.diagnostics.append({"reason_code": "source_time_decreased"})
        elif right.time_s == left.time_s and not (
            video and (_pause_marker(left.value) or _pause_marker(right.value))
        ):
            invalid_order = True
            stream.diagnostics.append({"reason_code": "source_time_duplicate_without_pause"})
    if invalid_order:
        stream.diagnostics.append({"reason_code": "source_time_not_nondecreasing"})
        stream.reason = "source_time_not_nondecreasing"
        stream.status = "unavailable"
        stream.samples = samples
        return stream
    stream.samples = samples
    stream.resolution_s = _derive_resolution(samples, _sample_resolution(config, source))
    for left, right in pairwise(samples):
        if right.time_s - left.time_s > max(stream.resolution_s * 2.0, 1e-12):
            stream.gaps.append(
                {
                    "start_s": left.time_s,
                    "end_s": right.time_s,
                    "duration_s": right.time_s - left.time_s,
                }
            )
    stream.status = "partial" if stream.diagnostics else "available"
    stream.reason = "" if stream.status == "available" else "source_rows_skipped"
    return stream


def _pause_marker(value: Any) -> bool:
    """Return whether a mapped video row explicitly declares a legal pause."""

    if not isinstance(value, Mapping):
        return False
    if value.get("is_pause") is True or value.get("pause") is True:
        return True
    if value.get("pause_duration_s") is not None:
        return True
    semantics = value.get("semantics", value.get("mapping_semantics"))
    if isinstance(semantics, Mapping):
        semantics = semantics.get("type", semantics.get("kind", semantics.get("mode")))
    if isinstance(semantics, Sequence) and not isinstance(semantics, (str, bytes)):
        return any(_pause_marker({"semantics": item}) for item in semantics)
    return isinstance(semantics, str) and semantics.strip().lower() in {
        "pause",
        "hold",
        "presentation_pause",
    }


def _media_time_from(row: Mapping[str, Any]) -> float | None:
    """Extract explicit presentation/media time for monotonicity checks.

    Returns:
        Finite media/presentation time, or ``None`` when it is not declared.
    """

    for key in ("presentation_t_s", "media_t_s", "media_time_s", "pts_s"):
        if key in row and row[key] is not None:
            try:
                return _finite_number(row[key], key)
            except _InputError:
                return None
    return None


def nearest_sample(stream: _Stream | Mapping[str, Any], target_time_s: float) -> NearestSample:
    """Select one nearest sample within the stream's declared resolution.

    Ties choose the earlier sample.  A missing sample is not replaced by a
    neighbour, and a sample outside the declared resolution is unavailable.

    Returns:
        Nearest sample metadata, or an explicit unavailable result.
    """

    target = _finite_number(target_time_s, "target time_s")
    if isinstance(stream, Mapping):
        samples = [
            Sample(
                time_s=float(row["time_s"]),
                value=row.get("value", row),
                source_index=int(row.get("source_index", index)),
                missing=bool(row.get("missing", False)),
            )
            for index, row in enumerate(stream.get("samples", []))
            if isinstance(row, Mapping) and row.get("time_s") is not None
        ]
        resolution = float(stream.get("resolution_s", DEFAULT_SAMPLE_RESOLUTION_S))
        status = str(stream.get("status", "available"))
        unavailable_reason = str(stream.get("reason", "stream_unavailable"))
    else:
        samples = stream.samples
        resolution = stream.resolution_s
        status = stream.status
        unavailable_reason = stream.reason or "stream_unavailable"
    if not samples or status == "unavailable":
        return NearestSample(
            "unavailable", target, None, None, resolution, reason=unavailable_reason
        )
    times = [sample.time_s for sample in samples]
    right = bisect_left(times, target)
    candidates: list[Sample] = []
    if right < len(samples):
        candidates.append(samples[right])
    if right > 0:
        candidates.append(samples[right - 1])
    selected = min(candidates, key=lambda sample: (abs(sample.time_s - target), sample.time_s))
    error = abs(selected.time_s - target)
    if error > resolution + 1e-12:
        return NearestSample(
            "unavailable",
            target,
            selected.time_s,
            error,
            resolution,
            source_index=selected.source_index,
            reason="outside_declared_resolution",
        )
    if selected.missing:
        return NearestSample(
            "unavailable",
            target,
            selected.time_s,
            error,
            resolution,
            source_index=selected.source_index,
            reason="sample_missing",
        )
    return NearestSample(
        "available",
        target,
        selected.time_s,
        error,
        resolution,
        source_index=selected.source_index,
        value=selected.value,
    )


def _unwrap_scene_rows(source: _LoadedSource) -> list[dict[str, Any]]:
    payload = source.payload
    rows = _rows(payload, "frames", "steps", "samples")
    if _payload_schema(source) == SIMULATION_TIMELINE_SCHEMA_VERSION:
        return [
            {
                **dict(frame),
                "value": frame.get("state", {}),
                "state": frame.get("state", {}),
                "time_s": frame.get("time_s"),
            }
            for frame in rows
            if isinstance(frame, Mapping)
        ]
    if _payload_schema(source) in {ANALYSIS_TRACE_RECORD_SCHEMA_VERSION, "analysis-trace.v1"}:
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _scene_stream(source: _LoadedSource, config: Mapping[str, Any]) -> _Stream:
    rows = _unwrap_scene_rows(source)
    normalized = _normalize_samples(rows, source=source, config=config)
    normalized.name = "scene"
    normalized.samples = [
        Sample(
            sample.time_s,
            _scene_value(sample.value),
            sample.source_index,
            sample.missing,
        )
        for sample in normalized.samples
    ]
    payload = source.payload if isinstance(source.payload, Mapping) else {}
    surface = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "map": payload.get("map"),
        "trajectory": payload.get("trajectory", []),
        "view": payload.get("view"),
        "fidelity": payload.get("fidelity"),
        "limitations": payload.get("limitations", []),
    }
    normalized.surface = {
        key: value for key, value in surface.items() if value not in (None, [], {})
    }
    return normalized


def _scene_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"value": value}
    state = value.get("state") if isinstance(value.get("state"), Mapping) else value
    result = dict(state)
    if "robot" not in result and isinstance(value.get("robot"), Mapping):
        result["robot"] = dict(value["robot"])
    if "pedestrians" not in result and isinstance(value.get("pedestrians"), list):
        result["pedestrians"] = value["pedestrians"]
    if "frame_index" in value:
        result["frame_index"] = value["frame_index"]
    if "frame_idx" in value:
        result["frame_idx"] = value["frame_idx"]
    if "step" in value:
        result["step"] = value["step"]
    return result


def _video_stream(source: _LoadedSource, config: Mapping[str, Any]) -> _Stream:  # noqa: C901
    rows = _rows(source.payload, "entries", "mappings", "frames", "samples", "mapping")
    if not rows:
        presentation = config.get("presentation")
        timestamp_map = (
            presentation.get("presentation_timestamp_map")
            if isinstance(presentation, Mapping)
            else None
        )
        if isinstance(timestamp_map, Mapping):
            rows = _rows(timestamp_map, "mapping", "mappings")
    stream = _normalize_samples(rows, source=source, config=config, video=True)
    stream.name = "video"
    presentation = config.get("presentation")
    media_uri = config.get("media_uri", config.get("video_uri"))
    if isinstance(presentation, Mapping) and not media_uri:
        media_uri = presentation.get("media_uri", presentation.get("video_uri"))
    if not media_uri:
        media_uri = source.identity.get("media_uri")
    if not media_uri and Path(source.ref.uri).suffix.lower() in {
        ".mp4",
        ".webm",
        ".ogg",
        ".ogv",
        ".mov",
        ".m4v",
    }:
        media_uri = source.ref.uri
    if isinstance(media_uri, str) and media_uri:
        stream.media_uri = media_uri
    media_times = [_media_time_from(row) if isinstance(row, Mapping) else None for row in rows]
    media_order_invalid = False
    for index, (left, right) in enumerate(pairwise(media_times)):
        if left is None or right is None:
            continue
        if right < left:
            media_order_invalid = True
            stream.diagnostics.append({"reason_code": "media_time_decreased", "index": index + 1})
        elif right == left:
            left_row = rows[index] if isinstance(rows[index], Mapping) else {}
            right_row = rows[index + 1] if isinstance(rows[index + 1], Mapping) else {}
            if not (_pause_marker(left_row) or _pause_marker(right_row)):
                media_order_invalid = True
                stream.diagnostics.append(
                    {
                        "reason_code": "media_time_duplicate_without_pause",
                        "index": index + 1,
                    }
                )
    if media_order_invalid:
        stream.status = "unavailable"
        stream.reason = "media_time_not_nondecreasing"
    # ``_normalize_samples`` intentionally does not accept presentation PTS as
    # source time.  Retain useful media fields while exposing the mapped time.
    return stream


def _metric_entries(source: _LoadedSource) -> list[dict[str, Any]]:
    payload = source.payload
    if not isinstance(payload, Mapping):
        return []
    raw = payload.get("metrics", payload.get("series"))
    if isinstance(raw, Mapping):
        entries: list[dict[str, Any]] = []
        for metric_id, value in raw.items():
            if isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("metric_id", str(metric_id))
            else:
                entry = {"metric_id": str(metric_id), "samples": value}
            entries.append(entry)
        return entries
    if isinstance(raw, list):
        return [dict(entry) for entry in raw if isinstance(entry, Mapping)]
    # An analysis trace/timeline may explicitly carry a metrics map on each row.
    rows = _rows(payload, "frames", "steps")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        time_s = _time_from(row)
        state = row.get("state") if isinstance(row.get("state"), Mapping) else row
        metrics = state.get("metrics") if isinstance(state, Mapping) else None
        if not isinstance(metrics, Mapping) or time_s is None:
            continue
        for metric_id, metric_value in metrics.items():
            entry = by_id.setdefault(str(metric_id), {"metric_id": str(metric_id), "samples": []})
            entry["samples"].append({"time_s": time_s, "value": metric_value})
    return list(by_id.values())


def _metric_samples(entry: Mapping[str, Any], source: _LoadedSource) -> list[dict[str, Any]]:
    if "time_s" in entry or "t_s" in entry or "source_time_s" in entry or "source_t_s" in entry:
        return [{**dict(entry), "value": _metric_scalar(entry.get("value"))}]
    raw = entry.get("samples", entry.get("values", entry.get("data", [])))
    if isinstance(raw, Mapping):
        # A mapping from timestamp to value is accepted only because each key is
        # an explicit timestamp, never because an index/fps was inferred.
        return [{"time_s": key, "value": _metric_scalar(value)} for key, value in raw.items()]
    if isinstance(raw, list):
        if raw and all(isinstance(row, Mapping) for row in raw):
            return [{**dict(row), "value": _metric_scalar(row.get("value"))} for row in raw]
        payload = source.payload if isinstance(source.payload, Mapping) else {}
        times = payload.get("times_s", payload.get("time_s"))
        if isinstance(times, list) and len(times) == len(raw):
            return [
                {"time_s": time_s, "value": _metric_scalar(value)}
                for time_s, value in zip(times, raw, strict=True)
            ]
    return []


def _metric_scalar(value: Any) -> Any:
    """Unwrap the scalar in a metric value envelope without inventing data.

    Returns:
        The unwrapped scalar or the original value when no scalar envelope exists.
    """

    current = value
    for _ in range(3):
        if not isinstance(current, Mapping) or "value" not in current:
            break
        nested = current["value"]
        if isinstance(nested, Mapping):
            current = nested
            continue
        return nested
    return current


def _metric_streams(
    sources: Sequence[_LoadedSource], config: Mapping[str, Any]
) -> tuple[dict[str, _Stream], dict[str, dict[str, Any]]]:
    streams: dict[str, _Stream] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for source in sources:
        for entry in _metric_entries(source):
            metric_id = _bounded_text(
                entry.get("metric_id", entry.get("id", entry.get("name"))), "unnamed_metric"
            )
            if not metric_id or metric_id in streams:
                continue
            rows = _metric_samples(entry, source)
            stream = _normalize_samples(rows, source=source, config=config, value_only=True)
            stream.name = f"metric:{metric_id}"
            streams[metric_id] = stream
            unit = entry.get("unit", entry.get("units"))
            definitions[metric_id] = {
                "metric_id": metric_id,
                "label": entry.get("label", metric_id),
                "unit": unit if isinstance(unit, str) and unit else None,
                "units_status": "available" if isinstance(unit, str) and unit else "unavailable",
                "source": source.ref.artifact_id,
                "derived": bool(entry.get("derived", False)),
                "description": entry.get("description", ""),
            }
    return streams, definitions


def _derive_metrics_from_scene(  # noqa: C901
    scene: _Stream, goal: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> tuple[dict[str, _Stream], dict[str, dict[str, Any]]]:
    """Derive only transparent geometric metrics from recorded scene state.

    Returns:
        Derived streams and their unit/provenance definitions.
    """

    if config.get("derive_metrics", True) is False or not scene.samples:
        return {}, {}
    goal_point = goal.get("goal_point") if isinstance(goal, Mapping) else None
    if goal_point is None and isinstance(goal, Mapping):
        point_state = goal.get("point")
        if isinstance(point_state, Mapping):
            goal_point = point_state.get("value")
    derived_rows: dict[str, list[dict[str, Any]]] = {}
    for sample in scene.samples:
        value = sample.value if isinstance(sample.value, Mapping) else {}
        robot = value.get("robot") if isinstance(value.get("robot"), Mapping) else {}
        velocity = robot.get("velocity")
        if (
            isinstance(velocity, Sequence)
            and not isinstance(velocity, (str, bytes))
            and len(velocity) >= 2
        ):
            try:
                speed = math.hypot(_finite_number(velocity[0]), _finite_number(velocity[1]))
            except _InputError:
                speed = None
            if speed is not None:
                derived_rows.setdefault("speed", []).append(
                    {"time_s": sample.time_s, "value": speed}
                )
        heading_rate = robot.get("angular_velocity", robot.get("turn_rate_rad_s"))
        if heading_rate is not None:
            try:
                derived_rows.setdefault("angular_velocity", []).append(
                    {"time_s": sample.time_s, "value": _finite_number(heading_rate)}
                )
            except _InputError:
                pass
        if isinstance(goal_point, Sequence) and not isinstance(goal_point, (str, bytes)):
            position = robot.get("position")
            if isinstance(position, Sequence) and len(position) >= 2 and len(goal_point) >= 2:
                try:
                    distance = math.dist(
                        [
                            _finite_number(position[0]),
                            _finite_number(position[1]),
                        ],
                        [
                            _finite_number(goal_point[0]),
                            _finite_number(goal_point[1]),
                        ],
                    )
                except _InputError:
                    distance = None
                if distance is not None:
                    derived_rows.setdefault("goal_distance", []).append(
                        {"time_s": sample.time_s, "value": distance}
                    )
    streams: dict[str, _Stream] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for metric_id, rows in derived_rows.items():
        source = _LoadedSource(
            ref=SourceRef("derived-scene", "", "derived"),
            payload={},
            digest=None,
            integrity="unverifiable",
            availability="ok",
            identity={"artifact_id": "derived-scene", "format": "derived"},
        )
        stream = _normalize_samples(rows, source=source, config=config, value_only=True)
        stream.name = f"metric:{metric_id}"
        streams[metric_id] = stream
        definitions[metric_id] = {
            "metric_id": metric_id,
            "label": metric_id.replace("_", " "),
            "unit": {"speed": "m/s", "angular_velocity": "rad/s", "goal_distance": "m"}.get(
                metric_id
            ),
            "units_status": "available",
            "source": "derived-scene",
            "derived": True,
            "description": "deterministically derived from recorded scene fields",
        }
    return streams, definitions


def _event_rows(source: _LoadedSource) -> list[dict[str, Any]]:  # noqa: C901
    payload = source.payload
    if not isinstance(payload, Mapping):
        return []
    if _payload_schema(source) == FAILURE_DIAGNOSIS_SCHEMA_VERSION:
        rows = payload.get("records", [])
        result: list[dict[str, Any]] = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, Mapping):
                continue
            interval = row.get("onset_interval", row.get("time_interval_s"))
            if isinstance(interval, Sequence) and len(interval) >= 2:
                result.append(
                    {
                        "event_id": row.get("record_id", f"diagnosis-{index}"),
                        "event_type": row.get("failure_type", "diagnosis"),
                        "start_s": interval[0],
                        "end_s": interval[1],
                        "category": row.get("failure_level"),
                        "status": row.get("validity_status", "observed"),
                        "source_record": dict(row),
                    }
                )
        return result
    if _payload_schema(source) in {
        SCENE_SCHEMA_VERSION,
        SIMULATION_TIMELINE_SCHEMA_VERSION,
        ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
        "analysis-trace.v1",
        "simulation_trace_export.v1",
    }:
        result = []
        top_level = payload.get("events")
        if isinstance(top_level, list):
            result.extend(dict(row) for row in top_level if isinstance(row, Mapping))
        for frame in _rows(payload, "frames", "steps"):
            if not isinstance(frame, Mapping):
                continue
            frame_time = _time_from(frame)
            event_rows = frame.get("events")
            planner = frame.get("planner")
            if not isinstance(event_rows, list) and isinstance(planner, Mapping):
                event_rows = [planner] if planner.get("event") or planner.get("event_id") else []
            if isinstance(event_rows, list):
                for row in event_rows:
                    if not isinstance(row, Mapping):
                        continue
                    result.append(
                        {
                            **dict(row),
                            "time_s": row.get("time_s", frame_time),
                            "event_id": row.get(
                                "event_id",
                                frame.get("event_id", f"frame-event-{frame.get('step', 0)}"),
                            ),
                        }
                    )
        return result
    raw = payload.get("intervals", payload.get("events", payload.get("records", [])))
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, Mapping)]
    return []


def _event_intervals(
    sources: Sequence[_LoadedSource],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intervals: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for source in sources:
        for index, row in enumerate(_event_rows(source)[:MAX_EVENTS]):
            start = row.get("start_s", row.get("time_s", row.get("onset_time_s")))
            end = row.get("end_s", start)
            try:
                start_s = _finite_number(start, "event start_s")
                end_s = _finite_number(end, "event end_s")
            except _InputError:
                diagnostics.append(
                    {
                        "artifact_id": source.ref.artifact_id,
                        "reason_code": "event_time_unavailable",
                        "index": index,
                    }
                )
                continue
            if end_s < start_s:
                diagnostics.append(
                    {
                        "artifact_id": source.ref.artifact_id,
                        "reason_code": "event_interval_reversed",
                        "index": index,
                    }
                )
                continue
            event_id = _bounded_text(row.get("event_id", row.get("interval_id", f"event-{index}")))
            intervals.append(
                {
                    "event_id": event_id,
                    "interval_id": row.get("interval_id", event_id),
                    "event_type": row.get(
                        "event_type", row.get("kind", row.get("category", "event"))
                    ),
                    "start_s": start_s,
                    "end_s": end_s,
                    "actor_ids": list(row.get("actor_ids", []))
                    if isinstance(row.get("actor_ids"), list)
                    else [],
                    "category": row.get("category", ""),
                    "metric_value": row.get("metric_value"),
                    "precursor_ids": list(row.get("precursor_ids", []))
                    if isinstance(row.get("precursor_ids"), list)
                    else [],
                    "recovery_ids": list(row.get("recovery_ids", []))
                    if isinstance(row.get("recovery_ids"), list)
                    else [],
                    "source": source.ref.artifact_id,
                }
            )
    intervals.sort(key=lambda row: (row["start_s"], row["end_s"], str(row["event_id"])))
    return intervals, diagnostics


def _goal_geometry(  # noqa: C901, PLR0912
    sources: Sequence[_LoadedSource], scene: _Stream, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return actual goal/completion geometry or explicit unavailable states."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for source in sources:
        if isinstance(source.payload, Mapping):
            candidates.append((source.ref.artifact_id, source.payload))
            metadata = source.payload.get("metadata")
            if isinstance(metadata, Mapping):
                candidates.append((source.ref.artifact_id, metadata))
    for sample in scene.samples[:1]:
        if isinstance(sample.value, Mapping):
            candidates.append(("scene-sample", sample.value))
    point: list[float] | None = None
    point_source = ""
    boundary: Any = None
    boundary_source = ""
    for source_id, payload in candidates:
        for key in ("goal_point", "goal", "robot_goal", "target_point"):
            raw = payload.get(key)
            candidate = raw.get("point") if isinstance(raw, Mapping) else raw
            if (
                isinstance(candidate, Sequence)
                and not isinstance(candidate, (str, bytes))
                and len(candidate) >= 2
            ):
                try:
                    point = [_finite_number(candidate[0]), _finite_number(candidate[1])]
                except _InputError:
                    continue
                point_source = source_id
                break
        for key in (
            "completion_boundary",
            "goal_boundary",
            "goal_radius_m",
            "goal_zone",
            "robot_goal_zones",
        ):
            if key in payload and payload[key] is not None:
                boundary = payload[key]
                boundary_source = source_id
                break
        if point is not None and boundary is not None:
            break
    if point is None and isinstance(config.get("goal_point"), Sequence):
        candidate = config["goal_point"]
        if len(candidate) >= 2:
            try:
                point = [_finite_number(candidate[0]), _finite_number(candidate[1])]
                point_source = "request-config"
            except _InputError:
                pass
    if boundary is None and "completion_boundary" in config:
        boundary = config["completion_boundary"]
        boundary_source = "request-config"
    point_state = {
        "status": "available" if point is not None else "unavailable",
        "value": point,
        "source": point_source or None,
        "reason": "" if point is not None else "goal_point_not_recorded",
    }
    boundary_state = {
        "status": "available" if boundary is not None else "unavailable",
        "value": boundary,
        "source": boundary_source or None,
        "reason": "" if boundary is not None else "completion_boundary_not_recorded",
    }
    return {
        "point": point_state,
        "completion_boundary": boundary_state,
        "goal_point": point,
        "goal_point_status": point_state["status"],
        "completion_boundary_value": boundary,
        "completion_boundary_status": boundary_state["status"],
        "coordinate_frame": next(
            (
                source.identity.get("coordinate_frame")
                for source in sources
                if source.identity.get("coordinate_frame")
            ),
            None,
        ),
    }


def _context_from_sources(
    sources: Sequence[_LoadedSource],
    config: Mapping[str, Any],
    cursor_time_s: float,
    interval_id: str | None,
) -> ReviewContext:
    def _selected(name: str, *aliases: str) -> str | None:
        value = config.get(name)
        if value is not None:
            return str(value)
        keys = (name, *aliases)
        for key in aliases:
            value = config.get(key)
            if value is not None:
                return str(value)
        for source in sources:
            for key in keys:
                value = source.identity.get(key)
                if value is not None:
                    return str(value)
        return str(value) if value is not None else None

    return ReviewContext(
        campaign_id=_selected("campaign_id"),
        execution_id=_selected("execution_id", "run_id", "trace_id"),
        episode_id=_selected("episode_id"),
        interval_id=interval_id,
        actor_id=_selected("actor_id", "actor", "ego_actor_id", "robot_actor_id"),
        cursor=SourceTimeCursor(cursor_time_s, 0, "initial", interval_id),
        context_revision=0,
    )


def _global_times(
    streams: Sequence[_Stream], intervals: Sequence[Mapping[str, Any]]
) -> tuple[float, float]:
    times = [sample.time_s for stream in streams for sample in stream.samples]
    times.extend(float(row[key]) for row in intervals for key in ("start_s", "end_s"))
    if not times:
        return 0.0, 0.0
    return min(times), max(times)


def _interval_config(  # noqa: C901
    config: Mapping[str, Any], start_s: float, end_s: float
) -> tuple[str | None, float, float, list[dict[str, Any]]]:
    """Normalize selected interval and optional interval menu.

    Returns:
        Selected interval ID, start/end, and available interval menu.
    """

    raw_intervals = config.get("intervals", config.get("available_intervals", []))
    intervals: list[dict[str, Any]] = []
    if isinstance(raw_intervals, list):
        for index, raw in enumerate(raw_intervals):
            if not isinstance(raw, Mapping):
                continue
            try:
                left = _finite_number(raw.get("start_s"), "interval start_s")
                right = _finite_number(raw.get("end_s"), "interval end_s")
            except _InputError:
                continue
            if right < left:
                continue
            intervals.append(
                {
                    "interval_id": str(raw.get("interval_id", raw.get("id", f"interval-{index}"))),
                    "start_s": left,
                    "end_s": right,
                    "label": raw.get("label", ""),
                }
            )
    selected = config.get("selected_interval", config.get("interval"))
    selected_id: str | None = None
    if isinstance(selected, str):
        selected_id = selected
        match = next((row for row in intervals if row["interval_id"] == selected), None)
        if match:
            start_s, end_s = match["start_s"], match["end_s"]
    elif isinstance(selected, Mapping):
        selected_id = str(selected.get("interval_id", selected.get("id", ""))) or None
        if selected.get("start_s") is not None and selected.get("end_s") is not None:
            start_s = _finite_number(selected["start_s"], "selected interval start_s")
            end_s = _finite_number(selected["end_s"], "selected interval end_s")
    elif (
        isinstance(selected, Sequence)
        and not isinstance(selected, (str, bytes))
        and len(selected) >= 2
    ):
        start_s = _finite_number(selected[0], "selected interval start_s")
        end_s = _finite_number(selected[1], "selected interval end_s")
    if end_s < start_s:
        raise _InputError("invalid_input: selected interval end precedes start")
    return selected_id, start_s, end_s, intervals


def _stream_document(stream: _Stream) -> dict[str, Any]:
    return {
        "status": stream.status,
        "reason": stream.reason,
        "resolution_s": stream.resolution_s,
        "source_ids": list(stream.source_ids),
        "source_identity": dict(stream.source_identity),
        "media_uri": stream.media_uri,
        "surface": dict(stream.surface),
        "gaps": list(stream.gaps),
        "samples": [
            {
                "time_s": sample.time_s,
                "value": sample.value,
                "source_index": sample.source_index,
                "missing": sample.missing,
            }
            for sample in stream.samples
        ],
        "diagnostics": list(stream.diagnostics),
        "range": (
            {"start_s": stream.samples[0].time_s, "end_s": stream.samples[-1].time_s}
            if stream.samples
            else None
        ),
    }


def _resolution_summary(streams: Mapping[str, _Stream]) -> dict[str, float | None]:
    return {
        name: stream.resolution_s if stream.samples else None for name, stream in streams.items()
    }


def _panel_snapshot(stream: _Stream, cursor_time_s: float, *, panel_name: str) -> dict[str, Any]:
    selected = nearest_sample(stream, cursor_time_s)
    return {
        "panel": panel_name,
        "status": selected.status,
        "reason": selected.reason,
        "cursor_time_s": cursor_time_s,
        "sample_time_s": selected.sample_time_s,
        "temporal_error_s": selected.temporal_error_s,
        "resolution_s": selected.resolution_s,
        "source_index": selected.source_index,
        "value": selected.value if selected.status == "available" else None,
        "media_uri": stream.media_uri,
        "surface": dict(stream.surface),
    }


def _render_html(document: Mapping[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, allow_nan=False).replace("</", "<\\/")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Robot SF synchronized review panels</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#f8fafc}"
        "main{max-width:1400px;margin:auto;padding:1rem}.toolbar{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}"
        "button,select{font:inherit;padding:.35rem .55rem;background:#1f2937;color:inherit;border:1px solid #4b5563;border-radius:.3rem}"
        "input[type=range]{width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.75rem}"
        ".panel{background:#1f2937;border-radius:.4rem;padding:.75rem;min-height:9rem}.panel h2{margin:.1rem 0 .5rem;font-size:1rem}"
        ".muted{color:#cbd5e1;font-size:.85rem}.unavailable{color:#fca5a5}.metric-row{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap}"
        ".review-scene-canvas,.review-video{display:block;width:100%;max-height:24rem;background:#0f172a;border-radius:.3rem}"
        ".metric-trace{display:flex;gap:.2rem;align-items:center;width:100%}.metric-sample{padding:0 .2rem;border:0;background:transparent;color:#38bdf8}"
        "pre{white-space:pre-wrap;overflow:auto;font-size:.76rem;max-height:15rem}"
        "@media(max-width:800px){.grid{grid-template-columns:1fr}}"
        "</style></head><body><main>"
        '<h1>Synchronized review panels</h1><div id="review-panels-root"></div>'
        '<script type="application/json" id="review-panels-data">'
        + payload
        + '</script><script type="module" src="./components/review_panels/review_panels.js"></script>'
        "</main></body></html>"
    )


def _write_text(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise _InputError("resource_limit: output bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise _InputError("output_collision: artifact already exists") from error
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(encoded)


def _write_json(path: Path, payload: Any) -> str:
    return _write_text(path, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _reserve_output(root: Path, value: str) -> Path:
    path = _resolve_under(root, value, kind="output")
    if path.exists() or path.is_symlink():
        raise _InputError("output_collision: output directory already exists")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _copy_web_component(output_dir: Path) -> Path:
    destination = output_dir / "components" / "review_panels" / "review_panels.js"
    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = resources.files("robot_sf.render.web_assets").joinpath(
        "components", "review_panels", "review_panels.js"
    )
    with resources.as_file(asset) as asset_path:
        shutil.copyfile(asset_path, destination)
    return destination


def _panel_document(  # noqa: C901, PLR0912, PLR0915
    request: ComponentRequest,
    sources: Sequence[_LoadedSource],
    config: Mapping[str, Any],
    *,
    load_diagnostics: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Build the complete renderer-neutral model and return diagnostics/status.

    Returns:
        Panel model, diagnostics, and complete/partial/unavailable status.
    """

    diagnostics: list[dict[str, Any]] = [dict(item) for item in load_diagnostics]
    for source in sources:
        diagnostics.extend(source.diagnostics)
    usable = [source for source in sources if source.payload is not None]
    scene_sources = [source for source in usable if _is_scene(source)]
    video_sources = [source for source in usable if _is_video(source)]
    metric_sources = [source for source in usable if _is_metric(source)]
    event_sources = [source for source in usable if _is_event(source)]
    scene = _scene_stream(scene_sources[0], config) if scene_sources else _Stream("scene")
    if scene_sources:
        diagnostics.extend(scene.diagnostics)
    video = _video_stream(video_sources[0], config) if video_sources else _Stream("video")
    if video_sources:
        diagnostics.extend(video.diagnostics)
        # A video document with no explicit mapped source-time row is not a
        # video stream at the simulation clock; retain it as unavailable.
        if not video.samples:
            video.reason = "presentation_timestamp_map_missing"
            video.status = "unavailable"
            diagnostics.append(
                {
                    "artifact_id": video_sources[0].ref.artifact_id,
                    "reason_code": "presentation_timestamp_map_missing",
                    "detail": "video requires explicit source_time_s mapping; FPS alignment is forbidden",
                }
            )
    intervals, event_diagnostics = _event_intervals(event_sources)
    diagnostics.extend(event_diagnostics)
    metric_streams, metric_definitions = _metric_streams(metric_sources, config)
    goal = _goal_geometry(sources, scene, config)
    derived_streams, derived_definitions = _derive_metrics_from_scene(scene, goal, config)
    for metric_id, stream in derived_streams.items():
        metric_streams.setdefault(metric_id, stream)
        metric_definitions.setdefault(metric_id, derived_definitions[metric_id])
    if scene.status == "unavailable" and not scene_sources:
        diagnostics.append(
            {"reason_code": "scene_unavailable", "detail": "no scene source declared"}
        )
    if video_sources and video.status == "unavailable":
        pass
    streams: dict[str, _Stream] = {"scene": scene, "video": video}
    streams.update({f"metric:{name}": stream for name, stream in metric_streams.items()})
    source_start_s, source_end_s = _global_times(list(streams.values()), intervals)
    interval_id, interval_start_s, interval_end_s, configured_intervals = _interval_config(
        config, source_start_s, source_end_s
    )
    cursor_value = config.get("cursor_time_s", config.get("initial_time_s", interval_start_s))
    cursor_time_s = _finite_number(cursor_value, "cursor_time_s")
    cursor_time_s = (
        min(max(cursor_time_s, interval_start_s), interval_end_s)
        if interval_end_s >= interval_start_s
        else cursor_time_s
    )
    context = _context_from_sources(sources, config, cursor_time_s, interval_id)
    source_identity = {
        source.ref.artifact_id: {
            **source.identity,
            "computed_sha256": source.digest,
            "integrity": source.integrity,
            "availability": source.availability,
            "admission": "not_evaluated",
        }
        for source in sources
    }
    visibility = config.get("metric_visibility", config.get("metrics_visible"))
    visibility_declared = visibility is not None
    if isinstance(visibility, Mapping):
        visible_ids = {str(key) for key, value in visibility.items() if value is True}
    elif isinstance(visibility, list):
        visible_ids = {str(value) for value in visibility}
    else:
        visible_ids = set()
    default_ids = [metric_id for metric_id in DEFAULT_METRIC_IDS if metric_id in metric_definitions]
    if not visibility_declared:
        visible_ids = set(default_ids[:4])
    metrics_document: dict[str, Any] = {}
    requested_metric_ids = config.get(
        "requested_metrics", config.get("metric_ids", config.get("metrics"))
    )
    if isinstance(requested_metric_ids, list):
        for requested in requested_metric_ids:
            metric_id = (
                str(requested.get("metric_id", requested.get("id", "")))
                if isinstance(requested, Mapping)
                else str(requested)
            )
            if metric_id and metric_id not in metric_definitions:
                metric_definitions[metric_id] = {
                    "metric_id": metric_id,
                    "label": metric_id,
                    "unit": None,
                    "units_status": "unavailable",
                    "source": None,
                    "derived": False,
                    "description": "metric was requested but not recorded",
                    "unavailable_reason": "metric_not_recorded",
                }
    for metric_id in sorted(metric_definitions):
        stream = metric_streams.get(metric_id, _Stream(f"metric:{metric_id}"))
        definition = dict(metric_definitions[metric_id])
        if not stream.samples and definition.get("unavailable_reason"):
            stream.reason = str(definition["unavailable_reason"])
        definition["visible"] = metric_id in visible_ids
        definition["toggleable"] = True
        definition["missingness"] = {
            "sample_count": len(stream.samples),
            "missing_count": sum(sample.missing for sample in stream.samples),
            "has_gaps": bool(stream.gaps),
        }
        definition["stream"] = _stream_document(stream)
        definition["current"] = _panel_snapshot(
            stream, cursor_time_s, panel_name=f"metric:{metric_id}"
        )
        metrics_document[metric_id] = definition
    events_document = []
    for row in intervals:
        events_document.append(
            {
                **row,
                "selected": row["start_s"] <= cursor_time_s <= row["end_s"],
                "seek_time_s": row["start_s"],
            }
        )
    panel_statuses = {
        "scene": scene.status,
        "video": video.status,
        "metrics": (
            "available"
            if any(
                stream.samples and stream.status in {"available", "partial"}
                for stream in metric_streams.values()
            )
            else "unavailable"
        ),
        "events": "available" if events_document else "unavailable",
    }
    required_panels = config.get("required_panels", [])
    if not isinstance(required_panels, list):
        raise _InputError("invalid_input: required_panels must be a list")
    required_panels = list(required_panels)
    required_panels.extend(
        capability
        for capability in request.required_capabilities
        if capability in panel_statuses and capability not in required_panels
    )
    missing_required = [
        name for name in required_panels if panel_statuses.get(str(name)) != "available"
    ]
    if missing_required:
        diagnostics.extend(
            {"reason_code": "required_panel_unavailable", "panel": name}
            for name in missing_required
        )
    declared_bad = any(source.payload is None for source in sources)
    status = STATUS_PARTIAL if diagnostics or missing_required else STATUS_COMPLETE
    if not usable:
        status = STATUS_UNAVAILABLE
    if missing_required and not any(stream.samples for stream in streams.values()):
        status = STATUS_UNAVAILABLE
    document = {
        "schema_version": PANEL_MODEL_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "status": status,
        "evidence_boundary": "analysis_workbench_only",
        "diagnostic_only": True,
        "time": {
            "authority": "simulation_time",
            "origin_s": source_start_s,
            "terminal_s": source_end_s,
            "interval": {
                "start_s": interval_start_s,
                "end_s": interval_end_s,
                "interval_id": interval_id,
            },
            "cursor": context.cursor.to_dict(),
            "resolution_s": _resolution_summary(streams),
            "rule": "nearest_sample_within_declared_resolution; no_interpolation",
            "video_alignment": "explicit_source_time_mapping_only",
        },
        "context": context.to_dict(),
        "source_identity": source_identity,
        "source_identity_status": {
            source_id: {
                "status": (
                    "available"
                    if any(
                        source.identity.get(key) is not None
                        for key in (
                            "source_commit",
                            "config_identity",
                            "episode_id",
                            "execution_id",
                            "scenario_id",
                            "planner_id",
                            "seed",
                            "actor_id",
                        )
                    )
                    else "unavailable"
                ),
                "reason": ""
                if any(
                    source.identity.get(key) is not None
                    for key in (
                        "source_commit",
                        "config_identity",
                        "episode_id",
                        "execution_id",
                        "scenario_id",
                        "planner_id",
                        "seed",
                        "actor_id",
                    )
                )
                else "source identity fields were not recorded",
            }
            for source_id, source in ((source.ref.artifact_id, source) for source in sources)
        },
        "goal_geometry": goal,
        "goal": goal,
        "scene_surface": dict(scene.surface),
        "surfaces": {
            "scene": {
                "renderer": "offline-canvas",
                "mount": "mountSceneViewer",
                "contract": SCENE_SCHEMA_VERSION,
                "status": scene.status,
            },
            "video": {
                "renderer": "HTMLVideoElement",
                "mount": "review-video",
                "mapping": "explicit_source_time_only",
                "media_uri": video.media_uri,
                "status": video.status,
            },
        },
        "streams": {name: _stream_document(stream) for name, stream in streams.items()},
        "panels": {
            "scene": {
                **_panel_snapshot(scene, cursor_time_s, panel_name="scene"),
                "goal_geometry": goal,
            },
            "video": {
                **_panel_snapshot(video, cursor_time_s, panel_name="video"),
            },
            "metrics": metrics_document,
            "events": events_document,
        },
        "metrics": metrics_document,
        "events": events_document,
        "intervals": configured_intervals,
        "panel_status": panel_statuses,
        "controls": {
            "keyboard": {
                "play_pause": "Space",
                "previous_step": "ArrowLeft",
                "next_step": "ArrowRight",
                "speed_down": "-",
                "speed_up": "+",
            },
            "speeds": [0.25, 0.5, 1.0, 2.0, 4.0],
            "default_speed": float(config.get("speed", 1.0)),
            "read_only": True,
            "feedback_loop_policy": "one_shared_cursor_dispatch",
        },
        "provenance": {
            "source_identity": source_identity,
            "config": dict(config),
            "admission": "not_evaluated",
            "evidence_status": "diagnostic_only",
            "generated_by": COMPONENT_ID,
            "declared_bad_sources": declared_bad,
        },
        "diagnostics": diagnostics,
    }
    return document, diagnostics, status


def descriptor_document() -> dict[str, Any]:
    """Return the versioned component descriptor."""

    return json.loads(json.dumps(_DESCRIPTOR_DOCUMENT, allow_nan=False))


def build_panel_model(request: ComponentRequest, *, base: Path | None = None) -> dict[str, Any]:
    """Build a panel model without reserving or writing an output directory.

    Returns:
        Renderer-neutral ``review-panels.v1`` document.
    """

    if not isinstance(request, ComponentRequest):
        raise TypeError("request must be a ComponentRequest")
    root = (base or Path.cwd()).resolve()
    sources, load_diagnostics = _load_sources(request, root)
    document, diagnostics, _status = _panel_document(
        request, sources, request.config, load_diagnostics=load_diagnostics
    )
    document["diagnostics"] = diagnostics
    return document


def descriptor() -> dict[str, Any]:
    """Alias for callers that use the other SREV component API.

    Returns:
        Versioned component descriptor.
    """

    return descriptor_document()


def result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize and validate a component-result.v1 envelope.

    Returns:
        JSON-safe result document.
    """

    document = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": result.request_id,
        "component_id": result.component_id,
        "status": result.status,
        "artifacts": [dict(artifact) for artifact in result.artifacts],
        "diagnostics": [dict(diagnostic) for diagnostic in result.diagnostics],
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
        "admission": "not_evaluated",
        "evidence_status": "diagnostic_only",
        "evidence_boundary": "analysis_workbench_only",
    }
    if provenance:
        result_provenance.update(dict(provenance))
    return ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=status,
        artifacts=tuple(artifacts),
        diagnostics=tuple(diagnostics),
        provenance=result_provenance,
        reason=reason,
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Build one offline synchronized panel model for a validated request.

    Returns:
        Component result with truthful status and provenance.
    """

    if not isinstance(request, ComponentRequest):
        return _result(
            request if isinstance(request, Mapping) else {},
            STATUS_FAILED,
            reason="invalid_request: expected ComponentRequest",
        )
    if request.component_id != COMPONENT_ID:
        return _result(
            request, STATUS_UNAVAILABLE, reason=f"unsupported_component: {request.component_id}"
        )
    supported = set(REQUIRED_CAPABILITIES) | set(OPTIONAL_CAPABILITIES)
    supported.update(
        {
            "scene-time-sync",
            "actor-identity",
            "artifact-provenance",
            "panel-model",
            "review-panels",
        }
    )
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return _result(
            request,
            STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    if request.config.get("cancelled") is True:
        return _result(request, STATUS_CANCELLED, reason="cancelled: request cancelled before read")
    minimum = request.config.get(
        "min_component_version", request.config.get("required_component_version")
    )
    if minimum is not None:
        try:
            requested_version = tuple(int(part) for part in str(minimum).split("."))
            current_version = tuple(int(part) for part in COMPONENT_VERSION.split("."))
        except (TypeError, ValueError):
            requested_version = (10**9,)
            current_version = (0,)
        if not isinstance(minimum, str) or requested_version > current_version:
            return _result(
                request,
                STATUS_FAILED,
                reason=f"incompatible_component_version: request needs {minimum}, component is {COMPONENT_VERSION}",
            )
    root = (base or Path.cwd()).resolve()
    output_dir: Path | None = None
    try:
        sources, load_diagnostics = _load_sources(request, root)
        output_dir = _reserve_output(root, request.output_directory)
        document, diagnostics, status = _panel_document(
            request, sources, request.config, load_diagnostics=load_diagnostics
        )
        document["diagnostics"] = diagnostics
        emitted: list[dict[str, Any]] = []
        model_name = f"{PANEL_MODEL_SCHEMA_VERSION}.json"
        model_digest = _write_json(output_dir / model_name, document)
        emitted.append(
            {
                "artifact_id": model_name,
                "uri": str(Path(request.output_directory) / model_name),
                "sha256": model_digest,
            }
        )
        html_name = f"{PANEL_MODEL_SCHEMA_VERSION}.html"
        html_digest = _write_text(output_dir / html_name, _render_html(document))
        emitted.append(
            {
                "artifact_id": html_name,
                "uri": str(Path(request.output_directory) / html_name),
                "sha256": html_digest,
            }
        )
        component_path = _copy_web_component(output_dir)
        component_id = "components/review_panels/review_panels.js"
        emitted.append(
            {
                "artifact_id": component_id,
                "uri": str(Path(request.output_directory) / component_id),
                "sha256": _sha256(component_path.read_bytes()),
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
        capability_name = "missing-capability-report.json"
        capability_digest = _write_json(output_dir / capability_name, capability_document)
        emitted.append(
            {
                "artifact_id": capability_name,
                "uri": str(Path(request.output_directory) / capability_name),
                "sha256": capability_digest,
            }
        )
        result_status = status
        # The shared result contract only advertises complete artifacts for a
        # complete run.  Partial files remain on disk as diagnostic output and
        # are named in provenance for local inspection.
        result_artifacts = emitted if result_status == STATUS_COMPLETE else ()
        return _result(
            request,
            result_status,
            artifacts=result_artifacts,
            diagnostics=diagnostics,
            provenance={
                "output_directory": request.output_directory,
                "model_schema": PANEL_MODEL_SCHEMA_VERSION,
                "emitted_artifacts": emitted,
                "source_integrity": {
                    source.ref.artifact_id: source.integrity for source in sources
                },
                "source_identity": document["source_identity"],
                "context_revision": document["context"]["context_revision"],
            },
        )
    except (
        _InputError,
        ReviewContractsValidationError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        if output_dir is not None and not any(output_dir.iterdir()):
            output_dir.rmdir()
        return _result(request, STATUS_FAILED, reason=str(error)[:512])


def _cli_identity(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        return "unknown", COMPONENT_ID
    return (
        str(payload.get("request_id", "unknown")),
        str(payload.get("component_id", COMPONENT_ID)),
    )


def _cli_failure(reason: str, payload: Any = None) -> int:
    request_id, component_id = _cli_identity(payload)
    result = _result(
        {"request_id": request_id, "component_id": component_id},
        STATUS_FAILED,
        reason=reason,
    )
    print(json.dumps(result_document(result), sort_keys=True, indent=2))  # noqa: T201
    return 1


def _strict_cli_json(path: str) -> Any:
    return _strict_loads(_read_regular(Path(path)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build offline synchronized scenario-review panels."
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
    """CLI entry point; non-complete states never return success.

    Returns:
        Process exit code (zero only for a complete result).
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
            else (
                2 if result.status in {STATUS_PARTIAL, STATUS_UNAVAILABLE, STATUS_CANCELLED} else 1
            )
        )
    except (
        ReviewContractsValidationError,
        _InputError,
        OSError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        return _cli_failure(f"invalid_input: {error}", payload)


__all__ = [
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "CONTEXT_SCHEMA_VERSION",
    "DEFAULT_METRIC_IDS",
    "DESCRIPTOR",
    "OUTPUT_HTML_FILENAME",
    "OUTPUT_MODEL_FILENAME",
    "PANEL_MODEL_SCHEMA_VERSION",
    "REQUIRED_CAPABILITIES",
    "NearestSample",
    "ReviewContext",
    "SourceTimeCursor",
    "build_panel_model",
    "descriptor",
    "descriptor_document",
    "main",
    "nearest_sample",
    "result_document",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
