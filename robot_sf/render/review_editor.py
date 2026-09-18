"""Offline review-editor component (SREV-17, issue #9287).

The editor is deliberately a thin client of the SREV-16 panel model and the
BA-03 audit contracts.  It owns review interaction state (annotation speed,
selection, overlays and storyboard edits), but it does not introduce a trace
format or a second canonical store.  Durable writes go through
``PersistenceAdapter`` (also exported as ``AuditPersistenceAdapter``);
``AuditStoreAdapter`` is the current local
implementation and the protocol is the seam for the not-yet-merged BA-05
service.

The Python API is useful in headless/offline workflows and the emitted browser
module has the same intentionally small, dependency-free surface.  Neither
surface starts a simulator, opens a server, fetches media, or uses
``localStorage``.
"""

# The editor deliberately keeps the public interaction API in one offline
# boundary.  A few constructors/factories have intentionally rich keyword
# surfaces and the state transition code is easier to audit as a cohesive
# unit than as speculative micro-abstractions.
# ruff: noqa: C901, D102, D103, D105, D107, DOC201, PLR0912, PLR0913, PLR0915, T201

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path, PureWindowsPath
from typing import Any, Protocol, cast

from robot_sf.analysis_workbench.audit_contracts import (
    ANNOTATION_CLASSIFICATIONS,
    ANNOTATION_MODES,
    AUTHOR_KINDS,
    ActionRecord,
    Annotation,
    Reference,
    ReviewRecord,
    TimeInterval,
    canonical_json,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_store import (
    AuditConflictError,
    AuditStore,
    AuditStoreError,
    CommitResult,
    StoredRecord,
)
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
from robot_sf.render.review_panels import NearestSample, nearest_sample

COMPONENT_ID = "srev17-review-editor"
COMPONENT_VERSION = "1.0.0"
EDITOR_MODEL_SCHEMA_VERSION = "review-editor.v1"
STORYBOARD_SCHEMA_VERSION = "review-storyboard-edit.v1"
MISSING_CAPABILITY_SCHEMA_VERSION = "missing-capability-report.v1"
AUTOSAVE_STATES = ("pending", "saved", "error")
ANNOTATION_SPEEDS = ANNOTATION_MODES
TRIAGE_CLASSIFICATIONS = {
    "normal": "normal",
    "suspicious": "interesting_valid",
    "bug": "planner_defect",
    "unsure": "unclear",
}
TRIAGE_LABELS = tuple(TRIAGE_CLASSIFICATIONS)
OVERLAY_KINDS = ("numbered", "highlights", "arrows", "rings", "distances")
REFERENCE_TARGETS = ("actor", "goal", "waypoint", "map", "event", "metric")
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_DESCRIPTOR_DOCUMENT = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": ["review-editor", "source-bound-annotations"],
    "optional_capabilities": [
        "audit-store",
        "audit-service-adapter",
        "storyboard-editing",
        "spatial-references",
        "offline-browser",
    ],
    "output_types": [
        EDITOR_MODEL_SCHEMA_VERSION,
        STORYBOARD_SCHEMA_VERSION,
        MISSING_CAPABILITY_SCHEMA_VERSION,
    ],
}
component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)
DESCRIPTOR: ComponentDescriptor = component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)
OUTPUT_MODEL_FILENAME = f"{EDITOR_MODEL_SCHEMA_VERSION}.json"
OUTPUT_HTML_FILENAME = f"{EDITOR_MODEL_SCHEMA_VERSION}.html"


class ReviewEditorError(ValueError):
    """Base error for invalid editor input or an unsafe edit."""


class InvalidIntervalError(ReviewEditorError):
    """Raised when a storyboard interval is not valid for its source."""


class StaleSelectionError(ReviewEditorError):
    """Raised when an edit targets an older panel selection revision."""

    def __init__(self, expected: int, actual: int):
        self.expected_revision = expected
        self.actual_revision = actual
        super().__init__(
            f"stale selection revision: expected {expected}, current selection is {actual}"
        )


class SourceStaleError(ReviewEditorError):
    """Raised only when a caller asks to rebind a changed source silently."""


class PersistenceAdapter(Protocol):
    """BA-05-compatible persistence boundary used by the editor.

    BA-05 is not merged at this source revision.  The editor therefore talks
    to this narrow protocol and ships an adapter for the current BA-03
    ``AuditStore``.  A service adapter can implement the same operations
    without changing editor semantics.
    """

    def get(self, record_id: str, *, include_deleted: bool = False) -> StoredRecord | None:
        """Load one record and its current revision."""

    def get_revision(self, record_id: str) -> int:
        """Return zero for a record that has never been committed."""

    def save(
        self,
        record: Any,
        *,
        operation_id: str,
        expected_revision: int | None,
        actor: str | Mapping[str, Any],
        actor_id: str = "",
    ) -> CommitResult:
        """Commit one record using operation ID and compare-and-swap revision."""


# The longer name makes the BA-03/BA-05 boundary obvious to integrations that
# discover this module by API rather than by reading the component descriptor.
# It is an alias, not a second persistence contract.
AuditPersistenceAdapter = PersistenceAdapter


class AuditStoreAdapter:
    """Adapter over the current BA-03 ``AuditStore``.

    This class intentionally delegates all journal, projection, conflict,
    replay, crash-recovery and backup behaviour to ``AuditStore``.  It never
    appends to ``audit.ndjson`` itself and never uses browser storage.
    """

    adapter_id = "ba03-audit-store"
    service_boundary = "ba05-service-compatible"

    def __init__(self, root: str | Path, **store_options: Any):
        self.root = Path(root)
        self.store = AuditStore(self.root, **store_options)

    def __enter__(self) -> AuditStoreAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()

    def get(self, record_id: str, *, include_deleted: bool = False) -> StoredRecord | None:
        return self.store.get(record_id, include_deleted=include_deleted)

    def get_revision(self, record_id: str) -> int:
        return self.store.get_revision(record_id)

    def save(
        self,
        record: Any,
        *,
        operation_id: str,
        expected_revision: int | None,
        actor: str | Mapping[str, Any],
        actor_id: str = "",
    ) -> CommitResult:
        return self.store.save(
            record,
            operation_id=operation_id,
            expected_revision=expected_revision,
            actor=actor,
            actor_id=actor_id,
        )


class ServicePersistenceAdapter:
    """Adapt a future BA-05 service without making it a competing store.

    ``service`` may expose ``save``/``get`` with the same keyword contract as
    ``AuditStore``, or a ``commit`` method plus ``get``.  The adapter does not
    interpret or merge records; it only forwards operation and revision
    metadata and normalizes the returned receipt.
    """

    adapter_id = "ba05-service"
    service_boundary = "ba05-service-compatible"

    def __init__(self, service: Any):
        self.service = service

    def get(self, record_id: str, *, include_deleted: bool = False) -> StoredRecord | None:
        getter = getattr(self.service, "get", None) or getattr(self.service, "load", None)
        if getter is None:
            raise ReviewEditorError("service adapter requires get/load")
        return getter(record_id, include_deleted=include_deleted)

    def get_revision(self, record_id: str) -> int:
        getter = getattr(self.service, "get_revision", None)
        if getter is not None:
            return int(getter(record_id))
        stored = self.get(record_id, include_deleted=True)
        return int(stored.revision) if stored is not None else 0

    def save(
        self,
        record: Any,
        *,
        operation_id: str,
        expected_revision: int | None,
        actor: str | Mapping[str, Any],
        actor_id: str = "",
    ) -> CommitResult:
        kwargs = {
            "operation_id": operation_id,
            "expected_revision": expected_revision,
            "actor": actor,
            "actor_id": actor_id,
        }
        saver = getattr(self.service, "save", None)
        if saver is None:
            saver = getattr(self.service, "commit", None)
            if saver is None:
                raise ReviewEditorError("service adapter requires save/commit")
            result = saver([record], **kwargs)
        else:
            result = saver(record, **kwargs)
        if isinstance(result, CommitResult):
            return result
        if hasattr(result, "revision") and hasattr(result, "record_id"):
            return cast("CommitResult", result)
        raise ReviewEditorError("service save did not return a commit receipt")


@dataclass(frozen=True, slots=True)
class AutosaveStatus:
    """Durability state exposed by the Python and browser editor surfaces."""

    state: str = "saved"
    operation_id: str = ""
    record_id: str = ""
    expected_revision: int | None = None
    saved_revision: int | None = None
    selection_revision: int = 0
    error: str = ""
    conflict: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state not in AUTOSAVE_STATES:
            raise ReviewEditorError(f"autosave state must be one of {AUTOSAVE_STATES}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OverlayState:
    """Toggleable overlay flags; geometry remains source-bound."""

    numbered: bool = True
    highlights: bool = True
    arrows: bool = False
    rings: bool = False
    distances: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {kind: bool(getattr(self, kind)) for kind in OVERLAY_KINDS}


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewEditorError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReviewEditorError(f"{name} must be a finite number")
    return result


def _point(value: Any, *, name: str = "point") -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ReviewEditorError(f"{name} must contain exactly two coordinates")
    return (_finite(value[0], name=f"{name}[0]"), _finite(value[1], name=f"{name}[1]"))


def _text(value: Any, *, name: str, allow_empty: bool = False, limit: int = 10_000) -> str:
    if not isinstance(value, str) or len(value) > limit or (not allow_empty and not value.strip()):
        raise ReviewEditorError(f"{name} must be {'optional ' if allow_empty else ''}text")
    return value


def _strict_loads(raw: bytes) -> Any:
    def reject_constant(token: str) -> Any:
        raise ReviewEditorError(f"non-finite JSON constant: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReviewEditorError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    if len(raw) > MAX_SOURCE_BYTES:
        raise ReviewEditorError("source exceeds review-editor input limit")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ReviewEditorError("source is not strict UTF-8 JSON") from error


def _unsafe_path(value: str) -> bool:
    path = Path(value)
    windows = PureWindowsPath(value)
    return (
        not value
        or path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in path.parts
        or ".." in windows.parts
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


def _resolve_under(root: Path, value: str, *, kind: str) -> Path:
    if _unsafe_path(value):
        raise ReviewEditorError(f"unsafe_{kind}_path: {value!r}")
    root = root.resolve(strict=True)
    # Retain the lexical path so callers can reject a source symlink before
    # reading it.  Use a separate resolved path only for the containment check;
    # returning the resolved target would make ``Path.is_symlink()`` useless.
    candidate = root / value
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(root)
    except ValueError as error:
        raise ReviewEditorError(f"unsafe_{kind}_path: path escapes base") from error
    return candidate


def _open_parent_no_follow(path: Path) -> tuple[int, str]:
    """Open a path's parent directory without traversing symlinks.

    The returned directory descriptor remains stable if an attacker swaps a
    parent path after validation.  Callers must close it after operating on
    the returned basename with ``dir_fd``.
    """

    absolute = path if path.is_absolute() else Path.cwd() / path
    parts = absolute.parts
    if not parts:
        raise ReviewEditorError("unsafe_path: empty path")
    if absolute.anchor:
        descriptor = os.open(
            absolute.anchor,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        components = parts[1:]
    else:  # pragma: no cover - ``absolute`` is normally anchored above.
        descriptor = os.open(
            ".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        components = parts
    if not components:
        os.close(descriptor)
        raise ReviewEditorError("unsafe_path: path has no basename")
    try:
        for component in components[:-1]:
            if component in {"", "."}:
                continue
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, components[-1]


def _read_regular_path(path: Path, *, kind: str) -> bytes:
    """Read one bounded regular file through a no-follow descriptor."""

    parent_fd, basename = _open_parent_no_follow(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ReviewEditorError(f"{kind}_not_regular_file")
        if opened.st_size > MAX_SOURCE_BYTES:
            raise ReviewEditorError(f"{kind}_too_large")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(MAX_SOURCE_BYTES + 1)
        if len(payload) > MAX_SOURCE_BYTES:
            raise ReviewEditorError(f"{kind}_too_large")
        return payload
    except OSError as error:
        raise ReviewEditorError(f"{kind}_unreadable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _mkdir_no_follow(path: Path, *, exist_ok: bool = True) -> None:
    """Create a directory tree without following swapped-in symlink parents."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    parts = absolute.parts
    descriptor = os.open(
        absolute.anchor or ".",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        components = parts[1:] if absolute.anchor else parts
        for index, component in enumerate(components):
            if component in {"", "."}:
                continue
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                if index == len(components) - 1 and not exist_ok:
                    os.close(next_descriptor)
                    raise FileExistsError(component, "mkdir", str(path))
            except FileNotFoundError:
                os.mkdir(component, dir_fd=descriptor)
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
    except FileExistsError:
        if not exist_ok:
            raise ReviewEditorError(f"output_collision: {path}")
        raise
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    """Write bytes atomically, rejecting symlink destinations and races."""

    if len(payload) > MAX_OUTPUT_BYTES:
        raise ReviewEditorError("editor output exceeds size limit")
    parent_fd, basename = _open_parent_no_follow(path)
    temporary = f".{basename}.tmp-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        try:
            existing = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not overwrite:
            raise ReviewEditorError(f"output_collision: {path}")
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ReviewEditorError(f"output_symlink_or_nonregular: {path}")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, basename, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        else:
            os.link(
                temporary,
                basename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=parent_fd)
    except OSError as error:
        raise ReviewEditorError(f"cannot write output: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def _read_source(root: Path, ref: SourceRef) -> tuple[Any, str, bool]:
    path = _resolve_under(root, ref.uri, kind="source")
    try:
        raw = _read_regular_path(path, kind=f"source:{ref.artifact_id}")
    except ReviewEditorError as error:
        raise ReviewEditorError(f"source_unreadable: {ref.artifact_id}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if ref.sha256 and digest.lower() != ref.sha256.lower():
        # The bytes remain available for diagnostics, but any attachment to
        # them is stale and must not be silently rebound.
        try:
            payload = _strict_loads(raw)
        except ReviewEditorError:
            payload = None
        return payload, digest, False
    return _strict_loads(raw), digest, True


def _source_identity(
    model: Mapping[str, Any],
    source_refs: Mapping[str, SourceRef],
    source_digests: Mapping[str, str],
) -> dict[str, Any]:
    identity = model.get("source_identity")
    result: dict[str, Any] = dict(identity) if isinstance(identity, Mapping) else {}
    context = model.get("context")
    if isinstance(context, Mapping):
        for key in (
            "episode_id",
            "execution_id",
            "campaign_id",
            "scenario_id",
            "planner_id",
            "seed",
        ):
            if key not in result and context.get(key) is not None:
                result[key] = context[key]
    result.setdefault("sources", {})
    source_entries = result["sources"] if isinstance(result["sources"], Mapping) else {}
    result["sources"] = dict(source_entries)
    for artifact_id, ref in source_refs.items():
        result["sources"].setdefault(artifact_id, {})
        if not isinstance(result["sources"][artifact_id], Mapping):
            result["sources"][artifact_id] = {}
        result["sources"][artifact_id] = {
            **dict(result["sources"][artifact_id]),
            "artifact_id": artifact_id,
            "uri": ref.uri,
            "format": ref.format,
            "sha256": source_digests.get(artifact_id) or ref.sha256,
            "declared_sha256": ref.sha256,
            "source_commit": ref.source_commit,
            "schema": ref.schema,
            "config_identity": ref.config_identity,
            "units": ref.units,
            "coordinate_frame": ref.coordinate_frame,
        }
    return result


def _source_revision_payload(
    source_refs: Mapping[str, SourceRef], source_digests: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    """Return every source field that can change a persisted storyboard binding."""

    return {
        artifact_id: {
            "sha256": source_digests.get(artifact_id, ref.sha256),
            "source_commit": ref.source_commit,
            "schema": ref.schema,
            "config_identity": ref.config_identity,
        }
        for artifact_id, ref in sorted(source_refs.items())
    }


def _source_revision_token(
    model: Mapping[str, Any],
    source_refs: Mapping[str, SourceRef],
    source_digests: Mapping[str, str],
) -> str:
    """Return the cross-runtime canonical storyboard source revision token.

    The browser editor computes this same compact, sorted JSON token.  Keep the
    source revision readable rather than hashing it so a generated Python
    model can be saved by the browser without an adapter-specific translation.
    ``source_identity`` remains the SHA-256 identity token; this token is the
    revision of the identity inputs.
    """

    context = model.get("context")
    context_revision = context.get("source_revision", "") if isinstance(context, Mapping) else ""
    return canonical_json(
        {
            "revisions": _source_revision_payload(source_refs, source_digests),
            "context": context_revision or "",
        }
    )


def _context_value(model: Mapping[str, Any], name: str, default: Any = None) -> Any:
    context = model.get("context")
    if isinstance(context, Mapping) and name in context:
        return context[name]
    identity = model.get("source_identity")
    if isinstance(identity, Mapping):
        return identity.get(name, default)
    return default


def _current_time(model: Mapping[str, Any], fallback: float = 0.0) -> float:
    context = model.get("context")
    if isinstance(context, Mapping):
        cursor = context.get("cursor")
        if isinstance(cursor, Mapping) and cursor.get("time_s") is not None:
            return _finite(cursor["time_s"], name="cursor.time_s")
    time_block = model.get("time")
    if isinstance(time_block, Mapping) and time_block.get("cursor") is not None:
        cursor = time_block["cursor"]
        if isinstance(cursor, Mapping) and cursor.get("time_s") is not None:
            return _finite(cursor["time_s"], name="cursor.time_s")
        return _finite(cursor, name="cursor time")
    return _finite(fallback, name="fallback time")


def _time_bounds(model: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Return the recorded source-time origin and terminal, if declared."""

    time_block = model.get("time")
    if not isinstance(time_block, Mapping):
        return None, None
    origin = time_block.get("origin_s", time_block.get("start_s"))
    terminal = time_block.get("terminal_s", time_block.get("end_s"))
    return (
        None if origin is None else _finite(origin, name="time.origin_s"),
        None if terminal is None else _finite(terminal, name="time.terminal_s"),
    )


def _validate_model_interval(model: Mapping[str, Any], interval: TimeInterval) -> TimeInterval:
    origin, terminal = _time_bounds(model)
    if origin is not None and interval.start_s < origin:
        raise InvalidIntervalError("interval must be contained in source time range")
    if terminal is not None and interval.end_s > terminal:
        raise InvalidIntervalError("interval must be contained in source time range")
    return interval


def _selected_interval(model: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the currently selected recorded storyboard interval, if any."""

    context = model.get("context")
    interval_id = context.get("interval_id") if isinstance(context, Mapping) else None
    if not interval_id:
        return None
    storyboard = model.get("storyboard")
    intervals = storyboard.get("intervals", []) if isinstance(storyboard, Mapping) else []
    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)):
        return None
    return next(
        (
            item
            for item in intervals
            if isinstance(item, Mapping)
            and str(item.get("interval_id", item.get("id", ""))) == str(interval_id)
        ),
        None,
    )


def _interval_from(value: Any, *, default_time: float | None = None) -> TimeInterval | None:
    if value is None:
        return None if default_time is None else TimeInterval(default_time)
    if isinstance(value, TimeInterval):
        return value
    if isinstance(value, Mapping):
        if "start_s" not in value and "start" not in value:
            raise InvalidIntervalError("interval.start_s is required")
        start = value.get("start_s", value.get("start"))
        end = value.get("end_s", value.get("end", start))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in (1, 2):
            raise InvalidIntervalError("interval must contain one or two times")
        start = value[0]
        end = value[-1]
    else:
        start = end = value
    start_value = _finite(start, name="interval.start_s")
    end_value = _finite(end, name="interval.end_s")
    if end_value < start_value:
        raise InvalidIntervalError("interval.end_s must be >= interval.start_s")
    return TimeInterval(start_value, end_value)


def validate_interval(
    value: Any, *, duration_s: float | None = None, origin_s: float = 0.0
) -> TimeInterval:
    """Validate a non-reversed interval and optional absolute source bounds."""

    interval = _interval_from(value)
    if interval is None:  # pragma: no cover - _interval_from always returns here.
        raise InvalidIntervalError("interval is required")
    if duration_s is not None:
        duration = _finite(duration_s, name="duration_s")
        origin = _finite(origin_s, name="origin_s")
        if interval.start_s < origin or interval.end_s > duration:
            raise InvalidIntervalError("interval must be contained in source duration")
    return interval


def _classification(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = TRIAGE_CLASSIFICATIONS.get(normalized, normalized)
    if normalized not in ANNOTATION_CLASSIFICATIONS:
        raise ReviewEditorError(
            f"classification must be one of {ANNOTATION_CLASSIFICATIONS} or {TRIAGE_LABELS}"
        )
    return normalized


def _source_ref_for(
    source_refs: Mapping[str, SourceRef], source_id: str | None = None
) -> SourceRef | None:
    if source_id and source_id in source_refs:
        return source_refs[source_id]
    for ref in source_refs.values():
        if ref.format.lower().startswith(("scene", "threejs", "simulation", "video", "image")):
            return ref
    return next(iter(source_refs.values()), None)


def _annotation_source_fields(
    model: Mapping[str, Any],
    source_refs: Mapping[str, SourceRef],
    source_digests: Mapping[str, str],
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    if not source_refs:
        model_refs, model_digests = _source_bindings_from_model(model)
        source_refs = model_refs
        source_digests = {**model_digests, **source_digests}
    ref = _source_ref_for(source_refs, source_id)
    digest = source_digests.get(ref.artifact_id, "") if ref is not None else ""
    source_identity = digest if digest else ""
    source_revision = ref.source_commit if ref is not None else ""
    if ref is None or not digest or not ref.sha256 or not source_identity:
        provenance = "unavailable"
    elif ref.sha256 and digest.lower() != ref.sha256.lower():
        # Preserve the changed byte identity while making the attachment
        # explicitly stale.  The editor never silently rebinds old points.
        provenance = "stale"
    elif not isinstance(source_revision, str) or not source_revision.strip():
        provenance = "unavailable"
    else:
        provenance = "verified"
    # BA-03 upgrades an ``unavailable`` annotation to verified when its source
    # hash is present but no source reference is attached.  A source with no
    # revision is intentionally not a verified binding, so omit that partial
    # reference rather than letting the downstream constructor guess.
    annotation_ref = (
        None
        if provenance == "unavailable"
        and ref is not None
        and digest
        and ref.sha256
        and digest.lower() == ref.sha256.lower()
        and not ref.source_commit
        else ref
    )
    return {
        "source_ref": annotation_ref,
        "source_identity": source_identity,
        "source_revision": source_revision,
        "provenance_status": provenance,
    }


def _episode_id(model: Mapping[str, Any]) -> str:
    value = _context_value(model, "episode_id", "episode")
    return _text(value, name="episode_id", limit=512)


def _selection_revision(model: Mapping[str, Any]) -> int:
    context = model.get("context")
    if isinstance(context, Mapping):
        value = context.get("context_revision", context.get("selection_revision", 0))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _source_bindings_from_model(
    model: Mapping[str, Any],
) -> tuple[dict[str, SourceRef], dict[str, str]]:
    """Recover typed source bindings embedded in a panel/editor model.

    ``ReviewEditorSession`` is often constructed from the persisted editor
    model rather than alongside the original component request.  Keeping this
    small conversion here means those sessions still create source-bound
    annotations without making the model itself a second source registry.
    """

    identity = model.get("source_identity")
    raw_sources = identity.get("sources") if isinstance(identity, Mapping) else None
    if not isinstance(raw_sources, Mapping):
        return {}, {}
    refs: dict[str, SourceRef] = {}
    digests: dict[str, str] = {}
    source_fields = {
        "artifact_id",
        "uri",
        "format",
        "schema",
        "sha256",
        "source_commit",
        "config_identity",
        "units",
        "coordinate_frame",
    }
    for artifact_id, raw in raw_sources.items():
        if not isinstance(raw, Mapping):
            continue
        values = {key: raw.get(key, "") for key in source_fields}
        values["sha256"] = raw.get("declared_sha256", values.get("sha256", ""))
        values["artifact_id"] = str(values.get("artifact_id") or artifact_id)
        if not values.get("uri") or not values.get("format"):
            continue
        try:
            ref = SourceRef(**{key: str(value) for key, value in values.items()})
        except (TypeError, ValueError):
            continue
        refs[ref.artifact_id] = ref
        digest = raw.get("sha256")
        if isinstance(digest, str) and digest:
            digests[ref.artifact_id] = digest
    return refs, digests


def _metadata_with(
    metadata: Mapping[str, Any] | None = None,
    *,
    actors: Sequence[str] = (),
    notes: str = "",
    execution_id: str = "",
    selection_revision: int = 0,
) -> dict[str, Any]:
    result = dict(metadata or {})
    if actors:
        result["actors"] = list(dict.fromkeys(str(item) for item in actors))
    if notes:
        result["notes"] = notes
    if execution_id:
        result["execution_id"] = execution_id
    result["selection_revision"] = selection_revision
    return result


def make_one_click_annotation(
    model: Mapping[str, Any],
    label: str,
    *,
    annotation_id: str | None = None,
    time_s: float | None = None,
    interval: Any = None,
    tags: Sequence[str] = (),
    author_kind: str = "human",
    author_id: str = "",
    source_refs: Mapping[str, SourceRef] | None = None,
    source_digests: Mapping[str, str] | None = None,
    source_id: str | None = None,
) -> Annotation:
    """Create a Normal/Suspicious/Bug/Unsure triage annotation."""

    if author_kind not in AUTHOR_KINDS:
        raise ReviewEditorError(f"author_kind must be one of {AUTHOR_KINDS}")
    refs = source_refs or {}
    fields = _annotation_source_fields(model, refs, source_digests or {}, source_id=source_id)
    selected_interval = _interval_from(
        interval if interval is not None or time_s is not None else _selected_interval(model),
        default_time=_current_time(model) if time_s is None else _finite(time_s, name="time_s"),
    )
    if selected_interval is not None:
        selected_interval = _validate_model_interval(model, selected_interval)
    return Annotation(
        annotation_id=annotation_id or f"annotation-{uuid.uuid4().hex}",
        episode_id=_episode_id(model),
        classification=_classification(label),
        mode="one_click",
        author_kind=author_kind,
        author_id=author_id,
        interval=selected_interval,
        tags=tuple(str(tag) for tag in tags),
        review_scope="interval",
        metadata=_metadata_with(
            actors=(),
            execution_id=str(_context_value(model, "execution_id", "")),
            selection_revision=_selection_revision(model),
        ),
        **fields,
    )


def make_quick_annotation(
    model: Mapping[str, Any],
    classification: str,
    *,
    annotation_id: str | None = None,
    tags: Sequence[str] = (),
    time_s: float | None = None,
    interval: Any = None,
    observed_behavior: str = "",
    author_kind: str = "human",
    author_id: str = "",
    source_refs: Mapping[str, SourceRef] | None = None,
    source_digests: Mapping[str, str] | None = None,
    source_id: str | None = None,
) -> Annotation:
    """Create a quick classification/tag note without causal fields."""

    if author_kind not in AUTHOR_KINDS:
        raise ReviewEditorError(f"author_kind must be one of {AUTHOR_KINDS}")
    refs = source_refs or {}
    fields = _annotation_source_fields(model, refs, source_digests or {}, source_id=source_id)
    selected_interval = _interval_from(
        interval if interval is not None or time_s is not None else _selected_interval(model),
        default_time=_current_time(model) if time_s is None else _finite(time_s, name="time_s"),
    )
    if selected_interval is not None:
        selected_interval = _validate_model_interval(model, selected_interval)
    return Annotation(
        annotation_id=annotation_id or f"annotation-{uuid.uuid4().hex}",
        episode_id=_episode_id(model),
        classification=_classification(classification),
        mode="quick",
        author_kind=author_kind,
        author_id=author_id,
        interval=selected_interval,
        observed_behavior=_text(observed_behavior, name="observed_behavior", allow_empty=True),
        tags=tuple(str(tag) for tag in tags),
        review_scope="interval",
        metadata=_metadata_with(
            execution_id=str(_context_value(model, "execution_id", "")),
            selection_revision=_selection_revision(model),
        ),
        **fields,
    )


def make_full_annotation(
    model: Mapping[str, Any],
    classification: str,
    *,
    annotation_id: str | None = None,
    observed_behavior: str = "",
    interval: Any = None,
    actors: Sequence[str] = (),
    references: Sequence[Reference | Mapping[str, Any]] = (),
    measured_evidence: Sequence[Mapping[str, Any]] = (),
    hypothesis: str = "",
    confidence: float | None = None,
    notes: str = "",
    tags: Sequence[str] = (),
    full_episode: bool = False,
    author_kind: str = "human",
    author_id: str = "",
    source_refs: Mapping[str, SourceRef] | None = None,
    source_digests: Mapping[str, str] | None = None,
    source_id: str | None = None,
) -> Annotation:
    """Create a structured observation with separated evidence and hypothesis."""

    if author_kind not in AUTHOR_KINDS:
        raise ReviewEditorError(f"author_kind must be one of {AUTHOR_KINDS}")
    refs = source_refs or {}
    fields = _annotation_source_fields(model, refs, source_digests or {}, source_id=source_id)
    normalized_refs = tuple(
        item if isinstance(item, Reference) else Reference(**dict(item)) for item in references
    )
    selected_interval = _interval_from(
        interval if interval is not None else _selected_interval(model),
        default_time=_current_time(model),
    )
    if selected_interval is not None:
        selected_interval = _validate_model_interval(model, selected_interval)
    normalized_evidence_items: list[dict[str, Any]] = []
    for item in measured_evidence:
        if not isinstance(item, Mapping):
            raise ReviewEditorError("measured_evidence entries must be mappings")
        normalized_evidence_items.append(dict(item))
    normalized_evidence = tuple(normalized_evidence_items)
    return Annotation(
        annotation_id=annotation_id or f"annotation-{uuid.uuid4().hex}",
        episode_id=_episode_id(model),
        classification=_classification(classification),
        mode="full",
        author_kind=author_kind,
        author_id=author_id,
        interval=selected_interval,
        observed_behavior=_text(observed_behavior, name="observed_behavior", allow_empty=True),
        suspected_cause=_text(hypothesis, name="hypothesis", allow_empty=True),
        evidence=normalized_evidence,
        confidence=confidence,
        references=normalized_refs,
        tags=tuple(str(tag) for tag in tags),
        review_scope="full_episode" if full_episode else "interval",
        metadata=_metadata_with(
            actors=actors,
            notes=_text(notes, name="notes", allow_empty=True),
            execution_id=str(_context_value(model, "execution_id", "")),
            selection_revision=_selection_revision(model),
        ),
        **fields,
    )


def make_full_episode_review(
    model: Mapping[str, Any],
    *,
    review_id: str | None = None,
    outcome: str = "",
    author_kind: str = "human",
    author_id: str = "",
    annotation_ids: Sequence[str] = (),
    notes: str = "",
    source_revision: int | None = None,
) -> ReviewRecord:
    """Create the explicit coverage receipt for a full-episode review."""

    revision = _selection_revision(model) if source_revision is None else source_revision
    return ReviewRecord(
        review_id=review_id or f"review-{uuid.uuid4().hex}",
        episode_id=_episode_id(model),
        scope="full_episode",
        outcome=outcome,
        author_kind=author_kind,
        author_id=author_id,
        source_revision=revision,
        annotation_ids=tuple(str(item) for item in annotation_ids),
        notes=notes,
    )


def coverage_summary(records: Sequence[Any]) -> dict[str, Any]:
    """Count explicit human full-episode receipts only.

    Triage, interval annotations, agent records, and detector records remain
    visible in the totals but never contribute to human full-episode coverage.
    """

    reviews = [item for item in records if isinstance(item, ReviewRecord)]
    annotations = [item for item in records if isinstance(item, Annotation)]
    explicit_human = {
        item.episode_id
        for item in reviews
        if item.scope == "full_episode" and item.author_kind == "human"
    }
    explicit_human.update(
        item.episode_id
        for item in annotations
        if item.review_scope == "full_episode" and item.author_kind == "human"
    )
    return {
        "human_full_episode_reviewed_episode_ids": sorted(explicit_human),
        "human_full_episode_count": len(explicit_human),
        "annotation_count": len(annotations),
        "review_receipt_count": len(reviews),
        "triage_count": sum(item.mode == "one_click" for item in annotations),
        "agent_annotation_count": sum(item.author_kind == "agent" for item in annotations),
        "interval_annotation_count": sum(item.review_scope == "interval" for item in annotations),
    }


def _normalize_stream(stream: Mapping[str, Any]) -> dict[str, Any]:
    """Apply SREV-16 missing-row aliases before nearest-sample selection."""

    normalized = dict(stream)
    rows = []
    for row in stream.get("samples", []):
        if not isinstance(row, Mapping):
            rows.append(row)
            continue
        copied = dict(row)
        copied["missing"] = bool(
            row.get("missing")
            or row.get("available") is False
            or ("value" in row and row.get("value") is None)
            or row.get("missing_reason")
        )
        rows.append(copied)
    normalized["samples"] = rows
    return normalized


def _nearest_stream_sample(stream: Mapping[str, Any], time_s: float) -> NearestSample:
    return nearest_sample(_normalize_stream(stream), time_s)


def _stream_sample(model: Mapping[str, Any], stream_name: str, time_s: float) -> NearestSample:
    """Use the SREV-16 nearest-sample contract for every snap consumer."""

    streams = model.get("streams")
    stream = streams.get(stream_name) if isinstance(streams, Mapping) else None
    if not isinstance(stream, Mapping):
        stream = {"status": "unavailable", "reason": f"{stream_name}_stream_missing"}
    return _nearest_stream_sample(stream, time_s)


def _sample_at(model: Mapping[str, Any], time_s: float) -> Mapping[str, Any] | None:
    """Return an available scene sample under the SREV-16 alignment rules."""

    selected = _stream_sample(model, "scene", time_s)
    if selected.status != "available" or not isinstance(selected.value, Mapping):
        return None
    return selected.value


def _actor_point(sample: Mapping[str, Any] | None, actor_id: str) -> tuple[float, float] | None:
    value = sample.get("value", sample) if isinstance(sample, Mapping) else {}
    if not isinstance(value, Mapping):
        return None
    robot = value.get("robot")
    if isinstance(robot, Mapping):
        identity = robot.get("actor_id", robot.get("id", "robot"))
        if str(identity) == actor_id:
            return _maybe_point(robot.get("position", robot.get("point")))
    for actor in (
        value.get("pedestrians", []) if isinstance(value.get("pedestrians"), Sequence) else ()
    ):
        if (
            isinstance(actor, Mapping)
            and str(actor.get("actor_id", actor.get("id", ""))) == actor_id
        ):
            return _maybe_point(actor.get("position", actor.get("point")))
    return None


def _maybe_point(value: Any) -> tuple[float, float] | None:
    try:
        return _point(value)
    except ReviewEditorError:
        return None


def _goal_point(model: Mapping[str, Any]) -> tuple[float, float] | None:
    goal = model.get("goal_geometry")
    if isinstance(goal, Mapping):
        for candidate in (
            goal.get("goal_point"),
            goal.get("point"),
            goal.get("active_goal"),
            goal.get("final_goal"),
        ):
            if isinstance(candidate, Mapping):
                candidate = candidate.get("point", candidate.get("value"))
            point = _maybe_point(candidate)
            if point is not None:
                return point
    return None


def _units_for(model: Mapping[str, Any], source: SourceRef | None) -> str:
    if source is not None and source.units:
        return source.units
    identity = model.get("source_identity")
    if isinstance(identity, Mapping):
        value = identity.get("units")
        if isinstance(value, str) and value:
            return value
    return ""


def _validate_reference_timestamp(model: Mapping[str, Any], timestamp_s: float) -> float:
    """Require a reference timestamp to lie within declared source bounds."""

    timestamp = _finite(timestamp_s, name="timestamp_s")
    origin, terminal = _time_bounds(model)
    if origin is not None and timestamp < origin:
        raise InvalidIntervalError("reference timestamp is before source origin")
    if terminal is not None and timestamp > terminal:
        raise InvalidIntervalError("reference timestamp is after source terminal")
    return timestamp


def create_reference(
    model: Mapping[str, Any],
    point: Sequence[Any],
    *,
    reference_id: str | None = None,
    coordinate_frame: str = "world",
    timestamp_s: float | None = None,
    source: SourceRef | None = None,
    source_revision: str = "",
    calibration: Mapping[str, Any] | None = None,
    source_point: Sequence[Any] | None = None,
    actor_id: str = "",
    object_id: str = "",
    goal_id: str = "",
    waypoint_id: str = "",
    metric_id: str = "",
    event_id: str = "",
) -> Reference:
    """Create a source-bound world/image reference.

    Image references retain the original source point and calibration/crop
    identity.  World references are accepted only when the scene declares a
    world frame or the caller supplies an explicit calibration for media.
    """

    if coordinate_frame not in {"world", "image"}:
        raise ReviewEditorError("coordinate_frame must be 'world' or 'image'")
    point_value = _point(point)
    source_value = source
    if source_value is None:
        model_refs, _model_digests = _source_bindings_from_model(model)
        source_value = next(iter(model_refs.values()), None)
    identity_frame = model.get("source_identity")
    declared_frame = (
        identity_frame.get("coordinate_frame") if isinstance(identity_frame, Mapping) else ""
    )
    if coordinate_frame == "world" and source_value is not None:
        media = source_value.format.lower().startswith(("video", "image"))
        if media and not calibration and declared_frame != "world":
            raise ReviewEditorError(
                "world reference requires verified scene geometry or calibration"
            )
        if media and calibration and declared_frame != "world":
            calibration_kind = (
                str(calibration.get("kind", "")).strip().lower()
                if isinstance(calibration, Mapping)
                else ""
            )
            if not calibration_kind.startswith("verified"):
                raise ReviewEditorError("world reference requires a calibration marked verified")
    if coordinate_frame == "image" and source_value is None:
        raise ReviewEditorError("image reference requires a media source identity")
    timestamp = _validate_reference_timestamp(
        model,
        _current_time(model) if timestamp_s is None else timestamp_s,
    )
    return Reference(
        reference_id=reference_id or f"reference-{uuid.uuid4().hex}",
        coordinate_frame=coordinate_frame,
        point=point_value,
        source=source_value,
        timestamp_s=timestamp,
        source_revision=source_revision,
        calibration=calibration,
        source_point=None if source_point is None else _point(source_point, name="source_point"),
        actor_id=actor_id,
        object_id=object_id,
        goal_id=goal_id,
        waypoint_id=waypoint_id,
        metric_id=metric_id,
        event_id=event_id,
        seek_identity=str(_context_value(model, "execution_id", "")),
    )


def snap_reference(
    model: Mapping[str, Any],
    target: str,
    *,
    target_id: str = "",
    timestamp_s: float | None = None,
    source: SourceRef | None = None,
    source_revision: str = "",
    reference_id: str | None = None,
) -> Reference:
    """Snap to a recorded actor/goal/waypoint/map/event/metric sample."""

    target = str(target).strip().lower()
    if target not in REFERENCE_TARGETS:
        raise ReviewEditorError(f"snap target must be one of {REFERENCE_TARGETS}")
    timestamp = (
        _current_time(model) if timestamp_s is None else _finite(timestamp_s, name="timestamp_s")
    )
    source_value = source
    if target == "actor":
        if not target_id:
            target_id = str(_context_value(model, "actor_id", "robot"))
        sample_result = _stream_sample(model, "scene", timestamp)
        point = _actor_point(
            sample_result.value if sample_result.status == "available" else None,
            target_id,
        )
        if point is None:
            reason = sample_result.reason or "sample_unavailable"
            raise ReviewEditorError(f"recorded actor geometry unavailable: {target_id} ({reason})")
        return create_reference(
            model,
            point,
            reference_id=reference_id,
            timestamp_s=timestamp,
            source=source_value,
            source_revision=source_revision,
            actor_id=target_id,
        )
    if target == "goal":
        point = _goal_point(model)
        if point is None:
            raise ReviewEditorError("recorded goal geometry unavailable")
        return create_reference(
            model,
            point,
            reference_id=reference_id,
            timestamp_s=timestamp,
            source=source_value,
            source_revision=source_revision,
            goal_id=target_id or "goal",
        )
    if target in {"waypoint", "map"}:
        geometry = model.get("scene_surface")
        values: list[Any] = []
        if isinstance(geometry, Mapping):
            candidate = (
                geometry.get("waypoints") if target == "waypoint" else geometry.get("objects")
            )
            if isinstance(candidate, Sequence):
                for item in candidate:
                    if isinstance(item, Mapping) and (
                        not target_id or str(item.get("id", item.get("object_id", ""))) == target_id
                    ):
                        values.append(item.get("point", item.get("position")))
                    elif not isinstance(item, Mapping):
                        values.append(item)
        if not values:
            raise ReviewEditorError(f"recorded {target} geometry unavailable: {target_id}")
        point = _maybe_point(values[0])
        if point is None:
            raise ReviewEditorError(f"recorded {target} geometry unavailable: {target_id}")
        return create_reference(
            model,
            point,
            reference_id=reference_id,
            timestamp_s=timestamp,
            source=source_value,
            source_revision=source_revision,
            waypoint_id=target_id if target == "waypoint" else "",
            object_id=target_id if target == "map" else "",
        )
    if target == "event":
        events = model.get("events", [])
        event = next(
            (
                item
                for item in events
                if isinstance(item, Mapping)
                and str(item.get("event_id", item.get("interval_id", ""))) == target_id
            ),
            None,
        )
        if event is None:
            raise ReviewEditorError(f"recorded event unavailable: {target_id}")
        event_time = _finite(
            event.get("seek_time_s", event.get("start_s", timestamp)), name="event time"
        )
        actor_ids = event.get("actor_ids", [])
        actor_candidates = (
            [str(item) for item in actor_ids]
            if isinstance(actor_ids, Sequence) and not isinstance(actor_ids, (str, bytes))
            else []
        )
        sample_result = _stream_sample(model, "scene", event_time)
        sample = sample_result.value if sample_result.status == "available" else None
        actor_id = ""
        point = None
        for candidate in actor_candidates:
            point = _actor_point(sample, candidate)
            if point is not None:
                actor_id = candidate
                break
        if point is None:
            point = _maybe_point(event.get("point", event.get("position")))
        if point is None:
            reason = sample_result.reason or "sample_unavailable"
            raise ReviewEditorError(f"recorded event geometry unavailable: {target_id} ({reason})")
        return create_reference(
            model,
            point,
            reference_id=reference_id,
            timestamp_s=event_time,
            source=source_value,
            source_revision=source_revision,
            actor_id=actor_id,
            event_id=target_id,
        )
    # A metric has no spatial geometry.  Its point is the recorded chart
    # coordinate (time, value) and is explicitly image/chart space; no metres
    # or screen-pixel distance is inferred from it.
    metrics = model.get("metrics", {})
    metric = metrics.get(target_id) if isinstance(metrics, Mapping) else None
    stream = metric.get("stream", metric) if isinstance(metric, Mapping) else None
    sample_result = _nearest_stream_sample(
        {"status": "unavailable", "reason": "metric_stream_missing"}
        if not isinstance(stream, Mapping)
        else stream,
        timestamp,
    )
    if sample_result.status != "available" or sample_result.sample_time_s is None:
        reason = sample_result.reason or "sample_unavailable"
        raise ReviewEditorError(f"recorded metric sample unavailable: {target_id} ({reason})")
    raw_value = sample_result.value
    if isinstance(raw_value, Mapping):
        raw_value = raw_value.get("value")
    value = _finite(raw_value, name="metric value")
    return create_reference(
        model,
        (float(sample_result.sample_time_s), value),
        reference_id=reference_id,
        coordinate_frame="image",
        timestamp_s=float(sample_result.sample_time_s),
        source=source_value,
        source_revision=source_revision,
        metric_id=target_id,
    )


# Short aliases make the three speed API easy to discover without creating a
# second annotation model.  The ``make_*`` names remain the canonical owners.
one_click_annotation = make_one_click_annotation
quick_annotation = make_quick_annotation
structured_annotation = make_full_annotation
snap_to = snap_reference


def build_overlay_commands(
    model: Mapping[str, Any],
    references: Sequence[Reference | Mapping[str, Any]],
    *,
    overlays: OverlayState | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build overlays from recorded coordinates/units only.

    No command derives a world coordinate or distance from browser pixels.
    Image-space references can still be numbered/highlighted, while measured
    distance lines require two world references and declared units.
    """

    state = (
        overlays
        if isinstance(overlays, OverlayState)
        else OverlayState(
            **{
                kind: bool(overlays.get(kind, getattr(OverlayState(), kind)))
                for kind in OVERLAY_KINDS
            }
        )
        if isinstance(overlays, Mapping)
        else OverlayState()
    )
    normalized = tuple(
        item if isinstance(item, Reference) else Reference(**dict(item)) for item in references
    )
    commands: list[dict[str, Any]] = []
    for index, reference in enumerate(normalized, start=1):
        source_point = (
            list(reference.source_point)
            if reference.coordinate_frame == "image" and reference.source_point is not None
            else None
        )
        display_point = None
        if source_point is not None and reference.calibration is not None:
            transform = reference.calibration
            mapper = getattr(transform, "source_to_display", None)
            if callable(mapper):
                display_point = list(mapper(source_point))
        if state.numbered:
            commands.append(
                {
                    "kind": "numbered",
                    "number": index,
                    "reference_id": reference.reference_id,
                    "coordinate_frame": reference.coordinate_frame,
                    "point": list(reference.point),
                    "source_point": source_point,
                    "display_point": display_point,
                    "timestamp_s": reference.timestamp_s,
                }
            )
        if state.highlights and reference.actor_id:
            commands.append(
                {
                    "kind": "highlight",
                    "reference_id": reference.reference_id,
                    "actor_id": reference.actor_id,
                    "source_point": source_point,
                    "display_point": display_point,
                    "timestamp_s": reference.timestamp_s,
                }
            )
        if state.arrows and reference.actor_id:
            commands.append(
                {
                    "kind": "arrow",
                    "reference_id": reference.reference_id,
                    "actor_id": reference.actor_id,
                    "point": list(reference.point),
                    "source_point": source_point,
                    "display_point": display_point,
                    "coordinate_frame": reference.coordinate_frame,
                }
            )
        if state.rings:
            commands.append(
                {
                    "kind": "ring",
                    "reference_id": reference.reference_id,
                    "point": list(reference.point),
                    "source_point": source_point,
                    "display_point": display_point,
                    "coordinate_frame": reference.coordinate_frame,
                }
            )
    if state.distances:
        world = [item for item in normalized if item.coordinate_frame == "world"]
        units = _units_for(model, world[0].source if world else None)
        world_units = {_units_for(model, item.source) for item in world}
        world_sources = {
            (
                item.source.artifact_id if item.source is not None else "",
                item.source.sha256 if item.source is not None else "",
            )
            for item in world
        }
        if len(world) >= 2 and len(world_units) == 1 and len(world_sources) == 1 and units:
            first, second = world[:2]
            distance = math.hypot(
                first.point[0] - second.point[0], first.point[1] - second.point[1]
            )
            commands.append(
                {
                    "kind": "distance",
                    "from": first.reference_id,
                    "to": second.reference_id,
                    "distance": distance,
                    "units": units,
                }
            )
    return tuple(commands)


@dataclass(frozen=True, slots=True)
class StoryboardState:
    """Mutable-in-practice storyboard data represented as immutable snapshots."""

    intervals: tuple[Mapping[str, Any], ...] = ()
    order: tuple[str, ...] = ()
    captions: Mapping[str, str] = field(default_factory=dict)
    source_identity: str = ""
    duration_s: float | None = None
    origin_s: float = 0.0
    source_revision: str = ""

    def __post_init__(self) -> None:
        entries: list[dict[str, Any]] = []
        identifiers: list[str] = []
        for raw_entry in self.intervals:
            if not isinstance(raw_entry, Mapping):
                raise ReviewEditorError("storyboard intervals must contain objects")
            entry = dict(raw_entry)
            interval_id = _text(
                entry.get("interval_id", entry.get("id", "")), name="interval_id", limit=512
            )
            interval = validate_interval(entry, duration_s=self.duration_s, origin_s=self.origin_s)
            entry["interval_id"] = interval_id
            entry["start_s"] = interval.start_s
            entry["end_s"] = interval.end_s
            entry["caption"] = str(entry.get("caption", self.captions.get(interval_id, "")))
            entries.append(entry)
            identifiers.append(interval_id)
        order = tuple(str(item) for item in self.order)
        if set(order) != set(identifiers) or len(order) != len(identifiers):
            raise ReviewEditorError("storyboard order must contain every interval exactly once")
        captions = {
            str(key): str(value) for key, value in self.captions.items() if str(key) in identifiers
        }
        captions.update({entry["interval_id"]: str(entry.get("caption", "")) for entry in entries})
        object.__setattr__(self, "intervals", tuple(entries))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "captions", captions)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        duration_s: float | None = None,
        source_identity: str = "",
        origin_s: float = 0.0,
        source_revision: str = "",
    ) -> StoryboardState:
        if value is None:
            return cls(
                duration_s=duration_s,
                source_identity=source_identity,
                origin_s=origin_s,
                source_revision=source_revision,
            )
        if not isinstance(value, Mapping):
            raise ReviewEditorError("storyboard must be an object")
        version = value.get("schema_version")
        if version != STORYBOARD_SCHEMA_VERSION:
            if version is None:
                raise ReviewEditorError("storyboard schema_version is required")
            raise ReviewEditorError(f"unsupported storyboard schema_version: {version!r}")
        required_fields = ("intervals", "order", "captions")
        missing = [field_name for field_name in required_fields if field_name not in value]
        if missing:
            raise ReviewEditorError("storyboard fields are required: " + ", ".join(missing))
        raw_intervals = value["intervals"]
        if not isinstance(raw_intervals, Sequence) or isinstance(raw_intervals, (str, bytes)):
            raise ReviewEditorError("storyboard intervals must be a list")
        if any(not isinstance(item, Mapping) for item in raw_intervals):
            raise ReviewEditorError("storyboard intervals must contain objects")
        order = value["order"]
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ReviewEditorError("storyboard order must be a list")
        captions = value["captions"]
        if not isinstance(captions, Mapping):
            raise ReviewEditorError("storyboard captions must be a mapping")
        raw_source_identity = value.get("source_identity", "")
        raw_source_revision = value.get("source_revision", "")
        if not isinstance(raw_source_identity, str):
            raise ReviewEditorError("storyboard source_identity must be text")
        if not isinstance(raw_source_revision, str):
            raise ReviewEditorError("storyboard source_revision must be text")
        return cls(
            tuple(dict(item) for item in raw_intervals),
            tuple(str(item) for item in order),
            dict(captions),
            source_identity or raw_source_identity,
            duration_s if duration_s is not None else value.get("duration_s"),
            _finite(value.get("origin_s", origin_s), name="origin_s"),
            source_revision or raw_source_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STORYBOARD_SCHEMA_VERSION,
            "source_identity": self.source_identity,
            "duration_s": self.duration_s,
            "origin_s": self.origin_s,
            "source_revision": self.source_revision,
            "intervals": [dict(item) for item in self.intervals],
            "order": list(self.order),
            "captions": dict(self.captions),
        }


class StoryboardEditor:
    """Undoable interval/caption/reorder editor with source-bound validation."""

    def __init__(
        self,
        storyboard: StoryboardState | Mapping[str, Any] | None = None,
        *,
        duration_s: float | None = None,
        source_identity: str = "",
        origin_s: float = 0.0,
        source_revision: str = "",
    ):
        self._state = (
            storyboard
            if isinstance(storyboard, StoryboardState)
            else StoryboardState.from_mapping(
                storyboard,
                duration_s=duration_s,
                source_identity=source_identity,
                origin_s=origin_s,
                source_revision=source_revision,
            )
        )
        self._undo: list[StoryboardState] = []
        self._redo: list[StoryboardState] = []

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        duration_s: float | None = None,
        source_identity: str = "",
        origin_s: float = 0.0,
        source_revision: str = "",
    ) -> StoryboardEditor:
        """Build an editor from a versioned or compatible storyboard mapping."""

        return cls(
            StoryboardState.from_mapping(
                value,
                duration_s=duration_s,
                source_identity=source_identity,
                origin_s=origin_s,
                source_revision=source_revision,
            )
        )

    @property
    def state(self) -> StoryboardState:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        return self._state.to_dict()

    def _replace(self, state: StoryboardState) -> StoryboardState:
        self._undo.append(self._state)
        self._state = state
        self._redo.clear()
        return self._state

    def add_interval(
        self, interval_id: str, start_s: float, end_s: float, *, caption: str = ""
    ) -> StoryboardState:
        normalized_id = _text(interval_id, name="interval_id", limit=512)
        if any(item["interval_id"] == normalized_id for item in self._state.intervals):
            raise ReviewEditorError(f"duplicate storyboard interval: {normalized_id}")
        interval = validate_interval(
            {"start_s": start_s, "end_s": end_s},
            duration_s=self._state.duration_s,
            origin_s=self._state.origin_s,
        )
        entry = {
            "interval_id": normalized_id,
            "start_s": interval.start_s,
            "end_s": interval.end_s,
            "caption": str(caption),
        }
        return self._replace(
            StoryboardState(
                self._state.intervals + (entry,),
                self._state.order + (normalized_id,),
                {**self._state.captions, normalized_id: str(caption)},
                self._state.source_identity,
                self._state.duration_s,
                self._state.origin_s,
                self._state.source_revision,
            )
        )

    def update_interval(self, interval_id: str, start_s: float, end_s: float) -> StoryboardState:
        interval = validate_interval(
            {"start_s": start_s, "end_s": end_s},
            duration_s=self._state.duration_s,
            origin_s=self._state.origin_s,
        )
        entries = [dict(item) for item in self._state.intervals]
        for entry in entries:
            if entry["interval_id"] == interval_id:
                entry["start_s"], entry["end_s"] = interval.start_s, interval.end_s
                return self._replace(
                    StoryboardState(
                        tuple(entries),
                        self._state.order,
                        self._state.captions,
                        self._state.source_identity,
                        self._state.duration_s,
                        self._state.origin_s,
                        self._state.source_revision,
                    )
                )
        raise ReviewEditorError(f"unknown storyboard interval: {interval_id}")

    def set_caption(self, interval_id: str, caption: str) -> StoryboardState:
        if interval_id not in self._state.order:
            raise ReviewEditorError(f"unknown storyboard interval: {interval_id}")
        entries = [dict(item) for item in self._state.intervals]
        for entry in entries:
            if entry["interval_id"] == interval_id:
                entry["caption"] = str(caption)
        return self._replace(
            StoryboardState(
                tuple(entries),
                self._state.order,
                {**self._state.captions, interval_id: str(caption)},
                self._state.source_identity,
                self._state.duration_s,
                self._state.origin_s,
                self._state.source_revision,
            )
        )

    def reorder(self, order: Sequence[str]) -> StoryboardState:
        return self._replace(
            StoryboardState(
                self._state.intervals,
                tuple(str(item) for item in order),
                self._state.captions,
                self._state.source_identity,
                self._state.duration_s,
                self._state.origin_s,
                self._state.source_revision,
            )
        )

    def remove(self, interval_id: str) -> StoryboardState:
        if interval_id not in self._state.order:
            raise ReviewEditorError(f"unknown storyboard interval: {interval_id}")
        entries = tuple(
            item for item in self._state.intervals if item["interval_id"] != interval_id
        )
        captions = {key: value for key, value in self._state.captions.items() if key != interval_id}
        return self._replace(
            StoryboardState(
                entries,
                tuple(item for item in self._state.order if item != interval_id),
                captions,
                self._state.source_identity,
                self._state.duration_s,
                self._state.origin_s,
                self._state.source_revision,
            )
        )

    def undo(self) -> StoryboardState:
        if not self._undo:
            return self._state
        self._redo.append(self._state)
        self._state = self._undo.pop()
        return self._state

    def redo(self) -> StoryboardState:
        if not self._redo:
            return self._state
        self._undo.append(self._state)
        self._state = self._redo.pop()
        return self._state

    def save(self, destination: str | Path, *, overwrite: bool = False) -> Path:
        path = Path(destination)
        payload = (canonical_json(self._state.to_dict()) + "\n").encode("utf-8")
        _write_atomic(path, payload, overwrite=overwrite)
        return path

    export = save

    @classmethod
    def load(cls, path: str | Path) -> StoryboardEditor:
        payload = _strict_loads(_read_regular_path(Path(path), kind="storyboard"))
        if not isinstance(payload, Mapping):
            raise ReviewEditorError("storyboard export must be an object")
        return cls(StoryboardState.from_mapping(payload))


@dataclass(frozen=True, slots=True)
class SaveConflict:
    """Human-readable conflict receipt preserving both local and remote edits."""

    record_id: str
    expected_revision: int | None
    actual_revision: int
    local_record: Mapping[str, Any]
    remote_record: Mapping[str, Any] | None
    selection_revision: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewEditorSession:
    """Headless editor session shared by tests, CLI integrations and a browser.

    The session never writes source artifacts.  ``save_annotation`` is the
    only canonical mutation and requires an adapter, operation ID and expected
    revision.  Selection revision checks prevent a delayed save from being
    attached to a newer episode/interval.
    """

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        source_refs: Mapping[str, SourceRef] | None = None,
        source_digests: Mapping[str, str] | None = None,
        adapter: PersistenceAdapter | None = None,
        actor_kind: str = "human",
        actor_id: str = "",
        storyboard: StoryboardEditor | None = None,
    ):
        self._state_lock = threading.Lock()
        self.model = json.loads(canonical_json(model))
        model_refs, model_digests = _source_bindings_from_model(self.model)
        self.source_refs = dict(model_refs if source_refs is None else source_refs)
        self.source_digests = dict(model_digests if source_digests is None else source_digests)
        self.adapter = adapter
        if actor_kind not in AUTHOR_KINDS:
            raise ReviewEditorError(f"actor_kind must be one of {AUTHOR_KINDS}")
        self.actor_kind = actor_kind
        self.actor_id = actor_id
        self.selection_revision = _selection_revision(self.model)
        self.storyboard = storyboard
        if self.storyboard is None:
            duration = (
                self.model.get("time", {}).get("terminal_s")
                if isinstance(self.model.get("time"), Mapping)
                else None
            )
            origin = (
                self.model.get("time", {}).get("origin_s", 0.0)
                if isinstance(self.model.get("time"), Mapping)
                else 0.0
            )
            self.storyboard = StoryboardEditor(
                StoryboardState.from_mapping(
                    self.model.get("storyboard"),
                    duration_s=duration,
                    origin_s=origin,
                    source_identity=self._source_identity_token(),
                    source_revision=self._source_revision_token(),
                )
            )
        self.autosave_status = AutosaveStatus(
            state="saved", selection_revision=self.selection_revision
        )
        self.pending_record: Any | None = None

    def _source_identity_token(self) -> str:
        identity = _source_identity(self.model, self.source_refs, self.source_digests)
        return hashlib.sha256(canonical_json(identity).encode()).hexdigest()

    def _source_revision_token(self) -> str:
        return _source_revision_token(self.model, self.source_refs, self.source_digests)

    def select(
        self,
        *,
        time_s: float | None = None,
        interval_id: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        with self._state_lock:
            context = (
                dict(self.model.get("context", {}))
                if isinstance(self.model.get("context"), Mapping)
                else {}
            )
            if time_s is not None:
                _finite(time_s, name="time_s")
                context["cursor"] = {
                    **(
                        context.get("cursor", {})
                        if isinstance(context.get("cursor"), Mapping)
                        else {}
                    ),
                    "time_s": float(time_s),
                }
            if interval_id is not None:
                context["interval_id"] = interval_id
            if episode_id is not None:
                context["episode_id"] = episode_id
            self.selection_revision += 1
            context["context_revision"] = self.selection_revision
            self.model["context"] = context
            return self.selection_revision

    def _check_selection(self, expected_selection_revision: int | None) -> None:
        if (
            expected_selection_revision is not None
            and expected_selection_revision != self.selection_revision
        ):
            raise StaleSelectionError(expected_selection_revision, self.selection_revision)

    def create_one_click(self, label: str, **kwargs: Any) -> Annotation:
        kwargs.setdefault("author_kind", self.actor_kind)
        return make_one_click_annotation(
            self.model,
            label,
            source_refs=self.source_refs,
            source_digests=self.source_digests,
            **kwargs,
        )

    def create_quick(self, classification: str, **kwargs: Any) -> Annotation:
        kwargs.setdefault("author_kind", self.actor_kind)
        return make_quick_annotation(
            self.model,
            classification,
            source_refs=self.source_refs,
            source_digests=self.source_digests,
            **kwargs,
        )

    def create_full(self, classification: str, **kwargs: Any) -> Annotation:
        kwargs.setdefault("author_kind", self.actor_kind)
        return make_full_annotation(
            self.model,
            classification,
            source_refs=self.source_refs,
            source_digests=self.source_digests,
            **kwargs,
        )

    def _save(
        self,
        record: Any,
        *,
        operation_id: str | None,
        expected_revision: int | None,
        expected_selection_revision: int | None,
    ) -> CommitResult:
        # Selection changes and the adapter's compare-and-swap form one
        # critical section.  This lock intentionally spans a blocking adapter
        # call: a different thread cannot advance the captured context after
        # the preflight check but before the durable write.
        with self._state_lock:
            return self._save_locked(
                record,
                operation_id=operation_id,
                expected_revision=expected_revision,
                expected_selection_revision=expected_selection_revision,
            )

    def _save_locked(
        self,
        record: Any,
        *,
        operation_id: str | None,
        expected_revision: int | None,
        expected_selection_revision: int | None,
    ) -> CommitResult:
        bound_selection_revision = expected_selection_revision
        if bound_selection_revision is None:
            metadata = getattr(record, "metadata", None)
            if isinstance(metadata, Mapping):
                value = metadata.get("selection_revision")
                if isinstance(value, int) and not isinstance(value, bool):
                    bound_selection_revision = value
            elif isinstance(record, ReviewRecord):
                bound_selection_revision = record.source_revision
            if bound_selection_revision is None:
                bound_selection_revision = self.selection_revision
        self._check_selection(bound_selection_revision)
        if self.adapter is None:
            raise ReviewEditorError("a BA-03/BA-05 persistence adapter is required for save")
        op_id = operation_id or f"review-editor-{uuid.uuid4().hex}"
        record_id = getattr(
            record,
            "annotation_id",
            getattr(record, "review_id", getattr(record, "action_id", "")),
        )
        existing = self.adapter.get(record_id, include_deleted=True)
        declared_author = getattr(record, "author_kind", getattr(record, "actor_kind", None))
        if (
            existing is not None
            and existing.record is not None
            and declared_author is not None
            and getattr(
                existing.record,
                "author_kind",
                getattr(existing.record, "actor_kind", declared_author),
            )
            != declared_author
        ):
            raise ReviewEditorError(
                "author_kind change requires a new record; preserve the original history"
            )
        self.pending_record = record
        self.autosave_status = AutosaveStatus(
            "pending",
            op_id,
            record_id,
            expected_revision,
            None,
            self.selection_revision,
        )
        try:
            receipt = self.adapter.save(
                record,
                operation_id=op_id,
                expected_revision=expected_revision,
                actor=self.actor_kind,
                actor_id=self.actor_id,
            )
        except (AuditConflictError, AuditStoreError, OSError, ReviewEditorError) as error:
            conflict: SaveConflict | None = None
            if isinstance(error, AuditConflictError):
                try:
                    remote = self.adapter.get(error.record_id, include_deleted=True)
                except Exception:  # noqa: BLE001 - preserve the local failed edit
                    remote = None
                conflict = SaveConflict(
                    error.record_id,
                    error.expected_revision,
                    error.actual_revision,
                    record_to_dict(record),
                    record_to_dict(remote.record) if remote and remote.record is not None else None,
                    self.selection_revision,
                )
            self.autosave_status = AutosaveStatus(
                "error",
                op_id,
                self.autosave_status.record_id,
                expected_revision,
                None,
                self.selection_revision,
                str(error),
                conflict.to_dict() if conflict else None,
            )
            raise
        except Exception as error:
            self.autosave_status = AutosaveStatus(
                "error",
                op_id,
                self.autosave_status.record_id,
                expected_revision,
                None,
                self.selection_revision,
                str(error),
                None,
            )
            raise
        self.pending_record = None
        self.autosave_status = AutosaveStatus(
            "saved",
            op_id,
            receipt.record_id,
            expected_revision,
            receipt.revision,
            self.selection_revision,
        )
        return receipt

    def save_annotation(
        self,
        annotation: Annotation,
        *,
        operation_id: str | None = None,
        expected_revision: int | None = None,
        expected_selection_revision: int | None = None,
    ) -> CommitResult:
        return self._save(
            annotation,
            operation_id=operation_id,
            expected_revision=expected_revision,
            expected_selection_revision=expected_selection_revision,
        )

    def save_review(
        self,
        review: ReviewRecord,
        *,
        operation_id: str | None = None,
        expected_revision: int | None = None,
        expected_selection_revision: int | None = None,
    ) -> CommitResult:
        return self._save(
            review,
            operation_id=operation_id,
            expected_revision=expected_revision,
            expected_selection_revision=expected_selection_revision,
        )

    def storyboard_record_id(self) -> str:
        """Return the stable BA-03 record ID for this source storyboard."""

        return "storyboard-" + hashlib.sha256(self._source_identity_token().encode()).hexdigest()

    def save_storyboard(
        self,
        *,
        operation_id: str | None = None,
        expected_revision: int | None = None,
        expected_selection_revision: int | None = None,
    ) -> CommitResult:
        """Persist the storyboard through the BA-03/BA-05 adapter.

        BA-03 has no separate storyboard record type, so the edit is an
        auditable ``ActionRecord`` whose closed details carry the versioned
        storyboard document.  It remains in the canonical journal and uses
        the same compare-and-swap and operation-id semantics as annotations.
        """

        storyboard = self.storyboard.snapshot()
        record = ActionRecord(
            action_id=self.storyboard_record_id(),
            action_type="storyboard_edit",
            actor_kind=self.actor_kind,
            actor_id=self.actor_id,
            target_id=self._source_identity_token(),
            details={
                "schema_version": STORYBOARD_SCHEMA_VERSION,
                "record_id": self.storyboard_record_id(),
                "source_identity": self._source_identity_token(),
                "source_revision": self._source_revision_token(),
                "storyboard": storyboard,
            },
        )
        return self._save(
            record,
            operation_id=operation_id,
            expected_revision=expected_revision,
            expected_selection_revision=expected_selection_revision,
        )

    def reload_storyboard(self) -> StoryboardEditor:
        """Reload the last durable storyboard edit from the shared adapter."""

        if self.adapter is None:
            raise ReviewEditorError("a persistence adapter is required for reload")
        stored = self.adapter.get(self.storyboard_record_id(), include_deleted=True)
        if stored is None:
            return self.storyboard
        if stored.deleted or stored.record is None:
            raise ReviewEditorError("stored storyboard tombstone is not reloadable")
        if (
            not isinstance(stored.revision, int)
            or isinstance(stored.revision, bool)
            or stored.revision < 0
        ):
            raise ReviewEditorError("stored storyboard record revision is invalid")
        if stored.record_type != "action_record" or not isinstance(stored.record, ActionRecord):
            raise ReviewEditorError("stored storyboard record type is invalid")
        record = stored.record
        if (
            record.action_type != "storyboard_edit"
            or record.action_id != self.storyboard_record_id()
        ):
            raise ReviewEditorError(
                "stored storyboard record identity does not match current source"
            )
        if record.status != "committed":
            raise ReviewEditorError("stored storyboard action status is not committed")
        if record.target_id != self._source_identity_token():
            raise ReviewEditorError("stored storyboard target identity is stale")
        details = record.details
        if (
            not isinstance(details, Mapping)
            or details.get("schema_version") != STORYBOARD_SCHEMA_VERSION
        ):
            raise ReviewEditorError("stored storyboard schema_version is invalid")
        required_details = ("record_id", "source_identity", "source_revision", "storyboard")
        missing_details = [name for name in required_details if name not in details]
        if missing_details:
            raise ReviewEditorError(
                "stored storyboard provenance is incomplete: " + ", ".join(missing_details)
            )
        if details.get("record_id") != self.storyboard_record_id():
            raise ReviewEditorError("stored storyboard record_id does not match current source")
        if details.get("source_identity") != self._source_identity_token():
            raise ReviewEditorError("stored storyboard source identity is stale")
        if details.get("source_revision") != self._source_revision_token():
            raise ReviewEditorError("stored storyboard source revision is stale")
        payload = details.get("storyboard")
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != STORYBOARD_SCHEMA_VERSION
        ):
            raise ReviewEditorError("stored storyboard schema_version is invalid")
        if payload.get("source_identity") != self._source_identity_token():
            raise ReviewEditorError("stored storyboard source identity is stale")
        if "source_revision" not in payload:
            raise ReviewEditorError("stored storyboard source revision is required")
        if payload.get("source_revision") != self._source_revision_token():
            raise ReviewEditorError("stored storyboard source revision is stale")
        time_origin, time_terminal = _time_bounds(self.model)
        self.storyboard = StoryboardEditor.from_mapping(
            payload,
            duration_s=time_terminal,
            origin_s=time_origin or 0.0,
            source_identity=self._source_identity_token(),
            source_revision=self._source_revision_token(),
        )
        self.pending_record = None
        self.autosave_status = AutosaveStatus(
            "saved",
            record_id=self.storyboard_record_id(),
            saved_revision=stored.revision,
            selection_revision=self.selection_revision,
        )
        return self.storyboard

    autosave = save_annotation

    def reload(self, record_id: str) -> StoredRecord | None:
        if self.adapter is None:
            raise ReviewEditorError("a persistence adapter is required for reload")
        loaded = self.adapter.get(record_id, include_deleted=True)
        self.pending_record = None
        self.autosave_status = AutosaveStatus(
            "saved",
            record_id=record_id,
            saved_revision=loaded.revision if loaded else 0,
            selection_revision=self.selection_revision,
        )
        return loaded

    def editor_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": EDITOR_MODEL_SCHEMA_VERSION,
            "context": {
                **dict(self.model.get("context", {})),
                "selection_revision": self.selection_revision,
            },
            "storyboard": self.storyboard.snapshot(),
            "autosave": self.autosave_status.to_dict(),
            "pending_record": record_to_dict(self.pending_record)
            if self.pending_record is not None
            else None,
        }


# Keep the concise public name available for integrations that treat the
# session as the editor facade.  This is intentionally the same class, not a
# second state or persistence implementation.
ReviewEditor = ReviewEditorSession


def _normalize_storyboard(
    value: Mapping[str, Any] | None,
    *,
    duration_s: float | None,
    source_identity: str,
    source_revision: str = "",
    origin_s: float = 0.0,
) -> dict[str, Any]:
    return StoryboardState.from_mapping(
        value,
        duration_s=duration_s,
        origin_s=origin_s,
        source_identity=source_identity,
        source_revision=source_revision,
    ).to_dict()


def _derive_time_bounds(model: Mapping[str, Any]) -> tuple[float | None, float | None]:
    time_block = model.get("time")
    if isinstance(time_block, Mapping):
        origin = time_block.get("origin_s", time_block.get("start_s"))
        terminal = time_block.get("terminal_s", time_block.get("end_s"))
        if isinstance(terminal, (int, float)) and not isinstance(terminal, bool):
            return (
                float(origin) if isinstance(origin, (int, float)) else 0.0,
                float(terminal),
            )
    streams = model.get("streams")
    samples_times: list[float] = []
    if isinstance(streams, Mapping):
        for stream in streams.values():
            if isinstance(stream, Mapping):
                samples = stream.get("samples", [])
                for sample in samples if isinstance(samples, Sequence) else ():
                    if isinstance(sample, Mapping) and isinstance(
                        sample.get("time_s"), (int, float)
                    ):
                        samples_times.append(float(sample["time_s"]))
    return (min(samples_times), max(samples_times)) if samples_times else (None, None)


def _derive_duration(model: Mapping[str, Any]) -> float | None:
    return _derive_time_bounds(model)[1]


def _annotation_hints(model: Mapping[str, Any]) -> dict[str, Any]:
    """Expose recorded context hints without deriving new geometry.

    The browser can show these beside the annotation controls.  Values remain
    ``None`` when the panel source did not declare them; in particular, this
    helper never guesses an actor, goal, or completion boundary from pixels or
    from a nominal frame rate.
    """

    context = model.get("context") if isinstance(model.get("context"), Mapping) else {}
    goal = model.get("goal_geometry")
    goal_mapping = goal if isinstance(goal, Mapping) else {}
    active_goal = goal_mapping.get("active_goal", goal_mapping.get("active_goal_point"))
    final_goal = goal_mapping.get(
        "final_goal", goal_mapping.get("final_goal_point", goal_mapping.get("goal_point"))
    )
    completion: list[dict[str, Any]] = []
    events = model.get("events", [])
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            event_kind = str(event.get("kind", event.get("event_type", ""))).lower()
            if "complet" in event_kind or "success" in event_kind or "goal" in event_kind:
                completion.append(
                    {
                        "event_id": str(event.get("event_id", event.get("id", ""))),
                        "start_s": event.get("start_s", event.get("time_s")),
                        "end_s": event.get("end_s", event.get("time_s")),
                    }
                )
    return {
        "selected_actor": context.get("actor_id"),
        "active_goal": json.loads(canonical_json(active_goal)) if active_goal is not None else None,
        "final_goal": json.loads(canonical_json(final_goal)) if final_goal is not None else None,
        "completion_boundaries": completion,
    }


def _rebind_annotation_provenance(
    item: Mapping[str, Any],
    source_refs: Mapping[str, SourceRef],
    source_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Carry forward raw annotations while marking changed source bindings stale."""

    result = dict(item)
    raw_source = item.get("source_ref")
    artifact_id = ""
    if isinstance(raw_source, Mapping):
        artifact_id = str(raw_source.get("artifact_id", ""))
    if not artifact_id and len(source_refs) == 1:
        artifact_id = next(iter(source_refs))
    ref = source_refs.get(artifact_id)
    raw_identity = item.get("source_identity")
    identity = raw_identity if isinstance(raw_identity, str) else ""
    if ref is None and identity:
        ref = next(
            (
                candidate
                for candidate_id, candidate in source_refs.items()
                if identity
                in {
                    candidate_id,
                    candidate.sha256,
                    source_digests.get(candidate_id, ""),
                }
            ),
            None,
        )
    if ref is None:
        if item.get("provenance_status") == "verified":
            result["provenance_status"] = "unavailable"
            result["provenance_reason"] = "annotation source binding is unavailable"
        return result
    if not artifact_id:
        artifact_id = ref.artifact_id
    digest = source_digests.get(artifact_id, "")
    raw_revision = item.get("source_revision")
    revision = raw_revision if isinstance(raw_revision, str) else ""
    stale_reason = ""
    unavailable_reason = ""
    if not digest or not ref.sha256 or not ref.source_commit:
        unavailable_reason = "annotation source bytes are unavailable"
    elif ref.sha256.lower() != digest.lower():
        stale_reason = "source bytes do not match declared source hash"
    elif not identity or identity != digest:
        stale_reason = "annotation source identity does not match current source"
    elif not revision or revision != ref.source_commit:
        stale_reason = "annotation source revision does not match current source"
    if unavailable_reason and item.get("provenance_status") == "verified":
        result["provenance_status"] = "unavailable"
        result["provenance_reason"] = unavailable_reason
    elif stale_reason:
        result["provenance_status"] = "stale"
        result["provenance_reason"] = stale_reason
    return result


def _build_model(
    request: ComponentRequest,
    *,
    panel_model: Mapping[str, Any],
    source_refs: Mapping[str, SourceRef],
    source_digests: Mapping[str, str],
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    identity = _source_identity(panel_model, source_refs, source_digests)
    source_identity_token = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    source_revision_token = _source_revision_token(panel_model, source_refs, source_digests)
    origin, duration = _derive_time_bounds(panel_model)
    storyboard_value = panel_model.get("storyboard")
    if storyboard_value is None:
        storyboard_value = request.config.get("storyboard")
    storyboard = _normalize_storyboard(
        storyboard_value if isinstance(storyboard_value, Mapping) else None,
        duration_s=duration,
        origin_s=origin or 0.0,
        source_identity=source_identity_token,
        source_revision=source_revision_token,
    )
    annotations = request.config.get("annotations", panel_model.get("annotations", []))
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        raise ReviewEditorError("annotations must be a list")
    serialized_annotations = [
        _rebind_annotation_provenance(item, source_refs, source_digests)
        for item in annotations
        if isinstance(item, Mapping)
    ]
    context = (
        dict(panel_model.get("context", {}))
        if isinstance(panel_model.get("context"), Mapping)
        else {}
    )
    context.setdefault("episode_id", _context_value(panel_model, "episode_id", "episode"))
    context.setdefault("execution_id", _context_value(panel_model, "execution_id", ""))
    context["selection_revision"] = _selection_revision(panel_model)
    overlay_config = request.config.get("overlays", panel_model.get("overlays", {}))
    overlay_state = (
        OverlayState(
            **{
                kind: bool(overlay_config.get(kind, getattr(OverlayState(), kind)))
                for kind in OVERLAY_KINDS
            }
        )
        if isinstance(overlay_config, Mapping)
        else OverlayState()
    )
    return {
        "schema_version": EDITOR_MODEL_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "context": context,
        "source_identity": identity,
        "storyboard_record_id": "storyboard-"
        + hashlib.sha256(source_identity_token.encode()).hexdigest(),
        # Keep the renderer-neutral panel data available to snapping and
        # offline reference inspection.  It is copied, never edited in place;
        # source files remain immutable.
        "panel_model": json.loads(canonical_json(panel_model)),
        "streams": json.loads(canonical_json(panel_model.get("streams", {}))),
        "scene_surface": json.loads(canonical_json(panel_model.get("scene_surface", {}))),
        "goal_geometry": json.loads(canonical_json(panel_model.get("goal_geometry", {}))),
        "metrics": json.loads(canonical_json(panel_model.get("metrics", {}))),
        "events": json.loads(canonical_json(panel_model.get("events", []))),
        "time": json.loads(canonical_json(panel_model.get("time", {}))),
        "annotation_hints": _annotation_hints(panel_model),
        "source_identity_status": {
            artifact_id: {
                "sha256": digest,
                "status": (
                    "unavailable"
                    if not source_refs[artifact_id].sha256
                    else (
                        "verified"
                        if source_refs[artifact_id].sha256.lower() == digest.lower()
                        else "stale"
                    )
                ),
            }
            for artifact_id, digest in source_digests.items()
            if artifact_id in source_refs
        },
        "annotation_modes": list(ANNOTATION_SPEEDS),
        "classifications": list(ANNOTATION_CLASSIFICATIONS),
        "triage_labels": dict(TRIAGE_CLASSIFICATIONS),
        "annotations": serialized_annotations,
        "overlay_state": overlay_state.to_dict(),
        "storyboard": storyboard,
        "controls": {
            "typing_suppresses_shortcuts": True,
            "shortcuts": {"undo": "Ctrl+Z", "redo": "Ctrl+Shift+Z", "save": "Ctrl+S"},
            "source_time_authority": "simulation_time",
        },
        "persistence": {
            "adapter": "ba03-audit-store",
            "service_boundary": "ba05-service-compatible",
            "canonical": "AuditStore/audit.ndjson",
            "browser_local_storage": False,
            "operation_id_required": True,
            "expected_revision_required": True,
            "acknowledge_after_durable_commit": True,
        },
        "coverage": coverage_summary(()),
        "diagnostics": [dict(item) for item in diagnostics],
        "provenance": {"evidence_status": "diagnostic_only", "source_mutation": "none"},
    }


def _find_panel_model(
    request: ComponentRequest, *, base: Path, diagnostics: list[dict[str, Any]]
) -> tuple[Mapping[str, Any], dict[str, SourceRef], dict[str, str]]:
    refs = {ref.artifact_id: ref for ref in request.sources}
    digests: dict[str, str] = {}
    payloads: dict[str, Any] = {}
    for ref in request.sources:
        try:
            payload, digest, matches = _read_source(base, ref)
        except ReviewEditorError as error:
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason_code": "source_unreadable",
                    "detail": str(error),
                }
            )
            continue
        digests[ref.artifact_id] = digest
        payloads[ref.artifact_id] = payload
        if not matches:
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason_code": "source_stale",
                    "detail": "source bytes do not match declared hash; references remain stale",
                }
            )
    inline = request.config.get("panel_model")
    if isinstance(inline, Mapping):
        if inline.get("schema_version") != "review-panels.v1":
            raise ReviewEditorError("panel_model schema_version must be review-panels.v1")
        return inline, refs, digests
    for ref in request.sources:
        payload = payloads.get(ref.artifact_id)
        if isinstance(payload, Mapping) and payload.get("schema_version") == "review-panels.v1":
            return payload, refs, digests
        if isinstance(payload, Mapping) and ref.format in {"review-panels", "review-panels.v1"}:
            raise ReviewEditorError(
                f"source {ref.artifact_id!r} does not declare review-panels.v1 schema"
            )
    # A direct scene model is useful for a minimal offline fixture.  Keep it
    # clearly diagnostic and do not attempt to run a simulator or renderer.
    for ref in request.sources:
        payload = payloads.get(ref.artifact_id)
        if isinstance(payload, Mapping) and isinstance(payload.get("frames"), Sequence):
            scene = payload
            identity = (
                scene.get("source_identity", {})
                if isinstance(scene.get("source_identity"), Mapping)
                else {}
            )
            frame_samples = [
                {"time_s": row.get("time_s"), "value": row}
                for row in scene.get("frames", [])
                if isinstance(row, Mapping) and isinstance(row.get("time_s"), (int, float))
            ]
            return (
                {
                    "schema_version": "review-panels.v1",
                    "context": dict(identity),
                    "source_identity": dict(identity),
                    "streams": {
                        "scene": {
                            "status": "available",
                            "samples": frame_samples,
                            "resolution_s": 0.0,
                        }
                    },
                    "scene_surface": scene.get("map", {}),
                    "goal_geometry": {
                        "goal_point": scene.get("goal", {}).get("point")
                        if isinstance(scene.get("goal"), Mapping)
                        else None
                    },
                    "time": {
                        "terminal_s": max((row["time_s"] for row in frame_samples), default=0.0),
                        "cursor": {"time_s": frame_samples[0]["time_s"] if frame_samples else 0.0},
                    },
                },
                refs,
                digests,
            )
    raise ReviewEditorError("review-panels.v1 source or inline panel_model is required")


def build_editor_model(request: ComponentRequest, *, base: Path | None = None) -> dict[str, Any]:
    """Build an editor model without writing output files."""

    if not isinstance(request, ComponentRequest):
        raise TypeError("request must be ComponentRequest")
    diagnostics: list[dict[str, Any]] = []
    panel_model, refs, digests = _find_panel_model(
        request, base=(base or Path.cwd()).resolve(), diagnostics=diagnostics
    )
    return _build_model(
        request,
        panel_model=panel_model,
        source_refs=refs,
        source_digests=digests,
        diagnostics=diagnostics,
    )


def _write_json(path: Path, payload: Any) -> str:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    _write_atomic(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _render_html(model: Mapping[str, Any]) -> str:
    payload = canonical_json(model).replace("</", "<\\/")
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Robot SF review editor</title>
<style>body{font-family:system-ui,sans-serif;margin:1rem;color:#111}button{margin:.2rem}.editor-grid{display:grid;grid-template-columns:2fr 1fr;gap:1rem}.surface{min-height:18rem;border:1px solid #aaa;padding:1rem}.stale{color:#9a3412}.autosave{font-weight:600}</style></head>
<body><h1>Offline review editor</h1><div class="editor-grid"><section class="surface" id="review-editor-root" tabindex="0"><p>Recorded scene/video surface; source-time references only.</p></section></div>
<script type="application/json" id="review-editor-data">"""
        + payload
        + r"""</script>
<script type="module" src="./components/review_editor/review_editor.js"></script></body></html>
"""
    )


def _copy_web_component(output_dir: Path) -> Path:
    destination = output_dir / "components" / "review_editor" / "review_editor.js"
    _mkdir_no_follow(destination.parent, exist_ok=True)
    asset = resources.files("robot_sf.render.web_assets").joinpath(
        "components", "review_editor", "review_editor.js"
    )
    _write_atomic(destination, asset.read_bytes())
    return destination


def descriptor_document() -> dict[str, Any]:
    """Return the shared component descriptor."""

    return json.loads(json.dumps(_DESCRIPTOR_DOCUMENT, allow_nan=False))


def descriptor() -> dict[str, Any]:
    return descriptor_document()


def result_document(result: ComponentResult) -> dict[str, Any]:
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
    artifacts: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[Mapping[str, Any]] = (),
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
    base_provenance = {
        "component_version": COMPONENT_VERSION,
        "evidence_status": "diagnostic_only",
        "source_mutation": "none",
        "service_boundary": "ba05-service-compatible",
    }
    if provenance:
        base_provenance.update(dict(provenance))
    return ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=status,
        artifacts=tuple(dict(item) for item in artifacts),
        diagnostics=tuple(dict(item) for item in diagnostics),
        provenance=base_provenance,
        reason=reason,
    )


def _reserve_output(root: Path, value: str) -> Path:
    path = _resolve_under(root, value, kind="output")
    try:
        _mkdir_no_follow(path, exist_ok=False)
    except FileExistsError as error:
        raise ReviewEditorError(f"output_collision: {value}") from error
    return path


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Build the standalone offline editor model and browser module."""

    if not isinstance(request, ComponentRequest):
        return _result({}, STATUS_FAILED, reason="invalid_request: expected ComponentRequest")
    if request.component_id != COMPONENT_ID:
        return _result(
            request, STATUS_UNAVAILABLE, reason=f"unsupported_component: {request.component_id}"
        )
    supported = (
        set(DESCRIPTOR.required_capabilities)
        | set(DESCRIPTOR.optional_capabilities)
        | {"annotations", "storyboard", "source-bound-references", "audit-store"}
    )
    missing = sorted(set(request.required_capabilities) - supported)
    if missing:
        return _result(
            request,
            STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(missing)}",
        )
    if request.config.get("cancelled") is True:
        return _result(request, STATUS_CANCELLED, reason="cancelled: request cancelled before read")
    root = (base or Path.cwd()).resolve()
    output_dir: Path | None = None
    diagnostics: list[dict[str, Any]] = []
    emitted: list[dict[str, Any]] = []
    try:
        model = build_editor_model(request, base=root)
        diagnostics.extend(model.get("diagnostics", []))
        output_dir = _reserve_output(root, request.output_directory)
        model_digest = _write_json(output_dir / OUTPUT_MODEL_FILENAME, model)
        emitted.append(
            {
                "artifact_id": OUTPUT_MODEL_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_MODEL_FILENAME),
                "sha256": model_digest,
            }
        )
        html = _render_html(model)
        html_bytes = html.encode("utf-8")
        _write_atomic(output_dir / OUTPUT_HTML_FILENAME, html_bytes)
        emitted.append(
            {
                "artifact_id": OUTPUT_HTML_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_HTML_FILENAME),
                "sha256": hashlib.sha256(html_bytes).hexdigest(),
            }
        )
        component_path = _copy_web_component(output_dir)
        component_id = "components/review_editor/review_editor.js"
        emitted.append(
            {
                "artifact_id": component_id,
                "uri": str(Path(request.output_directory) / component_id),
                "sha256": hashlib.sha256(component_path.read_bytes()).hexdigest(),
            }
        )
        capability = {
            "schema_version": MISSING_CAPABILITY_SCHEMA_VERSION,
            "requested": list(request.required_capabilities),
            "available": list(DESCRIPTOR.required_capabilities),
            "missing": missing,
            "diagnostics": diagnostics,
        }
        capability_name = "missing-capability-report.json"
        capability_digest = _write_json(output_dir / capability_name, capability)
        emitted.append(
            {
                "artifact_id": capability_name,
                "uri": str(Path(request.output_directory) / capability_name),
                "sha256": capability_digest,
            }
        )
        status = STATUS_PARTIAL if diagnostics else STATUS_COMPLETE
        return _result(
            request,
            status,
            artifacts=emitted if status == STATUS_COMPLETE else (),
            diagnostics=diagnostics,
            provenance={
                "output_directory": request.output_directory,
                "emitted_artifacts": emitted,
                "model_schema": EDITOR_MODEL_SCHEMA_VERSION,
                "source_identity": model.get("source_identity", {}),
            },
        )
    except (
        ReviewEditorError,
        ReviewContractsValidationError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        if output_dir is not None and output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        return _result(request, STATUS_FAILED, reason=str(error)[:512], diagnostics=diagnostics)


def _strict_cli_json(path: str) -> Any:
    try:
        return _strict_loads(_read_regular_path(Path(path), kind="config"))
    except ReviewEditorError:
        raise
    except OSError as error:
        raise ReviewEditorError(f"cannot read request: {error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline Robot SF review editor.")
    parser.add_argument("--input", default=None, help="component-request.v1 JSON")
    parser.add_argument(
        "--config", default=None, help="optional JSON config merged into request config"
    )
    parser.add_argument("--output", default=None, help="new output directory relative to --base")
    parser.add_argument("--base", default=None, help="base directory for sources/output")
    parser.add_argument("--descriptor", action="store_true", help="print component descriptor")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.descriptor:
        print(json.dumps(descriptor_document(), sort_keys=True, indent=2))
        return 0
    if args.input is None or args.output is None:
        print(
            json.dumps(
                {
                    "status": STATUS_FAILED,
                    "reason": "invalid_input: --input and --output are required",
                }
            )
        )
        return 1
    payload: Any = None
    try:
        payload = _strict_cli_json(args.input)
        if not isinstance(payload, Mapping):
            raise ReviewEditorError("request must be an object")
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            raise ReviewEditorError("request config must be an object")
        if args.config:
            extra = _strict_cli_json(args.config)
            if not isinstance(extra, Mapping):
                raise ReviewEditorError("config must be an object")
            config = {**dict(config), **dict(extra)}
        request = component_request_from_dict(
            {**dict(payload), "config": dict(config), "output_directory": args.output},
            source=args.input,
        )
        result = run(request, base=Path(args.base) if args.base else None)
        print(json.dumps(result_document(result), sort_keys=True, indent=2))
        return (
            0
            if result.status == STATUS_COMPLETE
            else (
                2 if result.status in {STATUS_PARTIAL, STATUS_UNAVAILABLE, STATUS_CANCELLED} else 1
            )
        )
    except (
        ReviewContractsValidationError,
        ReviewEditorError,
        OSError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        result = _result(
            {
                "request_id": str(payload.get("request_id", "unknown"))
                if isinstance(payload, Mapping)
                else "unknown",
                "component_id": str(payload.get("component_id", COMPONENT_ID))
                if isinstance(payload, Mapping)
                else COMPONENT_ID,
            },
            STATUS_FAILED,
            reason=f"invalid_input: {error}",
        )
        print(json.dumps(result_document(result), sort_keys=True, indent=2))
        return 1


__all__ = [
    "ANNOTATION_CLASSIFICATIONS",
    "ANNOTATION_SPEEDS",
    "AUTOSAVE_STATES",
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "DESCRIPTOR",
    "EDITOR_MODEL_SCHEMA_VERSION",
    "MISSING_CAPABILITY_SCHEMA_VERSION",
    "STORYBOARD_SCHEMA_VERSION",
    "AuditPersistenceAdapter",
    "AuditStoreAdapter",
    "AutosaveStatus",
    "InvalidIntervalError",
    "OverlayState",
    "PersistenceAdapter",
    "Reference",
    "ReviewEditor",
    "ReviewEditorError",
    "ReviewEditorSession",
    "SaveConflict",
    "ServicePersistenceAdapter",
    "SourceStaleError",
    "StaleSelectionError",
    "StoryboardEditor",
    "StoryboardState",
    "build_editor_model",
    "build_overlay_commands",
    "coverage_summary",
    "create_reference",
    "descriptor",
    "descriptor_document",
    "main",
    "make_full_annotation",
    "make_full_episode_review",
    "make_one_click_annotation",
    "make_quick_annotation",
    "one_click_annotation",
    "quick_annotation",
    "result_document",
    "run",
    "snap_reference",
    "snap_to",
    "structured_annotation",
    "validate_interval",
]


if __name__ == "__main__":
    raise SystemExit(main())
