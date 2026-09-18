"""Offline campaign accounting and detector orchestration for BA-01.

``scan_campaign`` accepts a current RobotSF campaign-result document (or a
JSONL episode source), indexes every expected row, and runs the registered
detectors against readable rows.  It never invokes a simulator, shell,
network, renderer, or diagnostic executor.  A report is useful even when
large parts of the source are missing: coverage, detector evaluability,
review, and diagnostic counters are intentionally separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from robot_sf.analysis_workbench.audit_contracts import (
    AuditContractError,
    CampaignAudit,
    EpisodeRef,
    Signal,
    canonical_json,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_detectors import (
    DEFAULT_REGISTRY,
    DETECTOR_ENGINE_VERSION,
    DetectorRegistry,
    DetectorSpec,
    default_registry,
    detect,
    normalize_detector_ids,
    signal_status_counts,
    unavailable_signal,
)
from robot_sf.analysis_workbench.review_context import (
    _assert_output_directory_current,
    _assert_output_parent_current,
    _atomic_materialize_no_replace,
    _open_output_parent_guard,
    _open_output_temporary,
    _publish_output,
    _read_descriptor_backed_file,
    _read_regular_file_no_follow,
    _release_empty_output,
    _reserve_output_directory,
    _snapshot_directory,
)
from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_request_from_dict,
)

if TYPE_CHECKING:
    from robot_sf.analysis_workbench.audit_store import AuditStore

COMPONENT_ID = "ba01-audit-scan"
COMPONENT_VERSION = "1.0.0"
AUDIT_SCAN_SCHEMA_VERSION = "benchmark-audit.v1"
AUDIT_REPORT_FILENAME = "audit-report.json"
AUDIT_REGISTRY_FILENAME = "audit-detector-registry.json"
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
INVENTORY_STATUSES = ("readable", "missing", "duplicate", "invalid", "unsupported")
EXECUTION_UNSUPPORTED = frozenset(
    {
        "fallback",
        "degraded",
        "unavailable",
        "not_available",
        "failed",
        "failure",
        "error",
        "unsupported",
        "partial",
        "partial_failure",
        "truncated",
        "diagnostic_only",
        "diagnostic_stub",
        "skipped",
        "cancelled",
    }
)
_STATUS_SURFACE_FIELDS = (
    "row_status",
    "status",
    "availability_status",
    "readiness_status",
    "campaign_execution_status",
    "execution_status",
    "execution_mode",
    "preflight_status",
)
_FALLBACK_COUNTER_FIELDS = (
    "fallback_count",
    "fallback_steps",
    "fallback_actions",
    "fallback_events",
    "fallback_invocations",
    "fallback_used",
    "fallback_counter",
    "fallback_counters",
)
_DESCRIPTIVE_STATUS_FIELDS = frozenset({"status", "execution_status"})
_DESCRIPTIVE_STATUS_VALUES = frozenset(
    {
        "success",
        "completed",
        "collision",
        "timeout",
        "timed_out",
        "horizon",
        "horizon_reached",
        "time_limit",
        "max_steps",
    }
)
SUPPORTED_SOURCE_SCHEMAS = frozenset(
    {"campaign-result.v1", "campaign-result-store.v2", "episode-jsonl.v1", ""}
)
SUPPORTED_SOURCE_FORMATS = frozenset(
    {"campaign-result", "campaign-result-store", "episode-jsonl", "json", "jsonl", "ndjson"}
)
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_EPISODES = 100_000
MAX_CONFIG_BYTES = 256 * 1024
MAX_SOURCE_FILES = 4096
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_IDENTITY_DIGEST_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class AuditScanError(ValueError):
    """Raised when a campaign cannot be safely inspected."""


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditScanError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise AuditScanError(f"{name} must be finite")
    return result


def _strict_walk(value: Any, *, path: str = "$", max_depth: int = 64) -> None:  # noqa: C901
    """Reject non-JSON values before they influence a digest or detector."""

    pending: list[tuple[Any, str, int]] = [(value, path, 0)]
    seen: set[int] = set()
    while pending:
        current, current_path, depth = pending.pop()
        if depth > max_depth:
            raise AuditScanError(f"{current_path} exceeds maximum nesting depth")
        if isinstance(current, float) and not math.isfinite(current):
            raise AuditScanError(f"{current_path} contains a non-finite number")
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                raise AuditScanError(f"{current_path} contains a recursive reference")
            seen.add(marker)
            for key, item in current.items():
                if not isinstance(key, str):
                    raise AuditScanError(f"{current_path} contains a non-string field name")
                pending.append((item, f"{current_path}.{key}", depth + 1))
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in seen:
                raise AuditScanError(f"{current_path} contains a recursive reference")
            seen.add(marker)
            for index, item in enumerate(current):
                pending.append((item, f"{current_path}[{index}]", depth + 1))
        elif current is not None and not isinstance(current, (str, int, float, bool)):
            raise AuditScanError(f"{current_path} contains unsupported {type(current).__name__}")


def _contains_nonfinite(value: Any) -> bool:
    """Return whether a row contains a non-finite numeric value."""

    pending = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            return True
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            pending.extend(current)
    return False


def _status_token(value: Any, *, path: str) -> tuple[str | None, str | None]:
    """Normalize one canonical status while retaining malformed input.

    Returns:
        A normalized status token and an error message when malformed.
    """

    if not isinstance(value, str) or not value.strip():
        return None, f"{path} is malformed"
    return re.sub(r"[-_\s]+", "_", value.strip().lower()), None


def _counter_status(  # noqa: C901
    value: Any, *, path: str
) -> tuple[str | None, str | None]:
    """Classify a fallback counter without truthiness coercion.

    Returns:
        An unsupported/invalid state and a reason, or ``(None, None)``.
    """

    if isinstance(value, bool):
        return ("unsupported", f"{path} indicates fallback") if value else (None, None)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or value < 0:
            return None, f"{path} is malformed"
        return ("unsupported", f"{path} is non-zero") if value > 0 else (None, None)
    if isinstance(value, Mapping):
        found: list[tuple[str | None, str | None]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                return None, f"{path} contains a non-string field name"
            found.append(_counter_status(item, path=f"{path}.{key}"))
        for state, reason in found:
            if reason is not None:
                return state, reason
        return None, None
    if isinstance(value, list):
        if value:
            return "unsupported", f"{path} contains fallback entries"
        return None, None
    return None, f"{path} is malformed"


def _execution_status(  # noqa: C901, PLR0912
    value: Mapping[str, Any], *, path: str = "row"
) -> tuple[str, str]:
    """Inspect every canonical execution/readiness status surface.

    The result is ``(state, reason)`` where state is ``native``,
    ``unsupported`` or ``invalid``.  A blocked nested surface can never be
    hidden by an outer ``row_status=native`` declaration.

    Returns:
        A state/reason pair for the admitted execution metadata.
    """

    surfaces: list[tuple[str, Any]] = []
    for key in _STATUS_SURFACE_FIELDS:
        if key in value:
            surfaces.append((f"{path}.{key}", value[key]))
    algorithm_metadata = value.get("algorithm_metadata")
    if algorithm_metadata is not None:
        if not isinstance(algorithm_metadata, Mapping):
            return "invalid", f"{path}.algorithm_metadata is malformed"
        for key in _STATUS_SURFACE_FIELDS:
            if key in algorithm_metadata:
                surfaces.append((f"{path}.algorithm_metadata.{key}", algorithm_metadata[key]))
        analysis_trace = algorithm_metadata.get("analysis_trace")
        if analysis_trace is not None:
            if not isinstance(analysis_trace, Mapping):
                return "invalid", f"{path}.algorithm_metadata.analysis_trace is malformed"
            for key in _STATUS_SURFACE_FIELDS:
                if key in analysis_trace:
                    surfaces.append(
                        (f"{path}.algorithm_metadata.analysis_trace.{key}", analysis_trace[key])
                    )
    for container_name in ("metrics", "operational_metrics"):
        container = value.get(container_name)
        if container is None:
            continue
        if not isinstance(container, Mapping):
            return "invalid", f"{path}.{container_name} is malformed"
        for key in _STATUS_SURFACE_FIELDS:
            if key in container:
                surfaces.append((f"{path}.{container_name}.{key}", container[key]))
    top_level_trace = value.get("analysis_trace")
    if top_level_trace is not None:
        if not isinstance(top_level_trace, Mapping):
            return "invalid", f"{path}.analysis_trace is malformed"
        for key in _STATUS_SURFACE_FIELDS:
            if key in top_level_trace:
                surfaces.append((f"{path}.analysis_trace.{key}", top_level_trace[key]))
    state = "native"
    state_reason = ""
    for surface_path, raw in surfaces:
        token, error = _status_token(raw, path=surface_path)
        if error is not None:
            return "invalid", error
        if token in EXECUTION_UNSUPPORTED:
            return "unsupported", f"non-admissible execution status: {token}"
        if token in {
            "native",
            "adapter",
            "available",
            "ready",
            "complete",
            "success",
            "collision",
            "ok",
            "passed",
            "pass",
            "admitted",
            "included",
            "running",
        }:
            continue
        # Canonical v2 reserves row_status for execution mode.  Its
        # status/execution_status projection may describe the terminal outcome
        # (for example ``collision`` or ``max_steps``) without changing the
        # row's execution admissibility.
        if (
            surface_path.rsplit(".", 1)[-1] in _DESCRIPTIVE_STATUS_FIELDS
            and token in _DESCRIPTIVE_STATUS_VALUES
        ):
            continue
        # Unknown status labels are retained for diagnosis but are not allowed
        # to advertise a native row.  This avoids silently trusting a future
        # status vocabulary before BA-01 has been updated.
        if surface_path.endswith(".row_status") or surface_path.endswith(
            (".availability_status", ".readiness_status", ".campaign_execution_status")
        ):
            return "invalid", f"{surface_path} is an unknown status: {token}"
        state = "unsupported"
        state_reason = f"{surface_path} has an unrecognized status: {token}"
    containers: list[tuple[str, Mapping[str, Any]]] = [(path, value)]
    for container_name in ("metrics", "operational_metrics"):
        container_value = value.get(container_name)
        if container_value is not None:
            if not isinstance(container_value, Mapping):
                return "invalid", f"{path}.{container_name} is malformed"
            containers.append((f"{path}.{container_name}", container_value))
    if isinstance(algorithm_metadata, Mapping):
        containers.append((f"{path}.algorithm_metadata", algorithm_metadata))
        analysis_trace = algorithm_metadata.get("analysis_trace")
        if isinstance(analysis_trace, Mapping):
            containers.append((f"{path}.algorithm_metadata.analysis_trace", analysis_trace))
    if isinstance(top_level_trace, Mapping):
        containers.append((f"{path}.analysis_trace", top_level_trace))
    for container_path, container in containers:
        for key in _FALLBACK_COUNTER_FIELDS:
            if key not in container:
                continue
            counter_state, counter_reason = _counter_status(
                container[key], path=f"{container_path}.{key}"
            )
            if counter_reason is not None:
                if counter_state == "unsupported":
                    return "unsupported", counter_reason
                return "invalid", counter_reason
            if counter_state == "unsupported":
                return "unsupported", counter_reason or f"{container_path}.{key} indicates fallback"
    return state, state_reason


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(value: str | Path, *, root: Path) -> Path:  # noqa: C901
    """Resolve one local path under ``root`` without following symlink escapes.

    Returns:
        A resolved regular file or campaign-store directory.
    """

    raw = str(value)
    split = urlsplit(raw)
    if (
        split.scheme
        or split.query
        or split.fragment
        or "\\" in raw
        or "\x00" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise AuditScanError("source URI must be a local path without a scheme")
    path = Path(raw)
    if ".." in path.parts:
        raise AuditScanError("source path traversal is rejected")
    try:
        root_resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuditScanError("admitted root is unavailable") from exc
    if not root_resolved.is_dir():
        raise AuditScanError("admitted root must be a directory")
    if path.is_absolute():
        candidate = path
    else:
        candidate = root_resolved / path
    try:
        lexical_relative = candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise AuditScanError("source path escapes admitted root") from exc
    current = root_resolved
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise AuditScanError("source path contains a symlink component")
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AuditScanError("source path escapes admitted root") from exc
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise AuditScanError("source is unavailable") from exc
    if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
        raise AuditScanError("source is missing or is not a regular file/directory")
    if stat.S_ISREG(mode) and resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise AuditScanError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    return resolved


def _strict_json(raw: bytes, *, label: str, validate: bool = True) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditScanError(f"{label} contains duplicate field {key!r}")
            result[key] = value
        return result

    def reject(token: str) -> Any:
        if not validate:
            return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[token]
        raise AuditScanError(f"{label} contains non-finite JSON constant {token}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AuditScanError, RecursionError) as exc:
        raise AuditScanError(f"{label} is not strict JSON") from exc
    if validate:
        _strict_walk(payload, path=label)
    return payload


def _jsonl_rows(raw: bytes, *, label: str) -> list[tuple[Mapping[str, Any] | None, int]]:
    """Parse bounded JSONL while retaining malformed rows for accounting.

    A campaign scan must distinguish a corrupt row from an absent row.  The
    line number is therefore retained even when decoding fails; row-level
    non-finite values are parsed in a permissive pass and classified by
    ``_row_validation`` rather than discarded at this boundary.

    Returns:
        One row entry per non-blank source line.
    """

    rows: list[tuple[Mapping[str, Any] | None, int]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = _strict_json(line, label=f"{label} line {line_number}", validate=False)
        except AuditScanError:
            rows.append((None, line_number))
            continue
        rows.append((value if isinstance(value, Mapping) else None, line_number))
    return rows


def _source_ref_from_value(  # noqa: C901, PLR0912
    source: SourceRef | Mapping[str, Any] | None,
    *,
    source_path: Path | None,
    observed_digest: str,
    payload: Mapping[str, Any] | None = None,
) -> SourceRef:
    """Build a BA-03 source pointer and enforce an advertised hash binding.

    Returns:
        The source identity bound to the observed bytes.
    """

    if isinstance(source, SourceRef):
        ref = source
        if any(
            not isinstance(getattr(ref, key), str)
            for key in (
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
            raise AuditScanError("source reference fields must be strings")
    elif isinstance(source, Mapping):
        known = {
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
        unknown = set(source) - known
        if unknown:
            raise AuditScanError(
                f"source reference contains unknown fields: {', '.join(sorted(unknown))}"
            )
        missing = {"artifact_id", "uri", "format"} - set(source)
        if missing:
            raise AuditScanError(
                f"source reference is missing fields: {', '.join(sorted(missing))}"
            )
        invalid = [key for key in known if key in source and not isinstance(source[key], str)]
        if invalid:
            raise AuditScanError(
                "source reference fields must be strings: " + ", ".join(sorted(invalid))
            )
        ref = SourceRef(**{key: source.get(key, "") for key in known})
    else:
        payload_schema = _payload_schema(payload) if isinstance(payload, Mapping) else ""
        payload_commit = (
            _identity_aliases(
                payload,
                names=("source_commit", "git_hash", "commit_sha", "commit"),
                label="source_commit",
                include_episode_rows=True,
            )
            if isinstance(payload, Mapping)
            else None
        )
        payload_config_identity = (
            _identity_aliases(
                payload,
                names=("config_identity", "config_hash"),
                label="config_identity",
                include_episode_rows=True,
            )
            if isinstance(payload, Mapping)
            else None
        )
        source_format = "campaign-result"
        if source_path is not None:
            if source_path.is_dir():
                source_format = "campaign-result-store"
            elif source_path.suffix.lower() in {".jsonl", ".ndjson"}:
                source_format = "episode-jsonl"
            elif source_path.suffix.lower() not in {".json", ""}:
                source_format = "unsupported"
        if not payload_schema:
            payload_schema = (
                "campaign-result-store.v2"
                if source_format == "campaign-result-store"
                else "episode-jsonl.v1"
                if source_format == "episode-jsonl"
                else "campaign-result.v1"
            )
        ref = SourceRef(
            artifact_id=source_path.name if source_path is not None else "in-memory-campaign",
            uri=source_path.name if source_path is not None else "in-memory-campaign",
            format=source_format,
            schema=str(payload_schema) if isinstance(payload_schema, str) else "",
            sha256=observed_digest,
            source_commit=payload_commit or "",
            config_identity=payload_config_identity or "",
        )
    payload_schema = _payload_schema(payload) if isinstance(payload, Mapping) else ""
    payload_commit = (
        _identity_aliases(
            payload,
            names=("source_commit", "git_hash", "commit_sha", "commit"),
            label="source_commit",
            include_episode_rows=True,
        )
        if isinstance(payload, Mapping)
        else None
    )
    payload_config_identity = (
        _identity_aliases(
            payload,
            names=("config_identity", "config_hash"),
            label="config_identity",
            include_episode_rows=True,
        )
        if isinstance(payload, Mapping)
        else None
    )
    if not ref.uri.strip() or not ref.format.strip():
        raise AuditScanError("source reference URI and format are required")
    if not ref.schema.strip():
        raise AuditScanError("source reference schema is required")
    if any(
        isinstance(getattr(ref, key), str) and getattr(ref, key) and not getattr(ref, key).strip()
        for key in ("source_commit", "config_identity")
    ):
        raise AuditScanError("source reference identity is malformed")
    if payload_schema and ref.schema.strip() != payload_schema:
        # The caller still receives an unsupported, fully accounted source;
        # rows must never be treated as native under a conflicting schema.
        ref = SourceRef(
            artifact_id=ref.artifact_id,
            uri=ref.uri,
            format=ref.format,
            schema=ref.schema.strip(),
            sha256=ref.sha256,
            source_commit=ref.source_commit,
            config_identity=ref.config_identity,
            units=ref.units,
            coordinate_frame=ref.coordinate_frame,
        )
    if ref.source_commit and payload_commit and ref.source_commit.strip() != payload_commit:
        raise AuditScanError("source reference source_commit conflicts with payload")
    if (
        ref.config_identity
        and payload_config_identity
        and ref.config_identity.strip() != payload_config_identity
    ):
        raise AuditScanError("source reference config_identity conflicts with payload")
    declared = ref.sha256.strip()
    if (
        not ref.artifact_id.strip()
        or ref.artifact_id in {".", ".."}
        or "/" in ref.artifact_id
        or "\\" in ref.artifact_id
        or "\x00" in ref.artifact_id
        or any(ord(character) < 32 or ord(character) == 127 for character in ref.artifact_id)
        or PureWindowsPath(ref.artifact_id).is_absolute()
        or bool(PureWindowsPath(ref.artifact_id).drive)
    ):
        raise AuditScanError("source artifact_id is unsafe")
    uri = urlsplit(ref.uri)
    if (
        uri.scheme
        or uri.query
        or uri.fragment
        or "\\" in ref.uri
        or "\x00" in ref.uri
        or any(ord(character) < 32 or ord(character) == 127 for character in ref.uri)
        or ".." in Path(ref.uri).parts
        or Path(ref.uri).is_absolute()
    ):
        raise AuditScanError("source reference URI must be an admitted relative path")
    if declared and (
        not _DIGEST_RE.fullmatch(declared) or declared.lower() != observed_digest.lower()
    ):
        raise AuditScanError("source digest does not match observed bytes")
    return SourceRef(
        artifact_id=ref.artifact_id,
        uri=ref.uri,
        format=ref.format,
        schema=ref.schema.strip(),
        sha256=declared.lower() if declared else observed_digest,
        source_commit=ref.source_commit.strip() or payload_commit or "",
        config_identity=ref.config_identity.strip() or payload_config_identity or "",
        units=ref.units,
        coordinate_frame=ref.coordinate_frame,
    )


def _regular_directory_files(source_path: Path) -> list[Path]:
    """Enumerate a source directory and reject every non-regular entry.

    Returns:
        Sorted regular files under ``source_path``.
    """

    try:
        entries = sorted(source_path.rglob("*"))
    except OSError as exc:
        raise AuditScanError("source directory cannot be enumerated") from exc
    files: list[Path] = []
    for item in entries:
        if item.is_symlink():
            raise AuditScanError("source directory contains a symlink")
        try:
            mode = item.stat().st_mode
        except OSError as exc:
            raise AuditScanError("source directory contains an unreadable entry") from exc
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise AuditScanError("source directory contains a special file")
        files.append(item)
    return files


def _read_admitted_file(path: Path, *, label: str) -> bytes:
    """Read a regular source file through the shared no-follow admission helper.

    Returns:
        The bounded file bytes.
    """

    try:
        return _read_regular_file_no_follow(path, limit=MAX_SOURCE_BYTES, kind=label)
    except (OSError, ReviewContractsValidationError) as exc:
        raise AuditScanError(f"{label} is not a readable regular file") from exc


def _canonical_store_rows(store_path: Path) -> tuple[list[tuple[Mapping[str, Any], int]], str]:
    """Load a v2 store through the canonical case-workbench owner.

    Returns:
        Canonical rows with stable positions and the source execution state.
    """

    try:
        from robot_sf.benchmark.case_workbench import (  # noqa: PLC0415
            _load_records,
            _v2_integrity_errors,
        )

        integrity_errors = _v2_integrity_errors(store_path)
        if integrity_errors:
            return [], "unsupported"
        records = _load_records(store_path)
    except Exception:  # noqa: BLE001
        # A malformed or dependency-unavailable table is retained as an
        # unsupported source, with expected IDs still accounted from its
        # manifest.  No fallback reader is permitted here.
        return [], "unsupported"
    return [(dict(record), index) for index, record in enumerate(records, start=1)], "native"


def _payload_schema(payload: Mapping[str, Any]) -> str:
    """Return one non-conflicting schema declaration from a payload."""

    values: list[tuple[str, Any]] = []
    for key in ("schema_version", "schema"):
        if key in payload:
            values.append((key, payload[key]))
    nonempty: list[tuple[str, str]] = []
    for key, value in values:
        if not isinstance(value, str) or not value.strip():
            raise AuditScanError(f"campaign {key} is malformed")
        nonempty.append((key, value.strip()))
    if len({value for _key, value in nonempty}) > 1:
        raise AuditScanError("campaign schema declarations conflict")
    return nonempty[0][1] if nonempty else ""


def _identity_aliases(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    names: Sequence[str],
    label: str,
    include_episode_rows: bool = False,
) -> str | None:
    """Collect one identity value and reject conflicting aliases.

    Returns:
        One normalized identity, or ``None`` when no declaration is present.
    """

    values: list[tuple[str, str]] = []

    def collect(container: Any, prefix: str) -> None:
        if not isinstance(container, Mapping):
            return
        for name in names:
            if name not in container or container[name] in (None, ""):
                continue
            value = container[name]
            if not isinstance(value, str) or not value.strip():
                raise AuditScanError(f"{prefix}.{name} is malformed")
            values.append((f"{prefix}.{name}", value.strip()))

    def collect_record(record: Mapping[str, Any], prefix: str) -> None:
        collect(record, prefix)
        for nested_name in ("provenance", "result_provenance", "cell_context", "config"):
            collect(record.get(nested_name), f"{prefix}.{nested_name}")
        metadata = record.get("algorithm_metadata")
        if isinstance(metadata, Mapping):
            collect(metadata, f"{prefix}.algorithm_metadata")
            collect(metadata.get("analysis_trace"), f"{prefix}.algorithm_metadata.analysis_trace")

    collect_record(payload, label)
    if include_episode_rows:
        entries = payload.get("episodes")
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                if isinstance(entry, Mapping):
                    collect_record(entry, f"{label}.episodes[{index}]")
    unique = {value for _name, value in values}
    if len(unique) > 1:
        names_text = ", ".join(name for name, _value in values)
        raise AuditScanError(f"conflicting {label} aliases: {names_text}")
    return next(iter(unique)) if unique else None


def _row_source_digest_error(row: Mapping[str, Any], source_ref: SourceRef) -> str | None:
    """Validate any row-level source digest declaration against admitted bytes.

    Returns:
        A bounded binding error, or ``None`` when the row has no conflict.
    """

    source_digest_values: list[str] = []
    source_digest_containers: list[tuple[str, Any]] = [("episode", row)]
    for name in ("provenance", "result_provenance", "cell_context", "config"):
        source_digest_containers.append((f"episode.{name}", row.get(name)))
    metadata = row.get("algorithm_metadata")
    source_digest_containers.append(("episode.algorithm_metadata", metadata))
    if isinstance(metadata, Mapping):
        source_digest_containers.append(
            ("episode.algorithm_metadata.analysis_trace", metadata.get("analysis_trace"))
        )
    for path, container in source_digest_containers:
        if not isinstance(container, Mapping) or "source_digest" not in container:
            continue
        value = container["source_digest"]
        if not isinstance(value, str) or not value.strip():
            return f"{path}.source_digest is malformed"
        source_digest_values.append(value.strip())
    if len(set(source_digest_values)) > 1:
        return "episode source_digest aliases conflict"
    if source_digest_values:
        value = source_digest_values[0]
        if _DIGEST_RE.fullmatch(value) is None:
            return "episode source_digest is not a hexadecimal digest"
        if value.lower() != source_ref.sha256.lower():
            return "episode source_digest conflicts with admitted source"
    return None


def _row_binding_error(
    row: Mapping[str, Any],
    source_ref: SourceRef,
    *,
    expected_campaign_id: str | None = None,
    expected_config_identity: str | None = None,
) -> str | None:
    """Check row provenance aliases against the admitted source identity.

    Returns:
        A conflict description, or ``None`` when the row is bound.
    """

    digest_error = _row_source_digest_error(row, source_ref)
    if digest_error is not None:
        return digest_error

    try:
        row_campaign = _identity_aliases(
            row,
            names=("campaign_id", "study_id", "campaign"),
            label="episode.campaign_id",
        )
        row_commit = _identity_aliases(
            row,
            names=("source_commit", "git_hash", "commit_sha", "commit"),
            label="episode.source_commit",
        )
        row_config = _identity_aliases(
            row,
            names=("config_identity", "config_hash"),
            label="episode.config_identity",
        )
    except AuditScanError as exc:
        return str(exc)
    if expected_campaign_id and row_campaign and expected_campaign_id.strip() != row_campaign:
        return "episode campaign_id conflicts with campaign"
    if source_ref.source_commit and row_commit and source_ref.source_commit != row_commit:
        return "episode source_commit conflicts with admitted source"
    if source_ref.config_identity and row_config and source_ref.config_identity != row_config:
        return "episode config_identity conflicts with admitted source"
    if expected_config_identity and row_config and expected_config_identity != row_config:
        return "episode config_identity conflicts with requested config"
    try:
        _identity_aliases(
            row,
            names=("execution_id", "run_id"),
            label="episode.execution_id",
        )
        _identity_aliases(
            row,
            names=("checkpoint_digest", "checkpoint_hash"),
            label="episode.checkpoint_digest",
        )
        _identity_aliases(
            row,
            names=("environment_digest", "environment_hash"),
            label="episode.environment_digest",
        )
        _identity_aliases(
            row,
            names=("config_digest",),
            label="episode.config_digest",
        )
    except AuditScanError as exc:
        return str(exc)
    return None


@dataclass(frozen=True, slots=True)
class EpisodeInventory:
    """One expected or observed input row and its accounting status."""

    episode_id: str
    status: str
    expected: bool = True
    source_artifact_id: str = ""
    line_number: int | None = None
    reason: str = ""
    row: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate status, identity and strict row data."""
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise AuditScanError("inventory episode_id must be non-empty")
        if self.status not in INVENTORY_STATUSES:
            raise AuditScanError(f"inventory status must be one of {INVENTORY_STATUSES}")
        if not isinstance(self.expected, bool):
            raise AuditScanError("inventory expected must be boolean")
        if self.line_number is not None and (
            not isinstance(self.line_number, int) or self.line_number < 1
        ):
            raise AuditScanError("inventory line_number must be positive")
        if self.row is not None:
            if not isinstance(self.row, Mapping):
                raise AuditScanError("inventory row must be a mapping")
            _strict_walk(self.row, path="inventory.row")

    @property
    def readable(self) -> bool:
        """Return whether this entry has a detector-readable row."""

        return self.status == "readable" and self.row is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable inventory entry.

        Returns:
            A strict JSON-compatible mapping.
        """

        result: dict[str, Any] = {
            "episode_id": self.episode_id,
            "status": self.status,
            "expected": self.expected,
            "source_artifact_id": self.source_artifact_id,
            "line_number": self.line_number,
            "reason": self.reason,
        }
        if self.row is not None:
            result["row"] = dict(self.row)
        return result


@dataclass(frozen=True, slots=True)
class AuditScanReport:
    """Portable BA-01 scan result with strict source and denominator metadata."""

    audit: CampaignAudit
    inventory: tuple[EpisodeInventory, ...]
    signals: tuple[Signal, ...]
    detector_registry: DetectorRegistry
    counts: Mapping[str, Any]
    provenance: Mapping[str, Any]
    enrichment_requests: tuple[Mapping[str, Any], ...] = ()
    cache_key: str = ""
    episode_refs: tuple[EpisodeRef, ...] = ()
    schema_version: str = AUDIT_SCAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate report identity and nested strict JSON values."""
        if self.schema_version != AUDIT_SCAN_SCHEMA_VERSION:
            raise AuditScanError(f"unsupported audit scan schema: {self.schema_version}")
        if not isinstance(self.audit, CampaignAudit):
            raise AuditScanError("audit must be a CampaignAudit")
        object.__setattr__(self, "inventory", tuple(self.inventory))
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "episode_refs", tuple(self.episode_refs))
        if any(not isinstance(item, EpisodeInventory) for item in self.inventory):
            raise AuditScanError("inventory must contain EpisodeInventory values")
        if any(not isinstance(item, Signal) for item in self.signals):
            raise AuditScanError("signals must contain Signal values")
        if not isinstance(self.detector_registry, DetectorRegistry):
            raise AuditScanError("detector_registry must be a DetectorRegistry")
        if any(not isinstance(item, EpisodeRef) for item in self.episode_refs):
            raise AuditScanError("episode_refs must contain EpisodeRef values")
        if not isinstance(self.counts, Mapping) or not isinstance(self.provenance, Mapping):
            raise AuditScanError("counts and provenance must be mappings")
        if any(not isinstance(item, Mapping) for item in self.enrichment_requests):
            raise AuditScanError("enrichment_requests must contain mappings")
        _strict_walk(self.counts, path="counts")
        _strict_walk(self.provenance, path="provenance")
        _strict_walk(self.enrichment_requests, path="enrichment_requests")
        if not isinstance(self.cache_key, str) or not _DIGEST_RE.fullmatch(self.cache_key):
            raise AuditScanError("cache_key must be a SHA-256 digest")

    @property
    def report_digest(self) -> str:
        """Return the deterministic logical report digest."""

        return _sha256(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report using BA-03 record serialization.

        Returns:
            A strict JSON-compatible report mapping.
        """

        return {
            "schema_version": self.schema_version,
            "audit": record_to_dict(self.audit),
            "inventory": [item.to_dict() for item in self.inventory],
            "episode_refs": [record_to_dict(item) for item in self.episode_refs],
            "signals": [record_to_dict(item) for item in self.signals],
            "detector_registry": self.detector_registry.to_dict(),
            "counts": dict(self.counts),
            "provenance": dict(self.provenance),
            "enrichment_requests": [dict(item) for item in self.enrichment_requests],
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True, slots=True)
class _LoadedCampaign:
    payload: dict[str, Any]
    rows: tuple[tuple[Mapping[str, Any] | None, int], ...]
    source_ref: SourceRef
    source_digest: str
    source_bytes: bytes
    source_status: str
    source_error: str = ""


def _load_source(  # noqa: C901, PLR0912, PLR0915
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    root: str | Path | None = None,
    source_ref: SourceRef | Mapping[str, Any] | None = None,
) -> _LoadedCampaign:
    """Load one bounded local/in-memory campaign without executing anything.

    Returns:
        Parsed rows and the source identity used by the scan.
    """

    source_path: Path | None = None
    source_bytes: bytes
    payload: dict[str, Any]
    source_status = "native"
    if isinstance(source, (str, Path)):
        root_path = Path(root) if root is not None else Path.cwd()
        try:
            source_path = _safe_path(source, root=root_path)
            if source_path.is_dir():
                files = _regular_directory_files(source_path)
                if len(files) > MAX_SOURCE_FILES:
                    raise AuditScanError(f"source directory exceeds {MAX_SOURCE_FILES} files")
                total = sum(item.stat().st_size for item in files)
                if total > MAX_SOURCE_BYTES:
                    raise AuditScanError(f"source directory exceeds {MAX_SOURCE_BYTES} bytes")
                # Bind the digest, manifest, owner loader, and rows to one
                # descriptor-backed snapshot.  Reopening ``source_path``
                # after integrity validation would allow a replaced parquet
                # file to be admitted under the original digest.
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="ba01-source-snapshot-"
                    ) as snapshot_dir:
                        snapshot_root = Path(snapshot_dir)
                        snapshot_files, snapshot_digest = _snapshot_directory(
                            source_path, files, snapshot_root
                        )
                        source_bytes = b"".join(
                            str(item.relative_to(snapshot_root)).encode("utf-8")
                            + b"\0"
                            + _read_admitted_file(item, label=f"campaign source {item.name}")
                            for item in snapshot_files
                        )
                        if _sha256(source_bytes) != snapshot_digest:
                            raise AuditScanError(
                                "source snapshot digest computation is inconsistent"
                            )
                        manifest_path = snapshot_root / "manifest.json"
                        manifest: dict[str, Any] = {}
                        if manifest_path.is_file():
                            manifest_value = _strict_json(
                                _read_admitted_file(manifest_path, label="campaign manifest"),
                                label="campaign manifest",
                            )
                            if not isinstance(manifest_value, Mapping):
                                raise AuditScanError("campaign manifest must be an object")
                            manifest = dict(manifest_value)
                            _strict_walk(manifest, path="campaign manifest")
                        canonical_store = snapshot_root
                        nested_store = snapshot_root / "campaign-result-store.v2"
                        if (
                            not (canonical_store / "episodes.parquet").is_file()
                            and nested_store.is_dir()
                            and (nested_store / "episodes.parquet").is_file()
                        ):
                            canonical_store = nested_store
                            nested_manifest_path = canonical_store / "manifest.json"
                            if nested_manifest_path.is_file():
                                nested_manifest_value = _strict_json(
                                    _read_admitted_file(
                                        nested_manifest_path, label="campaign manifest"
                                    ),
                                    label="campaign manifest",
                                )
                                if not isinstance(nested_manifest_value, Mapping):
                                    raise AuditScanError("campaign manifest must be an object")
                                manifest = dict(nested_manifest_value)
                        canonical_rows: list[tuple[Mapping[str, Any], int]] = []
                        canonical_loaded = False
                        canonical_present = (canonical_store / "episodes.parquet").is_file()
                        if canonical_present:
                            if manifest.get("schema_version") == "campaign-result-store.v2":
                                canonical_loaded = True
                                canonical_rows, canonical_status = _canonical_store_rows(
                                    canonical_store
                                )
                                if canonical_status != "native":
                                    source_status = canonical_status
                            else:
                                source_status = "unsupported"
                        row_path = next(
                            (
                                snapshot_root / name
                                for name in ("episodes.jsonl", "records.jsonl")
                                if (snapshot_root / name).is_file()
                            ),
                            None,
                        )
                        rows = (
                            canonical_rows
                            if canonical_present
                            else _jsonl_rows(
                                _read_admitted_file(row_path, label="campaign source"),
                                label="source",
                            )
                            if row_path is not None
                            else []
                        )
                        payload = {
                            **manifest,
                            "schema_version": manifest.get(
                                "schema_version", "campaign-result-store.v2"
                            ),
                            "episodes": (
                                [dict(item) for item, _line in rows if isinstance(item, Mapping)]
                                if row_path is not None or canonical_present
                                else manifest.get("episodes", [])
                                if isinstance(manifest.get("episodes"), list)
                                else []
                            ),
                        }
                        if row_path is None and not canonical_loaded:
                            source_status = "unsupported"
                except ReviewContractsValidationError as exc:
                    raise AuditScanError("campaign source snapshot is invalid") from exc
            else:
                source_bytes = _read_admitted_file(source_path, label="campaign source")
                if source_path.suffix.lower() in {".jsonl", ".ndjson"}:
                    rows = _jsonl_rows(source_bytes, label="campaign source")
                    payload = {
                        "schema_version": "episode-jsonl.v1",
                        "episodes": [dict(item) for item, _line in rows if item is not None],
                    }
                elif source_path.suffix.lower() in {".json", ""}:
                    loaded = _strict_json(source_bytes, label="campaign source", validate=False)
                    if isinstance(loaded, Mapping):
                        payload = dict(loaded)
                    elif isinstance(loaded, list):
                        payload = {"schema_version": "episode-jsonl.v1", "episodes": loaded}
                    else:
                        raise AuditScanError("campaign source must be an object or row array")
                    rows = (
                        [
                            (item if isinstance(item, Mapping) else None, index)
                            for index, item in enumerate(payload.get("episodes", []), start=1)
                        ]
                        if isinstance(payload.get("episodes"), list)
                        else []
                    )
                else:
                    # Do not pretend that an unreadable binary/table format
                    # is an absent campaign.  Retain its bytes and identity
                    # so expected IDs can be accounted for as unsupported.
                    payload = {
                        "schema_version": f"unsupported{source_path.suffix.lower()}",
                        "episodes": [],
                    }
                    rows = []
                    source_status = "unsupported"
        except (OSError, UnicodeError) as exc:
            raise AuditScanError(f"source unavailable: {type(exc).__name__}") from exc
    elif isinstance(source, Mapping):
        payload = dict(source)
        try:
            source_bytes = (canonical_json(payload) + "\n").encode("utf-8")
        except (AuditContractError, TypeError, ValueError, RecursionError):
            try:
                source_bytes = (json.dumps(payload, sort_keys=True, allow_nan=True) + "\n").encode(
                    "utf-8"
                )
            except (TypeError, ValueError, RecursionError) as exc:
                raise AuditScanError("source is not JSON-compatible") from exc
        rows = (
            [
                (item if isinstance(item, Mapping) else None, index)
                for index, item in enumerate(payload.get("episodes", []), start=1)
            ]
            if isinstance(payload.get("episodes"), list)
            else []
        )
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        rows = [
            (item if isinstance(item, Mapping) else None, index)
            for index, item in enumerate(source, start=1)
        ]
        payload = {
            "schema_version": "episode-jsonl.v1",
            "episodes": [dict(item) if isinstance(item, Mapping) else item for item, _ in rows],
        }
        try:
            source_bytes = (canonical_json(payload) + "\n").encode("utf-8")
        except (AuditContractError, TypeError, ValueError, RecursionError):
            try:
                source_bytes = (json.dumps(payload, sort_keys=True, allow_nan=True) + "\n").encode(
                    "utf-8"
                )
            except (TypeError, ValueError, RecursionError) as exc:
                raise AuditScanError("source is not JSON-compatible") from exc
    else:
        raise AuditScanError("source must be a local path, mapping, or row sequence")
    if "episodes" in payload and not isinstance(payload["episodes"], list):
        raise AuditScanError("campaign episodes must be an array")
    metadata_payload = dict(payload)
    if isinstance(metadata_payload.get("episodes"), list):
        # Row-level non-finite values are retained as invalid inventory rows;
        # campaign metadata and source identity are rejected immediately.
        metadata_payload["episodes"] = []
    _strict_walk(metadata_payload, path="campaign")
    if len(source_bytes) > MAX_SOURCE_BYTES:
        raise AuditScanError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    digest = _sha256(source_bytes)
    ref = _source_ref_from_value(
        source_ref, source_path=source_path, observed_digest=digest, payload=payload
    )
    if source_path is not None and source_ref is not None:
        root_path = Path(root) if root is not None else Path.cwd()
        try:
            expected_uri = source_path.relative_to(root_path.resolve(strict=True)).as_posix()
        except (OSError, ValueError, RuntimeError) as exc:
            raise AuditScanError("source reference root binding is unavailable") from exc
        if ref.uri != expected_uri:
            raise AuditScanError("source reference URI does not bind the admitted path")
    if ref.format not in SUPPORTED_SOURCE_FORMATS or ref.schema not in SUPPORTED_SOURCE_SCHEMAS:
        source_status = "unsupported"
    declared_payload_schema = _payload_schema(payload)
    _identity_aliases(
        payload,
        names=("campaign_id", "study_id", "campaign"),
        label="campaign_id",
        include_episode_rows=True,
    )
    if declared_payload_schema and ref.schema != declared_payload_schema:
        source_status = "unsupported"
    execution_state, execution_reason = _execution_status(payload, path="campaign")
    if execution_state == "invalid":
        source_status = "invalid"
    elif execution_state == "unsupported" and source_status == "native":
        source_status = "unsupported"
    if execution_reason and source_status == "native":
        source_status = "invalid"
    return _LoadedCampaign(
        payload, tuple(rows), ref, digest, source_bytes, source_status, execution_reason
    )


def _expected_ids(  # noqa: C901, PLR0912
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, ...]:
    """Extract explicit expected IDs; observed IDs are the fallback population.

    Returns:
        Sorted expected episode identifiers.
    """

    candidates: Any = None
    declared = False
    for container in (config, payload):
        for key in ("expected_episode_ids", "expected_episodes", "episode_expectations"):
            if key in container:
                candidates = container[key]
                declared = True
                break
        if declared:
            break
    if declared and candidates is None:
        return ()
    if candidates is None:
        candidates = []
    if isinstance(candidates, Mapping):
        candidates = list(candidates)
    elif isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        if declared:
            raise AuditScanError("expected episode IDs must be an array or mapping")
        candidates = []
    ids: list[str] = []
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for item in candidates:
            if isinstance(item, Mapping):
                value = item.get("episode_id") or item.get("id")
            else:
                value = item
            if not isinstance(value, str) or not value.strip():
                if declared:
                    raise AuditScanError("expected episode IDs must be non-empty strings")
                continue
            normalized = value.strip()
            if normalized not in ids:
                ids.append(normalized)
    if declared:
        return tuple(sorted(ids))
    observed: list[str] = []
    episodes = payload.get("episodes")
    if isinstance(episodes, list):
        for item in episodes:
            value = item.get("episode_id") if isinstance(item, Mapping) else None
            if isinstance(value, str) and value.strip() and value.strip() not in observed:
                observed.append(value.strip())
    return tuple(sorted(observed))


def _row_episode_id(row: Mapping[str, Any] | None, *, line: int) -> str:
    if isinstance(row, Mapping):
        value = row.get("episode_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"invalid-line-{line}"


def _row_validation(  # noqa: C901, PLR0912
    row: Mapping[str, Any] | None, *, line: int
) -> tuple[str, str, str]:
    if row is None:
        return _row_episode_id(row, line=line), "invalid", "row is not a JSON object"
    if _contains_nonfinite(row):
        return _row_episode_id(row, line=line), "invalid", "row contains a non-finite value"
    try:
        _strict_walk(row, path=f"row[{line}]")
    except AuditScanError as exc:
        return _row_episode_id(row, line=line), "invalid", str(exc)
    episode_id = _row_episode_id(row, line=line)
    raw_id = row.get("episode_id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        return episode_id, "invalid", "episode_id is missing or empty"
    # Identity fields are descriptive and deliberately not constrained to a
    # fixed release/planner vocabulary, but a promised identity must still be
    # a non-empty string.  Do not coerce a mapping/list into ``str(...)``: that
    # would let malformed untrusted data influence peer cohorts or EpisodeRef
    # joins while appearing readable.
    for key in (
        "campaign_id",
        "study_id",
        "planner_id",
        "planner",
        "algo",
        "algorithm",
        "scenario_id",
        "scenario",
        "config_id",
        "config_identity",
        "source_commit",
        "git_hash",
        "commit_sha",
        "commit",
    ):
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return episode_id, "invalid", f"{key} is malformed"
    try:
        _identity_aliases(
            row,
            names=("source_commit", "git_hash", "commit_sha", "commit"),
            label=f"row[{line}].source_commit",
        )
        _identity_aliases(
            row,
            names=("config_identity", "config_hash"),
            label=f"row[{line}].config_identity",
        )
    except AuditScanError as exc:
        return episode_id, "invalid", str(exc)
    for key in ("seed",):
        if (
            key in row
            and row[key] is not None
            and (isinstance(row[key], bool) or not isinstance(row[key], (int, str)))
        ):
            return episode_id, "invalid", f"{key} has unsupported type"
    for key in ("execution_id", "run_id"):
        if key in row and (not isinstance(row[key], str) or not row[key].strip()):
            return episode_id, "invalid", f"{key} is malformed"
    for key in (
        "checkpoint_digest",
        "checkpoint_hash",
        "environment_digest",
        "environment_hash",
        "config_digest",
    ):
        value = row.get(key)
        if value not in (None, "") and (
            not isinstance(value, str) or _IDENTITY_DIGEST_RE.fullmatch(value) is None
        ):
            return episode_id, "invalid", f"{key} is not a hexadecimal digest"
    config_block = row.get("config")
    if isinstance(config_block, Mapping):
        for key in ("config_digest", "config_hash"):
            value = config_block.get(key)
            if value not in (None, "") and (
                not isinstance(value, str) or _IDENTITY_DIGEST_RE.fullmatch(value) is None
            ):
                return episode_id, "invalid", f"config.{key} is not a hexadecimal digest"
    if "attempt" in row and (
        isinstance(row["attempt"], bool)
        or not isinstance(row["attempt"], int)
        or row["attempt"] < 0
    ):
        return episode_id, "invalid", "attempt is malformed"
    for key in ("metrics", "config", "provenance", "outcome"):
        if key in row and row[key] is not None and not isinstance(row[key], Mapping):
            # Outcome is allowed as a scalar label; all other containers are
            # intentionally closed to avoid silently changing detector meaning.
            if key != "outcome" or not isinstance(row[key], str):
                return episode_id, "invalid", f"{key} must be an object"
    for key in (
        "analysis_trace",
        "trace",
        "simulation_trace",
        "events",
        "commands",
        "control",
        "geometry",
        "initial_state",
        "capabilities",
        "telemetry_contract",
    ):
        value = row.get(key)
        if value is not None and not isinstance(value, (Mapping, list)):
            return episode_id, "invalid", f"{key} must be an object or array"
    for key in ("analysis_trace", "trace", "simulation_trace"):
        value = row.get(key)
        if isinstance(value, Mapping):
            frames = next(
                (value[name] for name in ("frames", "steps", "trajectory") if name in value),
                None,
            )
            if frames is not None and (
                not isinstance(frames, list)
                or not all(isinstance(frame, Mapping) for frame in frames)
            ):
                return episode_id, "invalid", f"{key}.frames must be an array of objects"
        elif isinstance(value, list) and not all(isinstance(frame, Mapping) for frame in value):
            return episode_id, "invalid", f"{key} must contain objects"
    algorithm_metadata = row.get("algorithm_metadata")
    if algorithm_metadata is not None and not isinstance(algorithm_metadata, Mapping):
        return episode_id, "invalid", "algorithm_metadata must be an object"
    if isinstance(algorithm_metadata, Mapping):
        analysis_trace = algorithm_metadata.get("analysis_trace")
        if analysis_trace is not None and not isinstance(analysis_trace, Mapping):
            return episode_id, "invalid", "algorithm_metadata.analysis_trace must be an object"
        if isinstance(analysis_trace, Mapping):
            frames = next(
                (
                    analysis_trace[name]
                    for name in ("frames", "steps", "trajectory")
                    if name in analysis_trace
                ),
                None,
            )
            if frames is not None and (
                not isinstance(frames, list)
                or not all(isinstance(frame, Mapping) for frame in frames)
            ):
                return episode_id, "invalid", "analysis_trace.steps must be an array of objects"
    execution_state, execution_reason = _execution_status(row, path=f"row[{line}]")
    if execution_state == "invalid":
        return episode_id, "invalid", execution_reason
    if execution_state == "unsupported":
        return episode_id, "unsupported", execution_reason or "non-admissible execution status"
    return episode_id, "readable", ""


def _identity_digest(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and _IDENTITY_DIGEST_RE.fullmatch(value):
        return value.lower()
    try:
        canonical = canonical_json(value if value is not None else fallback)
    except (AuditContractError, TypeError, ValueError, RecursionError) as exc:
        raise AuditScanError("campaign identity is not strict JSON") from exc
    return _sha256(canonical.encode("utf-8"))


def _row_config_digest(row: Mapping[str, Any]) -> str:
    for key in ("config_digest", "config_hash"):
        value = row.get(key)
        if isinstance(value, str) and value:
            if _IDENTITY_DIGEST_RE.fullmatch(value):
                return value.lower()
            return _sha256(value.encode("utf-8"))
    config = row.get("config")
    if isinstance(config, Mapping):
        for key in ("config_digest", "config_hash"):
            value = config.get(key)
            if isinstance(value, str) and value:
                if _IDENTITY_DIGEST_RE.fullmatch(value):
                    return value.lower()
                return _sha256(value.encode("utf-8"))
        return _sha256(canonical_json(config).encode("utf-8"))
    config_identity = row.get("config_identity")
    if isinstance(config_identity, str) and config_identity:
        return _sha256(config_identity.encode("utf-8"))
    return ""


def _episode_ref(
    row: Mapping[str, Any], *, audit: CampaignAudit, source_ref: SourceRef
) -> EpisodeRef:
    checkpoint = row.get("checkpoint_digest") or row.get("checkpoint_hash")
    environment = row.get("environment_digest") or row.get("environment_hash")
    execution_id = row.get("execution_id") or row.get("run_id") or row.get("episode_id")
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise AuditScanError("episode execution_id is missing")
    return EpisodeRef(
        campaign_digest=audit.campaign_digest,
        source_digest=audit.source_digest,
        execution_id=execution_id.strip(),
        planner_id=str(row.get("planner_id") or row.get("planner") or row.get("algo") or ""),
        scenario_id=str(row.get("scenario_id") or row.get("scenario") or ""),
        seed=row.get("seed"),
        attempt=row.get("attempt", 0),
        config_digest=_row_config_digest(row),
        checkpoint_digest=checkpoint
        if isinstance(checkpoint, str) and _IDENTITY_DIGEST_RE.fullmatch(checkpoint)
        else "",
        environment_digest=environment
        if isinstance(environment, str) and _IDENTITY_DIGEST_RE.fullmatch(environment)
        else "",
        source=source_ref,
    )


def _campaign_audit(
    payload: Mapping[str, Any],
    *,
    source_ref: SourceRef,
    source_digest: str,
    config: Mapping[str, Any],
) -> CampaignAudit:
    campaign_id = (
        payload.get("campaign_id")
        or config.get("campaign_id")
        or payload.get("study_id")
        or "campaign"
    )
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise AuditScanError("campaign identity is missing")
    campaign_digest = _identity_digest(
        payload.get("campaign_digest") or payload.get("campaign_id") or campaign_id,
        fallback=campaign_id,
    )
    source_commit = (
        payload.get("source_commit") or payload.get("git_hash") or source_ref.source_commit
    )
    config_identity = (
        payload.get("config_identity")
        or payload.get("config_hash")
        or config.get("config_identity")
        or source_ref.config_identity
    )
    metadata = {
        "campaign_id": campaign_id.strip(),
        "source_artifact_id": source_ref.artifact_id,
        "source_format": source_ref.format,
        "source_schema": source_ref.schema,
        "source_commit": source_commit if isinstance(source_commit, str) else "",
        "config_identity": config_identity if isinstance(config_identity, str) else "",
        "evidence_boundary": "diagnostic_only",
        "claim_boundary": "candidate_signals_only; no benchmark validity verdict",
    }
    audit_id = "audit-" + _sha256(
        canonical_json(
            {"campaign_digest": campaign_digest, "source_digest": source_digest, "config": config}
        ).encode("utf-8")
    )
    return CampaignAudit(
        audit_id=audit_id,
        campaign_digest=campaign_digest,
        source_digest=source_digest,
        title=str(payload.get("title") or campaign_id),
        protocol_version=str(config.get("protocol_version") or "audit-protocol.v1"),
        protocol_digest=_identity_digest(
            config.get("protocol_digest") or "audit-protocol.v1", fallback="audit-protocol.v1"
        ),
        source=source_ref,
        metadata=metadata,
    )


def _cache_key(
    *,
    source_digest: str,
    config: Mapping[str, Any],
    registry: DetectorRegistry,
    detector_ids: Sequence[str],
) -> str:
    payload = {
        "schema_version": AUDIT_SCAN_SCHEMA_VERSION,
        "source_digest": source_digest,
        "registry_digest": registry.digest,
        "detector_ids": sorted(detector_ids),
        "config": dict(config),
    }
    return _sha256(canonical_json(payload).encode("utf-8"))


def _review_counts(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    readable_ids: set[str],
    readable_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    human: set[str] = set()
    agent: set[str] = set()
    full_episode: set[str] = set()
    human_full: set[str] = set()
    agent_full: set[str] = set()
    finding_representatives: set[str] = set()
    confirmed_members: set[str] = set()
    rows_by_id = {
        str(row.get("episode_id")): row
        for row in readable_rows
        if isinstance(row.get("episode_id"), str)
    }

    def stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
        outcome = row.get("outcome")
        if isinstance(outcome, Mapping):
            outcome_value = outcome.get("label") or outcome.get("status")
        else:
            outcome_value = outcome
        group = (
            row.get("scenario_group")
            or row.get("scenario_group_id")
            or row.get("group")
            or row.get("scenario_id")
            or "unknown"
        )
        planner = row.get("planner_id") or row.get("planner") or "unknown"
        return str(planner), str(group), str(outcome_value or "unknown")

    strata_by_id = {episode_id: stratum(row) for episode_id, row in rows_by_id.items()}
    observed_strata = set(strata_by_id.values())
    control_ids = {
        episode_id
        for episode_id, row in rows_by_id.items()
        if row.get("is_control") is True
        or row.get("control_candidate") is True
        or row.get("ordinary_control") is True
        or row.get("role") in {"control", "ordinary", "normal_control"}
        or row.get("case_role") in {"control", "ordinary", "normal_control"}
    }
    for container in (payload, config):
        values = container.get("reviewed_episode_ids")
        if isinstance(values, list):
            human.update(item for item in values if isinstance(item, str) and item in readable_ids)
        values = container.get("agent_reviewed_episode_ids")
        if isinstance(values, list):
            agent.update(item for item in values if isinstance(item, str) and item in readable_ids)
        values = container.get("full_episode_reviewed_ids")
        if isinstance(values, list):
            full_episode.update(
                item for item in values if isinstance(item, str) and item in readable_ids
            )
        for key, target in (
            ("reviewed_finding_representatives", finding_representatives),
            ("finding_representative_episode_ids", finding_representatives),
            ("confirmed_affected_episode_ids", confirmed_members),
            ("confirmed_finding_member_ids", confirmed_members),
        ):
            values = container.get(key)
            if isinstance(values, list):
                target.update(
                    item for item in values if isinstance(item, str) and item in readable_ids
                )
        records = container.get("review_records")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, Mapping) or record.get("episode_id") not in readable_ids:
                    continue
                if record.get("author_kind") == "agent":
                    agent.add(record["episode_id"])
                elif record.get("author_kind") == "human":
                    human.add(record["episode_id"])
                if (
                    record.get("scope") == "full_episode"
                    or record.get("review_scope") == "full_episode"
                ):
                    full_episode.add(record["episode_id"])
                    if record.get("author_kind") == "human":
                        human_full.add(record["episode_id"])
                    elif record.get("author_kind") == "agent":
                        agent_full.add(record["episode_id"])
    # Some retained campaign exports attach review receipts to the episode
    # rather than maintaining a campaign-level list.  Read only the compact
    # identity/scope fields here; the full review-record contract remains
    # owned by BA-03.
    for row in readable_rows:
        sidecars = [row.get("review_context"), row.get("review_records")]
        for sidecar in sidecars:
            records = sidecar if isinstance(sidecar, list) else [sidecar]
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                episode_id = record.get("episode_id") or row.get("episode_id")
                if episode_id not in readable_ids:
                    continue
                if record.get("author_kind") == "agent":
                    agent.add(episode_id)
                elif record.get("author_kind") == "human":
                    human.add(episode_id)
                if (
                    record.get("scope") == "full_episode"
                    or record.get("review_scope") == "full_episode"
                ):
                    full_episode.add(episode_id)
                    if record.get("author_kind") == "human":
                        human_full.add(episode_id)
                    elif record.get("author_kind") == "agent":
                        agent_full.add(episode_id)
    reviewed_union = human | agent
    human_strata = {strata_by_id[item] for item in human if item in strata_by_id}
    agent_strata = {strata_by_id[item] for item in agent if item in strata_by_id}
    # Legacy list-style inputs do not carry author-specific scope, so an ID
    # explicitly listed as both human-reviewed and full is treated as a human
    # full review.  Typed review records retain the author boundary above.
    full_human = human_full | (
        {
            episode_id
            for episode_id in full_episode
            if episode_id in human and episode_id not in agent_full
        }
    )
    control_reviewed = control_ids & reviewed_union
    control_strata = {strata_by_id[item] for item in control_ids if item in strata_by_id}
    control_reviewed_strata = {
        strata_by_id[item] for item in control_ids & reviewed_union if item in strata_by_id
    }
    control_groups = {(planner, group) for planner, group, _outcome in control_strata}
    control_reviewed_groups = {
        (planner, group) for planner, group, _outcome in control_reviewed_strata
    }
    return {
        "human_reviewed": len(human),
        "agent_reviewed": len(agent),
        "full_episode_reviewed": len(full_episode),
        "interval_reviewed": len((human | agent) - full_episode),
        "reviewed_episode_union": len(reviewed_union),
        "finding_representatives": len(finding_representatives),
        "confirmed_affected_members": len(confirmed_members),
        "denominator": len(readable_ids),
        "observed_planner_group_outcome_strata": len(observed_strata),
        "human_reviewed_strata": len(human_strata),
        "agent_reviewed_strata": len(agent_strata),
        "full_human_reviewed_strata": len(
            {strata_by_id[item] for item in full_human if item in strata_by_id}
        ),
        "unreviewed_strata": [
            {
                "planner_id": planner,
                "scenario_group": group,
                "outcome": outcome,
            }
            for planner, group, outcome in sorted(observed_strata - (human_strata | agent_strata))
        ],
        "ordinary_control_candidates": len(control_ids),
        "ordinary_control_reviewed": len(control_reviewed),
        "ordinary_control_gaps": sorted(control_ids - reviewed_union),
        "ordinary_control_strata": len(control_strata),
        "ordinary_control_reviewed_strata": len(control_reviewed_strata),
        "ordinary_control_gap_strata": [
            {
                "planner_id": planner,
                "scenario_group": group,
                "outcome": outcome,
            }
            for planner, group, outcome in sorted(control_strata - control_reviewed_strata)
        ],
        "ordinary_control_planner_group_count": len(control_groups),
        "ordinary_control_reviewed_planner_group_count": len(control_reviewed_groups),
        "ordinary_control_gap_planner_groups": [
            {"planner_id": planner, "scenario_group": group}
            for planner, group in sorted(control_groups - control_reviewed_groups)
        ],
    }


def _diagnostic_counts(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    requests = []
    executed = []
    original_evidence = []
    for container in (payload, config):
        for key, target in (
            ("enrichment_requests", requests),
            ("diagnostic_requests", requests),
            ("diagnostic_executions", executed),
            ("recorded_evidence", original_evidence),
            ("original_recorded_evidence", original_evidence),
            ("evidence_records", original_evidence),
        ):
            values = container.get(key)
            if isinstance(values, list):
                target.extend(item for item in values if isinstance(item, Mapping))
    return {
        "requested": len(requests),
        "executed": len(executed),
        "successful": sum(1 for item in executed if item.get("status") in {"complete", "success"}),
        "failed": sum(1 for item in executed if item.get("status") in {"failed", "error"}),
        "original_recorded_evidence": len(original_evidence),
        "new_diagnostic_executions": len(executed),
    }


def _enrichment_requests(
    signals: Sequence[Signal], *, source_ref: SourceRef, source_digest: str
) -> tuple[Mapping[str, Any], ...]:
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    capability_map = {
        "trace": "trace",
        "geometry": "geometry",
        "commands": "commands",
        "forces": "forces",
        "events": "events",
        "cohort": "cohort",
        "metrics": "metrics",
    }
    for signal in signals:
        if signal.status not in {"unavailable", "error"}:
            continue
        fields = signal.missingness or (signal.reason_code,)
        for field_name in fields:
            token = str(field_name).split(".", 1)[0]
            capability = capability_map.get(token)
            if capability is None:
                continue
            key = (signal.episode_id, capability)
            requests[key] = {
                "request_id": "enrich-"
                + _sha256(
                    canonical_json(
                        {
                            "episode_id": signal.episode_id,
                            "capability": capability,
                            "source_digest": source_digest,
                        }
                    ).encode("utf-8")
                ),
                "episode_id": signal.episode_id,
                "capability": capability,
                "status": "requested",
                "reason": signal.reason_code,
                "source_artifact_id": source_ref.artifact_id,
                "source_digest": source_digest,
                "producer": "ba01-audit-scan",
                "execution": "not_requested_here",
            }
    return tuple(requests[key] for key in sorted(requests))


def _make_inventory_signal(
    spec: Any, row: Mapping[str, Any], status: str, config: Mapping[str, Any]
) -> Signal:
    """Represent a missing/invalid/duplicate input as an unavailable attempt.

    Returns:
        A strict BA-03 unavailable signal.
    """

    return unavailable_signal(
        spec,
        row,
        f"episode_{status}",
        missing=(f"inventory.{status}",),
        config=config,
    )


def scan_campaign(  # noqa: C901, PLR0912, PLR0915
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    root: str | Path | None = None,
    source_ref: SourceRef | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    detector_ids: Sequence[str] | None = None,
    include_advisory: bool = True,
    registry: DetectorRegistry | None = None,
    store: AuditStore | None = None,
) -> AuditScanReport:
    """Scan one campaign source into a deterministic, missing-aware report.

    ``store`` is optional.  When supplied, the campaign audit and all signal
    records are committed through BA-03's existing idempotent transaction API;
    the scan itself still reads only the caller-supplied source.

    Returns:
        A deterministic missing-aware audit report.
    """

    if config is not None and not isinstance(config, Mapping):
        raise AuditScanError("config must be a mapping")
    requested_config = dict(config or {})
    _strict_walk(requested_config, path="config")
    loaded = _load_source(source, root=root, source_ref=source_ref)
    source_config = loaded.payload.get("config")
    if source_config is not None and not isinstance(source_config, Mapping):
        raise AuditScanError("campaign config must be a mapping")
    source_config_identity = _identity_aliases(
        loaded.payload,
        names=("config_identity", "config_hash"),
        label="campaign.config_identity",
    )
    requested_config_identity = _identity_aliases(
        requested_config,
        names=("config_identity", "config_hash"),
        label="requested.config_identity",
    )
    if (
        source_config_identity
        and requested_config_identity
        and source_config_identity != requested_config_identity
    ):
        raise AuditScanError("requested config identity conflicts with campaign")
    requested_source_commit = _identity_aliases(
        requested_config,
        names=("source_commit", "git_hash", "commit_sha", "commit"),
        label="requested.source_commit",
    )
    if (
        loaded.source_ref.source_commit
        and requested_source_commit
        and loaded.source_ref.source_commit != requested_source_commit
    ):
        raise AuditScanError("requested source_commit conflicts with campaign")
    config_mapping = {
        **(dict(source_config) if isinstance(source_config, Mapping) else {}),
        **requested_config,
    }
    _strict_walk(config_mapping, path="config")
    if len(canonical_json(config_mapping).encode("utf-8")) > MAX_CONFIG_BYTES:
        raise AuditScanError(f"config exceeds {MAX_CONFIG_BYTES} bytes")
    if len(loaded.rows) > MAX_EPISODES:
        raise AuditScanError(f"observed episode count exceeds {MAX_EPISODES}")
    active_registry = registry or default_registry(include_advisory=include_advisory)
    selected_ids = (
        normalize_detector_ids(
            detector_ids,
            include_advisory=any(item.advisory for item in active_registry),
            registry=active_registry,
        )
        if detector_ids is not None
        else active_registry.ids
    )
    if any(item not in active_registry.ids for item in selected_ids):
        raise AuditScanError("requested detector is not present in supplied registry")
    audit = _campaign_audit(
        loaded.payload,
        source_ref=loaded.source_ref,
        source_digest=loaded.source_digest,
        config=config_mapping,
    )
    source_campaign_identity = _identity_aliases(
        loaded.payload,
        names=("campaign_id", "study_id", "campaign"),
        label="campaign_id",
        include_episode_rows=True,
    )
    requested_campaign_identity = _identity_aliases(
        requested_config,
        names=("campaign_id", "study_id", "campaign"),
        label="requested.campaign_id",
    )
    if (
        source_campaign_identity
        and requested_campaign_identity
        and source_campaign_identity != requested_campaign_identity
    ):
        raise AuditScanError("requested campaign_id conflicts with campaign")
    expected_campaign_identity = (
        source_campaign_identity
        or requested_campaign_identity
        or str(audit.metadata.get("campaign_id") or "")
    )
    expected_config_identity = source_config_identity or requested_config_identity
    expected_ids = _expected_ids(loaded.payload, config_mapping)
    if len(expected_ids) > MAX_EPISODES:
        raise AuditScanError(f"expected episode count exceeds {MAX_EPISODES}")
    rows_by_id: dict[str, list[tuple[Mapping[str, Any] | None, int, str, str]]] = {}
    for row, line in loaded.rows:
        episode_id, status, reason = _row_validation(row, line=line)
        if status == "readable" and row is not None:
            binding_error = _row_binding_error(
                row,
                loaded.source_ref,
                expected_campaign_id=expected_campaign_identity,
                expected_config_identity=expected_config_identity,
            )
            if binding_error is not None:
                status, reason = "invalid", binding_error
        rows_by_id.setdefault(episode_id, []).append((row, line, status, reason))
    all_ids = sorted(set(expected_ids) | set(rows_by_id))
    inventory: list[EpisodeInventory] = []
    readable_rows: list[Mapping[str, Any]] = []
    episode_refs: list[EpisodeRef] = []
    for episode_id in all_ids:
        matches = rows_by_id.get(episode_id, [])
        # When expectations are omitted, ``_expected_ids`` falls back to the
        # valid observed IDs.  Anonymous malformed rows therefore remain
        # unexpected observations rather than silently inflating the expected
        # denominator; an explicitly empty expectation list has the same
        # fail-closed behavior.
        expected = episode_id in expected_ids
        if loaded.source_status == "invalid":
            inventory.append(
                EpisodeInventory(
                    episode_id,
                    "invalid",
                    expected,
                    loaded.source_ref.artifact_id,
                    matches[0][1] if matches else None,
                    loaded.source_error or "source execution metadata is malformed",
                    None,
                )
            )
            continue
        if loaded.source_status == "unsupported":
            inventory.append(
                EpisodeInventory(
                    episode_id,
                    "unsupported",
                    expected,
                    loaded.source_ref.artifact_id,
                    matches[0][1] if matches else None,
                    "source format or schema is unsupported",
                    None,
                )
            )
            continue
        if loaded.source_status in EXECUTION_UNSUPPORTED:
            inventory.append(
                EpisodeInventory(
                    episode_id,
                    "unsupported",
                    expected,
                    loaded.source_ref.artifact_id,
                    matches[0][1] if matches else None,
                    f"source execution status is {loaded.source_status}",
                    None,
                )
            )
            continue
        if not matches:
            inventory.append(
                EpisodeInventory(
                    episode_id,
                    "missing",
                    expected,
                    loaded.source_ref.artifact_id,
                    None,
                    "expected episode is absent",
                    None,
                )
            )
            continue
        if len(matches) > 1:
            inventory.append(
                EpisodeInventory(
                    episode_id,
                    "duplicate",
                    expected,
                    loaded.source_ref.artifact_id,
                    matches[0][1],
                    "episode_id occurs more than once",
                    None,
                )
            )
            continue
        row, line, status, reason = matches[0]
        if status != "readable" or row is None:
            inventory.append(
                EpisodeInventory(
                    episode_id, status, expected, loaded.source_ref.artifact_id, line, reason, None
                )
            )
            continue
        row_copy = dict(row)
        # Join keys are canonicalized once at admission and carried into
        # every retained row, EpisodeRef, and Signal.
        row_copy["episode_id"] = episode_id
        for canonical, aliases in (
            ("planner_id", ("planner", "algo", "algorithm")),
            ("scenario_id", ("scenario",)),
            ("source_commit", ("git_hash", "commit_sha", "commit")),
            ("config_identity", ("config_hash",)),
        ):
            if canonical not in row_copy:
                alias_value = next(
                    (row_copy.get(alias) for alias in aliases if row_copy.get(alias) is not None),
                    None,
                )
                if alias_value is None and canonical == "planner_id":
                    params = row_copy.get("scenario_params")
                    if isinstance(params, Mapping):
                        alias_value = next(
                            (
                                params.get(alias)
                                for alias in aliases
                                if params.get(alias) is not None
                            ),
                            None,
                        )
                if alias_value is not None:
                    row_copy[canonical] = alias_value
        # Campaign-level provenance is authoritative for rows that omit a
        # repeated copy.  Keep row overrides intact so contradictory values
        # remain visible to the provenance detector.
        row_provenance = row_copy.get("provenance")
        if not isinstance(row_provenance, Mapping):
            row_provenance = {}
        inherited_provenance = {
            "campaign_id": loaded.payload.get("campaign_id") or loaded.payload.get("study_id"),
            "source_commit": (
                loaded.payload.get("source_commit")
                or loaded.payload.get("git_hash")
                or loaded.source_ref.source_commit
            ),
            "config_identity": loaded.payload.get("config_identity")
            or loaded.payload.get("config_hash")
            or loaded.source_ref.config_identity,
        }
        merged_provenance = {
            key: value
            for key, value in {**inherited_provenance, **dict(row_provenance)}.items()
            if isinstance(value, str) and value.strip()
        }
        if merged_provenance:
            row_copy["provenance"] = merged_provenance
        for key, value in inherited_provenance.items():
            if isinstance(value, str) and value.strip():
                row_copy.setdefault(key, value)
        row_copy.setdefault("campaign_digest", audit.campaign_digest)
        row_copy.setdefault("source_digest", loaded.source_digest)
        row_copy.setdefault("source_artifact_id", loaded.source_ref.artifact_id)
        row_copy.setdefault("source_uri", loaded.source_ref.uri)
        row_copy.setdefault("execution_id", episode_id)
        for alias, identity_key in (
            ("checkpoint_hash", "checkpoint_digest"),
            ("environment_hash", "environment_digest"),
        ):
            value = row_copy.get(alias)
            if identity_key not in row_copy and isinstance(value, str):
                row_copy[identity_key] = value
        if "config_digest" not in row_copy:
            config_digest = _row_config_digest(row_copy)
            if config_digest:
                row_copy["config_digest"] = config_digest
        row_copy.setdefault(
            "expected_provenance",
            {
                key: value
                for key, value in {
                    "source_commit": loaded.source_ref.source_commit,
                    "config_identity": loaded.source_ref.config_identity,
                }.items()
                if isinstance(value, str) and value.strip()
            },
        )
        inventory.append(
            EpisodeInventory(
                episode_id, "readable", expected, loaded.source_ref.artifact_id, line, "", row_copy
            )
        )
        readable_rows.append(row_copy)
    inventory.sort(key=lambda item: (item.episode_id, item.line_number or 0, item.status))
    readable_ids = {item.episode_id for item in inventory if item.readable}
    signals: list[Signal] = []
    # A non-readable expected row still receives an explicit attempt result so
    # unavailable/error accounting cannot disappear into an omitted denominator.
    for item in inventory:
        row = (
            dict(item.row)
            if item.readable and item.row is not None
            else {
                "episode_id": item.episode_id,
                "campaign_digest": audit.campaign_digest,
                "source_digest": loaded.source_digest,
                "execution_id": item.episode_id,
            }
        )
        for detector_id in selected_ids:
            if item.readable:
                signal = detect(
                    detector_id,
                    row,
                    cohort=readable_rows,
                    config=config_mapping,
                    registry=active_registry,
                )
            else:
                spec = active_registry.get(detector_id)
                signal = _make_inventory_signal(spec, row, item.status, config_mapping)
            signals.append(signal)
    signals.sort(key=lambda item: (item.episode_id, item.detector_id, item.signal_id))
    inventory_counts = Counter(item.status for item in inventory if item.expected)
    observed_counts = Counter(item.status for item in inventory)
    status_counts = signal_status_counts(signals)
    detector_evaluable = status_counts["flagged"] + status_counts["clear"]
    coverage = {
        "expected": len(expected_ids),
        "indexed": sum(
            inventory_counts.get(status, 0)
            for status in ("readable", "duplicate", "invalid", "unsupported")
        ),
        "readable": inventory_counts.get("readable", 0),
        "missing": inventory_counts.get("missing", 0),
        "duplicate": inventory_counts.get("duplicate", 0),
        "invalid": inventory_counts.get("invalid", 0),
        "unsupported": inventory_counts.get("unsupported", 0),
        "unexpected_observed": sum(
            count
            for status, count in observed_counts.items()
            if status and status not in INVENTORY_STATUSES
        )
        + sum(1 for item in inventory if not item.expected),
        "observed_rows": len(loaded.rows),
    }
    detectors = {
        "scheduled": len(signals),
        "evaluable": detector_evaluable,
        "flagged": status_counts["flagged"],
        "clear": status_counts["clear"],
        "unavailable": status_counts["unavailable"],
        "error": status_counts["error"],
        "by_status": status_counts,
        "by_detector": {
            detector_id: signal_status_counts(
                item for item in signals if item.detector_id == detector_id
            )
            for detector_id in selected_ids
        },
        "candidate_signals": status_counts["flagged"],
        "confirmed_findings": 0,
    }
    review = _review_counts(loaded.payload, config_mapping, readable_ids, readable_rows)
    diagnostics = _diagnostic_counts(loaded.payload, config_mapping)
    cache_key = _cache_key(
        source_digest=loaded.source_digest,
        config=config_mapping,
        registry=active_registry,
        detector_ids=selected_ids,
    )
    enrichment = _enrichment_requests(
        signals, source_ref=loaded.source_ref, source_digest=loaded.source_digest
    )
    diagnostics["requested"] += len(enrichment)
    counts = {
        "coverage": coverage,
        "detectors": detectors,
        "review": review,
        "diagnostic": diagnostics,
    }
    provenance = {
        "source": {
            "artifact_id": loaded.source_ref.artifact_id,
            "uri": loaded.source_ref.uri,
            "format": loaded.source_ref.format,
            "schema": loaded.source_ref.schema,
            "sha256_declared": loaded.source_ref.sha256,
            "sha256_observed": loaded.source_digest,
            "source_commit": loaded.source_ref.source_commit,
            "config_identity": loaded.source_ref.config_identity,
            "integrity_status": "digest_verified",
            "execution_status": loaded.source_status,
        },
        "detector_engine_version": DETECTOR_ENGINE_VERSION,
        "detector_registry_version": active_registry.version,
        "detector_registry_digest": active_registry.digest,
        "config": dict(config_mapping),
        "cache_key": cache_key,
        "evidence_status": "diagnostic_only",
        "claim_boundary": "candidate signals are not confirmed findings or benchmark validity verdicts",
        "no_simulation_execution": True,
        "no_network_or_shell": True,
    }
    report = AuditScanReport(
        audit,
        tuple(inventory),
        tuple(signals),
        active_registry,
        counts,
        provenance,
        enrichment,
        cache_key,
        tuple(episode_refs),
    )
    for item in inventory:
        if item.readable and item.row is not None:
            try:
                episode_refs.append(
                    _episode_ref(item.row, audit=audit, source_ref=loaded.source_ref)
                )
            except (AuditScanError, AuditContractError, TypeError, ValueError):
                # The inventory remains readable for detector purposes, while
                # a weak optional EpisodeRef is explicitly represented in
                # provenance rather than silently synthesized as an identity.
                continue
    if episode_refs:
        report = AuditScanReport(
            audit,
            tuple(inventory),
            tuple(signals),
            active_registry,
            counts,
            provenance,
            enrichment,
            cache_key,
            tuple(sorted(episode_refs, key=lambda item: item.episode_id)),
        )
    if store is not None:
        records: list[Any] = [report.audit, *report.signals]
        store.commit(
            records, operation_id=f"audit-scan:{cache_key}", actor="detector", actor_id=COMPONENT_ID
        )
    return report


def scan(*args: Any, **kwargs: Any) -> AuditScanReport:
    """Short alias for :func:`scan_campaign`.

    Returns:
        The generated audit report.
    """

    return scan_campaign(*args, **kwargs)


def audit_campaign(*args: Any, **kwargs: Any) -> AuditScanReport:
    """Compatibility alias for :func:`scan_campaign`.

    Returns:
        The generated audit report.
    """

    return scan_campaign(*args, **kwargs)


def _registry_from_document(value: Any) -> DetectorRegistry | None:
    """Reconstruct a validated registry from a serialized report document.

    Returns:
        A validated registry, or ``None`` for a malformed document.
    """

    if not isinstance(value, Mapping):
        return None
    raw_detectors = value.get("detectors")
    if not isinstance(raw_detectors, list):
        return None
    if value.get("engine_version") != DETECTOR_ENGINE_VERSION:
        return None
    try:
        specs = []
        for raw in raw_detectors:
            if not isinstance(raw, Mapping):
                return None
            specs.append(
                DetectorSpec(
                    detector_id=raw["detector_id"],
                    family=raw["family"],
                    description=raw["description"],
                    version=raw.get("version", "1.0.0"),
                    required_capabilities=tuple(raw.get("required_capabilities", ())),
                    optional_capabilities=tuple(raw.get("optional_capabilities", ())),
                    cohort_definition=raw.get("cohort_definition", {}),
                    parameters=raw.get("parameters", {}),
                    units=raw.get("units", {}),
                    provenance=raw.get("provenance", {}),
                    advisory=raw.get("advisory", False),
                )
            )
        return DetectorRegistry(
            version=str(value.get("schema_version", value.get("version", ""))),
            detectors=tuple(specs),
        )
    except (AuditContractError, KeyError, TypeError, ValueError, RecursionError):
        return None


def _report_detector_ids(report: AuditScanReport | Mapping[str, Any]) -> tuple[str, ...] | None:
    """Read the actual selected detector set when a report records it.

    Returns:
        Sorted selected IDs, or ``None`` when the report has no selection map.
    """

    if isinstance(report, AuditScanReport):
        detectors = report.counts.get("detectors", {})
    else:
        counts = report.get("counts")
        detectors = counts.get("detectors", {}) if isinstance(counts, Mapping) else {}
    by_detector = detectors.get("by_detector") if isinstance(detectors, Mapping) else None
    if not isinstance(by_detector, Mapping):
        return None
    return tuple(sorted(item for item in by_detector if isinstance(item, str)))


def cache_is_current(  # noqa: C901
    report: AuditScanReport | Mapping[str, Any],
    *,
    source_digest: str,
    config: Mapping[str, Any] | None = None,
    registry: DetectorRegistry | None = None,
    detector_ids: Sequence[str] | None = None,
) -> bool:
    """Return whether a cached report matches source, config and detector versions."""

    if not isinstance(report, (AuditScanReport, Mapping)):
        return False
    if not isinstance(source_digest, str) or _DIGEST_RE.fullmatch(source_digest) is None:
        return False
    source_digest = source_digest.lower()
    if registry is not None:
        active_registry = registry
    elif isinstance(report, AuditScanReport):
        active_registry = report.detector_registry
    else:
        serialized_registry = report.get("detector_registry")
        active_registry = _registry_from_document(serialized_registry)
        if active_registry is None:
            return False
    if isinstance(report, AuditScanReport):
        if report.audit.source_digest != source_digest:
            return False
        source_provenance = report.provenance.get("source")
    else:
        audit_document = report.get("audit")
        if (
            not isinstance(audit_document, Mapping)
            or not isinstance(audit_document.get("source_digest"), str)
            or audit_document.get("source_digest").lower() != source_digest
        ):
            return False
        source_provenance = (
            report.get("provenance", {}).get("source")
            if isinstance(report.get("provenance"), Mapping)
            else None
        )
    if isinstance(source_provenance, Mapping):
        observed = source_provenance.get("sha256_observed")
        if observed is not None and (
            not isinstance(observed, str) or observed.lower() != source_digest
        ):
            return False
    recorded_ids = _report_detector_ids(report)
    if detector_ids is None:
        selected = recorded_ids or active_registry.ids
    else:
        selected = normalize_detector_ids(detector_ids, registry=active_registry)
    recorded_provenance = (
        report.provenance
        if isinstance(report, AuditScanReport)
        else report.get("provenance")
        if isinstance(report.get("provenance"), Mapping)
        else {}
    )
    recorded_config = (
        recorded_provenance.get("config")
        if isinstance(recorded_provenance.get("config"), Mapping)
        else {}
    )
    effective_config = {**dict(recorded_config), **dict(config or {})}
    try:
        expected = _cache_key(
            source_digest=source_digest,
            config=effective_config,
            registry=active_registry,
            detector_ids=selected,
        )
    except (AuditContractError, TypeError, ValueError, RecursionError):
        return False
    actual = report.cache_key if isinstance(report, AuditScanReport) else report.get("cache_key")
    return actual == expected


def descriptor() -> dict[str, Any]:
    """Return the component descriptor for offline BA-01 invocation."""

    return {
        "schema_version": "component-descriptor.v1",
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "supported_input_versions": ["component-request.v1"],
        "output_types": [AUDIT_SCAN_SCHEMA_VERSION, "component-result.v1"],
        "required_capabilities": ["campaign-result"],
        "optional_capabilities": [
            "episode-selection",
            "trace",
            "events",
            "commands",
            "geometry",
            "forces",
            "cohort",
        ],
    }


def capability_report() -> dict[str, Any]:
    """Return the component's declared required and optional capabilities."""

    return {
        "schema_version": "audit-capability-report.v1",
        "component_id": COMPONENT_ID,
        "required_capabilities": descriptor()["required_capabilities"],
        "optional_capabilities": descriptor()["optional_capabilities"],
        "detectors": list(DEFAULT_REGISTRY.ids),
    }


def _result_document(result: ComponentResult) -> dict[str, Any]:
    payload = {"schema_version": "component-result.v1", **asdict(result)}
    return payload


def _reject_component_request(request: ComponentRequest) -> ComponentResult | None:
    """Return a stable result when a component request is out of scope."""

    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request.request_id,
            request.component_id,
            STATUS_UNAVAILABLE,
            reason="unsupported_component",
        )
    if not isinstance(request.config, Mapping):
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            STATUS_FAILED,
            reason="invalid_request: config must be an object",
        )
    supported = set(descriptor()["required_capabilities"]) | set(
        descriptor()["optional_capabilities"]
    )
    missing = sorted(item for item in request.required_capabilities if item not in supported)
    if missing:
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(missing)}",
        )
    minimum = request.config.get("min_component_version")
    if minimum is not None:
        version_pattern = re.compile(r"^(?:v)?(\d+)\.(\d+)\.(\d+)$")
        wanted = version_pattern.fullmatch(str(minimum))
        ours = version_pattern.fullmatch(COMPONENT_VERSION)
        if wanted is None or ours is None:
            return ComponentResult(
                request.request_id,
                COMPONENT_ID,
                STATUS_FAILED,
                reason="incompatible_component_version: malformed min_component_version",
            )
        if tuple(int(value) for value in wanted.groups()) > tuple(
            int(value) for value in ours.groups()
        ):
            return ComponentResult(
                request.request_id,
                COMPONENT_ID,
                STATUS_FAILED,
                reason=(
                    "incompatible_component_version: "
                    f"request needs v{str(minimum).lstrip('v')}, component is v{COMPONENT_VERSION}"
                ),
            )
    return None


def _component_scan_status(report: AuditScanReport) -> tuple[str, str]:
    """Classify a report without treating partial evidence as completion.

    Returns:
        The BA-03 component status and a semicolon-delimited reason.
    """

    coverage = report.counts["coverage"]
    input_blocking = (
        any(coverage[status] > 0 for status in ("missing", "duplicate", "invalid", "unsupported"))
        or coverage.get("unexpected_observed", 0) > 0
    )
    detector_error = report.counts["detectors"].get("error", 0) > 0
    source_execution = report.provenance.get("source", {}).get("execution_status")
    source_token = (
        _status_token(source_execution, path="source.execution_status")[0]
        if isinstance(source_execution, str)
        else None
    )
    source_blocking = source_token not in {"native", "adapter"}
    reasons: list[str] = []
    if detector_error:
        reasons.append("detector_evaluation_contains_errors")
    if source_blocking:
        reasons.append("source_execution_is_not_native")
    if input_blocking:
        reasons.append("input_accounting_contains_non_readable_rows")
    if detector_error:
        status = STATUS_FAILED
    elif source_blocking or input_blocking:
        status = STATUS_PARTIAL
    else:
        status = STATUS_COMPLETE
    return status, ";".join(reasons)


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Run the BA-01 component request and return a BA-03 result envelope.

    Returns:
        A component result whose artifacts contain the scan report.
    """

    if not isinstance(request, ComponentRequest):
        return ComponentResult(
            "unknown",
            COMPONENT_ID,
            STATUS_FAILED,
            reason="invalid_request: expected ComponentRequest",
        )
    rejected = _reject_component_request(request)
    if rejected is not None:
        return rejected
    if any(not isinstance(ref, SourceRef) for ref in request.sources):
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            STATUS_FAILED,
            reason="invalid_request: sources must contain SourceRef values",
        )
    if any(
        not isinstance(capability, str) or not capability.strip()
        for capability in request.required_capabilities
    ):
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            STATUS_FAILED,
            reason="invalid_request: required_capabilities must contain strings",
        )
    config = dict(request.config)
    source_refs = [ref for ref in request.sources if ref.format in SUPPORTED_SOURCE_FORMATS]
    if not source_refs:
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            STATUS_UNAVAILABLE,
            reason="required_source_family_missing: campaign-result",
        )
    selected_ref = source_refs[0]
    output_root = base if base is not None else Path.cwd()
    try:
        report = scan_campaign(
            selected_ref.uri,
            root=output_root,
            source_ref=selected_ref,
            config=config,
            detector_ids=config.get("detector_ids"),
            include_advisory=config.get("include_advisory", True),
        )
        output_directory = Path(request.output_directory)
        if (
            output_directory.is_absolute()
            or ".." in output_directory.parts
            or "\\" in str(output_directory)
        ):
            raise AuditScanError("output directory escapes base")
        # Keep the public output-path contract check separate from publication:
        # the descriptor-relative reservation below is the authority that
        # remains safe if this path is replaced between the two operations.
        _safe_path_for_output(str(output_directory / AUDIT_REPORT_FILENAME), root=output_root)
        _safe_path_for_output(str(output_directory / AUDIT_REGISTRY_FILENAME), root=output_root)
        reserved_output = None
        try:
            reserved_output = _reserve_output_directory(str(output_directory), output_root)
            report_text = (
                json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            registry_text = (
                json.dumps(
                    report.detector_registry.to_dict(), sort_keys=True, indent=2, allow_nan=False
                )
                + "\n"
            )
            report_digest = _atomic_materialize_no_replace(
                reserved_output.path / AUDIT_REPORT_FILENAME,
                report_text,
                output_directory=reserved_output,
            )
            registry_digest = _atomic_materialize_no_replace(
                reserved_output.path / AUDIT_REGISTRY_FILENAME,
                registry_text,
                output_directory=reserved_output,
            )
            _assert_output_directory_current(reserved_output)
            artifacts = (
                {
                    "artifact_id": AUDIT_REPORT_FILENAME,
                    "uri": str(output_directory / AUDIT_REPORT_FILENAME),
                    "sha256": report_digest,
                },
                {
                    "artifact_id": AUDIT_REGISTRY_FILENAME,
                    "uri": str(output_directory / AUDIT_REGISTRY_FILENAME),
                    "sha256": registry_digest,
                },
            )
        except BaseException:
            _release_empty_output(reserved_output)
            raise
        finally:
            if reserved_output is not None:
                reserved_output.close()
        status, reason = _component_scan_status(report)
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            status,
            artifacts=artifacts if status == STATUS_COMPLETE else (),
            provenance=report.provenance,
            reason=reason,
        )
    except (AuditScanError, AuditContractError, OSError, TypeError, ValueError) as exc:
        return ComponentResult(
            request.request_id,
            COMPONENT_ID,
            STATUS_FAILED,
            reason=f"audit_scan_failed:{type(exc).__name__}",
        )


def _read_cli_json(
    path: str,
    *,
    root: Path,
    max_bytes: int = MAX_SOURCE_BYTES,
    validate: bool = True,
) -> Any:
    """Read one CLI JSON input under the admitted root.

    Returns:
        The parsed JSON value.
    """

    target = _safe_path(path, root=root)
    resolved_root = root.resolve(strict=True)
    try:
        relative = target.relative_to(resolved_root)
        raw = _read_descriptor_backed_file(
            relative.parts,
            resolved_root,
            limit=max_bytes,
            kind=f"CLI input {path}",
        )
    except (OSError, ReviewContractsValidationError) as exc:
        raise AuditScanError("CLI input was replaced or is not a readable regular file") from exc
    return _strict_json(raw, label=path, validate=validate)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline RobotSF benchmark audit scan.")
    parser.add_argument(
        "--input", required=False, help="Campaign JSON/JSONL or component request JSON."
    )
    parser.add_argument("--output", default=None, help="Report JSON path; stdout when omitted.")
    parser.add_argument("--base", default=None, help="Admitted root for local source paths.")
    parser.add_argument(
        "--config", default=None, help="Optional JSON config merged over input config."
    )
    parser.add_argument(
        "--detector", action="append", dest="detectors", help="Detector ID; repeatable."
    )
    parser.add_argument(
        "--no-advisory", action="store_true", help="Skip advisory statistical detectors."
    )
    parser.add_argument("--descriptor", action="store_true", help="Print the component descriptor.")
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    """Execute a bounded CLI and print strict machine-readable JSON.

    Returns:
        Zero on success and two on a contract or safety error.
    """

    try:
        args = _build_parser().parse_args(argv)
        if args.descriptor:
            print(json.dumps(descriptor(), sort_keys=True))  # noqa: T201
            return 0
        if args.input is None:
            raise AuditScanError("--input is required unless --descriptor is requested")
        base = Path(args.base) if args.base is not None else Path.cwd()
        input_path = _safe_path(args.input, root=base)
        input_payload: Any = None
        if input_path.suffix.lower() not in {".jsonl", ".ndjson"}:
            input_payload = _read_cli_json(args.input, root=base, validate=False)
        config: dict[str, Any] = {}
        if isinstance(input_payload, Mapping) and isinstance(input_payload.get("config"), Mapping):
            config.update(input_payload["config"])
        if args.config is not None:
            loaded_config = _read_cli_json(args.config, root=base, max_bytes=MAX_CONFIG_BYTES)
            if not isinstance(loaded_config, Mapping):
                raise AuditScanError("CLI config must be an object")
            config.update(loaded_config)
        if (
            isinstance(input_payload, Mapping)
            and input_payload.get("schema_version") == "component-request.v1"
        ):
            request_payload = dict(input_payload)
            request_payload["output_directory"] = request_payload.get(
                "output_directory", "audit-output"
            )
            request_payload["config"] = config
            request = component_request_from_dict(request_payload, source=args.input)
            result = run(request, base=base)
            document = _result_document(result)
        else:
            source_ref = None
            if isinstance(input_payload, Mapping) and isinstance(
                input_payload.get("source_ref"), Mapping
            ):
                source_ref = input_payload["source_ref"]
            report = scan_campaign(
                args.input,
                root=base,
                source_ref=source_ref,
                config=config,
                detector_ids=args.detectors,
                include_advisory=not args.no_advisory,
            )
            document = report.to_dict()
        serialized = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            _write_cli_output(args.output, serialized, root=base)
        print(serialized, end="")  # noqa: T201
        return 0
    except (
        AuditScanError,
        AuditContractError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        document = {
            "schema_version": "audit-scan-error.v1",
            "status": "failed",
            "reason": type(exc).__name__,
        }
        print(json.dumps(document, sort_keys=True))  # noqa: T201
        return 2


def _safe_path_for_output(value: str, *, root: Path) -> Path:
    raw = Path(value)
    windows = PureWindowsPath(value)
    if (
        raw.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in raw.parts
        or "\\" in str(value)
        or "\x00" in str(value)
        or any(ord(character) < 32 or ord(character) == 127 for character in str(value))
    ):
        raise AuditScanError("output path escapes admitted root")
    resolved_root = root.resolve(strict=True)
    lexical = resolved_root / raw
    current = resolved_root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise AuditScanError("output path contains a symlink component")
    target = lexical.resolve(strict=False)
    target.relative_to(resolved_root)
    if target.exists():
        raise AuditScanError("output path already exists")
    return target


def _write_cli_output(value: str, serialized: str, *, root: Path) -> str:
    """Publish CLI output through a retained descriptor-relative parent.

    Returns:
        The SHA-256 digest of the published UTF-8 bytes.
    """

    target = _safe_path_for_output(value, root=root)
    resolved_root = root.resolve(strict=True)
    try:
        relative = target.relative_to(resolved_root)
    except ValueError as exc:
        raise AuditScanError("output path escapes admitted root") from exc
    if not relative.parts:
        raise AuditScanError("output path must name a file")
    encoded = serialized.encode("utf-8")
    parent_guard = None
    temporary_fd = -1
    retained_fd = -1
    temporary_name: str | None = None
    try:
        parent_guard = _open_output_parent_guard(resolved_root, relative.parts)
        _assert_output_parent_current(parent_guard)
        parent_fd = parent_guard.parent_fd
        temporary_fd, temporary_name = _open_output_temporary(parent_fd, relative.name)
        retained_fd = os.dup(temporary_fd)
        handle = os.fdopen(temporary_fd, "wb")
        temporary_fd = -1
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_output(
            parent_fd,
            temporary_name,
            relative.name,
            None,
            set(),
            temporary_fd=retained_fd,
            parent_guard=parent_guard,
        )
        temporary_name = None
        return _sha256(encoded)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if retained_fd >= 0:
            os.close(retained_fd)
        if temporary_name is not None and parent_guard is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_guard.parent_fd)
            except FileNotFoundError:
                pass
        if parent_guard is not None:
            parent_guard.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_REGISTRY_FILENAME",
    "AUDIT_REPORT_FILENAME",
    "AUDIT_SCAN_SCHEMA_VERSION",
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "INVENTORY_STATUSES",
    "AuditScanError",
    "AuditScanReport",
    "EpisodeInventory",
    "audit_campaign",
    "cache_is_current",
    "capability_report",
    "descriptor",
    "main",
    "run",
    "scan",
    "scan_campaign",
]
