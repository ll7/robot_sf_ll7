"""SREV-05 review-events component: index event/phase intervals with links.

This module owns the SREV-05 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts and builds an ordered
event/phase interval index with precursor/recovery links. Overlap and nesting
retain identities (nothing is merged); missing links resolve to explicit
unavailable reasons (never invented); existing metric values and failure
categories pass through unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
)

COMPONENT_ID = "srev05-review-events"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = (
    "event-list",
    "phase-list",
)
OPTIONAL_CAPABILITIES = ("predicate-report",)

OUTPUT_INDEX_FILENAME = "event-index.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("event-index.v1", "missing-capability-report.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)

_SEMVER_RE = re.compile(r"^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass
class _IndexedInterval:
    """One normalized interval with explicit links and availability."""

    interval_id: str
    kind: str
    start_s: float
    end_s: float
    actor_ids: list[str] = field(default_factory=list)
    category: str = ""
    metric_value: Any = None
    precursor_ids: list[str] = field(default_factory=list)
    recovery_ids: list[str] = field(default_factory=list)
    links_available: bool = True
    links_reason: str = ""


def descriptor() -> dict[str, Any]:
    """Return this component's self-contained capability descriptor."""
    return {
        "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
        "component_id": _DESCRIPTOR.component_id,
        "component_version": _DESCRIPTOR.component_version,
        "supported_input_versions": list(_DESCRIPTOR.supported_input_versions),
        "required_capabilities": list(_DESCRIPTOR.required_capabilities),
        "optional_capabilities": list(_DESCRIPTOR.optional_capabilities),
        "output_types": list(_DESCRIPTOR.output_types),
    }


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write strict-JSON.

    Returns:
        Hex digest of the written bytes.
    """
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ReviewContractsValidationError(
            [f"output parent is not a real directory: {path.parent}"]
        )
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise ReviewContractsValidationError(
            [f"output_collision: temporary artifact already exists: {tmp_path.name}"]
        ) from error
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    try:
        # Hard-linking a complete temporary file creates the final name with
        # no-replace semantics.  ``Path.replace`` would silently overwrite a
        # file created by a concurrent producer between the two operations.
        try:
            os.link(tmp_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ReviewContractsValidationError(
                [f"output_collision: artifact already exists: {path.name}"]
            ) from error
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return _sha256_bytes(text.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies it.
    """
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    wanted_match = _SEMVER_RE.fullmatch(str(minimum))
    ours_match = _SEMVER_RE.fullmatch(COMPONENT_VERSION)
    if wanted_match is None or ours_match is None:
        return f"incompatible_component_version: malformed min_component_version: {minimum!r}"
    wanted = tuple(int(value) for value in wanted_match.groups())
    ours = tuple(int(value) for value in ours_match.groups())
    if wanted > ours:
        return (
            "incompatible_component_version: "
            f"request needs v{str(minimum).lstrip('v')}, component is v{COMPONENT_VERSION}"
        )
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject requests this component cannot serve.

    Returns:
        An unavailable/failed result, or None when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=version_error,
        )
    return None


def _resolve_source(path_value: str, base: Path) -> Path:
    """Resolve a source URI under the base directory.

    Returns:
        Resolved path; absolute URIs and traversal are rejected.
    """
    candidate = Path(path_value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ReviewContractsValidationError(
            [f"source uri rejected (absolute or traversal): {path_value}"]
        )
    try:
        root = base.resolve(strict=True)
        if not root.is_dir():
            raise ReviewContractsValidationError([f"source base is not a directory: {base}"])
        unresolved = root.joinpath(candidate)
        current = root
        for part in candidate.parts:
            current /= part
            if current.is_symlink():
                raise ReviewContractsValidationError(
                    [f"source uri rejected (symlink component): {path_value}"]
                )
        resolved = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReviewContractsValidationError(
            [f"source uri cannot be resolved safely: {path_value}: {type(error).__name__}"]
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ReviewContractsValidationError(
            [f"source uri rejected (outside base or not a file): {path_value}"]
        )
    return resolved


def _read_source_bytes(path_value: str, base: Path) -> bytes:
    """Read one source through a directory-FD chain without TOCTOU escapes.

    The lexical and resolved-path checks remain useful diagnostics, but a
    second path-based ``read_bytes`` would reopen a potentially swapped path.
    Opening every directory component and the final file with ``O_NOFOLLOW``
    binds the read to the checked base directory and rejects symlink races.

    Returns:
        Raw bytes read from the regular source file.
    """
    _resolve_source(path_value, base)
    candidate = Path(path_value)
    root = base.resolve(strict=True)
    try:
        nofollow = os.O_NOFOLLOW
        directory = os.O_DIRECTORY
    except AttributeError as error:
        raise ReviewContractsValidationError(
            ["source safe read unavailable: platform lacks no-follow directory opens"]
        ) from error
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    directory_flags = common_flags | directory
    directory_fd = os.open(root, directory_flags)
    try:
        for component in candidate.parts[:-1]:
            next_directory_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_directory_fd
        source_fd = os.open(candidate.parts[-1], common_flags, dir_fd=directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise ReviewContractsValidationError(
                    [f"source uri rejected (not a regular file): {path_value}"]
                )
            with os.fdopen(source_fd, "rb") as handle:
                source_fd = -1
                return handle.read()
        finally:
            if source_fd != -1:
                os.close(source_fd)
    finally:
        os.close(directory_fd)


def _reserve_output_directory(output_directory: str, base: Path) -> Path:  # noqa: C901 - path guard
    """Atomically reserve a real output directory below ``base``.

    Returns:
        The newly created output directory.
    """
    relative = Path(output_directory)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ReviewContractsValidationError(
            [f"output directory rejected (absolute or traversal): {output_directory}"]
        )
    try:
        root = base.resolve(strict=True)
        if not root.is_dir():
            raise ReviewContractsValidationError([f"output base is not a directory: {base}"])
        parent = root
        for part in relative.parts[:-1]:
            parent = parent / part
            if parent.is_symlink():
                raise ReviewContractsValidationError(
                    [f"output directory rejected (symlink component): {output_directory}"]
                )
            try:
                parent.mkdir()
            except FileExistsError:
                pass
            if parent.is_symlink() or not parent.is_dir():
                raise ReviewContractsValidationError(
                    [f"output directory parent is not a real directory: {output_directory}"]
                )
            if not parent.resolve(strict=True).is_relative_to(root):
                raise ReviewContractsValidationError(
                    [f"output directory escapes base: {output_directory}"]
                )
        output = parent / relative.parts[-1]
        if output.is_symlink() or output.exists():
            raise ReviewContractsValidationError(
                [f"output_collision: already exists: {output_directory}"]
            )
        output.mkdir()
        resolved = output.resolve(strict=True)
    except FileExistsError as error:
        raise ReviewContractsValidationError(
            [f"output_collision: already exists: {output_directory}"]
        ) from error
    except (OSError, RuntimeError) as error:
        raise ReviewContractsValidationError(
            [
                f"output directory cannot be reserved safely: {output_directory}: {type(error).__name__}"
            ]
        ) from error
    if not resolved.is_relative_to(root) or resolved != output:
        raise ReviewContractsValidationError([f"output directory escapes base: {output_directory}"])
    return output


def _release_empty_output(output_directory: Path | None) -> None:
    """Release an output reservation when no artifact has been written."""
    if output_directory is None:
        return
    try:
        if output_directory.is_dir() and not output_directory.is_symlink():
            output_directory.rmdir()
    except OSError:
        pass


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or None for malformed input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _normalize_interval(
    item: Any, *, index: int, kind: str
) -> tuple[_IndexedInterval | None, str | None]:
    """Normalize one raw interval, clipping nothing yet.

    Returns:
        Tuple of (interval or None, diagnostic code or None).
    """
    if not isinstance(item, dict):
        return None, f"{kind}_row_{index}_malformed"
    interval_id = item.get("interval_id")
    start = _finite_number(item.get("start_s"))
    end = _finite_number(item.get("end_s"))
    if not isinstance(interval_id, str) or not interval_id:
        return None, f"{kind}_row_{index}_missing_id"
    if start is None or end is None or not end > start:
        return None, f"{kind}_row_{index}_malformed"
    actor_ids = item.get("actor_ids", [])
    precursors = item.get("precursor_ids", [])
    recoveries = item.get("recovery_ids", [])
    if not isinstance(actor_ids, list) or not isinstance(precursors, list):
        return None, f"{kind}_row_{index}_malformed"
    if not isinstance(recoveries, list):
        return None, f"{kind}_row_{index}_malformed"
    if not all(isinstance(actor_id, str) for actor_id in actor_ids):
        return None, f"{kind}_row_{index}_typed_actor_id"
    if not all(isinstance(link_id, str) for link_id in [*precursors, *recoveries]):
        return None, f"{kind}_row_{index}_typed_link_id"
    category = item.get("category", "")
    if not isinstance(category, str):
        return None, f"{kind}_row_{index}_typed_category"
    return (
        _IndexedInterval(
            interval_id=interval_id,
            kind=kind,
            start_s=start,
            end_s=end,
            actor_ids=list(actor_ids),
            category=category,
            metric_value=item.get("metric_value"),
            precursor_ids=list(precursors),
            recovery_ids=list(recoveries),
        ),
        None,
    )


def _clip_interval(
    interval: _IndexedInterval, t0: float, terminal: float
) -> tuple[_IndexedInterval | None, str | None]:
    """Clip one interval into range, preserving its identity.

    Returns:
        Tuple of (clipped interval or None when fully outside, diagnostic or None).
    """
    if interval.end_s <= t0 or interval.start_s >= terminal:
        return None, f"{interval.interval_id}: outside_range"
    clipped = _IndexedInterval(
        interval_id=interval.interval_id,
        kind=interval.kind,
        start_s=max(interval.start_s, t0),
        end_s=min(interval.end_s, terminal),
        actor_ids=list(interval.actor_ids),
        category=interval.category,
        metric_value=interval.metric_value,
        precursor_ids=list(interval.precursor_ids),
        recovery_ids=list(interval.recovery_ids),
    )
    if clipped.start_s != interval.start_s or clipped.end_s != interval.end_s:
        return clipped, f"{interval.interval_id}: clipped_to_range"
    return clipped, None


def _resolve_links(
    intervals: list[_IndexedInterval],
) -> list[str]:
    """Validate declared precursor/recovery links without inventing any.

    Returns:
        Diagnostic codes for dangling or absent links.
    """
    diagnostics: list[str] = []
    known = {interval.interval_id for interval in intervals}
    for interval in intervals:
        dangling = [
            link for link in interval.precursor_ids + interval.recovery_ids if link not in known
        ]
        if dangling:
            interval.links_available = False
            interval.links_reason = f"dangling_links:{','.join(sorted(dangling))}"
            diagnostics.append(f"{interval.interval_id}: dangling_links")
        elif not interval.precursor_ids and not interval.recovery_ids:
            interval.links_available = False
            interval.links_reason = "links_unavailable:undeclared"
            diagnostics.append(f"{interval.interval_id}: links_unavailable")
    return diagnostics


def _load_family(
    refs: list[Any],
    family: str,
    root: Path,
    diagnostics: list[str],
    source_provenance: list[dict[str, Any]],
    loaded_families: dict[str, int],
) -> list[dict[str, Any]]:
    """Load and parse all sources of one family.

    Returns:
        Parsed raw interval dicts.
    """
    items: list[dict[str, Any]] = []
    for ref in refs:
        provenance = {
            "artifact_id": ref.artifact_id,
            "uri": ref.uri,
            "format": ref.format,
            "schema_declared": ref.schema or None,
            "sha256_declared": ref.sha256.lower() if ref.sha256 else None,
            "source_commit": ref.source_commit or None,
            "config_identity": ref.config_identity or None,
            "units": ref.units or None,
            "coordinate_frame": ref.coordinate_frame or None,
            "sha256_observed": None,
            "integrity_status": "unverified",
        }
        try:
            raw = _read_source_bytes(ref.uri, root)
        except (OSError, ReviewContractsValidationError):
            diagnostics.append(f"{ref.artifact_id}: source_unreadable_or_unsafe")
            source_provenance.append(provenance)
            continue
        observed_sha = _sha256_bytes(raw)
        provenance["sha256_observed"] = observed_sha
        if not ref.sha256:
            diagnostics.append(f"{ref.artifact_id}: source_digest_missing")
            source_provenance.append(provenance)
            continue
        if ref.sha256.lower() != observed_sha:
            diagnostics.append(f"{ref.artifact_id}: source_digest_mismatch")
            provenance["integrity_status"] = "digest_mismatch"
            source_provenance.append(provenance)
            continue
        expected_schema = f"{family}.v1"
        if ref.schema != expected_schema:
            diagnostics.append(f"{ref.artifact_id}: source_schema_mismatch:{expected_schema}")
            provenance["integrity_status"] = "schema_mismatch"
            source_provenance.append(provenance)
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            diagnostics.append(f"{ref.artifact_id}: source_not_json")
            provenance["integrity_status"] = "invalid_json"
            source_provenance.append(provenance)
            continue
        if not isinstance(payload, dict):
            diagnostics.append(f"{ref.artifact_id}: source_not_json_object")
            provenance["integrity_status"] = "source_shape_invalid"
            source_provenance.append(provenance)
            continue
        if payload.get("schema_version") != expected_schema:
            diagnostics.append(f"{ref.artifact_id}: source_payload_schema_mismatch")
            provenance["integrity_status"] = "schema_mismatch"
            source_provenance.append(provenance)
            continue
        entries = payload.get("intervals")
        if not isinstance(entries, list):
            diagnostics.append(f"{ref.artifact_id}: intervals_not_a_list")
            provenance["integrity_status"] = "source_shape_invalid"
            source_provenance.append(provenance)
            continue
        provenance["integrity_status"] = "digest_and_schema_verified"
        source_provenance.append(provenance)
        loaded_families[family] = loaded_families.get(family, 0) + 1
        items.extend(entries)
    return items


_FAMILY_KINDS = {"event-list": "event", "phase-list": "phase", "predicate-report": "predicate"}


def _collect_family_refs(
    request: ComponentRequest, diagnostics: list[str]
) -> dict[str, list[Any]] | None:
    """Group sources by family, recording format problems.

    Returns:
        Family-to-refs mapping, or None when a required family is absent.
    """
    by_format: dict[str, list[Any]] = {}
    for ref in request.sources:
        if ref.format not in REQUIRED_CAPABILITIES + OPTIONAL_CAPABILITIES:
            diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
            continue
        if ref.format in OPTIONAL_CAPABILITIES and ref.format not in request.required_capabilities:
            diagnostics.append(f"{ref.artifact_id}: optional_stream_skipped:{ref.format}")
            continue
        by_format.setdefault(ref.format, []).append(ref)
    required_families = set(REQUIRED_CAPABILITIES) | set(request.required_capabilities)
    for family in sorted(required_families):
        if family not in by_format:
            diagnostics.append(f"required_source_family_missing:{family}")
    if any(code.startswith("required_source_family_missing") for code in diagnostics):
        return None
    return by_format


def _normalize_families(
    by_format: dict[str, list[Any]],
    root: Path,
    t0: float,
    terminal: float,
    diagnostics: list[str],
    source_provenance: list[dict[str, Any]],
    loaded_families: dict[str, int],
) -> list[_IndexedInterval]:
    """Normalize, clip, and deduplicate intervals across families.

    Returns:
        Unique clipped intervals (unsorted).
    """
    intervals: list[_IndexedInterval] = []
    for family in REQUIRED_CAPABILITIES + OPTIONAL_CAPABILITIES:
        for ref in by_format.get(family, []):
            raw_items = _load_family(
                [ref], family, root, diagnostics, source_provenance, loaded_families
            )
            for position, item in enumerate(raw_items):
                interval, problem = _normalize_interval(
                    item, index=position, kind=_FAMILY_KINDS[family]
                )
                if problem is not None or interval is None:
                    diagnostics.append(f"{ref.artifact_id}: {problem or 'interval_unavailable'}")
                    continue
                clipped, clip_note = _clip_interval(interval, t0, terminal)
                if clip_note is not None:
                    diagnostics.append(f"{ref.artifact_id}: {clip_note}")
                if clipped is not None:
                    intervals.append(clipped)
    seen: dict[str, _IndexedInterval] = {}
    unique: list[_IndexedInterval] = []
    for interval in intervals:
        previous = seen.get(interval.interval_id)
        if previous is not None:
            if (
                previous.kind,
                previous.start_s,
                previous.end_s,
                previous.actor_ids,
                previous.category,
                previous.metric_value,
                previous.precursor_ids,
                previous.recovery_ids,
            ) != (
                interval.kind,
                interval.start_s,
                interval.end_s,
                interval.actor_ids,
                interval.category,
                interval.metric_value,
                interval.precursor_ids,
                interval.recovery_ids,
            ):
                diagnostics.append(f"conflicting_duplicate_interval_id:{interval.interval_id}")
            else:
                diagnostics.append(f"duplicate_interval_id:{interval.interval_id}")
            continue
        seen[interval.interval_id] = interval
        unique.append(interval)
    return unique


_INFORMATIONAL_DIAGNOSTIC_CODES = frozenset({"optional_stream_skipped", "clipped_to_range"})


def _diagnostic_code(diagnostic: str) -> str:
    """Extract the structured code suffix from a rendered diagnostic.

    Returns:
        The diagnostic code, or an empty string for an unstructured value.
    """
    _, separator, suffix = diagnostic.rpartition(": ")
    if not separator:
        return ""
    return suffix.split(":", 1)[0]


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Index event/phase intervals with explicit precursor/recovery links.

    Args:
        request: Validated component request with event-list and phase-list sources.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when every interval indexed with
        resolvable links, ``partial`` on unavailable links or skipped rows,
        ``unavailable`` when the component or a required capability does not
        apply, ``failed`` on validation, version, collision, or internal errors.
    """
    if not isinstance(request, ComponentRequest):
        request_id = (
            request.get("request_id")
            if isinstance(request, dict) and isinstance(request.get("request_id"), str)
            else "unknown"
        )
        component_id = (
            request.get("component_id")
            if isinstance(request, dict) and isinstance(request.get("component_id"), str)
            else COMPONENT_ID
        )
        return ComponentResult(
            request_id=request_id,
            component_id=component_id,
            status=STATUS_FAILED,
            reason="invalid_request: expected validated ComponentRequest",
        )
    root = base if base is not None else Path.cwd()
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    t0 = _finite_number(request.config.get("t0_s", 0.0))
    terminal = _finite_number(request.config.get("terminal_s"))
    if t0 is None or terminal is None or not terminal > t0:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="invalid_config: t0_s/terminal_s must be finite with terminal_s > t0_s",
        )
    diagnostics: list[str] = []
    source_provenance: list[dict[str, Any]] = []
    loaded_families: dict[str, int] = {}
    output_dir: Path | None = None
    try:
        output_dir = _reserve_output_directory(request.output_directory, root)
        by_format = _collect_family_refs(request, diagnostics)
        if by_format is None:
            _release_empty_output(output_dir)
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="required_source_family_missing: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        unique = _normalize_families(
            by_format,
            root,
            t0,
            terminal,
            diagnostics,
            source_provenance,
            loaded_families,
        )
        required_families = set(REQUIRED_CAPABILITIES) | set(request.required_capabilities)
        unavailable_required = sorted(
            family for family in required_families if loaded_families.get(family, 0) == 0
        )
        diagnostics.extend(
            f"required_source_unavailable:{family}" for family in unavailable_required
        )
        if not unique:
            _release_empty_output(output_dir)
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="no_intervals_indexed: " + "; ".join(sorted(set(diagnostics))[:5]),
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance={"sources": source_provenance},
            )
        if unavailable_required:
            _release_empty_output(output_dir)
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason=(
                    "required_source_unavailable: "
                    + ", ".join(unavailable_required)
                    + "; "
                    + "; ".join(sorted(set(diagnostics))[:5])
                ),
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance={"sources": source_provenance},
            )
        diagnostics.extend(_resolve_links(unique))
        unique.sort(key=lambda row: (row.start_s, row.end_s))
        document = {
            "schema_version": "event-index.v1",
            "t0_s": t0,
            "terminal_s": terminal,
            "intervals": [
                {
                    "interval_id": row.interval_id,
                    "kind": row.kind,
                    "start_s": row.start_s,
                    "end_s": row.end_s,
                    "actor_ids": row.actor_ids,
                    "category": row.category,
                    "metric_value": row.metric_value,
                    "precursor_ids": row.precursor_ids,
                    "recovery_ids": row.recovery_ids,
                    "links_available": row.links_available,
                    "links_reason": row.links_reason,
                }
                for row in unique
            ],
        }
        declared_optional = {
            ref.format for ref in request.sources if ref.format in OPTIONAL_CAPABILITIES
        }
        requested_capabilities = set(request.required_capabilities)
        capability_payload = {
            "schema_version": "missing-capability-report.v1",
            "missing_capabilities": sorted(set(OPTIONAL_CAPABILITIES) - declared_optional),
            "skipped_optional_streams": sorted(declared_optional - requested_capabilities),
            "diagnostics": sorted(set(diagnostics)),
        }
        blocking = [
            item
            for item in diagnostics
            if _diagnostic_code(item) not in _INFORMATIONAL_DIAGNOSTIC_CODES
        ]
        partial = bool(blocking)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        index_digest = _write_json(output_dir / "event-index.json", document)
        capability_digest = _write_json(output_dir / OUTPUT_CAPABILITY_FILENAME, capability_payload)
        artifacts: tuple[dict[str, Any], ...] = (
            (
                {
                    "artifact_id": "event-index.json",
                    "uri": str(Path(request.output_directory) / "event-index.json"),
                    "sha256": index_digest,
                },
                {
                    "artifact_id": OUTPUT_CAPABILITY_FILENAME,
                    "uri": str(Path(request.output_directory) / OUTPUT_CAPABILITY_FILENAME),
                    "sha256": capability_digest,
                },
            )
            if status == STATUS_COMPLETE
            else ()
        )
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=artifacts,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            provenance={
                "output_directory": request.output_directory,
                "intervals": len(unique),
                "sources": source_provenance,
                "output_artifacts": {
                    "event-index.json": index_digest,
                    OUTPUT_CAPABILITY_FILENAME: capability_digest,
                },
                "source_integrity": "digest_and_schema_verified",
            },
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        _release_empty_output(output_dir)
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            provenance={"sources": source_provenance},
            reason="; ".join(error.errors),
        )
    except Exception as error:  # noqa: BLE001 - defensive API boundary
        _release_empty_output(output_dir)
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            provenance={"sources": source_provenance},
            reason=f"internal_error: {type(error).__name__}",
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-events component."""
    parser = argparse.ArgumentParser(description="Index event/phase intervals with links.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def _result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize a result using the shared component-result.v1 envelope.

    Returns:
        JSON-safe component-result.v1 document.
    """
    return {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)}


def _cli_failure_result(payload: Any, reason: str) -> ComponentResult:
    """Build a safe failure result when request parsing cannot complete.

    Returns:
        Failed component result with safe identity fields.
    """
    if isinstance(payload, dict):
        request_id = (
            payload.get("request_id") if isinstance(payload.get("request_id"), str) else "unknown"
        )
        component_id = (
            payload.get("component_id")
            if isinstance(payload.get("component_id"), str)
            else COMPONENT_ID
        )
    else:
        request_id = "unknown"
        component_id = COMPONENT_ID
    return ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=STATUS_FAILED,
        reason=reason,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-events component.

    Returns:
        Process exit code (0 only when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    payload: Any = None
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReviewContractsValidationError(["request must be a JSON object"])
        if args.config is not None:
            try:
                config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
            if not isinstance(config, dict):
                raise ReviewContractsValidationError(["config must be a JSON object"])
            request_config = payload.get("config", {})
            if not isinstance(request_config, dict):
                raise ReviewContractsValidationError(["request config must be a JSON object"])
            payload = {**payload, "config": {**request_config, **config}}
        payload = {**payload, "output_directory": args.output}
        request = component_request_from_dict(payload, source=args.input)
        result = run(request, base=Path(args.base) if args.base is not None else None)
    except ReviewContractsValidationError as error:
        result = _cli_failure_result(payload, "; ".join(error.errors))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        result = _cli_failure_result(payload, f"invalid_request: {type(error).__name__}")
    except Exception as error:  # noqa: BLE001 - defensive CLI boundary
        result = _cli_failure_result(payload, f"internal_error: {type(error).__name__}")
    print(json.dumps(_result_document(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
