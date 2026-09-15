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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
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
    return asdict(_DESCRIPTOR)


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


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies it.
    """
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
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReviewContractsValidationError(
            [f"source uri rejected (absolute or traversal): {path_value}"]
        )
    return base / candidate


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
    return (
        _IndexedInterval(
            interval_id=interval_id,
            kind=kind,
            start_s=start,
            end_s=end,
            actor_ids=[str(a) for a in actor_ids],
            category=str(item.get("category", "")),
            metric_value=item.get("metric_value"),
            precursor_ids=[str(p) for p in precursors],
            recovery_ids=[str(r) for r in recoveries],
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
    refs: list[Any], family: str, root: Path, diagnostics: list[str]
) -> list[dict[str, Any]]:
    """Load and parse all sources of one family.

    Returns:
        Parsed raw interval dicts.
    """
    items: list[dict[str, Any]] = []
    for ref in refs:
        try:
            raw = _resolve_source(ref.uri, root).read_bytes()
        except OSError:
            diagnostics.append(f"{ref.artifact_id}: source_unreadable")
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            diagnostics.append(f"{ref.artifact_id}: source_not_json")
            continue
        entries = payload.get("intervals", payload)
        if not isinstance(entries, list):
            diagnostics.append(f"{ref.artifact_id}: intervals_not_a_list")
            continue
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
    for family in REQUIRED_CAPABILITIES:
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
) -> list[_IndexedInterval]:
    """Normalize, clip, and deduplicate intervals across families.

    Returns:
        Unique clipped intervals (unsorted).
    """
    intervals: list[_IndexedInterval] = []
    for family in REQUIRED_CAPABILITIES + OPTIONAL_CAPABILITIES:
        for ref in by_format.get(family, []):
            raw_items = _load_family([ref], family, root, diagnostics)
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
    seen: set[str] = set()
    unique: list[_IndexedInterval] = []
    for interval in intervals:
        if interval.interval_id in seen:
            diagnostics.append(f"duplicate_interval_id:{interval.interval_id}")
            continue
        seen.add(interval.interval_id)
        unique.append(interval)
    return unique


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
    root = base if base is not None else Path.cwd()
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=f"output_collision: already exists: {request.output_directory}",
        )
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
    try:
        by_format = _collect_family_refs(request, diagnostics)
        if by_format is None:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="required_source_family_missing: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        unique = _normalize_families(by_format, root, t0, terminal, diagnostics)
        if not unique:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="no_intervals_indexed: " + "; ".join(sorted(set(diagnostics))[:5]),
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
        capability_payload = {
            "missing_capabilities": [],
            "diagnostics": sorted(set(diagnostics)),
        }
        index_digest = _write_json(output_dir / "event-index.json", document)
        _write_json(output_dir / OUTPUT_CAPABILITY_FILENAME, capability_payload)
        informational = {"optional_stream_skipped", "clipped_to_range"}
        blocking = [item for item in diagnostics if not any(tag in item for tag in informational)]
        partial = bool(blocking)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = (
            (
                {
                    "artifact_id": "event-index.json",
                    "uri": str(Path(request.output_directory) / "event-index.json"),
                    "sha256": index_digest,
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
            },
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="; ".join(error.errors),
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-events component."""
    parser = argparse.ArgumentParser(description="Index event/phase intervals with links.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-events component.

    Returns:
        Process exit code (0 only when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
