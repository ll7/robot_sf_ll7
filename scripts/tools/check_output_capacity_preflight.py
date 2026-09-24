#!/usr/bin/env python3
"""Fail-closed output-capacity and post-run preservation preflight (issue #8841).

Estimates lower/expected/conservative-upper bounds for bytes, files/inodes, peak
temporary space, checkpoint volume, logs, harvest manifests, checksums, compression
workspace, and destination transfer volume from one versioned packet, then checks the
conservative bounds plus a reserved safety margin against one sanitized
storage-capability projection. Task output, scheduler logs, temporary scratch,
durable-required artifacts, and disposable post-verification data stay separate, and
every empirical size basis must name a compatible source/config/output-schema identity;
declared or unavailable values remain explicit. ``capacity_ok`` requires a full
conservative fit; conservative misses are ``capacity_exceeded``; unbounded or
unavailable scaling, workspace, inode, destination, transfer-rate, or deadline inputs
are ``capacity_unknown`` and never pass. Check-only: the tool reads two JSON documents,
never deletes artifacts, and never mutates campaigns. CLI: ``--check --packet <json>
--storage-projection <json> [--format json|text]``; exit codes 0 capacity_ok, 2
capacity_exceeded or capacity_unknown, 3 malformed input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.identity.hash_utils import stable_hash  # noqa: E402
from scripts.tools.classify_scheduler_failure import sanitize_text  # noqa: E402
from scripts.tools.validate_receipt_projection import private_content_reason  # noqa: E402

PACKET_SCHEMA = "robot_sf.output_capacity_preflight_packet.v1"
PROJECTION_SCHEMA = "robot_sf.storage_capability_projection.v1"
REPORT_SCHEMA = "robot_sf.output_capacity_preflight.v1"
CLAIM_BOUNDARY = (
    "Operational capacity estimate only. Conservative bounds and sanitized storage "
    "projections never assert transfer completion, artifact custody, benchmark eligibility, "
    "or scientific success; this check-only tool deletes nothing and mutates no campaign."
)
STATUS_OK, STATUS_EXCEEDED, STATUS_UNKNOWN = (
    "capacity_ok",
    "capacity_exceeded",
    "capacity_unknown",
)
STORAGE_CLASSES = (
    "task_output",
    "scheduler_log",
    "temporary_scratch",
    "durable_required",
    "disposable_post_verification",
)
OUTPUT_KINDS = (
    "rows",
    "logs",
    "checkpoints",
    "harvest_manifest",
    "checksums",
    "compression_workspace",
    "temporary_workspace",
    "other",
)
DESTINATION_CLASSES = frozenset(("task_output", "scheduler_log", "durable_required"))
BASES = ("empirical", "declared", "unavailable")
COMPATIBILITIES = ("compatible", "incompatible")
UNCERTAINTIES = ("bounded", "unbounded", "unknown")
ROW_SCALINGS = ("bounded", "unbounded", "unavailable")
EXPLICIT_TOKENS = frozenset(
    ("", "unavailable", "redacted", "not_observed", "not_applicable", "none", "null")
)
EXCEEDED_CODES = frozenset(
    (
        "source_bytes_exceeded",
        "source_inodes_exceeded",
        "destination_bytes_exceeded",
        "destination_inodes_exceeded",
        "transfer_deadline_missed",
    )
)
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PACKET_KEYS = frozenset(
    ("schema", "packet_id", "expected_rows", "row_scaling", "components", "transfer")
)
COMPONENT_KEYS = frozenset(
    "component_id storage_class output_kind basis source_identity compatibility uncertainty "
    "bytes files peak_bytes".split()
)
DIMENSION_KEYS = frozenset(("per_row", "fixed"))
TRANSFER_KEYS = frozenset(("task_completion_utc", "task_completion_latest_utc"))
PROJECTION_KEYS = frozenset(
    "schema projection_id generated_at source destination safety_margin access_deadline_utc "
    "transfer".split()
)
MARGIN_KEYS = frozenset(("bytes", "inodes", "time_seconds"))
PROJECTION_TRANSFER_KEYS = frozenset(
    ("route_identity", "rate_bytes_per_second", "rate_uncertainty")
)


@dataclass(frozen=True, slots=True)
class Issue:
    """One sanitized, fail-closed issue that never carries a private value."""

    code: str
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class Bounds:
    """One lower/expected/conservative-upper estimate triple."""

    lower: int
    expected: int
    upper: int

    def scaled(self, rows: int) -> Bounds:
        """Multiply the triple by a bounded row count."""
        return Bounds(self.lower * rows, self.expected * rows, self.upper * rows)

    def plus(self, other: Bounds) -> Bounds:
        """Add two triples term by term."""
        return Bounds(
            self.lower + other.lower, self.expected + other.expected, self.upper + other.upper
        )

    def as_dict(self) -> dict[str, int]:
        """Return the serializable triple."""
        return {"lower": self.lower, "expected": self.expected, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class Dimension:
    """A per-row plus fixed term pair for one estimated quantity."""

    per_row: Bounds
    fixed: Bounds

    def totals(self, rows: int) -> Bounds:
        """Resolve the dimension for a bounded row count."""
        return self.per_row.scaled(rows).plus(self.fixed)


@dataclass(frozen=True, slots=True)
class Component:
    """One output component with provenance, class separation, and size bounds."""

    component_id: str
    storage_class: str
    output_kind: str
    basis: str
    source_identity: str | None
    compatibility: str
    uncertainty: str
    bytes: Dimension | None
    files: Dimension | None
    peak_bytes: Dimension | None


@dataclass(frozen=True, slots=True)
class Packet:
    """One parsed capacity packet; unavailable fields stay ``None``."""

    packet_id: str
    expected_rows: int | None
    row_scaling: str
    components: tuple[Component, ...]
    task_completion_utc: datetime | None
    task_completion_latest_utc: datetime | None


@dataclass(frozen=True, slots=True)
class Projection:
    """One parsed sanitized storage-capability projection; unknown stays ``None``."""

    projection_id: str
    source: dict[str, Any]
    destination: dict[str, Any]
    margins: dict[str, int | None]
    access_deadline_utc: datetime | None
    route_identity: str | None
    rate: Bounds | None
    rate_uncertainty: str


def _issue(code: str, location: str, message: str) -> Issue:
    return Issue(code=code, location=location, message=message)


def _slug(value: Any) -> str | None:
    """Accept a public-safe schema slug; reject paths, hosts, or private identities."""
    if isinstance(value, str) and SLUG_RE.fullmatch(value) and not private_content_reason(value):
        return value
    return None


def _is_explicit(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in EXPLICIT_TOKENS


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or _is_explicit(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _render_timestamp(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _scan_unsanitized(value: Any, location: str, issues: list[Issue]) -> None:
    """Flag private paths, hosts, identities, credentials, or signed URLs in inputs."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _scan_unsanitized(item, f"{location}/{sanitize_text(key, limit=80)}", issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_unsanitized(item, f"{location}[{index}]", issues)
    elif isinstance(value, str) and (reason := private_content_reason(value)):
        issues.append(_issue("unsanitized_input", location, reason))


def _check_keys(
    raw: Mapping[str, Any],
    allowed: frozenset[str],
    required: Sequence[str],
    location: str,
    issues: list[Issue],
) -> None:
    for key in sorted(set(raw) - allowed):
        issues.append(
            _issue("unknown_field", f"{location}/{sanitize_text(key, limit=80)}", "not in schema")
        )
    for key in sorted(set(required) - set(raw)):
        issues.append(_issue("missing_field", f"{location}/{key}", "required"))


def _parse_bounds(
    value: Any,
    location: str,
    issues: list[Issue],
    *,
    allow_explicit: bool = True,
    positive: bool = False,
) -> Bounds | None:
    """Parse one ``[lower, expected, upper]`` triple, or an explicit token."""
    if _is_explicit(value):
        if not allow_explicit:
            issues.append(_issue("invalid_bounds", location, "explicit token is not allowed here"))
        return None
    floor = 1 if positive else 0
    ordered = (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= floor for item in value
        )
        and value[0] <= value[1] <= value[2]
    )
    if not ordered:
        issues.append(_issue("invalid_bounds", location, "expected ordered non-negative integers"))
        return None
    return Bounds(*value)


def _parse_dimension(value: Any, location: str, issues: list[Issue]) -> Dimension | None:
    if _is_explicit(value):
        issues.append(_issue("bounds_unavailable", location, "dimension declared unavailable"))
        return None
    if not isinstance(value, Mapping):
        issues.append(_issue("invalid_bounds", location, "dimension mapping required"))
        return None
    if set(value) - DIMENSION_KEYS:
        issues.append(_issue("unknown_field", location, "unknown dimension term"))
    per_row = _parse_bounds(
        value.get("per_row"), f"{location}/per_row", issues, allow_explicit=False
    )
    fixed = _parse_bounds(value.get("fixed"), f"{location}/fixed", issues, allow_explicit=False)
    return None if per_row is None or fixed is None else Dimension(per_row=per_row, fixed=fixed)


def _component_identity(  # noqa: C901 - one validation pass
    raw: Mapping[str, Any], location: str, issues: list[Issue]
) -> dict[str, Any] | None:
    fields = (
        ("component_id", _slug(raw.get("component_id")), None, "slug required"),
        ("storage_class", raw.get("storage_class"), STORAGE_CLASSES, "unknown storage class"),
        ("output_kind", raw.get("output_kind"), OUTPUT_KINDS, "unknown output kind"),
        ("basis", raw.get("basis"), BASES, "unknown basis"),
        ("uncertainty", raw.get("uncertainty"), UNCERTAINTIES, "unknown uncertainty"),
    )
    values: dict[str, Any] = {}
    valid = True
    for name, value, allowed, message in fields:
        if value is None or (allowed is not None and value not in allowed):
            valid = False
            issues.append(_issue("invalid_field", f"{location}/{name}", message))
        values[name] = value
    basis, uncertainty = values["basis"], values["uncertainty"]
    compatibility = raw.get("compatibility")
    if basis == "unavailable":
        issues.append(_issue("basis_unavailable", f"{location}/basis", "no empirical basis"))
    if basis == "empirical":
        if _slug(raw.get("source_identity")) is None:
            issues.append(
                _issue(
                    "missing_source_identity",
                    f"{location}/source_identity",
                    "empirical basis requires a source identity",
                )
            )
        if compatibility != "compatible":
            issues.append(
                _issue(
                    "incompatible_source_identity",
                    f"{location}/compatibility",
                    "source identity is not compatible",
                )
            )
    elif compatibility is not None and compatibility not in COMPATIBILITIES:
        issues.append(_issue("invalid_field", f"{location}/compatibility", "unknown compatibility"))
    if uncertainty == "unbounded":
        issues.append(
            _issue("uncertainty_unbounded", f"{location}/uncertainty", "unbounded output")
        )
    elif uncertainty == "unknown":
        issues.append(_issue("uncertainty_unknown", f"{location}/uncertainty", "unknown output"))
    if not valid:
        return None
    return {
        **values,
        "source_identity": _slug(raw.get("source_identity")),
        "compatibility": compatibility if compatibility in COMPATIBILITIES else "unavailable",
    }


def _parse_component(raw: Any, location: str, issues: list[Issue]) -> Component | None:
    if not isinstance(raw, Mapping):
        issues.append(_issue("invalid_field", location, "mapping required"))
        return None
    _check_keys(
        raw,
        COMPONENT_KEYS,
        COMPONENT_KEYS - {"source_identity", "compatibility"},
        location,
        issues,
    )
    identity = _component_identity(raw, location, issues)
    if identity is None:
        return None
    return Component(
        **identity,
        bytes=_parse_dimension(raw.get("bytes"), f"{location}/bytes", issues),
        files=_parse_dimension(raw.get("files"), f"{location}/files", issues),
        peak_bytes=_parse_dimension(raw.get("peak_bytes"), f"{location}/peak_bytes", issues),
    )


def _parse_packet_transfer(
    payload: Mapping[str, Any], location: str, issues: list[Issue]
) -> tuple[datetime | None, datetime | None]:
    transfer = payload.get("transfer")
    if not isinstance(transfer, Mapping):
        issues.append(_issue("invalid_field", f"{location}/transfer", "mapping required"))
        transfer = {}
    else:
        _check_keys(transfer, TRANSFER_KEYS, TRANSFER_KEYS, f"{location}/transfer", issues)
    completion = _parse_timestamp(transfer.get("task_completion_utc"))
    completion_latest = _parse_timestamp(transfer.get("task_completion_latest_utc"))
    for key, parsed in (
        ("task_completion_utc", completion),
        ("task_completion_latest_utc", completion_latest),
    ):
        if parsed is None:
            issues.append(
                _issue("task_completion_unavailable", f"{location}/transfer/{key}", "required")
            )
    return completion, completion_latest


def _parse_row_scaling(
    payload: Mapping[str, Any], location: str, issues: list[Issue]
) -> tuple[str, int | None]:
    row_scaling = payload.get("row_scaling")
    if row_scaling not in ROW_SCALINGS:
        issues.append(_issue("invalid_field", f"{location}/row_scaling", "unknown scaling"))
        return "unavailable", None
    expected_rows = _int_or_none(payload.get("expected_rows"))
    if row_scaling == "bounded" and (expected_rows is None or expected_rows <= 0):
        issues.append(
            _issue("invalid_field", f"{location}/expected_rows", "positive integer required")
        )
        return row_scaling, None
    if row_scaling in ("unbounded", "unavailable"):
        issues.append(
            _issue(
                f"row_scaling_{row_scaling}",
                f"{location}/row_scaling",
                f"row count {row_scaling}",
            )
        )
    return row_scaling, expected_rows


def _parse_packet(payload: Mapping[str, Any], issues: list[Issue]) -> Packet:
    location = "/packet"
    _check_keys(payload, PACKET_KEYS, PACKET_KEYS, location, issues)
    if payload.get("schema") != PACKET_SCHEMA:
        issues.append(_issue("invalid_schema", f"{location}/schema", f"must equal {PACKET_SCHEMA}"))
    packet_id = _slug(payload.get("packet_id"))
    if packet_id is None:
        issues.append(_issue("invalid_field", f"{location}/packet_id", "slug required"))
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        issues.append(_issue("invalid_field", f"{location}/components", "list required"))
        raw_components = []
    components = [
        parsed
        for index, raw in enumerate(raw_components)
        if (parsed := _parse_component(raw, f"{location}/components[{index}]", issues)) is not None
    ]
    if not components:
        issues.append(_issue("no_components", f"{location}/components", "no usable component"))
    row_scaling, expected_rows = _parse_row_scaling(payload, location, issues)
    completion, completion_latest = _parse_packet_transfer(payload, location, issues)
    components.sort(key=lambda item: item.component_id)
    return Packet(
        packet_id=packet_id or "unavailable",
        expected_rows=expected_rows,
        row_scaling=row_scaling,
        components=tuple(components),
        task_completion_utc=completion,
        task_completion_latest_utc=completion_latest,
    )


def _parse_capacity(raw: Any, location: str, issues: list[Issue]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        issues.append(_issue("invalid_field", location, "mapping required"))
        raw = {}
    free_bytes, free_inodes = (
        _int_or_none(raw.get("free_bytes")),
        _int_or_none(raw.get("free_inodes")),
    )
    retention_class = _slug(raw.get("retention_class"))
    if free_bytes is None or free_inodes is None:
        issues.append(_issue("capacity_unavailable", location, "free bytes or inodes unavailable"))
    if retention_class is None:
        issues.append(
            _issue("retention_class_unavailable", location, "retention class unavailable")
        )
    return {
        "free_bytes": free_bytes,
        "free_inodes": free_inodes,
        "retention_class": retention_class,
    }


def _parse_projection(payload: Mapping[str, Any], issues: list[Issue]) -> Projection:  # noqa: C901
    location = "/storage_projection"
    _check_keys(payload, PROJECTION_KEYS, PROJECTION_KEYS, location, issues)
    if payload.get("schema") != PROJECTION_SCHEMA:
        issues.append(
            _issue("invalid_schema", f"{location}/schema", f"must equal {PROJECTION_SCHEMA}")
        )
    projection_id = _slug(payload.get("projection_id"))
    if projection_id is None:
        issues.append(_issue("invalid_field", f"{location}/projection_id", "slug required"))
    if _parse_timestamp(payload.get("generated_at")) is None:
        issues.append(_issue("invalid_field", f"{location}/generated_at", "timestamp required"))
    margins: dict[str, int | None] = {}
    margin = payload.get("safety_margin")
    if not isinstance(margin, Mapping):
        issues.append(_issue("safety_margin_unavailable", f"{location}/safety_margin", "required"))
        margin = {}
    else:
        _check_keys(margin, MARGIN_KEYS, MARGIN_KEYS, f"{location}/safety_margin", issues)
    for key in MARGIN_KEYS:
        value = _int_or_none(margin.get(key))
        margins[key] = value if value is not None and value >= 0 else None
        if margins[key] is None:
            issues.append(
                _issue("safety_margin_unavailable", f"{location}/safety_margin/{key}", "required")
            )
    deadline = _parse_timestamp(payload.get("access_deadline_utc"))
    if deadline is None:
        issues.append(
            _issue("access_deadline_unavailable", f"{location}/access_deadline_utc", "required")
        )
    route = payload.get("transfer")
    if not isinstance(route, Mapping):
        issues.append(_issue("invalid_field", f"{location}/transfer", "mapping required"))
        route = {}
    else:
        _check_keys(
            route,
            PROJECTION_TRANSFER_KEYS,
            PROJECTION_TRANSFER_KEYS,
            f"{location}/transfer",
            issues,
        )
    route_identity = _slug(route.get("route_identity"))
    if route_identity is None:
        issues.append(
            _issue("transfer_route_unavailable", f"{location}/transfer/route_identity", "required")
        )
    rate = _parse_bounds(
        route.get("rate_bytes_per_second"),
        f"{location}/transfer/rate_bytes_per_second",
        issues,
        positive=True,
    )
    if rate is None:
        issues.append(_issue("transfer_rate_unavailable", f"{location}/transfer", "required"))
    rate_uncertainty = route.get("rate_uncertainty")
    if rate_uncertainty != "bounded":
        label = "unknown" if rate_uncertainty == "unknown" else "unavailable"
        issues.append(
            _issue(
                "transfer_rate_uncertainty_unknown", f"{location}/transfer/rate_uncertainty", label
            )
        )
    return Projection(
        projection_id=projection_id or "unavailable",
        source=_parse_capacity(payload.get("source"), f"{location}/source", issues),
        destination=_parse_capacity(payload.get("destination"), f"{location}/destination", issues),
        margins=margins,
        access_deadline_utc=deadline,
        route_identity=route_identity,
        rate=rate,
        rate_uncertainty=rate_uncertainty if isinstance(rate_uncertainty, str) else "unavailable",
    )


def _component_totals(component: Component, rows: int | None) -> dict[str, Bounds | None]:
    if rows is None:
        return {"bytes": None, "files": None, "peak_bytes": None}
    return {
        name: dimension.totals(rows) if dimension else None
        for name, dimension in (
            ("bytes", component.bytes),
            ("files", component.files),
            ("peak_bytes", component.peak_bytes),
        )
    }


def _sum_bounds(values: Sequence[Bounds | None]) -> Bounds | None:
    if not values or any(value is None for value in values):
        return None
    total = Bounds(0, 0, 0)
    for value in values:
        total = total.plus(value)
    return total


def _group_totals(entries: Sequence[tuple[str, dict[str, Bounds | None]]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Bounds | None]]] = {}
    for name, totals in entries:
        groups.setdefault(name, []).append(totals)
    return {
        name: {
            "component_count": len(rows),
            **{
                key: _sum_bounds([row[key] for row in rows])
                for key in ("bytes", "files", "peak_bytes")
            },
        }
        for name, rows in sorted(groups.items())
    }


def _build_estimate(packet: Packet, issues: list[Issue]) -> dict[str, Any]:
    rows = packet.expected_rows if packet.row_scaling == "bounded" else None
    per_component = [
        (component, _component_totals(component, rows)) for component in packet.components
    ]
    destination = [
        t for component, t in per_component if component.storage_class in DESTINATION_CLASSES
    ]
    return {
        "row_count": rows,
        "row_scaling": packet.row_scaling,
        "totals": {
            "component_count": len(per_component),
            **{
                key: _sum_bounds([t[key] for _, t in per_component])
                for key in ("bytes", "files", "peak_bytes")
            },
        },
        "destination": {
            "storage_classes": sorted(DESTINATION_CLASSES),
            **{
                key: _sum_bounds([t[key] for t in destination])
                for key in ("bytes", "files", "peak_bytes")
            },
        },
        "by_storage_class": _group_totals(
            [(component.storage_class, t) for component, t in per_component]
        ),
        "by_output_kind": _group_totals(
            [(component.output_kind, t) for component, t in per_component]
        ),
        "declared_unavailable": sorted(
            issue.location
            for issue in issues
            if issue.code in {"bounds_unavailable", "basis_unavailable"}
        ),
        "components": [
            {
                "component_id": component.component_id,
                "storage_class": component.storage_class,
                "output_kind": component.output_kind,
                "basis": component.basis,
                "source_identity": component.source_identity,
                "compatibility": component.compatibility,
                "uncertainty": component.uncertainty,
                **{
                    key: t[key].as_dict() if t[key] else None
                    for key in ("bytes", "files", "peak_bytes")
                },
            }
            for component, t in per_component
        ],
    }


def _check_storage(
    estimate: Mapping[str, Any], projection: Projection, issues: list[Issue]
) -> dict[str, Any]:
    totals, destination = estimate["totals"], estimate["destination"]
    checks: dict[str, Any] = {
        "safety_margin": {
            key: projection.margins[key] for key in ("bytes", "inodes", "time_seconds")
        }
    }
    required = {
        "source": (totals["peak_bytes"], totals["files"]),
        "destination": (destination["bytes"], destination["files"]),
    }
    for name, capacity in (("source", projection.source), ("destination", projection.destination)):
        required_bytes, required_files = required[name]
        margin_bytes, margin_inodes = projection.margins["bytes"], projection.margins["inodes"]
        fits_bytes = (
            None
            if None in (capacity["free_bytes"], required_bytes, margin_bytes)
            else capacity["free_bytes"] - margin_bytes >= required_bytes.upper
        )
        fits_inodes = (
            None
            if None in (capacity["free_inodes"], required_files, margin_inodes)
            else capacity["free_inodes"] - margin_inodes >= required_files.upper
        )
        checks[name] = {
            "retention_class": capacity["retention_class"],
            "free_bytes": capacity["free_bytes"],
            "free_inodes": capacity["free_inodes"],
            "required_bytes": required_bytes.as_dict() if required_bytes else None,
            "required_inodes": required_files.as_dict() if required_files else None,
            "fits_bytes": fits_bytes,
            "fits_inodes": fits_inodes,
        }
    for name, dimension, code in (
        ("source", "fits_bytes", "source_bytes_exceeded"),
        ("source", "fits_inodes", "source_inodes_exceeded"),
        ("destination", "fits_bytes", "destination_bytes_exceeded"),
        ("destination", "fits_inodes", "destination_inodes_exceeded"),
    ):
        if checks[name][dimension] is False:
            unit = "bytes" if dimension.endswith("bytes") else "inodes"
            issues.append(
                _issue(
                    code,
                    f"/storage_projection/{name}/free_{unit}",
                    "conservative bound exceeds capacity",
                )
            )
    return checks


def _check_transfer(
    estimate: Mapping[str, Any], projection: Projection, packet: Packet, issues: list[Issue]
) -> dict[str, Any]:
    volume = estimate["destination"]["bytes"]
    window = (
        (projection.access_deadline_utc - packet.task_completion_latest_utc).total_seconds()
        if projection.access_deadline_utc and packet.task_completion_latest_utc
        else None
    )
    durations: dict[str, float | None] = {"lower": None, "expected": None, "upper": None}
    if volume and projection.rate:
        durations = {
            "lower": volume.lower / projection.rate.upper,
            "expected": volume.expected / projection.rate.expected,
            "upper": volume.upper / projection.rate.lower,
        }
    fit = "unknown"
    if (
        window is not None
        and durations["upper"] is not None
        and projection.margins["time_seconds"] is not None
        and projection.rate_uncertainty == "bounded"
    ):
        fits = durations["upper"] + projection.margins["time_seconds"] <= window
        fit = "fits" if fits else "misses"
        if not fits:
            issues.append(
                _issue(
                    "transfer_deadline_missed",
                    "/storage_projection/access_deadline_utc",
                    "conservative transfer exceeds access window",
                )
            )
    return {
        "destination_volume_bytes": volume.as_dict() if volume else None,
        "rate_bytes_per_second": projection.rate.as_dict() if projection.rate else None,
        "rate_uncertainty": projection.rate_uncertainty,
        "route_identity": projection.route_identity,
        "task_completion_utc": _render_timestamp(packet.task_completion_utc),
        "task_completion_latest_utc": _render_timestamp(packet.task_completion_latest_utc),
        "access_deadline_utc": _render_timestamp(projection.access_deadline_utc),
        "window_seconds": window,
        "duration_seconds": durations,
        "deadline_fit": fit,
    }


def _jsonable(value: Any) -> Any:
    """Convert nested bounds triples into plain JSON-compatible structures."""
    if isinstance(value, Bounds):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_report(
    packet_payload: Mapping[str, Any], projection_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one deterministic, sanitized capacity-preflight report."""
    issues: list[Issue] = []
    _scan_unsanitized(packet_payload, "/packet", issues)
    _scan_unsanitized(projection_payload, "/storage_projection", issues)
    packet = _parse_packet(packet_payload, issues)
    projection = _parse_projection(projection_payload, issues)
    estimate = _build_estimate(packet, issues)
    checks = _check_storage(estimate, projection, issues)
    transfer = _check_transfer(estimate, projection, packet, issues)
    deduped = sorted({(issue.code, issue.location, issue.message) for issue in issues})
    status = (
        STATUS_EXCEEDED
        if deduped and all(code in EXCEEDED_CODES for code, _, _ in deduped)
        else STATUS_UNKNOWN
    )
    if not deduped:
        status = STATUS_OK
    return _jsonable(
        {
            "schema": REPORT_SCHEMA,
            "check_only": True,
            "status": status,
            "packet_id": packet.packet_id,
            "projection_id": projection.projection_id,
            "packet_sha256": stable_hash(packet_payload),
            "storage_projection_sha256": stable_hash(projection_payload),
            "claim_boundary": CLAIM_BOUNDARY,
            "estimate": estimate,
            "storage": checks,
            "transfer": transfer,
            "issues": [
                {"code": code, "location": location, "message": message}
                for code, location, message in deduped
            ],
        }
    )


def render_report_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable sorted-key JSON with a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_report_text(payload: Mapping[str, Any]) -> str:
    """Return a concise human explanation of the capacity verdict."""
    estimate, checks, transfer = payload["estimate"], payload["storage"], payload["transfer"]
    lines = [
        f"{payload['status']}: packet={payload['packet_id']} rows={estimate['row_count']}",
        f"- source peak bytes upper={_fmt(estimate['totals']['peak_bytes'])}"
        f" inodes upper={_fmt(estimate['totals']['files'])}",
        f"- destination bytes upper={_fmt(estimate['destination']['bytes'])}",
        f"- source fits={checks['source']['fits_bytes']}/{checks['source']['fits_inodes']}"
        f" destination fits={checks['destination']['fits_bytes']}/{checks['destination']['fits_inodes']}",
        f"- transfer deadline={transfer['deadline_fit']}",
    ]
    if payload["issues"]:
        lines.append(f"- issues: {', '.join(issue['code'] for issue in payload['issues'])}")
    return "\n".join(lines) + "\n"


def _fmt(bounds: Mapping[str, Any] | None) -> str:
    return "unavailable" if bounds is None else str(bounds["upper"])


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the check-only preflight argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate only; never writes")
    parser.add_argument("--packet", type=Path, required=True, help="capacity packet JSON")
    parser.add_argument(
        "--storage-projection", type=Path, required=True, help="sanitized storage projection JSON"
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("document must be a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check-only capacity preflight and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("only --check is supported; this tool never writes or deletes artifacts")
    try:
        packet_payload = _load_json(args.packet)
        projection_payload = _load_json(args.storage_projection)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"FAIL malformed_input: {sanitize_text(f'{type(exc).__name__}: {exc}')}\n")
        return 3
    report = build_report(packet_payload, projection_payload)
    sys.stdout.write(
        render_report_json(report) if args.format == "json" else render_report_text(report)
    )
    return 0 if report["status"] == STATUS_OK else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
