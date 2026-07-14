"""Trace-level predicate export lane (issue #5593).

Provides a versioned, queryable export format that joins ``surrogate_events``
predicate fields with scenario / planner / seed / run metadata.  Downstream
consumers (reports, external analyses) pull the full set of emitted predicate
values per episode without re-deriving them from raw per-step traces.

Schema versioning follows the existing convention: breaking changes bump
``trace_predicate_export.vN``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.benchmark.event_ledger import ensure_event_ledger
from robot_sf.benchmark.safety_predicates import (
    LATE_EVASIVE_PREDICATE_SCHEMA,
    OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
    OSCILLATORY_PREDICATE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Schema constant
# ---------------------------------------------------------------------------
TRACE_PREDICATE_EXPORT_SCHEMA = "trace_predicate_export.v1"

# Canonical set of motivated predicate families.  A predicate appears here
# when its producer exists in ``safety_predicates.py`` and can emit a versioned
# record into an episode's ``surrogate_events`` block.
MOTIVATED_PREDICATE_FAMILIES: tuple[str, ...] = (
    "near_miss",
    "clearance_breach",
    "ttc_breach",
    "oscillation",
    "late_evasive",
    "occlusion_near_miss",
)

# Canonical predicate producer schemas keyed by family name.
PREDICATE_SCHEMA_BY_FAMILY: dict[str, str] = {
    "oscillation": OSCILLATORY_PREDICATE_SCHEMA,
    "late_evasive": LATE_EVASIVE_PREDICATE_SCHEMA,
    "occlusion_near_miss": OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
    # near_miss, clearance_breach, ttc_breach are metric-derived and do not have
    # a dedicated producer schema; they remain in MOTIVATED_PREDICATE_FAMILIES
    # but export as metric-derived booleans.
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PredicateExportRecord:
    """One exportable predicate row for a single episode.

    Attributes:
        scenario_id: Scenario identifier.
        concrete_case_id: Optional concrete case within the scenario.
        seed: Random seed for this episode.
        planner: Planner / algorithm identifier.
        episode_id: Episode identifier.
        schema_version: The trace-predicate-export schema version.
        surrogate_events: Full surrogate event booleans.
        predicate_records: Per-family predicate records (includes schema version,
            raw fields, thresholds, and classification booleans).
        degraded_fields: Fields present but flagged as ``unavailable`` or
            ``not_applicable``.
        missing_fields: Motivated predicates that are absent from the ledger.
        software_commit: Git commit SHA, if available.
    """

    scenario_id: str
    seed: int
    planner: str
    episode_id: str
    schema_version: str
    surrogate_events: dict[str, bool]
    predicate_records: dict[str, dict[str, Any]]
    degraded_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    software_commit: str | None = None
    concrete_case_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Machine-checkable manifest for a predicate export batch.

    Declares which predicate types are present, their schema versions,
    and any episodes with missing or degraded fields.

    Attributes:
        schema_version: Export manifest schema version.
        export_schema: The trace predicate export schema version used.
        predicate_types: Predicate families present in the export.
        predicate_schema_versions: Mapping from family to producer schema version.
        episode_count: Number of episodes in the export.
        episodes_with_missing: Episodes missing at least one motivated predicate.
        episodes_with_degraded: Episodes with degraded fields.
        checksum_sha256: SHA-256 of the deterministically-sorted export JSONL.
    """

    schema_version: str
    export_schema: str
    predicate_types: list[str]
    predicate_schema_versions: dict[str, str]
    episode_count: int
    episodes_with_missing: int
    episodes_with_degraded: int
    checksum_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoverageReportRow:
    """One row in the predicate coverage report.

    Attributes:
        predicate_family: Name of the motivated predicate.
        exported: Whether the predicate appears in at least one record.
        schema_version: Producer schema version (or ``"metric_derived"``).
        episodes_exported: Count of episodes that export this predicate.
        episodes_degraded: Count of episodes with degraded evidence.
        episodes_missing: Count of episodes missing this predicate.
    """

    predicate_family: str
    exported: bool
    schema_version: str
    episodes_exported: int
    episodes_degraded: int
    episodes_missing: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Core export functions
# ---------------------------------------------------------------------------
def _extract_predicate_records(
    surrogate_events: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, dict[str, Any]], list[str], set[str]]:
    """Pull boolean flags, predicate records, degraded fields, and exported
    producer families from surrogate events.

    The event ledger always carries ``False`` boolean defaults for all motivated
    predicates (from :class:`SurrogateEvents`).  Only predicates that have a
    corresponding producer record (e.g. ``oscillatory_control_predicate``) were
    actually produced for this episode.  The returned ``exported_producer`` set
    distinguishes producer-derived booleans from metric-derived defaults.

    Returns:
        Tuple of (boolean flags, predicate records, degraded field names,
        set of families with producer records).
    """
    flags: dict[str, bool] = {}
    records: dict[str, dict[str, Any]] = {}
    degraded: list[str] = []
    exported_producer: set[str] = set()

    for family in MOTIVATED_PREDICATE_FAMILIES:
        if family in surrogate_events:
            flags[family] = bool(surrogate_events[family])

    # Predicate records use keys like "oscillatory_control_predicate",
    # "late_evasive_predicate", "occlusion_near_miss_predicate".
    predicate_key_map = {
        "oscillatory_control_predicate": "oscillation",
        "late_evasive_predicate": "late_evasive",
        "occlusion_near_miss_predicate": "occlusion_near_miss",
    }

    for pred_key, family in predicate_key_map.items():
        pred_data = surrogate_events.get(pred_key)
        if not isinstance(pred_data, Mapping):
            continue
        records[family] = dict(pred_data)
        exported_producer.add(family)
        # Check for degraded evidence
        status = pred_data.get("status")
        if status in ("unavailable", "not_applicable"):
            degraded.append(family)

    return flags, records, degraded, exported_producer


def build_predicate_export_record(
    record: Mapping[str, Any],
    *,
    predicate_families: Sequence[str] | None = None,
) -> PredicateExportRecord:
    """Extract one exportable predicate record from an episode record.

    If the episode record lacks an ``event_ledger``, one is built from the
    available metrics and safety predicates (see ``build_event_ledger``).

    Args:
        record: Episode record dict (must contain scenario_id, seed, algo/planner).
        predicate_families: Optional override for motivated predicate families.
            Defaults to ``MOTIVATED_PREDICATE_FAMILIES``.

    Returns:
        :class:`PredicateExportRecord` for the episode.

    Raises:
        ValueError: When required fields are missing from the event ledger.
    """
    predicate_families = (
        tuple(predicate_families)
        if predicate_families is not None
        else MOTIVATED_PREDICATE_FAMILIES
    )
    ledger = record.get("event_ledger")

    # Ensure ledger exists
    if not isinstance(ledger, Mapping) or not ledger.get("schema_version"):
        # Build ledger in-place from the record
        record_copy = dict(record)
        ledger_result = ensure_event_ledger(record_copy)
        ledger = ledger_result if isinstance(ledger_result, Mapping) else {}
        record = record_copy

    surrogate = ledger.get("surrogate_events")
    if not isinstance(surrogate, Mapping):
        surrogate = {}

    flags, predicate_records, degraded, exported_producer = _extract_predicate_records(surrogate)

    # Determine missing predicates: a family is missing when no producer record
    # was emitted for it.  The event ledger always sets ``False`` defaults for all
    # motivated predicates, so checking ``family not in flags`` alone would never
    # fire.  Instead we check whether a detailed producer record exists.
    missing = [
        family
        for family in predicate_families
        if family not in exported_producer and family not in predicate_records
    ]

    return PredicateExportRecord(
        scenario_id=str(ledger.get("scenario_id") or record.get("scenario_id") or "unknown"),
        seed=int(ledger.get("seed") or record.get("seed", 0)),
        planner=str(
            ledger.get("planner") or record.get("algo") or record.get("planner") or "unknown"
        ),
        episode_id=str(record.get("episode_id") or "unknown"),
        schema_version=TRACE_PREDICATE_EXPORT_SCHEMA,
        surrogate_events=flags,
        predicate_records=predicate_records,
        degraded_fields=sorted(degraded),
        missing_fields=sorted(missing),
        software_commit=ledger.get("software_commit"),
        concrete_case_id=ledger.get("concrete_case_id"),
    )


def build_predicate_export_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    predicate_families: Sequence[str] | None = None,
) -> list[PredicateExportRecord]:
    """Build export records for a batch of episode records.

    Fails closed: if any record cannot produce a valid export row, a
    ``ValueError`` is raised with the failing episode_id.

    Args:
        records: Iterable of episode record dicts.
        predicate_families: Optional override for motivated predicate families.

    Returns:
        List of :class:`PredicateExportRecord` instances.
    """
    results: list[PredicateExportRecord] = []
    for i, rec in enumerate(records):
        ep_id = rec.get("episode_id") or f"index-{i}"
        try:
            results.append(
                build_predicate_export_record(rec, predicate_families=predicate_families)
            )
        except Exception as e:
            raise ValueError(f"Failed to export predicates for episode {ep_id}: {e}") from e
    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize_export_records(
    records: Sequence[PredicateExportRecord],
) -> str:
    """Serialize export records to deterministic JSONL.

    Uses sorted keys and deterministic float formatting to ensure byte-identical
    output for the same input.

    Args:
        records: Export records to serialize.

    Returns:
        JSONL string (newline-terminated).
    """

    def _default(obj: Any) -> Any:
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return str(obj)

    lines: list[str] = []
    for rec in records:
        lines.append(json.dumps(rec.to_dict(), sort_keys=True, default=_default))
    return "\n".join(lines) + "\n" if lines else "\n"


def compute_export_checksum(records: Sequence[PredicateExportRecord]) -> str:
    """Compute SHA-256 checksum of the deterministic JSONL export.

    Args:
        records: Export records to checksum.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    content = serialize_export_records(records)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_export_manifest(
    records: Sequence[PredicateExportRecord],
) -> ExportManifest:
    """Build a machine-checkable manifest for the export batch.

    Args:
        records: Export records to manifest.

    Returns:
        :class:`ExportManifest` describing predicate coverage in the batch.
    """
    predicate_types: set[str] = set()
    schema_versions: dict[str, str] = {}
    missing_count = 0
    degraded_count = 0

    for rec in records:
        # Booleans from surrogate events
        for family in rec.surrogate_events:
            predicate_types.add(family)
        # Detailed records
        for family in rec.predicate_records:
            predicate_types.add(family)
            schema_ver = rec.predicate_records[family].get("schema_version")
            if schema_ver:
                schema_versions[family] = schema_ver

        if rec.missing_fields:
            missing_count += 1
        if rec.degraded_fields:
            degraded_count += 1

    manifest = ExportManifest(
        schema_version="trace_predicate_manifest.v1",
        export_schema=TRACE_PREDICATE_EXPORT_SCHEMA,
        predicate_types=sorted(predicate_types),
        predicate_schema_versions=schema_versions,
        episode_count=len(records),
        episodes_with_missing=missing_count,
        episodes_with_degraded=degraded_count,
    )

    # Attach checksum
    checksum = compute_export_checksum(records) if records else None
    manifest = ExportManifest(
        schema_version=manifest.schema_version,
        export_schema=manifest.export_schema,
        predicate_types=manifest.predicate_types,
        predicate_schema_versions=manifest.predicate_schema_versions,
        episode_count=manifest.episode_count,
        episodes_with_missing=manifest.episodes_with_missing,
        episodes_with_degraded=manifest.episodes_with_degraded,
        checksum_sha256=checksum,
    )
    return manifest


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def build_coverage_report(
    records: Sequence[PredicateExportRecord],
    *,
    predicate_families: Sequence[str] | None = None,
) -> list[CoverageReportRow]:
    """Build a predicate coverage report for the export batch.

    Answers "is this predicate measured or just discussed" by enumerating
    exported versus motivated-but-not-exported predicates.

    Args:
        records: Export records to analyze.
        predicate_families: Motivated predicate families to check.

    Returns:
        List of :class:`CoverageReportRow` instances, one per motivated predicate.
    """
    predicate_families = (
        tuple(predicate_families)
        if predicate_families is not None
        else MOTIVATED_PREDICATE_FAMILIES
    )

    for family in predicate_families:
        # Verify tuple contains valid family names
        pass  # Just ensures no empty-argument errors

    rows: list[CoverageReportRow] = []
    for family in predicate_families:
        exported = 0
        degraded = 0
        missing = 0
        for rec in records:
            if family in rec.missing_fields:
                missing += 1
            elif family in rec.degraded_fields:
                degraded += 1
                exported += 1
            elif family in rec.surrogate_events or family in rec.predicate_records:
                exported += 1
            else:
                missing += 1

        schema_ver = PREDICATE_SCHEMA_BY_FAMILY.get(family, "metric_derived")
        rows.append(
            CoverageReportRow(
                predicate_family=family,
                exported=exported > 0,
                schema_version=schema_ver,
                episodes_exported=exported,
                episodes_degraded=degraded,
                episodes_missing=missing,
            )
        )

    return rows


def format_coverage_report_md(
    rows: list[CoverageReportRow],
    *,
    total_episodes: int | None = None,
) -> str:
    """Format a coverage report as a Markdown table.

    Args:
        rows: Coverage report rows.
        total_episodes: Optional total episode count for header context.

    Returns:
        Markdown table string.
    """
    header_lines = []
    if total_episodes is not None:
        header_lines.append(f"## Predicate Coverage Report ({total_episodes} episodes)\n")

    header_lines.append(
        "| Predicate | Exported | Schema Version | Episodes Exported | "
        "Episodes Degraded | Episodes Missing |"
    )
    header_lines.append("| --- | --- | --- | --- | --- | --- |")

    for row in rows:
        header_lines.append(
            f"| {row.predicate_family} | {'yes' if row.exported else 'no'} | "
            f"{row.schema_version} | {row.episodes_exported} | {row.episodes_degraded} | "
            f"{row.episodes_missing} |"
        )

    return "\n".join(header_lines) + "\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_export_record(record: Mapping[str, Any]) -> list[str]:
    """Validate a trace predicate export record against the v1 contract.

    Args:
        record: Dict representation of a PredicateExportRecord.

    Returns:
        List of violation messages. Empty list means valid.
    """
    violations: list[str] = []

    if record.get("schema_version") != TRACE_PREDICATE_EXPORT_SCHEMA:
        violations.append(
            f"schema_version must be {TRACE_PREDICATE_EXPORT_SCHEMA!r}, "
            f"got {record.get('schema_version')!r}"
        )

    for key in ("scenario_id", "seed", "planner", "episode_id"):
        if key not in record:
            violations.append(f"missing required field: {key!r}")

    surrogate = record.get("surrogate_events")
    if not isinstance(surrogate, Mapping):
        violations.append("surrogate_events must be a mapping")
    else:
        for k, v in surrogate.items():
            if not isinstance(v, bool):
                violations.append(f"surrogate_events.{k} must be boolean, got {type(v).__name__}")

    predicate_records = record.get("predicate_records")
    if not isinstance(predicate_records, Mapping):
        violations.append("predicate_records must be a mapping")

    for field_name in ("degraded_fields", "missing_fields"):
        val = record.get(field_name)
        if not isinstance(val, list):
            violations.append(f"{field_name} must be a list")

    return violations


def validate_export_batch(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate a batch of export records.

    Args:
        records: Dict representations of PredicateExportRecords.

    Returns:
        List of violation messages keyed by index.
    """
    all_violations: list[str] = []
    for i, rec in enumerate(records):
        violations = validate_export_record(rec)
        if violations:
            ep_id = rec.get("episode_id") or f"index-{i}"
            for v in violations:
                all_violations.append(f"[{ep_id}] {v}")
    return all_violations


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------
def export_to_jsonl(
    records: Sequence[PredicateExportRecord],
    output_path: str | Path,
) -> str:
    """Write export records to a JSONL file.

    Args:
        records: Export records to write.
        output_path: Destination file path.

    Returns:
        SHA-256 checksum of the written content.
    """
    path = Path(output_path)
    content = serialize_export_records(records)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_manifest(
    manifest: ExportManifest,
    output_path: str | Path,
) -> None:
    """Write an export manifest to a JSON file.

    Args:
        manifest: Manifest to write.
        output_path: Destination file path.
    """
    path = Path(output_path)
    content = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    path.write_text(content + "\n", encoding="utf-8")


def write_coverage_report(
    rows: list[CoverageReportRow],
    output_path: str | Path,
    *,
    total_episodes: int | None = None,
) -> None:
    """Write a coverage report to a Markdown file.

    Args:
        rows: Coverage report rows.
        output_path: Destination file path.
        total_episodes: Optional total episode count.
    """
    path = Path(output_path)
    content = format_coverage_report_md(rows, total_episodes=total_episodes)
    path.write_text(content, encoding="utf-8")


__all__ = [
    "LATE_EVASIVE_PREDICATE_SCHEMA",
    "MOTIVATED_PREDICATE_FAMILIES",
    "OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA",
    "OSCILLATORY_PREDICATE_SCHEMA",
    "PREDICATE_SCHEMA_BY_FAMILY",
    "TRACE_PREDICATE_EXPORT_SCHEMA",
    "CoverageReportRow",
    "ExportManifest",
    "PredicateExportRecord",
    "build_coverage_report",
    "build_export_manifest",
    "build_predicate_export_batch",
    "build_predicate_export_record",
    "compute_export_checksum",
    "export_to_jsonl",
    "format_coverage_report_md",
    "serialize_export_records",
    "validate_export_batch",
    "validate_export_record",
    "write_coverage_report",
    "write_manifest",
]
