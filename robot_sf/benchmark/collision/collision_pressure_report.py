"""Deterministic collision-event counts over a declared typed-ledger slice.

This module consumes existing ``EpisodeEventLedger.v2`` episode rows. It does
not define collision semantics, execute campaigns, or estimate probability,
severity, causality, or physical risk. The caller declares the scenario-family
slice and provenance identity; incomplete rows remain explicit exclusions and
cannot silently enter the denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.event_ledger import (
    COLLISION_PARTNER_TYPES,
    EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
    reconcile_event_ledger,
)

COLLISION_PRESSURE_REPORT_SCHEMA_VERSION = "collision_pressure_report.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBSTACLE_PARTNER_TYPES = frozenset({"static_geometry", "boundary", "goal_artifact"})
_CSV_FIELDS = ("metric", "value", "unit", "status")


class CollisionPressureReportError(ValueError):
    """Raised when a declared typed-ledger slice cannot be audited safely."""


def build_collision_pressure_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligible_families: Sequence[str],
    source_commit: str,
    release_id: str,
    bundle_id: str,
    input_checksums: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic collision-pressure report.

    Args:
        rows: Episode rows carrying an ``event_ledger`` mapping or a ledger
            mapping directly. Each eligible row must carry ``episode_id`` (or
            ``episode_key``) and ``scenario_family``.
        eligible_families: Caller-declared scenario-family selection.
        source_commit: Code commit that produced the source rows.
        release_id: Release or bundle release identity.
        bundle_id: Durable input-bundle identity.
        input_checksums: Named SHA-256 checksums for the input artifacts.

    Returns:
        A JSON-safe ``collision_pressure_report.v1`` mapping.

    Raises:
        CollisionPressureReportError: If the requested slice has no auditable
            eligible rows, duplicate episode keys, or invalid provenance.
    """
    families = _normalise_families(eligible_families)
    _require_identity(source_commit, "source_commit")
    _require_identity(release_id, "release_id")
    _require_identity(bundle_id, "bundle_id")
    checksums = _normalise_checksums(input_checksums)

    selected, exclusions = _select_rows(rows, families)

    duplicate_keys = _duplicates(key for key, _, _ in selected)
    if duplicate_keys:
        raise CollisionPressureReportError(
            "duplicate eligible episode keys prevent an exact denominator: "
            + ", ".join(duplicate_keys)
        )
    if not selected:
        reasons = sorted({str(item["reason"]) for item in exclusions})
        detail = "; ".join(reasons) if reasons else "no rows matched eligible_families"
        raise CollisionPressureReportError(f"no auditable eligible episodes: {detail}")

    aggregate = _aggregate_selected(selected)
    contact_keys = aggregate["contact_keys"]
    obstacle_contact_keys = aggregate["obstacle_contact_keys"]
    partner_episode_counts = aggregate["partner_episode_counts"]
    family_counts = aggregate["family_counts"]
    missing_optional_fields = aggregate["missing_optional_fields"]
    total_collision_events = aggregate["total_collision_events"]
    overlap_counts = aggregate["overlap_counts"]

    denominator_keys = sorted(key for key, _, _ in selected)
    denominator_digest = _json_digest(denominator_keys)
    exclusion_counts = Counter(str(item["reason"]) for item in exclusions)
    report = {
        "schema_version": COLLISION_PRESSURE_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "claim_boundary": (
            "Descriptive exact collision-event counts for the caller-declared typed-ledger slice. "
            "This is not a collision probability, causal mechanism, severity, physical-risk, "
            "or real-world safety metric."
        ),
        "selection": {
            "eligible_families": families,
            "family_field": "scenario_family",
            "input_row_count": len(rows),
            "excluded_row_count": len(exclusions),
            "exclusions": sorted(exclusions, key=lambda item: (item["row_index"], item["reason"])),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "denominator": {
            "unit": "episode",
            "eligible_episode_count": len(selected),
            "eligible_episode_key_sha256": denominator_digest,
            "eligible_episode_keys": denominator_keys,
        },
        "counts": {
            "contact_episode_count": len(contact_keys),
            "collision_event_count": total_collision_events,
            "partner_type_episode_counts": {
                partner_type: partner_episode_counts.get(partner_type, 0)
                for partner_type in COLLISION_PARTNER_TYPES
            },
            "obstacle_rollup_episode_count": len(obstacle_contact_keys),
            "pedestrian_obstacle_overlap_episode_counts": dict(overlap_counts),
        },
        "family_counts": {family: family_counts[family] for family in sorted(family_counts)},
        "missingness": {
            "optional_collision_event_fields": dict(sorted(missing_optional_fields.items())),
            "excluded_rows_by_reason": dict(sorted(exclusion_counts.items())),
        },
        "provenance": {
            "source_commit": source_commit.strip(),
            "release_id": release_id.strip(),
            "bundle_id": bundle_id.strip(),
            "input_checksums": checksums,
        },
    }
    return report


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[str],
) -> tuple[list[tuple[str, str, Mapping[str, Any]]], list[dict[str, Any]]]:
    """Select auditable rows and retain explicit exclusions.

    Returns:
        A selected ``(episode_key, family, ledger)`` list and exclusion records.
    """
    selected: list[tuple[str, str, Mapping[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            exclusions.append(_exclusion(index, None, "row_not_mapping"))
            continue
        family = _text(row.get("scenario_family"))
        if family is None:
            exclusions.append(_exclusion(index, _episode_key(row), "missing_scenario_family"))
            continue
        if family not in families:
            continue
        key = _episode_key(row)
        if key is None:
            exclusions.append(_exclusion(index, None, "missing_episode_key"))
            continue
        ledger = _ledger_from_row(row)
        reason = _ledger_exclusion_reason(ledger)
        if reason is not None:
            exclusions.append(_exclusion(index, key, reason))
            continue
        if ledger is None:
            raise CollisionPressureReportError("selected row is missing an auditable event ledger")
        selected.append((key, family, ledger))
    return selected, exclusions


def _aggregate_selected(selected: Sequence[tuple[str, str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Aggregate selected ledgers without changing event semantics.

    Returns:
        Internal aggregate sets, counters, and family counts used by the report.
    """
    contact_keys: set[str] = set()
    obstacle_contact_keys: set[str] = set()
    partner_episode_counts: Counter[str] = Counter()
    family_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"eligible_episode_count": 0, "contact_episode_count": 0}
    )
    missing_optional_fields: Counter[str] = Counter()
    total_collision_events = 0
    overlap_counts = Counter(
        {"pedestrian_only": 0, "obstacle_only": 0, "pedestrian_and_obstacle": 0}
    )
    for key, family, ledger in selected:
        family_counts[family]["eligible_episode_count"] += 1
        events = ledger["collision_events"]
        if not events:
            continue
        contact_keys.add(key)
        family_counts[family]["contact_episode_count"] += 1
        total_collision_events += len(events)
        partner_types = {str(event["collision_partner_type"]) for event in events}
        for partner_type in partner_types:
            partner_episode_counts[partner_type] += 1
        has_pedestrian = "pedestrian" in partner_types
        has_obstacle = bool(partner_types & _OBSTACLE_PARTNER_TYPES)
        if has_obstacle:
            obstacle_contact_keys.add(key)
        overlap_key = (
            "pedestrian_and_obstacle"
            if has_pedestrian and has_obstacle
            else "pedestrian_only"
            if has_pedestrian
            else "obstacle_only"
        )
        overlap_counts[overlap_key] += 1
        for event in events:
            for field in ("collision_partner_id", "relative_speed_at_contact"):
                if event.get(field) is None:
                    missing_optional_fields[field] += 1
    return {
        "contact_keys": contact_keys,
        "obstacle_contact_keys": obstacle_contact_keys,
        "partner_episode_counts": partner_episode_counts,
        "family_counts": family_counts,
        "missing_optional_fields": missing_optional_fields,
        "total_collision_events": total_collision_events,
        "overlap_counts": overlap_counts,
    }


def build_compact_report_row(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build one stable report row for tables or CSV consumers.

    Returns:
        A compact mapping containing only descriptive report fields.
    """
    counts = report["counts"]
    partners = counts["partner_type_episode_counts"]
    overlap = counts["pedestrian_obstacle_overlap_episode_counts"]
    denominator = report["denominator"]
    selection = report["selection"]
    provenance = report["provenance"]
    return {
        "schema_version": report["schema_version"],
        "source_commit": provenance["source_commit"],
        "release_id": provenance["release_id"],
        "bundle_id": provenance["bundle_id"],
        "eligible_episode_count": denominator["eligible_episode_count"],
        "contact_episode_count": counts["contact_episode_count"],
        "pedestrian_contact_episode_count": partners["pedestrian"],
        "obstacle_contact_episode_count": counts["obstacle_rollup_episode_count"],
        "pedestrian_obstacle_overlap_episode_count": overlap["pedestrian_and_obstacle"],
        "excluded_row_count": selection["excluded_row_count"],
        "claim_boundary": report["claim_boundary"],
    }


def write_collision_pressure_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, Path]:
    """Write deterministic JSON and compact CSV report outputs.

    Returns:
        Paths to the written JSON and CSV files.
    """
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row = build_compact_report_row(report)
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    return {"json": json_target, "csv": csv_target}


def _ledger_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return a ledger mapping from an episode row."""
    candidate = row.get("event_ledger")
    if isinstance(candidate, Mapping):
        return candidate
    if row.get("schema_version") == EPISODE_EVENT_LEDGER_SCHEMA_VERSION:
        return row
    return None


def _ledger_exclusion_reason(ledger: Mapping[str, Any] | None) -> str | None:
    """Return an explicit exclusion reason for an incomplete ledger."""
    if ledger is None:
        return "missing_event_ledger"
    if ledger.get("schema_version") != EPISODE_EVENT_LEDGER_SCHEMA_VERSION:
        return "unsupported_event_ledger_schema"
    violations = reconcile_event_ledger(ledger)
    if violations:
        return "event_ledger_reconciliation_failed"
    reconciliation = ledger.get("reconciliation")
    if not isinstance(reconciliation, Mapping) or reconciliation.get("audit_result") != "pass":
        return "event_ledger_not_audited"
    exact = ledger.get("exact_events")
    events = ledger.get("collision_events")
    if not isinstance(exact, Mapping) or not isinstance(events, list):
        return "missing_collision_fields"
    collision = exact.get("collision")
    if not isinstance(collision, bool):
        return "invalid_collision_flag"
    if collision and not events:
        return "collision_event_records_missing"
    if not collision and events:
        return "collision_event_without_exact_collision"
    return None


def _episode_key(row: Mapping[str, Any]) -> str | None:
    """Return an explicit episode identity."""
    return _text(row.get("episode_key")) or _text(row.get("episode_id"))


def _exclusion(row_index: int, episode_key: str | None, reason: str) -> dict[str, Any]:
    """Build one deterministic exclusion record.

    Returns:
        A row-indexed exclusion mapping.
    """
    return {"row_index": row_index, "episode_key": episode_key, "reason": reason}


def _normalise_families(values: Sequence[str]) -> list[str]:
    """Normalize and require the caller-declared family set.

    Returns:
        Sorted unique family names.
    """
    families = sorted({_text(value) for value in values if _text(value) is not None})
    if not families:
        raise CollisionPressureReportError("eligible_families must contain at least one family")
    return families


def _normalise_checksums(values: Mapping[str, str]) -> dict[str, str]:
    """Normalize and validate named input checksums.

    Returns:
        Sorted checksum mapping.
    """
    if not values:
        raise CollisionPressureReportError("input_checksums must not be empty")
    result: dict[str, str] = {}
    for name, checksum in sorted(values.items()):
        key = _text(name)
        value = _text(checksum)
        if key is None or value is None or _SHA256_RE.fullmatch(value) is None:
            raise CollisionPressureReportError(f"invalid SHA-256 input checksum for {name!r}")
        result[key] = value
    return result


def _require_identity(value: str, field: str) -> None:
    """Require a non-empty provenance identity."""
    if not _text(value):
        raise CollisionPressureReportError(f"{field} must be non-empty")


def _text(value: Any) -> str | None:
    """Return a stripped non-empty string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _duplicates(values: Sequence[str] | Any) -> list[str]:
    """Return sorted duplicate values from an iterable."""
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _json_digest(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-safe value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL episode rows as mappings.

    Returns:
        Parsed JSON object rows.
    """
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CollisionPressureReportError(
                f"invalid JSON on line {line_number} of {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise CollisionPressureReportError(f"JSONL line {line_number} is not an object")
        rows.append(payload)
    return rows


def _parse_checksums(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``name=sha256`` CLI arguments.

    Returns:
        Parsed checksum mapping.
    """
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise CollisionPressureReportError("--input-checksum must use name=sha256")
        name, checksum = item.split("=", 1)
        parsed[name] = checksum
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the collision-pressure report CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="Input episode JSONL path.")
    parser.add_argument(
        "--eligible-family",
        action="append",
        required=True,
        help="Caller-declared scenario family; repeat for multiple families.",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--input-checksum", action="append", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report CLI and return a shell-friendly exit code.

    Returns:
        Zero on success, or two when the report is blocked.
    """
    args = build_arg_parser().parse_args(argv)
    try:
        report = build_collision_pressure_report(
            _load_jsonl(args.rows),
            eligible_families=args.eligible_family,
            source_commit=args.source_commit,
            release_id=args.release_id,
            bundle_id=args.bundle_id,
            input_checksums=_parse_checksums(args.input_checksum),
        )
        paths = write_collision_pressure_report(
            report,
            json_path=args.json_out,
            csv_path=args.csv_out,
        )
    except (OSError, CollisionPressureReportError) as exc:
        sys.stderr.write(f"collision-pressure report blocked: {exc}\n")
        return 2
    sys.stdout.write(f"collision-pressure report written: {paths['json']} and {paths['csv']}\n")
    return 0


__all__ = [
    "COLLISION_PRESSURE_REPORT_SCHEMA_VERSION",
    "CollisionPressureReportError",
    "build_arg_parser",
    "build_collision_pressure_report",
    "build_compact_report_row",
    "main",
    "write_collision_pressure_report",
]
