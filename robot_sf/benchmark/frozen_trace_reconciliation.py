"""Frozen-trace before/after reconciliation for episode event ledgers.

This module compares already-materialized ``EpisodeEventLedger.v1`` rows from two frozen
analyzer outputs. It does not rebuild ledgers, rerun benchmarks, or choose metric semantics; it
only reports event-count deltas, changed and unchanged episode rows, and which declared
claim/table/figure artifacts consume changed event fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from robot_sf.benchmark.event_ledger import EPISODE_EVENT_LEDGER_SCHEMA_VERSION

FROZEN_TRACE_RECONCILIATION_SCHEMA = "frozen_trace_event_reconciliation.v1"

_EVENT_BLOCKS = ("exact_events", "surrogate_events")


def _coerce_identifier(value: Any, *, default: str = "unknown") -> str:
    """Return a stable string identifier while preserving valid falsy values."""

    if value is None:
        return default
    text = str(value)
    return text if text else default


def _first_present_identifier(*values: Any, default: str = "unknown") -> str:
    """Return the first non-``None`` identifier candidate as a stable string."""

    for value in values:
        if value is not None:
            return _coerce_identifier(value, default=default)
    return default


def _ledger_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the existing event ledger carried by a frozen row."""

    ledger = (
        row
        if row.get("schema_version") == EPISODE_EVENT_LEDGER_SCHEMA_VERSION
        else row.get("event_ledger")
    )
    if not isinstance(ledger, Mapping):
        raise ValueError("frozen reconciliation rows must contain event_ledger")
    if ledger.get("schema_version") != EPISODE_EVENT_LEDGER_SCHEMA_VERSION:
        raise ValueError("frozen reconciliation rows must contain EpisodeEventLedger.v1 payloads")
    return ledger


def _episode_key(row: Mapping[str, Any], ledger: Mapping[str, Any]) -> str:
    """Build a stable key for pairing frozen before/after rows.

    Returns:
        Pipe-delimited episode identity string.
    """

    scenario_id = _first_present_identifier(ledger.get("scenario_id"), row.get("scenario_id"))
    concrete_case_id = _first_present_identifier(
        ledger.get("concrete_case_id"), row.get("concrete_case_id"), row.get("episode_id")
    )
    seed = _coerce_identifier(
        ledger.get("seed") if ledger.get("seed") is not None else row.get("seed")
    )
    planner = _first_present_identifier(ledger.get("planner"), row.get("planner"), row.get("algo"))
    return f"{scenario_id}|{concrete_case_id}|{seed}|{planner}"


def _event_values(ledger: Mapping[str, Any]) -> dict[str, bool]:
    """Flatten boolean exact and surrogate event fields from a ledger.

    Returns:
        Mapping of event field paths to boolean values.
    """

    values: dict[str, bool] = {}
    for block_name in _EVENT_BLOCKS:
        block = ledger.get(block_name)
        if not isinstance(block, Mapping):
            continue
        for field, value in block.items():
            if isinstance(value, bool):
                values[f"{block_name}.{field}"] = value
    return values


def _index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    """Index frozen rows by episode key with flattened event values.

    Returns:
        Episode-keyed event value mapping.
    """

    indexed: dict[str, dict[str, bool]] = {}
    for row in rows:
        ledger = _ledger_from_row(row)
        key = _episode_key(row, ledger)
        if key in indexed:
            raise ValueError(f"duplicate frozen reconciliation row for episode key: {key}")
        indexed[key] = _event_values(ledger)
    return indexed


def _event_counts(indexed_rows: Mapping[str, Mapping[str, bool]]) -> dict[str, int]:
    """Count true event values across indexed frozen rows.

    Returns:
        Event field counts sorted by field path.
    """

    counts: dict[str, int] = {}
    for events in indexed_rows.values():
        for field, value in events.items():
            counts.setdefault(field, 0)
            if value:
                counts[field] += 1
    return dict(sorted(counts.items()))


def _changed_and_unchanged_rows(
    old_index: Mapping[str, Mapping[str, bool]],
    new_index: Mapping[str, Mapping[str, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Classify paired episode rows by changed or unchanged event fields.

    Returns:
        Changed row details and unchanged row identities.
    """

    changed_rows: list[dict[str, Any]] = []
    unchanged_rows: list[dict[str, str]] = []
    for key in sorted(set(old_index) & set(new_index)):
        old_events = old_index[key]
        new_events = new_index[key]
        changed_fields = [
            field
            for field in sorted(set(old_events) | set(new_events))
            if old_events.get(field, False) != new_events.get(field, False)
        ]
        if not changed_fields:
            unchanged_rows.append({"episode_key": key})
            continue
        changed_rows.append(
            {
                "episode_key": key,
                "changed_event_fields": changed_fields,
                "old_events": {field: old_events.get(field, False) for field in changed_fields},
                "new_events": {field: new_events.get(field, False) for field in changed_fields},
            }
        )
    return changed_rows, unchanged_rows


def _event_field_names(events: Mapping[str, bool]) -> list[str]:
    """Return sorted event field names present on one frozen ledger row."""

    return sorted(events)


def _affected_artifacts(
    artifact_manifest: Iterable[Mapping[str, Any]],
    changed_rows: Iterable[Mapping[str, Any]],
    added_rows: Iterable[Mapping[str, Any]],
    removed_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Classify declared claim/table/figure artifacts by changed consumed event fields.

    Returns:
        Manifest entries annotated with reconciliation status.
    """

    changed_fields_by_row = [
        set(row.get("changed_event_fields", ()))
        for row in changed_rows
        if isinstance(row.get("changed_event_fields"), list)
    ]
    added_fields_by_row = [
        set(row.get("event_fields", ()))
        for row in added_rows
        if isinstance(row.get("event_fields"), list)
    ]
    removed_fields_by_row = [
        set(row.get("event_fields", ()))
        for row in removed_rows
        if isinstance(row.get("event_fields"), list)
    ]
    all_delta_fields = set().union(
        *changed_fields_by_row,
        *added_fields_by_row,
        *removed_fields_by_row,
    )
    has_row_delta = bool(changed_fields_by_row or added_fields_by_row or removed_fields_by_row)
    statuses: list[dict[str, Any]] = []
    for artifact in artifact_manifest:
        consumed = artifact.get("consumes_event_fields", ())
        consumed_fields = (
            {str(field) for field in consumed} if isinstance(consumed, list) else set()
        )
        affected_fields = sorted(consumed_fields & all_delta_fields)
        affected_row_count = sum(
            1 for row_fields in changed_fields_by_row if row_fields & consumed_fields
        )
        affected_added_row_count = sum(
            1 for row_fields in added_fields_by_row if row_fields & consumed_fields
        )
        affected_removed_row_count = sum(
            1 for row_fields in removed_fields_by_row if row_fields & consumed_fields
        )
        if affected_fields:
            status = "affected_reconciliation_required"
        elif has_row_delta:
            status = "unchanged_by_event_delta"
        else:
            status = "unchanged"
        statuses.append(
            {
                "artifact_id": str(artifact.get("artifact_id", "unknown")),
                "artifact_type": str(artifact.get("artifact_type", "unknown")),
                "status": status,
                "consumes_event_fields": sorted(consumed_fields),
                "affected_event_fields": affected_fields,
                "affected_changed_row_count": affected_row_count,
                "affected_added_row_count": affected_added_row_count,
                "affected_removed_row_count": affected_removed_row_count,
            }
        )
    return statuses


def build_frozen_trace_reconciliation_report(
    old_rows: Iterable[Mapping[str, Any]],
    new_rows: Iterable[Mapping[str, Any]],
    *,
    artifact_manifest: Iterable[Mapping[str, Any]] = (),
    old_label: str = "old",
    new_label: str = "new",
) -> dict[str, Any]:
    """Compare frozen old/new event-ledger rows and return a reviewable report.

    Returns:
        JSON-serializable before/after reconciliation report.
    """

    old_index = _index_rows(old_rows)
    new_index = _index_rows(new_rows)
    changed_rows, unchanged_rows = _changed_and_unchanged_rows(old_index, new_index)
    added_keys = sorted(set(new_index) - set(old_index))
    removed_keys = sorted(set(old_index) - set(new_index))
    added_rows = [
        {"episode_key": key, "event_fields": _event_field_names(new_index[key])}
        for key in added_keys
    ]
    removed_rows = [
        {"episode_key": key, "event_fields": _event_field_names(old_index[key])}
        for key in removed_keys
    ]
    return {
        "schema_version": FROZEN_TRACE_RECONCILIATION_SCHEMA,
        "summary": {
            "old_label": old_label,
            "new_label": new_label,
            "old_row_count": len(old_index),
            "new_row_count": len(new_index),
            "changed_row_count": len(changed_rows),
            "unchanged_row_count": len(unchanged_rows),
            "added_row_count": len(added_keys),
            "removed_row_count": len(removed_keys),
        },
        "old_event_counts": _event_counts(old_index),
        "new_event_counts": _event_counts(new_index),
        "changed_rows": changed_rows,
        "unchanged_rows": unchanged_rows,
        "added_rows": added_rows,
        "removed_rows": removed_rows,
        "affected_artifacts": _affected_artifacts(
            artifact_manifest,
            changed_rows,
            added_rows,
            removed_rows,
        ),
        "semantics": {
            "input_contract": "existing EpisodeEventLedger.v1 rows only",
            "benchmark_semantics_changed": False,
            "claim_promotion": "none",
        },
    }


__all__ = [
    "FROZEN_TRACE_RECONCILIATION_SCHEMA",
    "build_frozen_trace_reconciliation_report",
]
