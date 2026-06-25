"""Tests for the metric-semantics reconciliation table (issue #3482)."""

from __future__ import annotations

from robot_sf.benchmark.event_ledger import build_event_ledger
from robot_sf.benchmark.event_ledger_reconciliation import (
    RECONCILIATION_TABLE_SCHEMA,
    build_metric_semantics_table,
)

_EXPECTED_FIELDS = {
    "collision_count",
    "near_miss",
    "clearance_breach",
    "ttc_breach",
    "oscillation",
}


def _record(*, collision_event: bool = False, collisions: float = 0.0) -> dict:
    """Build a minimal episode record (mirrors the event-ledger test fixture)."""
    return {
        "scenario_id": "s",
        "seed": 0,
        "algo": "demo",
        "metrics": {
            "success": 0.0 if collision_event else 1.0,
            "collisions": collisions,
        },
        "termination_reason": "collision" if collision_event else "success",
        "outcome": {
            "route_complete": not collision_event,
            "collision_event": collision_event,
        },
    }


def test_table_has_one_row_per_reported_field() -> None:
    """The table must expose a provenance row for every reported metric field."""
    ledger = build_event_ledger(_record(collision_event=True, collisions=1.0))
    table = build_metric_semantics_table(ledger)

    assert table["schema_version"] == RECONCILIATION_TABLE_SCHEMA
    assert {row["field"] for row in table["rows"]} == _EXPECTED_FIELDS
    for row in table["rows"]:
        assert row["level"] == "episode"
        assert row["kind"] in {"exact_or_sampled", "derived"}
        assert row["producer"]
        assert isinstance(row["representative_consumers"], list)


def test_collision_count_is_exact_or_sampled_with_a_producer() -> None:
    """The collision field must be exact/sampled and name its producing metric."""
    ledger = build_event_ledger(_record(collision_event=True, collisions=1.0))
    table = build_metric_semantics_table(ledger)

    collision_row = next(r for r in table["rows"] if r["field"] == "collision_count")
    assert collision_row["kind"] == "exact_or_sampled"
    assert collision_row["producer"] != "missing"


def test_clean_ledger_reconciles() -> None:
    """A consistent ledger must report a passing audit and no violations."""
    ledger = build_event_ledger(_record(collision_event=False, collisions=0.0))
    table = build_metric_semantics_table(ledger)

    assert table["reconciles"] is True
    assert table["violations"] == []
    assert table["audit_result"] == "pass"


def test_exact_collision_with_zero_metric_surfaces_violation() -> None:
    """An exact collision with zero collision metric must surface, not be silently resolved."""
    ledger = build_event_ledger(_record(collision_event=True, collisions=1.0))
    ledger["reconciliation"]["collision_metric_value"] = 0.0
    table = build_metric_semantics_table(ledger)

    assert table["reconciles"] is False
    assert any("collision" in violation for violation in table["violations"])
