"""Tests for the canonical episode event ledger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.event_ledger import (
    EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
    build_event_ledger,
    reconcile_event_ledger,
    validate_record_event_ledger,
)
from robot_sf.benchmark.map_runner_jsonl import write_validated_to_handle
from robot_sf.benchmark.runner import validate_and_write

if TYPE_CHECKING:
    from pathlib import Path


def _record(*, collision_event: bool = False, collisions: float = 0.0) -> dict:
    """Return a minimal episode record with outcome and metric fields."""
    return {
        "version": "v1",
        "episode_id": "episode-1",
        "scenario_id": "scenario-1",
        "seed": 7,
        "algo": "goal",
        "git_hash": "abc123",
        "metrics": {
            "success": 0.0 if collision_event else 1.0,
            "collisions": collisions,
            "near_misses": 1.0,
            "min_clearance": 0.2,
        },
        "termination_reason": "collision" if collision_event else "success",
        "outcome": {
            "route_complete": not collision_event,
            "collision_event": collision_event,
            "timeout_event": False,
        },
    }


def _schema(path: Path) -> Path:
    """Write a permissive schema for writer-path tests."""
    schema_path = path / "episode.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": True,
            }
        ),
        encoding="utf-8",
    )
    return schema_path


def test_build_event_ledger_records_exact_and_surrogate_events() -> None:
    """Ledger payloads should expose exact outcomes and derived event definitions."""
    ledger = build_event_ledger(_record(collision_event=True, collisions=1.0))

    assert ledger["schema_version"] == EPISODE_EVENT_LEDGER_SCHEMA_VERSION
    assert ledger["scenario_id"] == "scenario-1"
    assert ledger["seed"] == 7
    assert ledger["planner"] == "goal"
    assert ledger["software_commit"] == "abc123"
    assert ledger["exact_events"]["collision"] is True
    assert ledger["exact_events"]["goal_reached"] is False
    assert ledger["surrogate_events"]["near_miss"] is True
    assert ledger["metric_definitions"]["collision_count"]["source"] == "metrics.collisions"
    assert ledger["reconciliation"]["audit_result"] == "pass"


def test_build_event_ledger_parses_string_booleans_without_truthiness_leaks() -> None:
    """String booleans from JSON-adjacent records should not rely on Python truthiness."""
    record = _record(collision_event=False, collisions=0.0)
    record["outcome"]["collision_event"] = "false"
    record["outcome"]["timeout_event"] = "0"
    record["outcome"]["route_complete"] = "true"

    ledger = build_event_ledger(record)

    assert ledger["exact_events"]["collision"] is False
    assert ledger["exact_events"]["timeout"] is False
    assert ledger["exact_events"]["goal_reached"] is True


def test_build_event_ledger_defaults_malformed_seed() -> None:
    """Malformed seed values should not break the validation write path."""
    record = _record(collision_event=False, collisions=0.0)
    record["seed"] = "not-an-int"

    assert build_event_ledger(record)["seed"] == 0


def test_reconcile_event_ledger_surfaces_exact_collision_metric_disagreement() -> None:
    """An exact collision event must not silently coexist with zero collision metrics."""
    record = _record(collision_event=True, collisions=1.0)
    ledger = build_event_ledger(record)
    ledger["reconciliation"]["collision_metric_value"] = 0.0

    assert reconcile_event_ledger(ledger) == ["exact collision event requires collision metric > 0"]


def test_reconcile_event_ledger_rejects_goal_and_invalid_run_overlap() -> None:
    """Goal and invalid-run exact events are mutually exclusive."""
    ledger = build_event_ledger(_record(collision_event=False, collisions=0.0))
    ledger["exact_events"]["invalid_run"] = True

    assert reconcile_event_ledger(ledger) == ["goal_reached and invalid_run are mutually exclusive"]


def test_reconcile_event_ledger_requires_metric_definitions() -> None:
    """Every ledger metric family should carry a definition entry."""
    ledger = build_event_ledger(_record(collision_event=False, collisions=0.0))
    del ledger["metric_definitions"]["ttc_breach"]

    assert reconcile_event_ledger(ledger) == ["missing metric definitions: ttc_breach"]


def test_validate_record_event_ledger_attaches_canonical_payload() -> None:
    """Validation should materialize the canonical ledger on legacy episode records."""
    record = _record(collision_event=False, collisions=0.0)

    assert validate_record_event_ledger(record) == []
    assert record["event_ledger"]["schema_version"] == EPISODE_EVENT_LEDGER_SCHEMA_VERSION


def test_jsonl_writer_rejects_collision_ledger_disagreement(tmp_path: Path) -> None:
    """The write path should fail closed when a supplied ledger contradicts exact events."""
    record = _record(collision_event=True, collisions=1.0)
    record["event_ledger"] = build_event_ledger(record)
    record["event_ledger"]["reconciliation"]["collision_metric_value"] = 0.0

    with (tmp_path / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        with pytest.raises(ValueError, match="exact collision event requires collision metric > 0"):
            write_validated_to_handle(handle, json.loads(_schema(tmp_path).read_text()), record)


def test_runner_validate_and_write_attaches_event_ledger(tmp_path: Path) -> None:
    """The classic runner writer should persist a canonical event ledger on legacy records."""
    record = _record(collision_event=False, collisions=0.0)
    out_path = tmp_path / "episodes.jsonl"

    validate_and_write(record, _schema(tmp_path), out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["event_ledger"]["schema_version"] == EPISODE_EVENT_LEDGER_SCHEMA_VERSION
    assert payload["event_ledger"]["exact_events"]["goal_reached"] is True
