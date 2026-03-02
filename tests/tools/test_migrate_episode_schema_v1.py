"""Tests for episode schema v1 migration helper."""

from __future__ import annotations

from scripts.tools import migrate_episode_schema_v1


def test_migrate_record_backfills_new_required_v1_fields() -> None:
    """Legacy records should gain required v1 outcome/termination/integrity fields."""
    record = {
        "scenario_params": {"id": "legacy_scenario"},
        "metrics": {"success": 0.0, "collisions": 1.0},
        "status": "collision",
        "created_at": 1_700_000_000.0,
    }
    migrated = migrate_episode_schema_v1._migrate_record(record)
    assert migrated["version"] == "v1"
    assert migrated["scenario_id"] == "legacy_scenario"
    assert migrated["seed"] == 0
    assert migrated["episode_id"] == "legacy_scenario::seed=0"
    assert migrated["termination_reason"] == "collision"
    assert migrated["outcome"] == {
        "route_complete": False,
        "collision_event": True,
        "timeout_event": False,
    }
    assert isinstance(migrated["integrity"]["contradictions"], list)
    assert "timestamps" in migrated


def test_migrate_record_preserves_existing_contract_fields() -> None:
    """Records that already satisfy the new contract should keep their explicit values."""
    record = {
        "version": "v1",
        "episode_id": "ep-1",
        "scenario_id": "s1",
        "seed": 7,
        "metrics": {"success": 1.0, "collisions": 0.0},
        "termination_reason": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "integrity": {"contradictions": ["keep-me"]},
    }
    migrated = migrate_episode_schema_v1._migrate_record(record)
    assert migrated["episode_id"] == "ep-1"
    assert migrated["scenario_id"] == "s1"
    assert migrated["seed"] == 7
    assert migrated["termination_reason"] == "success"
    assert migrated["outcome"]["route_complete"] is True
    assert migrated["integrity"]["contradictions"] == ["keep-me"]
