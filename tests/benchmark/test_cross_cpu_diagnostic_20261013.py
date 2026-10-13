"""Meaningful controls for the reconstructed scientific-row comparison."""

from __future__ import annotations

import json

import pytest

from scripts.validation import cross_cpu_dispatch_20261013 as study


def episode():
    """Provide one known development coordinate with explicit event outcomes."""
    return {
        "scenario_id": "test_scenario",
        "seed": 1001,
        "steps": 20,
        "metrics": {"collisions": 2, "energy": 1.0},
        "metric_values": {"energy": 1.0},
        "status": "ok",
        "termination_reason": "collision",
        "outcome": {"route_complete": False, "collision_event": True, "timeout_event": False},
    }


def test_collision_counts_are_scientific_values_not_event_outcomes():
    """Two collision counts can differ while the native event outcome agrees."""
    first, second = episode(), episode()
    second["metrics"]["collisions"] = 3
    assert study.canonical(first["metrics"]) != study.canonical(second["metrics"])
    assert study.outcome(first) == study.outcome(second)


def test_termination_difference_is_detected_without_metric_change():
    """Do not infer outcomes solely from matching metric dictionaries."""
    first, second = episode(), episode()
    second["termination_reason"] = "max_steps"
    second["outcome"] = {"route_complete": False, "collision_event": False, "timeout_event": True}
    assert first["metrics"] == second["metrics"]
    assert study.outcome(first) != study.outcome(second)


def test_canonical_object_order_keeps_float_differences():
    """Object-order changes agree; adjacent float values remain different."""
    assert study.canonical({"a": 1, "b": 2}) == study.canonical({"b": 2, "a": 1})
    assert study.canonical({"x": 1.0}) != study.canonical({"x": 1.0000000000000002})
    assert study.canonical({"x": 0.0}) != study.canonical({"x": -0.0})


def test_missing_outcome_fails_closed():
    """Legacy/incomplete records cannot be counted as identical outcomes."""
    row = episode()
    del row["outcome"]
    with pytest.raises(ValueError, match="explicit native outcome"):
        study.outcome(row)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "wrong_seed"])
def test_bad_coordinate_inventory_is_rejected(tmp_path, fault):
    """Reject incomplete, duplicated, or unexpected paired episode inventories."""
    cfg = {"arms": ["goal"], "scenarios": ["test_scenario"], "seed_start": 1001, "seed_stop": 1001}
    destination = tmp_path / "campaigns/default-r1/runs/goal__differential_drive/episodes.jsonl"
    destination.parent.mkdir(parents=True)
    row = episode()
    if fault == "wrong_seed":
        row["seed"] = 1002
    lines = [] if fault == "missing" else [json.dumps(row)] * (2 if fault == "duplicate" else 1)
    destination.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="duplicate|missing|outside"):
        study.records(tmp_path, "default-r1", cfg)
