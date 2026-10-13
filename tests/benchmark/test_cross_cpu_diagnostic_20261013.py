"""Meaningful controls for the reconstructed scientific-row comparison."""

from __future__ import annotations

import json

import pytest
import yaml

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


def test_float64_config_binds_both_actors_without_changing_kernel_only_arm(tmp_path):
    """Float64 opt-in must agree at campaign and each learned-arm boundary."""
    study.prepare(tmp_path)
    pinned = yaml.safe_load((tmp_path / "pinned.yaml").read_text())
    numeric = yaml.safe_load((tmp_path / "float64.yaml").read_text())
    assert "numerical_mode" not in pinned
    assert numeric["numerical_mode"] == "pinned_float64_v1"
    assert pinned["seed_policy"]["seeds"] == list(range(1001, 1031))
    for arm in (p for p in numeric["planners"] if p["key"] in {"ppo", "guarded_ppo"}):
        generated = yaml.safe_load(open(arm["algo_config"], encoding="utf-8"))
        assert generated.pop("numerical_mode") == "pinned_float64_v1"
        original = next(p for p in pinned["planners"] if p["key"] == arm["key"])
        assert generated == yaml.safe_load((study.ROOT / original["algo_config"]).read_text())


def test_altered_seed_input_is_refused_before_launch(tmp_path, monkeypatch):
    """Do not let edited generated inputs bypass the study's dev seed contract."""
    study.prepare(tmp_path)
    path = tmp_path / "default.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["seed_policy"]["seeds"][0] = 1000
    path.write_text(yaml.safe_dump(payload))

    def refused_launch(*args, **kwargs):
        raise AssertionError("episode launch reached before input guard")

    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setattr(study, "identity", lambda: {})
    monkeypatch.setattr(study.subprocess, "Popen", refused_launch)
    with pytest.raises(ValueError, match="authorized development inventory"):
        study.run(tmp_path / "acquisition", tmp_path, selected_campaign="default-r1")
