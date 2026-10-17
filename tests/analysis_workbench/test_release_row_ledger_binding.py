"""Collision reconciliation cannot be borrowed from another episode."""

from copy import deepcopy

import pytest

from robot_sf.analysis_workbench import release_row_anomalies as checker
from robot_sf.benchmark.event_ledger import EPISODE_EVENT_LEDGER_SCHEMA_VERSION


def _fixture(monkeypatch):
    monkeypatch.setattr(checker, "_release_0_0_8_roster", lambda: {"goal-slot"})
    row = {
        "scenario_id": "ledger-binding",
        "seed": 1001,
        "algo": "goal",
        "_release_arm": "goal-slot",
        "steps": 100,
        "git_hash": "commit-a",
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "metrics": {
            "ped_collision_count": 0,
            "obstacle_collision_count": 0,
            "agent_collision_count": 0,
            "total_collision_count": 0,
            "collisions": 0,
        },
        "integrity": {"effective_view": {"observation_ped_count": 1}},
        "event_ledger": {
            "schema_version": EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
            "scenario_id": "ledger-binding",
            "seed": 1001,
            "planner": "goal",
            "software_commit": "commit-a",
            "exact_events": {"collision": False, "invalid_run": False},
            "reconciliation": {
                "collision_metric_value": 0,
                "collision_metric_source": "metrics.total_collision_count",
            },
        },
    }
    config = {
        "baseline_planner": "goal-slot",
        "pedestrian_aware_planners": [],
        "require_preflight": False,
        "min_planners_per_cell": 1,
        "collision_metric_contract": "release_0_0_8",
        "collision_roster_status": "frozen",
        "collision_expected_arm_count": 1,
        "max_unannotated_findings": 100,
    }
    return row, config


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("scenario_id", "another-scenario"),
        ("seed", 1002),
        ("seed", "1001"),
        ("seed", 1001.0),
        ("planner", "orca"),
        ("software_commit", "commit-b"),
    ],
)
def test_reconciled_ledger_from_wrong_episode_is_blocked(monkeypatch, field, wrong):
    """Equal collision numbers do not authorize a mismatched typed ledger."""
    row, config = _fixture(monkeypatch)
    assert checker.analyze_release_rows([row], config=config)["gate"]["blocked"] is False
    mutant = deepcopy(row)
    mutant["event_ledger"][field] = wrong
    report = checker.analyze_release_rows([mutant], config=config)
    assert report["gate"]["blocked"] is True
    finding = next(
        f for f in report["findings"] if f["detector_id"] == "collision_metric_inconsistent"
    )
    assert f"ledger_{field}_mismatch" in finding["measured"]["problems"]


@pytest.mark.parametrize("field", ["scenario_id", "seed", "planner", "software_commit"])
def test_typed_ledger_missing_identity_is_blocked(monkeypatch, field):
    """A typed ledger must retain every identity field emitted by its producer."""
    row, config = _fixture(monkeypatch)
    del row["event_ledger"][field]
    assert checker.analyze_release_rows([row], config=config)["gate"]["blocked"] is True
