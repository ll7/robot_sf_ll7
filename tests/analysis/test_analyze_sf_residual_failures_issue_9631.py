"""Tests for the issue #9631 social-force residual-failure analyzer."""

from __future__ import annotations

from scripts.analysis.analyze_sf_residual_failures_issue_9631 import (
    _collision_subtype,
    classify_episode,
    is_noncollision_timeout,
    summarize_timeouts,
)


def _row(
    *,
    status: str = "failure",
    termination: str = "terminated",
    route_complete: bool = False,
    collision_event: bool = False,
    timeout_event: bool = True,
    family: str = "doorway",
    seed: int = 111,
    metrics: dict | None = None,
) -> dict:
    return {
        "episode_id": f"classic_{family}_low--{seed}--abc",
        "scenario_id": f"classic_{family}_low",
        "scenario_params": {"metadata": {"archetype": family}},
        "seed": seed,
        "status": status,
        "termination_reason": termination,
        "outcome": {
            "route_complete": route_complete,
            "collision_event": collision_event,
            "timeout_event": timeout_event,
        },
        "metrics": metrics or {},
    }


def test_success_classification_prefers_route_complete() -> None:
    item = classify_episode(_row(status="success", route_complete=True, timeout_event=False))
    assert item.status == "success"
    assert item.scenario_family == "doorway"
    assert item.seed == 111
    assert item.collision_subtype is None


def test_collision_subtype_splits_pedestrian_obstacle_wall() -> None:
    row = _row(
        status="collision",
        termination="collision",
        collision_event=True,
        timeout_event=False,
        metrics={"ped_collision_count": 1, "obstacle_collision_count": 0, "wall_collisions": 2.0},
    )
    item = classify_episode(row)
    assert item.status == "collision"
    assert item.collision_subtype == "pedestrian+wall"
    assert _collision_subtype({}) == "untyped"


def test_timeout_split_counts_max_steps_and_deadlocks() -> None:
    rows = [
        _row(metrics={"deadlock": True, "failure_to_progress": 300.0}),
        _row(termination="max_steps", metrics={"deadlock": False, "avg_speed": 0.5}),
        _row(
            status="collision",
            termination="collision",
            collision_event=True,
            timeout_event=False,
        ),
    ]
    summary = summarize_timeouts(rows)
    assert summary["timeouts"] == 2
    assert summary["max_steps"] == 1
    assert summary["terminated"] == 1
    assert summary["deadlock_true"] == 1
    assert summary["deadlock_known"] == 2
    assert summary["deadlock_true_non_timeout"] == 0
    assert summary["deadlock_true_success"] == 0
    assert summary["failure_to_progress"]["count"] == 1
    assert summary["avg_speed_m_s"]["median"] == 0.5


def test_noncollision_timeout_requires_timeout_without_collision() -> None:
    assert is_noncollision_timeout(_row()) is True
    assert (
        is_noncollision_timeout(_row(collision_event=True, timeout_event=True, status="collision"))
        is False
    )
    assert is_noncollision_timeout(_row(route_complete=True, timeout_event=False)) is False
