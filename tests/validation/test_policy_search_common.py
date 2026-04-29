"""Tests for policy-search reporting helpers."""

from __future__ import annotations

from scripts.validation.policy_search_common import (
    classify_failure_mode,
    infer_scenario_family,
    summarize_policy_search_records,
)


def test_infer_scenario_family_prefers_explicit_fields() -> None:
    row = {"scenario_family": "Classic"}

    assert infer_scenario_family(row) == "classic"


def test_infer_scenario_family_falls_back_to_scenario_id_prefix() -> None:
    assert infer_scenario_family({"scenario_id": "classic_bottleneck_medium"}) == "classic"
    assert infer_scenario_family({"scenario_id": "francis2023_blind_corner"}) == "francis2023"
    assert infer_scenario_family({"scenario_id": "planner_sanity_simple"}) == "nominal"


def test_classify_failure_mode_distinguishes_collision_types() -> None:
    static_row = {
        "scenario_id": "corner_90_turn",
        "termination_reason": "collision",
        "scenario_params": {"humans": []},
        "metrics": {},
    }
    ped_row = {
        "scenario_id": "classic_crossing_medium",
        "termination_reason": "collision",
        "scenario_params": {"humans": [{"id": "p1"}]},
        "metrics": {},
    }

    assert classify_failure_mode(static_row) == "static_collision"
    assert classify_failure_mode(ped_row) == "pedestrian_collision"


def test_classify_failure_mode_uses_near_miss_and_low_speed_heuristics() -> None:
    near_miss_row = {
        "scenario_id": "francis2023_parallel_traffic",
        "termination_reason": "max_steps",
        "metrics": {"near_misses": 2},
    }
    bottleneck_row = {
        "scenario_id": "classic_bottleneck_medium",
        "termination_reason": "max_steps",
        "metrics": {"avg_speed": 0.03},
    }
    timeout_row = {
        "scenario_id": "francis2023_intersection_wait",
        "termination_reason": "max_steps",
        "metrics": {"avg_speed": 0.42, "goal_progress": 0.15},
    }

    assert classify_failure_mode(near_miss_row) == "near_miss_intrusive"
    assert classify_failure_mode(bottleneck_row) == "bottleneck_yield_failure"
    assert classify_failure_mode(timeout_row) == "timeout_low_progress"


def test_summarize_policy_search_records_builds_family_and_failure_counts() -> None:
    records = [
        {
            "scenario_id": "classic_bottleneck_medium",
            "termination_reason": "collision",
            "scenario_params": {"humans": [{"id": "p1"}]},
            "metrics": {"min_distance": 0.1, "avg_speed": 0.2},
        },
        {
            "scenario_id": "francis2023_parallel_traffic",
            "termination_reason": "max_steps",
            "metrics": {"near_misses": 1, "min_distance": 0.5, "avg_speed": 0.1},
        },
        {
            "scenario_id": "francis2023_blind_corner",
            "termination_reason": "success",
            "metrics": {"success": 1.0, "min_distance": 1.3, "avg_speed": 0.7},
        },
    ]

    summary = summarize_policy_search_records(records)

    assert summary["episodes"] == 3
    assert summary["collision_rate"] == 1 / 3
    assert summary["near_miss_rate"] == 1 / 3
    assert summary["failure_mode_counts"]["pedestrian_collision"] == 1
    assert summary["failure_mode_counts"]["near_miss_intrusive"] == 1
    assert summary["scenario_family"]["classic"]["episodes"] == 1
    assert summary["scenario_family"]["francis2023"]["episodes"] == 2
