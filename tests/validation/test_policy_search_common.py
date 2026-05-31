"""Tests for policy-search reporting helpers."""

from __future__ import annotations

from scripts.validation.policy_search_common import (
    classify_failure_mode,
    infer_scenario_family,
    normalize_scenario_exclusion,
    summarize_policy_search_records,
)


def test_infer_scenario_family_prefers_explicit_fields() -> None:
    """Explicit family metadata should win over inferred identifiers."""
    row = {"scenario_family": "Classic"}

    assert infer_scenario_family(row) == "classic"


def test_infer_scenario_family_falls_back_to_scenario_id_prefix() -> None:
    """Scenario ID prefixes should map to stable family names."""
    assert infer_scenario_family({"scenario_id": "classic_bottleneck_medium"}) == "classic"
    assert infer_scenario_family({"scenario_id": "francis2023_blind_corner"}) == "francis2023"
    assert infer_scenario_family({"scenario_id": "planner_sanity_simple"}) == "nominal"


def test_classify_failure_mode_distinguishes_collision_types() -> None:
    """Collision classification should distinguish static from pedestrian cases."""
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
    """Near-miss and low-speed timeout heuristics should produce readable labels."""
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
    """Summary aggregation should include rates, failure counts, and family splits."""
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


def test_summarize_policy_search_records_keeps_actuation_diagnostics_separate() -> None:
    """Synthetic actuation diagnostics should aggregate outside outcome rates."""
    records = [
        {
            "scenario_id": "classic_cross_trap_high",
            "termination_reason": "collision",
            "metrics": {
                "command_clip_fraction": 0.25,
                "yaw_rate_saturation_fraction": 0.5,
                "signed_braking_peak_m_s2": -1.0,
            },
        },
        {
            "scenario_id": "classic_cross_trap_high",
            "termination_reason": "success",
            "metrics": {
                "command_clip_fraction": 0.0,
                "yaw_rate_saturation_fraction": 0.0,
                "signed_braking_peak_m_s2": -0.5,
            },
        },
    ]

    summary = summarize_policy_search_records(records)

    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["synthetic_actuation"] == {
        "command_clip_fraction_mean": 0.125,
        "yaw_rate_saturation_fraction_mean": 0.25,
        "signed_braking_peak_m_s2_mean": -0.75,
    }


def test_policy_search_exclusions_require_explicit_evidence() -> None:
    """Invalid/impossible scenarios should only be excluded with explicit evidence."""
    excluded = {
        "scenario_id": "francis2023_narrow_doorway",
        "seed": 111,
        "termination_reason": "collision",
        "metrics": {"min_distance": 0.0},
        "scenario_exclusion": {
            "status": "impossible",
            "reason": "geometry_footprint_infeasible",
            "evidence": [
                "doorway gap is exactly 2.0 m",
                "robot radius is 1.0 m; any positive clearance margin is infeasible",
            ],
        },
    }
    unsupported = {
        "scenario_id": "classic_cross_trap_high",
        "seed": 112,
        "termination_reason": "collision",
        "scenario_exclusion": {
            "status": "impossible",
            "reason": "missing evidence should not be accepted",
        },
    }

    assert normalize_scenario_exclusion(excluded) == {
        "scenario_id": "francis2023_narrow_doorway",
        "seed": 111,
        "status": "impossible",
        "reason": "geometry_footprint_infeasible",
        "evidence": [
            "doorway gap is exactly 2.0 m",
            "robot radius is 1.0 m; any positive clearance margin is infeasible",
        ],
    }
    assert normalize_scenario_exclusion(unsupported) is None


def test_summarize_policy_search_records_reports_raw_and_adjusted_exclusions() -> None:
    """Raw rates stay visible while evidence-adjusted rates exclude proven scenario defects."""
    records = [
        {
            "scenario_id": "classic_cross_trap_high",
            "seed": 112,
            "termination_reason": "collision",
            "scenario_params": {"humans": [{"id": "p1"}]},
            "scenario_exclusion": {
                "status": "invalid",
                "reason": "zero_action_first_step_collision",
                "evidence": ["zero-action first step terminates before policy control matters"],
            },
        },
        {
            "scenario_id": "classic_cross_trap_high",
            "seed": 113,
            "termination_reason": "success",
        },
    ]

    summary = summarize_policy_search_records(records)

    assert summary["episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["failure_mode_counts"] == {}
    assert summary["scenario_exclusions"]["count"] == 1
    assert summary["scenario_exclusions"]["by_reason"] == {"zero_action_first_step_collision": 1}
    assert summary["evidence_adjusted"] == {
        "episodes": 1,
        "excluded_episodes": 1,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "near_miss_rate": 0.0,
    }
