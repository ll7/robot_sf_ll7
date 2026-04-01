"""Tests for artifact-driven scenario difficulty analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.scenario_difficulty import build_scenario_difficulty_analysis

if TYPE_CHECKING:
    from pathlib import Path


def _planner_rows() -> list[dict[str, str]]:
    return [
        {
            "planner_key": "goal",
            "algo": "goal",
            "planner_group": "core",
            "status": "ok",
            "preflight_status": "ok",
            "benchmark_success": "true",
        },
        {
            "planner_key": "orca",
            "algo": "orca",
            "planner_group": "core",
            "status": "ok",
            "preflight_status": "ok",
            "benchmark_success": "true",
        },
        {
            "planner_key": "stream_gap",
            "algo": "stream_gap",
            "planner_group": "experimental",
            "status": "ok",
            "preflight_status": "ok",
            "benchmark_success": "true",
        },
    ]


def _seed_payload() -> dict[str, object]:
    return {
        "rows": [
            {
                "scenario_id": "easy_case",
                "planner_key": "goal",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.02, "cv": 0.01},
                    "time_to_goal_norm": {"ci_half_width": 0.03, "cv": 0.02},
                    "snqi": {"ci_half_width": 0.04, "cv": 0.03},
                },
            },
            {
                "scenario_id": "easy_case",
                "planner_key": "orca",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.03, "cv": 0.02},
                    "time_to_goal_norm": {"ci_half_width": 0.04, "cv": 0.03},
                    "snqi": {"ci_half_width": 0.04, "cv": 0.03},
                },
            },
            {
                "scenario_id": "hard_case",
                "planner_key": "goal",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.11, "cv": 0.20},
                    "time_to_goal_norm": {"ci_half_width": 0.10, "cv": 0.14},
                    "snqi": {"ci_half_width": 0.08, "cv": 0.12},
                },
            },
            {
                "scenario_id": "hard_case",
                "planner_key": "orca",
                "seed_count": 3,
                "summary": {
                    "success": {"ci_half_width": 0.10, "cv": 0.18},
                    "time_to_goal_norm": {"ci_half_width": 0.09, "cv": 0.12},
                    "snqi": {"ci_half_width": 0.08, "cv": 0.11},
                },
            },
        ]
    }


def _preview_payload() -> dict[str, object]:
    return {
        "truncated": False,
        "route_clearance_warnings": [
            {
                "scenario": "hard_case",
                "warning_scope": "route",
                "min_clearance_margin_m": 0.2,
            }
        ],
        "scenarios": [
            {
                "name": "easy_case",
                "simulation_config": {"ped_density": 0.0},
                "metadata": {
                    "archetype": "easy_family",
                    "flow": "none",
                    "behavior": "none",
                    "primary_capability": "frame_consistency",
                    "target_failure_mode": "coordinate_transform",
                    "determinism": "deterministic",
                },
            },
            {
                "name": "hard_case",
                "simulation_config": {"ped_density": 0.5},
                "metadata": {
                    "archetype": "hard_family",
                    "flow": "perpendicular",
                    "behavior": "crowd",
                    "primary_capability": "dynamic_interaction",
                    "target_failure_mode": "social_collision",
                    "determinism": "stochastic",
                },
            },
        ],
    }


def test_build_scenario_difficulty_ranks_harder_scenarios_first() -> None:
    """Difficulty ranking should place scenarios that are weak across core planners above easier control cases because the proxy is meant to separate scenario hardness from planner quality."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:2],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.35",
                "collisions_mean": "0.45",
                "near_misses_mean": "0.70",
                "time_to_goal_norm_mean": "0.85",
                "snqi_mean": "-0.40",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.40",
                "collisions_mean": "0.35",
                "near_misses_mean": "0.65",
                "time_to_goal_norm_mean": "0.80",
                "snqi_mean": "-0.30",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
    )

    scenario_rows = {row["scenario_id"]: row for row in analysis["scenario_rows"]}
    assert analysis["scenario_rows"][0]["scenario_id"] == "hard_case"
    assert (
        scenario_rows["hard_case"]["difficulty_score"]
        > scenario_rows["easy_case"]["difficulty_score"]
    )
    assert scenario_rows["hard_case"]["route_clearance_warning"] is True
    assert scenario_rows["hard_case"]["seed_success_ci_half_width_mean"] == pytest.approx(0.105)


def test_build_scenario_difficulty_flags_easy_scenario_underperformance() -> None:
    """Residual summaries should flag planners that underperform on easier scenarios because that is the main signal for separating planner mismatch from globally hard scenarios."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows(),
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.96",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.92",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.08",
                "time_to_goal_norm_mean": "0.32",
                "snqi_mean": "0.25",
            },
            {
                "planner_key": "stream_gap",
                "algo": "stream_gap",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.20",
                "collisions_mean": "0.60",
                "near_misses_mean": "0.70",
                "time_to_goal_norm_mean": "0.95",
                "snqi_mean": "-0.50",
            },
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.45",
                "collisions_mean": "0.40",
                "near_misses_mean": "0.65",
                "time_to_goal_norm_mean": "0.80",
                "snqi_mean": "-0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.40",
                "collisions_mean": "0.35",
                "near_misses_mean": "0.60",
                "time_to_goal_norm_mean": "0.78",
                "snqi_mean": "-0.25",
            },
            {
                "planner_key": "stream_gap",
                "algo": "stream_gap",
                "scenario_family": "hard_family",
                "scenario_id": "hard_case",
                "episodes": "3",
                "success_mean": "0.38",
                "collisions_mean": "0.38",
                "near_misses_mean": "0.62",
                "time_to_goal_norm_mean": "0.81",
                "snqi_mean": "-0.27",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
    )

    planner_summary_rows = {row["planner_key"]: row for row in analysis["planner_summary_rows"]}
    stream_gap = planner_summary_rows["stream_gap"]
    assert stream_gap["easy_scenario_underperformance_count"] == 1
    assert stream_gap["worst_scenarios"][0] == "easy_case"


def test_build_scenario_difficulty_requests_verified_simple_rerun_without_overlap(
    tmp_path: Path,
) -> None:
    """Verified-simple assessment should ask for a bounded pilot when the current campaign has no overlap with the candidate subset because the recommendation would otherwise overclaim."""
    verified_simple_manifest = tmp_path / "verified_simple_subset.yaml"
    verified_simple_manifest.write_text(
        "scenarios:\n  - name: different_case\n    metadata:\n      archetype: control\n",
        encoding="utf-8",
    )

    analysis = build_scenario_difficulty_analysis(
        planner_rows=_planner_rows()[:2],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
        ],
        seed_variability_payload=_seed_payload(),
        preview_payload=_preview_payload(),
        verified_simple_manifest_path=verified_simple_manifest,
    )

    assessment = analysis["verified_simple_assessment"]
    assert assessment["status"] == "rerun_required"
    assert assessment["worth_adding"] is None
    assert "bounded pilot" in assessment["recommendation"]


def test_build_scenario_difficulty_reports_fallback_selection_when_core_set_missing() -> None:
    """Fallback consensus metadata should stay internally consistent when a non-paper campaign only exposes non-core planners."""
    analysis = build_scenario_difficulty_analysis(
        planner_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "planner_group": "experimental",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "planner_group": "experimental",
                "status": "ok",
                "preflight_status": "ok",
                "benchmark_success": "true",
            },
        ],
        scenario_breakdown_rows=[
            {
                "planner_key": "goal",
                "algo": "goal",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.95",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.05",
                "time_to_goal_norm_mean": "0.30",
                "snqi_mean": "0.30",
            },
            {
                "planner_key": "orca",
                "algo": "orca",
                "scenario_family": "easy_family",
                "scenario_id": "easy_case",
                "episodes": "3",
                "success_mean": "0.90",
                "collisions_mean": "0.00",
                "near_misses_mean": "0.10",
                "time_to_goal_norm_mean": "0.35",
                "snqi_mean": "0.20",
            },
        ],
    )

    assert (
        analysis["primary_proxy"]["eligible_planner_selection"]
        == "all planners (fallback: no eligible core set)"
    )
    assert analysis["primary_proxy"]["eligible_planner_count"] == 2
    assert any("fell back to all planners" in finding for finding in analysis["findings"])
