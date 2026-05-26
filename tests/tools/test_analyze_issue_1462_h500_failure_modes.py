"""Tests for issue #1462 S10/h500 failure-mode analysis."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.tools import analyze_issue_1462_h500_failure_modes

if TYPE_CHECKING:
    from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small CSV fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys())
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")


def test_grouping_compares_only_local_h500_candidates_against_stage_a_rows() -> None:
    """The issue #1462 candidate group should exclude older adapter baselines."""
    groups = analyze_issue_1462_h500_failure_modes._planner_groups(
        [
            {"planner_key": "orca", "planner_group": "core"},
            {"planner_key": "prediction_planner", "planner_group": "experimental"},
            {"planner_key": "socnav_sampling", "planner_group": "experimental"},
            {"planner_key": "hybrid_rule_v3_fast_progress", "planner_group": "experimental"},
            {
                "planner_key": "scenario_adaptive_hybrid_orca_v2_collision_guard",
                "planner_group": "experimental",
            },
        ]
    )

    assert groups["orca"] == "core"
    assert groups["prediction_planner"] == "core"
    assert groups["socnav_sampling"] == "core"
    assert groups["hybrid_rule_v3_fast_progress"] == "candidate"
    assert groups["scenario_adaptive_hybrid_orca_v2_collision_guard"] == "candidate"


def test_float_helper_defaults_non_numeric_optional_values() -> None:
    """Optional CSV/JSON values should not crash the analysis helper."""
    parse = analyze_issue_1462_h500_failure_modes._float

    assert parse(None) == 0.0
    assert parse("None") == 0.0
    assert parse("null") == 0.0
    assert parse("nan") == 0.0
    assert parse("not-a-number") == 0.0
    assert parse("1.25") == 1.25


def test_std_helper_uses_population_standard_deviation() -> None:
    """The variability table intentionally reports population standard deviation."""
    assert analyze_issue_1462_h500_failure_modes._std([]) == 0.0
    assert analyze_issue_1462_h500_failure_modes._std([1.0]) == 0.0
    assert analyze_issue_1462_h500_failure_modes._std([0.0, 1.0]) == 0.5


def test_analyzer_requires_raw_campaign_dir(monkeypatch) -> None:
    """Raw outcome rollups are required so benchmark evidence cannot silently zero-fill them."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_issue_1462_h500_failure_modes.py",
            "--evidence-dir",
            "docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23",
        ],
    )

    with pytest.raises(SystemExit):
        analyze_issue_1462_h500_failure_modes.parse_args()


def test_analyzer_writes_tables_with_raw_termination_rollups(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A tiny fixture should produce scenario, matrix, seed, and summary outputs."""
    evidence_dir = tmp_path / "evidence"
    reports_dir = evidence_dir / "reports"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"

    (evidence_dir / "run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "run_meta.json").write_text(
        json.dumps({"campaign_id": "fixture", "git_hash": "abc123"}), encoding="utf-8"
    )
    (reports_dir / "campaign_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (reports_dir / "campaign_summary.json").write_text(
        json.dumps({"campaign": {"campaign_id": "fixture", "episodes_written": 4}}),
        encoding="utf-8",
    )

    _write_csv(
        reports_dir / "campaign_table.csv",
        [
            {"planner_key": "orca", "planner_group": "core"},
            {"planner_key": "hybrid_rule_v3_fast_progress", "planner_group": "experimental"},
        ],
    )
    _write_csv(
        reports_dir / "scenario_breakdown.csv",
        [
            {
                "planner_key": "orca",
                "scenario_family": "bottleneck",
                "scenario_id": "classic_bottleneck_high",
                "episodes": 2,
                "success_mean": 0.0,
                "collisions_mean": 0.5,
                "ped_collision_count_mean": 0.5,
                "obstacle_collision_count_mean": 0.0,
                "near_misses_mean": 10.0,
                "time_to_goal_norm_mean": 0.0,
                "comfort_exposure_mean": 0.0,
                "snqi_mean": -0.5,
            },
            {
                "planner_key": "hybrid_rule_v3_fast_progress",
                "scenario_family": "bottleneck",
                "scenario_id": "classic_bottleneck_high",
                "episodes": 2,
                "success_mean": 1.0,
                "collisions_mean": 0.0,
                "ped_collision_count_mean": 0.0,
                "obstacle_collision_count_mean": 0.0,
                "near_misses_mean": 20.0,
                "time_to_goal_norm_mean": 0.8,
                "comfort_exposure_mean": 0.5,
                "snqi_mean": 0.3,
            },
        ],
    )
    _write_csv(
        reports_dir / "seed_variability_by_scenario.csv",
        [
            {
                "scenario_id": "classic_bottleneck_high",
                "planner_key": "orca",
                "seed": 111,
                "success_per_seed_mean": 0.0,
                "collisions_per_seed_mean": 1.0,
                "near_misses_per_seed_mean": 10.0,
                "time_to_goal_norm_per_seed_mean": 0.0,
                "snqi_per_seed_mean": -0.5,
            },
            {
                "scenario_id": "classic_bottleneck_high",
                "planner_key": "orca",
                "seed": 112,
                "success_per_seed_mean": 0.0,
                "collisions_per_seed_mean": 0.0,
                "near_misses_per_seed_mean": 5.0,
                "time_to_goal_norm_per_seed_mean": 0.0,
                "snqi_per_seed_mean": -0.5,
            },
            {
                "scenario_id": "classic_bottleneck_high",
                "planner_key": "hybrid_rule_v3_fast_progress",
                "seed": 111,
                "success_per_seed_mean": 1.0,
                "collisions_per_seed_mean": 0.0,
                "near_misses_per_seed_mean": 20.0,
                "time_to_goal_norm_per_seed_mean": 0.8,
                "snqi_per_seed_mean": 0.3,
            },
            {
                "scenario_id": "classic_bottleneck_high",
                "planner_key": "hybrid_rule_v3_fast_progress",
                "seed": 112,
                "success_per_seed_mean": 1.0,
                "collisions_per_seed_mean": 0.0,
                "near_misses_per_seed_mean": 15.0,
                "time_to_goal_norm_per_seed_mean": 0.8,
                "snqi_per_seed_mean": 0.3,
            },
        ],
    )
    episodes = [
        {
            "scenario_id": "classic_bottleneck_high",
            "status": "failure",
            "termination_reason": "terminated",
            "metrics": {"ped_collision_count": 0, "obstacle_collision_count": 0},
        },
        {
            "scenario_id": "classic_bottleneck_high",
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {"ped_collision_count": 1, "obstacle_collision_count": 0},
        },
    ]
    run_dir = raw_dir / "runs" / "orca__differential_drive"
    run_dir.mkdir(parents=True)
    (run_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    candidate_dir = raw_dir / "runs" / "hybrid_rule_v3_fast_progress__differential_drive"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "scenario_id": "classic_bottleneck_high",
                    "status": "success",
                    "termination_reason": "success",
                    "metrics": {"ped_collision_count": 0, "obstacle_collision_count": 0},
                }
            )
            + "\n"
            for _ in range(2)
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_issue_1462_h500_failure_modes.py",
            "--evidence-dir",
            str(evidence_dir),
            "--raw-campaign-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analyze_issue_1462_h500_failure_modes.main() == 0

    matrix = (output_dir / "candidate_vs_core_matrix.csv").read_text(encoding="utf-8")
    assert "classic_bottleneck_high,bottleneck,0.0,1.0,1.0" in matrix
    assert "core_time_to_goal_norm_mean" in matrix
    assert "candidate_comfort_exposure_mean" in matrix
    scenario_table = (output_dir / "scenario_difficulty_table.csv").read_text(encoding="utf-8")
    assert "candidate_specific_improvement" in scenario_table
    assert "0.25,0.25" in scenario_table
    assert "time_to_goal_norm_mean_all" in scenario_table
    assert "comfort_exposure_mean_all" in scenario_table
    assert "snqi_mean_all" in scenario_table
    seed_table_text = (output_dir / "seed_difficulty_table.csv").read_text(encoding="utf-8")
    assert "time_to_goal_norm_mean" in seed_table_text
    assert "snqi_mean" in seed_table_text
    variability_text = (output_dir / "planner_scenario_seed_variability.csv").read_text(
        encoding="utf-8"
    )
    assert "time_to_goal_norm_mean" in variability_text
    assert "snqi_mean" in variability_text
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "easiest_scenarios" in summary
    assert len(summary["easiest_scenarios"]) > 0
    assert summary["snqi_contract_status"] == "unknown"
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Easiest Scenarios" in readme
    assert "Easiest aggregate scenarios" in readme
    assert "Relation to H500 Solvability Mechanisms" in readme
    assert "SNQI Field" in readme
    assert "diagnostic inspection" in readme
