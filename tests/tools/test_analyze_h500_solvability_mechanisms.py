"""Tests for aggregate h500 solvability mechanism analysis."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from scripts.tools import analyze_h500_solvability_mechanisms

if TYPE_CHECKING:
    from pathlib import Path


def _delta_row(
    *,
    planner: str,
    scenario: str,
    family: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    """Build the comparison shape consumed by the analyzer."""
    success_base = metrics.get("success_base", 0.0)
    success_candidate = metrics["success_candidate"]
    unfinished_base = metrics.get("unfinished_base", 1.0)
    unfinished_candidate = metrics["unfinished_candidate"]
    collision_delta = metrics["collision_delta"]
    near_miss_delta = metrics["near_miss_delta"]
    candidate_time = metrics["candidate_time"]
    return {
        "planner_key": planner,
        "scenario_id": scenario,
        "scenario_family": family,
        "base_episodes": 3,
        "candidate_episodes": 3,
        "metrics": {
            "success_mean": {
                "base": success_base,
                "candidate": success_candidate,
                "delta": success_candidate - success_base,
            },
            "unfinished_mean": {
                "base": unfinished_base,
                "candidate": unfinished_candidate,
                "delta": unfinished_candidate - unfinished_base,
            },
            "collisions_mean": {
                "base": 0.0,
                "candidate": collision_delta,
                "delta": collision_delta,
            },
            "near_misses_mean": {
                "base": 0.0,
                "candidate": near_miss_delta,
                "delta": near_miss_delta,
            },
            "snqi_mean": {"base": -0.1, "candidate": -0.2, "delta": -0.1},
            "time_to_goal_norm_mean": {
                "base": 1.0,
                "candidate": candidate_time,
                "delta": candidate_time - 1.0,
            },
        },
    }


def test_analyzer_classifies_timeout_to_success_mechanisms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Aggregate deltas should produce mechanism cases, reps, and family rollups."""
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "base_campaign_id": "fixed",
                "candidate_campaign_id": "h500",
                "scenario_deltas": [
                    _delta_row(
                        planner="orca",
                        scenario="clean_fast",
                        family="bottleneck",
                        metrics={
                            "success_candidate": 1.0,
                            "unfinished_candidate": 0.0,
                            "collision_delta": 0.0,
                            "near_miss_delta": 0.0,
                            "candidate_time": 0.62,
                        },
                    ),
                    _delta_row(
                        planner="orca",
                        scenario="exposure",
                        family="crossing",
                        metrics={
                            "success_candidate": 1.0,
                            "unfinished_candidate": 0.0,
                            "collision_delta": 0.0,
                            "near_miss_delta": 3.0,
                            "candidate_time": 0.70,
                        },
                    ),
                    _delta_row(
                        planner="goal",
                        scenario="collision",
                        family="crossing",
                        metrics={
                            "success_candidate": 0.67,
                            "unfinished_candidate": 0.33,
                            "collision_delta": 0.33,
                            "near_miss_delta": 4.0,
                            "candidate_time": 0.90,
                        },
                    ),
                    _delta_row(
                        planner="goal",
                        scenario="unchanged",
                        family="blocked",
                        metrics={
                            "success_candidate": 0.0,
                            "unfinished_candidate": 1.0,
                            "collision_delta": 0.0,
                            "near_miss_delta": 0.0,
                            "candidate_time": 1.0,
                        },
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_h500_solvability_mechanisms.py",
            str(comparison),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert analyze_h500_solvability_mechanisms.main() == 0

    payload = json.loads((output_dir / "h500_solvability_mechanisms.json").read_text())
    assert payload["mechanism_counts"] == {
        "budget_limited_clean_completion": 1,
        "exposure_enabled_completion": 1,
        "safety_regressed_completion": 1,
    }
    assert len(payload["cases"]) == 3
    assert payload["interpretation_boundary"]["waiting_until_dynamic_obstacles_passed"] == (
        "trace_required"
    )
    assert (output_dir / "h500_solvability_cases.csv").exists()
    assert (output_dir / "h500_solvability_family_rollup.csv").exists()
    assert "aggregate fixed-vs-h500 comparison" in (
        output_dir / "h500_solvability_mechanisms.md"
    ).read_text(encoding="utf-8")
