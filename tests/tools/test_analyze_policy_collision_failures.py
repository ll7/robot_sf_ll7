"""Tests for policy-analysis collision failure summaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools import analyze_policy_collision_failures as analyzer

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSONL episode fixtures for collision-failure analysis."""
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_analyze_run_splits_collision_hotspots_by_scenario_and_seed(tmp_path: Path) -> None:
    """Scenario rows should expose collision type and seed concentration."""
    run_root = tmp_path / "issue193_policy_analysis_dyn_large_med_s231"
    run_root.mkdir()
    _write_jsonl(
        run_root / "episodes.jsonl",
        [
            {
                "scenario_id": "classic_bottleneck_high",
                "seed": 101,
                "termination_reason": "collision",
                "metrics": {
                    "collisions": 1.0,
                    "obstacle_collision_count": 1.0,
                    "ped_collision_count": 0.0,
                    "agent_collision_count": 0.0,
                },
            },
            {
                "scenario_id": "classic_bottleneck_high",
                "seed": 102,
                "termination_reason": "success",
                "metrics": {"collisions": 0.0},
            },
            {
                "scenario_id": "classic_cross_trap_medium",
                "seed": 201,
                "termination_reason": "collision",
                "metrics": {
                    "collisions": 1.0,
                    "obstacle_collision_count": 0.0,
                    "ped_collision_count": 1.0,
                    "agent_collision_count": 0.0,
                },
            },
        ],
    )

    payload = analyzer.analyze_run(run_root)

    assert payload["totals"]["episodes"] == 3
    assert payload["totals"]["collisions"] == 2
    assert payload["totals"]["ped_collisions"] == 1
    assert payload["totals"]["obstacle_collisions"] == 1
    bottleneck = next(
        row for row in payload["scenarios"] if row["scenario"] == "classic_bottleneck_high"
    )
    assert bottleneck["scenario"] == "classic_bottleneck_high"
    assert bottleneck["collision_seeds"] == ["101"]


def test_markdown_report_includes_summary_and_hotspots(tmp_path: Path) -> None:
    """Markdown output should include run-level and hotspot evidence."""
    run_root = tmp_path / "candidate"
    run_root.mkdir()
    _write_jsonl(
        run_root / "episodes.jsonl",
        [
            {
                "scenario": "classic_merging_low",
                "scenario_seed": "eval-1",
                "termination_reason": "collision",
                "metrics": {"obstacle_collision_count": 1},
            }
        ],
    )
    payload = analyzer.analyze_runs([run_root])

    report = analyzer.render_markdown(payload, top_n=5)

    assert "Policy Collision Failure Analysis" in report
    assert "`candidate`" in report
    assert "`classic_merging_low`" in report
    assert "eval-1" in report
