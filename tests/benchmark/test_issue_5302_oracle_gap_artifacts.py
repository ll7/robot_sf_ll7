"""Characterize issue #5302 report artifact writing through the production path."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Any

from robot_sf.benchmark.issue_5302_oracle_gap import (
    CEILING_IDS,
    EXPECTED_PLANNERS,
    run_full_oracle_gap_analysis,
    write_report_artifacts,
)


def _minimal_oracle_gap_rows() -> list[dict[str, Any]]:
    """Build a small native six-arm dataset with separate selection and evaluation families."""
    rows: list[dict[str, Any]] = []
    for split, family in (("selection", "selection_family"), ("evaluation", "evaluation_family")):
        for episode_index in range(2):
            episode_id = f"{split}_episode_{episode_index}"
            scenario_cell = f"{family}_cell"
            for planner_index, planner_id in enumerate(EXPECTED_PLANNERS):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "scenario_id": f"{scenario_cell}_scenario_{episode_index}",
                        "scenario_family": family,
                        "scenario_cell": scenario_cell,
                        "split": split,
                        "seed": 5302 + episode_index,
                        "planner_id": planner_id,
                        "row_status": "successful_evidence",
                        "execution_mode": "native",
                        "config_hash": f"config-{planner_id}",
                        "repo_commit": "abcdef1234567890abcdef1234567890abcdef12",
                        "selection_score": 0.7 + 0.01 * planner_index,
                        "collision_rate": 0.0,
                        "severe_intrusion_rate": 0.0,
                        "completion_rate": 1.0,
                        "timeout_rate": 0.0,
                        "tail_clearance": 0.8,
                        "jerk": 1.0,
                        "pedestrian_disturbance": 0.1,
                        "compute_time_ms": 10.0 + planner_index,
                    }
                )
    return rows


def test_write_report_artifacts_returns_ordered_complete_report_set(tmp_path: Path) -> None:
    """The production writer must emit all ten reports in its stable return order."""
    analysis_result = run_full_oracle_gap_analysis(
        _minimal_oracle_gap_rows(), n_bootstrap=5, seed=5302
    )

    written = write_report_artifacts(analysis_result, tmp_path)
    expected_files = [
        "preflight.json",
        "ceiling_summary.json",
        "ceiling_summary.csv",
        "family_breakdown.csv",
        "cell_breakdown.csv",
        "failure_mechanism_map.csv",
        "runtime_tail.csv",
        "pareto_dominance.json",
        "normalized_regret.csv",
        "bootstrap_intervals.json",
    ]

    assert [path.relative_to(tmp_path).as_posix() for path in written] == [
        f"reports/{name}" for name in expected_files
    ]
    assert all(path.is_file() for path in written)

    ceiling_summary = json.loads(
        (tmp_path / "reports" / "ceiling_summary.json").read_text(encoding="utf-8")
    )
    assert set(ceiling_summary["ceilings"]) == set(CEILING_IDS)
    assert all((tmp_path / "reports" / name).read_text(encoding="utf-8") for name in expected_files)
