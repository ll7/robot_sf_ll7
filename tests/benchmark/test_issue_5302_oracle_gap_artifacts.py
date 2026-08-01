"""Characterize issue #5302 report artifact writing through the production path."""

from __future__ import annotations

import csv
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
                collision_rate = 0.25 if split == "evaluation" and planner_id == "orca" else 0.0
                severe_intrusion_rate = (
                    0.25 if split == "evaluation" and planner_id == "ppo" else 0.0
                )
                completion_rate = (
                    0.0 if split == "evaluation" and planner_id == "prediction_mpc" else 1.0
                )
                timeout_rate = (
                    0.25 if split == "evaluation" and planner_id == "prediction_planner" else 0.0
                )
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
                        "collision_rate": collision_rate,
                        "severe_intrusion_rate": severe_intrusion_rate,
                        "completion_rate": completion_rate,
                        "timeout_rate": timeout_rate,
                        "tail_clearance": 0.8,
                        "jerk": 1.0,
                        "pedestrian_disturbance": 0.1,
                        "compute_time_ms": 10.0 + planner_index,
                    }
                )
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a report CSV into dictionaries for stable contract assertions."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_write_report_artifacts_returns_ordered_complete_report_set(tmp_path: Path) -> None:
    """The production writer must preserve all ten report payloads and their stable order."""
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

    reports_dir = tmp_path / "reports"
    assert (
        json.loads((reports_dir / "preflight.json").read_text(encoding="utf-8"))
        == (analysis_result["preflight"])
    )
    assert (
        json.loads((reports_dir / "pareto_dominance.json").read_text(encoding="utf-8"))
        == (analysis_result["pareto_dominance"])
    )
    assert (
        json.loads((reports_dir / "bootstrap_intervals.json").read_text(encoding="utf-8"))
        == (analysis_result["bootstrap_intervals"])
    )

    ceiling_summary = json.loads((reports_dir / "ceiling_summary.json").read_text(encoding="utf-8"))
    assert set(ceiling_summary["ceilings"]) == set(CEILING_IDS)
    assert ceiling_summary["best_fixed_planner_id"] == analysis_result["best_fixed_planner"]
    assert ceiling_summary["claim_gate"] == analysis_result["claim_gate"]
    for ceiling_id, metrics in analysis_result["ceiling_summary"].items():
        bootstrap_intervals = analysis_result["bootstrap_intervals"]
        expected_metrics = {
            **metrics,
            "selection_score_ci": bootstrap_intervals[f"ceiling.{ceiling_id}.selection_score"][
                "ci_95"
            ],
            "collision_rate_ci": bootstrap_intervals[f"ceiling.{ceiling_id}.collision_rate"][
                "ci_95"
            ],
        }
        assert ceiling_summary["ceilings"][ceiling_id] == expected_metrics

    ceiling_rows = _read_csv_rows(reports_dir / "ceiling_summary.csv")
    assert [row["estimand"] for row in ceiling_rows] == list(CEILING_IDS)
    assert ceiling_rows[0]["planner_id"] == analysis_result["best_fixed_planner"]

    family_rows = _read_csv_rows(reports_dir / "family_breakdown.csv")
    assert {(row["entity_type"], row["entity_id"]) for row in family_rows} == {
        *(("planner", planner_id) for planner_id in EXPECTED_PLANNERS),
        *(("ceiling", ceiling_id) for ceiling_id in CEILING_IDS),
    }
    assert all(row["scenario_family"] == "evaluation_family" for row in family_rows)

    cell_rows = _read_csv_rows(reports_dir / "cell_breakdown.csv")
    assert {(row["entity_type"], row["entity_id"]) for row in cell_rows} == {
        ("planner", planner_id) for planner_id in EXPECTED_PLANNERS
    }
    assert all(
        row["scenario_family"] == "evaluation_family"
        and row["scenario_cell"] == "evaluation_family_cell"
        for row in cell_rows
    )

    failure_rows = {
        row["entity_id"]: row for row in _read_csv_rows(reports_dir / "failure_mechanism_map.csv")
    }
    assert failure_rows["orca"]["collision_count"] == "2"
    assert failure_rows["ppo"]["severe_intrusion_count"] == "2"
    assert failure_rows["prediction_planner"]["timeout_count"] == "2"

    runtime_rows = _read_csv_rows(reports_dir / "runtime_tail.csv")
    assert [row["entity_id"] for row in runtime_rows] == [*EXPECTED_PLANNERS, *CEILING_IDS]
    assert all(row["count"] == "2" for row in runtime_rows)

    regret_rows = _read_csv_rows(reports_dir / "normalized_regret.csv")
    assert [row["planner_id"] for row in regret_rows] == list(EXPECTED_PLANNERS)
