"""Tests for the issue #4195 h600 aggregation artifact builder."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis.build_issue_4195_h600_aggregation_artifact import build_artifact

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture_reports(
    reports_dir: Path,
    *,
    scenario_matrix_hash: str = "matrix-a",
    planners: tuple[str, ...] = ("goal", "orca"),
) -> None:
    reports_dir.mkdir(parents=True)
    planner_rows = [
        {
            "planner_key": planner,
            "episodes": 4,
            "success_mean": "0.5000",
            "collisions_mean": "0.5000",
            "near_misses_mean": "1.5000",
            "comfort_exposure_mean": "0.2500",
            "snqi_mean": "0.1250",
        }
        for planner in planners
    ]
    runs = [
        {
            "planner": {"key": planner},
            "aggregates": {
                planner: {
                    "comfort_exposure": {
                        "mean": 0.25,
                        "mean_ci": [0.1, 0.4],
                    }
                }
            },
        }
        for planner in planners
    ]
    summary = {
        "campaign": {
            "campaign_id": reports_dir.name,
            "evidence_status": "valid",
            "benchmark_success": True,
            "scenario_matrix": "configs/scenarios/example.yaml",
            "scenario_matrix_hash": scenario_matrix_hash,
            "comparability_mapping_hash": "mapping-a",
            "created_at_utc": "2026-07-03T00:00:00Z",
            "git_hash": "abc123",
        },
        "planner_rows": planner_rows,
        "runs": runs,
    }
    (reports_dir / "campaign_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    with (reports_dir / "seed_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "scenario_id",
                "planner_key",
                "seed",
                "success",
                "collision",
                "near_miss",
                "snqi",
            ],
        )
        writer.writeheader()
        for planner in planners:
            writer.writerows(
                [
                    {
                        "episode_id": f"{planner}-111-0",
                        "scenario_id": "a",
                        "planner_key": planner,
                        "seed": "111",
                        "success": "1.0",
                        "collision": "0.0",
                        "near_miss": "1.0",
                        "snqi": "0.2",
                    },
                    {
                        "episode_id": f"{planner}-111-1",
                        "scenario_id": "b",
                        "planner_key": planner,
                        "seed": "111",
                        "success": "0.0",
                        "collision": "1.0",
                        "near_miss": "3.0",
                        "snqi": "0.0",
                    },
                    {
                        "episode_id": f"{planner}-112-0",
                        "scenario_id": "a",
                        "planner_key": planner,
                        "seed": "112",
                        "success": "1.0",
                        "collision": "0.0",
                        "near_miss": "0.0",
                        "snqi": "0.4",
                    },
                    {
                        "episode_id": f"{planner}-112-1",
                        "scenario_id": "b",
                        "planner_key": planner,
                        "seed": "112",
                        "success": "0.0",
                        "collision": "1.0",
                        "near_miss": "2.0",
                        "snqi": "0.2",
                    },
                ]
            )


def test_build_artifact_writes_tables_comparability_and_checksums(tmp_path: Path) -> None:
    """Fixture reports produce deterministic aggregation artifacts."""

    confirm_reports = tmp_path / "confirm" / "reports"
    extended_reports = tmp_path / "extended" / "reports"
    _write_fixture_reports(confirm_reports)
    _write_fixture_reports(extended_reports, planners=("goal", "orca", "prediction_mpc"))
    output_dir = tmp_path / "evidence"

    result = build_artifact(
        confirm_reports=confirm_reports,
        extended_reports=extended_reports,
        output_dir=output_dir,
        bootstrap_samples=100,
        confidence=0.95,
    )

    assert result["status"] == "ok"
    assert result["row_count"] == 25
    metric_rows = list(csv.DictReader((output_dir / "planner_metric_summary.csv").open()))
    goal_snqi = next(
        row
        for row in metric_rows
        if row["job_id"] == "13268" and row["planner_key"] == "goal" and row["metric"] == "snqi"
    )
    assert json.loads(goal_snqi["seed_values_json"]) == pytest.approx({"111": 0.1, "112": 0.3})
    assert goal_snqi["source"] == "seed_episode_rows"
    goal_comfort = next(
        row
        for row in metric_rows
        if row["job_id"] == "13268" and row["planner_key"] == "goal" and row["metric"] == "comfort"
    )
    assert goal_comfort["source"] == "campaign_summary_aggregate"
    assert goal_comfort["value_status"] == "no_seed_episode_column"
    assert goal_comfort["bootstrap_ci_low"] == "0.100000"
    assert goal_comfort["bootstrap_ci_high"] == "0.400000"

    comparability = json.loads((output_dir / "comparability_check.json").read_text())
    assert comparability["status"] == "pass"
    assert comparability["shared_planners"] == ["goal", "orca"]
    assert (
        (output_dir / "README.md")
        .read_text(encoding="utf-8")
        .startswith("# Issue 4195 h600 Aggregation Artifact")
    )
    checksums = (output_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert "planner_metric_summary.csv" in checksums
    assert "source_manifest.json" in checksums


def test_build_artifact_fails_status_on_shared_hash_mismatch(tmp_path: Path) -> None:
    """Shared planners are not comparable when scenario matrix hashes differ."""

    confirm_reports = tmp_path / "confirm" / "reports"
    extended_reports = tmp_path / "extended" / "reports"
    _write_fixture_reports(confirm_reports, scenario_matrix_hash="matrix-a")
    _write_fixture_reports(extended_reports, scenario_matrix_hash="matrix-b")

    result = build_artifact(
        confirm_reports=confirm_reports,
        extended_reports=extended_reports,
        output_dir=tmp_path / "evidence",
        bootstrap_samples=20,
        confidence=0.95,
    )

    assert result["status"] == "comparability_failed"
    comparability = json.loads((tmp_path / "evidence" / "comparability_check.json").read_text())
    assert comparability["status"] == "fail"
    assert comparability["checks"]["scenario_matrix_hash_match"] is False
