"""Tests for planner inclusion-check reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark import planner_inclusion
from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.planner_inclusion import InclusionCriteria, build_inclusion_report


def _aggregate(success: float, collisions: float) -> dict[str, Any]:
    """Build a minimal aggregate payload keyed like benchmark aggregation output."""
    return {
        "goal": {
            "success": {"mean": success, "median": success, "p95": success},
            "collisions": {"mean": collisions, "median": collisions, "p95": collisions},
        }
    }


def test_inclusion_report_passes_for_clean_reference_slice(tmp_path: Path) -> None:
    """A complete finite run above thresholds should produce a pass decision."""
    report = build_inclusion_report(
        algo="goal",
        algo_config=None,
        benchmark_profile="experimental",
        matrix=Path("matrix.yaml"),
        schema=Path("schema.json"),
        output_dir=tmp_path,
        episodes_path=tmp_path / "episodes.jsonl",
        summary={"total_jobs": 2, "written": 2, "failures": []},
        aggregates=_aggregate(success=1.0, collisions=0.0),
        runtime_sec=1.5,
        criteria=InclusionCriteria(min_episodes=1, min_success_rate=0.5),
    )

    assert report["schema_version"] == "planner-inclusion-check.v1"
    assert report["decision"] == "pass"
    assert report["failure_reasons"] == []
    assert report["checks"]["schema_valid"]["passed"] is True
    assert report["checks"]["no_nan_aggregates"]["passed"] is True
    assert report["metrics"]["success_rate"] == 1.0


def test_inclusion_report_fails_closed_with_explicit_reasons(tmp_path: Path) -> None:
    """Gate failures should be reviewable without reading runner logs."""
    report = build_inclusion_report(
        algo="goal",
        algo_config=None,
        benchmark_profile="experimental",
        matrix=Path("matrix.yaml"),
        schema=Path("schema.json"),
        output_dir=tmp_path,
        episodes_path=tmp_path / "episodes.jsonl",
        summary={"total_jobs": 2, "written": 1, "failures": [{"error": "boom"}]},
        aggregates=_aggregate(success=float("nan"), collisions=0.25),
        runtime_sec=61.0,
        criteria=InclusionCriteria(
            min_episodes=2,
            min_success_rate=0.5,
            max_collision_rate=0.0,
            max_runtime_sec=60.0,
        ),
    )

    assert report["decision"] == "revise"
    assert report["checks"]["schema_valid"]["passed"] is False
    assert report["checks"]["no_nan_aggregates"]["passed"] is False
    assert report["checks"]["bounded_runtime"]["passed"] is False
    assert report["checks"]["minimum_success_rate"]["passed"] is False
    assert report["checks"]["maximum_collision_rate"]["passed"] is False
    assert any("schema_valid" in reason for reason in report["failure_reasons"])


def test_planner_inclusion_cli_writes_report(monkeypatch, tmp_path: Path, capsys) -> None:
    """CLI command should run the checker and write a JSON report artifact."""
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump([{"name": "smoke", "repeats": 1}]), encoding="utf-8")

    def fake_run_batch(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["horizon"] == 250
        episodes_path = Path(kwargs["out_path"])
        record = {
            "episode_id": "e1",
            "scenario_id": "smoke",
            "seed": 0,
            "algo": "goal",
            "metrics": {"success": 1.0, "collisions": 0.0},
        }
        episodes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return {"total_jobs": 1, "written": 1, "failures": []}

    monkeypatch.setattr(planner_inclusion, "run_batch", fake_run_batch)

    rc = cli_main(
        [
            "planner-inclusion-check",
            "--algo",
            "goal",
            "--matrix",
            str(matrix),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-runtime-sec",
            "60",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    stdout_report = json.loads(captured.out)
    report_path = Path(stdout_report["artifacts"]["report_json"])
    assert report_path.exists()
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk_report["decision"] == "pass"
    assert disk_report["checks"]["schema_valid"]["passed"] is True
