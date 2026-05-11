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


def test_run_planner_inclusion_check_appends_when_resuming(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resume mode should preserve existing episode files by passing append=True to the runner."""

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump([{"name": "smoke", "repeats": 1}]), encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run_batch(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        Path(kwargs["out_path"]).write_text(
            json.dumps(
                {
                    "episode_id": "e1",
                    "scenario_id": "smoke",
                    "seed": 0,
                    "algo": "goal",
                    "metrics": {"success": 1.0, "collisions": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"total_jobs": 1, "written": 1, "failures": []}

    monkeypatch.setattr(planner_inclusion, "run_batch", fake_run_batch)

    planner_inclusion.run_planner_inclusion_check(
        algo="goal",
        matrix=matrix,
        schema=tmp_path / "schema.json",
        output_dir=tmp_path / "out",
        resume=True,
    )

    assert captured["resume"] is True
    assert captured["append"] is True


def test_inclusion_report_handles_deep_nested_non_finite_payloads(tmp_path: Path) -> None:
    """Deep aggregate payloads should not rely on recursion to find non-finite values."""

    aggregates: dict[str, Any] = {"goal": {"success": {"mean": 1.0}, "collisions": {"mean": 0.0}}}
    cursor: dict[str, Any] = aggregates
    depth = 1200
    for index in range(depth):
        child: dict[str, Any] = {}
        cursor[f"level_{index}"] = child
        cursor = child
    cursor["bad"] = float("nan")

    report = build_inclusion_report(
        algo="goal",
        algo_config=None,
        benchmark_profile="experimental",
        matrix=Path("matrix.yaml"),
        schema=Path("schema.json"),
        output_dir=tmp_path,
        episodes_path=tmp_path / "episodes.jsonl",
        summary={"total_jobs": 1, "written": 1, "failures": []},
        aggregates=aggregates,
        runtime_sec=1.0,
        criteria=InclusionCriteria(),
    )

    assert report["checks"]["no_nan_aggregates"]["passed"] is False
    assert len(report["checks"]["no_nan_aggregates"]["observed"]["non_finite_paths"]) == 1


def test_planner_inclusion_cli_serializes_nonstandard_report_values(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """CLI output should coerce Path-like and scalar-wrapper values into JSON-safe types."""

    class FakeScalar:
        def item(self) -> float:
            return 1.25

    def fake_run_planner_inclusion_check(**_kwargs: Any) -> dict[str, Any]:
        return {
            "decision": "pass",
            "artifacts": {"report_json": tmp_path / "report.json"},
            "metrics": {"runtime_sec": FakeScalar(), "bad_value": float("nan")},
        }

    monkeypatch.setattr(
        "robot_sf.benchmark.cli.run_planner_inclusion_check",
        fake_run_planner_inclusion_check,
    )

    rc = cli_main(
        [
            "planner-inclusion-check",
            "--algo",
            "goal",
            "--matrix",
            str(tmp_path / "matrix.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["artifacts"]["report_json"] == str(tmp_path / "report.json")
    assert payload["metrics"]["runtime_sec"] == 1.25
    assert payload["metrics"]["bad_value"] is None
