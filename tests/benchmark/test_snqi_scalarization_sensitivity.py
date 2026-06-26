"""Tests for SNQI scalarization-sensitivity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    SCALARIZATION_SENSITIVITY_SCHEMA,
    build_scalarization_sensitivity_report,
    format_pareto_svg,
    load_jsonl,
    write_diagnostic_artifacts,
)
from scripts.tools.analyze_snqi_scalarization_sensitivity import main as scalarization_cli_main


def _episodes() -> list[dict]:
    return [
        {
            "planner_key": "fast-risky",
            "metrics": {
                "success": 1,
                "time_to_goal_norm": 0.05,
                "collisions": 1,
                "near_misses": 0,
                "comfort_exposure": 0.0,
            },
        },
        {
            "planner_key": "fast-risky",
            "metrics": {
                "success": 1,
                "time_to_goal_norm": 0.05,
                "collisions": 1,
                "near_misses": 0,
                "comfort_exposure": 0.0,
            },
        },
        {
            "planner_key": "safe-slow",
            "metrics": {
                "success": 1,
                "time_to_goal_norm": 0.8,
                "collisions": 0,
                "near_misses": 0,
                "comfort_exposure": 0.1,
            },
        },
        {
            "planner_key": "safe-slow",
            "metrics": {
                "success": 1,
                "time_to_goal_norm": 0.8,
                "collisions": 0,
                "near_misses": 0,
                "comfort_exposure": 0.1,
            },
        },
        {
            "planner_key": "middle",
            "metrics": {
                "success": 1,
                "time_to_goal_norm": 0.9,
                "collisions": 0,
                "near_misses": 0,
                "comfort_exposure": 0.2,
            },
        },
        {
            "planner_key": "middle",
            "metrics": {
                "success": 1,
                "time_to_goal_norm": 0.9,
                "collisions": 0,
                "near_misses": 0,
                "comfort_exposure": 0.2,
            },
        },
    ]


def _weights() -> dict[str, float]:
    return {
        "w_success": 0.2,
        "w_time": 1.0,
        "w_collisions": 0.1,
        "w_near": 0.1,
        "w_comfort": 0.0,
        "w_force_exceed": 0.0,
        "w_jerk": 0.0,
    }


def _baseline() -> dict[str, dict[str, float]]:
    return {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }


def test_report_exports_decision_disagreement_and_weight_reversals() -> None:
    """Sensitivity report exposes rank disagreement and scalarization reversals."""

    report = build_scalarization_sensitivity_report(
        _episodes(),
        weights=_weights(),
        baseline=_baseline(),
        sweep_factors=[0.0, 1.0, 10.0],
    )

    assert report["schema_version"] == SCALARIZATION_SENSITIVITY_SCHEMA
    assert report["evidence_kind"] == "analysis_artifact_only"
    assert report["base_snqi_order"][0] == "fast-risky"
    assert report["constraints_first_order"][0] == "safe-slow"
    assert report["decision_disagreement"]["winner_disagreement"] is True
    assert report["summary"]["decision_disagreement_rate"] > 0.0
    assert report["weight_sweep"]["w_collisions"][-1]["winner_changed"] is True
    assert report["weight_zero_ablation"]["w_time"]["pairwise_reversal_count_vs_base"] > 0
    assert report["term_dominance"][0]["component"] == "w_time"
    assert {point["planner"] for point in report["pareto_front"]["points"]} == {
        "fast-risky",
        "safe-slow",
    }


def test_artifact_writer_creates_report_ready_files(tmp_path: Path) -> None:
    """Artifact writer emits report-ready JSON, CSV, Markdown, and SVG."""

    report = build_scalarization_sensitivity_report(
        _episodes(),
        weights=_weights(),
        baseline=_baseline(),
        sweep_factors=[0.0, 1.0],
    )

    artifacts = write_diagnostic_artifacts(report, tmp_path)

    assert json.loads(artifacts.json_path.read_text(encoding="utf-8"))["schema_version"]
    csv_text = artifacts.csv_path.read_text(encoding="utf-8")
    assert "planner,snqi_rank,constraints_first_rank" in csv_text
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "not benchmark evidence" in markdown
    svg = artifacts.svg_path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "safe-slow" in svg


def test_load_jsonl_rejects_non_object_records(tmp_path: Path) -> None:
    """JSONL loader fails closed on non-object records."""

    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(_episodes()[0]) + "\n[]\n", encoding="utf-8")

    try:
        load_jsonl(path)
    except ValueError as exc:
        assert "expected a JSON object" in str(exc)
    else:
        raise AssertionError("expected non-object JSONL records to fail closed")


def test_svg_renderer_marks_pareto_points() -> None:
    """SVG renderer includes a title and front styling."""

    report = build_scalarization_sensitivity_report(
        _episodes(),
        weights=_weights(),
        baseline=_baseline(),
        sweep_factors=[1.0],
    )

    svg = format_pareto_svg(report)

    assert "<title>SNQI scalarization sensitivity Pareto diagnostic</title>" in svg
    assert "#1f77b4" in svg


def test_cli_writes_scalarization_artifacts(tmp_path: Path, capsys) -> None:
    """CLI writes the same report-ready artifact family."""

    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        "\n".join(json.dumps(record) for record in _episodes()) + "\n",
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(_weights()), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline()), encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = scalarization_cli_main(
        [
            "--episodes",
            str(episodes_path),
            "--weights",
            str(weights_path),
            "--baseline",
            str(baseline_path),
            "--output-dir",
            str(output_dir),
            "--sweep-factors",
            "0",
            "1",
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert Path(stdout["json"]).exists()
    assert Path(stdout["csv"]).exists()
    assert Path(stdout["markdown"]).exists()
    assert Path(stdout["svg"]).exists()
