"""Tests for SNQI scalarization-sensitivity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    SCALARIZATION_SENSITIVITY_SCHEMA,
    SENSITIVITY_PREFLIGHT_BLOCKED,
    SENSITIVITY_PREFLIGHT_MALFORMED,
    SENSITIVITY_PREFLIGHT_READY,
    build_scalarization_sensitivity_report,
    classify_scalarization_sensitivity_inputs,
    format_pareto_svg,
    input_file_provenance,
    load_baseline_mapping,
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


def _preflight_episodes() -> list[dict]:
    records = []
    for horizon in (500, 600):
        records.extend(
            [
                {
                    "scenario_id": "crossing",
                    "horizon": horizon,
                    "planner_key": "fast-risky",
                    "metrics": {
                        "success": 1,
                        "time_to_goal_norm": 0.1,
                        "collisions": 1,
                        "near_misses": 0,
                        "comfort_exposure": 0.0,
                    },
                },
                {
                    "scenario_id": "crossing",
                    "horizon": horizon,
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
                    "scenario_id": "crossing",
                    "horizon": horizon,
                    "planner_key": "middle",
                    "metrics": {
                        "success": 1,
                        "time_to_goal_norm": 0.5,
                        "collisions": 0,
                        "near_misses": 1,
                        "comfort_exposure": 0.2,
                    },
                },
            ]
        )
    return records


def test_preflight_ready_for_rectangular_scenario_horizon_terms() -> None:
    """Ready input has planners, complete scenario-horizon cells, and SNQI terms."""
    report = classify_scalarization_sensitivity_inputs(
        _preflight_episodes(),
        weights=_weights(),
        baseline=_baseline(),
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_READY
    assert report["ready"] is True
    assert report["planner_count"] == 3
    assert report["scenario_horizon_count"] == 2
    assert report["issues"] == []


def test_preflight_blocks_missing_scenario_horizon_and_non_rectangular_table() -> None:
    """Missing scenario-horizon evidence and table cells block sensitivity export."""
    records = _preflight_episodes()
    records[0].pop("horizon")
    records = records[:-1]

    report = classify_scalarization_sensitivity_inputs(
        records,
        weights=_weights(),
        baseline=_baseline(),
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_BLOCKED
    codes = {issue["code"] for issue in report["issues"]}
    assert "missing_scenario_horizon" in codes
    assert "non_rectangular_planner_table" in codes


def test_preflight_malformed_for_non_finite_term() -> None:
    """Non-finite required SNQI terms are malformed rather than merely blocked."""
    records = _preflight_episodes()
    records[0]["metrics"]["time_to_goal_norm"] = "nan"

    report = classify_scalarization_sensitivity_inputs(
        records,
        weights=_weights(),
        baseline=_baseline(),
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(issue["code"] == "non_finite_required_term" for issue in report["issues"])


def test_preflight_malformed_for_non_mapping_baseline_value() -> None:
    """A baseline metric value that is not a mapping is classified malformed, not crashed."""
    baseline = _baseline()
    baseline["collisions"] = 1.0  # scalar instead of {"med": ..., "p95": ...}

    report = classify_scalarization_sensitivity_inputs(
        _preflight_episodes(),
        weights=_weights(),
        baseline=baseline,
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(issue["code"] == "malformed_baseline_stats" for issue in report["issues"])


def test_preflight_malformed_for_degenerate_baseline_range() -> None:
    """Degenerate normalized baseline stats cannot support sensitivity export."""
    baseline = _baseline()
    baseline["collisions"] = {"med": 1.0, "p95": 1.0}

    report = classify_scalarization_sensitivity_inputs(
        _preflight_episodes(),
        weights=_weights(),
        baseline=baseline,
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(issue["code"] == "degenerate_baseline_range" for issue in report["issues"])


def test_preflight_malformed_for_non_finite_baseline_stats() -> None:
    """Non-finite normalized baseline stats fail closed before export."""
    baseline = _baseline()
    baseline["near_misses"] = {"med": 0.0, "p95": float("nan")}

    report = classify_scalarization_sensitivity_inputs(
        _preflight_episodes(),
        weights=_weights(),
        baseline=baseline,
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(issue["code"] == "non_finite_baseline_stats" for issue in report["issues"])


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


def test_report_export_refuses_missing_normalized_time_term() -> None:
    """Direct report export fails closed when normalized time input is absent."""

    records = _episodes()
    del records[0]["metrics"]["time_to_goal_norm"]

    with pytest.raises(ValueError, match="missing required SNQI term 'time_to_goal_norm'"):
        build_scalarization_sensitivity_report(
            records,
            weights=_weights(),
            baseline=_baseline(),
        )


def test_report_export_refuses_non_finite_normalized_time_term() -> None:
    """Direct report export fails closed when normalized time input is invalid."""

    records = _episodes()
    records[0]["metrics"]["time_to_goal_norm"] = "nan"

    with pytest.raises(ValueError, match="non-finite required SNQI term 'time_to_goal_norm'"):
        build_scalarization_sensitivity_report(
            records,
            weights=_weights(),
            baseline=_baseline(),
        )


def test_preflight_blocks_missing_active_optional_normalized_term() -> None:
    """Weighted optional normalized terms must be present before export."""

    records = _preflight_episodes()
    weights = _weights()
    weights["w_jerk"] = 0.25

    report = classify_scalarization_sensitivity_inputs(
        records,
        weights=weights,
        baseline=_baseline(),
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_BLOCKED
    assert any(issue["code"] == "missing_weighted_optional_term" for issue in report["issues"])


def test_preflight_reports_non_mapping_weights_without_crashing() -> None:
    """Non-mapping weights stay a graceful malformed report, not an AttributeError."""

    report = classify_scalarization_sensitivity_inputs(
        _preflight_episodes(),
        weights=None,
        baseline=_baseline(),
    )

    assert report["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(issue["code"] == "weights_not_mapping" for issue in report["issues"])


def test_report_export_refuses_missing_active_optional_normalized_term() -> None:
    """Direct export fails closed instead of defaulting active optional terms to zero."""

    records = _episodes()
    weights = _weights()
    weights["w_force_exceed"] = 0.5

    with pytest.raises(ValueError, match="missing weighted SNQI term 'force_exceed_events'"):
        build_scalarization_sensitivity_report(
            records,
            weights=weights,
            baseline=_baseline(),
        )


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
    disagreement_csv_text = artifacts.decision_disagreement_csv_path.read_text(encoding="utf-8")
    assert "comparison,left_order,right_order,pairwise_reversal_count" in disagreement_csv_text
    assert "base_snqi_vs_constraints_first" in disagreement_csv_text
    # The order columns must be populated, not silently blank, so the table shows which
    # planners' rankings disagree (not just the aggregate reversal count).
    base_order = " > ".join(report["base_snqi_order"])
    constraints_order = " > ".join(report["constraints_first_order"])
    assert base_order in disagreement_csv_text
    assert constraints_order in disagreement_csv_text
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "not benchmark evidence" in markdown
    svg = artifacts.svg_path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "safe-slow" in svg


def test_report_records_input_provenance_for_auditable_weight_sweeps(tmp_path: Path) -> None:
    """Report payload can carry input hashes without changing SNQI semantics."""

    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(_weights(), sort_keys=True), encoding="utf-8")

    report = build_scalarization_sensitivity_report(
        _episodes(),
        weights=_weights(),
        baseline=_baseline(),
        sweep_factors=[1.0],
        input_provenance={"weights": input_file_provenance(weights_path)},
    )

    provenance = report["inputs"]["provenance"]["weights"]
    assert provenance["path"] == str(weights_path)
    assert len(provenance["sha256"]) == 64


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
    payload = json.loads(Path(stdout["json"]).read_text(encoding="utf-8"))
    assert payload["inputs"]["provenance"]["weights"]["path"] == str(weights_path)
    assert len(payload["inputs"]["provenance"]["weights"]["sha256"]) == 64


def test_cli_preflight_only_blocks_invalid_inputs(tmp_path: Path, capsys) -> None:
    """Preflight mode reports blocked inputs without writing diagnostic artifacts."""
    records = _preflight_episodes()
    records[0].pop("scenario_id")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(_weights()), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline()), encoding="utf-8")

    exit_code = scalarization_cli_main(
        [
            "--episodes",
            str(episodes_path),
            "--weights",
            str(weights_path),
            "--baseline",
            str(baseline_path),
            "--preflight-only",
        ]
    )

    assert exit_code == 2
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["status"] == SENSITIVITY_PREFLIGHT_BLOCKED
    assert any(issue["code"] == "missing_scenario_horizon" for issue in stdout["issues"])


def test_baseline_loader_refuses_missing_normalized_fixture_input(tmp_path: Path) -> None:
    """Baseline fixture loader fails closed before export on missing normalized stats."""
    baseline = _baseline()
    baseline.pop("jerk_mean")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required normalized metric 'jerk_mean'"):
        load_baseline_mapping(baseline_path)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"med": 0.0}, "must provide finite med/p95 values"),
        ({"med": 0.0, "p95": "nan"}, "has non-finite med/p95 values"),
        ({"med": 1.0, "p95": 1.0}, "must satisfy p95 > med"),
    ],
)
def test_baseline_loader_refuses_invalid_normalized_stats(
    tmp_path: Path, entry: dict[str, float | str], match: str
) -> None:
    """Baseline fixture loader validates normalized-stat shape and range."""
    baseline = _baseline()
    baseline["near_misses"] = entry
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_baseline_mapping(baseline_path)


def test_cli_refuses_malformed_baseline_without_writing_artifacts(tmp_path: Path, capsys) -> None:
    """CLI refuses malformed normalized baseline fixtures before artifacts exist."""
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        "\n".join(json.dumps(record) for record in _episodes()) + "\n",
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(_weights()), encoding="utf-8")
    baseline = _baseline()
    baseline["collisions"] = 1.0
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="baseline metric 'collisions' must provide med/p95"):
        scalarization_cli_main(
            [
                "--episodes",
                str(episodes_path),
                "--weights",
                str(weights_path),
                "--baseline",
                str(baseline_path),
                "--output-dir",
                str(output_dir),
            ]
        )

    capsys.readouterr()
    assert not output_dir.exists()
