"""Tests for the ORCA residual mechanism-trace emission script."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from robot_sf.benchmark.mechanism_trace import SCHEMA_VERSION
from scripts.tools.emit_orca_residual_mechanism_trace import (
    build_orca_residual_mechanism_trace_payload,
    load_planner_decision_trace,
    main,
    write_mechanism_trace_payload,
)


def test_build_orca_residual_mechanism_trace_payload_from_fixture() -> None:
    """Fixture-backed row emission should produce a validated mechanism-trace payload."""
    fixture_path = (
        Path(__file__).parents[1]
        / "benchmark"
        / "fixtures"
        / "orca_residuals_planner_decision_trace.v1.json"
    )
    planner_trace = load_planner_decision_trace(fixture_path)

    payload = build_orca_residual_mechanism_trace_payload(
        planner_trace,
        trace_uri=str(fixture_path),
        generated_at="2026-06-16T17:00:00Z",
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_at_utc"] == "2026-06-16T17:00:00Z"
    assert len(payload["rows"]) == 4
    assert payload["rows"][0]["mechanism_id"] == "orca_residuals"
    assert payload["rows"][0]["activation_step"] == 101


def test_emit_orca_residual_mechanism_trace_script_jsonl(tmp_path: Path) -> None:
    """Script entrypoint should write JSONL rows for downstream tooling."""
    fixture_path = (
        Path(__file__).parents[1]
        / "benchmark"
        / "fixtures"
        / "orca_residuals_planner_decision_trace.v1.json"
    )
    output_path = tmp_path / "orca_residual_mechanism_trace.jsonl"
    report_path = tmp_path / "orca_residual_emission_report.json"
    exit_code = main(
        [
            "--planner-decision-trace",
            str(fixture_path),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
            "--trace-uri",
            "fixtures/orca_residuals_planner_decision_trace.v1.json",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    first_row = json.loads(lines[0])
    assert first_row["mechanism_id"] == "orca_residuals"
    assert first_row["trace_uri"] == "fixtures/orca_residuals_planner_decision_trace.v1.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["claim_boundary"] == "diagnostic_only"
    assert report["rows_count"] == 4
    assert report["classification_counts"] == {"inactive": 1, "revise": 1, "slice-local": 2}
    assert report["source_commit"]
    assert shlex.split(report["command"])[0:4] == [
        "uv",
        "run",
        "python",
        "scripts/tools/emit_orca_residual_mechanism_trace.py",
    ]


def test_nested_fallback_adapter_in_emitted_payload(tmp_path: Path) -> None:
    """Nested ``fallback_controller_state.action_adaptation`` should be honored."""
    planner_trace = [
        {
            "selected_source": "dynamic_window",
            "selected_command": [0.8, 0.1],
            "route_progress_from_start_m": 0.0,
            "fallback_controller_state": {
                "action_adaptation": {
                    "mode": "prior_residual",
                    "raw_residual_action": [0.5, 0.0],
                    "bounded_residual_action": [0.5, 0.0],
                    "residual_bounds": {"linear": 0.2, "angular": 0.3},
                }
            },
        }
    ]
    payload = build_orca_residual_mechanism_trace_payload(
        planner_trace,
        trace_uri="fixture://nested-fallback",
        generated_at="2026-06-16T17:00:00Z",
    )
    assert payload["rows"][0]["command_source"] == "prior_residual_safe"
    assert payload["rows"][0]["trace_uri"] == "fixture://nested-fallback"

    output_path = tmp_path / "nested_fallback_mechanism_trace.json"
    write_mechanism_trace_payload(payload, output_path, "json")
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == payload
