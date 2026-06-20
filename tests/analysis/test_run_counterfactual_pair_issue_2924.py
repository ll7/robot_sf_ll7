"""Tests for the Issue #2924 counterfactual-pair runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import jsonschema
import pytest
import yaml

from scripts.analysis import run_counterfactual_pair_issue_2924 as runner

if TYPE_CHECKING:
    from pathlib import Path


def _mechanism_trace_payload(classification: str) -> dict[str, Any]:
    """Return a minimal mechanism_trace.v1 payload for prediction-risk gating."""

    return {
        "schema_version": "mechanism_trace.v1",
        "generated_at_utc": "2026-06-21T00:00:00Z",
        "rows": [
            {
                "mechanism_id": "prediction_risk_gating",
                "activation_step": 4,
                "input_condition": classification != "inactive",
                "selected_command": [0.2, 0.0],
                "command_source": "prediction_risk_gate",
                "risk_score": 0.82,
                "route_progress_delta": 0.2,
                "failure_mode": None,
                "trace_uri": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json",
                "classification": classification,
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON and return the path."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _observation(
    *,
    config_id: str,
    clearance: float,
    status: str = "available",
    readiness_status: str = "native",
    scenario: str = "issue_2924_prediction_risk_occluded_emergence",
) -> dict[str, Any]:
    """Return a compact observed-result fixture."""

    return {
        "schema_version": "counterfactual_pair_observation.v1",
        "config_id": config_id,
        "invariant_fields": {
            "scenario": scenario,
            "seed": 111,
            "planner": "hybrid_rule_v0_minimal",
            "artifact": "issue_2924_prediction_risk_fixture.v1",
        },
        "execution_mode": "native",
        "readiness_status": readiness_status,
        "availability_status": status,
        "metrics": {
            "min_clearance_m": clearance,
            "collision_count": 0,
        },
    }


def _manifest(tmp_path: Path) -> dict[str, Any]:
    """Create a valid manifest payload with temp fixture paths."""

    baseline_result = _write_json(
        tmp_path / "baseline_result.json",
        _observation(config_id="baseline", clearance=0.18),
    )
    intervention_result = _write_json(
        tmp_path / "intervention_result.json",
        _observation(config_id="intervention", clearance=0.42),
    )
    baseline_trace = _write_json(
        tmp_path / "baseline_mechanism_trace.json",
        _mechanism_trace_payload("inactive"),
    )
    intervention_trace = _write_json(
        tmp_path / "intervention_mechanism_trace.json",
        _mechanism_trace_payload("revise"),
    )
    invariants = {
        "scenario": "issue_2924_prediction_risk_occluded_emergence",
        "seed": 111,
        "planner": "hybrid_rule_v0_minimal",
        "artifact": "issue_2924_prediction_risk_fixture.v1",
    }
    return {
        "schema_version": "counterfactual_pair.v1",
        "pair_id": "issue_2924_prediction_risk_fixture",
        "evidence_tier": "analysis_only",
        "claim_boundary": "analysis_only_not_benchmark_or_paper_grade_evidence",
        "invariant_fields": invariants,
        "baseline_config": {
            "config_id": "baseline",
            "result_path": baseline_result.as_posix(),
            "mechanism_trace_path": baseline_trace.as_posix(),
            "trace_export_path": (
                "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
                "occluded_emergence_episode_0000.json"
            ),
            "invariant_fields": invariants,
        },
        "intervention_config": {
            "config_id": "intervention",
            "result_path": intervention_result.as_posix(),
            "mechanism_trace_path": intervention_trace.as_posix(),
            "trace_export_path": (
                "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
                "occluded_emergence_left_close_episode_0000.json"
            ),
            "invariant_fields": invariants,
        },
        "expected_mechanism": "prediction_risk_gating",
        "expected_metric_direction": {
            "metric": "min_clearance_m",
            "direction": "increase",
            "min_delta": 0.05,
        },
    }


def test_schema_is_valid_and_accepts_minimal_manifest(tmp_path: Path) -> None:
    """The counterfactual_pair.v1 schema should validate the runner manifest."""

    schema = runner.load_counterfactual_pair_schema()
    jsonschema.Draft202012Validator.check_schema(schema)

    payload = _manifest(tmp_path)
    runner.validate_counterfactual_pair_manifest(payload)


def test_runner_evaluates_pair_and_writes_report(tmp_path: Path) -> None:
    """A matched prediction-risk pair emits activation/outcome deltas and a survived verdict."""

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest(tmp_path), sort_keys=False), encoding="utf-8")
    output_path = tmp_path / "result.json"
    report_path = tmp_path / "report.md"

    result = runner.run_counterfactual_pair(
        manifest_path=manifest_path,
        output_path=output_path,
        markdown_output_path=report_path,
    )

    assert output_path.is_file()
    assert report_path.is_file()
    assert result["schema_version"] == "counterfactual_pair_run.v1"
    assert result["pair_id"] == "issue_2924_prediction_risk_fixture"
    assert result["activation_delta"]["active_count_delta"] == 1
    assert result["outcome_delta"]["metric"] == "min_clearance_m"
    assert result["outcome_delta"]["delta"] == pytest.approx(0.24)
    assert result["hypothesis_verdict"] == "survived"
    assert result["pair_result"]["activation_delta"] == 1
    assert "Hypothesis verdict: `survived`" in report_path.read_text(encoding="utf-8")


def test_runner_fails_clearly_on_invariant_mismatch(tmp_path: Path) -> None:
    """Baseline/intervention invariant mismatches must fail before deltas are computed."""

    payload = _manifest(tmp_path)
    payload["intervention_config"]["invariant_fields"] = {
        **payload["intervention_config"]["invariant_fields"],
        "scenario": "different_scenario",
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(runner.CounterfactualPairRunnerError, match="invariant mismatch.*scenario"):
        runner.run_counterfactual_pair(manifest_path=manifest_path)


def test_runner_fails_closed_on_unavailable_or_degraded_rows(tmp_path: Path) -> None:
    """Fallback, degraded, and not_available rows are excluded instead of producing evidence."""

    payload = _manifest(tmp_path)
    degraded_result = _write_json(
        tmp_path / "intervention_degraded.json",
        _observation(
            config_id="intervention",
            clearance=0.42,
            status="not_available",
            readiness_status="degraded",
        ),
    )
    payload["intervention_config"]["result_path"] = degraded_result.as_posix()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        runner.CounterfactualPairRunnerError,
        match="fail-closed.*intervention.*readiness_status=degraded.*availability_status=not_available",
    ):
        runner.run_counterfactual_pair(manifest_path=manifest_path)
