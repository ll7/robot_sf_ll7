"""Tests for local-navigation intervention mechanism trace schema validation and classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.mechanism_trace import (
    SCHEMA_VERSION,
    MechanismTraceValidationError,
    classify_mechanism_trace_row,
    emit_orca_residual_row,
    emit_orca_residual_rows,
    emit_static_recentering_row,
    emit_topology_guidance_row,
    generate_mechanism_trace_report,
    load_mechanism_trace_schema,
    validate_mechanism_trace_payload,
)


def test_schema_loads_and_is_valid() -> None:
    """The mechanism trace schema must load and be a valid JSON schema."""
    schema = load_mechanism_trace_schema()
    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_example_trace_validates_successfully() -> None:
    """The example trace fixture file must validate successfully against the schema."""
    example_path = Path(__file__).parent / "fixtures" / "mechanism_trace.v1.example.json"
    with example_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    validated = validate_mechanism_trace_payload(payload)
    assert validated["schema_version"] == SCHEMA_VERSION
    assert len(validated["rows"]) == 2
    assert validated["rows"][0]["mechanism_id"] == "static_recentering"
    assert validated["rows"][1]["mechanism_id"] == "topology_guidance"


def test_invalid_payload_throws_error() -> None:
    """Invalid payloads must raise MechanismTraceValidationError."""
    # Missing schema_version
    invalid_1 = {"generated_at_utc": "2026-06-16T17:00:00Z", "rows": []}
    with pytest.raises(MechanismTraceValidationError):
        validate_mechanism_trace_payload(invalid_1)

    # Invalid mechanism_id enum
    invalid_2 = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": [
            {
                "mechanism_id": "invalid_mechanism_name",
                "activation_step": 0,
                "input_condition": None,
                "selected_command": None,
                "command_source": "dw",
                "risk_score": None,
                "route_progress_delta": None,
                "failure_mode": None,
                "trace_uri": None,
                "classification": "inactive",
            }
        ],
    }
    with pytest.raises(MechanismTraceValidationError):
        validate_mechanism_trace_payload(invalid_2)

    # Invalid classification enum
    invalid_3 = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": [
            {
                "mechanism_id": "static_recentering",
                "activation_step": 0,
                "input_condition": None,
                "selected_command": None,
                "command_source": "dw",
                "risk_score": None,
                "route_progress_delta": None,
                "failure_mode": None,
                "trace_uri": None,
                "classification": "unknown_state",
            }
        ],
    }
    with pytest.raises(MechanismTraceValidationError):
        validate_mechanism_trace_payload(invalid_3)

    # Missing required field inside row (e.g. command_source)
    invalid_4 = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": [
            {
                "mechanism_id": "static_recentering",
                "activation_step": 0,
                "input_condition": None,
                "selected_command": None,
                "risk_score": None,
                "route_progress_delta": None,
                "failure_mode": None,
                "trace_uri": None,
                "classification": "inactive",
            }
        ],
    }
    with pytest.raises(MechanismTraceValidationError):
        validate_mechanism_trace_payload(invalid_4)

    # Invalid generated_at_utc format
    invalid_5 = {"schema_version": SCHEMA_VERSION, "generated_at_utc": "not-a-date", "rows": []}
    with pytest.raises(MechanismTraceValidationError):
        validate_mechanism_trace_payload(invalid_5)

    # Invalid selected command shape
    invalid_6 = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": [
            {
                "mechanism_id": "static_recentering",
                "activation_step": 0,
                "input_condition": None,
                "selected_command": [1.0],
                "command_source": "dw",
                "risk_score": None,
                "route_progress_delta": None,
                "failure_mode": None,
                "trace_uri": None,
                "classification": "inactive",
            }
        ],
    }
    with pytest.raises(MechanismTraceValidationError):
        validate_mechanism_trace_payload(invalid_6)


def test_schema_supports_all_issue_contract_mechanisms() -> None:
    """The v1 enum should cover every mechanism family named by issue #2923."""
    schema = load_mechanism_trace_schema()
    mechanism_ids = set(schema["properties"]["rows"]["items"]["properties"]["mechanism_id"]["enum"])
    assert mechanism_ids == {
        "static_recentering",
        "topology_guidance",
        "prediction_risk_gating",
        "orca_residuals",
        "signal_state_logic",
        "amv_actuation_constraints",
    }


def test_report_counts_all_classification_states() -> None:
    """The report should preserve and count every v1 classification state."""
    states = ["inactive", "active-but-irrelevant", "slice-local", "revise", "stop"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": [
            {
                "mechanism_id": "static_recentering",
                "activation_step": index,
                "input_condition": None,
                "selected_command": None if state == "stop" else [1.0, 0.0],
                "command_source": "dynamic_window",
                "risk_score": None,
                "route_progress_delta": None,
                "failure_mode": None,
                "trace_uri": None,
                "classification": state,
            }
            for index, state in enumerate(states)
        ],
    }
    report = generate_mechanism_trace_report(payload)
    assert report["summary"]["counts"] == dict.fromkeys(states, 1)


def test_classify_mechanism_trace_row() -> None:
    """classify_mechanism_trace_row should correctly classify states."""
    # Predefined classification should be respected
    row_predef = {"classification": "stop"}
    assert classify_mechanism_trace_row(row_predef) == "stop"

    # Rule: stop classification when selected_command is None or [0.0, 0.0]
    row_stop_none = {
        "mechanism_id": "static_recentering",
        "selected_command": None,
        "command_source": "dw",
    }
    assert classify_mechanism_trace_row(row_stop_none) == "stop"

    row_stop_zeros = {
        "mechanism_id": "static_recentering",
        "selected_command": [0.0, 0.0],
        "command_source": "dw",
    }
    assert classify_mechanism_trace_row(row_stop_zeros) == "stop"

    # Rule: inactive when input_condition is None/False
    row_inactive = {
        "mechanism_id": "static_recentering",
        "input_condition": None,
        "selected_command": [1.0, 0.0],
        "command_source": "dw",
    }
    assert classify_mechanism_trace_row(row_inactive) == "inactive"

    # Rule: active-but-irrelevant when mechanism is mismatch with selected command source
    row_irrelevant = {
        "mechanism_id": "static_recentering",
        "input_condition": True,
        "selected_command": [1.0, 0.0],
        "command_source": "dynamic_window",
    }
    assert classify_mechanism_trace_row(row_irrelevant) == "active-but-irrelevant"

    # Rule: revise when route progress delta is high
    row_revise = {
        "mechanism_id": "static_recentering",
        "input_condition": True,
        "selected_command": [1.0, 0.0],
        "command_source": "static_recenter",
        "route_progress_delta": 0.6,
    }
    assert classify_mechanism_trace_row(row_revise) == "revise"

    # Rule: slice-local otherwise
    row_slice_local = {
        "mechanism_id": "static_recentering",
        "input_condition": True,
        "selected_command": [1.0, 0.0],
        "command_source": "static_recenter",
        "route_progress_delta": 0.1,
    }
    assert classify_mechanism_trace_row(row_slice_local) == "slice-local"


def test_generate_mechanism_trace_report() -> None:
    """generate_mechanism_trace_report should validate and summarize row counts."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": [
            {
                "mechanism_id": "static_recentering",
                "activation_step": 10,
                "input_condition": None,
                "selected_command": [0.0, 0.0],
                "command_source": "stop",
                "risk_score": 0.9,
                "route_progress_delta": 0.0,
                "failure_mode": "collision",
                "trace_uri": "uri",
                "classification": "stop",
            },
            {
                "mechanism_id": "topology_guidance",
                "activation_step": 20,
                "input_condition": True,
                "selected_command": [1.0, 0.0],
                "command_source": "topology_hypothesis",
                "risk_score": 0.1,
                "route_progress_delta": 0.0,
                "failure_mode": None,
                "trace_uri": "uri",
                "classification": "slice-local",
            },
        ],
    }
    report = generate_mechanism_trace_report(payload)
    assert report["schema_version"] == "mechanism_trace_report.v1"
    assert report["summary"]["total_rows"] == 2
    assert report["summary"]["counts"]["stop"] == 1
    assert report["summary"]["counts"]["slice-local"] == 1


def test_emit_static_recentering_row() -> None:
    """emit_static_recentering_row should properly translate planner decisions."""
    decision_active = {
        "selected_command": [0.0, 0.2],
        "selected_source": "static_recenter",
        "selected_terms": {"static_recenter": 1.0},
        "risk_score": 0.3,
    }
    row = emit_static_recentering_row(5, decision_active, progress_delta=-0.02)
    validate_mechanism_trace_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": "2026-06-16T17:00:00Z",
            "rows": [row],
        }
    )
    assert row["mechanism_id"] == "static_recentering"
    assert row["activation_step"] == 5
    assert row["classification"] == "revise"
    assert row["selected_command"] == [0.0, 0.2]
    assert row["route_progress_delta"] == -0.02

    decision_inactive = {
        "selected_command": [1.2, 0.0],
        "selected_source": "dynamic_window",
        "selected_terms": {"static_recenter": 0.0},
        "risk_score": 0.0,
    }
    row_inactive = emit_static_recentering_row(6, decision_inactive)
    assert row_inactive["classification"] == "inactive"


def test_emit_topology_guidance_row() -> None:
    """emit_topology_guidance_row should properly translate planner decisions."""
    decision_active = {
        "selected_command": [1.0, -0.1],
        "selected_source": "topology_hypothesis",
        "topology_guided": {"active": True, "selected_hypothesis_id": "route_2"},
        "risk_score": 0.1,
    }
    row = emit_topology_guidance_row(12, decision_active, progress_delta=0.4)
    validate_mechanism_trace_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": "2026-06-16T17:00:00Z",
            "rows": [row],
        }
    )
    assert row["mechanism_id"] == "topology_guidance"
    assert row["activation_step"] == 12
    assert row["classification"] == "revise"
    assert row["input_condition"]["selected_hypothesis_id"] == "route_2"


def test_emit_orca_residual_row() -> None:
    """emit_orca_residual_row should emit residual rows with residual diagnostics."""
    decision_active = {
        "selected_command": [0.7, -0.2],
        "selected_source": "prior_residual",
        "selected_score": 0.31,
        "action_adaptation": {
            "mode": "prior_residual",
            "raw_residual_action": [0.25, -0.3],
            "bounded_residual_action": [0.2, -0.25],
            "residual_bounds": {
                "linear": 0.2,
                "angular": 0.3,
            },
            "residual_clipped": True,
        },
        "intervened": True,
        "intervention_reason": "bounded_prior_residual_improved_short_horizon_safety",
    }
    row = emit_orca_residual_row(9, decision_active, progress_delta=0.12)
    validate_mechanism_trace_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": "2026-06-16T17:00:00Z",
            "rows": [row],
        }
    )
    assert row["mechanism_id"] == "orca_residuals"
    assert row["activation_step"] == 9
    assert row["command_source"] == "prior_residual_safe"
    assert row["selected_command"] == [0.7, -0.2]
    assert row["input_condition"]["original_command_source"] == "prior_residual"
    assert row["input_condition"]["residual_norm"] == pytest.approx((0.25**2 + (-0.3) ** 2) ** 0.5)
    assert row["failure_mode"] == "bounded_prior_residual_improved_short_horizon_safety"
    assert row["classification"] == "slice-local"

    row_inactive = emit_orca_residual_row(
        10,
        {
            "selected_source": "dynamic_window",
            "selected_command": [1.0, 0.0],
        },
    )
    assert row_inactive["mechanism_id"] == "orca_residuals"
    assert row_inactive["classification"] == "inactive"
    assert row_inactive["selected_command"] == [1.0, 0.0]

    row_explicit_none = emit_orca_residual_row(
        10,
        {
            "selected_source": None,
            "selected_command": [1.0, 0.0],
            "action_adaptation": {"mode": None},
        },
    )
    assert row_explicit_none["classification"] == "inactive"
    assert row_explicit_none["command_source"] == "unknown"

    row_revise = emit_orca_residual_row(
        11,
        {
            "selected_source": "prior_residual_safe",
            "selected_command": [0.4, 0.1],
            "action_adaptation": {
                "mode": "prior_residual",
                "raw_residual_action": [0.1, 0.0],
                "bounded_residual_action": [0.1, 0.0],
                "residual_bounds": {
                    "linear": 0.2,
                    "angular": 0.3,
                },
            },
            "selected_score": 0.22,
        },
        progress_delta=1.0,
    )
    assert row_revise["classification"] == "revise"


def test_emit_orca_residual_rows_from_fixture() -> None:
    """Emit ORCA residual rows from a durable tracked planner-decision fixture."""
    fixture_path = (
        Path(__file__).parent / "fixtures" / "orca_residuals_planner_decision_trace.v1.json"
    )
    with fixture_path.open("r", encoding="utf-8") as f:
        planner_trace = json.load(f)

    rows = emit_orca_residual_rows(planner_trace, trace_uri=str(fixture_path))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-06-16T17:00:00Z",
        "rows": rows,
    }
    validate_mechanism_trace_payload(payload)
    report = generate_mechanism_trace_report(payload)

    assert len(rows) == 4
    assert rows[0]["mechanism_id"] == "orca_residuals"
    assert rows[0]["classification"] == "slice-local"
    assert rows[0]["command_source"] == "prior_residual_safe"
    assert rows[2]["classification"] == "revise"
    assert rows[3]["classification"] == "inactive"
    assert report["summary"]["counts"]["revise"] == 1
    assert report["summary"]["counts"]["slice-local"] == 2
    assert report["summary"]["counts"]["inactive"] == 1
