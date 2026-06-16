"""Mechanism trace validation, classification, and report generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_VERSION = "mechanism_trace.v1"


class MechanismTraceValidationError(ValueError):
    """Raised when mechanism trace inputs violate the JSON schema or constraints."""


def load_mechanism_trace_schema() -> dict[str, Any]:
    """Load the JSON schema for mechanism traces from disk.

    Returns:
        Parsed schema dictionary.
    """
    path = Path(__file__).parent / "schemas" / "mechanism_trace.schema.v1.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_mechanism_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a mechanism trace payload against the JSON Schema.

    Args:
        payload: The payload to validate

    Returns:
        The validated payload dictionary

    Raises:
        MechanismTraceValidationError: If the payload is invalid
    """
    schema = load_mechanism_trace_schema()
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        )
        validator.validate(payload)
    except jsonschema.ValidationError as e:
        raise MechanismTraceValidationError(f"Schema validation failed: {e.message}") from e

    # Additional semantic checks
    rows = payload.get("rows", [])
    for index, row in enumerate(rows):
        classification = row.get("classification")
        expected_states = {"inactive", "active-but-irrelevant", "slice-local", "revise", "stop"}
        if classification not in expected_states:
            raise MechanismTraceValidationError(
                f"Row {index} classification must be one of {expected_states}, got {classification!r}"
            )

    return payload


def classify_mechanism_trace_row(row: dict[str, Any]) -> str:  # noqa: C901
    """Classify a mechanism trace row into one of the 5 states.

    Inactive: Mechanism not active or condition not met.
    Active-but-irrelevant: Mechanism active but did not change command or outcome.
    Slice-local: Mechanism active, locally influenced command, but not primary recovery.
    Revise: Mechanism changed command source or revised command direction to avoid failure.
    Stop: Mechanism triggered a stop or fail-closed state.

    Returns:
        The classified state string.
    """
    val = row.get("classification")
    if val in {"inactive", "active-but-irrelevant", "slice-local", "revise", "stop"}:
        return val

    mechanism_id = row.get("mechanism_id")
    selected_cmd = row.get("selected_command")
    cmd_src = row.get("command_source")

    # Check stop
    is_stop = False
    if selected_cmd is None:
        is_stop = True
    elif isinstance(selected_cmd, list) and len(selected_cmd) >= 2:
        if abs(selected_cmd[0]) < 1e-5 and abs(selected_cmd[1]) < 1e-5:
            is_stop = True

    if is_stop:
        return "stop"

    # Check inactive
    input_cond = row.get("input_condition")
    if input_cond is False or input_cond is None:
        return "inactive"

    # Check active-but-irrelevant
    if cmd_src and mechanism_id:
        if mechanism_id == "static_recentering" and cmd_src != "static_recenter":
            return "active-but-irrelevant"
        if mechanism_id == "topology_guidance" and cmd_src not in {
            "topology_hypothesis",
            "topology_route",
        }:
            return "active-but-irrelevant"

    # Check revise vs slice-local
    progress_delta = row.get("route_progress_delta")
    if progress_delta is not None and abs(progress_delta) > 0.5:
        return "revise"

    return "slice-local"


def generate_mechanism_trace_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and classify mechanism trace rows, producing a summary report.

    Args:
        payload: The raw trace payload

    Returns:
        Report summary containing counts and classified rows
    """
    validated = validate_mechanism_trace_payload(payload)
    rows = validated.get("rows", [])

    counts = {
        "inactive": 0,
        "active-but-irrelevant": 0,
        "slice-local": 0,
        "revise": 0,
        "stop": 0,
    }

    classified_rows = []
    for row in rows:
        state = classify_mechanism_trace_row(row)
        counts[state] += 1

        row_copy = dict(row)
        row_copy["classification"] = state
        classified_rows.append(row_copy)

    return {
        "schema_version": "mechanism_trace_report.v1",
        "summary": {
            "total_rows": len(rows),
            "counts": counts,
        },
        "rows": classified_rows,
    }


def emit_static_recentering_row(
    step: int,
    last_decision: dict[str, Any],
    progress_delta: float | None = None,
    trace_uri: str | None = None,
) -> dict[str, Any]:
    """Convert static recentering planner diagnostics to a mechanism trace row.

    Args:
        step: Simulation step index
        last_decision: Planner step decision diagnostics
        progress_delta: Progress change compared to baseline
        trace_uri: Path to simulation trace file

    Returns:
        Formatted mechanism trace row dict
    """
    selected_terms = last_decision.get("selected_terms", {})
    recenter_term = selected_terms.get("static_recenter", 0.0)

    input_condition = {"static_recenter_term": recenter_term} if recenter_term > 0.0 else None
    selected_cmd = last_decision.get("selected_command")
    cmd_src = last_decision.get("selected_source", "unknown")

    if cmd_src == "static_recenter":
        classification = "revise"
    elif recenter_term > 0.0:
        classification = "active-but-irrelevant"
    else:
        classification = "inactive"

    return {
        "mechanism_id": "static_recentering",
        "activation_step": step,
        "input_condition": input_condition,
        "selected_command": selected_cmd,
        "command_source": cmd_src,
        "risk_score": float(last_decision.get("risk_score", 0.0))
        if last_decision.get("risk_score") is not None
        else None,
        "route_progress_delta": progress_delta,
        "failure_mode": last_decision.get("reason"),
        "trace_uri": trace_uri,
        "classification": classification,
    }


def emit_topology_guidance_row(
    step: int,
    last_decision: dict[str, Any],
    progress_delta: float | None = None,
    trace_uri: str | None = None,
) -> dict[str, Any]:
    """Convert topology guidance planner diagnostics to a mechanism trace row.

    Args:
        step: Simulation step index
        last_decision: Planner step decision diagnostics
        progress_delta: Progress change compared to baseline
        trace_uri: Path to simulation trace file

    Returns:
        Formatted mechanism trace row dict
    """
    topology_guided = last_decision.get("topology_guided", {})

    selected_cmd = last_decision.get("selected_command")
    cmd_src = last_decision.get("selected_source", "unknown")

    input_condition = None
    if topology_guided:
        input_condition = {
            "active": topology_guided.get("active"),
            "selected_hypothesis_id": topology_guided.get("selected_hypothesis_id"),
        }

    if cmd_src == "topology_fail_closed":
        classification = "stop"
    elif cmd_src in {"topology_hypothesis", "topology_route"}:
        classification = "revise"
    elif topology_guided.get("active"):
        classification = "active-but-irrelevant"
    else:
        classification = "inactive"

    return {
        "mechanism_id": "topology_guidance",
        "activation_step": step,
        "input_condition": input_condition,
        "selected_command": selected_cmd,
        "command_source": cmd_src,
        "risk_score": float(last_decision.get("risk_score", 0.0))
        if last_decision.get("risk_score") is not None
        else None,
        "route_progress_delta": progress_delta,
        "failure_mode": last_decision.get("reason"),
        "trace_uri": trace_uri,
        "classification": classification,
    }
