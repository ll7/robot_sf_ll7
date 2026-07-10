"""Mechanism trace validation, classification, and report generation utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema

from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "mechanism_trace.v1"


class MechanismTraceValidationError(RobotSfError, ValueError):
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

    generated_at_utc = payload.get("generated_at_utc")
    if isinstance(generated_at_utc, str):
        try:
            datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))
        except ValueError as e:
            raise MechanismTraceValidationError(
                "generated_at_utc must be a valid ISO 8601 datetime"
            ) from e
    else:
        raise MechanismTraceValidationError(
            "generated_at_utc must be a string in ISO 8601 datetime format"
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
        if mechanism_id == "orca_residuals" and cmd_src not in {
            "prior_residual",
            "prior_residual_safe",
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


def emit_orca_residual_row(
    step: int,
    last_decision: Mapping[str, Any],
    progress_delta: float | None = None,
    trace_uri: str | None = None,
) -> dict[str, Any]:
    """Convert an ORCA residual planning decision into a mechanism trace row.

    Args:
        step: Simulation step index
        last_decision: Planner step decision diagnostics
        progress_delta: Progress change compared to previous step
        trace_uri: Path to simulation trace file

    Returns:
        Formatted mechanism trace row dict
    """
    action_adaptation = last_decision.get("action_adaptation")
    if not isinstance(action_adaptation, Mapping):
        fallback_state = last_decision.get("fallback_controller_state")
        if isinstance(fallback_state, Mapping):
            action_adaptation = fallback_state.get("action_adaptation")
    adaptation = action_adaptation if isinstance(action_adaptation, Mapping) else {}
    adaptation_mode_value = adaptation.get("mode")
    adaptation_mode = (
        str(adaptation_mode_value).strip() if adaptation_mode_value is not None else ""
    )
    is_prior_residual = adaptation_mode == "prior_residual"

    selected_cmd = last_decision.get("selected_command")
    if isinstance(selected_cmd, list | tuple):
        selected_command_values = [
            float(value) for value in selected_cmd[:2] if isinstance(value, int | float)
        ]
        selected_command = selected_command_values if len(selected_command_values) >= 2 else None
    else:
        selected_command = None

    raw_residual = adaptation.get("raw_residual_action", [])
    bounded_residual = adaptation.get("bounded_residual_action", [])
    raw_residual_floats = (
        [float(value) for value in raw_residual[:2] if isinstance(value, int | float)]
        if isinstance(raw_residual, list | tuple)
        else []
    )
    bounded_residual_floats = (
        [float(value) for value in bounded_residual[:2] if isinstance(value, int | float)]
        if isinstance(bounded_residual, list | tuple)
        else []
    )

    residual_norm = None
    if len(raw_residual_floats) >= 2:
        residual_norm = float((raw_residual_floats[0] ** 2 + raw_residual_floats[1] ** 2) ** 0.5)

    original_source_value = last_decision.get("selected_source")
    original_command_source = (
        str(original_source_value) if original_source_value is not None else "unknown"
    )
    original_command_source = (
        "unknown" if original_command_source == "None" else original_command_source
    )

    input_condition = (
        {
            "adaptation_mode": adaptation_mode,
            "original_command_source": original_command_source,
            "raw_residual": raw_residual_floats,
            "bounded_residual": bounded_residual_floats,
            "residual_norm": residual_norm,
            "residual_clipped": bool(adaptation.get("residual_clipped", False)),
            "residual_bounds": adaptation.get("residual_bounds"),
        }
        if is_prior_residual
        else None
    )

    command_source = "prior_residual_safe" if is_prior_residual else original_command_source

    selected_score = last_decision.get("selected_score")
    risk_score = (
        float(selected_score)
        if is_prior_residual and isinstance(selected_score, int | float)
        else None
    )

    row = {
        "mechanism_id": "orca_residuals",
        "activation_step": step,
        "input_condition": input_condition,
        "selected_command": selected_command,
        "command_source": command_source,
        "risk_score": risk_score,
        "route_progress_delta": progress_delta,
        "failure_mode": None,
        "trace_uri": trace_uri,
        "classification": "inactive",
    }
    if is_prior_residual and bool(last_decision.get("intervened", False)):
        reason = last_decision.get("intervention_reason")
        row["failure_mode"] = str(reason) if reason is not None else "residual_intervention_active"
    class_input = dict(row)
    class_input.pop("classification", None)
    row["classification"] = classify_mechanism_trace_row(class_input)
    return row


def emit_orca_residual_rows(
    planner_decision_trace: list[dict[str, Any]],
    *,
    trace_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Build ORCA residual mechanism rows from a planner decision trace.

    Args:
        planner_decision_trace: Planner decision trace records.
        trace_uri: Optional trace URI attached to emitted rows.

    Returns:
        List of mechanism trace rows.
    """
    rows: list[dict[str, Any]] = []
    previous_progress = None
    for entry in planner_decision_trace:
        if not isinstance(entry, Mapping):
            continue
        route_progress = entry.get("route_progress_from_start_m")
        if isinstance(route_progress, int | float):
            route_progress_float = float(route_progress)
            progress_delta = (
                route_progress_float - previous_progress
                if isinstance(previous_progress, float)
                else None
            )
            previous_progress = route_progress_float
        else:
            progress_delta = None
            previous_progress = None

        step = entry.get("step", len(rows))
        activation_step = int(step) if isinstance(step, int | float) else len(rows)
        rows.append(
            emit_orca_residual_row(
                activation_step,
                last_decision=entry,
                progress_delta=progress_delta,
                trace_uri=trace_uri,
            )
        )
    return rows


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
