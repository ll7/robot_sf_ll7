"""Tests for the fixture-only prediction/planning/runtime safety contract (#7317).

These tests protect split identity, empirical coverage, hard-floor monotonicity, runtime
event accounting, and unavailable-field semantics.  They do not assert a navigation or
deployment-safety result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.benchmark.safety.prediction_planning_safety import (
    NominalPlanningTrace,
    PredictionHorizonTrace,
    RuntimeSafetyTrace,
    build_fixture_diagnostic_report,
    build_fixture_traces,
    build_prediction_planning_safety_diagnostic,
    validate_prediction_planning_safety_report,
)


def _partition_ids(traces):
    """Return the canonical fixture identity partitions."""
    return {
        "fit_trace_ids": {trace.trace_id for trace in traces if trace.split == "fit"},
        "calibration_trace_ids": {
            trace.trace_id for trace in traces if trace.split == "calibration"
        },
        "evaluation_trace_ids": {trace.trace_id for trace in traces if trace.split == "evaluation"},
    }


def _build(traces=None, **overrides):
    """Build a diagnostic from deterministic fixture rows."""
    selected = tuple(traces or build_fixture_traces())
    kwargs = {
        "traces": selected,
        "hard_floor_m": 0.3,
        "coverage_target": 0.8,
        "seed": 7317,
        **_partition_ids(selected),
    }
    kwargs.update(overrides)
    return build_prediction_planning_safety_diagnostic(**kwargs)


def test_fixture_report_separates_prediction_planning_and_runtime_mechanisms() -> None:
    """The fixture contract must expose all three mechanism cases and runtime events."""
    report = build_fixture_diagnostic_report()
    payload = report.to_dict()

    assert payload["evidence_tier"] == "smoke/diagnostic"
    assert payload["fixture_case_counts"] == {
        "good_prediction_poor_planning": 2,
        "poor_prediction_safe_fallback": 2,
        "verification_unavailable": 2,
    }
    assert payload["prediction_coverage"][0]["status"] == "under_covered"
    assert payload["same_seed_comparison"]["paired_trace_count"] == 3
    lane_by_id = {row["lane_id"]: row for row in payload["lanes"]}
    assert lane_by_id["baseline"]["event_counts"]["verified"] == 2
    assert lane_by_id["uncertainty_aware"]["event_counts"]["contingency_invoked"] == 1
    assert "collision" in lane_by_id["uncertainty_aware"]["unavailable_fields"]
    assert "per-encounter" in payload["claim_boundary"]


def test_fixture_report_is_deterministic_and_schema_valid() -> None:
    """Repeated fixture builds should be byte-equivalent after canonical JSON encoding."""
    first = build_fixture_diagnostic_report().to_dict()
    second = build_fixture_diagnostic_report().to_dict()

    assert first == second
    validate_prediction_planning_safety_report(first)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_split_identity_overlap_is_rejected() -> None:
    """A trace identity cannot be used in more than one fit/calibration/evaluation split."""
    traces = build_fixture_traces()
    partitions = _partition_ids(traces)
    overlap_id = next(iter(partitions["fit_trace_ids"]))
    partitions["calibration_trace_ids"].add(overlap_id)

    with pytest.raises(ValueError, match="trace identities overlap"):
        build_prediction_planning_safety_diagnostic(
            traces=traces,
            hard_floor_m=0.3,
            coverage_target=0.8,
            seed=7317,
            **partitions,
        )


def test_split_identity_must_match_trace_split_field() -> None:
    """Declared split membership must agree with the trace's own split label."""
    traces = list(build_fixture_traces())
    partitions = _partition_ids(traces)
    first = traces[0]
    traces[0] = replace(first, split="evaluation")

    with pytest.raises(ValueError, match="must carry split='fit'"):
        build_prediction_planning_safety_diagnostic(
            traces=traces,
            hard_floor_m=0.3,
            coverage_target=0.8,
            seed=7317,
            **partitions,
        )


def test_hard_floor_reduction_is_rejected() -> None:
    """An uncertainty-aware effective margin may not weaken the deterministic hard floor."""
    traces = list(build_fixture_traces())
    target = next(trace for trace in traces if trace.split == "evaluation")
    weakened = replace(
        target,
        planning=NominalPlanningTrace(
            status="available",
            planner_source="fixture",
            deterministic_margin_m=0.2,
            uncertainty_margin_m=0.0,
            effective_margin_m=0.2,
            command=(0.1, 0.0),
        ),
    )
    traces[traces.index(target)] = weakened

    with pytest.raises(ValueError, match="below hard floor"):
        _build(traces)


def test_prediction_and_runtime_unavailable_states_fail_closed() -> None:
    """Unavailable inputs must not carry fabricated residuals or omit required reasons."""
    with pytest.raises(ValueError, match="unavailable predictions"):
        PredictionHorizonTrace(
            horizon_step=1,
            representation="interval",
            status="unavailable",
            realized_error_m=0.1,
        )
    with pytest.raises(ValueError, match="requires reason"):
        RuntimeSafetyTrace(status="verification_unavailable")
    with pytest.raises(ValueError, match="requires contingency_action"):
        RuntimeSafetyTrace(status="contingency_invoked")


def test_schema_validation_rejects_wrong_version() -> None:
    """Serialized diagnostics remain bound to their versioned report schema."""
    payload = build_fixture_diagnostic_report().to_dict()
    payload["schema_version"] = "prediction_planning_safety.v0"

    with pytest.raises(ValueError, match="schema validation failed"):
        validate_prediction_planning_safety_report(payload)


def test_cli_emits_a_valid_fixture_report(tmp_path: Path) -> None:
    """The canonical validation entry point must produce a schema-valid JSON artifact."""
    output = tmp_path / "prediction_planning_safety.json"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validation"
        / "run_prediction_planning_safety_diagnostic.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "valid"
    validate_prediction_planning_safety_report(json.loads(output.read_text(encoding="utf-8")))
