"""Tests for the #1489 hybrid-learning prerequisite/status matrix helper.

These tests exercise the campaign-lifecycle classification (missing, blocked,
ready, complete) and the conservative synthesis gate that consumes it. Rows are
built from the shared evidence-matrix fixtures and mutated synthetically so each
lifecycle state has a focused, deterministic case.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from robot_sf.benchmark.hybrid_evidence_matrix import (
    DEFAULT_SYNTHESIS_PREREQUISITE_COUNT,
    HybridEvidenceMatrixValidationError,
    build_hybrid_prerequisite_matrix,
    build_hybrid_prerequisite_matrix_file,
    classify_component_campaign_state,
    load_hybrid_evidence_input,
)
from scripts.validation.validate_hybrid_evidence_matrix import main as validate_cli_main

FIXTURE_ROOT = Path("tests/fixtures/hybrid_evidence_matrix/v1")


def _complete_row() -> dict:
    """Return a deep copy of the synthesis-eligible stress-slice fixture row."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")
    return copy.deepcopy(rows[0])


def _blocked_launch_packet_row() -> dict:
    """Return a deep copy of the not-run launch-packet fixture row."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")
    return copy.deepcopy(rows[1])


def _ready_smoke_row() -> dict:
    """Build an executed smoke-only row that is valid but not synthesis-grade."""
    row = _complete_row()
    row["component"] = "shielded_ppo_repair_v1"
    row["source_issue"] = "#1474"
    row["evaluation_slice"] = "smoke"
    row["evidence_tier"] = "smoke_only"
    return row


def test_classify_component_campaign_state_maps_each_state() -> None:
    """The row-level classifier maps eligibility/tier/slice to the lifecycle state."""
    assert classify_component_campaign_state({"synthesis_eligible": True, "status": "valid"}) == (
        "complete"
    )
    assert classify_component_campaign_state({"status": "invalid"}) == "blocked"
    assert (
        classify_component_campaign_state({"status": "valid", "evaluation_slice": "not_run"})
        == "blocked"
    )
    assert (
        classify_component_campaign_state({"status": "valid", "evidence_tier": "failed"})
        == "blocked"
    )
    assert (
        classify_component_campaign_state(
            {"status": "valid", "evaluation_slice": "smoke", "evidence_tier": "smoke_only"}
        )
        == "ready"
    )


def test_complete_row_classifies_as_complete() -> None:
    """A synthesis-eligible stress row is the only state that feeds synthesis."""
    matrix = build_hybrid_prerequisite_matrix([_complete_row()])

    assert matrix["lanes"][0]["state"] == "complete"
    assert matrix["lanes"][0]["synthesis_eligible"] is True
    assert matrix["state_counts"]["complete"] == 1


def test_blocked_launch_packet_row_classifies_as_blocked() -> None:
    """A not-run launch packet is pre-runtime evidence and must stay blocked."""
    matrix = build_hybrid_prerequisite_matrix([_blocked_launch_packet_row()])

    assert matrix["lanes"][0]["state"] == "blocked"
    assert matrix["state_counts"]["blocked"] == 1


def test_ready_smoke_row_classifies_as_ready() -> None:
    """An executed smoke-only row produced runtime evidence but is not synthesis-grade."""
    matrix = build_hybrid_prerequisite_matrix([_ready_smoke_row()])

    assert matrix["lanes"][0]["state"] == "ready"
    assert matrix["lanes"][0]["synthesis_eligible"] is False
    assert matrix["state_counts"]["ready"] == 1


def test_invalid_row_classifies_as_blocked() -> None:
    """An invalid row cannot be trusted, so it is blocked rather than ready/complete."""
    row = _complete_row()
    row["source_issue"] = "not-an-issue-ref"

    matrix = build_hybrid_prerequisite_matrix([row])

    assert matrix["rows_valid"] is False
    assert matrix["invalid_row_count"] == 1
    assert matrix["lanes"][0]["state"] == "blocked"


def test_expected_component_without_row_is_missing() -> None:
    """Expected components with no row should surface as a 'missing' lane."""
    matrix = build_hybrid_prerequisite_matrix(
        [_complete_row()],
        expected_components=["learned_risk_model_v1", "orca_residual_bc_v1"],
    )

    states = {lane["component"]: lane["state"] for lane in matrix["lanes"]}
    assert states["learned_risk_model_v1"] == "complete"
    assert states["orca_residual_bc_v1"] == "missing"
    assert matrix["state_counts"]["missing"] == 1


def test_gate_blocked_with_single_complete_lane() -> None:
    """One synthesis-eligible lane is below the default prerequisite of two."""
    matrix = build_hybrid_prerequisite_matrix(
        [_complete_row(), _ready_smoke_row(), _blocked_launch_packet_row()]
    )

    assert matrix["prerequisite_count"] == DEFAULT_SYNTHESIS_PREREQUISITE_COUNT
    assert matrix["complete_count"] == 1
    assert matrix["prerequisite_met"] is False
    assert matrix["gate"] == "blocked"


def test_gate_opens_only_when_two_lanes_are_complete() -> None:
    """The gate opens at exactly two durable comparable (complete) lanes."""
    first = _complete_row()
    second = _complete_row()
    second["component"] = "orca_residual_learned_v1"
    second["source_issue"] = "#1358"

    matrix = build_hybrid_prerequisite_matrix([first, second])

    assert matrix["complete_count"] == 2
    assert matrix["prerequisite_met"] is True
    assert matrix["gate"] == "ready_for_synthesis"


def test_ready_lanes_never_open_the_gate() -> None:
    """Even many executed-but-not-synthesis-grade lanes cannot open the gate."""
    rows = [_ready_smoke_row() for _ in range(3)]

    matrix = build_hybrid_prerequisite_matrix(rows)

    assert matrix["state_counts"]["ready"] == 3
    assert matrix["complete_count"] == 0
    assert matrix["gate"] == "blocked"


def test_custom_prerequisite_count_changes_gate() -> None:
    """A prerequisite of one opens the gate on a single complete lane."""
    matrix = build_hybrid_prerequisite_matrix([_complete_row()], prerequisite_count=1)

    assert matrix["prerequisite_count"] == 1
    assert matrix["gate"] == "ready_for_synthesis"


def test_invalid_prerequisite_count_is_rejected() -> None:
    """A non-positive prerequisite count is a usage error, not a silent default."""
    try:
        build_hybrid_prerequisite_matrix([_complete_row()], prerequisite_count=0)
    except HybridEvidenceMatrixValidationError as exc:
        assert "prerequisite_count" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected HybridEvidenceMatrixValidationError")


def test_blank_expected_component_is_rejected() -> None:
    """Expected-component entries must be non-empty strings."""
    try:
        build_hybrid_prerequisite_matrix([_complete_row()], expected_components=["  "])
    except HybridEvidenceMatrixValidationError as exc:
        assert "expected_components" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected HybridEvidenceMatrixValidationError")


def test_single_string_expected_components_is_rejected() -> None:
    """A bare string must not be iterated character-by-character into lanes."""
    try:
        build_hybrid_prerequisite_matrix(
            [_complete_row()], expected_components="learned_risk_model_v1"
        )
    except HybridEvidenceMatrixValidationError as exc:
        assert "expected_components" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected HybridEvidenceMatrixValidationError")


def test_file_helper_attaches_input_metadata() -> None:
    """The file helper should classify fixture rows and record the input path."""
    report = build_hybrid_prerequisite_matrix_file(FIXTURE_ROOT / "valid_rows.yaml")

    assert report["input_format"] == "rows"
    assert report["input_path"].endswith("valid_rows.yaml")
    assert report["lane_count"] == 2
    states = sorted(lane["state"] for lane in report["lanes"])
    assert states == ["blocked", "complete"]


def test_cli_prerequisite_matrix_emits_gate_decision(capsys) -> None:
    """The CLI prerequisite mode should emit the gate JSON and a clean exit code."""
    exit_code = validate_cli_main(
        [
            "--input",
            str(FIXTURE_ROOT / "valid_rows.yaml"),
            "--prerequisite-matrix",
            "--expected-component",
            "learned_risk_model_v1",
            "--expected-component",
            "shielded_ppo_repair_v1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate"] == "blocked"
    assert payload["complete_count"] == 1
    assert payload["state_counts"]["missing"] == 1
