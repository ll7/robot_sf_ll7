"""Tests for the #1489 hybrid-learning synthesis recommendation report.

These tests exercise :func:`build_hybrid_synthesis_report`, the synthesis-
deliverable half of the #1489 contract. They confirm the fail-closed promotion
rule (a ``continue``/``revise`` verdict is authoritative only when the
prerequisite gate is open), the terminal ``stop`` surfacing rule, and that no
pre-result, single-complete, or invalid-row matrix can promote a verdict. Rows
are built from the shared evidence-matrix fixtures and mutated synthetically so
each case is focused and deterministic.
"""

from __future__ import annotations

import copy
from pathlib import Path

from robot_sf.benchmark.hybrid_evidence_matrix import (
    SYNTHESIS_RECOMMENDATIONS,
    build_hybrid_prerequisite_matrix,
    build_hybrid_synthesis_report,
    build_hybrid_synthesis_report_file,
    load_hybrid_evidence_input,
)

FIXTURE_ROOT = Path("tests/fixtures/hybrid_evidence_matrix/v1")


def _complete_row() -> dict:
    """Return a deep copy of the synthesis-eligible stress-slice fixture row."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")
    return copy.deepcopy(rows[0])


def _blocked_launch_packet_row() -> dict:
    """Return a deep copy of the not-run launch-packet fixture row."""
    _input_format, rows = load_hybrid_evidence_input(FIXTURE_ROOT / "valid_rows.yaml")
    return copy.deepcopy(rows[1])


def _second_complete_row() -> dict:
    """Return a distinct second synthesis-eligible lane."""
    row = _complete_row()
    row["component"] = "orca_residual_learned_v1"
    row["source_issue"] = "#1358"
    return row


def _stop_full_slice_row() -> dict:
    """Return an executed stress-slice lane that concluded ``stop``."""
    row = _complete_row()
    row["component"] = "shielded_ppo_repair_v1"
    row["source_issue"] = "#1474"
    row["verdict"] = "stop"
    return row


def _synthesis_from_rows(rows: list[dict], **kwargs) -> dict:
    """Build the prerequisite matrix then the synthesis report."""
    matrix = build_hybrid_prerequisite_matrix(rows, **kwargs)
    return build_hybrid_synthesis_report(matrix)


def test_no_complete_lanes_stays_blocked_and_unpromoted() -> None:
    """A launch-packet-only matrix must fail closed with no promoted verdicts."""
    report = _synthesis_from_rows([_blocked_launch_packet_row()])

    assert report["status"] == "blocked"
    assert report["eligible"] is False
    assert report["promoted_verdict_count"] == 0
    assert report["blockers"]
    assert report["mechanisms"][0]["recommendation"] == "gather_more_evidence"
    assert report["mechanisms"][0]["synthesis_verdict_promoted"] is False


def test_single_complete_lane_does_not_open_the_gate() -> None:
    """One complete lane recommends its verdict but is not yet promoted."""
    report = _synthesis_from_rows([_complete_row()])

    assert report["status"] == "blocked"
    assert report["eligible"] is False
    assert report["promoted_verdict_count"] == 0
    lane = report["mechanisms"][0]
    assert lane["recommendation"] == "continue"
    assert lane["recommendation_basis"] == "durable_complete_lane"
    assert lane["synthesis_verdict_promoted"] is False


def test_two_complete_lanes_open_the_gate_and_promote() -> None:
    """At two durable complete lanes the gate opens and both verdicts promote."""
    report = _synthesis_from_rows([_complete_row(), _second_complete_row()])

    assert report["status"] == "ready_for_synthesis"
    assert report["eligible"] is True
    assert report["promoted_verdict_count"] == 2
    assert report["blockers"] == []
    for lane in report["mechanisms"]:
        assert lane["recommendation"] in {"continue", "revise"}
        assert lane["synthesis_verdict_promoted"] is True


def test_terminal_stop_is_surfaced_but_never_promoted() -> None:
    """A stress-slice lane concluding ``stop`` surfaces as stop, unpromoted."""
    report = _synthesis_from_rows([_stop_full_slice_row(), _complete_row()])

    stop_lane = next(m for m in report["mechanisms"] if m["component"] == "shielded_ppo_repair_v1")
    assert stop_lane["recommendation"] == "stop"
    assert stop_lane["synthesis_verdict_promoted"] is False
    # One complete + one stop lane => only one complete lane => gate stays closed.
    assert report["eligible"] is False
    assert report["promoted_verdict_count"] == 0


def test_invalid_row_keeps_gate_closed_even_with_two_complete() -> None:
    """An invalid row makes the matrix not rows_valid, so nothing promotes."""
    invalid = _complete_row()
    invalid.pop("outcomes")  # drop a required field -> invalid row
    report = _synthesis_from_rows([_complete_row(), _second_complete_row(), invalid])

    assert report["eligible"] is False
    assert report["status"] == "blocked"
    assert report["promoted_verdict_count"] == 0
    assert any("invalid" in blocker for blocker in report["blockers"])


def test_missing_expected_component_maps_to_gather_more_evidence() -> None:
    """An expected-but-absent lane is reported as gather_more_evidence."""
    report = _synthesis_from_rows(
        [_complete_row()], expected_components=["learned_risk_model_v1", "absent_component"]
    )

    missing = next(m for m in report["mechanisms"] if m["component"] == "absent_component")
    assert missing["state"] == "missing"
    assert missing["recommendation"] == "gather_more_evidence"
    assert missing["recommendation_basis"] == "no_campaign_row"
    assert missing["synthesis_verdict_promoted"] is False


def test_all_recommendations_use_the_shared_vocabulary() -> None:
    """Every emitted recommendation must be in the declared vocabulary."""
    report = _synthesis_from_rows(
        [_complete_row(), _stop_full_slice_row(), _blocked_launch_packet_row()]
    )

    for lane in report["mechanisms"]:
        assert lane["recommendation"] in SYNTHESIS_RECOMMENDATIONS


def test_file_helper_attaches_input_metadata() -> None:
    """The file helper classifies fixture rows and records the input path."""
    report = build_hybrid_synthesis_report_file(FIXTURE_ROOT / "valid_rows.yaml")

    assert report["issue"] == "#1489"
    # The fixture holds one complete lane and one launch packet => blocked.
    assert report["status"] == "blocked"
    assert report["eligible"] is False
    assert isinstance(report["input_format"], str) and report["input_format"]
    assert report["input_path"].endswith("valid_rows.yaml")
