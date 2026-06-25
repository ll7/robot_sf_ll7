"""Tests for why-first benchmark report generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import generate_why_first_report

if TYPE_CHECKING:
    from pathlib import Path

_NON_SUCCESS_STATUSES = ("fallback", "degraded", "failed", "not_available", "skipped")
_LIMITATION_TEXT = (
    "fallback/degraded/failed/not-available evidence and must not be counted as benchmark success"
)


def _compact_evidence() -> dict[str, object]:
    """Return compact fallback evidence for a why-first report."""
    return {
        "title": "Fixture why report",
        "planner": "orca",
        "scenario_id": "classic_head_on_corridor_low",
        "outcome": "collision avoided but adapter fallback was active",
        "execution_status": "fallback",
        "metrics": {"success": 1.0, "collision_rate": 0.0},
        "mechanism_activation": {
            "name": "route_offset_sensitivity",
            "status": "activated",
            "evidence": "intervention changed clearance margin",
        },
        "failure_mechanism": {
            "label": "fallback_adapter_confound",
            "rationale": "adapter fallback prevents native planner interpretation",
        },
        "paired_comparator": {
            "name": "baseline_noop",
            "outcome": "same success under unchanged seed",
            "delta": {"success": 0.0},
        },
        "trace_evidence": [
            "trace_review: docs/context/evidence/fixture/trace_summary.json",
        ],
        "alternative_explanations": [
            "route geometry rather than planner mechanism may explain the outcome",
        ],
        "decision": {
            "action": "revise",
            "rationale": "fallback row cannot support benchmark-strength interpretation",
            "next_step": "rerun with native planner support",
        },
        "dissertation": {
            "reader_takeaway": "Fallback-active ORCA evidence is useful only as a limitation case.",
            "allowed_wording": [
                "The diagnostic report identifies fallback as a confound.",
                "The row is not benchmark-strength planner evidence.",
            ],
            "not_claimed": [
                "ORCA outperforms other planners.",
                "Fallback execution is equivalent to native execution.",
            ],
            "figure_table_candidates": ["tab:robot_sf_release_planner_results"],
        },
    }


def test_generate_report_includes_required_sections_and_fallback_caveat() -> None:
    """The report should be why-first and fail-closed about fallback rows."""
    report = generate_why_first_report.generate_report(_compact_evidence())

    for section in generate_why_first_report.REQUIRED_SECTIONS:
        assert f"## {section}" in report
    assert "fallback/degraded/failed/not-available evidence" in report
    assert "must not be counted as benchmark success" in report
    assert "fallback_adapter_confound" in report
    assert "## Claim Boundary" in report
    assert "## Dissertation-Facing Handoff" not in report


def test_cli_writes_markdown_report(tmp_path: Path) -> None:
    """The CLI should convert compact JSON evidence into Markdown."""
    input_path = tmp_path / "evidence.json"
    output_path = tmp_path / "why_report.md"
    input_path.write_text(json.dumps(_compact_evidence()), encoding="utf-8")

    result = generate_why_first_report.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    report = output_path.read_text(encoding="utf-8")
    assert report.startswith("# Fixture why report\n")
    assert "## Continue / Revise / Stop Decision" in report
    assert "rerun with native planner support" in report


def test_dissertation_mode_emits_reader_takeaway_and_claim_boundaries() -> None:
    """Dissertation mode should expose bounded wording without promoting claims."""
    report = generate_why_first_report.generate_report(_compact_evidence(), mode="dissertation")

    assert "## Dissertation-Facing Handoff" in report
    assert "`reader_takeaway`: Fallback-active ORCA evidence" in report
    assert "`allowed_wording`: The diagnostic report identifies fallback as a confound" in report
    assert "`not_claimed`: ORCA outperforms other planners" in report
    assert "`figure_table_candidates`: tab:robot_sf_release_planner_results" in report
    assert "must not be counted as benchmark success" in report
    assert "## Claim Boundary" in report


def test_cli_dissertation_mode_writes_handoff_fields(tmp_path: Path) -> None:
    """The CLI should expose dissertation-facing fields when explicitly requested."""
    input_path = tmp_path / "evidence.json"
    output_path = tmp_path / "why_report.md"
    input_path.write_text(json.dumps(_compact_evidence()), encoding="utf-8")

    result = generate_why_first_report.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            "dissertation",
        ]
    )

    assert result == 0
    report = output_path.read_text(encoding="utf-8")
    assert "## Dissertation-Facing Handoff" in report
    assert "`not_claimed`" in report
    assert "`allowed_wording`" in report


def _minimal_evidence() -> dict[str, object]:
    """Return near-empty evidence to exercise fallback rendering paths."""
    return {"title": "Minimal report"}


# --- Edge-case contract coverage (issue #3464) -------------------------------
# The why-first report feeds research/dissertation decisions, so malformed or
# incomplete compact evidence must render explicit "not specified" rows and
# fail-closed status caveats rather than silently dropping report sections.


def test_minimal_evidence_keeps_all_required_sections() -> None:
    """Near-empty evidence must still render every required section and boundary."""
    report = generate_why_first_report.generate_report(_minimal_evidence())

    for section in generate_why_first_report.REQUIRED_SECTIONS:
        assert f"## {section}" in report, f"missing required section: {section}"
    assert "## Claim Boundary" in report
    assert generate_why_first_report.CLAIM_BOUNDARY in report
    # Sections must be explicit about missing inputs, never silently empty.
    assert "Mechanism activation: not specified in compact evidence." in report
    assert "No paired comparator was provided." in report
    assert "No trace evidence reference was provided." in report
    assert "No alternative explanations were provided." in report


def test_malformed_mechanism_activation_renders_explicit_fallback() -> None:
    """Non-mapping or empty mechanism activation must render an explicit fallback row."""
    for malformed in ("activated", ["activated"], {}):
        evidence = {**_minimal_evidence(), "mechanism_activation": malformed}
        report = generate_why_first_report.generate_report(evidence)
        assert "Mechanism activation: not specified in compact evidence." in report
        assert "## Mechanism Activation" in report


def test_partial_mechanism_activation_uses_named_unknown_defaults() -> None:
    """A mapping missing keys must surface explicit unknown markers, not crash."""
    evidence = {
        **_minimal_evidence(),
        "mechanism_activation": {"name": "route_offset_sensitivity"},
    }
    report = generate_why_first_report.generate_report(evidence)

    assert "Mechanism: `route_offset_sensitivity`." in report
    assert "Activation status: `unknown`." in report
    assert "Evidence: not specified." in report


def test_missing_trace_evidence_is_explicit_not_dropped() -> None:
    """Missing trace-review references must be reported explicitly, never dropped."""
    report = generate_why_first_report.generate_report(_minimal_evidence())

    assert "## Trace Evidence" in report
    assert "No trace evidence reference was provided." in report


def test_trace_evidence_mapping_form_is_rendered() -> None:
    """A mapping trace_evidence payload must render keyed references."""
    evidence = {
        **_minimal_evidence(),
        "trace_evidence": {"trace_review": "docs/context/evidence/fixture/trace.json"},
    }
    report = generate_why_first_report.generate_report(evidence)

    assert "trace_review: docs/context/evidence/fixture/trace.json." in report


@pytest.mark.parametrize("status", _NON_SUCCESS_STATUSES)
def test_non_success_statuses_emit_limitation_caveat(status: str) -> None:
    """Every non-success execution status must emit the fail-closed limitation row."""
    evidence = {**_minimal_evidence(), "execution_status": status}
    report = generate_why_first_report.generate_report(evidence)

    assert "- Limitation: this row is " + _LIMITATION_TEXT + "." in report
    assert f"Execution status: `{status}`." in report


@pytest.mark.parametrize("status", ("success", "completed", "native"))
def test_success_statuses_do_not_emit_limitation_caveat(status: str) -> None:
    """Success-style statuses must not be flagged as fallback/degraded evidence."""
    evidence = {**_minimal_evidence(), "execution_status": status}
    report = generate_why_first_report.generate_report(evidence)

    assert "- Limitation: this row is " not in report
    assert f"Execution status: `{status}`." in report


def test_uppercase_status_is_normalized_before_caveat_check() -> None:
    """Status comparison must be case-insensitive so caveats are not silently skipped."""
    evidence = {**_minimal_evidence(), "execution_status": "FALLBACK"}
    report = generate_why_first_report.generate_report(evidence)

    assert "- Limitation: this row is " + _LIMITATION_TEXT + "." in report


def test_readiness_status_used_when_execution_status_absent() -> None:
    """The caveat must still fire when only readiness_status carries the flag."""
    evidence = {**_minimal_evidence(), "readiness_status": "degraded"}
    report = generate_why_first_report.generate_report(evidence)

    assert "- Limitation: this row is " + _LIMITATION_TEXT + "." in report
    assert "Execution status: `degraded`." in report


def test_actuation_failure_label_from_string_classification() -> None:
    """A bare-string failure mechanism (e.g. actuation failure) must classify cleanly."""
    evidence = {**_minimal_evidence(), "failure_mechanism": "actuation_command_failure"}
    report = generate_why_first_report.generate_report(evidence)

    assert "Classification: `actuation_command_failure`." in report
    assert "Rationale: not specified." in report


def test_dissertation_handoff_declares_no_fallback_when_status_clean() -> None:
    """Dissertation handoff must state plainly when no fallback flag was declared."""
    evidence = {**_minimal_evidence(), "execution_status": "success"}
    report = generate_why_first_report.generate_report(evidence, mode="dissertation")

    assert "no fallback/degraded status was declared in the compact evidence" in report
    assert "## Dissertation-Facing Handoff" in report


def test_load_evidence_rejects_non_object_payload(tmp_path: Path) -> None:
    """Fail-closed: a non-object JSON payload must raise WhyFirstReportError."""
    input_path = tmp_path / "evidence.json"
    input_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(generate_why_first_report.WhyFirstReportError):
        generate_why_first_report.load_evidence(input_path)


def test_generate_report_rejects_unsupported_mode() -> None:
    """Fail-closed: an unknown render mode must raise rather than emit a partial report."""
    with pytest.raises(generate_why_first_report.WhyFirstReportError):
        generate_why_first_report.generate_report(_minimal_evidence(), mode="unsupported")


def test_non_success_status_fixture_tracks_generator_constant() -> None:
    """Guard against drift: the covered statuses must match the generator's source set."""
    assert set(_NON_SUCCESS_STATUSES) == generate_why_first_report._NON_SUCCESS_STATUSES
