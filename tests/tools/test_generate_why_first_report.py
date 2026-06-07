"""Tests for why-first benchmark report generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools import generate_why_first_report

if TYPE_CHECKING:
    from pathlib import Path


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
