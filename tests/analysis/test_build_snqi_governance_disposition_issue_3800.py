"""Tests for issue #3800 SNQI governance disposition packet."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.analysis.build_snqi_governance_disposition_issue_3800 import (
    GovernanceIssueDisposition,
    build_packet,
    main,
    render_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _synthetic_governance_report() -> dict:
    return {
        "schema_version": "snqi_governance_preflight.v1",
        "claim_boundary": "secondary_diagnostic_only",
        "status": "failed",
        "blockers": [
            {
                "issue": 3723,
                "kind": "weight_provenance_conflict",
                "detail": "canonical-labeled weight sets disagree",
            },
            {
                "issue": 3699,
                "kind": "mixed_normalization_basis",
                "detail": "raw and baseline-normalized terms mix",
            },
        ],
        "weights": {},
        "normalization": {},
    }


def test_packet_separates_resolved_checks_owner_decisions_and_claim_notes() -> None:
    """Synthetic metadata proves the packet keeps disposition categories distinct."""
    dispositions = (
        GovernanceIssueDisposition(
            issue=1,
            title="Synthetic resolved provenance issue",
            current_status="decision-required",
            resolved_provenance_checks=("inventory exists", "preflight exists"),
            remaining_owner_decisions=("choose owner policy",),
            claim_boundary_notes=("diagnostic only",),
            source_refs=("docs/example.md",),
        ),
    )

    packet = build_packet(_synthetic_governance_report(), dispositions)

    assert packet["packet_status"] == "blocked_on_owner_decisions"
    assert packet["summary"] == {
        "issue_count": 1,
        "resolved_provenance_check_count": 2,
        "remaining_owner_decision_count": 1,
        "claim_boundary_note_count": 1,
    }
    disposition = packet["issue_dispositions"][0]
    assert disposition["resolved_provenance_checks"] == ["inventory exists", "preflight exists"]
    assert disposition["remaining_owner_decisions"] == ["choose owner policy"]
    assert disposition["claim_boundary_notes"] == ["diagnostic only"]


def test_packet_reports_diagnostic_clear_when_no_blockers() -> None:
    """A clear governance report yields no owner decisions and a clear status."""
    clear_report = _synthetic_governance_report()
    clear_report["status"] = "passed"
    clear_report["blockers"] = []

    packet = build_packet(clear_report)

    assert packet["packet_status"] == "diagnostic_clear"
    assert packet["remaining_owner_decisions"] == []
    assert packet["summary"]["remaining_owner_decision_count"] == 0
    for disposition in packet["issue_dispositions"]:
        assert disposition["current_status"] == "diagnostic-clear"
        assert disposition["remaining_owner_decisions"] == []


def test_render_markdown_keeps_sections_explicit() -> None:
    """Generated Markdown has reviewable sections for the disposition packet."""
    packet = build_packet(_synthetic_governance_report())
    markdown = render_markdown(packet)

    assert "# SNQI Governance Disposition Packet" in markdown
    assert "Resolved provenance checks:" in markdown
    assert "Remaining owner decisions:" in markdown
    assert "Claim-boundary notes:" in markdown
    assert "does not run a benchmark campaign" in markdown
    assert "does not change SNQI scoring" in markdown


def test_cli_writes_generated_markdown_packet(tmp_path: Path) -> None:
    """CLI writes the generated JSON and Markdown packet."""
    output_dir = tmp_path / "packet"

    assert main(["--output-dir", str(output_dir)]) == 0

    json_path = output_dir / "snqi_governance_disposition.json"
    markdown_path = output_dir / "snqi_governance_disposition.md"
    assert json_path.exists()
    assert markdown_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["schema_version"] == "snqi_governance_disposition_issue_3800.v1"
    assert payload["packet_status"] == "blocked_on_owner_decisions"
    assert "Issue #3723" in markdown
    assert "Issue #3699" in markdown
