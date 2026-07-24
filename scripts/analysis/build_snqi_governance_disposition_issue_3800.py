#!/usr/bin/env python3
"""Build the issue #3800 SNQI governance disposition packet.

The packet is read-only analysis. It summarizes the current Social Navigation
Quality Index (SNQI) governance state from existing diagnostics without choosing
canonical weights, changing normalization, or changing benchmark semantics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.validation.check_snqi_governance import CLAIM_BOUNDARY, build_governance_report

SCHEMA_VERSION = "snqi_governance_disposition_issue_3800.v1"
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_3800_snqi_governance_disposition")


@dataclass(frozen=True)
class GovernanceIssueDisposition:
    """Disposition facts for one open SNQI governance issue."""

    issue: int
    title: str
    current_status: str
    resolved_provenance_checks: tuple[str, ...]
    remaining_owner_decisions: tuple[str, ...]
    claim_boundary_notes: tuple[str, ...]
    source_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return {
            "issue": self.issue,
            "title": self.title,
            "current_status": self.current_status,
            "resolved_provenance_checks": list(self.resolved_provenance_checks),
            "remaining_owner_decisions": list(self.remaining_owner_decisions),
            "claim_boundary_notes": list(self.claim_boundary_notes),
            "source_refs": list(self.source_refs),
        }


def default_dispositions(
    governance_report: dict[str, Any],
) -> tuple[GovernanceIssueDisposition, ...]:
    """Build current issue dispositions from the SNQI governance report."""
    blocker_kinds_by_issue: dict[int, list[str]] = {}
    for blocker in governance_report.get("blockers", []):
        issue = int(blocker["issue"])
        blocker_kinds_by_issue.setdefault(issue, []).append(str(blocker["kind"]))

    return (
        GovernanceIssueDisposition(
            issue=3723,
            title="Conflicting canonical-labeled SNQI weight sets",
            current_status="decision-required"
            if 3723 in blocker_kinds_by_issue
            else "diagnostic-clear",
            resolved_provenance_checks=(
                "Weight-source inventory enumerates code default and shipped JSON weight sets.",
                "Fail-closed preflight exposes blocking canonical-label conflicts.",
                "User guide documents the current provenance conflict without selecting a winner.",
            ),
            remaining_owner_decisions=(
                "Choose the single canonical SNQI weight source, or explicitly retire the canonical label.",
                "Update shipped artifact metadata and fail-closed tests after the canonical decision.",
            )
            if 3723 in blocker_kinds_by_issue
            else (),
            claim_boundary_notes=(
                "Current checks are secondary diagnostic evidence only.",
                "The packet does not make SNQI a primary safety ranking or dissertation claim.",
            ),
            source_refs=(
                "docs/snqi-weight-tools/weights_provenance.md",
                "robot_sf/benchmark/snqi/weights_inventory.py",
                "scripts/validation/check_snqi_governance.py",
            ),
        ),
        GovernanceIssueDisposition(
            issue=3699,
            title="Mixed SNQI normalization basis",
            current_status="decision-required"
            if 3699 in blocker_kinds_by_issue
            else "diagnostic-clear",
            resolved_provenance_checks=(
                "Normalization inventory mirrors the current compute_snqi term-scaling regimes.",
                "Combined governance preflight reports mixed raw and baseline-normalized penalty terms.",
                "Optional baseline coverage check reports missing normalized median/p95 inputs.",
            ),
            remaining_owner_decisions=(
                "Choose whether to normalize raw terms or document the bounded raw-term asymmetry.",
                "Only after that decision, update scoring semantics, docs, and tests together.",
            )
            if 3699 in blocker_kinds_by_issue
            else (),
            claim_boundary_notes=(
                "The packet does not change emitted SNQI values.",
                "Mixed-scale diagnostics must stay caveats for benchmark or paper-facing SNQI claims.",
            ),
            source_refs=(
                "docs/snqi-weight-tools/weights_provenance.md",
                "robot_sf/benchmark/snqi/normalization_inventory.py",
                "scripts/validation/check_snqi_governance.py",
            ),
        ),
    )


def build_packet(
    governance_report: dict[str, Any],
    dispositions: tuple[GovernanceIssueDisposition, ...] | None = None,
) -> dict[str, Any]:
    """Build a structured read-only disposition packet."""
    issue_dispositions = dispositions or default_dispositions(governance_report)
    remaining_decisions = [
        {
            "issue": disposition.issue,
            "decision": decision,
        }
        for disposition in issue_dispositions
        for decision in disposition.remaining_owner_decisions
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_status": "blocked_on_owner_decisions"
        if remaining_decisions
        else "diagnostic_clear",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_preflight_status": governance_report["status"],
        "source_blockers": governance_report.get("blockers", []),
        "summary": {
            "issue_count": len(issue_dispositions),
            "resolved_provenance_check_count": sum(
                len(disposition.resolved_provenance_checks) for disposition in issue_dispositions
            ),
            "remaining_owner_decision_count": len(remaining_decisions),
            "claim_boundary_note_count": sum(
                len(disposition.claim_boundary_notes) for disposition in issue_dispositions
            ),
        },
        "issue_dispositions": [disposition.to_dict() for disposition in issue_dispositions],
        "remaining_owner_decisions": remaining_decisions,
    }


def _markdown_list(items: list[str] | tuple[str, ...]) -> str:
    """Render Markdown list or an explicit empty-state sentence."""
    if not items:
        return "- None recorded."
    return "\n".join(f"- {item}" for item in items)


def render_markdown(packet: dict[str, Any]) -> str:
    """Render packet as a reviewable Markdown disposition note."""
    lines = [
        "# SNQI Governance Disposition Packet",
        "",
        "Plain-language summary: current Social Navigation Quality Index (SNQI) "
        "governance diagnostics are visible and fail closed, but owner decisions "
        "remain before any canonical SNQI weight or normalization claim can be made.",
        "",
        f"- Schema: `{packet['schema_version']}`",
        f"- Packet status: `{packet['packet_status']}`",
        f"- Source preflight status: `{packet['source_preflight_status']}`",
        f"- Claim boundary: {packet['claim_boundary']}",
        "",
        "## Summary",
        "",
        f"- Open governance issues summarized: {packet['summary']['issue_count']}",
        "- Resolved provenance checks separated here: "
        f"{packet['summary']['resolved_provenance_check_count']}",
        "- Remaining owner decisions separated here: "
        f"{packet['summary']['remaining_owner_decision_count']}",
        f"- Claim-boundary notes separated here: {packet['summary']['claim_boundary_note_count']}",
        "",
        "## Issue Dispositions",
        "",
    ]

    for disposition in packet["issue_dispositions"]:
        lines.extend(
            [
                f"### Issue #{disposition['issue']}: {disposition['title']}",
                "",
                f"- Current status: `{disposition['current_status']}`",
                "",
                "Resolved provenance checks:",
                _markdown_list(disposition["resolved_provenance_checks"]),
                "",
                "Remaining owner decisions:",
                _markdown_list(disposition["remaining_owner_decisions"]),
                "",
                "Claim-boundary notes:",
                _markdown_list(disposition["claim_boundary_notes"]),
                "",
                "Source references:",
                _markdown_list(disposition["source_refs"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Read-Only Boundary",
            "",
            "- This packet does not run a benchmark campaign.",
            "- This packet does not submit Slurm or GPU work.",
            "- This packet does not edit dissertation, paper, or benchmark claims.",
            "- This packet does not change SNQI scoring, weights, normalization, or artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true", help="Print packet JSON to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    governance_report = build_governance_report(repo_root=args.repo_root)
    packet = build_packet(governance_report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "snqi_governance_disposition.json"
    markdown_path = args.output_dir / "snqi_governance_disposition.md"
    json_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(packet), encoding="utf-8")

    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
