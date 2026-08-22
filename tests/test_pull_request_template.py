"""Tests for the repo's default pull request template contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PR_TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / "pr_default.md"


def test_pull_request_template_includes_proof_and_follow_up_sections() -> None:
    """Verify the PR template guides contributors toward proof-first reviews.

    This matters because the repo expects changes to include validation evidence,
    risk notes, and any deferred follow-up work before review.
    """

    text = PR_TEMPLATE.read_text(encoding="utf-8")

    assert "- Closes #<id>" in text
    assert "- Relates to #<id>" in text
    assert "- Closes `#<id>`" not in text
    assert "- Relates to `#<id>`" not in text

    for section in (
        "## Summary",
        "## Linked Issues",
        "## Stack / Dependency",
        "## What Changed",
        "## Why It Matters",
        "## Research / Evidence Notes",
        "## Validation / Proof",
        "## Risks / Rollback",
        "## Docs / Provenance",
        "## Downstream Propagation",
        "## Follow-Up / Residual Scope",
        "## Reviewer Notes",
    ):
        assert section in text

    # The v2 metadata block carries the machine-enforced approval, evidence,
    # performance, and follow-up fields that the old Markdown headings held.
    for field in (
        "<!-- pr-contract:v2",
        "change_class: tooling",
        "linked_issues:",
        "deferred_work:",
        "evidence:",
        "domain_approval:",
        "performance:",
    ):
        assert field in text

    assert "## Domain-Aware Approval" not in text
