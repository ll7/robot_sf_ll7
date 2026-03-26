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

    for section in (
        "## Summary",
        "## Linked Issues",
        "## What Changed",
        "## Why It Matters",
        "## Validation / Proof",
        "## Risks / Rollout",
        "## Docs / Provenance",
        "## Follow-Up Issues",
        "## Reviewer Notes",
    ):
        assert section in text

    assert "Commands run:" in text
    assert "Evidence that the change works here:" in text
    assert "Deferred work:" in text
    assert "Issues opened for follow-up:" in text
