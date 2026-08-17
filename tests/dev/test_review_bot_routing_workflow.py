"""Tests for the external review-bot routing workflow contract."""

from __future__ import annotations

from pathlib import Path


def test_review_bot_routing_retries_transient_api_errors_and_is_idempotent() -> None:
    """Routing must tolerate transient GitHub API failures and existing labels."""
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "review-bot-routing.yml").read_text(
        encoding="utf-8"
    )

    assert "const retryableStatuses = new Set([429, 500, 502, 503, 504]);" in workflow
    assert "const maxAttempts = 3;" in workflow
    assert "async function withGithubRetry(operation, description)" in workflow
    assert "failed after ${maxAttempts} attempts" in workflow
    assert "github.rest.issues.listLabelsOnIssue" in workflow
    assert (
        "await withGithubRetry(\n                  () => github.rest.issues.addLabels(" in workflow
    )
    assert (
        "await withGithubRetry(\n                () => github.rest.issues.removeLabel(" in workflow
    )
    assert "if (hasAutomaticLabel)" in workflow
    assert "error.status !== 404" in workflow
