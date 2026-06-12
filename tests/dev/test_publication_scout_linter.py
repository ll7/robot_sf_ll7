"""Tests for publication scout linter helpers and fixture-backed classifications."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.dev import publication_scout_linter as linter

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests/fixtures/publication_scout_linter"


def _fixture(name: str) -> object:
    """Load one JSON fixture from test resources."""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _fixed_now() -> datetime:
    """Deterministic timestamp used by recency tests."""
    return datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)


def test_validate_candidate_flags_stale_issue_state() -> None:
    """Closed issue state should create a conformance failure."""
    issue = _fixture("issue_closed.json")
    assert isinstance(issue, dict)
    comments = _fixture("comments_recent.json")

    findings = linter.validate_candidate(
        issue,
        comments,
        expected_repo="ll7/robot_sf_ll7",
        recent_comment_hours=24,
        now=_fixed_now(),
    )

    assert findings and findings[0].code == "issue_not_open"


def test_validate_candidate_flags_wrong_repo_owner() -> None:
    """Wrong repository owner/repo should be a hard failure."""
    issue = _fixture("issue_wrong_repo.json")
    assert isinstance(issue, dict)
    comments = _fixture("comments_recent.json")

    findings = linter.validate_candidate(
        issue,
        comments,
        expected_repo="ll7/robot_sf_ll7",
        recent_comment_hours=24,
        now=_fixed_now(),
    )
    assert any(finding.code == "issue_repo_mismatch" for finding in findings)


def test_validate_candidate_flags_missing_recent_comments() -> None:
    """Missing recency-window comments should fail readiness checks."""
    issue = _fixture("issue_open.json")
    assert isinstance(issue, dict)
    comments = _fixture("comments_stale.json")

    findings = linter.validate_candidate(
        issue,
        comments,
        expected_repo="ll7/robot_sf_ll7",
        recent_comment_hours=24,
        now=_fixed_now(),
    )

    assert any(finding.code == "missing_recent_comments" for finding in findings)


def test_classify_graphql_publish_payload_flags_forbidden() -> None:
    """GraphQL errors should be classified into non-ok publication status."""
    payload = _fixture("comment_graphql_error.json")
    classified = linter.classify_comment_publication_result(payload)

    assert classified["ok"] is False
    assert classified["status"] == "forbidden"


def test_main_reports_failures_for_stale_issue(
    capsys,
) -> None:
    """CLI should report deterministic failures for a stale issue payload."""
    issue_path = FIXTURE_DIR / "issue_closed.json"
    comments_path = FIXTURE_DIR / "comments_recent.json"
    exit_code = linter.main(
        [
            "--issue-json",
            str(issue_path),
            "--comments-json",
            str(comments_path),
            "--recent-comment-hours",
            "24",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["schema"] == "publication_scout_linter.v1"
    assert payload["ok"] is False
    assert any(
        check["name"] == "issue_state_open" and check["ok"] is False for check in payload["checks"]
    )
