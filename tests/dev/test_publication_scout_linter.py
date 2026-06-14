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


def test_classify_graphql_publish_payload_handles_null_extensions() -> None:
    """GraphQL error payloads may include null extensions and should not crash."""
    payload = _fixture("comment_graphql_null_extensions.json")
    classified = linter.classify_comment_publication_result(payload)

    assert classified["ok"] is False
    assert classified["status"] == "graphql_error"
    assert classified["error_type"] == "UNKNOWN"


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
            "999",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["schema"] == "publication_scout_linter.v1"
    assert payload["ok"] is False
    assert any(
        check["name"] == "issue_state_open" and check["ok"] is False for check in payload["checks"]
    )


def test_negative_result_guard_flags_repeat_without_rationale() -> None:
    """Repeated weak scenarios without a 'why this is different' rationale should be flagged."""
    issue = {
        "title": "Rerunning NR-001 topology reselection",
        "body": "We need to check the doorway_transfer scenario again with same thresholds.",
        "state": "open",
        "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
    }
    findings = linter.check_negative_result_guard(issue)
    assert any(
        finding.code == "repeated_negative_result_without_rationale"
        and finding.severity == "warning"
        for finding in findings
    )


def test_negative_result_guard_allows_repeat_with_rationale() -> None:
    """Repeated weak scenarios WITH a rationale should not be flagged."""
    issue = {
        "title": "Rerunning NR-002 observation-noise",
        "body": (
            "Observation-noise distant pedestrian test. Retest with a closer pedestrian and "
            "higher pedestrian count."
        ),
        "metadata": {
            "why_this_is_different": (
                "Addresses NR-002 by replacing the distant pedestrian with a near-field "
                "avoidance requirement."
            )
        },
        "state": "open",
        "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
    }
    findings = linter.check_negative_result_guard(issue)
    assert not any(
        finding.code == "repeated_negative_result_without_rationale" for finding in findings
    )


def test_negative_result_guard_ignores_unrelated_issues() -> None:
    """Unrelated issues should not trigger the negative result guard."""
    issue = {
        "title": "Fix bug in path planner",
        "body": "The A* implementation has a small bug when handling off-grid goals.",
        "state": "open",
        "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
    }
    findings = linter.check_negative_result_guard(issue)
    assert not findings


def test_negative_result_guard_flags_space_separated_observation_noise_repeat() -> None:
    """Observation noise wording should not need the exact hyphenated form."""
    issue = {
        "title": "Observation noise diagnostic for classic_bottleneck_medium",
        "body": "Repeat the scenario_too_weak distant pedestrian diagnostic.",
        "state": "open",
        "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
    }

    findings = linter.check_negative_result_guard(issue)

    assert any(finding.code == "repeated_negative_result_without_rationale" for finding in findings)


def test_negative_result_guard_ignores_null_optional_text_fields() -> None:
    """Null optional issue fields should not be coerced into searchable 'None' text."""
    issue = {
        "title": None,
        "body": 0,
        "labels": [{"name": None}, 0],
        "metadata": {"why_this_is_different": "0"},
        "state": "open",
        "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
    }

    findings = linter.check_negative_result_guard(issue)

    assert not findings


def test_negative_result_guard_ignores_non_list_labels() -> None:
    """Malformed scalar label payloads should not abort candidate text extraction."""
    issue = {
        "title": "Fix bug in path planner",
        "body": "No negative-result repeat here.",
        "labels": 1,
        "state": "open",
        "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
    }

    findings = linter.check_negative_result_guard(issue)

    assert not findings


def test_main_reports_negative_result_guard_as_warning(tmp_path, capsys) -> None:
    """Negative-result repeats should warn without failing the read-only linter."""
    issue_path = tmp_path / "issue.json"
    comments_path = tmp_path / "comments.json"
    issue_path.write_text(
        json.dumps(
            {
                "number": 2785,
                "url": "https://github.com/ll7/robot_sf_ll7/issues/2785",
                "title": "Observation noise diagnostic for classic_bottleneck_medium",
                "body": "Repeat the scenario_too_weak distant pedestrian diagnostic.",
                "state": "open",
                "repository": {"owner": {"login": "ll7"}, "name": "robot_sf_ll7"},
            }
        ),
        encoding="utf-8",
    )
    comments_path.write_text(
        json.dumps([{"createdAt": "2026-06-12T12:00:00Z"}]),
        encoding="utf-8",
    )

    exit_code = linter.main(
        [
            "--issue-json",
            str(issue_path),
            "--comments-json",
            str(comments_path),
            "--recent-comment-hours",
            "999",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["failure_summary"] is None
    assert payload["warnings"][0]["code"] == "repeated_negative_result_without_rationale"
