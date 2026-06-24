"""Tests for conservative PR babysitter snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev.pr_babysitter_snapshot import (
    _sanitize_payload_for_output,
    build_babysitter_snapshot,
    classify_pr_action,
    load_retry_state,
    save_retry_state,
)


def _base_pr(**overrides):  # type: ignore[no-untyped-def]
    pr = {
        "number": 2999,
        "title": "example",
        "state": "OPEN",
        "head_branch": "feature",
        "head_sha": "abc123",
        "mergeable": "MERGEABLE",
        "checks": {"overall": "success", "total": 1, "by_conclusion": {"success": 1}},
        "reviews": {},
        "review_snapshot": {"total": 0},
        "comment_snapshot": {"total": 0},
        "review_thread_snapshot": {"status": "ok", "unresolved": 0, "threads": []},
        "preflight": {"status": "healthy", "reasons": ["ok"]},
    }
    pr.update(overrides)
    return pr


def test_successful_mergeable_pr_routes_to_review_for_merge_ready() -> None:
    """Green mergeable PRs still need local merge-readiness review."""
    recommendation = classify_pr_action(_base_pr(), retry_state={"prs": {}})

    assert recommendation["action"] == "review_for_merge_ready"
    assert recommendation["mutation_guardrails"]["auto_merge"] is False


def test_unresolved_review_thread_routes_to_comment_processing() -> None:
    """Submitted unresolved review threads should take priority over green CI."""
    pr = _base_pr(review_thread_snapshot={"status": "ok", "unresolved": 1, "threads": []})

    recommendation = classify_pr_action(pr, retry_state={"prs": {}})

    assert recommendation["action"] == "process_review_comment"
    assert recommendation["review_feedback"]["unresolved_threads"] == 1
    assert recommendation["mutation_guardrails"]["auto_reply_to_humans"] is False


def test_resolved_thread_suppresses_comment_review_noise() -> None:
    """Resolved review threads should not keep COMMENTED reviews actionable forever."""
    pr = _base_pr(
        reviews={"COMMENTED": 1},
        review_thread_snapshot={"status": "ok", "unresolved": 0, "threads": []},
    )

    recommendation = classify_pr_action(pr, retry_state={"prs": {}})

    assert recommendation["action"] == "review_for_merge_ready"
    assert recommendation["review_feedback"]["has_actionable_feedback"] is False


def test_trusted_author_filter_ignores_untrusted_review_threads() -> None:
    """Optional trusted-author filtering should suppress untrusted review threads."""
    pr = _base_pr(
        review_thread_snapshot={
            "status": "ok",
            "unresolved": 1,
            "threads": [
                {
                    "resolved": False,
                    "comments": [{"author": "unknown-reviewer"}],
                }
            ],
        }
    )

    recommendation = classify_pr_action(
        pr,
        retry_state={"prs": {}},
        trusted_authors={"trusted-reviewer"},
    )

    assert recommendation["action"] == "review_for_merge_ready"
    assert recommendation["review_feedback"]["trusted_unresolved_threads"] == 0


def test_trusted_author_filter_accepts_trusted_review_threads() -> None:
    """Trusted unresolved review threads should route to comment processing."""
    pr = _base_pr(
        review_thread_snapshot={
            "status": "ok",
            "unresolved": 1,
            "threads": [
                {
                    "resolved": False,
                    "comments": [{"author": "trusted-reviewer"}],
                }
            ],
        }
    )

    recommendation = classify_pr_action(
        pr,
        retry_state={"prs": {}},
        trusted_authors={"trusted-reviewer"},
    )

    assert recommendation["action"] == "process_review_comment"
    assert recommendation["review_feedback"]["trusted_unresolved_threads"] == 1


def test_failed_ci_recommends_diagnosis_before_bounded_retry() -> None:
    """Failed CI should diagnose logs before any retry recommendation is acted on."""
    pr = _base_pr(checks={"overall": "failure", "by_conclusion": {"failure": 1}})

    recommendation = classify_pr_action(pr, retry_state={"prs": {}}, retry_budget=1)

    assert recommendation["action"] == "diagnose_ci_failure"
    assert recommendation["after_diagnosis_action"] == "retry_failed_checks"
    assert recommendation["retry_budget"]["remaining"] == 1


def test_retry_budget_exhaustion_stops_retry_recommendation() -> None:
    """Retry recommendations are keyed by PR and head SHA."""
    pr = _base_pr(checks={"overall": "failure", "by_conclusion": {"failure": 1}})
    retry_state = {"prs": {"2999:abc123": {"retry_recommendations": 1}}}

    recommendation = classify_pr_action(pr, retry_state=retry_state, retry_budget=1)

    assert recommendation["action"] == "diagnose_ci_failure"
    assert recommendation["after_diagnosis_action"] == "stop_retry_budget_exhausted"
    assert recommendation["retry_budget"]["exhausted"] is True


def test_sanitized_output_preserves_retry_budget_shape_without_raw_details() -> None:
    """CLI output summary keeps retry accounting but omits verbose check detail."""
    payload = {
        "schema": "pr_babysitter_snapshot.v1",
        "repo": "ll7/robot_sf_ll7",
        "source_schema": "pr_queue_snapshot.v1",
        "route_evidence_only": True,
        "prs": [
            {
                "number": 2999,
                "state": "OPEN",
                "mergeable": "MERGEABLE",
                "checks": {
                    "failed": [
                        {
                            "name": "CodeQL",
                            "details_url": "https://example.invalid/redacted-detail",
                        }
                    ]
                },
                "recommendation": {
                    "action": "diagnose_ci_failure",
                    "after_diagnosis_action": "retry_failed_checks",
                    "retry_budget": {
                        "key": "2999:abc123",
                        "budget": 2,
                        "retry_recommendations": 1,
                        "remaining": 1,
                        "exhausted": False,
                    },
                },
            }
        ],
    }

    safe_payload = _sanitize_payload_for_output(payload)

    assert "checks" not in safe_payload["prs"][0]
    assert safe_payload["prs"][0]["recommendation"]["retry_budget"] == {
        "budget": 2,
        "retry_recommendations": 1,
        "remaining": 1,
        "exhausted": False,
    }


def test_closed_pr_stops_babysitting() -> None:
    """Closed or merged PRs are terminal for the babysitter."""
    recommendation = classify_pr_action(_base_pr(state="MERGED"), retry_state={"prs": {}})

    assert recommendation["action"] == "stop_pr_closed"


def test_none_fields_do_not_stringify_to_none_or_crash() -> None:
    """Explicit None values from API JSON should be treated as missing."""
    pr = _base_pr(
        state=None,
        mergeable=None,
        checks={"overall": None},
        preflight={"status": None},
        comment_snapshot=None,
    )

    recommendation = classify_pr_action(pr, retry_state={"prs": {}})

    assert recommendation["action"] == "wait"
    assert recommendation["review_feedback"]["issue_comments"] == 0


def test_build_snapshot_wraps_compact_pr_queue_and_records_retry(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The wrapper should reuse compact queue snapshots and persist retry accounting."""
    state_file = tmp_path / "retry-state.json"
    pr_payload = {
        "number": 3001,
        "title": "failed PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/3001",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "def456",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "failure"}],
        "reviews": [],
        "comments": [],
    }
    thread_payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "totalCount": 0,
                        "nodes": [],
                    }
                }
            }
        }
    }
    retry_state = load_retry_state(state_file)
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr=""),
            MagicMock(returncode=0, stdout=json.dumps(thread_payload), stderr=""),
        ]
        payload = build_babysitter_snapshot(
            [3001],
            repo="ll7/robot_sf_ll7",
            retry_state=retry_state,
            retry_budget=1,
            record_retry_recommendations=True,
        )
    save_retry_state(state_file, retry_state)

    pr = payload["prs"][0]
    assert payload["schema"] == "pr_babysitter_snapshot.v1"
    assert pr["recommendation"]["action"] == "diagnose_ci_failure"
    assert pr["recommendation"]["after_diagnosis_action"] == "stop_retry_budget_exhausted"
    persisted = load_retry_state(state_file)
    assert persisted["prs"]["3001:def456"]["retry_recommendations"] == 1
