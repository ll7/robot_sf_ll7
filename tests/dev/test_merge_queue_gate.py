"""Regression tests for the merge-queue status-check gate."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev import check_pr_ci_status as ci_status
from scripts.dev import merge_queue_gate as merge_queue_gate_module
from scripts.dev.merge_queue_gate import (
    CI_PATHS_IGNORE_PATTERNS,
    _format_summary,
    _rest_check_rollup,
    _rest_requested_reviewers,
    _rest_reviews,
    _to_receipt_check_runs,
    evaluate_merge_gate,
    fetch_merge_queue_strategy,
    fetch_pr_snapshot,
    fetch_threads_resolved,
    main,
)
from scripts.dev.pr_metadata import metadata_digest, metadata_trailer

FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
METADATA_DIGEST = "b" * 64


def _gh_response(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Create a mock ``subprocess.CompletedProcess`` for GitHub CLI calls."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def test_receipt_check_projection_binds_focused_review_to_exact_head_and_metadata() -> None:
    checks = _to_receipt_check_runs(
        [
            {
                "name": "pr-contract-check",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
            {
                "name": "CI",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
            {"context": "CodeRabbit", "status": None, "conclusion": None},
        ],
        head_sha=FULL_SHA,
        expected_metadata_digest=METADATA_DIGEST,
    )

    assert checks[0]["head_sha"] == FULL_SHA
    assert checks[0]["approved_source"] is True
    assert checks[0]["metadata_digest"] == METADATA_DIGEST
    assert checks[1]["approved_source"] is False
    assert checks[1]["metadata_digest"] is None
    assert [check["name"] for check in checks] == ["pr-contract-check", "CI"]


def test_receipt_check_projection_drops_superseded_duplicate_runs() -> None:
    """Receipt evidence must use the newest run for a repeated Actions job."""
    checks = _to_receipt_check_runs(
        [
            None,
            "malformed rollup entry",
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "startedAt": "2026-08-21T13:46:14Z",
                "status": "COMPLETED",
                "conclusion": "CANCELLED",
            },
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "startedAt": "2026-08-21T13:46:30Z",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
            {
                "__typename": "CheckRun",
                "name": "CI",
                "workflowName": "CI",
                "startedAt": "2026-08-21T13:46:12Z",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
        ],
        head_sha=FULL_SHA,
        expected_metadata_digest=METADATA_DIGEST,
    )

    assert [check["name"] for check in checks] == ["pr-body-contracts", "CI"]
    assert checks[0]["conclusion"] == "success"


def _exact_changed_coverage_response(
    *, head_sha: str = FULL_SHA, status: str = "completed", conclusion: str | None = "success"
) -> MagicMock:
    """Build the exact-head REST check-run response used by live gate fixtures."""
    return _gh_response(
        stdout=json.dumps(
            {
                "total_count": 1,
                "check_runs": [
                    {
                        "id": 7001,
                        "name": "changed-coverage-gate",
                        "head_sha": head_sha,
                        "status": status,
                        "conclusion": conclusion,
                        "started_at": "2026-08-18T01:00:00Z",
                        "completed_at": "2026-08-18T01:01:00Z" if status == "completed" else None,
                    }
                ],
            }
        )
    )


def _changed_files_response(*filenames: str) -> MagicMock:
    """Build a REST pull-files response for missing-proof scope tests."""
    return _gh_response(stdout=json.dumps([{"filename": filename} for filename in filenames]))


def _raw_pr(
    *,
    body: str = "",
    carrier: str = "comments",
    author_association: str = "OWNER",
) -> dict[str, object]:
    """Build raw ``gh pr view`` data with an optional comment/review body."""
    payload: dict[str, object] = {
        "number": 42,
        "title": "merge queue test PR",
        "body": "final body",
        "state": "OPEN",
        "mergedAt": None,
        "isDraft": False,
        "headRefOid": FULL_SHA,
        "labels": [{"name": "merge-ready"}],
        "statusCheckRollup": [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {
                "name": "changed-coverage-gate",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
        ],
        "comments": [
            {
                "body": metadata_trailer(metadata_digest("merge queue test PR", "final body")),
                "authorAssociation": "OWNER",
            }
        ],
        "reviews": [],
        "reviewRequests": [],
    }
    if body:
        payload[carrier] = [
            *payload.get(carrier, []),
            {"body": body, "authorAssociation": author_association},
        ]
    return payload


def _review_threads_payload(
    *, nodes: list[dict[str, object]], total_count: int, has_next_page: bool
) -> dict[str, object]:
    """Build a GraphQL review-thread connection payload."""
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": nodes,
                        "totalCount": total_count,
                        "pageInfo": {"hasNextPage": has_next_page},
                    }
                }
            }
        }
    }


def _merge_queue_strategy_payload(strategy: str) -> dict[str, object]:
    """Build a GraphQL merge-queue configuration payload."""
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "mergeQueueEntry": {
                        "mergeQueue": {"configuration": {"mergingStrategy": strategy}}
                    }
                }
            }
        }
    }


def test_fetch_pr_snapshot_uses_supported_gh_fields_and_rest_base_sha() -> None:
    """Live snapshots avoid unsupported ``baseRefOid`` and obtain ``base.sha`` via REST."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr())),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["base_sha"] == "base_sha"
    assert snapshot["pr_state"] == "OPEN"
    assert snapshot["pr_merged_at"] is None
    assert snapshot["data_source"] == "graphql"
    assert snapshot["evidence_provenance"]["ordinary_facts"]["check_rollup"] == "graphql"
    assert snapshot["evidence_provenance"]["ordinary_facts"]["base_sha"] == "rest"
    first_call = mock_gh.call_args_list[0].args[0]
    assert first_call[:3] == ["pr", "view", "42"]
    fields = first_call[first_call.index("--json") + 1]
    assert "baseRefOid" not in fields
    assert "reviewRequests" in fields
    assert mock_gh.call_args_list[1].args[0] == ["api", "repos/owner/repo/pulls/42"]


def test_fetch_pr_snapshot_preserves_terminal_merged_state() -> None:
    raw_pr = _raw_pr()
    raw_pr["state"] = "CLOSED"
    raw_pr["mergedAt"] = "2026-08-23T19:28:15Z"
    raw_pr["mergeCommit"] = {"oid": FULL_SHA}
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(raw_pr)),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["pr_state"] == "MERGED"
    assert snapshot["pr_merged_at"] == "2026-08-23T19:28:15Z"
    assert snapshot["merge_commit_sha"] == FULL_SHA


def _rest_pull_response(*, head_sha: str = FULL_SHA, body: str = "") -> MagicMock:
    """Build a REST ``pulls/{n}`` response carrying the gate-critical fields."""
    payload = {
        "number": 42,
        "title": "merge queue test PR",
        "body": body or "final body",
        "state": "open",
        "merged_at": None,
        "draft": False,
        "head": {"sha": head_sha},
        "base": {"sha": "base_sha"},
        "labels": [{"name": "merge-ready"}],
    }
    return _gh_response(stdout=json.dumps(payload))


def _rest_comments_response(*bodies: str) -> MagicMock:
    """Build a REST issue-comments response with owner association."""
    return _gh_response(
        stdout=json.dumps(
            [
                {
                    "body": body,
                    "author_association": "OWNER",
                    "user": {"login": "maintainer"},
                }
                for body in bodies
            ]
        )
    )


def test_fetch_pr_snapshot_rest_fallback_when_graphql_quota_exhausted() -> None:
    """Issue #7705: quota-exhausted gh pr view falls back to REST reads.

    The GraphQL-backed ``gh pr view`` is blocked, but the gate must still
    refresh hosted-check evidence from REST endpoints (PR, comments, reviews,
    requested reviewers, and head check-runs) rather than reporting a generic
    unavailable snapshot.
    """
    quota_error = _gh_response(
        stdout="gh: GraphQL: API rate limit already exceeded (403)",
        returncode=0,
    )
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            quota_error,  # gh pr view -> GraphQL quota exhausted
            _rest_pull_response(head_sha=FULL_SHA),  # REST pulls/42 (core)
            _rest_comments_response("comment one"),  # issues/42/comments
            _gh_response(stdout=json.dumps([])),  # pulls/42/reviews
            _gh_response(  # pulls/42/requested_reviewers
                stdout=json.dumps(
                    {
                        "users": [{"login": "external-reviewer"}],
                        "teams": [{"slug": "core-reviewers", "name": "Core Reviewers"}],
                    }
                )
            ),
            _exact_changed_coverage_response(
                head_sha=FULL_SHA
            ),  # commits/{sha}/check-runs (rollup)
            _gh_response(stdout=json.dumps([])),  # commits/{sha}/statuses (legacy contexts)
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),  # base pulls/42
            _exact_changed_coverage_response(head_sha=FULL_SHA),  # changed-coverage check-runs
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None, error
    assert snapshot["head_sha"] == FULL_SHA
    assert snapshot["pr_state"] == "OPEN"
    assert snapshot["pr_merged_at"] is None
    assert snapshot["base_sha"] == "base_sha"
    assert snapshot["draft"] is False
    assert snapshot["labels"] == ["merge-ready"]
    assert snapshot["reviewers_requested"] is True
    assert snapshot["requested_reviewers"] == ["external-reviewer"]
    assert snapshot["requested_teams"] == ["Core Reviewers"]
    assert snapshot["comment_snapshot"]["latest"][0]["body_excerpt"] == "comment one"
    assert snapshot["data_source"] == "rest_fallback_graphql_quota"
    assert snapshot["evidence_provenance"]["data_source"] == "rest_fallback_graphql_quota"
    assert snapshot["evidence_provenance"]["ordinary_facts"]["check_rollup"] == "rest"
    assert snapshot["evidence_provenance"]["ordinary_facts"]["labels"] == "rest"
    assert snapshot["evidence_provenance"]["review_threads"] == {
        "source": "graphql",
        "status": "separate_query",
    }
    assert mock_gh.call_args_list[2].args[0][-1].endswith("/issues/42/comments?per_page=100&page=1")
    assert mock_gh.call_args_list[3].args[0][-1].endswith("/pulls/42/reviews?per_page=100&page=1")
    # The REST fallback must be exercised (pr view hit with quota failure first).
    first_call = mock_gh.call_args_list[0].args[0]
    assert first_call[:3] == ["pr", "view", "42"]


def test_rest_check_rollup_paginates_and_includes_legacy_statuses() -> None:
    """The REST rollup must not discard later check runs or commit-status failures."""
    check = {
        "id": 7001,
        "name": "CI",
        "head_sha": FULL_SHA,
        "status": "completed",
        "conclusion": "success",
    }
    page_one = {"total_count": 101, "check_runs": [check] * 100}
    page_two = {"total_count": 101, "check_runs": [check]}
    status = {
        "context": "legacy-gate",
        "state": "failure",
        "target_url": "https://example.test/status",
    }
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(page_one)),
            _gh_response(stdout=json.dumps(page_two)),
            _gh_response(stdout=json.dumps([status])),
        ]
        rollup, error = _rest_check_rollup(owner="owner", name="repo", head_sha=FULL_SHA)

    assert error is None
    assert len(rollup) == 102
    assert any(item["name"] == "legacy-gate" and item["conclusion"] == "FAILURE" for item in rollup)
    assert "page=2" in mock_gh.call_args_list[1].args[0][-1]
    assert mock_gh.call_args_list[2].args[0][-1].endswith("/statuses?per_page=100&page=1")


def test_rest_check_rollup_enriches_workflow_identity_for_superseded_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST fallback must project Actions identity before deduplicating reruns."""
    predecessor = {
        "name": "pr-body-contracts",
        "head_sha": FULL_SHA,
        "status": "completed",
        "conclusion": "cancelled",
        "started_at": "2026-08-21T13:46:14Z",
        "completed_at": "2026-08-21T13:46:20Z",
        "details_url": "https://github.com/ll7/robot_sf_ll7/actions/runs/101/job/1001",
    }
    replacement = {
        **predecessor,
        "conclusion": "success",
        "started_at": "2026-08-21T13:46:30Z",
        "completed_at": "2026-08-21T13:46:36Z",
        "details_url": "https://github.com/ll7/robot_sf_ll7/actions/runs/102/job/1002",
    }

    monkeypatch.setattr(
        merge_queue_gate_module,
        "_rest_check_runs",
        lambda **_: ([predecessor, replacement], None),
    )
    monkeypatch.setattr(merge_queue_gate_module, "_rest_commit_statuses", lambda **_: ([], None))
    monkeypatch.setattr(ci_status, "_WORKFLOW_ID_BY_RUN_ID", {})

    def _fake_rest(path: str) -> dict[str, int] | None:
        return {"workflow_id": 9001} if path in {"actions/runs/101", "actions/runs/102"} else None

    monkeypatch.setattr(ci_status, "_rest_api_get", _fake_rest)

    rollup, error = _rest_check_rollup(owner="owner", name="repo", head_sha=FULL_SHA)

    assert error is None
    assert [item["workflowId"] for item in rollup] == ["9001", "9001"]
    effective, superseded = merge_queue_gate_module._latest_check_runs(rollup)
    assert superseded == 1
    assert len(effective) == 1
    assert effective[0]["conclusion"] == "success"


def test_rest_check_rollup_rejects_incomplete_total_count() -> None:
    """A short page that contradicts total_count cannot become a green rollup."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout=json.dumps(
                {
                    "total_count": 101,
                    "check_runs": [
                        {
                            "name": "CI",
                            "head_sha": FULL_SHA,
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ],
                }
            )
        )
        rollup, error = _rest_check_rollup(owner="owner", name="repo", head_sha=FULL_SHA)

    assert rollup == []
    assert error is not None
    assert "incomplete" in error


def test_rest_reviews_reject_malformed_entry_and_paginates() -> None:
    """Review state is complete only when every bounded page has valid identity data."""
    valid_review = {
        "id": 1,
        "body": "",
        "state": "APPROVED",
        "user": {"login": "reviewer"},
        "author_association": "OWNER",
        "commit_id": FULL_SHA,
    }
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps([valid_review] * 100)),
            _gh_response(stdout=json.dumps([{**valid_review, "state": "CHANGES_REQUESTED"}])),
        ]
        reviews, error = _rest_reviews(owner="owner", name="repo", pr_number=42)

    assert error is None
    assert len(reviews) == 101
    assert reviews[-1]["state"] == "CHANGES_REQUESTED"

    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout=json.dumps([{"state": "APPROVED", "author_association": "OWNER"}])
        )
        reviews, error = _rest_reviews(owner="owner", name="repo", pr_number=42)

    assert reviews == []
    assert error is not None
    assert "malformed" in error


def test_rest_requested_reviewers_reject_malformed_objects() -> None:
    """Missing reviewer identity must not be normalized to an empty request set."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps({"users": [{}], "teams": []}))
        reviewers, error = _rest_requested_reviewers(owner="owner", name="repo", pr_number=42)

    assert reviewers == []
    assert error is not None
    assert "malformed" in error


def test_fetch_pr_snapshot_does_not_fallback_on_transient_graphql_exhaustion() -> None:
    """Only recognized GraphQL quota exhaustion may enter the REST fallback route."""
    with (
        patch(
            "scripts.dev.merge_queue_gate._gh",
            side_effect=[_gh_response(returncode=1, stderr="HTTP 503 Service Unavailable")] * 3,
        ),
        patch("scripts.dev.github_graphql_retry.time.sleep", lambda _seconds: None),
    ):
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert snapshot == {}
    assert error is not None
    assert "after 3 attempts" in error


def test_fetch_pr_snapshot_rest_fallback_fails_closed_on_rest_failure() -> None:
    """Issue #7705: if REST is also down the snapshot fails closed, never green."""
    quota_error = _gh_response(
        stderr="gh: GraphQL: API rate limit already exceeded (403)",
        returncode=1,
    )
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            quota_error,  # gh pr view -> GraphQL quota exhausted
            _gh_response(stderr="gh api failed (HTTP 500)", returncode=1),  # REST pull down
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert snapshot == {}
    assert error is not None
    assert "HTTP 500" in error


def test_fetch_pr_snapshot_rejects_missing_review_request_data() -> None:
    """Missing reviewer-request state cannot bypass the merger preflight."""
    raw_pr = _raw_pr()
    raw_pr.pop("reviewRequests")
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(raw_pr))
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert snapshot == {}
    assert error is not None
    assert "reviewRequests" in error


def test_fetch_pr_snapshot_rejects_missing_draft_state() -> None:
    """Missing draft metadata cannot be interpreted as a non-draft PR."""
    raw_pr = _raw_pr()
    raw_pr.pop("isDraft")
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(raw_pr))
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert snapshot == {}
    assert error is not None
    assert "isDraft" in error


def test_fetch_pr_snapshot_records_outstanding_requested_reviewer() -> None:
    """A live reviewer request is preserved for the gate's fail-closed evaluation."""
    raw_pr = _raw_pr()
    raw_pr["reviewRequests"] = [{"requestedReviewer": {"login": "external-reviewer"}}]
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(raw_pr)),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["reviewers_requested"] is True


def test_fetch_pr_snapshot_ignores_superseded_failed_check_run() -> None:
    """The live gate uses the canonical current-run CI classification."""
    raw_pr = _raw_pr()
    raw_pr["statusCheckRollup"] = [
        {
            "__typename": "CheckRun",
            "name": "test",
            "workflowName": "CI",
            "startedAt": "2026-07-25T12:00:00Z",
            "status": "COMPLETED",
            "conclusion": "FAILURE",
        },
        {
            "__typename": "CheckRun",
            "name": "test",
            "workflowName": "CI",
            "startedAt": "2026-07-25T12:05:00Z",
            "status": "COMPLETED",
            "conclusion": "SUCCESS",
        },
    ]
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(raw_pr)),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["checks"] == {"overall": "success"}


def test_fetch_pr_snapshot_ignores_current_gate_check() -> None:
    """The PR-head gate does not wait on its own in-progress check run."""
    raw_pr = _raw_pr()
    raw_pr["statusCheckRollup"] = [
        {
            "__typename": "CheckRun",
            "name": "merge-queue-gate",
            "workflowName": "Merge Queue Gate",
            "startedAt": "2026-07-25T12:05:00Z",
            "status": "IN_PROGRESS",
            "conclusion": None,
        },
        {
            "__typename": "CheckRun",
            "name": "test",
            "workflowName": "CI",
            "startedAt": "2026-07-25T12:00:00Z",
            "status": "COMPLETED",
            "conclusion": "SUCCESS",
        },
    ]
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(raw_pr)),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["checks"] == {"overall": "success"}


@pytest.mark.parametrize(
    ("rollup", "expected"),
    [
        (
            [
                {
                    "__typename": "CheckRun",
                    "name": "merge-queue-gate",
                    "workflowName": "Merge Queue Gate",
                    "startedAt": "2026-07-25T12:05:00Z",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
            "pending",
        ),
        ([{"status": "COMPLETED", "conclusion": "STALE"}], "failure"),
        ([{"status": "COMPLETED", "conclusion": None}], "pending"),
        ([{"status": "BROKEN", "conclusion": "SUCCESS"}], "unknown"),
    ],
)
def test_fetch_pr_snapshot_fails_closed_on_non_green_ci_rollups(
    rollup: list[dict[str, object]], expected: str
) -> None:
    """A missing, stale, or malformed current CI rollup cannot become green."""
    raw_pr = _raw_pr()
    raw_pr["statusCheckRollup"] = rollup
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(raw_pr)),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["checks"] == {"overall": expected}


def test_fetch_pr_snapshot_binds_changed_coverage_to_exact_head() -> None:
    """A passing coverage check on a later or unrelated SHA cannot prove this head."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr())),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _exact_changed_coverage_response(head_sha="b" * 40),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["changed_coverage"]["status"] == "stale"
    audit = evaluate_merge_gate(snapshot, main_sha="base_sha", threads_resolved=True)
    assert audit.passed is False
    assert "changed_coverage_proof_stale" in audit.reasons


def test_fetch_pr_snapshot_requires_a_changed_coverage_check() -> None:
    """A missing proof remains a blocker when a non-ignored file changed."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr())),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _gh_response(stdout=json.dumps({"total_count": 0, "check_runs": []})),
            _changed_files_response("scripts/dev/merge_queue_gate.py"),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["changed_coverage"]["status"] == "missing"
    assert snapshot["changed_coverage"]["changed_files_complete"] is True
    audit = evaluate_merge_gate(snapshot, main_sha="base_sha", threads_resolved=True)
    assert audit.passed is False
    assert "changed_coverage_proof_missing" in audit.reasons


def test_fetch_pr_snapshot_accepts_complete_docs_only_scope_without_check() -> None:
    """A skipped CI workflow is explainable only for its exact ignored paths."""
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _gh_response(stdout=json.dumps({"total_count": 0, "check_runs": []})),
            _changed_files_response(
                "README.md",
                "docs/dev_guide.md",
                ".agents/skills/example/SKILL.md",
            ),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    audit = evaluate_merge_gate(
        snapshot,
        main_sha="base_sha",
        threads_resolved=True,
        reviewers_requested=False,
    )
    assert audit.passed is True
    assert audit.changed_coverage_status == "not_required"
    assert audit.changed_coverage_head_sha == ""


def test_evaluate_merge_gate_requires_complete_scope_for_docs_only_bypass() -> None:
    """An unproven or mixed changed-file set cannot satisfy missing coverage."""
    metadata = metadata_digest("merge queue test PR", "final body")
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "base_sha": FULL_SHA,
            "draft": False,
            "labels": ["merge-ready"],
            "checks": {"overall": "success"},
            "changed_coverage": {
                "status": "missing",
                "head_sha": FULL_SHA,
                "changed_files": ["README.md"],
                "changed_files_complete": False,
            },
            "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
            "metadata_digest": metadata,
            "metadata_verdicts": [metadata_trailer(metadata)],
        },
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is False
    assert audit.changed_coverage_status == "missing"
    assert "changed_coverage_proof_missing" in audit.reasons


def test_evaluate_merge_gate_does_not_trust_unproven_not_required_status() -> None:
    """The bypass status itself is not evidence of a docs-only changed set."""
    metadata = metadata_digest("merge queue test PR", "final body")
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "base_sha": FULL_SHA,
            "draft": False,
            "labels": ["merge-ready"],
            "checks": {"overall": "success"},
            "changed_coverage": {
                "status": "not_required",
                "changed_files": ["../README.md"],
                "changed_files_complete": True,
            },
            "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
            "metadata_digest": metadata,
            "metadata_verdicts": [metadata_trailer(metadata)],
        },
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is False
    assert audit.changed_coverage_status == "unknown"
    assert "changed_coverage_proof_unknown" in audit.reasons


def test_fetch_pr_snapshot_rejects_incomplete_exact_head_check_runs() -> None:
    """A paginated check-run response cannot silently omit a newer proof."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr())),
            _gh_response(stdout=json.dumps({"base": {"sha": "base_sha"}})),
            _gh_response(
                stdout=json.dumps(
                    {
                        "total_count": 2,
                        "check_runs": [
                            {
                                "id": 7001,
                                "name": "changed-coverage-gate",
                                "head_sha": FULL_SHA,
                                "status": "completed",
                                "conclusion": "success",
                            }
                        ],
                    }
                )
            ),
            _gh_response(stderr="gh: Not Found (HTTP 404)", returncode=1),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert snapshot == {}
    assert error is not None
    assert "incomplete" in error


def test_evaluate_merge_gate_accepts_current_changed_coverage_proof() -> None:
    """A successful changed-coverage check must bind to the evaluated PR head."""
    metadata = metadata_digest("merge queue test PR", "final body")
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "base_sha": FULL_SHA,
            "draft": False,
            "labels": ["merge-ready"],
            "checks": {"overall": "success"},
            "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
            "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
            "metadata_digest": metadata,
            "metadata_verdicts": [metadata_trailer(metadata)],
        },
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is True
    assert audit.changed_coverage_status == "success"
    assert audit.changed_coverage_head_sha == FULL_SHA


def test_workflow_keeps_merge_group_hard_and_source_pr_advisory() -> None:
    """Only native merge-group evaluation is a failing required check."""
    workflow = Path(".github/workflows/merge-queue-gate.yml").read_text(encoding="utf-8")

    assert "PR_NUMBER: ${{ inputs.pr_number }}" in workflow
    assert '--pr "$PR_NUMBER"' in workflow
    assert '--pr "${{ inputs.pr_number }}"' not in workflow
    assert "pull_request:" in workflow
    assert "pull_request_review:" not in workflow
    assert "pull_request_review_comment:" not in workflow
    for activity in ("labeled", "unlabeled", "synchronize"):
        assert activity in workflow
    for noisy_activity in (
        "opened",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "review_requested",
        "review_request_removed",
    ):
        assert noisy_activity not in workflow
    assert "PR_NUMBER: ${{ github.event.pull_request.number }}" in workflow
    assert "Run merge-admission audit (source PR head)" in workflow
    assert "Source-PR admission is advisory; merge_group remains fail-closed." in workflow
    assert workflow.count("--advisory") == 2
    merge_group_step, source_pr_step = workflow.split(
        "- name: Run merge-admission audit (source PR head)",
        maxsplit=1,
    )
    assert "--advisory" not in merge_group_step
    assert "--from-event" in merge_group_step
    assert "--advisory" in source_pr_step
    assert "exit 0" in workflow  # Bootstrap skip remains advisory before the gate exists on main.
    assert "MERGE_GROUP_BASE_SHA: ${{ github.event.merge_group.base_sha }}" in workflow
    assert "PULL_REQUEST_BASE_SHA: ${{ github.event.pull_request.base.sha }}" not in workflow
    assert "PULL_REQUEST_BASE_REF: ${{ github.event.pull_request.base.ref }}" in workflow
    assert "encoded_branch=" in workflow
    assert "'$value|@uri'" in workflow
    assert 'gh api "repos/$REPOSITORY/branches/$encoded_branch"' in workflow
    assert 'printf \'ref=%s\\n\' "$trusted_ref" >> "$GITHUB_OUTPUT"' in workflow
    assert "Trusted gate revision: $trusted_ref" in workflow
    assert "checks: read" in workflow
    assert "issues: read" in workflow
    assert "ref: ${{ steps.trusted-gate.outputs.ref }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "statuses: read" in workflow
    assert "Trusted base does not contain scripts/dev/merge_queue_gate.py" in workflow
    assert "python -m pip install --quiet pyyaml==6.0.3" in workflow
    assert "conversation resolution before merging" in workflow
    assert "exit 0" in workflow


def test_docs_only_bypass_matches_ci_workflow_path_filters() -> None:
    """The gate's exemption cannot drift from the workflow that skips CI."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    for pattern in CI_PATHS_IGNORE_PATTERNS:
        assert f'      - "{pattern}"' in workflow


@pytest.mark.parametrize("carrier", ["comments", "reviews"])
def test_fetch_pr_snapshot_preserves_long_gate_verdict_trailers(carrier: str) -> None:
    """Accepted trailers after compact-body truncation remain available to the live gate."""
    long_prefix = "Detailed review feedback paragraph line. " * 6
    trailer = f"gate-verdict: accepted @ {FULL_SHA}"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(
                stdout=json.dumps(_raw_pr(body=f"{long_prefix}\n\n{trailer}", carrier=carrier))
            ),
            _gh_response(stdout=json.dumps({"base": {"sha": FULL_SHA}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["gate_verdicts"] == [trailer]
    audit = evaluate_merge_gate(snapshot, main_sha=FULL_SHA, threads_resolved=True)
    assert audit.passed is True


@pytest.mark.parametrize("carrier", ["comments", "reviews"])
def test_fetch_pr_snapshot_preserves_trusted_exact_head_base_policy(carrier: str) -> None:
    trailer = f"base-policy: ordinary-cas @ {FULL_SHA}"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=trailer, carrier=carrier))),
            _gh_response(stdout=json.dumps({"base": {"sha": "older_base"}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["base_policy"] == [trailer]


def test_remote_head_only_marker_addition_is_base_sensitive() -> None:
    marker_source = "import pytest\npytestmark = pytest.mark.base_sensitive\n"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(
                stdout=json.dumps(
                    [
                        {
                            "filename": "tests/test_remote_marker.py",
                            "status": "added",
                        }
                    ]
                )
            ),
            _gh_response(stdout=json.dumps({"sha": "c" * 40})),
            _gh_response(
                stdout=json.dumps(
                    {
                        "encoding": "base64",
                        "content": base64.b64encode(marker_source.encode()).decode(),
                    }
                )
            ),
            _gh_response(stderr="gh: Not Found (HTTP 404)", returncode=1),
        ]

        inventory, error = merge_queue_gate_module.fetch_pr_changed_file_marker_inventory(
            42,
            repo="owner/repo",
            base_sha="b" * 40,
            head_sha=FULL_SHA,
            current_main_sha="c" * 40,
        )

    assert error is None
    assert inventory is not None
    assert inventory["changed_file_records"] == [
        {
            "filename": "tests/test_remote_marker.py",
            "previous_filename": None,
            "status": "added",
        }
    ]
    assert inventory["changed_sensitive_files"] == ["tests/test_remote_marker.py"]
    assert inventory["content_provenance"] == [
        {
            "base": None,
            "current_main": [
                {
                    "contains_marker": False,
                    "exists": False,
                    "path": "tests/test_remote_marker.py",
                    "ref": "c" * 40,
                }
            ],
            "filename": "tests/test_remote_marker.py",
            "head": {
                "contains_marker": True,
                "path": "tests/test_remote_marker.py",
                "ref": FULL_SHA,
            },
            "previous_filename": None,
            "status": "added",
        }
    ]


def test_marker_removed_by_modified_test_remains_base_sensitive() -> None:
    base_source = "import pytest\npytestmark = pytest.mark.base_sensitive\n"
    head_source = "def test_still_exists():\n    assert True\n"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(
                stdout=json.dumps([{"filename": "tests/test_marker.py", "status": "modified"}])
            ),
            _gh_response(stdout=json.dumps({"sha": "c" * 40})),
            _gh_response(
                stdout=json.dumps(
                    {
                        "encoding": "base64",
                        "content": base64.b64encode(base_source.encode()).decode(),
                    }
                )
            ),
            _gh_response(
                stdout=json.dumps(
                    {
                        "encoding": "base64",
                        "content": base64.b64encode(head_source.encode()).decode(),
                    }
                )
            ),
            _gh_response(
                stdout=json.dumps(
                    {
                        "encoding": "base64",
                        "content": base64.b64encode(head_source.encode()).decode(),
                    }
                )
            ),
        ]

        inventory, error = merge_queue_gate_module.fetch_pr_changed_file_marker_inventory(
            42,
            repo="owner/repo",
            base_sha="b" * 40,
            head_sha=FULL_SHA,
            current_main_sha="c" * 40,
        )

    assert error is None
    assert inventory is not None
    assert inventory["changed_sensitive_files"] == ["tests/test_marker.py"]
    assert inventory["content_provenance"][0]["base"]["contains_marker"] is True
    assert inventory["content_provenance"][0]["head"]["contains_marker"] is False


def test_marker_added_only_on_current_main_is_base_sensitive() -> None:
    ordinary_source = "def test_still_exists():\n    assert True\n"
    current_source = "import pytest\npytestmark = pytest.mark.base_sensitive\n"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(
                stdout=json.dumps([{"filename": "tests/test_marker.py", "status": "modified"}])
            ),
            _gh_response(stdout=json.dumps({"sha": "c" * 40})),
            *[
                _gh_response(
                    stdout=json.dumps(
                        {
                            "encoding": "base64",
                            "content": base64.b64encode(source.encode()).decode(),
                        }
                    )
                )
                for source in (ordinary_source, ordinary_source, current_source)
            ],
        ]

        inventory, error = merge_queue_gate_module.fetch_pr_changed_file_marker_inventory(
            42,
            repo="owner/repo",
            base_sha="b" * 40,
            head_sha=FULL_SHA,
            current_main_sha="c" * 40,
        )

    assert error is None
    assert inventory is not None
    assert inventory["changed_sensitive_files"] == ["tests/test_marker.py"]
    assert inventory["content_provenance"][0]["current_main"][0]["contains_marker"] is True


def test_changed_file_inventory_fails_closed_at_github_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(merge_queue_gate_module, "_CHANGED_FILES_PAGE_SIZE", 2)
    monkeypatch.setattr(merge_queue_gate_module, "_MAX_GITHUB_PR_FILES", 6, raising=False)
    page = [
        {"filename": "robot_sf/a.py", "status": "modified"},
        {"filename": "robot_sf/b.py", "status": "modified"},
    ]
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [_gh_response(stdout=json.dumps(page)) for _ in range(3)]

        records, error = merge_queue_gate_module._fetch_pr_changed_file_records(
            42, repo="owner/repo"
        )

    assert records is None
    assert error == "changed-file inventory reached GitHub's 6-file cap"


def test_renaming_sensitive_test_to_non_test_filename_remains_base_sensitive() -> None:
    marker_source = "import pytest\npytestmark = pytest.mark.base_sensitive\n"
    ordinary_source = "def helper():\n    return True\n"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(
                stdout=json.dumps(
                    [
                        {
                            "filename": "robot_sf/marker_helper.py",
                            "previous_filename": "tests/test_marker.py",
                            "status": "renamed",
                        }
                    ]
                )
            ),
            _gh_response(stdout=json.dumps({"sha": "c" * 40})),
            *[
                _gh_response(
                    stdout=json.dumps(
                        {
                            "encoding": "base64",
                            "content": base64.b64encode(source.encode()).decode(),
                        }
                    )
                )
                for source in (marker_source, ordinary_source)
            ],
            _gh_response(stderr="gh: Not Found (HTTP 404)", returncode=1),
            _gh_response(
                stdout=json.dumps(
                    {
                        "encoding": "base64",
                        "content": base64.b64encode(marker_source.encode()).decode(),
                    }
                )
            ),
        ]

        inventory, error = merge_queue_gate_module.fetch_pr_changed_file_marker_inventory(
            42,
            repo="owner/repo",
            base_sha="b" * 40,
            head_sha=FULL_SHA,
            current_main_sha="c" * 40,
        )

    assert error is None
    assert inventory is not None
    assert inventory["changed_file_records"] == [
        {
            "filename": "robot_sf/marker_helper.py",
            "previous_filename": "tests/test_marker.py",
            "status": "renamed",
        }
    ]
    assert inventory["candidate_files"] == ["robot_sf/marker_helper.py"]
    assert inventory["changed_sensitive_files"] == ["robot_sf/marker_helper.py"]
    assert inventory["content_provenance"][0]["previous_filename"] == "tests/test_marker.py"


def test_invalid_current_main_ref_cannot_be_treated_as_path_absence() -> None:
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(
                stdout=json.dumps([{"filename": "tests/test_marker.py", "status": "added"}])
            ),
            _gh_response(stderr="gh: Not Found (HTTP 404)", returncode=1),
        ]

        inventory, error = merge_queue_gate_module.fetch_pr_changed_file_marker_inventory(
            42,
            repo="owner/repo",
            base_sha="b" * 40,
            head_sha=FULL_SHA,
            current_main_sha="c" * 40,
        )

    assert inventory is None
    assert error == "gh: Not Found (HTTP 404)"


def test_changed_file_marker_inventory_rejects_unknown_status() -> None:
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout=json.dumps([{"filename": "robot_sf/example.py", "status": "mystery"}])
        )

        inventory, error = merge_queue_gate_module.fetch_pr_changed_file_marker_inventory(
            42,
            repo="owner/repo",
            base_sha="b" * 40,
            head_sha=FULL_SHA,
            current_main_sha="c" * 40,
        )

    assert inventory is None
    assert error == "changed file has unsupported status mystery: robot_sf/example.py"


@pytest.mark.parametrize("carrier", ["comments", "reviews"])
def test_fetch_pr_snapshot_ignores_untrusted_gate_verdict_authors(carrier: str) -> None:
    """A contributor cannot self-approve a retained merge-ready label after pushing."""
    raw_pr = _raw_pr(
        body=f"gate-verdict: accepted @ {FULL_SHA}",
        carrier=carrier,
        author_association="CONTRIBUTOR",
    )
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(raw_pr)),
            _gh_response(stdout=json.dumps({"base": {"sha": FULL_SHA}})),
            _exact_changed_coverage_response(),
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["gate_verdicts"] == []
    audit = evaluate_merge_gate(snapshot, main_sha=FULL_SHA, threads_resolved=True)
    assert audit.passed is False
    assert "missing_exact_head_gate_verdict" in audit.reasons


def test_fetch_threads_resolved_rejects_incomplete_connection() -> None:
    """An unresolved thread beyond the first page must fail closed rather than bypass the gate."""
    resolved = {"isResolved": True, "isOutdated": False}
    payload = _review_threads_payload(nodes=[resolved] * 100, total_count=101, has_next_page=True)
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(payload))
        resolved_state, error = fetch_threads_resolved(42, repo="owner/repo")

    assert resolved_state is None
    assert error is not None
    assert "incomplete" in error
    query = mock_gh.call_args.args[0][mock_gh.call_args.args[0].index("-f") + 1]
    assert "totalCount" in query
    assert "pageInfo" in query


def test_fetch_threads_resolved_accepts_complete_resolved_connection() -> None:
    """A complete all-resolved thread connection passes the actionable-thread check."""
    payload = _review_threads_payload(
        nodes=[{"isResolved": True, "isOutdated": False}], total_count=1, has_next_page=False
    )
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(payload))
        resolved_state, error = fetch_threads_resolved(42, repo="owner/repo")

    assert resolved_state is True
    assert error is None


def test_fetch_merge_queue_strategy_reads_allgreen_configuration() -> None:
    """The live gate can prove every constituent queue entry must pass."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout=json.dumps(_merge_queue_strategy_payload("ALLGREEN"))
        )
        strategy, error = fetch_merge_queue_strategy(42, repo="owner/repo")

    assert strategy == "ALLGREEN"
    assert error is None


def test_fetch_merge_queue_strategy_rejects_missing_queue_entry() -> None:
    """A race or incomplete GraphQL response cannot bypass the queue strategy gate."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout=json.dumps({"data": {"repository": {"pullRequest": {"mergeQueueEntry": None}}}})
        )
        strategy, error = fetch_merge_queue_strategy(42, repo="owner/repo")

    assert strategy is None
    assert error is not None


def test_fetch_merge_queue_strategy_rejects_partial_graphql_errors() -> None:
    """Partial GraphQL data cannot bypass the queue-strategy fail-closed check."""
    payload = _merge_queue_strategy_payload("ALLGREEN")
    payload["errors"] = [{"message": "configuration may be incomplete"}]
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(payload))
        strategy, error = fetch_merge_queue_strategy(42, repo="owner/repo")

    assert strategy is None
    assert error is not None
    assert "incomplete" in error


def test_fetch_merge_queue_strategy_classifies_stdout_quota_before_json_parsing() -> None:
    """A quota diagnostic on stdout stays a quota failure even with exit code zero."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(
            stdout="GraphQL: API rate limit already exceeded.",
            returncode=0,
        )
        strategy, error = fetch_merge_queue_strategy(42, repo="owner/repo")

    assert strategy is None
    assert error is not None
    assert "GraphQL quota exhausted" in error


def test_fetch_threads_resolved_rejects_partial_graphql_errors() -> None:
    """Partial GraphQL data cannot hide an unresolved review thread."""
    payload = _review_threads_payload(nodes=[], total_count=0, has_next_page=False)
    payload["errors"] = [{"message": "thread data may be incomplete"}]
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(payload))
        resolved_state, error = fetch_threads_resolved(42, repo="owner/repo")

    assert resolved_state is None
    assert error is not None
    assert "incomplete" in error


def test_fetch_threads_resolved_retries_transient_graphql_and_fails_closed() -> None:
    """The merge gate never treats an exhausted GraphQL outage as thread-free."""
    with (
        patch(
            "scripts.dev.merge_queue_gate._gh",
            side_effect=[_gh_response(returncode=1, stderr="HTTP 503 Service Unavailable")] * 3,
        ),
        patch("scripts.dev.github_graphql_retry.time.sleep", lambda _seconds: None),
    ):
        resolved_state, error = fetch_threads_resolved(42, repo="owner/repo")

    assert resolved_state is None
    assert error is not None
    assert "after 3 attempts" in error


def test_fetch_threads_resolved_quota_exhaustion_carries_reset_handoff() -> None:
    """Quota-blocked thread reads stay fail-closed but name the reset and retry (issue #8282)."""
    handoff = {
        "quota_reset_at": 1_800_000_000,
        "reset_in_seconds": 30,
        "retry_after_utc": "2027-01-01T00:00:00Z",
        "retry_command": "uv run python scripts/dev/single_account_merge_receipt.py --repo owner/repo --pr 42 --mode report-only --output output/validation/pr-42-merge-receipt.json",
        "handoff": "GraphQL quota exhausted; quota resets at 2027-01-01T00:00:00Z (in ~30s). Retry after reset with: uv run python scripts/dev/single_account_merge_receipt.py --repo owner/repo --pr 42 --mode report-only --output output/validation/pr-42-merge-receipt.json. Never admit merge-ready from unknown thread state.",
    }
    with (
        patch(
            "scripts.dev.merge_queue_gate._gh",
            return_value=_gh_response(
                returncode=0, stdout="GraphQL: API rate limit already exceeded."
            ),
        ),
        patch(
            "scripts.dev.merge_queue_gate.quota_reset_handoff",
            return_value=handoff,
        ) as mock_handoff,
    ):
        resolved_state, error = fetch_threads_resolved(42, repo="owner/repo")

    mock_handoff.assert_called_once()
    assert resolved_state is None
    assert error is not None
    assert "2027-01-01T00:00:00Z" in error
    assert "single_account_merge_receipt.py" in error
    assert "report-only" in error


def test_evaluate_live_query_failure_preserves_unknown_thread_audit() -> None:
    """A failed thread query must audit unknown state, not unresolved state."""
    base_sha = "b" * 40
    title = "live gate"
    body = "final body"
    digest = metadata_digest(title, body)
    snapshot = {
        "number": 42,
        "title": title,
        "body": body,
        "head_sha": FULL_SHA,
        "base_sha": base_sha,
        "labels": ["merge-ready"],
        "draft": False,
        "checks": {"overall": "success"},
        "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
        "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
        "metadata_digest": digest,
        "metadata_verdicts": [metadata_trailer(digest)],
        "reviewers_requested": False,
    }
    with (
        patch.object(
            merge_queue_gate_module,
            "fetch_pr_snapshot",
            return_value=(snapshot, None),
        ),
        patch.object(
            merge_queue_gate_module,
            "get_pr_commit_messages",
            return_value="repair commit\n",
        ),
        patch.object(merge_queue_gate_module, "fetch_main_sha", return_value=base_sha),
        patch.object(
            merge_queue_gate_module,
            "fetch_threads_resolved",
            return_value=(None, "quota handoff"),
        ),
    ):
        audit, error = merge_queue_gate_module._evaluate_live(42, repo="owner/repo")

    assert audit.thread_resolution == "not_evaluated"
    assert "review_threads_not_evaluated" in audit.reasons
    assert "unresolved_review_threads" not in audit.reasons
    assert audit.passed is False
    assert error == "thread resolution query failed: quota handoff"


def test_quota_thread_retry_command_quotes_repo() -> None:
    """Merge-gate retry guidance must preserve a hostile repository value as one argument."""
    with patch(
        "scripts.dev.merge_queue_gate.quota_reset_handoff",
        side_effect=lambda *, retry_command: {"handoff": retry_command},
    ) as mock_handoff:
        merge_queue_gate_module._quota_exhausted_thread_diagnostic(
            42, repo="owner/repo; touch pwned"
        )

    command = mock_handoff.call_args.kwargs["retry_command"]
    assert shlex.split(command) == [
        "uv",
        "run",
        "python",
        "scripts/dev/single_account_merge_receipt.py",
        "--repo",
        "owner/repo; touch pwned",
        "--pr",
        "42",
        "--mode",
        "report-only",
        "--output",
        "output/validation/pr-42-merge-receipt.json",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"data": None},
        {"data": {"repository": None}},
        {"data": {"repository": {"pullRequest": None}}},
    ],
)
def test_graphql_queries_reject_malformed_pull_request_data(
    payload: dict[str, object],
) -> None:
    """Malformed GraphQL data fails closed without an uncaught attribute error."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps(payload))

        strategy, strategy_error = fetch_merge_queue_strategy(42, repo="owner/repo")
        threads, threads_error = fetch_threads_resolved(42, repo="owner/repo")

    assert strategy is None
    assert strategy_error is not None
    assert threads is None
    assert threads_error is not None
    assert mock_gh.call_count == 2


def test_headgreen_merge_queue_strategy_fails_closed() -> None:
    """A passing tail entry cannot carry an earlier ungated entry through."""
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "labels": ["merge-ready"],
            "gate_verdicts": [gate_verdict],
            "checks": {"overall": "success"},
        },
        threads_resolved=True,
        merge_group_head_sha=FULL_SHA[:12],
        queue_merging_strategy="HEADGREEN",
    )

    assert audit.passed is False
    assert audit.queue_merging_strategy == "HEADGREEN"
    assert "unsafe_merge_queue_strategy:HEADGREEN" in audit.reasons


def test_evaluate_merge_gate_requires_verified_closing_discipline() -> None:
    """A live snapshot without a passing semantic-close recheck cannot be admitted."""
    body = "final body"
    digest = metadata_digest("merge queue test PR", body)
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "labels": ["merge-ready"],
            "draft": False,
            "body": body,
            "metadata_digest": digest,
            "metadata_verdicts": [metadata_trailer(digest)],
            "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
            "checks": {"overall": "success"},
            "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
            "closing_discipline": {
                "status": "unavailable",
                "blockers": ["commit metadata unavailable"],
            },
        },
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is False
    assert audit.closing_discipline_status == "unavailable"
    assert audit.closing_discipline_blockers == ["commit metadata unavailable"]
    assert "closing_discipline_unavailable" in audit.reasons


def test_evaluate_live_carries_current_closing_discipline_result() -> None:
    """The live evaluator binds the merge decision to the fresh contract recheck."""
    body = "final body"
    digest = metadata_digest("live gate", body)
    snapshot = {
        "number": 42,
        "title": "live gate",
        "body": body,
        "head_sha": FULL_SHA,
        "base_sha": FULL_SHA,
        "labels": ["merge-ready"],
        "draft": False,
        "checks": {"overall": "success"},
        "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
        "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
        "metadata_digest": digest,
        "metadata_verdicts": [metadata_trailer(digest)],
        "reviewers_requested": False,
    }
    with (
        patch.object(merge_queue_gate_module, "fetch_pr_snapshot", return_value=(snapshot, None)),
        patch.object(merge_queue_gate_module, "fetch_main_sha", return_value=FULL_SHA),
        patch.object(merge_queue_gate_module, "fetch_threads_resolved", return_value=(True, None)),
        patch.object(
            merge_queue_gate_module,
            "get_pr_commit_messages",
            return_value="Closes: #8414",
        ) as mock_commits,
        patch.object(
            merge_queue_gate_module,
            "check_closes_discipline",
            return_value=["incident blocker"],
        ) as mock_check,
    ):
        audit, error = merge_queue_gate_module._evaluate_live(42, repo="owner/repo")

    assert error is None
    assert audit.passed is False
    assert audit.closing_discipline_status == "blocked"
    assert audit.closing_discipline_blockers == ["incident blocker"]
    assert "closing_discipline_blocked" in audit.reasons
    mock_commits.assert_called_once_with("42", "owner/repo")
    mock_check.assert_called_once_with(
        body,
        "owner/repo",
        commit_messages="Closes: #8414",
        commit_messages_checked=True,
    )


def test_outstanding_requested_reviewer_fails_closed() -> None:
    """An explicit reviewer request receives the same fail-closed merger-preflight treatment."""
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "labels": ["merge-ready"],
            "gate_verdicts": [gate_verdict],
            "checks": {"overall": "success"},
        },
        threads_resolved=True,
        reviewers_requested=True,
    )

    assert audit.passed is False
    assert audit.reviewer_request_status == "requested"
    assert "outstanding_requested_reviewers" in audit.reasons


def test_metadata_verdict_is_required_for_native_queue_admission() -> None:
    """A current gate verdict alone cannot admit a changed final PR state."""
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "labels": ["merge-ready"],
            "draft": False,
            "gate_verdicts": [gate_verdict],
            "checks": {"overall": "success"},
        },
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is False
    assert audit.metadata_verdict_status == "missing"
    assert "missing_pr_metadata_verdict" in audit.reasons


def test_stale_metadata_verdict_is_distinguished_from_missing() -> None:
    """A prior metadata digest is reported as stale rather than accepted."""
    current_digest = metadata_digest("current title", "current body")
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "labels": ["merge-ready"],
            "draft": False,
            "metadata_digest": current_digest,
            "metadata_verdicts": [metadata_trailer(metadata_digest("old title", "old body"))],
            "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
            "checks": {"overall": "success"},
        },
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is False
    assert audit.metadata_digest == current_digest
    assert audit.metadata_verdict_status == "stale"
    assert "stale_pr_metadata_verdict" in audit.reasons


def test_evaluate_merge_gate_fails_closed_when_runtime_dimensions_are_missing() -> None:
    """The pure evaluator must not pass when live preflight dimensions are omitted."""
    audit = evaluate_merge_gate(
        {
            "number": 42,
            "head_sha": FULL_SHA,
            "draft": False,
            "labels": ["merge-ready"],
            "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
        }
    )

    assert audit.passed is False
    assert "ci_not_green:unknown" in audit.reasons
    assert "review_threads_not_evaluated" in audit.reasons
    assert "requested_reviewers_not_evaluated" in audit.reasons


def test_from_event_resolves_canonical_queue_ref_and_binds_pr_head(tmp_path) -> None:
    """The live merge_group path uses its encoded PR and matching source SHA."""
    event_path = tmp_path / "merge_group.json"
    event_path.write_text(
        json.dumps(
            {
                "event_name": "merge_group",
                "merge_group": {
                    "head_ref": (f"refs/heads/gh-readonly-queue/main/pr-42-{FULL_SHA[:12]}"),
                    "base_sha": "queue_base_sha",
                },
            }
        ),
        encoding="utf-8",
    )
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    threads = _review_threads_payload(nodes=[], total_count=0, has_next_page=False)

    with (
        patch("scripts.dev.merge_queue_gate._gh") as mock_gh,
        patch.object(
            merge_queue_gate_module,
            "get_pr_commit_messages",
            return_value="repair commit\n",
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": "stale_base_sha"}})),
            _exact_changed_coverage_response(),
            _gh_response(stdout=json.dumps(_merge_queue_strategy_payload("ALLGREEN"))),
            _gh_response(stdout=json.dumps(threads)),
        ]
        exit_code = main(["--from-event", str(event_path), "--repo", "owner/repo"])

    assert exit_code == 0
    calls = [call.args[0] for call in mock_gh.call_args_list]
    assert calls[0][:3] == ["pr", "view", "42"]
    assert ["pr", "list"] not in [call[:2] for call in calls]


def test_from_event_accepts_branch_name_queue_ref(tmp_path) -> None:
    """The event payload's branch-name queue ref resolves like its full ref form."""
    event_path = tmp_path / "merge_group.json"
    event_path.write_text(
        json.dumps(
            {
                "merge_group": {
                    "head_ref": f"gh-readonly-queue/main/pr-42-{FULL_SHA[:12]}",
                    "base_sha": "queue_base_sha",
                }
            }
        ),
        encoding="utf-8",
    )
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    threads = _review_threads_payload(nodes=[], total_count=0, has_next_page=False)

    with (
        patch("scripts.dev.merge_queue_gate._gh") as mock_gh,
        patch.object(
            merge_queue_gate_module,
            "get_pr_commit_messages",
            return_value="repair commit\n",
        ),
    ):
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": "stale_base_sha"}})),
            _exact_changed_coverage_response(),
            _gh_response(stdout=json.dumps(_merge_queue_strategy_payload("ALLGREEN"))),
            _gh_response(stdout=json.dumps(threads)),
        ]
        exit_code = main(["--from-event", str(event_path), "--repo", "owner/repo"])

    assert exit_code == 0


@pytest.mark.parametrize(
    "event",
    [[], {"event_name": "pull_request", "merge_group": {}}, {"merge_group": {}}],
)
def test_from_event_rejects_malformed_event_payload(tmp_path, capsys, event) -> None:
    """Malformed or non-queue payloads fail closed before any GitHub query."""
    event_path = tmp_path / "merge_group.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        exit_code = main(["--from-event", str(event_path), "--repo", "owner/repo"])

    assert exit_code == 1
    assert mock_gh.call_count == 0
    assert "failing closed" in capsys.readouterr().err


def test_from_event_fails_closed_when_encoded_head_differs_from_pr(tmp_path, capsys) -> None:
    """A queue ref cannot be rebound to a newer or unrelated PR head."""
    encoded_sha = "deadbeefcafe"
    event_path = tmp_path / "merge_group.json"
    event_path.write_text(
        json.dumps(
            {
                "merge_group": {
                    "head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{encoded_sha}",
                    "base_sha": "queue_base_sha",
                }
            }
        ),
        encoding="utf-8",
    )
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    threads = _review_threads_payload(nodes=[], total_count=0, has_next_page=False)

    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": "stale_base_sha"}})),
            _exact_changed_coverage_response(),
            _gh_response(stdout=json.dumps(_merge_queue_strategy_payload("ALLGREEN"))),
            _gh_response(stdout=json.dumps(threads)),
        ]
        exit_code = main(["--from-event", str(event_path), "--repo", "owner/repo"])

    audit = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert audit["merge_group_head_sha"] == encoded_sha
    assert audit["merge_group_head_binding"] == "mismatch"
    assert "merge_group_head_sha_mismatch" in audit["reasons"]


def test_pr_mode_fails_closed_when_current_main_sha_is_unavailable(capsys) -> None:
    """Standalone source-head evaluation cannot pass without a current main SHA."""
    gate_verdict = f"gate-verdict: accepted @ {FULL_SHA}"
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": FULL_SHA}})),
            _exact_changed_coverage_response(),
            _gh_response(returncode=1, stderr="main ref unavailable"),
        ]
        exit_code = main(["--pr", "42", "--repo", "owner/repo"])

    audit = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert audit["main_sha"] == ""
    assert "main_sha_unavailable" in audit["reasons"]


def test_pr_mode_records_snapshot_failure_in_audit(capsys) -> None:
    """A PR snapshot failure is visible in the machine-readable fail-closed audit."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="snapshot unavailable")
        exit_code = main(["--pr", "42", "--repo", "owner/repo"])

    audit = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert "pr_snapshot_unavailable" in audit["reasons"]


def test_pr_advisory_mode_preserves_failed_audit_but_exits_zero(capsys) -> None:
    """Source-PR audit remains truthful without presenting ordinary CI as red."""
    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="snapshot unavailable")
        exit_code = main(["--pr", "42", "--repo", "owner/repo", "--advisory"])

    captured = capsys.readouterr()
    audit = json.loads(captured.out)
    assert exit_code == 0
    assert audit["passed"] is False
    assert "pr_snapshot_unavailable" in audit["reasons"]
    assert "Source-PR admission is advisory" in captured.err


def test_merge_group_cannot_opt_into_advisory_mode(capsys) -> None:
    """Queue-time evaluation cannot bypass its fail-closed exit policy."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--from-event", "event.json", "--repo", "owner/repo", "--advisory"])

    assert excinfo.value.code == 2
    assert "--advisory is valid only with --pr" in capsys.readouterr().err


@pytest.mark.parametrize(
    "stale_body, expected_sentinel",
    [
        (
            "The PR remains unapproved and not merge-ready pending independent exact-head review and current hosted checks.",
            "not merge-ready",
        ),
        (
            "This branch remains unapproved pending independent review.",
            "remains unapproved",
        ),
        (
            "WIP: do not merge yet.",
            "do not merge",
        ),
        (
            "The change is unapproved and not merge-ready.",
            "unapproved and not merge-ready",
        ),
    ],
)
def test_evaluate_merge_gate_fails_closed_on_stale_not_ready_body_narrative(
    stale_body: str,
    expected_sentinel: str,
) -> None:
    """A merge-ready PR carrying unapproved/not-ready narrative sentinels fails closed."""
    title = "fix: valid title"
    digest = metadata_digest(title, stale_body)
    snapshot = {
        "number": 42,
        "title": title,
        "body": stale_body,
        "draft": False,
        "head_sha": FULL_SHA,
        "base_sha": FULL_SHA,
        "labels": ["merge-ready"],
        "checks": {"overall": "success"},
        "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
        "gate_verdict": {"verdict": "accepted", "sha": FULL_SHA},
        "metadata_verdicts": [metadata_trailer(digest)],
        "metadata_digest": digest,
    }
    audit = evaluate_merge_gate(
        snapshot,
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )
    assert audit.passed is False
    assert "stale_not_ready_body_narrative" in audit.reasons
    assert audit.body_narrative_status == "stale"
    assert any(expected_sentinel.lower() in s.lower() for s in audit.body_not_ready_sentinels)


def test_evaluate_merge_gate_passes_with_clean_body_narrative() -> None:
    """A merge-ready PR carrying clean reconciliation narrative passes the gate."""
    title = "fix: valid title"
    clean_body = (
        "## Summary\n\nAll review comments addressed and PR verified ready.\n\n"
        f"pr-metadata: reconciled @ {metadata_digest(title, 'clean')}"
    )
    digest = metadata_digest(title, clean_body)
    snapshot = {
        "number": 42,
        "title": title,
        "body": clean_body,
        "draft": False,
        "head_sha": FULL_SHA,
        "base_sha": FULL_SHA,
        "labels": ["merge-ready"],
        "checks": {"overall": "success"},
        "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
        "gate_verdict": {"verdict": "accepted", "sha": FULL_SHA},
        "metadata_verdicts": [metadata_trailer(digest)],
        "metadata_digest": digest,
    }
    audit = evaluate_merge_gate(
        snapshot,
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )
    assert audit.passed is True
    assert audit.reasons == []
    assert audit.body_narrative_status == "clean"
    assert audit.body_not_ready_sentinels == []


# ---------------------------------------------------------------------------
# Issue #7515: stacked ancestry fails the merge gate closed
# ---------------------------------------------------------------------------


def _gate_ready_pr(**overrides: object) -> dict[str, object]:
    """Build a PR snapshot that otherwise passes every merge-gate dimension."""
    metadata = metadata_digest("merge queue test PR", "final body")
    payload: dict[str, object] = {
        "number": 42,
        "head_sha": FULL_SHA,
        "base_sha": FULL_SHA,
        "draft": False,
        "labels": ["merge-ready"],
        "checks": {"overall": "success"},
        "changed_coverage": {"status": "success", "head_sha": FULL_SHA},
        "gate_verdicts": [f"gate-verdict: accepted @ {FULL_SHA}"],
        "metadata_digest": metadata,
        "metadata_verdicts": [metadata_trailer(metadata)],
    }
    payload.update(overrides)
    return payload


def test_stacked_ancestry_fails_gate_closed() -> None:
    """A stacked-not-independently-mergeable PR must never pass the merge gate."""
    for state in (
        "undeclared_stack",
        "mismatched_declaration",
        "parent_invalidated",
        "stacked",
        "parent_merged",
    ):
        audit = evaluate_merge_gate(
            _gate_ready_pr(ancestry={"state": state}),
            main_sha=FULL_SHA,
            threads_resolved=True,
            reviewers_requested=False,
        )

        assert audit.passed is False, state
        assert audit.ancestry_state == state
        assert "stacked_ancestry_not_independently_mergeable" in audit.reasons


def test_clean_ancestry_passes_gate() -> None:
    """A clean ancestry block leaves the otherwise-ready PR mergeable."""
    audit = evaluate_merge_gate(
        _gate_ready_pr(ancestry={"state": "clean"}),
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is True
    assert audit.ancestry_state == "clean"
    assert "stacked_ancestry_not_independently_mergeable" not in audit.reasons


def test_missing_ancestry_block_is_not_evaluated() -> None:
    """Legacy snapshots without an ancestry block keep the pre-gate verdict."""
    audit = evaluate_merge_gate(
        _gate_ready_pr(),
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )

    assert audit.passed is True
    assert audit.ancestry_state == ""
    assert "stacked_ancestry_not_independently_mergeable" not in audit.reasons


def test_audit_summary_records_ancestry_state(capsys) -> None:
    """The human summary surfaces the evaluated ancestry state."""
    audit = evaluate_merge_gate(
        _gate_ready_pr(ancestry={"state": "stacked"}),
        main_sha=FULL_SHA,
        threads_resolved=True,
        reviewers_requested=False,
    )
    summary = _format_summary(audit)
    assert "ancestry state: `stacked`" in summary
    assert "stacked_ancestry_not_independently_mergeable" in summary
