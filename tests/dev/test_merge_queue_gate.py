"""Regression tests for the merge-queue status-check gate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.merge_queue_gate import (
    evaluate_merge_gate,
    fetch_merge_queue_strategy,
    fetch_pr_snapshot,
    fetch_threads_resolved,
    main,
)

FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"


def _gh_response(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Create a mock ``subprocess.CompletedProcess`` for GitHub CLI calls."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _raw_pr(*, body: str = "", carrier: str = "comments") -> dict[str, object]:
    """Build raw ``gh pr view`` data with an optional comment/review body."""
    payload: dict[str, object] = {
        "number": 42,
        "isDraft": False,
        "headRefOid": FULL_SHA,
        "labels": [{"name": "merge-ready"}],
        "statusCheckRollup": [{"status": "COMPLETED", "conclusion": "SUCCESS"}],
        "comments": [],
        "reviews": [],
        "reviewRequests": [],
    }
    if body:
        payload[carrier] = [{"body": body}]
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
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["base_sha"] == "base_sha"
    first_call = mock_gh.call_args_list[0].args[0]
    assert first_call[:3] == ["pr", "view", "42"]
    fields = first_call[first_call.index("--json") + 1]
    assert "baseRefOid" not in fields
    assert "reviewRequests" in fields
    assert mock_gh.call_args_list[1].args[0] == ["api", "repos/owner/repo/pulls/42"]


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
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["checks"] == {"overall": expected}


def test_workflow_uses_fail_closed_source_head_gate_and_safe_manual_input() -> None:
    """Source-head state changes cannot retain an older passing required check."""
    workflow = Path(".github/workflows/merge-queue-gate.yml").read_text(encoding="utf-8")

    assert "PR_NUMBER: ${{ inputs.pr_number }}" in workflow
    assert '--pr "$PR_NUMBER"' in workflow
    assert '--pr "${{ inputs.pr_number }}"' not in workflow
    assert "pull_request:" in workflow
    assert "pull_request_review:" in workflow
    assert "pull_request_review_comment:" in workflow
    for activity in (
        "ready_for_review",
        "converted_to_draft",
        "review_requested",
        "review_request_removed",
        "submitted",
        "dismissed",
        "created",
        "deleted",
    ):
        assert activity in workflow
    assert "pull_request|pull_request_review|pull_request_review_comment)" in workflow
    assert "PR_NUMBER: ${{ github.event.pull_request.number }}" in workflow
    assert "Run merge-queue gate (source PR head)" in workflow
    assert "PR-head evaluation is advisory; merge_group enforces the gate." not in workflow
    assert "status=0" not in workflow
    assert "exit 0" in workflow  # Bootstrap skip remains advisory before the gate exists on main.
    assert "MERGE_GROUP_BASE_SHA: ${{ github.event.merge_group.base_sha }}" in workflow
    assert "PULL_REQUEST_BASE_SHA: ${{ github.event.pull_request.base.sha }}" in workflow
    assert "checks: read" in workflow
    assert "issues: read" in workflow
    assert "ref: ${{ steps.trusted-gate.outputs.ref }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "statuses: read" in workflow
    assert "Trusted base does not contain scripts/dev/merge_queue_gate.py" in workflow
    assert "conversation resolution before merging" in workflow
    assert "exit 0" in workflow


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
        ]
        snapshot, error = fetch_pr_snapshot(42, repo="owner/repo")

    assert error is None
    assert snapshot["gate_verdicts"] == [trailer]
    audit = evaluate_merge_gate(snapshot, main_sha=FULL_SHA, threads_resolved=True)
    assert audit.passed is True


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

    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": "stale_base_sha"}})),
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

    with patch("scripts.dev.merge_queue_gate._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps(_raw_pr(body=gate_verdict))),
            _gh_response(stdout=json.dumps({"base": {"sha": "stale_base_sha"}})),
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
