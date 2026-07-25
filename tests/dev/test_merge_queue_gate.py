"""Regression tests for the merge-queue status-check gate."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.merge_queue_gate import (
    evaluate_merge_gate,
    fetch_pr_snapshot,
    fetch_threads_resolved,
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
    assert mock_gh.call_args_list[1].args[0] == ["api", "repos/owner/repo/pulls/42"]


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
