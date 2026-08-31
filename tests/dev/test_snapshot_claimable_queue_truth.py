"""Regression tests for truthful claimable-queue discovery."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev import goal_issue_admission, issue_implementability, snapshot_issue_batch
from scripts.dev.github_quota import RateLimitSnapshot


def _healthy_rate_limit() -> RateLimitSnapshot:
    return RateLimitSnapshot(
        status="ok",
        graphql_remaining=4_000,
        graphql_reset_at=1_800_000_000,
        core_remaining=4_000,
        core_reset_at=1_800_000_000,
    )


def _claim_status(number: int) -> dict[str, object]:
    return {
        "ok": True,
        "claimed": False,
        "claim_ref": f"agent-claims/issue-{number}",
        "sha": None,
    }


def _ready_issue(number: int) -> dict[str, object]:
    return {
        "number": number,
        "title": f"ready issue {number}",
        "state": "OPEN",
        "url": f"https://github.test/issues/{number}",
        "labels": [{"name": issue_implementability.READY_LABEL}],
        "assignees": [],
    }


def _listing(
    listed: list[dict[str, object]],
    *,
    status: str = "ok",
    error: str = "",
    resume_cursor: dict[str, object] | None = None,
    data_source: str = "graphql",
) -> dict[str, object]:
    return {
        "status": status,
        "listed": listed,
        "error": error,
        "data_source": data_source,
        "rate_limit": {},
        "quota": {},
        "resume_cursor": resume_cursor,
    }


def _admission(number: int, *, ready: bool) -> dict[str, object]:
    preflight = {
        "schema": "issue_implementability.v1",
        "classification": "ready" if ready else "needs_spec",
        "admission_reason": "claimable" if ready else "needs_spec",
        "reasons": ["issue is ready" if ready else "missing required contract section"],
        "ready": ready,
        "write_allowed": ready,
        "claim": _claim_status(number),
    }
    return {
        "schema": goal_issue_admission.SCHEMA,
        "ok": ready,
        "outcome": "ready_check_only" if ready else "not_admitted",
        "write_attempted": False,
        "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
        "preflight": preflight,
        "claim": preflight["claim"],
    }


def test_claimable_discovery_filters_ready_candidates_before_limit() -> None:
    """The bounded page must be scoped to ready candidates before it is limited."""
    with patch(
        "scripts.dev.snapshot_issue_batch._list_open_issues",
        return_value=_listing([]),
    ) as listing:
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    listing.assert_called_once_with(
        repo="ll7/robot_sf_ll7",
        limit=20,
        min_graphql_remaining=snapshot_issue_batch.DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
        resume_page=1,
        label=issue_implementability.READY_LABEL,
    )
    assert payload["candidate_scope"] == issue_implementability.READY_LABEL
    assert payload["candidate_universe_complete"] is True
    assert payload["queue_status"] == "exhausted"
    assert payload["zero_work_authoritative"] is True


def test_graphql_claimable_listing_receives_ready_label_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GraphQL-backed gh command must carry the server-side ready filter."""
    monkeypatch.setattr(snapshot_issue_batch, "_rate_limit_snapshot", _healthy_rate_limit)
    with patch("scripts.dev.snapshot_issue_batch._gh") as gh:
        gh.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    args = gh.call_args.args[0]
    assert args[0:2] == ["issue", "list"]
    assert args[args.index("--label") + 1] == issue_implementability.READY_LABEL
    assert payload["queue_status"] == "exhausted"
    assert payload["zero_work_authoritative"] is True


def test_rest_claimable_listing_receives_ready_label_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The low-GraphQL REST fallback must preserve the same ready-candidate scope."""
    monkeypatch.setattr(
        snapshot_issue_batch,
        "_rate_limit_snapshot",
        lambda: RateLimitSnapshot(
            status="ok",
            graphql_remaining=0,
            graphql_reset_at=1_800_000_123,
            core_remaining=4_000,
            core_reset_at=1_800_000_456,
        ),
    )
    with patch("scripts.dev.snapshot_issue_batch._gh") as gh:
        gh.return_value = MagicMock(returncode=0, stdout=json.dumps([]), stderr="")
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    args = gh.call_args.args[0]
    assert args[0:2] == ["api", "repos/ll7/robot_sf_ll7/issues"]
    assert f"labels={issue_implementability.READY_LABEL}" in args
    assert payload["data_source"] == "rest"
    assert payload["queue_status"] == "exhausted"
    assert payload["zero_work_authoritative"] is True


def test_partial_zero_is_incomplete_not_genuine_zero_work() -> None:
    """A resumable ready-candidate page may expose an observed zero, never exhaustion."""
    issues = [_ready_issue(8120), _ready_issue(8121)]
    claims = {number: _claim_status(number) for number in (8120, 8121)}
    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing(
                issues,
                resume_cursor={"source": "rest", "page": 2, "limit": 2},
                data_source="rest",
            ),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value=claims,
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            side_effect=[_admission(8120, ready=False), _admission(8121, ready=False)],
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=2,
        )

    assert payload["claimable_count"] == 0
    assert payload["candidate_universe_complete"] is False
    assert payload["queue_status"] == "incomplete"
    assert payload["zero_work_authoritative"] is False


def test_complete_nonclaimable_ready_scan_is_authoritative_zero() -> None:
    """A complete successful ready-candidate scan may prove genuine zero work."""
    issue = _ready_issue(8121)
    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing([issue]),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={8121: _claim_status(8121)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            return_value=_admission(8121, ready=False),
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    assert payload["claimable_count"] == 0
    assert payload["candidate_universe_complete"] is True
    assert payload["queue_status"] == "exhausted"
    assert payload["zero_work_authoritative"] is True


def test_claimable_row_wins_even_when_candidate_scan_is_partial() -> None:
    """Found work remains actionable even when more ready candidates exist later."""
    issue = _ready_issue(8045)
    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing(
                [issue],
                resume_cursor={"source": "rest", "page": 2, "limit": 1},
                data_source="rest",
            ),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={8045: _claim_status(8045)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            return_value=_admission(8045, ready=True),
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=1,
        )

    assert payload["claimable_count"] == 1
    assert payload["queue_status"] == "claimable"
    assert payload["zero_work_authoritative"] is False


def test_unavailable_or_resumed_zero_is_not_authoritative() -> None:
    """Quota failures and scans that start after page one cannot prove global exhaustion."""
    with patch(
        "scripts.dev.snapshot_issue_batch._list_open_issues",
        return_value=_listing(
            [],
            status="quota_blocked",
            error="quota unavailable",
            resume_cursor={"source": "rest", "page": 1, "limit": 20},
            data_source="none",
        ),
    ):
        unavailable = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    with patch(
        "scripts.dev.snapshot_issue_batch._list_open_issues",
        return_value=_listing([], data_source="rest"),
    ):
        resumed = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
            resume_page=2,
        )

    assert unavailable["queue_status"] == "unavailable"
    assert unavailable["zero_work_authoritative"] is False
    assert resumed["candidate_universe_complete"] is False
    assert resumed["queue_status"] == "incomplete"
    assert resumed["zero_work_authoritative"] is False
