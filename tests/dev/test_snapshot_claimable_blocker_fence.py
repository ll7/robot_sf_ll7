"""Regression coverage for blocker-receipt dispatch fences in claimable queues."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.dev import goal_issue_admission, issue_implementability, snapshot_issue_batch


def _claim_status(number: int) -> dict[str, object]:
    """Return a compact unclaimed claim-state fixture."""
    return {
        "ok": True,
        "claimed": False,
        "claim_ref": f"agent-claims/issue-{number}",
        "sha": None,
    }


def _ready_issue(number: int) -> dict[str, object]:
    """Return one ready candidate-list row."""
    return {
        "number": number,
        "title": f"ready issue {number}",
        "state": "OPEN",
        "url": f"https://github.test/issues/{number}",
        "labels": [{"name": issue_implementability.READY_LABEL}],
        "assignees": [],
    }


def _listing(issue: dict[str, object]) -> dict[str, object]:
    """Return a complete one-row candidate listing."""
    return {
        "status": "ok",
        "listed": [issue],
        "error": "",
        "data_source": "graphql",
        "rate_limit": {},
        "quota": {},
        "resume_cursor": None,
    }


def _ready_admission(number: int) -> dict[str, object]:
    """Return the canonical wrapper shape for a live-admitted ready issue."""
    preflight = {
        "schema": "issue_implementability.v1",
        "classification": "ready",
        "admission_reason": "claimable",
        "reasons": ["issue is ready"],
        "ready": True,
        "write_allowed": True,
        "claim": _claim_status(number),
    }
    return {
        "schema": goal_issue_admission.SCHEMA,
        "ok": True,
        "outcome": "ready_check_only",
        "write_attempted": False,
        "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
        "preflight": preflight,
        "claim": preflight["claim"],
    }


@pytest.mark.parametrize(
    ("status", "expected_reason"),
    [
        ("blocked_unchanged", "blocked_receipt"),
        ("blocker_changed", "needs_re_evaluation"),
        ("re_evaluate", "needs_re_evaluation"),
    ],
)
def test_blocker_receipt_removes_ready_row_from_claimable_queue(
    tmp_path, status: str, expected_reason: str
) -> None:  # type: ignore[no-untyped-def]
    """A live-ready row remains non-dispatchable when its blocker receipt fences it."""
    number = 8045
    decision_path = tmp_path / "blocker-decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "issue": number,
                "status": status,
                "reason": "external blocker decision",
                "receipt_digest": "a" * 64,
                "current_fingerprint": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing(_ready_issue(number)),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={number: _claim_status(number)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            return_value=_ready_admission(number),
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=2,
            blocker_decision_paths=[str(decision_path)],
        )

    row = payload["issues"][0]
    assert row["dispatch_allowed"] is False
    assert row["classification"] == expected_reason
    assert payload["claimable_issues"] == []
    assert payload["claimable_count"] == 0
    assert payload["admission_reason_histogram"] == {expected_reason: 1}
    assert payload["candidate_universe_complete"] is True
    assert payload["queue_status"] == "exhausted"
    assert payload["zero_work_authoritative"] is True
