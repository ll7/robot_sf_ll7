"""Regression tests for fail-closed claimable-queue truth."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from scripts.dev import (
    goal_issue_admission,
    issue_implementability,
    snapshot_issue_batch,
)


def _claim_status(number: int, *, ok: bool = True) -> dict[str, object]:
    """Return one compact claim-state fixture."""
    return {
        "ok": ok,
        "claimed": False if ok else None,
        "claim_ref": f"agent-claims/issue-{number}",
        "sha": None,
    }


def _ready_issue(number: int) -> dict[str, object]:
    """Return one state:ready candidate-list row."""
    return {
        "number": number,
        "title": f"ready issue {number}",
        "state": "OPEN",
        "url": f"https://github.test/issues/{number}",
        "labels": [{"name": issue_implementability.READY_LABEL}],
        "assignees": [],
    }


def _listing(
    rows: list[dict[str, object]],
    *,
    data_source: str = "graphql",
    resume_cursor: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return one successful bounded listing fixture."""
    return {
        "status": "ok",
        "listed": rows,
        "error": "",
        "data_source": data_source,
        "rate_limit": {},
        "quota": {},
        "resume_cursor": resume_cursor,
    }


def _admission(number: int, *, ready: bool) -> dict[str, object]:
    """Return one canonical live-admission wrapper fixture."""
    claim = _claim_status(number)
    preflight = {
        "schema": "issue_implementability.v1",
        "classification": "ready" if ready else "needs_spec",
        "admission_reason": "claimable" if ready else "needs_spec",
        "reasons": ["issue is ready" if ready else "missing contract section"],
        "ready": ready,
        "write_allowed": ready,
        "claim": claim,
    }
    return {
        "schema": goal_issue_admission.SCHEMA,
        "ok": ready,
        "outcome": "ready_check_only" if ready else "not_admitted",
        "write_attempted": False,
        "source_ref": goal_issue_admission.DEFAULT_SOURCE_REF,
        "preflight": preflight,
        "claim": claim,
    }


def test_resumed_final_page_is_not_global_complete() -> None:
    """A standalone final resume page cannot prove prior pages were scanned."""
    with patch(
        "scripts.dev.snapshot_issue_batch._list_open_issues",
        return_value=_listing([], data_source="rest"),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
            resume_page=2,
        )

    assert payload["claimable_count"] == 0
    assert payload["queue_completeness"] == "incomplete"


def test_admission_error_makes_queue_unavailable() -> None:
    """An admission exception must forbid a complete zero-work conclusion."""
    number = 9001
    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing([_ready_issue(number)]),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={number: _claim_status(number)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            side_effect=RuntimeError("admission unavailable"),
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    assert payload["claimable_count"] == 0
    assert payload["queue_completeness"] == "unavailable"
    assert payload["admission_reason_histogram"] == {"error": 1}


def test_unavailable_claim_read_makes_queue_unavailable() -> None:
    """An unusable claim read must not become complete zero-work evidence."""
    number = 9002
    unavailable_claim = _claim_status(number, ok=False)
    preflight = {
        "schema": "issue_implementability.v1",
        "classification": "needs_spec",
        "admission_reason": "needs_spec",
        "reasons": ["missing contract section"],
        "ready": False,
        "write_allowed": False,
        "claim": unavailable_claim,
    }
    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing([_ready_issue(number)]),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={number: unavailable_claim},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.issue_implementability.evaluate_issue",
            return_value=preflight,
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    assert payload["claimable_count"] == 0
    assert payload["queue_completeness"] == "unavailable"


def test_complete_nonclaimable_scan_can_prove_zero_work() -> None:
    """A page-one exhaustive, fully evaluated scan may remain complete."""
    number = 9003
    with (
        patch(
            "scripts.dev.snapshot_issue_batch._list_open_issues",
            return_value=_listing([_ready_issue(number)]),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={number: _claim_status(number)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            return_value=_admission(number, ready=False),
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
        )

    assert payload["claimable_count"] == 0
    assert payload["queue_completeness"] == "complete"


@pytest.mark.parametrize(
    ("status", "expected_reason"),
    [
        ("blocked_unchanged", "blocked_receipt"),
        ("blocker_changed", "needs_re_evaluation"),
        ("re_evaluate", "needs_re_evaluation"),
    ],
)
def test_blocker_receipt_fences_final_claimable_selection(
    tmp_path: Path, status: str, expected_reason: str
) -> None:
    """A final blocker receipt must override a ready live admission."""
    number = 9004
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
            return_value=_listing([_ready_issue(number)]),
        ),
        patch(
            "scripts.dev.snapshot_issue_batch._batch_claim_statuses",
            return_value={number: _claim_status(number)},
        ),
        patch(
            "scripts.dev.snapshot_issue_batch.goal_issue_admission.admit_issue",
            return_value=_admission(number, ready=True),
        ),
    ):
        payload = snapshot_issue_batch.snapshot_claimable_issues(
            repo="ll7/robot_sf_ll7",
            remote="origin",
            body_limit=150,
            limit=20,
            blocker_decision_paths=[str(decision_path)],
        )

    row = payload["issues"][0]
    assert row["dispatch_allowed"] is False
    assert row["classification"] == expected_reason
    assert payload["claimable_issues"] == []
    assert payload["claimable_count"] == 0
    assert payload["admission_reason_histogram"] == {expected_reason: 1}
    assert payload["queue_completeness"] == "complete"
