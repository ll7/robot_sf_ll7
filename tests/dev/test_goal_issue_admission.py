"""Tests for the issue-claim admission wrapper."""

from __future__ import annotations

from unittest.mock import patch

from scripts.dev.goal_issue_admission import admit_issue


def _preflight(*, ready: bool) -> dict[str, object]:
    return {
        "schema": "issue_implementability.v1",
        "classification": "ready" if ready else "needs_spec",
        "ready": ready,
        "write_allowed": ready,
    }


def test_non_ready_issue_never_attempts_claim() -> None:
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=False),
        ),
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["outcome"] == "not_admitted"
    assert payload["write_attempted"] is False
    acquire.assert_not_called()


def test_check_only_ready_issue_performs_no_write() -> None:
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=True),
        ),
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=True,
        )

    assert payload["ok"] is True
    assert payload["outcome"] == "ready_check_only"
    assert payload["write_attempted"] is False
    acquire.assert_not_called()


def test_ready_issue_calls_atomic_claim_once() -> None:
    claim = {"ok": True, "claimed": True, "sha": "abc"}
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=True),
        ),
        patch(
            "scripts.dev.goal_issue_admission.issue_claim.acquire_issue",
            return_value=claim,
        ) as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["ok"] is True
    assert payload["outcome"] == "claim_acquired"
    assert payload["write_attempted"] is True
    assert payload["claim"] == claim
    acquire.assert_called_once_with(
        7611,
        repo="ll7/robot_sf_ll7",
        remote="origin",
        source_ref="origin/main",
    )


def test_atomic_claim_failure_remains_explicit() -> None:
    claim = {"ok": False, "claimed": False, "error": "claim exists"}
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=True),
        ),
        patch(
            "scripts.dev.goal_issue_admission.issue_claim.acquire_issue",
            return_value=claim,
        ),
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["ok"] is False
    assert payload["outcome"] == "claim_failed"
    assert payload["write_attempted"] is True
