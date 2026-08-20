"""Tests for the issue-claim admission wrapper."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

from scripts.dev.goal_issue_admission import admit_issue, compact_admission


def _preflight(*, ready: bool) -> dict[str, object]:
    return {
        "schema": "issue_implementability.v1",
        "issue": {
            "number": 7611,
            "title": "bounded workflow repair",
            "state": "OPEN",
            "labels": ["state:ready"],
            "assignees": [],
        },
        "classification": "ready" if ready else "needs_spec",
        "contract": {"body_sha256": "body-a"},
        "claim": {"ok": True, "claimed": False, "claim_ref": None, "sha": None},
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
    assert payload["revalidation"] == {"performed": True, "inputs_match": True}
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


def test_changed_issue_inputs_fail_closed_before_claim_write() -> None:
    initial = _preflight(ready=True)
    changed = _preflight(ready=True)
    changed["contract"] = {"body_sha256": "body-b"}
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            side_effect=[initial, changed],
        ) as live_report,
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
    assert payload["revalidation"] == {"performed": True, "inputs_match": False}
    assert live_report.call_count == 2
    acquire.assert_not_called()


def test_changed_label_that_makes_issue_non_ready_fails_closed() -> None:
    initial = _preflight(ready=True)
    changed = _preflight(ready=False)
    changed["issue"] = {
        **changed["issue"],
        "labels": ["state:ready", "state:blocked"],
    }
    changed["classification"] = "blocked"
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            side_effect=[initial, changed],
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
    assert payload["preflight"]["classification"] == "blocked"
    assert payload["claim_outcome"] == "unclaimed"
    acquire.assert_not_called()


def test_compact_admission_preserves_claim_and_write_boundary() -> None:
    """Queue projections must retain the canonical admission and claim outcomes."""
    payload = compact_admission(
        {
            "ok": True,
            "outcome": "ready_check_only",
            "write_attempted": False,
            "source_ref": "origin/main",
            "preflight": {
                "classification": "ready",
                "reasons": ["contract is complete"],
                "ready": True,
                "write_allowed": True,
                "claim": {
                    "ok": True,
                    "claimed": False,
                    "claim_ref": "agent-claims/issue-7611",
                    "sha": None,
                },
            },
        }
    )

    assert payload["outcome"] == "ready_check_only"
    assert payload["write_attempted"] is False
    assert payload["ready"] is True
    assert payload["write_allowed"] is True
    assert payload["claim_outcome"] == "unclaimed"


def test_only_goal_admission_calls_the_atomic_issue_claim_owner() -> None:
    """Repository call sites must not bypass the canonical admission wrapper."""
    root = Path(__file__).resolve().parents[2]
    call_sites: set[str] = set()
    for base in (root / "scripts" / "dev", root / "tests" / "dev"):
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                is_issue_claim_owner = (
                    isinstance(owner, ast.Name) and owner.id == "issue_claim"
                ) or (isinstance(owner, ast.Attribute) and owner.attr == "issue_claim")
                if node.func.attr == "acquire_issue" and is_issue_claim_owner:
                    call_sites.add(path.relative_to(root).as_posix())

    assert call_sites == {"scripts/dev/goal_issue_admission.py"}
