"""Tests for the ready/triage contradiction reconciliation planner (issue #9012).

The planner must derive one evidence-backed action per contradictory issue and must not mutate
labels without a current, drift-checked classification.
"""

from __future__ import annotations

from typing import Any

from scripts.dev import ready_triage_reconcile as reconcile

COMPLETE_BODY = """## Objective
Reconcile contradictory labels.

## Scope
- Open issues only.

## Inputs
- Live labels and issue contract.

## Acceptance Criteria
- [ ] Contradiction removed.

## Verification
- Focused tests pass.
"""

INCOMPLETE_BODY = "## Objective\nOnly an objective section exists.\n"


def _raw_issue(
    *,
    labels: list[str],
    body: str = COMPLETE_BODY,
    number: int = 7611,
) -> dict[str, Any]:
    """Build one normalized-live issue payload for planner tests."""
    return {
        "number": number,
        "title": "friction: reconcile ready and triage",
        "body": body,
        "state": "open",
        "url": f"https://github.test/issues/{number}",
        "html_url": f"https://github.test/issues/{number}",
        "labels": [{"name": label} for label in labels],
        "assignees": [],
    }


def test_plan_removes_triage_when_readiness_is_current() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"]),
        repo="ll7/robot_sf_ll7",
    )

    assert row["action"] == "remove_triage"
    assert row["classification_without_triage"] == "ready"
    assert row["blocking_labels"] == ["needs-triage"]


def test_plan_drops_ready_when_readiness_is_stale() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "needs-triage"], body=INCOMPLETE_BODY),
        repo="ll7/robot_sf_ll7",
    )

    assert row["action"] == "remove_ready"
    assert row["classification_without_triage"] == "needs_spec"


def test_plan_removes_triage_when_compute_gate_still_blocks() -> None:
    """A valid readiness decision survives a separate compute gate; only triage is removed."""
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "needs-triage", "slurm", "type:workflow"]),
        repo="ll7/robot_sf_ll7",
    )

    assert row["action"] == "remove_triage"
    assert row["classification_without_triage"] == "needs_compute"


def test_plan_reports_additional_blocking_labels() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "needs-triage", "state:blocked"]),
        repo="ll7/robot_sf_ll7",
    )

    assert row["action"] == "report_only"
    assert row["blocking_labels"] == ["needs-triage", "state:blocked"]


def test_apply_removes_triage_after_drift_check() -> None:
    raw = _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"])
    row = reconcile.plan_row(raw, repo="ll7/robot_sf_ll7")
    removed: list[tuple[int, str]] = []

    def fake_remove(number: int, label: str, *, repo: str = "") -> dict[str, Any]:
        removed.append((number, label))
        return {"status": "ok"}

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": raw,
        label_remover=fake_remove,
    )

    assert applied["applied"] is True
    assert removed == [(7611, "needs-triage")]


def test_apply_defers_on_label_drift() -> None:
    raw = _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"])
    row = reconcile.plan_row(raw, repo="ll7/robot_sf_ll7")
    drifted = _raw_issue(labels=["state:ready", "type:workflow"])

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": drifted,
        label_remover=lambda number, label, repo="": {"status": "ok"},
    )

    assert applied["action"] == "deferred"
    assert applied["applied"] is False


def test_apply_defers_when_removal_fails() -> None:
    raw = _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"])
    row = reconcile.plan_row(raw, repo="ll7/robot_sf_ll7")

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": raw,
        label_remover=lambda number, label, repo="": {"status": "error", "error": "boom"},
    )

    assert applied["action"] == "deferred"


def test_apply_never_mutates_report_only_rows() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "needs-triage", "state:blocked"]),
        repo="ll7/robot_sf_ll7",
    )

    def exploding_remover(number: int, label: str, *, repo: str = "") -> dict[str, Any]:
        raise AssertionError("report_only rows must not be mutated")

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": _raw_issue(labels=["state:ready", "needs-triage"]),
        label_remover=exploding_remover,
    )

    assert applied == row
