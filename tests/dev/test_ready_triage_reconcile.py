"""Tests for the ready/triage contradiction reconciliation planner (issue #9012).

The planner must derive one evidence-backed action per contradictory issue and must not mutate
labels without a current, drift-checked classification.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

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


def test_apply_defers_when_live_classification_changes_even_if_action_does_not() -> None:
    raw = _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"])
    row = reconcile.plan_row(raw, repo="ll7/robot_sf_ll7")
    live = _raw_issue(
        labels=["state:ready", "needs-triage", "type:workflow", "slurm"],
    )
    removed: list[tuple[int, str]] = []

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": live,
        label_remover=lambda number, label, repo="": (
            removed.append((number, label)) or {"status": "ok"}
        ),
    )

    assert applied["action"] == "deferred"
    assert "stale plan" in applied["reason"]
    assert removed == []


def test_apply_defers_when_live_issue_is_closed() -> None:
    raw = _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"])
    row = reconcile.plan_row(raw, repo="ll7/robot_sf_ll7")
    live = _raw_issue(labels=["state:ready", "needs-triage", "type:workflow"])
    live["state"] = "closed"
    removed: list[tuple[int, str]] = []

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": live,
        label_remover=lambda number, label, repo="": (
            removed.append((number, label)) or {"status": "ok"}
        ),
    )

    assert applied["action"] == "deferred"
    assert "not open" in applied["reason"]
    assert removed == []


def test_apply_defers_when_live_contract_is_incomplete() -> None:
    live = _raw_issue(labels=["state:ready", "needs-triage"], body=INCOMPLETE_BODY)
    row = reconcile.plan_row(live, repo="ll7/robot_sf_ll7")
    planned_action = row["action"]
    removed: list[tuple[int, str]] = []

    applied = reconcile.apply_row(
        row,
        repo="ll7/robot_sf_ll7",
        fetch_issue=lambda number, repo="": live,
        label_remover=lambda number, label, repo="": (
            removed.append((number, label)) or {"status": "ok"}
        ),
    )

    assert planned_action == "remove_ready"
    assert applied["action"] == "deferred"
    assert "contract is incomplete" in applied["reason"]
    assert removed == []


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


@pytest.mark.parametrize("max_pages", [0, -1])
def test_inventory_rejects_nonpositive_page_budgets(
    max_pages: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(path: str) -> object:
        raise AssertionError(f"REST must not run for max_pages={max_pages}: {path}")

    monkeypatch.setattr(reconcile, "run_gh_api_or_raise", fail_if_called)

    with pytest.raises(ValueError, match="max_pages must be >= 1"):
        reconcile.list_contradictory_issues("ll7/robot_sf_ll7", max_pages=max_pages)


def test_inventory_fails_closed_when_page_budget_ends_on_full_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_page = [
        _raw_issue(
            labels=["state:ready", "needs-triage", "type:workflow"],
            number=number,
        )
        for number in range(1, reconcile.PAGE_SIZE + 1)
    ]
    paths: list[str] = []

    def fake_run(path: str) -> object:
        paths.append(path)
        return object()

    monkeypatch.setattr(reconcile, "run_gh_api_or_raise", fake_run)
    monkeypatch.setattr(reconcile, "parse_json", lambda result, *, what: (full_page, None))

    with pytest.raises(reconcile.InventoryIncompleteError, match="incomplete"):
        reconcile.list_contradictory_issues("ll7/robot_sf_ll7", max_pages=1)

    assert paths == [
        "repos/ll7/robot_sf_ll7/issues?state=open&labels=state:ready,needs-triage"
        "&per_page=100&page=1"
    ]


def test_main_reports_incomplete_inventory_without_planning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_scan(repo: str, *, max_pages: int) -> list[dict[str, Any]]:
        raise reconcile.InventoryIncompleteError(pages_read=max_pages, max_pages=max_pages)

    monkeypatch.setattr(reconcile, "list_contradictory_issues", fail_scan)

    exit_code = reconcile.main(["--json", "--max-pages", "1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["inventory"]["complete"] is False
    assert payload["inventory"]["truncated"] is True
    assert payload["rows"] == []
