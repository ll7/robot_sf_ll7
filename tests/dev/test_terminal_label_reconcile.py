"""Focused tests for the terminal-label reconciliation planner."""

from __future__ import annotations

import json
from unittest.mock import patch

from scripts.dev.terminal_label_reconcile import (
    ACTIVE_LABELS,
    TERMINAL_CLASSES,
    fetch_item_state,
    plan_for_terminal,
    reconcile_item,
)


def _labels(*names: str) -> list[str]:
    return sorted(names)


def test_completed_removes_active_labels_and_adds_done() -> None:
    """A completed item loses active dispatch labels and gains state:done."""
    plan = plan_for_terminal(
        "completed",
        _labels("state:ready", "needs-review", "type:workflow", "ruled"),
        reason="completed",
    )
    assert plan["add"] == ["state:done"]
    assert plan["remove"] == ["needs-review", "state:ready"]
    assert plan["preserved"] == ["ruled", "type:workflow"]


def test_completed_resolves_blocked_and_decision_labels() -> None:
    """blocked:needs-maintainer and decision-required are resolved by completion."""
    plan = plan_for_terminal(
        "completed",
        _labels("blocked:needs-maintainer", "decision-required", "type:docs"),
    )
    assert plan["remove"] == ["blocked:needs-maintainer", "decision-required"]
    assert plan["preserved"] == ["type:docs"]


def test_not_planned_preserves_manual_and_removes_active() -> None:
    """not_planned keeps manual labels and clears active execution labels."""
    plan = plan_for_terminal(
        "not_planned",
        _labels("agent-ready", "state:working", "priority:4", "type:research"),
        reason="not_planned",
    )
    assert plan["remove"] == ["agent-ready", "state:working"]
    assert set(plan["preserved"]) == {"priority:4", "type:research"}
    assert "state:done" in plan["add"]


def test_duplicate_resolves_decision_and_blocked() -> None:
    """A duplicate closure resolves its decision and dependency holds."""
    plan = plan_for_terminal(
        "duplicate",
        _labels("decision-required", "blocked:some-dep", "type:docs"),
        reason="duplicate",
    )
    assert "decision-required" in plan["remove"]
    assert "blocked:some-dep" in plan["remove"]


def test_reopened_clears_terminal_marker_only() -> None:
    """A reopen must not drop active labels; only state:done is cleared."""
    plan = plan_for_terminal(
        "reopened",
        _labels("state:done", "state:running"),
        reason="reopened",
    )
    assert plan["remove"] == ["state:done"]
    assert plan["preserved"] == ["state:running"]
    assert plan["add"] == []


def test_terminal_unverified_needs_no_mutation() -> None:
    """Without a receipt the planner makes no plan and surfaces no mutation."""
    plan = plan_for_terminal(
        "terminal_unverified",
        _labels("state:working", "agent-ready"),
        reason=None,
    )
    assert plan["remove"] == []
    assert plan["add"] == []
    assert plan["preserved"] == ["agent-ready", "state:working"]


def test_pr_merged_policy() -> None:
    """A merged PR clears active review labels and merge-ready."""
    plan = plan_for_terminal(
        "pr_merged",
        _labels("merge-ready", "needs-review", "review-bot-auto", "state:done"),
        reason=None,
    )
    assert plan["remove"] == ["merge-ready", "needs-review"]
    assert "review-bot-auto" in plan["preserved"]  # bot marker is not active-only


def test_pr_closed_unmerged_policy() -> None:
    """A closed-unmerged PR clears active labels but keeps bot markers."""
    plan = plan_for_terminal(
        "pr_closed_unmerged",
        _labels("needs-review", "review-bot-auto", "state:done"),
        reason=None,
    )
    assert plan["remove"] == ["needs-review"]
    assert "review-bot-auto" in plan["preserved"]


def test_all_terminal_classes_are_supported() -> None:
    """Every declared terminal class produces a plan without error."""
    for terminal_class in TERMINAL_CLASSES:
        plan = plan_for_terminal(
            terminal_class,
            _labels("state:ready", "type:docs"),
            reason="completed" if terminal_class == "terminal_unverified" else None,
        )
        assert plan["terminal_class"] == terminal_class


def test_unknown_labels_preserved_by_default() -> None:
    """Non-controlled labels survive the plan."""
    plan = plan_for_terminal(
        "completed",
        _labels("state:ready", "evidence:smoke", "type:benchmark"),
    )
    assert plan["preserved"] == ["evidence:smoke", "type:benchmark"]


def test_active_labels_constant_matches_taxonomy() -> None:
    """The controlled active set contains the taxonomy dispatch/review labels."""
    assert "state:ready" in ACTIVE_LABELS
    assert "state:running" in ACTIVE_LABELS
    assert "needs-review" in ACTIVE_LABELS
    assert "merge-ready" in ACTIVE_LABELS


def test_report_mode_is_read_only(tmp_path) -> None:
    """Report mode computes a plan and performs no label mutations."""
    live_payload = {
        "number": 42,
        "state": "closed",
        "state_reason": "completed",
        "labels": [
            {"name": "state:ready"},
            {"name": "type:workflow"},
            {"name": "needs-review"},
        ],
    }
    with (
        patch("scripts.dev.terminal_label_reconcile.gh_api_get") as mock_get,
        patch("scripts.dev.terminal_label_reconcile.add_label") as mock_add,
        patch("scripts.dev.terminal_label_reconcile.remove_label") as mock_remove,
    ):
        mock_get.return_value = type(
            "R", (), {"returncode": 0, "stdout": json.dumps(live_payload), "stderr": ""}
        )()
        result = reconcile_item(42, "completed", repo="o/r", apply=False)

    assert result["applied"] is False
    assert result["ok"] is True
    assert result["remove"] == ["needs-review", "state:ready"]
    assert "type:workflow" in result["preserved"]
    mock_add.assert_not_called()
    mock_remove.assert_not_called()


def test_fetch_item_state_parses_labels() -> None:
    """fetch_item_state normalizes REST labels and reason fields."""
    payload = {
        "number": 7,
        "state": "closed",
        "state_reason": "completed",
        "labels": [{"name": "state:running"}, {"name": "type:docs"}],
    }
    with patch("scripts.dev.terminal_label_reconcile.gh_api_get") as mock_get:
        mock_get.return_value = type(
            "R", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        state = fetch_item_state(7, repo="o/r")
    assert state["ok"] is True
    assert state["reason"] == "completed"
    assert state["labels"] == ["state:running", "type:docs"]
