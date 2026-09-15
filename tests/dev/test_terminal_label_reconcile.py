"""Focused tests for the terminal-label reconciliation planner."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from scripts.dev.terminal_label_reconcile import (
    ACTIVE_LABELS,
    INVENTORY_SCHEMA,
    STATE_QUALIFIER_CLASSIFICATION,
    TERMINAL_CLASSES,
    _collect_closed_items,
    _terminal_class_from_state,
    fetch_item_state,
    main,
    plan_for_terminal,
    reconcile_item,
    run_terminal_inventory,
)


def _labels(*names: str) -> list[str]:
    return sorted(names)


def _pr_state(
    *labels: str,
    state: str = "closed",
    is_pull_request: bool = True,
    merged_at: str | None = "2026-09-15T00:00:00Z",
) -> dict[str, Any]:
    """Build a normalized live item state for apply-mode receipt tests."""
    return {
        "ok": True,
        "number": 9356,
        "state": state,
        "reason": None,
        "labels": _labels(*labels),
        "is_pull_request": is_pull_request,
        "pull_request": {} if is_pull_request else None,
        "merged_at": merged_at if is_pull_request else None,
        "html_url": "https://github.com/o/r/pulls/9356"
        if is_pull_request
        else "https://github.com/o/r/issues/9356",
    }


def _terminal_remove_ok(label: str) -> dict[str, Any]:
    """Build a guarded terminal-removal success receipt for mocked reconciler calls."""
    return {
        "status": "ok",
        "number": 9356,
        "label": label,
        "action": "remove",
        "repo": "o/r",
        "target": "pr",
        "operation": "terminal_label_remove",
        "expected_head_sha": "a" * 40,
        "expected_base_sha": "b" * 40,
        "observed_state": "CLOSED",
        "observed_head_sha": "a" * 40,
        "observed_base_sha": "b" * 40,
        "merged_at": "2026-09-15T00:00:00Z",
    }


def test_cli_help_lists_every_terminal_class(capsys: pytest.CaptureFixture[str]) -> None:
    """The item syntax is discoverable without a failed live invocation."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for terminal_class in TERMINAL_CLASSES:
        assert terminal_class in help_text


def test_cli_rejects_unknown_terminal_class_before_live_reads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid classes fail before report construction can read GitHub state."""
    with patch("scripts.dev.terminal_label_reconcile.build_report") as mock_build:
        return_code = main(["--item", "42=merged"])

    assert return_code == 2
    mock_build.assert_not_called()
    assert "unknown terminal class 'merged'" in capsys.readouterr().out


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
        _labels(
            "merge-ready", "merge-if-ci-green", "needs-review", "review-bot-auto", "state:done"
        ),
        reason=None,
    )
    assert plan["remove"] == ["merge-if-ci-green", "merge-ready", "needs-review"]
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
    assert state["is_pull_request"] is False


def test_fetch_item_state_preserves_pull_request_merge_identity() -> None:
    """Exact-item reads retain enough PR identity to classify merged rows."""
    payload = {
        "number": 8,
        "state": "closed",
        "state_reason": None,
        "labels": [],
        "pull_request": {"merged_at": "2026-08-01T00:00:00Z"},
    }
    with patch("scripts.dev.terminal_label_reconcile.gh_api_get") as mock_get:
        mock_get.return_value = type(
            "R", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        state = fetch_item_state(8, repo="o/r")
    assert state["ok"] is True
    assert state["is_pull_request"] is True
    assert state["merged_at"] == "2026-08-01T00:00:00Z"
    assert _terminal_class_from_state(state) == "pr_merged"


def test_inventory_rejects_malformed_listing_labels() -> None:
    """Malformed label entries make an inventory non-applicable."""
    item = _closed_item(17, labels=["type:docs"])
    item["labels"] = [{"not_name": "type:docs"}]
    import pytest as _pytest

    with _pytest.raises(ValueError, match="label entry 0 was malformed"):
        run_terminal_inventory(fixture=[item], observed_at="2026-08-25T00:00:00Z")


def test_inventory_rejects_exact_item_state_drift() -> None:
    """A reopen between listing and exact read fails closed."""
    listed = _closed_item(18, labels=["type:docs"])
    exact = {**listed, "state": "open", "state_reason": "reopened"}
    responses = [
        type("R", (), {"returncode": 0, "stdout": json.dumps([listed]), "stderr": ""})(),
        type("R", (), {"returncode": 0, "stdout": json.dumps(exact), "stderr": ""})(),
    ]
    import pytest as _pytest

    with (
        patch("scripts.dev.terminal_label_reconcile.gh_api_get", side_effect=responses),
        _pytest.raises(ValueError, match="inconsistent terminal state"),
    ):
        run_terminal_inventory(repo="o/r", observed_at="2026-08-25T00:00:00Z")


def test_inventory_rejects_exact_item_label_drift() -> None:
    """A concurrent label change between listing and exact read fails closed."""
    listed = _closed_item(19, labels=["type:docs"])
    exact = {**listed, "labels": [{"name": "type:research"}]}
    responses = [
        type("R", (), {"returncode": 0, "stdout": json.dumps([listed]), "stderr": ""})(),
        type("R", (), {"returncode": 0, "stdout": json.dumps(exact), "stderr": ""})(),
    ]
    import pytest as _pytest

    with (
        patch("scripts.dev.terminal_label_reconcile.gh_api_get", side_effect=responses),
        _pytest.raises(ValueError, match="inconsistent labels"),
    ):
        run_terminal_inventory(repo="o/r", observed_at="2026-08-25T00:00:00Z")


def test_inventory_rejects_rate_limit_or_rest_error() -> None:
    """REST and rate-limit failures are surfaced as non-applicable inventory."""
    response = type("R", (), {"returncode": 1, "stdout": "", "stderr": "API rate limit exceeded"})()
    import pytest as _pytest

    with (
        patch("scripts.dev.terminal_label_reconcile.gh_api_get", return_value=response),
        _pytest.raises(ValueError, match="pagination read failed"),
    ):
        run_terminal_inventory(repo="o/r", observed_at="2026-08-25T00:00:00Z")


def test_inventory_complete_multi_page_fixture_and_aggregates() -> None:
    """A full page followed by a short page is complete and classified."""
    first_page = [_closed_item(number, labels=[]) for number in range(20, 120)]
    second_page = [_closed_item(120, labels=["agent"], is_pr=True, merged=True, reason=None)]
    report = run_terminal_inventory(
        fixture_pages=[first_page, second_page],
        observed_at="2026-08-25T00:00:00Z",
    )
    assert report["pagination"] == {
        "pages_read": 2,
        "item_count": 101,
        "complete": True,
        "bounded_by_max_items": False,
        "termination": "short_page",
    }
    assert report["source"]["api"] == "github-rest-v3"
    assert report["aggregate_counts_by"]["item_kind"] == {"issue": 100, "pull_request": 1}
    assert report["aggregate_counts_by"]["verdict"] == {
        "changes_required": 101,
    }
    assert report["aggregate_counts_by"]["label"] == {"agent": 1}


def test_inventory_truncated_fixture_fails_closed() -> None:
    """A full page budget without a terminating page is truncation."""
    pages = [
        [_closed_item(number, labels=[]) for number in range(130, 230)],
        [_closed_item(230, labels=[])],
    ]
    import pytest as _pytest

    with _pytest.raises(ValueError, match="pagination truncated"):
        run_terminal_inventory(
            fixture_pages=pages,
            max_pages=1,
            observed_at="2026-08-25T00:00:00Z",
        )


def test_collect_closed_items_rejects_invalid_limits() -> None:
    """Pagination limits are explicit positive integers."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="max_pages must be positive"):
        _collect_closed_items("o/r", max_pages=0, max_items=None, fixture=None, fixture_pages=None)
    with _pytest.raises(ValueError, match="max_items must be positive"):
        _collect_closed_items("o/r", max_pages=1, max_items=0, fixture=None, fixture_pages=None)


# --- issue #7896: terminal-label inventory mode ---


def _closed_item(
    number: int,
    *,
    labels: list[str],
    reason: str = "completed",
    is_pr: bool = False,
    merged: bool = False,
) -> dict:
    item: dict = {
        "number": number,
        "state": "closed",
        "state_reason": reason,
        "labels": [{"name": label} for label in labels],
        "html_url": f"https://github.com/o/r/issues/{number}",
    }
    if is_pr:
        item["pull_request"] = {"url": f"https://api.github.com/repos/o/r/pulls/{number}"}
        item["merged_at"] = "2026-08-01T00:00:00Z" if merged else None
    return item


def test_inventory_removes_agent_from_completed() -> None:
    """agent is an active dispatch label and is removed for verified terminal classes."""
    assert "agent" in ACTIVE_LABELS
    report = run_terminal_inventory(
        fixture=[_closed_item(1, labels=["agent", "agent-ready", "type:docs"])],
        observed_at="2026-08-25T00:00:00Z",
    )
    assert report["ok"] is True
    row = report["items"][0]
    assert row["terminal_class"] == "completed"
    assert "agent" in row["remove"]
    assert "agent-ready" in row["remove"]
    assert "type:docs" in row["preserved"]
    assert report["mutation_authorized"] is False


def test_inventory_reopened_and_unverified_retain_active_labels() -> None:
    """Reopened and terminal_unverified items receive no false terminal plan."""
    reopened = _closed_item(2, labels=["state:running", "state:done"], reason="reopened")
    reopened["state"] = "open"
    unverified = _closed_item(3, labels=["agent-ready"], reason=None)
    unverified["state_reason"] = None
    report = run_terminal_inventory(
        fixture=[reopened, unverified],
        observed_at="2026-08-25T00:00:00Z",
    )
    rows = {row["number"]: row for row in report["items"]}
    assert rows[2]["terminal_class"] == "reopened"
    assert "state:running" in rows[2]["preserved"]
    assert rows[3]["terminal_class"] == "terminal_unverified"
    assert rows[3]["remove"] == []


def test_inventory_not_planned_and_duplicate() -> None:
    """not_planned and duplicate closures produce the documented plans."""
    report = run_terminal_inventory(
        fixture=[
            _closed_item(4, labels=["state:ready", "blocked:some-dep"], reason="not_planned"),
            _closed_item(5, labels=["decision-required", "state:review"], reason="duplicate"),
        ],
        observed_at="2026-08-25T00:00:00Z",
    )
    rows = {row["number"]: row for row in report["items"]}
    assert rows[4]["terminal_class"] == "not_planned"
    assert "state:ready" in rows[4]["remove"]
    assert rows[5]["terminal_class"] == "duplicate"
    assert "decision-required" in rows[5]["remove"]


def test_inventory_merged_and_closed_unmerged_prs() -> None:
    """PR kinds derive from pull_request and merged_at fields."""
    report = run_terminal_inventory(
        fixture=[
            _closed_item(6, labels=["merge-ready"], is_pr=True, merged=True, reason=None),
            _closed_item(7, labels=["needs-review"], is_pr=True, merged=False, reason=None),
        ],
        observed_at="2026-08-25T00:00:00Z",
    )
    rows = {row["number"]: row for row in report["items"]}
    assert rows[6]["kind"] == "pull_request"
    assert rows[6]["terminal_class"] == "pr_merged"
    assert "merge-ready" in rows[6]["remove"]
    assert rows[7]["terminal_class"] == "pr_closed_unmerged"


def test_inventory_preserves_unknown_and_provenance_labels() -> None:
    """Unknown/manual/provenance labels remain preserved by default."""
    report = run_terminal_inventory(
        fixture=[
            _closed_item(
                8,
                labels=[
                    "state:ready",
                    "evidence:smoke",
                    "priority:4",
                    "artifact:durable-required",
                    "custom-manual-label",
                ],
            )
        ],
        observed_at="2026-08-25T00:00:00Z",
    )
    row = report["items"][0]
    for preserved in (
        "evidence:smoke",
        "priority:4",
        "artifact:durable-required",
        "custom-manual-label",
    ):
        assert preserved in row["preserved"]


def test_inventory_duplicate_numbers_fail_closed() -> None:
    """Duplicate item numbers across pages make the report non-applicable."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="duplicate item numbers"):
        run_terminal_inventory(
            fixture=[
                _closed_item(9, labels=["type:docs"]),
                _closed_item(9, labels=["type:docs"]),
            ],
            observed_at="2026-08-25T00:00:00Z",
        )


def test_inventory_ambiguous_state_qualifier_fails_closed() -> None:
    """An unclassified state:* label fails closed instead of wildcard removal."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="ambiguous"):
        run_terminal_inventory(
            fixture=[_closed_item(10, labels=["state:mystery-new-state"])],
            observed_at="2026-08-25T00:00:00Z",
        )


def test_inventory_unknown_terminal_class_fails_closed() -> None:
    """A missing/unknown state reason cannot produce a trustworthy plan."""
    import pytest as _pytest

    item = _closed_item(11, labels=["type:docs"], reason="some_unknown_reason")
    with _pytest.raises(ValueError, match="no trustworthy terminal class"):
        run_terminal_inventory(fixture=[item], observed_at="2026-08-25T00:00:00Z")


def test_inventory_repeated_runs_are_byte_stable() -> None:
    """Repeated runs over a fixed fixture are byte-stable."""
    fixture = [
        _closed_item(12, labels=["agent", "state:ready", "type:docs"]),
        _closed_item(13, labels=["needs-review"], is_pr=True, merged=True, reason=None),
    ]
    first = run_terminal_inventory(fixture=fixture, observed_at="2026-08-25T00:00:00Z")
    second = run_terminal_inventory(fixture=fixture, observed_at="2026-08-25T00:00:00Z")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_inventory_schema_and_aggregate_counts() -> None:
    """The inventory report carries schema, pagination, and aggregate counts."""
    report = run_terminal_inventory(
        fixture=[
            _closed_item(14, labels=["agent", "state:ready", "type:docs"]),
            _closed_item(15, labels=["merge-ready"], is_pr=True, merged=True, reason=None),
        ],
        observed_at="2026-08-25T00:00:00Z",
    )
    assert report["schema"] == INVENTORY_SCHEMA
    assert report["pagination"]["item_count"] == 2
    assert report["aggregate_counts"]["completed"] == 1
    assert report["aggregate_counts"]["pr_merged"] == 1
    assert report["aggregate_counts"]["agent"] == 1
    assert report["aggregate_counts"]["merge-ready"] == 1


def test_inventory_zero_mutation_proof() -> None:
    """The inventory command never calls label mutation helpers."""
    fixture = [_closed_item(16, labels=["agent", "type:docs"])]
    with (
        patch("scripts.dev.terminal_label_reconcile.add_label") as mock_add,
        patch("scripts.dev.terminal_label_reconcile.remove_label") as mock_remove,
    ):
        report = run_terminal_inventory(fixture=fixture, observed_at="2026-08-25T00:00:00Z")
    assert report["ok"] is True
    mock_add.assert_not_called()
    mock_remove.assert_not_called()


def test_state_qualifier_classification_is_explicit() -> None:
    """Every known state:* label is explicitly classified or would fail closed."""
    for label, classification in STATE_QUALIFIER_CLASSIFICATION.items():
        assert label.startswith("state:")
        assert classification in {"active", "terminal", "historical"}
    assert STATE_QUALIFIER_CLASSIFICATION["state:done"] == "terminal"
    assert STATE_QUALIFIER_CLASSIFICATION["state:hold"] == "historical"


def test_terminal_class_from_state() -> None:
    """REST state/reason/merged fields map to the documented terminal classes."""
    assert (
        _terminal_class_from_state({"state": "closed", "state_reason": "completed"}) == "completed"
    )
    assert (
        _terminal_class_from_state({"state": "closed", "state_reason": "not_planned"})
        == "not_planned"
    )
    assert (
        _terminal_class_from_state({"state": "closed", "state_reason": "duplicate"}) == "duplicate"
    )
    assert _terminal_class_from_state({"state": "open", "state_reason": None}) == "reopened"
    pr = {"state": "closed", "state_reason": None, "pull_request": {}, "merged_at": "x"}
    assert _terminal_class_from_state(pr) == "pr_merged"


def test_apply_merged_pr_removes_merge_ready_and_records_final_labels() -> None:
    """A merged PR clears merge-ready and reports state:done in final labels."""
    before = _pr_state("merge-ready", "review-bot-auto")
    after_removal = _pr_state("review-bot-auto")
    final = _pr_state("review-bot-auto", "state:done")
    with (
        patch(
            "scripts.dev.terminal_label_reconcile.fetch_item_state",
            side_effect=[before, before, after_removal, final],
        ),
        patch(
            "scripts.dev.terminal_label_reconcile.remove_terminal_pr_label",
            return_value=_terminal_remove_ok("merge-ready"),
        ) as mock_terminal_remove,
        patch("scripts.dev.terminal_label_reconcile.remove_label") as mock_issue_remove,
        patch(
            "scripts.dev.terminal_label_reconcile.add_label",
            return_value={"status": "ok"},
        ),
    ):
        report = reconcile_item(9356, "pr_merged", repo="o/r", apply=True)

    assert report["ok"] is True, report
    mock_terminal_remove.assert_called_once_with(9356, "merge-ready", repo="o/r")
    mock_issue_remove.assert_not_called()
    assert report["final_labels"] == ["review-bot-auto", "state:done"]
    assert report["applied_changes"]["remove"] == [
        {
            "label": "merge-ready",
            "skipped": False,
            "status": "ok",
            "target": "pr",
            "operation": "terminal_label_remove",
            "expected_head_sha": "a" * 40,
            "expected_base_sha": "b" * 40,
            "observed_state": "CLOSED",
            "observed_head_sha": "a" * 40,
            "observed_base_sha": "b" * 40,
            "merged_at": "2026-09-15T00:00:00Z",
        }
    ]


def test_apply_closed_unmerged_pr_routes_all_active_removals_to_terminal_guard() -> None:
    """Closed-unmerged PR active labels share the terminal PR guard."""
    before = _pr_state("merge-ready", "needs-review", "state:running", "review-bot-auto")
    after_removals = _pr_state("review-bot-auto")
    final = _pr_state("review-bot-auto", "state:done")
    with (
        patch(
            "scripts.dev.terminal_label_reconcile.fetch_item_state",
            side_effect=[before, before, before, before, after_removals, final],
        ),
        patch(
            "scripts.dev.terminal_label_reconcile.remove_terminal_pr_label",
            side_effect=[
                _terminal_remove_ok("merge-ready"),
                _terminal_remove_ok("needs-review"),
                _terminal_remove_ok("state:running"),
            ],
        ) as mock_terminal_remove,
        patch("scripts.dev.terminal_label_reconcile.remove_label") as mock_issue_remove,
        patch(
            "scripts.dev.terminal_label_reconcile.add_label",
            return_value={"status": "ok"},
        ),
    ):
        report = reconcile_item(9356, "pr_closed_unmerged", repo="o/r", apply=True)

    assert report["ok"] is True, report
    assert [call.args for call in mock_terminal_remove.call_args_list] == [
        (9356, "merge-ready"),
        (9356, "needs-review"),
        (9356, "state:running"),
    ]
    mock_issue_remove.assert_not_called()
    assert report["final_labels"] == ["review-bot-auto", "state:done"]


def test_apply_reopened_pr_does_not_remove_active_labels() -> None:
    """A reopened PR is rejected before any terminal label DELETE is attempted."""
    reopened = _pr_state("merge-ready", "state:done", state="open", merged_at=None)
    with (
        patch("scripts.dev.terminal_label_reconcile.fetch_item_state", return_value=reopened),
        patch("scripts.dev.terminal_label_reconcile.remove_terminal_pr_label") as mock_terminal,
        patch("scripts.dev.terminal_label_reconcile.remove_label") as mock_issue_remove,
        patch("scripts.dev.terminal_label_reconcile.add_label") as mock_add,
    ):
        report = reconcile_item(9356, "pr_merged", repo="o/r", apply=True)

    assert report["ok"] is False
    assert "reopened" in report["error"]
    assert report["failures"] == [
        {
            "label": "merge-ready",
            "status": "review_skipped_stale_state",
            "reason": "pr_not_terminal",
            "error": "item reopened (state=open); plan aborted",
        }
    ]
    mock_terminal.assert_not_called()
    mock_issue_remove.assert_not_called()
    mock_add.assert_not_called()


def test_apply_records_terminal_guard_failure_and_truthful_final_labels() -> None:
    """A failed PR CAS leaves labels observable and never reports success."""
    before = _pr_state("merge-ready")
    final = _pr_state("merge-ready", "state:done")
    stale = {
        "status": "review_skipped_stale_state",
        "reason": "head_sha_changed",
        "error": "head moved before terminal label removal",
        "target": "pr",
        "operation": "terminal_label_remove",
        "expected_head_sha": "a" * 40,
        "expected_base_sha": "b" * 40,
        "observed_state": "CLOSED",
        "observed_head_sha": "c" * 40,
        "observed_base_sha": "b" * 40,
        "merged_at": "2026-09-15T00:00:00Z",
    }
    with (
        patch(
            "scripts.dev.terminal_label_reconcile.fetch_item_state",
            side_effect=[before, before, before, final],
        ),
        patch(
            "scripts.dev.terminal_label_reconcile.remove_terminal_pr_label",
            return_value=stale,
        ),
        patch(
            "scripts.dev.terminal_label_reconcile.add_label",
            return_value={"status": "ok"},
        ),
    ):
        report = reconcile_item(9356, "pr_merged", repo="o/r", apply=True)

    assert report["ok"] is False
    assert report["final_labels"] == ["merge-ready", "state:done"]
    failure = report["applied_changes"]["failures"][0]
    assert failure == {
        "label": "merge-ready",
        "error": "head moved before terminal label removal",
        "status": "review_skipped_stale_state",
        "reason": "head_sha_changed",
        "target": "pr",
        "operation": "terminal_label_remove",
        "expected_head_sha": "a" * 40,
        "expected_base_sha": "b" * 40,
        "observed_state": "CLOSED",
        "observed_head_sha": "c" * 40,
        "observed_base_sha": "b" * 40,
        "merged_at": "2026-09-15T00:00:00Z",
    }


def test_apply_final_label_readback_failure_is_not_success() -> None:
    """A missing final readback makes the reconciliation receipt fail closed."""
    before = _pr_state("review-bot-auto")
    final_error = {"ok": False, "error": "labels read failed"}
    with (
        patch(
            "scripts.dev.terminal_label_reconcile.fetch_item_state",
            side_effect=[before, before, final_error],
        ),
        patch("scripts.dev.terminal_label_reconcile.add_label", return_value={"status": "ok"}),
    ):
        report = reconcile_item(9356, "pr_closed_unmerged", repo="o/r", apply=True)

    assert report["ok"] is False
    assert report["final_labels"] is None
    assert report["failures"][-1] == {
        "label": "__final_state__",
        "stage": "final_readback",
        "error": "labels read failed",
    }


def test_apply_issue_removals_keep_legacy_issue_helper() -> None:
    """Issue terminal rows keep the existing unguarded issue-target path."""
    before = _pr_state(
        "needs-review",
        "state:ready",
        is_pull_request=False,
        merged_at=None,
    )
    after_removal = _pr_state(is_pull_request=False, merged_at=None)
    final = _pr_state("state:done", is_pull_request=False, merged_at=None)
    with (
        patch(
            "scripts.dev.terminal_label_reconcile.fetch_item_state",
            side_effect=[before, before, before, after_removal, final],
        ),
        patch(
            "scripts.dev.terminal_label_reconcile.remove_label",
            return_value={"status": "ok"},
        ) as mock_issue_remove,
        patch("scripts.dev.terminal_label_reconcile.remove_terminal_pr_label") as mock_terminal,
        patch(
            "scripts.dev.terminal_label_reconcile.add_label",
            return_value={"status": "ok"},
        ),
    ):
        report = reconcile_item(9356, "completed", repo="o/r", apply=True)

    assert report["ok"] is True, report
    assert [call.args for call in mock_issue_remove.call_args_list] == [
        (9356, "needs-review"),
        (9356, "state:ready"),
    ]
    mock_terminal.assert_not_called()
    assert report["final_labels"] == ["state:done"]


def test_apply_is_idempotent_when_terminal_pr_labels_are_already_reconciled() -> None:
    """A repeated terminal reconciliation performs no label mutation when already clean."""
    clean = _pr_state("review-bot-auto", "state:done")
    with (
        patch(
            "scripts.dev.terminal_label_reconcile.fetch_item_state",
            side_effect=[clean, clean, clean],
        ),
        patch("scripts.dev.terminal_label_reconcile.remove_terminal_pr_label") as mock_terminal,
        patch("scripts.dev.terminal_label_reconcile.remove_label") as mock_issue_remove,
        patch("scripts.dev.terminal_label_reconcile.add_label") as mock_add,
    ):
        report = reconcile_item(9356, "pr_merged", repo="o/r", apply=True)

    assert report["ok"] is True, report
    assert report["final_labels"] == ["review-bot-auto", "state:done"]
    mock_terminal.assert_not_called()
    mock_issue_remove.assert_not_called()
    mock_add.assert_not_called()
