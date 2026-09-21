"""Tests for the lifecycle state reconciler (issue #9535).

Every case runs offline through injected evidence: no GitHub access, no label
writes, no claim reads. The reconciler must repair only evidence-provable
contradictions and retain everything ambiguous untouched.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.dev import lifecycle_state_reconcile as reconcile


def _raw_issue(
    *,
    number: int = 7601,
    state: str = "open",
    labels: list[str],
    assignees: list[str] | None = None,
    body: str = "## Objective\nFix the lifecycle drift.\n",
) -> dict[str, Any]:
    """Build one open-issue payload for planner tests."""
    return {
        "number": number,
        "title": "workflow: reconcile lifecycle state",
        "body": body,
        "state": state,
        "url": f"https://github.test/issues/{number}",
        "html_url": f"https://github.test/issues/{number}",
        "labels": [{"name": label} for label in labels],
        "assignees": [{"login": login} for login in (assignees or [])],
    }


def _claim(*, claimed: bool, ok: bool = True) -> dict[str, Any]:
    """Build one atomic-claim evidence payload."""
    return {"ok": ok, "claimed": claimed, "claim_ref": "agent-claims/issue-7601"}


def _live_issue(
    raw: dict[str, Any],
    *,
    labels: list[str] | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    """Build a live re-read payload mirroring the normalized reader shape."""
    payload = dict(raw)
    if labels is not None:
        payload["labels"] = [{"name": label} for label in labels]
    if body is not None:
        payload["body"] = body
    return payload


def _remover_factory(calls: list[tuple[int, str]]) -> Any:
    """Build an injectable label remover recording its invocations."""

    def _remove(number: int, label: str, **_: Any) -> dict[str, Any]:
        calls.append((number, label))
        return {"status": "ok"}

    return _remove


# Classification: repair codes.


def test_plan_removes_running_without_claim_or_pr() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:running", "type:workflow"]),
        claim=_claim(claimed=False),
        covering_prs=[],
    )

    assert row["reason_code"] == "stale_running_without_claim_or_pr"
    assert row["action"] == "remove_labels"
    assert row["target_labels"] == ["state:running"]


def test_plan_removes_ready_under_evidenced_running() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "state:running"]),
        claim=_claim(claimed=True),
        covering_prs=[],
    )

    assert row["reason_code"] == "evidenced_running_ready_stale"
    assert row["action"] == "remove_labels"
    assert row["target_labels"] == ["state:ready"]


def test_plan_removes_ready_when_open_pr_covers_running() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "state:running"]),
        claim=_claim(claimed=False),
        covering_prs=[7710],
    )

    assert row["reason_code"] == "evidenced_running_ready_stale"
    assert row["target_labels"] == ["state:ready"]


# Classification: report-only codes.


def test_plan_reports_ready_and_parked() -> None:
    row = reconcile.plan_row(_raw_issue(labels=["state:ready", "state:parked"]))

    assert row["reason_code"] == "ready_and_parked"
    assert row["action"] == "report_only"
    assert row["target_labels"] == []


def test_plan_reports_ready_and_decision_required() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "decision-required", "type:workflow"])
    )

    assert row["reason_code"] == "ready_and_decision_required"
    assert row["action"] == "report_only"


def test_plan_reports_assigned_ready() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready"], assignees=["octocat"]),
    )

    assert row["reason_code"] == "ready_but_assigned"
    assert row["action"] == "report_only"


def test_plan_reports_ready_running_with_assignee() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:ready", "state:running"], assignees=["octocat"]),
        claim=_claim(claimed=False),
        covering_prs=[],
    )

    assert row["reason_code"] == "ready_but_assigned"
    assert row["action"] == "report_only"


def test_plan_reports_execution_state_conflict_with_blocked() -> None:
    row = reconcile.plan_row(_raw_issue(labels=["state:ready", "state:blocked"]))

    assert row["reason_code"] == "execution_state_conflict"
    assert row["action"] == "report_only"


def test_plan_reports_unknown_state_labels() -> None:
    row = reconcile.plan_row(_raw_issue(labels=["state:ready", "state:frobnicated"]))

    assert row["reason_code"] == "ambiguous_do_not_mutate"
    assert row["action"] == "report_only"


# Classification: consistent rows.


def test_plan_keeps_active_claim_untouched() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:running", "type:workflow"]),
        claim=_claim(claimed=True),
        covering_prs=[],
    )

    assert row["reason_code"] == "consistent"
    assert row["action"] == "no_action"


def test_plan_keeps_covering_pr_untouched() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:running"]),
        claim=_claim(claimed=False),
        covering_prs=[7710],
    )

    assert row["reason_code"] == "consistent"
    assert row["action"] == "no_action"


def test_plan_keeps_deliberate_states_untouched() -> None:
    for labels in (
        ["state:blocked"],
        ["state:hold"],
        ["state:blocked-external-input"],
        ["state:ready"],
        [],
    ):
        row = reconcile.plan_row(_raw_issue(labels=list(labels)))

        assert row["action"] == "no_action", labels
        assert row["reason_code"] == "consistent", labels


def test_repair_is_idempotent_after_label_removal() -> None:
    """Re-running the report after a successful repair plans no action."""
    row = reconcile.plan_row(
        _raw_issue(labels=["type:workflow"]),
        claim=_claim(claimed=False),
        covering_prs=[],
    )

    assert row["action"] == "no_action"


# Classification: unavailable evidence.


def test_plan_defers_when_claim_read_fails() -> None:
    row = reconcile.plan_row(
        _raw_issue(labels=["state:running"]),
        claim=_claim(claimed=False, ok=False),
        covering_prs=[],
    )

    assert row["reason_code"] == "ambiguous_do_not_mutate"
    assert row["action"] == "deferred"


def test_plan_defers_running_without_evidence() -> None:
    row = reconcile.plan_row(_raw_issue(labels=["state:running"]))

    assert row["action"] == "deferred"


# Apply path with drift checks.


def test_apply_removes_stale_running_with_matching_live_state() -> None:
    raw = _raw_issue(labels=["state:running", "type:workflow"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=False), covering_prs=[])
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(
        row,
        live_issue=_live_issue(raw),
        claim=_claim(claimed=False),
        covering_prs=[],
        label_remover=_remover_factory(calls),
    )

    assert result["applied"] is True
    assert result["applied_labels"] == ["state:running"]
    assert calls == [(7601, "state:running")]


def test_apply_defers_on_label_drift() -> None:
    """A concurrently resolved contradiction must not be re-mutated."""
    raw = _raw_issue(labels=["state:running", "type:workflow"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=False), covering_prs=[])
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(
        row,
        live_issue=_live_issue(raw, labels=["type:workflow"]),
        claim=_claim(claimed=False),
        covering_prs=[],
        label_remover=_remover_factory(calls),
    )

    assert result["action"] == "deferred"
    assert result["applied"] is False
    assert calls == []


def test_apply_defers_on_body_drift() -> None:
    raw = _raw_issue(labels=["state:running", "type:workflow"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=False), covering_prs=[])
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(
        row,
        live_issue=_live_issue(raw, body="## Objective\nEdited concurrently.\n"),
        claim=_claim(claimed=False),
        covering_prs=[],
        label_remover=_remover_factory(calls),
    )

    assert result["action"] == "deferred"
    assert calls == []


def test_apply_defers_when_claim_appears_before_write() -> None:
    """A claim acquired between plan and apply keeps the running label."""
    raw = _raw_issue(labels=["state:running", "type:workflow"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=False), covering_prs=[])
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(
        row,
        live_issue=_live_issue(raw),
        claim=_claim(claimed=True),
        covering_prs=[],
        label_remover=_remover_factory(calls),
    )

    assert result["action"] == "deferred"
    assert calls == []


def test_apply_proceeds_when_evidence_shifts_but_conclusion_matches() -> None:
    """A claim released while a covering PR appears still evidences running."""
    raw = _raw_issue(labels=["state:ready", "state:running"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=True), covering_prs=[])
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(
        row,
        live_issue=_live_issue(raw),
        claim=_claim(claimed=False),
        covering_prs=[7710],
        label_remover=_remover_factory(calls),
    )

    assert result["applied"] is True
    assert result["applied_labels"] == ["state:ready"]
    assert calls == [(7601, "state:ready")]


def test_apply_defers_when_live_issue_closes() -> None:
    raw = _raw_issue(labels=["state:running", "type:workflow"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=False), covering_prs=[])
    live = _live_issue(raw)
    live["state"] = "closed"
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(
        row,
        live_issue=live,
        claim=_claim(claimed=False),
        covering_prs=[],
        label_remover=_remover_factory(calls),
    )

    assert result["action"] == "deferred"
    assert calls == []


def test_apply_ignores_non_repair_rows() -> None:
    row = reconcile.plan_row(_raw_issue(labels=["state:ready", "state:parked"]))
    calls: list[tuple[int, str]] = []

    result = reconcile.apply_row(row, label_remover=_remover_factory(calls))

    assert result["action"] == "report_only"
    assert calls == []


def test_apply_records_failed_removal_without_mutation_claim() -> None:
    raw = _raw_issue(labels=["state:running", "type:workflow"])
    row = reconcile.plan_row(raw, claim=_claim(claimed=False), covering_prs=[])
    calls: list[tuple[int, str]] = []

    def _failing(number: int, label: str, **_: Any) -> dict[str, Any]:
        calls.append((number, label))
        return {"status": "error", "error": "boom"}

    result = reconcile.apply_row(
        row,
        live_issue=_live_issue(raw),
        claim=_claim(claimed=False),
        covering_prs=[],
        label_remover=_failing,
    )

    assert result["action"] == "deferred"
    assert result["applied"] is False
    assert result["applied_labels"] == []
    assert calls == [(7601, "state:running")]


# Summary accounting for the controller receipt.


def test_summary_counts_repaired_and_unresolved_drift() -> None:
    rows = [
        reconcile.plan_row(
            _raw_issue(number=1, labels=["state:running"]),
            claim=_claim(claimed=False),
            covering_prs=[],
        ),
        reconcile.plan_row(_raw_issue(number=2, labels=["state:ready", "state:parked"])),
        reconcile.plan_row(_raw_issue(number=3, labels=["state:ready"])),
    ]
    rows[0]["applied"] = True

    summary = reconcile._summarize(rows, mode="apply", origin_main_sha=None)

    assert summary["schema"] == "lifecycle_state_reconcile.v1"
    assert summary["actions"]["remove_labels"] == 1
    assert summary["actions"]["report_only"] == 1
    assert summary["actions"]["no_action"] == 1
    assert summary["repaired_count"] == 1
    assert summary["unresolved_drift_count"] == 1
    assert summary["reason_counts"]["ready_and_parked"] == 1


def test_summary_report_mode_counts_planned_repairs_as_unresolved() -> None:
    rows = [
        reconcile.plan_row(
            _raw_issue(number=1, labels=["state:running"]),
            claim=_claim(claimed=False),
            covering_prs=[],
        )
    ]

    summary = reconcile._summarize(rows, mode="report", origin_main_sha=None)

    assert summary["repaired_count"] == 0
    assert summary["unresolved_drift_count"] == 1


# Inventory failure boundary.


def test_collect_open_rows_fails_closed_on_truncated_inventory(monkeypatch: Any) -> None:
    """An exhausted page budget never yields a partial candidate set."""
    import json as json_module

    full_page = [{"number": index, "labels": [], "assignees": []} for index in range(100)]

    class _Result:
        returncode = 0
        stdout = json_module.dumps(full_page)
        stderr = ""

    monkeypatch.setattr(reconcile, "run_gh_api_or_raise", lambda path: _Result())
    monkeypatch.setattr(
        reconcile,
        "parse_json",
        lambda result, what: (json_module.loads(result.stdout), None),
    )

    with pytest.raises(reconcile.InventoryIncompleteError):
        reconcile.collect_open_rows("ll7/robot_sf_ll7", max_pages=2)


def test_main_reports_blocked_inventory_without_mutation(monkeypatch: Any, capsys: Any) -> None:
    def _blocked(*_: Any, **__: Any) -> Any:
        raise reconcile.InventoryIncompleteError(pages_read=1, max_pages=1)

    monkeypatch.setattr(reconcile, "collect_open_rows", _blocked)

    assert reconcile.main(["--repo", "ll7/robot_sf_ll7", "--skip-closed"]) == 2
    assert "blocked" in capsys.readouterr().err
