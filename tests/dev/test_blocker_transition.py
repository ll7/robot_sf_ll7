"""Tests for the fail-closed blocker transition planner and apply guard."""

from __future__ import annotations

import json
from subprocess import CompletedProcess
from typing import Any

import pytest

from scripts.dev.blocker_transition import TransitionError, apply_transition, plan_transition


def _issue(*labels: str, number: int = 7672) -> dict[str, Any]:
    return {
        "number": number,
        "title": "workflow: bounded transition",
        "state": "OPEN",
        "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "body": "## Objective\nPlan one transition.\n",
        "labels": list(labels),
        "updated_at": "2026-08-20T15:00:00Z",
    }


def test_ruling_without_child_is_explicit_and_does_not_infer_readiness() -> None:
    plan = plan_transition(
        _issue("state:blocked", "decision-required", "state:blocked-no-code-slice"),
        ruling={"valid": True, "token": "ruling-7652", "carrier": "issue-comment-1"},
    )

    assert plan["schema"] == "blocker_transition_plan.v1"
    assert plan["blocker_class"] == "ruled_pending_child"
    assert plan["next_permitted_state"] == "blocked"
    assert "missing_bounded_child" in plan["reason_codes"]
    assert plan["proposed_label_delta"] == {
        "add": ["dependency:has-blockers", "needs-triage", "parent", "ruled"],
        "remove": ["decision-required", "state:blocked-no-code-slice"],
    }
    assert plan["no_write"] is True
    assert plan["required_child_contract"]["status"] == "required"
    assert plan["required_child_contract"]["executable"] is False


def test_authorized_child_makes_no_code_slice_an_invalid_transition() -> None:
    plan = plan_transition(
        _issue("ruled", "state:blocked", "state:blocked-no-code-slice"),
        ruling={"valid": True, "token": "ruling-7652", "carrier": "comment/1"},
        children=[{"number": 7669, "state": "OPEN", "contract_ready": True}],
    )

    assert plan["blocker_class"] == "invalid_or_conflicting_state"
    assert "ruled_child_exists_with_no_code_slice" in plan["reason_codes"]
    assert plan["proposed_label_delta"]["remove"] == ["state:blocked-no-code-slice"]


def test_resolved_child_is_still_dependency_bound_until_rechecked() -> None:
    plan = plan_transition(
        _issue("state:blocked", "decision-required"),
        ruling={"status": "recorded", "ruling_token": "ruling-7652", "source_url": "comment/1"},
        children=[{"number": 7669, "state": "OPEN", "revision": "child-sha"}],
    )

    assert plan["blocker_class"] == "ruled_pending_child"
    assert plan["required_child_or_pr_links"]["children"] == [7669]
    assert plan["next_action"] == "execute the named bounded child, then re-evaluate the parent"


def test_dependency_link_alone_is_not_satisfaction() -> None:
    plan = plan_transition(_issue("state:blocked", "dependency:has-blockers"), dependencies=[])

    assert plan["blocker_class"] == "dependency_predicate"
    assert plan["reason_codes"] == ["dependency_observation_missing"]
    assert plan["next_permitted_state"] == "blocked"
    assert plan["proposed_label_delta"]["add"] == []


def test_satisfied_dependency_and_fresh_implementability_propose_ready() -> None:
    plan = plan_transition(
        _issue("state:blocked", "dependency:has-blockers"),
        dependencies=[{"number": 7619, "state": "MERGED", "revision": "merge-sha"}],
        implementability={"ready": True},
    )

    assert plan["blocker_class"] == "dependency_predicate"
    assert plan["source_observations"]["dependencies"] == ["satisfied"]
    assert plan["next_permitted_state"] == "ready"
    assert plan["proposed_label_delta"] == {
        "add": ["state:ready"],
        "remove": ["dependency:has-blockers", "state:blocked"],
    }


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (("state:ready", "state:blocked"), "invalid_or_conflicting_state"),
        (("state:running", "state:working"), "invalid_or_conflicting_state"),
        (("state:parked",), "parked_or_deferred"),
        (("state:blocked-external-input",), "external_input"),
        (("resource:slurm",), "compute_required"),
    ],
)
def test_blocker_classes_remain_distinct(labels: tuple[str, ...], expected: str) -> None:
    assert plan_transition(_issue(*labels))["blocker_class"] == expected


def test_stale_pr_routes_to_refresh_owner() -> None:
    plan = plan_transition(
        _issue("state:blocked"),
        affected_prs=[{"number": 7493, "stale_base": True, "head_sha": "head", "base_sha": "old"}],
    )

    assert plan["blocker_class"] == "stale_base_or_metadata"
    assert plan["required_child_or_pr_links"]["affected_prs"] == [7493]
    assert "refresh" in plan["next_action"]
    assert plan["required_child_or_pr_links"]["affected_pr_owners"] == [
        {"number": 7493, "owner": ""}
    ]


def test_manual_regression_packets_remain_bounded() -> None:
    prerequisite = plan_transition(
        _issue("state:blocked", "dependency:has-blockers", number=7650),
        dependencies=[{"number": 7619, "state": "MERGED", "revision": "merge-sha"}],
        implementability={"ready": True, "contract_digest": "contract-sha"},
    )
    assert prerequisite["next_permitted_state"] == "ready"
    assert prerequisite["proposed_label_delta"]["add"] == ["state:ready"]

    for number, child_number, pr_numbers in (
        (7652, 7669, ()),
        (7653, 7670, (7555, 7562)),
        (7654, 7671, (7556,)),
    ):
        plan = plan_transition(
            _issue(
                "decision-required", "state:blocked", "state:blocked-no-code-slice", number=number
            ),
            ruling={
                "valid": True,
                "token": f"ruling-{number}",
                "carrier": f"issue-comment-{number}",
            },
            children=[{"number": child_number, "state": "OPEN", "contract_ready": True}],
            affected_prs=[
                {
                    "number": pr_number,
                    "owner": f"issue-{child_number}",
                    "head_sha": f"head-{pr_number}",
                }
                for pr_number in pr_numbers
            ],
        )

        assert plan["next_permitted_state"] == "blocked"
        assert plan["required_child_or_pr_links"]["children"] == [child_number]
        assert plan["required_child_or_pr_links"]["affected_prs"] == list(pr_numbers)
        assert "state:blocked-no-code-slice" in plan["proposed_label_delta"]["remove"]


def test_stale_and_conflicting_rulings_fail_closed() -> None:
    stale = plan_transition(
        _issue("state:blocked"),
        ruling={"valid": True, "token": "ruling-1", "carrier": ""},
    )
    conflicting = plan_transition(
        _issue("state:blocked"),
        ruling={
            "valid": True,
            "tokens": ["ruling-1", "ruling-2"],
            "carrier": "issue-comment-1",
        },
    )

    assert stale["blocker_class"] == "invalid_or_conflicting_state"
    assert "ruling_carrier_missing" in stale["reason_codes"]
    assert conflicting["blocker_class"] == "invalid_or_conflicting_state"
    assert "conflicting_ruling_tokens" in conflicting["reason_codes"]


def test_parent_child_link_drift_is_not_repaired_by_inference() -> None:
    plan = plan_transition(
        _issue("state:blocked"),
        children=[{"number": 7669, "parent_number": 7000, "state": "OPEN"}],
    )

    assert plan["blocker_class"] == "invalid_or_conflicting_state"
    assert "parent_child_link_drift" in plan["reason_codes"]
    assert plan["no_write"] is True


def test_closed_items_are_delegated_to_terminal_label_reconciliation() -> None:
    issue = _issue("state:ready")
    issue["state"] = "CLOSED"
    plan = plan_transition(issue)

    assert plan["blocker_class"] == "none"
    assert plan["terminal_reconciliation"] == {
        "delegated": True,
        "owner": "#7651",
        "no_write": True,
    }
    assert plan["next_action"].startswith("delegate terminal-label reconciliation")


def test_plan_digest_is_stable_and_observation_time_is_not_an_input() -> None:
    first = plan_transition(_issue("state:blocked", "dependency:has-blockers"))
    second = plan_transition(_issue("state:blocked", "dependency:has-blockers"))

    assert first == second
    assert len(first["plan_digest"]) == 64


def test_apply_requires_explicit_digest_and_verifies_label_readback() -> None:
    issue = _issue("state:blocked", "dependency:has-blockers", number=42)
    plan = plan_transition(
        issue,
        dependencies=[{"number": 7619, "state": "MERGED"}],
        implementability={"ready": True},
        mode="apply",
        authorized=True,
    )
    live = {
        "number": 42,
        "state": "open",
        "body": issue["body"],
        "labels": [{"name": label} for label in plan["item"]["labels"]],
    }
    calls: list[tuple[str, str | None, object | None]] = []

    def runner(path: str, payload: object | None, method: str | None) -> CompletedProcess[str]:
        calls.append((path, method, payload))
        if method == "PUT":
            assert isinstance(payload, dict)
            live["labels"] = [{"name": label} for label in payload["labels"]]
            return CompletedProcess(["gh", "api"], 0, json.dumps(live["labels"]), "")
        return CompletedProcess(["gh", "api"], 0, json.dumps(live), "")

    result = apply_transition(
        plan,
        repo="owner/repo",
        expected_plan_digest=plan["plan_digest"],
        authorized=True,
        runner=runner,
        source_revalidator=lambda expected: expected,
    )

    assert result["status"] == "applied"
    assert result["readback"] is True
    assert [method for _, method, _ in calls] == [None, "PUT", None]
    assert "state:ready" in result["labels"]
    assert "state:blocked" not in result["labels"]


def test_apply_requires_source_revalidation_for_dependency_transition() -> None:
    issue = _issue("state:blocked", "dependency:has-blockers", number=42)
    plan = plan_transition(
        issue,
        dependencies=[{"number": 7619, "state": "MERGED"}],
        implementability={"ready": True},
        mode="apply",
        authorized=True,
    )

    with pytest.raises(TransitionError, match="source revalidation"):
        apply_transition(
            plan,
            repo="owner/repo",
            expected_plan_digest=plan["plan_digest"],
            authorized=True,
            runner=lambda _path, _payload, _method: CompletedProcess(
                ["gh", "api"],
                0,
                json.dumps(
                    {
                        "number": 42,
                        "state": "open",
                        "body": issue["body"],
                        "labels": [{"name": label} for label in plan["item"]["labels"]],
                    }
                ),
                "",
            ),
        )


def test_apply_aborts_on_body_drift() -> None:
    issue = _issue("state:blocked", number=42)
    plan = plan_transition(issue, mode="apply", authorized=True)

    def runner(path: str, payload: object | None, method: str | None) -> CompletedProcess[str]:
        del path, payload, method
        return CompletedProcess(
            ["gh", "api"],
            0,
            json.dumps(
                {
                    "number": 42,
                    "state": "open",
                    "body": "changed body",
                    "labels": [{"name": "state:blocked"}],
                }
            ),
            "",
        )

    with pytest.raises(TransitionError, match="body drifted"):
        apply_transition(
            plan,
            repo="owner/repo",
            expected_plan_digest=plan["plan_digest"],
            authorized=True,
            runner=runner,
        )


def test_apply_aborts_on_concurrent_label_drift() -> None:
    plan = plan_transition(_issue("state:blocked", number=42), mode="apply", authorized=True)

    def runner(path: str, payload: object | None, method: str | None) -> CompletedProcess[str]:
        del path, payload, method
        current = {
            "number": 42,
            "state": "open",
            "body": "body",
            "labels": [{"name": "state:blocked"}, {"name": "needs-triage"}],
        }
        return CompletedProcess(["gh", "api"], 0, json.dumps(current), "")

    with pytest.raises(TransitionError, match="labels drifted"):
        apply_transition(
            plan,
            repo="owner/repo",
            expected_plan_digest=plan["plan_digest"],
            authorized=True,
            runner=runner,
        )
