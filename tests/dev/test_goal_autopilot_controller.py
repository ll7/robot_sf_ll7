"""Regression tests for parent-only goal-autopilot arbitration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.dev import goal_autopilot_controller as controller

ORIGIN = "a" * 40
FRESHNESS = {
    "issue_state_digest": "b" * 64,
    "claim_state_digest": "c" * 64,
    "pr_head_digest": "d" * 64,
    "preparation_audit_digest": "e" * 64,
    "discovery_relevant_paths_digest": "f" * 64,
}


def _snapshot() -> dict:
    """Return a complete, terminal-capable controller evidence fixture."""
    return {
        "origin_main_sha": ORIGIN,
        "freshness": dict(FRESHNESS),
        "implementation": {
            "candidate_scope": "state:ready",
            "queue_completeness": "complete",
            "zero_work_authoritative": True,
            "claimable_count": 0,
            "admission_reason_histogram": {},
        },
        "pull_requests": {
            "open_count": 0,
            "recoverable_active_count": 0,
            "review_eligible_count": 0,
            "merge_ready_count": 0,
        },
        "preparation": {
            "audit_digest": "e" * 64,
            "promotable_count": 0,
            "formalizable_count": 0,
            "blocker_reconciliation_count": 0,
            "blocker_reconciliation_complete": True,
        },
        "discovery": {
            "lane": "documentation_and_ci_contract_drift",
            "relevant_head_sha": ORIGIN,
            "status": "saturated",
            "created_issue_numbers": [],
            "readiness_outcomes": [],
            "readiness_outcomes_complete": True,
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("merge_ready_count", 1, "merge"),
        ("review_eligible_count", 1, "review"),
        ("recoverable_active_count", 1, "recover_pr"),
    ],
)
def test_controller_routes_pr_work_before_zero_work(field: str, value: int, expected: str) -> None:
    """Merge, review, and active recovery all beat an empty issue queue."""
    snapshot = _snapshot()
    snapshot["pull_requests"][field] = value
    snapshot["pull_requests"]["open_count"] = 1
    result = controller.arbitrate_controller(snapshot)

    assert result["global_zero_work"] is False
    assert result["next_action"] == expected
    assert result["stop_reason"] is None


def test_controller_routes_claimable_issue_before_zero_work() -> None:
    """A claimable issue remains implementation work even with no PR work."""
    snapshot = _snapshot()
    snapshot["implementation"]["claimable_count"] = 1
    result = controller.arbitrate_controller(snapshot)

    assert result["next_action"] == "implement"
    assert result["global_zero_work"] is False


@pytest.mark.parametrize(
    ("field", "expected"),
    [("promotable_count", "gate_readiness"), ("formalizable_count", "formalize_issue")],
)
def test_controller_routes_preparation_work_before_discovery(field: str, expected: str) -> None:
    """Readiness replenishment and formalization are controller work."""
    snapshot = _snapshot()
    snapshot["preparation"][field] = 1
    result = controller.arbitrate_controller(snapshot)

    assert result["next_action"] == expected
    assert result["global_zero_work"] is False


def test_ungated_discovery_cannot_be_saturated() -> None:
    """A discovery result without readiness outcomes must return to discovery."""
    snapshot = _snapshot()
    snapshot["discovery"]["readiness_outcomes_complete"] = False
    result = controller.arbitrate_controller(snapshot)

    assert result["next_action"] == "discover"
    assert result["global_zero_work"] is False
    assert "discovery_readiness_outcomes_incomplete" in result["reasons"]


def test_incomplete_issue_queue_cannot_produce_global_zero_work() -> None:
    """A lane-local empty implementation queue is not a controller stop."""
    snapshot = _snapshot()
    snapshot["implementation"]["queue_completeness"] = "incomplete"
    snapshot["implementation"]["zero_work_authoritative"] = False
    result = controller.arbitrate_controller(snapshot)

    assert result["global_zero_work"] is False
    assert result["next_action"] == "refresh_issue_queue"
    assert result["stop_reason"] is None


def test_true_terminal_case_emits_head_bound_zero_work_proof() -> None:
    """Only complete empty lanes produce the controller-only terminal receipt."""
    result = controller.arbitrate_controller(_snapshot())

    assert result["global_zero_work"] is True
    assert result["next_action"] is None
    assert result["stop_reason"] == controller.GLOBAL_ZERO_WORK
    proof = result["zero_work_proof"]
    assert proof["schema"] == controller.ZERO_WORK_PROOF_SCHEMA
    assert proof["origin_main_sha"] == ORIGIN
    assert proof["implementation"]["claimable_count"] == 0
    assert proof["pull_requests"]["open_count"] == 0
    assert proof["preparation"]["blocker_reconciliation_count"] == 0
    assert proof["discovery"]["status"] == "saturated"


def test_stale_zero_work_proof_fails_closed_for_all_freshness_inputs() -> None:
    """Head, issue, claim, PR, audit, and discovery drift invalidate a receipt."""
    snapshot = _snapshot()
    proof = controller.arbitrate_controller(snapshot)["zero_work_proof"]
    assert (
        controller.validate_zero_work_proof(
            proof,
            origin_main_sha=ORIGIN,
            freshness=FRESHNESS,
            snapshot=snapshot,
        )["valid"]
        is True
    )

    for field in (
        "issue_state_digest",
        "claim_state_digest",
        "pr_head_digest",
        "preparation_audit_digest",
        "discovery_relevant_paths_digest",
    ):
        changed = deepcopy(FRESHNESS)
        changed[field] = "9" * 64
        validation = controller.validate_zero_work_proof(
            proof,
            origin_main_sha=ORIGIN,
            freshness=changed,
        )
        assert validation["valid"] is False

    changed_snapshot = deepcopy(snapshot)
    changed_snapshot["origin_main_sha"] = "9" * 40
    changed_snapshot["discovery"]["relevant_head_sha"] = "9" * 40
    stale_result = controller.arbitrate_controller(
        changed_snapshot,
        prior_zero_work_proof=proof,
    )
    assert stale_result["global_zero_work"] is False
    assert stale_result["next_action"] == "refresh_controller_evidence"
    assert any(reason.startswith("stale_zero_work_proof:") for reason in stale_result["reasons"])


def test_lane_result_rejects_global_terminal_state() -> None:
    """Delegated workers cannot manufacture the parent controller stop state."""
    with pytest.raises(ValueError, match="controller-only"):
        controller.lane_result("implementation", controller.GLOBAL_ZERO_WORK)

    result = controller.lane_result(
        "implementation",
        "implementation_queue_exhausted",
        evidence={"claimable_count": 0},
    )
    assert result["global_terminal"] is False
    assert result["status"] == "implementation_queue_exhausted"
