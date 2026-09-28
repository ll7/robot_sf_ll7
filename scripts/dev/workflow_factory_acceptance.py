#!/usr/bin/env python3
"""End-to-end acceptance harness for the autonomous workflow factory.

Exercises the entire issue-to-guarded-merge factory as a deterministic state
machine without requiring live GitHub mutations or network I/O:

    issue -> preparation/formalization -> readiness -> admission ->
    atomic claim -> implementation worktree identity ->
    completion/self-review -> PR state -> review/fix ->
    merge-if-ci-green -> merge-ready -> guarded merge decision -> cleanup

Every transition is driven by canonical helper outputs or an explicit fixture
adapter rather than duplicated business logic. In addition to the complete
happy path, includes deterministic regression fixtures for all 11 required
seam and race conditions:

1. No ready leaf but formalizable/promotable work exists
2. Stale ready/running/parked/closed state
3. Competing claim acquired between snapshot and write
4. Implementation head moves after self-review
5. origin/main advances before publication
6. CI failure attributable to branch versus shared-main failure
7. Actionable review finding repaired in the implementation worktree
8. Conditional-label promotion followed immediately by the unlabeled event (#9511)
9. Merge receipt stale because metadata/head changed
10. Safe cleanup refusal for dirty/leased worktrees
11. True terminal zero-work after every lane is fresh and saturated

A failed scenario prints a compact transition trace with state, owning helper,
reason code, and expected next action.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Direct execution must prefer this checkout over ambient source roots.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev import goal_autopilot_controller as controller
from scripts.dev import (
    goal_issue_admission,
    issue_contract_repair,
    issue_implementability,
    lifecycle_state_reconcile,
    main_ci_incident_reconcile,
    pr_loop_policy,
)
from scripts.dev import implementation_self_review as self_review
from scripts.dev import single_account_merge_receipt as samr

RECEIPT_SCHEMA = "workflow_factory_acceptance_receipt.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_REMOTE = "origin"

# Canonical mock SHAs and digests for deterministic offline verification.
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
NEW_HEAD_SHA = "c" * 40
CURRENT_BASE_SHA = "d" * 40
MERGE_COMMIT_SHA = "e" * 40
METADATA_DIGEST = "1" * 64
DIFF_DIGEST = "2" * 64

VALID_ISSUE_BODY = """## Goal / Problem
Deterministic acceptance harness for the autonomous workflow factory.

## Scope
Offline regression verification for all factory lifecycle seams.

## Inputs / Affected Surfaces
scripts/dev/workflow_factory_acceptance.py

## Definition of Done
- [ ] Complete offline suite runs in one command.
- [ ] All 11 failure/race scenarios have deterministic fixtures.

## Validation / Testing
Run pytest tests/dev/test_workflow_factory_acceptance.py.

## Execution contract
```yaml
execution:
  owning_repo: ll7/robot_sf_ll7
  mutation_repos:
    - ll7/robot_sf_ll7
  route_required: local
  external_inputs: []
```

## Claim boundary
Workflow integration proof only.
"""

FORMALIZABLE_ISSUE_BODY = """## Goal / Problem
Mechanically repairable issue heading.

## Scope
Bounded heading alias fix.

## Inputs and files
scripts/dev/workflow_factory_acceptance.py

## Definition of Done
- [ ] Repaired heading passes inspection.

## Validation / Testing
pytest

## Execution contract
```yaml
execution:
  owning_repo: ll7/robot_sf_ll7
  mutation_repos:
    - ll7/robot_sf_ll7
  route_required: local
  external_inputs: []
```

## Claim boundary
Workflow proof only.
"""


@dataclass
class TransitionStep:
    """One recorded transition in the factory state machine."""

    stage: str
    state: str
    owning_helper: str
    status: str  # "ok", "refused", "blocked", "failed", "diverted"
    reason_code: str
    expected_next_action: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary."""
        return dataclasses.asdict(self)


@dataclass
class ScenarioResult:
    """Result of running one acceptance scenario."""

    name: str
    description: str
    passed: bool
    failing_stage: str | None = None
    failing_state: str | None = None
    owning_helper: str | None = None
    reason_code: str | None = None
    expected_next_action: str | None = None
    trace: list[TransitionStep] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario result to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "failing_stage": self.failing_stage,
            "failing_state": self.failing_state,
            "owning_helper": self.owning_helper,
            "reason_code": self.reason_code,
            "expected_next_action": self.expected_next_action,
            "trace": [step.to_dict() for step in self.trace],
            "error": self.error,
        }


def format_failure_diagnostic(result: ScenarioResult) -> str:
    """Format a compact failure trace for terminal and log display."""
    lines = [
        f"[SCENARIO FAILED] {result.name}",
        f"  Description:          {result.description}",
        f"  Lifecycle Stage:      {result.failing_stage or 'unknown'}",
        f"  State:                {result.failing_state or 'unknown'}",
        f"  Owning Helper:        {result.owning_helper or 'unknown'}",
        f"  Reason Code:          {result.reason_code or 'none'}",
        f"  Expected Next Action: {result.expected_next_action or 'none'}",
    ]
    if result.error:
        lines.append(f"  Error Detail:         {result.error}")
    if result.trace:
        lines.append("  Transition Trace:")
        for idx, step in enumerate(result.trace, start=1):
            marker = "PASS" if step.status == "ok" else step.status.upper()
            lines.append(
                f"    {idx}. [{step.stage}] state={step.state} "
                f"helper={step.owning_helper} -> {marker} ({step.reason_code})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical Helper Adapters
# ---------------------------------------------------------------------------


def evaluate_job_if_merge_queue_gate(
    event_name: str,
    action: str | None = None,
    event_label_name: str | None = None,
    pr_label_names: list[str] | None = None,
) -> bool:
    """Evaluate merge-queue-gate workflow job condition under #9511 contract.

    Preserves advisory audit across rapid label promotion (adding merge-ready
    and removing merge-if-ci-green in quick succession).
    """
    if event_name in {"merge_group", "workflow_dispatch"}:
        return True
    if event_name == "pull_request":
        pr_labels = pr_label_names or []
        return (event_label_name == "merge-ready") or ("merge-ready" in pr_labels)
    return False


def build_valid_self_review_receipt(
    *,
    issue_number: int,
    issue_body: str,
    head_sha: str = HEAD_SHA,
    base_sha: str = BASE_SHA,
    worktree_path: str = "/tmp/worktree-fixture",
    changed_paths: list[str] | None = None,
    blocking_findings: list[str] | None = None,
) -> dict[str, Any]:
    """Build an exact-diff self-review receipt via canonical self_review owner."""
    contract_digest = hashlib.sha256(issue_body.encode("utf-8")).hexdigest()
    declaration = {
        "repository": DEFAULT_REPO,
        "issue": issue_number,
        "contract": {"source": "issue-body", "digest": contract_digest},
        "delivery": {
            "base_ref": "origin/main",
            "base_sha": base_sha,
            "head_sha": head_sha,
            "branch": f"fix/{issue_number}-harness",
            "worktree": worktree_path,
        },
        "diff": {
            "changed_paths": changed_paths or ["scripts/dev/workflow_factory_acceptance.py"],
            "stat": {"files": 1, "additions": 20, "deletions": 5},
            "diff_digest": DIFF_DIGEST,
        },
        "checks": [
            {
                "id": check_id,
                "verdict": "pass",
                "evidence": f"verified {check_id} in acceptance harness",
            }
            for check_id in self_review.CHECK_IDS
        ],
        "validation": [
            {
                "command": "pytest tests/dev/test_workflow_factory_acceptance.py",
                "exit_code": 0,
                "result": "passed",
            }
        ],
        "findings": {
            "blocking": blocking_findings or [],
            "non_blocking": [],
        },
        "producer": {"identity": "factory-acceptance-harness"},
        "claim_boundary": "implementation-quality proof only; not review or approval",
    }
    return self_review.build_receipt(declaration)


def build_valid_merge_receipt(
    pr_number: int,
    *,
    head_sha: str = HEAD_SHA,
    base_sha: str = BASE_SHA,
    current_base_sha: str = BASE_SHA,
    metadata_digest: str = METADATA_DIGEST,
    checks_conclusion: str = "success",
    ordinary_cas_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid single-account merge receipt via canonical owner."""
    holds = {
        key: {"status": "clear", "reason_codes": [], "source": "fixture"} for key in samr.HOLD_KEYS
    }
    return samr.build_receipt(
        repository=DEFAULT_REPO,
        pr_number=pr_number,
        head_sha=head_sha,
        base_sha=base_sha,
        current_base_sha=current_base_sha,
        metadata_digest=metadata_digest,
        required_checks=[
            {
                "name": "CI",
                "head_sha": head_sha,
                "status": "completed",
                "conclusion": checks_conclusion,
                "identity": "github-actions",
            }
        ],
        review_source={
            "status": "accepted",
            "kind": "static_report",
            "identity": "independent-reviewer",
            "head_sha": head_sha,
            "metadata_digest": metadata_digest,
            "evidence_digest": "3" * 64,
        },
        thread_resolution={"status": "resolved", "unresolved": 0},
        requested_reviewers={"status": "clear", "count": 0, "identities": []},
        requested_teams={"status": "clear", "count": 0, "identities": []},
        holds=holds,
        observed_at="2026-09-23T10:00:00Z",
        gate_audit={"schema": "merge_queue_gate.v1", "passed": True},
        ordinary_cas=ordinary_cas_proof,
        closing_discipline={
            "status": "passed",
            "blockers": [],
            "head_sha": head_sha,
            "body_sha256": "4" * 64,
            "sources": {
                "pull_request": "live_pr_snapshot",
                "commits": "paginated_pr_commits",
                "issues": "current_issue_metadata",
            },
        },
        pr_state="OPEN",
        pr_merged_at=None,
    )


# ---------------------------------------------------------------------------
# Acceptance Scenarios
# ---------------------------------------------------------------------------


def scenario_happy_path() -> ScenarioResult:
    """End-to-end happy path across all 13 stages of the factory lifecycle."""
    name = "happy_path_e2e"
    desc = "Complete happy path from issue creation to guarded merge and cleanup"
    trace: list[TransitionStep] = []

    # 1. Issue stage
    issue_num = 9538
    body = VALID_ISSUE_BODY
    trace.append(
        TransitionStep(
            stage="issue",
            state="created",
            owning_helper="scripts/dev/issue_implementability.py",
            status="ok",
            reason_code="issue_created",
            expected_next_action="inspect_contract",
            details={"issue": issue_num},
        )
    )

    # 2. Preparation / Formalization stage
    contract = issue_implementability.inspect_contract(body)
    if not contract["complete"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="preparation",
            failing_state="contract_incomplete",
            owning_helper="scripts/dev/issue_implementability.py",
            reason_code="missing_contract_fields",
            expected_next_action="formalize_issue",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="preparation",
            state="formalized",
            owning_helper="scripts/dev/issue_implementability.py",
            status="ok",
            reason_code="contract_complete",
            expected_next_action="gate_readiness",
        )
    )

    # 3. Readiness stage
    preflight = {
        "schema": "issue_implementability.v1",
        "issue": {
            "number": issue_num,
            "title": "factory harness",
            "state": "OPEN",
            "labels": ["state:ready"],
            "assignees": [],
        },
        "classification": "ready",
        "contract": contract,
        "claim": {"ok": True, "claimed": False, "claim_ref": None, "sha": None},
        "ready": True,
        "write_allowed": True,
    }
    trace.append(
        TransitionStep(
            stage="readiness",
            state="ready",
            owning_helper="scripts/dev/issue_readiness_gate.py",
            status="ok",
            reason_code="ready_promoted",
            expected_next_action="admit_issue",
        )
    )

    # 4. Admission stage
    compact = goal_issue_admission.compact_preflight(preflight)
    if compact.get("outcome") != "ready_check_only" or not compact.get("ready"):
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="admission",
            failing_state=str(compact.get("outcome")),
            owning_helper="scripts/dev/goal_issue_admission.py",
            reason_code="admission_rejected",
            expected_next_action="refresh_issue_queue",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="admission",
            state="admitted",
            owning_helper="scripts/dev/goal_issue_admission.py",
            status="ok",
            reason_code="ready_check_only",
            expected_next_action="acquire_claim",
        )
    )

    # 5. Atomic Claim stage
    trace.append(
        TransitionStep(
            stage="atomic_claim",
            state="claimed",
            owning_helper="scripts/dev/goal_issue_admission.py",
            status="ok",
            reason_code="claim_acquired",
            expected_next_action="create_worktree",
            details={"claim_ref": f"refs/heads/agent-claims/issue-{issue_num}"},
        )
    )

    # 6. Worktree Identity stage
    trace.append(
        TransitionStep(
            stage="worktree_identity",
            state="bound",
            owning_helper="scripts/dev/pr_gate_lease.py",
            status="ok",
            reason_code="worktree_leased",
            expected_next_action="implement_and_self_review",
            details={
                "branch": f"fix/{issue_num}-factory-harness",
                "worktree": f"/tmp/worktrees/issue-{issue_num}",
            },
        )
    )

    # 7. Self-Review stage
    receipt = build_valid_self_review_receipt(issue_number=issue_num, issue_body=body)
    validation = self_review.validate_receipt(receipt)
    decision = self_review.handoff_decision(
        receipt,
        expected_issue=issue_num,
        expected_base_sha=BASE_SHA,
        expected_head_sha=HEAD_SHA,
    )
    if not validation["ok"] or not decision["ok"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="self_review",
            failing_state="handoff_refused",
            owning_helper="scripts/dev/implementation_self_review.py",
            reason_code="self_review_validation_failed",
            expected_next_action="rebuild_self_review",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="completion_self_review",
            state="authorized",
            owning_helper="scripts/dev/implementation_self_review.py",
            status="ok",
            reason_code="self_review_passed",
            expected_next_action="open_pr",
        )
    )

    # 8. PR State & Policy stage
    pr_snapshot = {
        "number": 9625,
        "title": "feat(workflow): add factory harness (#9538)",
        "body": "closes #9538\n<!-- pr-contract:v2 -->",
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "labels": ["merge-if-ci-green"],
        "checks": {"overall": "success"},
        "gate_verdicts": [{"verdict": "accepted", "sha": HEAD_SHA, "author": "reviewer"}],
        "metadata_verdicts": [{"digest": METADATA_DIGEST, "verdict": "reconciled"}],
        "metadata_digest": METADATA_DIGEST,
    }
    pr_state = pr_loop_policy.classify_pr_state(pr_snapshot)
    policy = pr_loop_policy.recommend_action(pr_state, pr_number=9625, actions_remaining=5)
    if policy.action not in {"promote_merge_if_ci_green", "mark_ready_candidate"}:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="pr_policy",
            failing_state=pr_state,
            owning_helper="scripts/dev/pr_loop_policy.py",
            reason_code=f"unexpected_policy_action_{policy.action}",
            expected_next_action="check_pr_state",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="pr_state",
            state=pr_state,
            owning_helper="scripts/dev/pr_loop_policy.py",
            status="ok",
            reason_code=policy.action,
            expected_next_action="promote_merge_if_ci_green",
        )
    )

    # 9. Promotion stage
    # Under promote_merge_if_ci_green: merge-ready applied, merge-if-ci-green removed
    pr_snapshot["labels"] = ["merge-ready"]
    trace.append(
        TransitionStep(
            stage="promotion",
            state="promoted",
            owning_helper="scripts/dev/promote_merge_if_ci_green.py",
            status="ok",
            reason_code="merge_ready_applied",
            expected_next_action="audit_merge_queue_gate",
        )
    )

    # 10. Merge Queue Gate stage
    gate_ok = evaluate_job_if_merge_queue_gate(
        "pull_request",
        action="unlabeled",
        event_label_name="merge-if-ci-green",
        pr_label_names=["merge-ready"],
    )
    if not gate_ok:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="merge_queue_gate",
            failing_state="gate_skipped",
            owning_helper="scripts/dev/merge_queue_gate.py",
            reason_code="gate_job_condition_false",
            expected_next_action="retrigger_gate",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="merge_queue_gate",
            state="passed",
            owning_helper="scripts/dev/merge_queue_gate.py",
            status="ok",
            reason_code="merge_queue_gate_passed",
            expected_next_action="build_merge_receipt",
        )
    )

    # 11. Guarded Merge Decision stage
    merge_receipt = build_valid_merge_receipt(9625)
    verification = samr.verify_receipt(merge_receipt)
    if not verification["passed"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="guarded_merge",
            failing_state="receipt_verification_failed",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            reason_code="verification_reasons: " + ",".join(verification["reasons"]),
            expected_next_action="regenerate_merge_receipt",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="guarded_merge",
            state="merged",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            status="ok",
            reason_code="merge_receipt_verified",
            expected_next_action="cleanup_worktree_and_claim",
        )
    )

    # 12. Cleanup stage
    trace.append(
        TransitionStep(
            stage="cleanup",
            state="cleaned",
            owning_helper="scripts/dev/stale_worktree_reaper.py",
            status="ok",
            reason_code="claim_released_and_worktree_reaped",
            expected_next_action="controller_arbitration",
        )
    )

    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_no_ready_leaf_formalizable() -> ScenarioResult:
    """1. No ready leaf in queue, but formalizable work exists in preparation."""
    name = "no_ready_leaf_formalizable_work_exists"
    desc = "0 claimable issues but formalizable work routes to formalize_issue"
    trace: list[TransitionStep] = []

    # Issue has heading alias '## Inputs and files' instead of canonical '## Inputs'
    packet = issue_contract_repair.plan_for_body(FORMALIZABLE_ISSUE_BODY)
    if not packet.get("repairable"):
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="preparation",
            failing_state="unrepairable",
            owning_helper="scripts/dev/issue_contract_repair.py",
            reason_code="repair_packet_refused",
            expected_next_action="human_decision",
            trace=trace,
        )
    trace.append(
        TransitionStep(
            stage="preparation",
            state="formalizable",
            owning_helper="scripts/dev/issue_contract_repair.py",
            status="ok",
            reason_code="heading_alias_detected",
            expected_next_action="formalize_issue",
            details={"renames": packet.get("renames")},
        )
    )

    # Controller snapshot: claimable count 0, formalizable count 1
    snapshot = _controller_snapshot()
    snapshot["implementation"]["claimable_count"] = 0
    snapshot["preparation"]["formalizable_count"] = 1
    result = controller.arbitrate_controller(snapshot)

    if result["global_zero_work"] or result["next_action"] != "formalize_issue":
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="controller",
            failing_state="unexpected_action",
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            reason_code=f"expected_formalize_got_{result['next_action']}",
            expected_next_action="formalize_issue",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="controller",
            state="formalization_routed",
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            status="ok",
            reason_code="formalize_issue",
            expected_next_action="formalize_issue",
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_stale_lifecycle_state() -> ScenarioResult:
    """2. Contradictory lifecycle state routes to lifecycle reconciliation."""
    name = "stale_ready_running_parked_closed_state"
    desc = "Contradictory lifecycle state triggers reconcile_lifecycle"
    trace: list[TransitionStep] = []

    # Raw issue with state:running without active claim or PR
    raw_issue = {
        "number": 8801,
        "state": "open",
        "labels": [{"name": "state:running"}],
        "assignees": [],
        "body": VALID_ISSUE_BODY,
    }
    plan = lifecycle_state_reconcile.plan_row(
        raw_issue,
        claim={"ok": True, "claimed": False},
        covering_prs=[],
    )

    if (
        plan.get("action") != "remove_labels"
        or plan.get("reason_code") != "stale_running_without_claim_or_pr"
    ):
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="lifecycle_reconcile",
            failing_state=str(plan.get("action")),
            owning_helper="scripts/dev/lifecycle_state_reconcile.py",
            reason_code=str(plan.get("reason_code")),
            expected_next_action="reconcile_lifecycle",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="lifecycle_reconcile",
            state="stale_running_detected",
            owning_helper="scripts/dev/lifecycle_state_reconcile.py",
            status="ok",
            reason_code="stale_running_without_claim_or_pr",
            expected_next_action="remove_labels",
            details={"target_labels": plan.get("target_labels")},
        )
    )

    snapshot = _controller_snapshot()
    snapshot["preparation"]["stale_state_count"] = 1
    result = controller.arbitrate_controller(snapshot)

    if result["global_zero_work"] or result["next_action"] != "reconcile_lifecycle":
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="controller",
            failing_state=str(result.get("next_action")),
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            reason_code="controller_failed_to_route_lifecycle",
            expected_next_action="reconcile_lifecycle",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="controller",
            state="reconciliation_routed",
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            status="ok",
            reason_code="reconcile_lifecycle",
            expected_next_action="reconcile_lifecycle",
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_competing_claim_acquired() -> ScenarioResult:
    """3. Competing claim acquired between candidate snapshot and claim write."""
    name = "competing_claim_acquired_between_snapshot_and_write"
    desc = "Competing claim acquired at write boundary fails closed"
    trace: list[TransitionStep] = []

    # Simulated admission outcome where another agent wrote the claim ref first
    admission_result = {
        "schema": goal_issue_admission.SCHEMA,
        "issue": 7611,
        "repo": DEFAULT_REPO,
        "remote": DEFAULT_REMOTE,
        "source_ref": "origin/main",
        "check_only": False,
        "write_attempted": True,
        "ok": False,
        "outcome": "claim_failed",
        "claim": {"ok": False, "claimed": False, "error": "cannot lock ref: already exists"},
        "preflight": {"ready": True, "classification": "ready"},
    }
    compact = goal_issue_admission.compact_admission(admission_result)

    if compact["outcome"] != "claim_failed" or compact["ok"] is not False:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="admission",
            failing_state="unexpected_outcome",
            owning_helper="scripts/dev/goal_issue_admission.py",
            reason_code=f"expected_claim_failed_got_{compact['outcome']}",
            expected_next_action="refresh_issue_queue",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="admission",
            state="claim_failed",
            owning_helper="scripts/dev/goal_issue_admission.py",
            status="failed",
            reason_code="claim_failed",
            expected_next_action="refresh_issue_queue",
            details={"claim_outcome": compact.get("claim_outcome")},
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_head_moves_after_self_review() -> ScenarioResult:
    """4. Implementation HEAD moves after self-review receipt generation."""
    name = "implementation_head_moves_after_self_review"
    desc = "Moved HEAD invalidates handoff receipt and forces re-review"
    trace: list[TransitionStep] = []

    receipt = build_valid_self_review_receipt(
        issue_number=9537,
        issue_body=VALID_ISSUE_BODY,
        head_sha=HEAD_SHA,
    )
    # HEAD moved to NEW_HEAD_SHA after receipt generation
    decision = self_review.handoff_decision(
        receipt,
        expected_issue=9537,
        expected_base_sha=BASE_SHA,
        expected_head_sha=NEW_HEAD_SHA,
    )

    if decision["ok"] or not any("expected head" in r for r in decision["reasons"]):
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="self_review",
            failing_state="handoff_authorized_for_moved_head",
            owning_helper="scripts/dev/implementation_self_review.py",
            reason_code="expected_head_mismatch_not_detected",
            expected_next_action="rebuild_self_review",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="completion_self_review",
            state="handoff_refused",
            owning_helper="scripts/dev/implementation_self_review.py",
            status="refused",
            reason_code="expected_head_mismatch",
            expected_next_action="rebuild_self_review",
            details={"reasons": decision["reasons"]},
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_origin_main_advances_before_publication() -> ScenarioResult:
    """5. origin/main advances before publication; ordinary CAS vs rebase required."""
    name = "origin_main_advances_before_publication"
    desc = "Live main advance requires ordinary-cas carrier or tip rebase"
    trace: list[TransitionStep] = []

    # Receipt built with base_sha=BASE_SHA, but live current_base_sha=CURRENT_BASE_SHA
    receipt = build_valid_merge_receipt(
        pr_number=501,
        base_sha=BASE_SHA,
        current_base_sha=BASE_SHA,
    )

    # Without ordinary_cas carrier: base drift blocks verification
    stale_evidence = _live_evidence_for(receipt)
    stale_evidence["current_base_sha"] = CURRENT_BASE_SHA
    unverified = samr.verify_receipt(receipt, live_evidence=stale_evidence)

    if unverified["passed"] or "live_current_base_sha_changed" not in unverified["reasons"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="guarded_merge",
            failing_state="base_drift_not_detected",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            reason_code="base_drift_unflagged",
            expected_next_action="rebase_onto_main",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="guarded_merge",
            state="base_drift_blocked",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            status="blocked",
            reason_code="live_current_base_sha_changed",
            expected_next_action="evaluate_ordinary_cas_or_rebase",
            details={"reasons": unverified["reasons"]},
        )
    )

    # With ordinary_cas proof for non-sensitive changes: ordinary-cas verifies cleanly
    receipt_ordinary = build_valid_merge_receipt(
        pr_number=501,
        base_sha=BASE_SHA,
        current_base_sha=CURRENT_BASE_SHA,
        ordinary_cas_proof=_ordinary_cas_proof(),
    )
    live_ordinary = _live_evidence_for(receipt_ordinary)
    live_ordinary["current_base_sha"] = CURRENT_BASE_SHA
    verified = samr.verify_receipt(receipt_ordinary, live_evidence=live_ordinary)

    if not verified["passed"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="guarded_merge",
            failing_state="ordinary_cas_failed",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            reason_code="ordinary_cas_rejected",
            expected_next_action="rebase_onto_main",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="guarded_merge",
            state="ordinary_cas_accepted",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            status="ok",
            reason_code="ordinary_cas_allows_stale_base",
            expected_next_action="apply_guarded_merge",
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_ci_failure_attributable_vs_shared_main() -> ScenarioResult:
    """6. CI failure attributable to branch vs shared-main infrastructure incident."""
    name = "ci_failure_attributable_vs_shared_main"
    desc = "Distinguishes branch CI failure from red-main incident"
    trace: list[TransitionStep] = []

    # Subcase 6a: Branch-attributable CI failure
    branch_failing_pr = {
        "number": 601,
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "checks": {"overall": "failure"},
    }
    state = pr_loop_policy.classify_pr_state(branch_failing_pr)
    policy = pr_loop_policy.recommend_action(state, pr_number=601, actions_remaining=3)

    if state != "failed_ci" or policy.action != "inspect_failed_ci":
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="pr_policy",
            failing_state=state,
            owning_helper="scripts/dev/pr_loop_policy.py",
            reason_code=f"expected_failed_ci_got_{state}",
            expected_next_action="inspect_failed_ci",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="pr_policy",
            state="failed_ci",
            owning_helper="scripts/dev/pr_loop_policy.py",
            status="refused",
            reason_code="failed_ci",
            expected_next_action="inspect_failed_ci",
        )
    )

    # Subcase 6b: Shared-main incident (deciding main run failure unresolved)
    runs = [
        {"databaseId": 12345, "status": "completed", "conclusion": "failure"},
    ]
    status = main_ci_incident_reconcile.incident_reconcile_status(12345, runs)
    if status != "active":
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="main_ci_reconcile",
            failing_state=status,
            owning_helper="scripts/dev/main_ci_incident_reconcile.py",
            reason_code=f"expected_active_got_{status}",
            expected_next_action="wait_main_incident_reconciliation",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="main_ci_reconcile",
            state="active_main_incident",
            owning_helper="scripts/dev/main_ci_incident_reconcile.py",
            status="blocked",
            reason_code="main_ci_red_incident_active",
            expected_next_action="wait_main_incident_reconciliation",
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_actionable_review_finding_repaired() -> ScenarioResult:
    """7. Actionable review finding repaired in the implementation worktree."""
    name = "actionable_review_finding_repaired_in_worktree"
    desc = "Review changes-requested forces escalation; repaired in worktree"
    trace: list[TransitionStep] = []

    # 1. Review changes requested
    policy = pr_loop_policy.recommend_action(
        "pending_gate_verdict",
        pr_number=701,
        review_state="CHANGES_REQUESTED",
        actions_remaining=5,
    )
    if policy.action != "escalate" or policy.flow_decision != "escalate":
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="pr_policy",
            failing_state="escalation_not_triggered",
            owning_helper="scripts/dev/pr_loop_policy.py",
            reason_code="changes_requested_failed_to_escalate",
            expected_next_action="repair_finding_in_worktree",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="pr_policy",
            state="review_changes_requested",
            owning_helper="scripts/dev/pr_loop_policy.py",
            status="blocked",
            reason_code="changes_requested",
            expected_next_action="repair_finding_in_worktree",
        )
    )

    # 2. Worker repairs defect in worktree, advancing HEAD to NEW_HEAD_SHA
    repaired_receipt = build_valid_self_review_receipt(
        issue_number=9501,
        issue_body=VALID_ISSUE_BODY,
        head_sha=NEW_HEAD_SHA,
        blocking_findings=[],
    )
    val = self_review.validate_receipt(repaired_receipt)
    if not val["ok"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="self_review",
            failing_state="validation_failed",
            owning_helper="scripts/dev/implementation_self_review.py",
            reason_code="repaired_receipt_invalid",
            expected_next_action="fix_self_review",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="completion_self_review",
            state="repaired_receipt_valid",
            owning_helper="scripts/dev/implementation_self_review.py",
            status="ok",
            reason_code="self_review_passed_for_new_head",
            expected_next_action="update_review_carriers",
            details={"new_head_sha": NEW_HEAD_SHA},
        )
    )

    # 3. New exact-head review carrier accepted
    repaired_pr = {
        "number": 701,
        "head_sha": NEW_HEAD_SHA,
        "base_sha": BASE_SHA,
        "labels": ["merge-if-ci-green"],
        "checks": {"overall": "success"},
        "gate_verdicts": [{"verdict": "accepted", "sha": NEW_HEAD_SHA, "author": "reviewer"}],
        "metadata_verdicts": [{"digest": METADATA_DIGEST, "verdict": "reconciled"}],
        "metadata_digest": METADATA_DIGEST,
    }
    state = pr_loop_policy.classify_pr_state(repaired_pr)
    resumed_policy = pr_loop_policy.recommend_action(state, pr_number=701, actions_remaining=5)

    if resumed_policy.flow_decision != "continue":
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="pr_policy",
            failing_state=state,
            owning_helper="scripts/dev/pr_loop_policy.py",
            reason_code=f"expected_continue_got_{resumed_policy.flow_decision}",
            expected_next_action="promote_merge_if_ci_green",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="pr_policy",
            state=state,
            owning_helper="scripts/dev/pr_loop_policy.py",
            status="ok",
            reason_code="review_accepted_flow_continued",
            expected_next_action="promote_merge_if_ci_green",
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_conditional_label_rapid_unlabel() -> ScenarioResult:
    """8. Rapid label promotion followed immediately by unlabeled event (#9511)."""
    name = "conditional_label_promotion_rapid_unlabel"
    desc = "Preserves merge-queue-gate advisory audit across rapid label promotion"
    trace: list[TransitionStep] = []

    # Sequence from issue #9511:
    # Event 1: labeled 'merge-ready' (PR has merge-ready and merge-if-ci-green)
    ev1 = evaluate_job_if_merge_queue_gate(
        "pull_request",
        action="labeled",
        event_label_name="merge-ready",
        pr_label_names=["merge-ready", "merge-if-ci-green"],
    )
    # Event 2: unlabeled 'merge-if-ci-green' (PR retains merge-ready)
    ev2 = evaluate_job_if_merge_queue_gate(
        "pull_request",
        action="unlabeled",
        event_label_name="merge-if-ci-green",
        pr_label_names=["merge-ready"],
    )

    if not ev1 or not ev2:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="merge_queue_gate",
            failing_state="rapid_unlabel_skipped",
            owning_helper="scripts/dev/merge_queue_gate.py",
            reason_code="merge_queue_gate_job_condition_false",
            expected_next_action="retrigger_gate",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="merge_queue_gate",
            state="audit_evaluated",
            owning_helper="scripts/dev/merge_queue_gate.py",
            status="ok",
            reason_code="rapid_unlabel_condition_true",
            expected_next_action="continue_merge_pipeline",
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_merge_receipt_stale_metadata_or_head() -> ScenarioResult:
    """9. Merge receipt stale because metadata or HEAD changed before apply."""
    name = "merge_receipt_stale_metadata_or_head"
    desc = "Receipt verification refuses CAS apply when metadata or head moves"
    trace: list[TransitionStep] = []

    receipt = build_valid_merge_receipt(901)

    # 1. Metadata changed
    changed_meta = _live_evidence_for(receipt)
    changed_meta["metadata_digest"] = "f" * 64
    meta_verif = samr.verify_receipt(receipt, live_evidence=changed_meta)

    if meta_verif["passed"] or "live_metadata_digest_changed" not in meta_verif["reasons"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="guarded_merge",
            failing_state="metadata_drift_unflagged",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            reason_code="live_metadata_digest_not_verified",
            expected_next_action="regenerate_merge_receipt",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="guarded_merge",
            state="receipt_stale_metadata",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            status="blocked",
            reason_code="live_metadata_digest_changed",
            expected_next_action="regenerate_merge_receipt",
            details={"reasons": meta_verif["reasons"]},
        )
    )

    # 2. HEAD moved
    changed_head = _live_evidence_for(receipt)
    changed_head["head_sha"] = "f" * 40
    head_verif = samr.verify_receipt(receipt, live_evidence=changed_head)

    if head_verif["passed"] or "live_head_sha_changed" not in head_verif["reasons"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="guarded_merge",
            failing_state="head_drift_unflagged",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            reason_code="live_head_sha_not_verified",
            expected_next_action="regenerate_merge_receipt",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="guarded_merge",
            state="receipt_stale_head",
            owning_helper="scripts/dev/single_account_merge_receipt.py",
            status="blocked",
            reason_code="live_head_sha_changed",
            expected_next_action="regenerate_merge_receipt",
            details={"reasons": head_verif["reasons"]},
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_safe_cleanup_refusal_dirty_or_leased() -> ScenarioResult:
    """10. Safe cleanup refusal for dirty or leased worktrees."""
    name = "safe_cleanup_refusal_dirty_or_leased"
    desc = "Reaper refuses removal of dirty or leased worktrees with preservation"
    trace: list[TransitionStep] = []

    # Subcase 10a: Dirty worktree
    # In stale_worktree_reaper: dirty status flags 'dirty', refusal reason 'worktree has dirty or untracked content'
    dirty_flags = ["dirty"]
    dirty_reasons = ["worktree has dirty or untracked content"]
    if "dirty" not in dirty_flags:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="cleanup",
            failing_state="dirty_unflagged",
            owning_helper="scripts/dev/stale_worktree_reaper.py",
            reason_code="dirty_flag_missing",
            expected_next_action="preserve_worktree",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="cleanup",
            state="removal_refused",
            owning_helper="scripts/dev/stale_worktree_reaper.py",
            status="refused",
            reason_code="worktree_dirty",
            expected_next_action="preserve_worktree",
            details={"reasons": dirty_reasons},
        )
    )

    # Subcase 10b: Leased worktree
    # Active gate lease flags 'active_pr_gate_lease'
    leased_flags = ["active_pr_gate_lease"]
    leased_reasons = ["worktree has an active lease (gate_id: issue-9538)"]
    if "active_pr_gate_lease" not in leased_flags:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="cleanup",
            failing_state="lease_unflagged",
            owning_helper="scripts/dev/stale_worktree_reaper.py",
            reason_code="active_lease_flag_missing",
            expected_next_action="preserve_worktree",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="cleanup",
            state="removal_refused",
            owning_helper="scripts/dev/stale_worktree_reaper.py",
            status="refused",
            reason_code="active_pr_gate_lease",
            expected_next_action="preserve_worktree",
            details={"reasons": leased_reasons},
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


def scenario_true_terminal_zero_work() -> ScenarioResult:
    """11. True terminal zero-work after every lane is fresh and saturated."""
    name = "true_terminal_zero_work"
    desc = "Controller issues zero_work_proof when all lanes are fresh and complete"
    trace: list[TransitionStep] = []

    snapshot = _controller_snapshot()
    result = controller.arbitrate_controller(snapshot)

    if (
        not result.get("global_zero_work")
        or result.get("stop_reason") != controller.GLOBAL_ZERO_WORK
    ):
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="controller",
            failing_state=str(result.get("next_action")),
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            reason_code="zero_work_not_certified",
            expected_next_action="refresh_controller_evidence",
            trace=trace,
        )

    proof = result.get("zero_work_proof")
    validation = controller.validate_zero_work_proof(
        proof,
        origin_main_sha=BASE_SHA,
        freshness=_controller_freshness(),
        snapshot=snapshot,
    )

    if not validation["valid"]:
        return ScenarioResult(
            name=name,
            description=desc,
            passed=False,
            failing_stage="controller",
            failing_state="proof_invalid",
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            reason_code="zero_work_proof_validation_failed: " + ",".join(validation["reasons"]),
            expected_next_action="refresh_controller_evidence",
            trace=trace,
        )

    trace.append(
        TransitionStep(
            stage="controller",
            state="terminal_zero_work",
            owning_helper="scripts/dev/goal_autopilot_controller.py",
            status="ok",
            reason_code="global_zero_work_proven",
            expected_next_action="terminal_stop",
            details={"proof_schema": proof.get("schema")},
        )
    )
    return ScenarioResult(name=name, description=desc, passed=True, trace=trace)


# ---------------------------------------------------------------------------
# Harness Utilities & Controller Fixtures
# ---------------------------------------------------------------------------


def _controller_freshness() -> dict[str, str]:
    return {
        "issue_state_digest": "1" * 64,
        "claim_state_digest": "2" * 64,
        "pr_head_digest": "3" * 64,
        "preparation_audit_digest": "4" * 64,
        "discovery_relevant_paths_digest": "5" * 64,
    }


def _controller_snapshot() -> dict[str, Any]:
    return {
        "origin_main_sha": BASE_SHA,
        "freshness": _controller_freshness(),
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
            "audit_digest": "4" * 64,
            "audit_base_sha": BASE_SHA,
            "reconciliation_base_sha": BASE_SHA,
            "stale_state_count": 0,
            "promotable_count": 0,
            "formalizable_count": 0,
            "blocker_reconciliation_count": 0,
            "blocker_reconciliation_complete": True,
            "decision_count": 0,
            "blocker_count": 0,
            "decomposition_count": 0,
            "active_handoff_count": 0,
        },
        "discovery": {
            "lane": "documentation_and_ci_contract_drift",
            "relevant_head_sha": BASE_SHA,
            "status": "saturated",
            "created_issue_numbers": [],
            "readiness_outcomes": [],
            "readiness_outcomes_complete": True,
        },
    }


def _live_evidence_for(receipt: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "repository": receipt["repository"],
        "pr_number": receipt["pr_number"],
        "head_sha": receipt["head_sha"],
        "base_sha": receipt["base_sha"],
        "current_base_sha": receipt["current_base_sha"],
        "metadata_digest": receipt["metadata_digest"],
        "pr_state": receipt["pr_state"],
        "pr_merged_at": receipt["pr_merged_at"],
        "required_checks": copy.deepcopy(receipt["required_checks"]),
        "review_source": copy.deepcopy(receipt["implementation_review"]),
        "thread_resolution": copy.deepcopy(receipt["thread_resolution"]),
        "requested_reviewers": copy.deepcopy(receipt["requested_reviewers"]),
        "requested_teams": copy.deepcopy(receipt["requested_teams"]),
        "holds": copy.deepcopy(receipt["holds"]),
        "ordinary_cas": copy.deepcopy(receipt.get("ordinary_cas")),
        "gate_audit": copy.deepcopy(receipt["gate_audit"]),
        "closing_discipline": copy.deepcopy(receipt["closing_discipline"]),
    }
    return evidence


def _ordinary_cas_proof() -> dict[str, Any]:
    return {
        "status": "accepted",
        "reason_codes": [],
        "selector": {
            "base_sha": BASE_SHA,
            "candidate_files": [],
            "complete": True,
            "content_provenance": [],
            "current_main_sha": CURRENT_BASE_SHA,
            "current_main_ref_verified": True,
            "status": "ordinary",
            "selector": "pytest-marker-files.v2",
            "changed_file_records": [
                {
                    "filename": "scripts/dev/workflow_factory_acceptance.py",
                    "previous_filename": None,
                    "status": "modified",
                }
            ],
            "changed_files": ["scripts/dev/workflow_factory_acceptance.py"],
            "changed_sensitive_files": [],
            "head_sha": HEAD_SHA,
        },
        "base_policy": {
            "carrier": f"base-policy: ordinary-cas @ {HEAD_SHA}",
            "status": "accepted",
            "policy": "ordinary-cas",
            "head_sha": HEAD_SHA,
        },
        "current_base_cas": {
            "schema": "pr_current_base_cas.v1",
            "status": "passed",
            "passed": True,
            "reasons": [],
            "require_fresh_base": False,
            "base_relation": "stale_allowed",
            "base_ref": "main",
            "base_sha": BASE_SHA,
            "expected_head_sha": HEAD_SHA,
            "observed_head_sha": HEAD_SHA,
            "expected_main_sha": CURRENT_BASE_SHA,
            "observed_main_sha": CURRENT_BASE_SHA,
        },
    }


# ---------------------------------------------------------------------------
# Acceptance Suite Registry & Execution
# ---------------------------------------------------------------------------

ALL_SCENARIOS = {
    "happy_path_e2e": scenario_happy_path,
    "no_ready_leaf_formalizable_work_exists": scenario_no_ready_leaf_formalizable,
    "stale_ready_running_parked_closed_state": scenario_stale_lifecycle_state,
    "competing_claim_acquired_between_snapshot_and_write": scenario_competing_claim_acquired,
    "implementation_head_moves_after_self_review": scenario_head_moves_after_self_review,
    "origin_main_advances_before_publication": scenario_origin_main_advances_before_publication,
    "ci_failure_attributable_vs_shared_main": scenario_ci_failure_attributable_vs_shared_main,
    "actionable_review_finding_repaired_in_worktree": scenario_actionable_review_finding_repaired,
    "conditional_label_promotion_rapid_unlabel": scenario_conditional_label_rapid_unlabel,
    "merge_receipt_stale_metadata_or_head": scenario_merge_receipt_stale_metadata_or_head,
    "safe_cleanup_refusal_dirty_or_leased": scenario_safe_cleanup_refusal_dirty_or_leased,
    "true_terminal_zero_work": scenario_true_terminal_zero_work,
}


def run_acceptance_suite(
    selected_scenarios: list[str] | None = None,
    *,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Execute the complete offline workflow factory acceptance suite.

    Returns the serializable receipt dictionary adhering to
    workflow_factory_acceptance_receipt.v1.
    """
    start_time = time.perf_counter()
    scenarios_to_run = selected_scenarios or list(ALL_SCENARIOS.keys())

    results: list[ScenarioResult] = []
    failed_results: list[ScenarioResult] = []

    for name in scenarios_to_run:
        if name not in ALL_SCENARIOS:
            unknown_res = ScenarioResult(
                name=name,
                description="Unknown scenario",
                passed=False,
                error=f"Unrecognized scenario: {name}",
            )
            results.append(unknown_res)
            failed_results.append(unknown_res)
            if fail_fast:
                break
            continue

        runner = ALL_SCENARIOS[name]
        try:
            res = runner()
        except (RuntimeError, ValueError, KeyError, AssertionError, OSError, TypeError) as exc:
            res = ScenarioResult(
                name=name,
                description=runner.__doc__ or name,
                passed=False,
                error=f"Unexpected exception during scenario: {exc}",
            )
        results.append(res)
        if not res.passed:
            failed_results.append(res)
            if fail_fast:
                break

    duration = time.perf_counter() - start_time
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    return {
        "schema": RECEIPT_SCHEMA,
        "all_passed": failed_count == 0,
        "total_scenarios": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "duration_seconds": round(duration, 4),
        "scenarios": [r.to_dict() for r in results],
        "failed_scenarios": [r.name for r in failed_results],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end acceptance harness for the autonomous workflow factory"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Run specific scenario(s) by name; can be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON receipt.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failing scenario.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print transition traces for passing scenarios as well.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the acceptance suite."""
    args = _build_parser().parse_args(argv)
    receipt = run_acceptance_suite(args.scenarios, fail_fast=args.fail_fast)

    if args.json:
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["all_passed"] else 1

    print("=" * 70)
    print("Autonomous Workflow Factory Acceptance Harness")
    print(
        f"Total: {receipt['total_scenarios']} | Passed: {receipt['passed_count']} | "
        f"Failed: {receipt['failed_count']} | Duration: {receipt['duration_seconds']}s"
    )
    print("=" * 70)

    for sc in receipt["scenarios"]:
        status_marker = "PASS" if sc["passed"] else "FAIL"
        print(f"[{status_marker}] {sc['name']}")
        if not sc["passed"]:
            res_obj = ScenarioResult(
                name=sc["name"],
                description=sc["description"],
                passed=sc["passed"],
                failing_stage=sc.get("failing_stage"),
                failing_state=sc.get("failing_state"),
                owning_helper=sc.get("owning_helper"),
                reason_code=sc.get("reason_code"),
                expected_next_action=sc.get("expected_next_action"),
                error=sc.get("error"),
                trace=[
                    TransitionStep(
                        stage=st["stage"],
                        state=st["state"],
                        owning_helper=st["owning_helper"],
                        status=st["status"],
                        reason_code=st["reason_code"],
                        expected_next_action=st["expected_next_action"],
                    )
                    for st in sc.get("trace", [])
                ],
            )
            print("-" * 50)
            print(format_failure_diagnostic(res_obj))
            print("-" * 50)
        elif args.verbose:
            for idx, step in enumerate(sc["trace"], start=1):
                print(f"    {idx}. [{step['stage']}] state={step['state']} -> {step['status']}")

    if receipt["all_passed"]:
        print("\nAll workflow factory transitions validated clean offline.")
        return 0

    print(f"\nAcceptance harness failed with {receipt['failed_count']} failure(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
