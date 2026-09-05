"""Focused contract tests for the shared issue-audit core."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, Any

import pytest

from scripts.dev import issue_audit_core
from scripts.dev.issue_audit_core import (
    _run_command,
    _run_gh,
    apply_mutations,
    attach_issue_comments,
    build_audit_plan,
    build_decision_envelope,
    build_pending_decision_queue,
    classify_issue,
    closure_evidence,
    compute_plan_digest,
    discover_issue_comments,
    discover_issue_timeline_merged_prs,
    label_api_path,
    main,
    parse_decision_answer,
    select_next_pending_decision,
    validate_decision_envelope,
)

if TYPE_CHECKING:
    from pathlib import Path


EXPECTED_ISSUE_UPDATED_AT = "2026-08-23T00:00:00Z"


def _expected_issue(state: str = "open") -> dict[str, str]:
    return {"state": state, "updated_at": EXPECTED_ISSUE_UPDATED_AT}


def _attach_valid_provenance(plan: dict[str, Any]) -> dict[str, Any]:
    """Populate valid commit/classifier provenance and compute plan_digest for test plans."""
    plan["source_sha"] = issue_audit_core.resolve_source_sha()
    plan["classifier_digest"] = issue_audit_core.resolve_classifier_digest()
    plan["producer"] = issue_audit_core.resolve_producer_identity()
    plan["provenance"] = {
        "schema": issue_audit_core.PROVENANCE_SCHEMA,
        "source_sha": plan["source_sha"],
        "classifier_digest": plan["classifier_digest"],
        "producer": plan["producer"],
    }
    plan["plan_digest"] = compute_plan_digest(plan)
    return plan


def _issue(
    number: int,
    *,
    labels: list[str] | None = None,
    body: str = "",
    title: str = "Implement bounded change",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "state": "open",
        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
        "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "author": "maintainer",
        "labels": labels or [],
        "body": body,
        "comments": [],
    }


def _healthy_quota_meta(*, core_remaining: int = 500) -> tuple:
    """Return deterministic quota preflight data for inventory unit tests."""
    return (
        issue_audit_core.RateLimitSnapshot(
            status="ok",
            graphql_remaining=500,
            graphql_reset_at=1_800_000_000,
            core_remaining=core_remaining,
            core_reset_at=1_800_000_100,
        ),
        {
            "available": True,
            "status": "ok",
            "core_remaining": core_remaining,
            "core_reset_at": 1_800_000_100,
            "available_budget": max(0, core_remaining - 10),
            "min_core_remaining": 10,
            "retry_command": "cmd",
            "next_action": "none",
            "reason": "sufficient core quota available",
            "errors": [],
            "quota_exhausted": False,
            "quota_uncertain": False,
            "budget_exhausted": False,
        },
    )


def test_state_contradiction_prefers_observed_active_work() -> None:
    """Active work wins over a contradictory ready label to avoid false readiness."""
    classification = classify_issue(
        _issue(
            101,
            labels=["state:ready", "state:running"],
            body="## Definition of Done\n- [ ] verify the change",
        ),
        open_prs=[{"number": 900, "title": "Fixes #101", "head_ref": "issue-101"}],
        available_labels={"state:ready", "state:running", "state:blocked", "decision-required"},
    )

    assert classification.classification == "running"
    assert {
        (mutation["operation"], mutation["value"]) for mutation in classification.mutations
    } == {("remove_label", "state:ready")}
    assert not any(
        mutation["operation"] == "add_label" and mutation["value"] == "state:ready"
        for mutation in classification.mutations
    )


def test_active_work_never_promotes_acceptance_text_to_ready() -> None:
    """Acceptance text cannot promote an issue while an active claim is present."""
    classification = classify_issue(
        _issue(
            102,
            body="## Definition of Done\n- [ ] verify the change\n## Validation / Testing\ncommand: pytest",
        ),
        claims={102: {"claimed": True, "sha": "abc123"}},
        available_labels={"state:ready", "state:running"},
    )

    assert classification.classification == "running"
    assert {mutation["value"] for mutation in classification.mutations} == {"state:running"}


def test_parent_issue_is_never_promoted_to_ready() -> None:
    """Parent coordination issues remain non-leaves despite complete-looking prose."""
    classification = classify_issue(
        _issue(
            122,
            labels=["parent"],
            body="## Acceptance Criteria\n- [ ] complete the bounded child work",
        ),
        available_labels={"state:ready"},
    )

    assert classification.classification == "unclassified"
    assert not any(
        mutation["operation"] == "add_label" and mutation["value"] == "state:ready"
        for mutation in classification.mutations
    )
    assert any("parent issue cannot be promoted" in finding for finding in classification.findings)


def test_stale_running_state_is_preserved_and_not_promoted_to_ready() -> None:
    """A stale running label remains uncertain instead of being guessed complete."""
    classification = classify_issue(
        _issue(
            103,
            labels=["state:running"],
            body="## Definition of Done\n- [ ] verify the change\n"
            "## Validation / Testing\ncommand: pytest",
        ),
        available_labels={"state:ready", "state:running"},
    )

    assert classification.classification == "running"
    assert classification.mutations == ()
    assert any("state:running" in finding for finding in classification.findings)


def test_terminal_review_status_replaces_stale_dispatch_state() -> None:
    """A terminal report routes completed execution to review instead of dispatch."""
    issue = _issue(124, labels=["state:ready"])
    issue["comments"] = [
        {
            "body": (
                "Report status: diagnostic_ready_for_domain_review; benchmark_success_allowed=true."
            ),
            "created_at": "2026-08-15T10:00:00Z",
        }
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "state:review", "decision-required"},
    )

    assert classification.classification == "decision-required"
    assert classification.terminal_review_evidence
    assert {
        (mutation["operation"], mutation["value"]) for mutation in classification.mutations
    } == {
        ("remove_label", "state:ready"),
        ("add_label", "state:review"),
        ("add_label", "decision-required"),
    }


def test_canonical_ruling_supersedes_older_terminal_review_status() -> None:
    """A reviewed result does not return to the decision queue after its ruling."""
    issue = _issue(6095, body="Owner decision required: classify the completed report.")
    issue["comments"] = [
        {
            "body": "Report status: diagnostic_ready_for_domain_review",
            "created_at": "2026-08-15T10:00:00Z",
        },
        {
            "body": "ll7/robot_sf_ll7#6095: approve-bounded-diagnostic",
            "created_at": "2026-08-16T10:00:00Z",
        },
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:review", "decision-required"},
    )

    assert classification.terminal_review_evidence == ()
    assert classification.decision_required is False
    assert classification.mutations == ()


def test_new_terminal_review_status_after_ruling_reopens_review_gate() -> None:
    """A later completed run still returns to review after an older ruling."""
    issue = _issue(6095)
    issue["comments"] = [
        {
            "body": "ll7/robot_sf_ll7#6095: approve-bounded-diagnostic",
            "created_at": "2026-08-16T10:00:00Z",
        },
        {
            "body": "Report status: diagnostic_ready_for_domain_review",
            "created_at": "2026-08-17T10:00:00Z",
        },
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:review", "decision-required"},
    )

    assert classification.terminal_review_evidence
    assert classification.decision_required is True
    assert {mutation["value"] for mutation in classification.mutations} == {
        "state:review",
        "decision-required",
    }


def test_terminal_review_status_does_not_hide_active_execution() -> None:
    """A visible active job keeps terminal-status evidence fail-closed."""
    issue = _issue(125, labels=["state:ready"])
    issue["comments"] = [{"body": "Report status: diagnostic_ready_for_domain_review"}]

    classification = classify_issue(
        issue,
        jobs=[{"name": "issue-125-campaign", "job_name": "issue-125-campaign"}],
        available_labels={"state:ready", "state:running", "state:review", "decision-required"},
    )

    assert classification.classification == "decision-required"
    assert {mutation["value"] for mutation in classification.mutations} == {
        "state:ready",
        "state:running",
        "decision-required",
    }
    assert not any(mutation["value"] == "state:review" for mutation in classification.mutations)


def test_terminal_review_status_supersedes_open_report_pr_for_dispatch() -> None:
    """An open report PR is not treated as active campaign execution."""
    issue = _issue(126, labels=["state:ready"])
    issue["comments"] = [{"body": "Report status: diagnostic_ready_for_domain_review"}]

    classification = classify_issue(
        issue,
        open_prs=[
            {
                "number": 902,
                "title": "Report repair for #126",
                "head_ref": "issue-126-report-repair",
            }
        ],
        available_labels={"state:ready", "state:running", "state:review", "decision-required"},
    )

    assert {mutation["value"] for mutation in classification.mutations} == {
        "state:ready",
        "state:review",
        "decision-required",
    }
    assert not any(mutation["value"] == "state:running" for mutation in classification.mutations)


def test_future_terminal_review_rule_is_not_current_status_evidence() -> None:
    """Acceptance prose about a future terminal run cannot suppress dispatch."""
    classification = classify_issue(
        _issue(
            127,
            labels=["state:ready"],
            body=(
                "If the campaign reaches a terminal state, review the report before "
                "interpreting it."
            ),
        ),
        available_labels={"state:ready", "state:review", "decision-required"},
    )

    assert classification.terminal_review_evidence == ()
    assert classification.mutations == ()


def test_state_qualifiers_are_preserved_during_execution_state_cleanup() -> None:
    """Composable state qualifiers survive cleanup of mutually exclusive states."""
    classification = classify_issue(
        _issue(
            104,
            labels=["state:blocked", "state:needs-artifact-promotion", "state:review"],
            body="The external checkpoint is missing.",
        ),
        available_labels={
            "state:blocked",
            "state:blocked-external-input",
            "state:needs-artifact-promotion",
            "state:review",
        },
    )

    assert classification.execution_state_labels == ("state:blocked",)
    assert {"state:needs-artifact-promotion", "state:review"}.issubset(classification.state_labels)
    assert not any(
        mutation["operation"] == "remove_label"
        and mutation["value"] in {"state:needs-artifact-promotion", "state:review"}
        for mutation in classification.mutations
    )


def test_issue_audit_and_admission_share_execution_state_taxonomy() -> None:
    """The audit and admission gates must not drift on execution-state labels."""
    from scripts.dev import issue_implementability
    from scripts.dev.issue_state_taxonomy import EXECUTION_STATE_LABELS

    classification = classify_issue(
        _issue(
            105,
            labels=["state:blocked", "state:parked"],
            body="The source run is parked until the prerequisite closes.",
        ),
        available_labels={"state:blocked", "state:parked"},
    )
    report = issue_implementability.evaluate_issue(
        {
            "number": 105,
            "title": "fixture issue",
            "body": (
                "## Objective\nFix it.\n\n## Scope\nOne file.\n\n## Inputs\n- file\n\n"
                "## Acceptance criteria\n- pass\n\n## Validation\n- pytest\n"
            ),
            "state": "OPEN",
            "url": "https://github.test/issues/105",
            "labels": ["state:blocked", "state:parked"],
            "assignees": [],
        },
        {"ok": True, "claimed": False, "claim_ref": None, "sha": None},
    )

    assert set(classification.execution_state_labels) == {"state:blocked"}
    assert set(EXECUTION_STATE_LABELS) == set(
        issue_implementability.execution_state_labels(set(EXECUTION_STATE_LABELS))
    )
    assert report["admission_reason"] != "state_label_conflict"


def test_missing_optional_job_visibility_preserves_slurm_issue_without_blocker_claim() -> None:
    """Unavailable SLURM visibility blocks promotion without inventing a blocker."""
    classification = classify_issue(
        _issue(
            111,
            labels=["resource:slurm"],
            body="## Definition of Done\n- [ ] verify the campaign\n## Validation / Testing\ncommand: sbatch",
        ),
        job_inventory_available=False,
        available_labels={"state:ready", "state:blocked", "resource:slurm"},
    )

    assert classification.classification == "unclassified"
    assert classification.mutations == ()
    assert any("SLURM job inventory unavailable" in finding for finding in classification.findings)


def test_gate_matching_does_not_pair_unrelated_status_and_topic_lines() -> None:
    """Gate detection requires issue-local topic and status evidence in one context."""
    proven = classify_issue(
        _issue(
            112,
            body="Blocked-by: #900\n"
            "Rights: license missing pending permission review.\n"
            "## Definition of Done\n- [ ] record the decision",
        ),
        available_labels={"state:blocked", "state:blocked-external-input"},
    )
    assert proven.classification == "blocked"
    assert {mutation["value"] for mutation in proven.mutations} == {"state:blocked"}

    unrelated = classify_issue(
        _issue(
            113,
            body="Status: blocked.\nThe license is documented and release-ready.",
        ),
        available_labels={"state:blocked", "state:blocked-external-input"},
    )
    assert unrelated.blocker_evidence == ()
    assert unrelated.mutations == ()


def test_gate_matching_ignores_aggregate_reports_and_conditional_rules() -> None:
    """Aggregate reports and hypothetical rules do not become current issue gates."""
    report = classify_issue(
        _issue(
            114,
            body=(
                "Result: 28 of 30 resource:slurm issues are gate-blocked.\n"
                "Stop if provenance cannot be resolved; report the provenance blocker."
            ),
        ),
        available_labels={"state:blocked", "state:blocked-external-input"},
    )
    assert report.blocker_evidence == ()
    assert report.mutations == ()

    current = classify_issue(
        _issue(
            115,
            body="Blocked-by: #901\nThis issue is blocked until the required checkpoint is staged.",
        ),
        available_labels={"state:blocked", "state:blocked-external-input"},
    )
    assert current.classification == "blocked"
    assert {mutation["value"] for mutation in current.mutations} == {"state:blocked-external-input"}


def test_decision_detection_distinguishes_resolved_records_from_open_gates() -> None:
    """Resolved decision records do not keep an issue in the pending queue."""
    resolved = classify_issue(
        _issue(116, body="## Maintainer Decision\nDecision: defer.\nDecision reaffirmed."),
        available_labels={"decision-required"},
    )
    assert resolved.decision_required is False
    assert resolved.mutations == ()

    historical = classify_issue(
        _issue(
            119,
            body=(
                "Owner decision required: choose the disposition.\n"
                "Later update: the decision-required label was removed; the disposition is settled."
            ),
        ),
        available_labels={"decision-required"},
    )
    assert historical.decision_required is False
    assert historical.mutations == ()

    answered = classify_issue(
        _issue(
            121,
            body=(
                "Owner decision required: approve the preregistration.\n"
                "Choose option (a) for the design sequence; do not launch yet."
            ),
        ),
        available_labels={"decision-required"},
    )
    assert answered.decision_required is False
    assert answered.mutations == ()

    negated = classify_issue(
        _issue(120, body="No owner decision is required for this mechanical fix."),
        available_labels={"decision-required"},
    )
    assert negated.decision_required is False
    assert negated.mutations == ()

    pending = classify_issue(
        _issue(117, body="Owner decisions required before merge-ready work."),
        available_labels={"decision-required"},
    )
    assert pending.decision_required is True
    assert {mutation["value"] for mutation in pending.mutations} == {"decision-required"}


def test_canonical_same_issue_ruling_suppresses_older_decision_prompt() -> None:
    """A later exact repository ruling ends a superseded decision prompt."""
    issue = _issue(
        7409,
        body="Owner decision required: choose the disposition.",
    )
    issue["comments"] = [
        {
            "body": "ll7/robot_sf_ll7#7409: hold-artifact-rows",
            "created_at": "2026-08-18T10:00:00Z",
        }
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )

    assert classification.decision_required is False
    assert classification.mutations == ()


def test_conditional_reopen_clause_after_ruling_does_not_reopen_gate() -> None:
    """A ruling's revival condition is not itself a fresh maintainer request."""
    issue = _issue(6155, body="Owner decision required: authorize publication.")
    issue["comments"] = [
        {
            "body": (
                "ll7/robot_sf_ll7#6155: approve-exact-publication\n\n"
                "Any byte, manifest, rights, or target-SHA drift reopens the release decision."
            ),
            "created_at": "2026-08-31T12:00:00Z",
        }
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:blocked", "decision-required"},
    )

    assert classification.decision_required is False
    assert not any(
        mutation["operation"] == "add_label" and mutation["value"] == "decision-required"
        for mutation in classification.mutations
    )


def test_decision_request_after_canonical_ruling_reopens_gate() -> None:
    """A genuine later reopen request remains actionable after a ruling."""
    issue = _issue(7411, body="Owner decision required: choose the registry policy.")
    issue["comments"] = [
        {"body": "ll7/robot_sf_ll7#7411: use-commit-pinned-repository-registry"},
        {"body": "Reopen the decision: the maintainer must choose a replacement."},
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )

    assert classification.decision_required is True
    assert {mutation["value"] for mutation in classification.mutations} == {"decision-required"}


def test_canonical_ruling_uses_timestamp_not_input_comment_order() -> None:
    """A newer ruling suppresses an older prompt even when REST rows are reversed."""
    issue = _issue(7410)
    issue["comments"] = [
        {
            "body": "ll7/robot_sf_ll7#7410: hold-artifact-rows",
            "created_at": "2026-08-18T10:00:00Z",
        },
        {
            "body": "Owner decision required: choose the disposition.",
            "created_at": "2026-08-18T09:00:00Z",
        },
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )

    assert classification.decision_required is False
    assert classification.mutations == ()


def test_incomplete_comment_order_does_not_suppress_decision_prompt() -> None:
    """An untimestamped source cannot be treated as later ruling evidence."""
    issue = _issue(7414)
    issue["comments"] = [
        {"body": "ll7/robot_sf_ll7#7414: hold-artifact-rows"},
        {
            "body": "Owner decision required: choose the disposition.",
            "created_at": "2026-08-18T09:00:00Z",
        },
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )

    assert classification.decision_required is True
    assert {mutation["value"] for mutation in classification.mutations} == {"decision-required"}


def test_copied_example_ruling_line_does_not_suppress_decision_prompt() -> None:
    """An exact ruling copied under an example marker is not a live ruling."""
    issue = _issue(7415, body="Owner decision required: choose the disposition.")
    issue["comments"] = [
        {
            "body": (
                "Example copied ruling (do not apply):\nll7/robot_sf_ll7#7415: hold-artifact-rows"
            ),
            "created_at": "2026-08-18T10:00:00Z",
        }
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )

    assert classification.decision_required is True
    assert {mutation["value"] for mutation in classification.mutations} == {"decision-required"}


def test_wrong_quoted_or_malformed_ruling_does_not_suppress_decision() -> None:
    """Only an exact same-issue ruling line is terminal decision evidence."""
    issue = _issue(7412, body="Owner decision required: choose the release disposition.")
    issue["comments"] = [
        {"body": "> ll7/robot_sf_ll7#7412: create-v2.1-and-preserve-v2"},
        {"body": "ll7/robot_sf_ll7#7413: unrelated-ruling"},
        {"body": "ll7/robot_sf_ll7#7412 create-v2.1-and-preserve-v2"},
    ]

    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )

    assert classification.decision_required is True
    assert {mutation["value"] for mutation in classification.mutations} == {"decision-required"}


def test_blocker_replaces_single_stale_execution_state() -> None:
    """A proven blocker replaces one stale execution state with its blocker state."""
    classification = classify_issue(
        _issue(
            118,
            labels=["state:running"],
            body="Blocked-by: #902\nThis issue is blocked until the required dataset is staged.",
        ),
        available_labels={
            "state:running",
            "state:blocked",
            "state:blocked-external-input",
        },
    )
    assert {
        (mutation["operation"], mutation["value"]) for mutation in classification.mutations
    } == {
        ("remove_label", "state:running"),
        ("add_label", "state:blocked-external-input"),
    }
    blocked_mutation = next(
        mutation
        for mutation in classification.mutations
        if mutation["value"] == "state:blocked-external-input"
    )
    assert blocked_mutation["blocked_reason"] == ["Blocked-by reference present: Blocked-by: #902"]


def test_blocker_without_reason_routes_to_needs_triage() -> None:
    """A prose-only blocker cannot create a dispatch-suppressing state label."""
    classification = classify_issue(
        _issue(122, body="This issue is blocked until the required dataset is staged."),
        available_labels={"state:blocked", "state:blocked-external-input", "needs-triage"},
    )

    assert classification.classification == "blocked"
    assert classification.blocked_reason_evidence == ()
    assert classification.blocked_label_decision == "declined-needs-triage"
    assert not any(
        mutation["operation"] == "add_label"
        and mutation["value"] in issue_audit_core.BLOCKED_LABELS
        for mutation in classification.mutations
    )
    assert {mutation["value"] for mutation in classification.mutations} == {"needs-triage"}
    assert any("declined state:blocked" in finding for finding in classification.findings)


@pytest.mark.parametrize(
    "phrase",
    ["none", "no blockers", "n/a", "not applicable", "clear", "resolved"],
)
def test_non_blocking_blocked_by_phrase_does_not_route_to_triage(phrase: str) -> None:
    """A documented absence of blockers is not current gate evidence."""
    classification = classify_issue(
        _issue(
            123,
            body=f"Blocked by: {phrase}.\n## Acceptance Criteria\n- [ ] implement the change",
        ),
        available_labels={"state:ready", "state:blocked", "needs-triage"},
    )

    assert classification.blocker_evidence == ()
    assert not any(mutation["value"] == "needs-triage" for mutation in classification.mutations)


@pytest.mark.parametrize(
    "declaration",
    [
        "Provenance is blocked by: none",
        "Provenance is blocked-by: none",
        "Dataset is blocked by: no blockers",
        "Compute remains blocked by: N/A",
    ],
)
def test_non_blocking_declaration_does_not_trigger_domain_gate(declaration: str) -> None:
    """Domain-specific gate detection also ignores explicit no-blocker text."""
    assert issue_audit_core._gate_evidence(declaration) == []


def test_multiline_non_blocking_declaration_does_not_route_to_triage() -> None:
    """A Markdown Blocked-by section with an explicit empty value is non-blocking."""
    classification = classify_issue(
        _issue(
            125,
            body="## Blocked by\n\nNone\n\n## Acceptance Criteria\n- [ ] implement the change",
        ),
        available_labels={"state:ready", "state:blocked", "needs-triage"},
    )

    assert classification.blocker_evidence == ()
    assert not any(mutation["value"] == "needs-triage" for mutation in classification.mutations)


@pytest.mark.parametrize(
    "declaration",
    ["Blocked by: none or #902", "Blocked-by: none, pending a decision"],
)
def test_ambiguous_non_blocking_value_remains_fail_closed(declaration: str) -> None:
    """A value that has extra content is not silently treated as clear."""
    evidence = issue_audit_core._gate_evidence(declaration)
    assert any(item["kind"] == "blocked" for item in evidence)


def test_non_blocking_declaration_does_not_hide_a_later_gate() -> None:
    """A later genuine gate on the same line remains current evidence."""
    classification = classify_issue(
        _issue(
            124,
            body="Blocked by: none. This issue remains blocked until the dataset is staged.",
        ),
        available_labels={"state:blocked", "needs-triage"},
    )

    assert {item["kind"] for item in classification.blocker_evidence} == {
        "external-input",
        "blocked",
    }


def test_audit_plan_does_not_route_explicit_no_blocker_text_to_triage() -> None:
    """The complete audit plan ignores an explicit no-blocker declaration."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(
                    123,
                    labels=["state:ready"],
                    body="Blocked by: none.\n## Acceptance Criteria\n- [ ] implement the change",
                )
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["state:ready", "state:blocked", "needs-triage"],
            "inventory": {},
        }
    )

    assert plan["issues"][0]["blocker_evidence"] == []
    assert not any(
        mutation["operation"] == "add_label" and mutation["value"] == "needs-triage"
        for mutation in plan["mutations"]
    )


def test_blocked_triage_block_binds_reason_to_blocked_label() -> None:
    """A complete blocked-triage block is accepted as explicit reason evidence."""
    body = (
        "<!-- blocked-triage-v1 tracking=7067 -->\n"
        "```yaml\n"
        "blocker_class: dependency\n"
        "unblock_condition: '#902 is closed'\n"
        "watcher: next queue pass\n"
        "next_check_at: next queue pass\n"
        "last_meaningful_progress_at: '2026-08-14'\n"
        "```\n"
        "This issue is blocked until the required dataset is staged."
    )
    classification = classify_issue(
        _issue(125, body=body),
        available_labels={"state:blocked", "state:blocked-external-input"},
    )

    assert classification.blocked_label_decision == "apply"
    blocked_mutation = next(
        mutation
        for mutation in classification.mutations
        if mutation["value"] == "state:blocked-external-input"
    )
    assert blocked_mutation["blocked_reason"] == ["blocked-triage-v1 reason block present"]


def test_needs_triage_label_blocks_ready_promotion() -> None:
    """An issue with needs-triage must never be promoted to state:ready."""
    classification = classify_issue(
        _issue(
            200,
            labels=["needs-triage"],
            body="## Acceptance Criteria\n- [ ] Works correctly\n## Validation\n- Run tests",
        ),
        available_labels={"state:ready", "needs-triage"},
    )

    assert classification.classification == "blocked"
    assert classification.blocker_evidence
    assert not any(
        mutation["operation"] == "add_label" and mutation["value"] == "state:ready"
        for mutation in classification.mutations
    )
    assert any(item["kind"] == "triage" for item in classification.blocker_evidence)


def test_audit_plan_reports_blocked_label_decision() -> None:
    """The plan exposes whether a blocked-label write was applied or declined."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [_issue(123, body="This issue is blocked until the dataset is staged.")],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["state:blocked", "state:blocked-external-input", "needs-triage"],
            "inventory": {},
        }
    )

    assert plan["blocked_label_report"] == [
        {
            "issue": 123,
            "decision": "declined-needs-triage",
            "blocker_evidence": [
                {"kind": "external-input", "text": "dataset evidence includes explicit gate"},
                {"kind": "blocked", "text": "explicit current blocker"},
            ],
            "reason_evidence": [],
            "fallback_label": "needs-triage",
        }
    ]
    assert plan["counts"]["blocked_label_decisions"] == 1
    assert all(
        mutation["expected_issue"] == {**_expected_issue(), "labels": []}
        for mutation in plan["mutations"]
    )


def test_type_mirror_requires_complete_valid_archetype_metadata() -> None:
    """Type labels mirror only complete, valid archetype metadata."""
    fence = chr(96) * 3
    valid_body = (
        "## Archetype Metadata\n\n"
        f"{fence}yaml\n"
        "archetype: docs\n"
        "evidence_tier: idea\n"
        "linked_policy:\n"
        "  - docs/context/issue_1512_issue_archetypes.md\n"
        f"{fence}\n"
    )
    valid = classify_issue(
        _issue(108, body=valid_body),
        available_labels={"type:docs"},
    )
    assert {mutation["value"] for mutation in valid.mutations} == {"type:docs"}

    malformed = classify_issue(
        _issue(
            109,
            body=f"## Archetype Metadata\n\n{fence}yaml\narchetype: docs\n{fence}\n",
        ),
        available_labels={"type:docs"},
    )
    assert not any(mutation["value"] == "type:docs" for mutation in malformed.mutations)


def test_comment_discovery_is_bounded_and_rest_normalized() -> None:
    """Comment discovery preserves REST metadata within the bounded read contract."""

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        assert args[0] == "api"
        assert "issues/110/comments" in args[1]
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                [
                    {
                        "body": "Maintainer decision required.",
                        "user": {"login": "owner"},
                        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/110#issuecomment-1",
                        "created_at": "2026-08-11T10:00:00Z",
                    }
                ]
            ),
            "",
        )

    comments, metadata = discover_issue_comments("ll7/robot_sf_ll7", 110, runner=runner)

    assert comments == [
        {
            "body": "Maintainer decision required.",
            "user": "owner",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/110#issuecomment-1",
            "created_at": "2026-08-11T10:00:00Z",
        }
    ]
    assert metadata["truncated"] is False
    assert metadata["errors"] == []


def test_comment_inventory_reports_degraded_reads_as_unavailable() -> None:
    """Failed comment reads cannot be mistaken for an empty decision thread."""

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        return subprocess.CompletedProcess(args, 1, "", "comment endpoint unavailable")

    issues = [_issue(109)]
    metadata = attach_issue_comments("ll7/robot_sf_ll7", issues, runner=runner)

    assert metadata["available"] is False
    assert metadata["errors"]
    assert issues[0]["comments"] == []


def test_comment_inventory_bails_out_after_actual_rate_limit() -> None:
    """A rate-limited comment thread stops optional comment reads immediately."""
    queried: list[int] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        endpoint = args[1]
        issue_number = int(endpoint.split("issues/")[1].split("/")[0])
        queried.append(issue_number)
        return subprocess.CompletedProcess(
            args,
            1,
            "",
            "gh: HTTP 403: API rate limit exceeded for user",
        )

    issues = [_issue(101), _issue(102), _issue(103)]
    metadata = attach_issue_comments(
        "ll7/robot_sf_ll7",
        issues,
        runner=runner,
        request_budget=issue_audit_core._RestRequestBudget.from_available(10),
    )

    assert queried == [101]
    assert metadata["processed_issue_count"] == 1
    assert metadata["requests_attempted"] == 1
    assert metadata["rate_limited"] is True
    assert metadata["quota_exhausted"] is True
    assert metadata["available"] is False


def test_generic_permission_403_is_not_quota_exhaustion() -> None:
    """A permission 403 remains an ordinary incomplete source, not quota exhaustion."""
    queried: list[int] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        endpoint = args[1]
        issue_number = int(endpoint.split("issues/")[1].split("/")[0])
        queried.append(issue_number)
        return subprocess.CompletedProcess(
            args,
            1,
            "",
            "gh: HTTP 403: Resource not accessible by integration",
        )

    _rows, metadata = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [101, 102, 103],
        runner=runner,
    )

    assert queried == [101, 102, 103]
    assert metadata["available"] is False
    assert metadata["errors"]
    assert metadata.get("rate_limited") is not True
    assert metadata.get("quota_exhausted") is not True


def test_issue_timeline_recovers_merged_cross_referenced_pr() -> None:
    """A bounded issue timeline supplies merged-PR coverage beyond global history."""

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        assert args[0] == "api"
        assert "issues/110/timeline" in args[1]
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                [
                    {
                        "event": "cross-referenced",
                        "created_at": "2026-08-14T10:00:00Z",
                        "source": {
                            "type": "issue",
                            "issue": {
                                "number": 901,
                                "title": "Add audit coverage",
                                "body": "Refs #110",
                                "state": "closed",
                                "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                                "pull_request": {
                                    "merged_at": "2026-08-14T09:00:00Z",
                                    "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                                },
                            },
                        },
                    },
                    {"event": "commented"},
                ]
            ),
            "",
        )

    rows, metadata = discover_issue_timeline_merged_prs("ll7/robot_sf_ll7", [110], runner=runner)

    assert [row["number"] for row in rows] == [901]
    assert rows[0]["coverage_source"] == "targeted_issue_timeline"
    assert rows[0]["timeline_issue"] == 110
    assert rows[0]["linked_issue_numbers"] == [110]
    assert metadata["available"] is True
    assert metadata["event_count"] == 2
    assert metadata["row_count"] == 1


def test_issue_timeline_failure_is_explicitly_unavailable() -> None:
    """Timeline/API failures never become an empty successful fallback."""

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        return subprocess.CompletedProcess(args, 1, "", "timeline unavailable")

    rows, metadata = discover_issue_timeline_merged_prs("ll7/robot_sf_ll7", [110], runner=runner)

    assert rows == []
    assert metadata["available"] is False
    assert metadata["errors"]


def test_inventory_uses_timeline_fallback_for_partial_closed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial global history triggers targeted merged-PR coverage."""
    issue = _issue(110)
    timeline_row = {
        "number": 901,
        "title": "Add audit coverage",
        "body": "Refs #110",
        "state": "closed",
        "url": "https://github.com/ll7/robot_sf_ll7/pull/901",
        "merged_at": "2026-08-14T09:00:00Z",
        "head_ref": "",
        "linked_issue_numbers": [110],
        "coverage_source": "targeted_issue_timeline",
    }

    monkeypatch.setattr(
        issue_audit_core,
        "discover_open_issues",
        lambda *args, **kwargs: ([issue], {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_core_quota",
        lambda *args, **kwargs: _healthy_quota_meta(),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_pull_requests",
        lambda _repo, *, state, max_pages, runner, request_budget: (
            ([], {"truncated": False, "errors": []})
            if state == "open"
            else ([], {"truncated": True, "errors": []})
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_issue_timeline_merged_prs",
        lambda *args, **kwargs: (
            [timeline_row],
            {"available": True, "truncated": False, "errors": [], "row_count": 1},
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_repository_labels",
        lambda *args, **kwargs: (set(), {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    inventory = issue_audit_core.discover_inventory("ll7/robot_sf_ll7")

    assert [row["number"] for row in inventory["merged_prs"]] == [901]
    assert inventory["inventory"]["closure_coverage"]["mode"] == ("issue_timeline_fallback")
    assert inventory["inventory"]["closure_coverage"]["complete_for_open_issues"] is True
    assert inventory["inventory"]["closed_prs"]["truncated"] is True


def test_inventory_uses_an_independent_closed_pr_page_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed-PR history can outgrow the smaller ordinary inventory page budget."""
    observed: list[tuple[str, int]] = []

    monkeypatch.setattr(
        issue_audit_core,
        "discover_open_issues",
        lambda *args, **kwargs: ([], {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_core_quota",
        lambda *args, **kwargs: _healthy_quota_meta(),
    )

    def discover_prs(
        _repo: str,
        *,
        state: str,
        max_pages: int,
        runner: object,
        request_budget: object,
    ) -> tuple:
        observed.append((state, max_pages))
        return [], {"truncated": False, "errors": []}

    monkeypatch.setattr(issue_audit_core, "discover_pull_requests", discover_prs)
    monkeypatch.setattr(
        issue_audit_core,
        "discover_repository_labels",
        lambda *args, **kwargs: (set(), {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    issue_audit_core.discover_inventory("ll7/robot_sf_ll7", max_pages=2, max_closed_pr_pages=47)

    assert observed == [("open", 2), ("closed", 47)]


def test_inventory_preflights_quota_before_comments_with_provided_runner() -> None:
    """The production inventory runner uses one bounded API path in quota-first order."""
    calls: list[list[str]] = []
    issue_payload = {
        "number": 110,
        "title": "Bounded issue",
        "body": "## Acceptance Criteria\n- [x] verified",
        "state": "open",
        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/110",
        "user": {"login": "maintainer"},
        "labels": [],
    }

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        calls.append(args)
        if args == ["api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "resources": {
                            "graphql": {"remaining": 500, "reset": 1_800_000_000},
                            "core": {"remaining": 100, "reset": 1_800_000_100},
                        }
                    }
                ),
                "",
            )
        endpoint = args[1]
        if "issues?state=open" in endpoint:
            payload: object = [issue_payload]
        elif "/comments?" in endpoint:
            payload = []
        elif "pulls?state=open" in endpoint or "pulls?state=closed" in endpoint:
            payload = []
        elif "/labels?" in endpoint:
            payload = []
        else:
            raise AssertionError(f"unexpected REST endpoint: {endpoint}")
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    def command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "", "")

    inventory = issue_audit_core.discover_inventory(
        "ll7/robot_sf_ll7",
        include_comments=True,
        runner=runner,
        command_runner=command_runner,
    )

    assert calls[0] == ["api", "rate_limit"]
    assert any("/comments?" in args[1] for args in calls[1:])
    assert inventory["inventory"]["comments"]["processed_issue_count"] == 1
    assert inventory["quota"]["request_budget"] == 90
    assert inventory["quota"]["requests_attempted"] == 5
    assert inventory["quota"]["requests_remaining"] == 85


def test_inventory_low_budget_suppresses_ready_mutations_with_production_runner() -> None:
    """A real bounded inventory run suppresses mutations when its shared budget truncates labels."""
    calls: list[list[str]] = []
    issue_payload = {
        "number": 111,
        "title": "Ready issue",
        "body": "## Acceptance Criteria\n- [x] verified\n\n## Validation\n- [x] pytest",
        "state": "open",
        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/111",
        "user": {"login": "maintainer"},
        "labels": [{"name": "state:running"}],
    }
    label_rows = [{"name": "state:ready"}, {"name": "state:running"}]
    label_rows.extend({"name": f"fixture-label-{index}"} for index in range(98))

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        calls.append(args)
        if args == ["api", "rate_limit"]:
            payload: object = {
                "resources": {
                    "graphql": {"remaining": 500, "reset": 1_800_000_000},
                    "core": {"remaining": 16, "reset": 1_800_000_100},
                }
            }
        else:
            endpoint = args[1]
            if "issues?state=open" in endpoint:
                payload = [issue_payload]
            elif "pulls?state=open" in endpoint or "pulls?state=closed" in endpoint:
                payload = []
            elif "/labels?" in endpoint:
                payload = label_rows
            else:
                raise AssertionError(f"unexpected REST endpoint: {endpoint}")
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    def command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "", "")

    inventory = issue_audit_core.discover_inventory(
        "ll7/robot_sf_ll7",
        runner=runner,
        command_runner=command_runner,
    )
    plan = build_audit_plan(inventory)

    assert inventory["quota"]["request_budget"] == 6
    assert inventory["quota"]["requests_attempted"] == 6
    assert inventory["quota"]["status"] == "insufficient"
    assert inventory["quota"]["quota_exhausted"] is False
    assert inventory["quota"]["budget_exhausted"] is True
    assert inventory["inventory"]["labels"]["truncated"] is True
    assert plan["classification_status"]["mutations_suppressed"] is True
    assert plan["mutations"] == []
    assert plan["issues"][0]["mutations"] == []
    assert "labels" in plan["truncation_or_errors"]


def test_inventory_mid_run_rate_limit_emits_reset_handoff() -> None:
    """A mid-run rate limit stops later REST reads and publishes retry metadata."""
    calls: list[list[str]] = []
    issue_payload = {
        "number": 112,
        "title": "Needs closure evidence",
        "body": "## Acceptance Criteria\n- [x] verified",
        "state": "open",
        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
        "html_url": "https://github.com/ll7/robot_sf_ll7/issues/112",
        "user": {"login": "maintainer"},
        "labels": [],
    }

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        assert input_text is None
        calls.append(args)
        if args == ["api", "rate_limit"]:
            payload: object = {
                "resources": {
                    "graphql": {"remaining": 500, "reset": 1_800_000_000},
                    "core": {"remaining": 50, "reset": 1_800_000_100},
                }
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        endpoint = args[1]
        if "issues?state=open" in endpoint:
            return subprocess.CompletedProcess(args, 0, json.dumps([issue_payload]), "")
        if "pulls?state=open" in endpoint:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if "pulls?state=closed" in endpoint:
            return subprocess.CompletedProcess(
                args,
                1,
                "",
                "gh: HTTP 403: API rate limit exceeded for user",
            )
        raise AssertionError(f"REST read should have stopped before: {endpoint}")

    def command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "", "")

    inventory = issue_audit_core.discover_inventory(
        "ll7/robot_sf_ll7",
        runner=runner,
        command_runner=command_runner,
    )

    assert [args[1] for args in calls[1:]]
    assert not any("/timeline?" in args[1] or "/labels?" in args[1] for args in calls[1:])
    quota = inventory["quota"]
    assert quota["status"] == "exhausted"
    assert quota["quota_exhausted"] is True
    assert quota["rate_limited"] is True
    assert quota["retry_after_utc"] == "2027-01-15T08:01:40Z"
    assert isinstance(quota["reset_in_seconds"], int)
    assert "Retry after reset with:" in quota["handoff"]
    assert quota["retry_command"] in quota["handoff"]
    plan = build_audit_plan(inventory)
    assert plan["classification_status"]["mutations_suppressed"] is True
    assert plan["mutations"] == []


def test_closure_evidence_preserves_targeted_timeline_provenance() -> None:
    """Closure evidence distinguishes targeted timeline coverage from global rows."""
    issue = _issue(110, body="Completion condition: merged PR #901")
    merged = [
        {
            "number": 901,
            "title": "Add audit coverage",
            "body": "",
            "merged_at": "2026-08-14T09:00:00Z",
            "linked_issue_numbers": [110],
            "coverage_source": "targeted_issue_timeline",
        }
    ]

    evidence = closure_evidence(issue, merged_prs=merged, open_issue_numbers={110})

    assert evidence["eligible"] is True
    assert evidence["merged_prs"] == [901]
    assert evidence["coverage_sources"] == ["targeted_issue_timeline"]
    assert evidence["targeted_merged_prs"] == [901]


def test_targeted_coverage_does_not_hide_global_history_truncation() -> None:
    """The plan exposes targeted coverage but apply remains fail-closed."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [_issue(110, body="Completion condition: merged PR #901")],
            "open_prs": [],
            "merged_prs": [
                {
                    "number": 901,
                    "title": "Add audit coverage",
                    "body": "",
                    "merged_at": "2026-08-14T09:00:00Z",
                    "linked_issue_numbers": [110],
                    "coverage_source": "targeted_issue_timeline",
                }
            ],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": [],
            "inventory": {
                "closed_prs": {"truncated": True, "errors": []},
                "issue_timeline_merged_prs": {
                    "available": True,
                    "truncated": False,
                    "errors": [],
                },
                "closure_coverage": {
                    "complete_for_open_issues": True,
                    "mode": "issue_timeline_fallback",
                },
            },
        }
    )

    assert plan["inventory_coverage"]["mode"] == "issue_timeline_fallback"
    assert plan["issues"][0]["closure_evidence"]["targeted_merged_prs"] == [901]
    assert "closed_prs" in plan["truncation_or_errors"]

    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError("truncated plan must not invoke mutation runner")

    result = apply_mutations(plan, runner=runner)
    assert result["ok"] is False
    assert result["applied"] == []
    assert "closed_prs" in result["failures"]
    assert calls == []


def test_command_timeouts_return_failures_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Command timeouts preserve fail-closed result handling for callers."""

    def raise_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    gh_result = _run_gh(["api", "repos/ll7/robot_sf_ll7/issues"])
    command_result = _run_command(["squeue", "--json"])

    assert gh_result.returncode == 124
    assert "timed out" in gh_result.stderr
    assert command_result.returncode == 124
    assert "timed out" in command_result.stderr


def test_run_gh_uses_remaining_budget_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    """The aggregate budget can shorten an individual gh subprocess timeout."""
    observed: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "[]", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _run_gh(["api", "repos/ll7/robot_sf_ll7/issues"], timeout_seconds=1.25)

    assert result.returncode == 0
    assert observed["timeout"] == 1.25


def test_deadline_runner_fails_closed_before_the_next_rest_call() -> None:
    """An exhausted audit budget returns a structured error without invoking the runner."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError("an exhausted budget must not invoke another REST call")

    bounded = issue_audit_core._deadline_runner(runner, 0)
    result = bounded(["api", "repos/ll7/robot_sf_ll7/issues"], None)

    assert result.returncode == 124
    assert "wall-time budget exhausted" in result.stderr
    assert calls == []


def test_production_runners_reject_results_returned_after_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production subprocess paths convert late successful results into timeouts."""
    clock_values = iter([0.0, 2.0, 0.0, 2.0])
    monkeypatch.setattr(issue_audit_core.time, "monotonic", lambda: next(clock_values))

    def complete_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(args[0], 0, "[]", "")

    monkeypatch.setattr(subprocess, "run", complete_run)

    rest_result = issue_audit_core._deadline_runner(
        issue_audit_core._run_gh,
        deadline=1.0,
    )(["api", "repos/ll7/robot_sf_ll7/issues"], None)
    command_result = issue_audit_core._deadline_command_runner(
        issue_audit_core._run_command,
        deadline=1.0,
    )(["squeue", "--json"])

    assert rest_result.returncode == 124
    assert command_result.returncode == 124
    assert "wall-time budget exhausted" in rest_result.stderr
    assert "wall-time budget exhausted" in command_result.stderr


@pytest.mark.skipif(
    not hasattr(issue_audit_core.signal, "SIGALRM")
    or not hasattr(issue_audit_core.signal, "setitimer"),
    reason="requires POSIX interval timers",
)
def test_slow_in_process_classification_is_interrupted_by_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A classifier that does not return cannot extend the CLI budget indefinitely."""

    def slow_classifier(*_args: Any, **_kwargs: Any) -> object:
        time.sleep(1.0)
        raise AssertionError("deadline interrupt should stop the classifier first")

    monkeypatch.setattr(issue_audit_core, "classify_issue", slow_classifier)
    deadline = time.monotonic() + 0.05
    started = time.monotonic()
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [_issue(206)],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": [],
            "inventory": {},
        },
        deadline=deadline,
    )

    assert time.monotonic() - started < 0.5
    assert plan["classification_status"]["status"] == "timed_out"
    assert plan["classification_status"]["classified_issues"] == 0
    assert plan["mutations"] == []
    assert all(row["mutations"] == [] for row in plan["issues"])


@pytest.mark.skipif(
    not hasattr(issue_audit_core.signal, "SIGALRM")
    or not hasattr(issue_audit_core.signal, "setitimer"),
    reason="requires POSIX interval timers",
)
def test_slow_in_process_discovery_emits_a_timeout_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A discovery phase that does not return cannot escape the CLI deadline."""

    def slow_discovery(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        time.sleep(1.0)
        raise AssertionError("deadline interrupt should stop discovery first")

    monkeypatch.setattr(issue_audit_core, "discover_inventory", slow_discovery)
    output = tmp_path / "issue-audit-plan.json"
    started = time.monotonic()

    result = main(["plan", "--max-wall-seconds", "0.05", "--output", str(output)])

    assert time.monotonic() - started < 0.5
    assert result == 2
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["classification_status"]["status"] == "timed_out"
    assert "during inventory discovery" in json.dumps(plan["inventory"])
    assert plan["mutations"] == []
    assert all(row["mutations"] == [] for row in plan["issues"])


def test_deadline_rejects_non_finite_budgets() -> None:
    """A non-finite CLI budget must not disable the aggregate timeout."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        issue_audit_core._deadline_from_seconds(float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        issue_audit_core._deadline_from_seconds(float("inf"))


def test_deadline_rejects_non_finite_absolute_deadlines() -> None:
    """An invalid shared absolute deadline must not admit a complete plan."""
    for deadline in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="deadline must be finite"):
            build_audit_plan({"issues": [], "inventory": {}}, deadline=deadline)


def test_plan_writes_fail_closed_artifact_when_wall_budget_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI callers receive a non-admitting plan artifact when REST budget is exhausted."""

    def fake_command(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(issue_audit_core, "_run_command", fake_command)
    output = tmp_path / "issue-audit-plan.json"

    result = main(
        [
            "plan",
            "--repo",
            "ll7/robot_sf_ll7",
            "--max-pages",
            "1",
            "--max-comment-pages",
            "1",
            "--max-wall-seconds",
            "0",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["mutations"] == []
    assert plan["truncation_or_errors"]
    assert "wall-time budget exhausted" in json.dumps(plan["inventory"])


def test_classification_timeout_is_explicit_and_suppresses_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial classification exposes a diagnostic cursor and cannot authorize writes."""
    issues = [
        _issue(
            201,
            body="## Definition of Done\n- [ ] implement the bounded change",
        ),
        _issue(
            202,
            body="## Definition of Done\n- [ ] implement the second bounded change",
        ),
    ]
    checks = 0

    def expire_after_classification(_deadline: float | None) -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(issue_audit_core, "_deadline_expired", expire_after_classification)

    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": issues,
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["state:ready"],
            "inventory": {},
        },
        deadline=10.0,
    )

    assert plan["classification_status"] == {
        "status": "timed_out",
        "reason": "issue-audit wall-time budget exhausted during issue classification",
        "classified_issues": 1,
        "total_issues": 2,
        "remaining_issue_numbers": [202],
        "resume_from_issue": 202,
        "resume_supported": False,
        "resume_requires_fresh_full_inventory": True,
        "mutations_suppressed": True,
    }
    assert plan["mutations"] == []
    assert all(row["mutations"] == [] for row in plan["issues"])
    assert plan["classification_status"]["resume_supported"] is False
    assert plan["classification_status"]["resume_requires_fresh_full_inventory"] is True
    assert "classification" in plan["truncation_or_errors"]


def test_final_classification_overrun_cannot_report_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline reached by the final classifier still produces a timeout plan."""
    checks = 0

    def expire_after_final_classification(_deadline: float | None) -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(issue_audit_core, "_deadline_expired", expire_after_final_classification)
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [_issue(204, body="## Definition of Done\n- [ ] implement the change")],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["state:ready"],
            "inventory": {},
        },
        deadline=10.0,
    )

    assert plan["classification_status"] == {
        "status": "timed_out",
        "reason": "issue-audit wall-time budget exhausted during issue classification",
        "classified_issues": 1,
        "total_issues": 1,
        "remaining_issue_numbers": [],
        "resume_from_issue": None,
        "resume_supported": False,
        "resume_requires_fresh_full_inventory": True,
        "mutations_suppressed": True,
    }
    assert plan["mutations"] == []
    assert all(row["mutations"] == [] for row in plan["issues"])


def test_final_merged_pr_record_crossing_deadline_discards_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline crossing after the last merged PR cannot return a usable index."""
    checks = 0

    def expire_after_final_record(_deadline: float | None) -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(issue_audit_core, "_deadline_expired", expire_after_final_record)
    index, timed_out = issue_audit_core._index_merged_prs(
        [
            {
                "number": 901,
                "title": "Implement audit support for #203",
                "body": "",
                "head_ref": "fix/issue-203-audit",
            }
        ],
        deadline=10.0,
    )

    assert timed_out is True
    assert index == {}


def test_plan_finalization_overrun_scrubs_all_mutation_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline crossed while finalizing a plan cannot leave a complete mutation artifact."""
    digest_completed = False
    original_digest = issue_audit_core.compute_plan_digest

    def digest(plan: dict[str, Any]) -> str:
        nonlocal digest_completed
        result = original_digest(plan)
        digest_completed = True
        return result

    monkeypatch.setattr(issue_audit_core, "compute_plan_digest", digest)
    monkeypatch.setattr(
        issue_audit_core,
        "_deadline_expired",
        lambda _deadline: digest_completed,
    )

    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [_issue(201, body="## Definition of Done\n- [ ] implement the change")],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["state:ready"],
            "inventory": {},
        },
        deadline=10.0,
    )

    assert plan["classification_status"]["status"] == "timed_out"
    assert "finalizing the audit plan" in plan["classification_status"]["reason"]
    assert plan["mutations"] == []
    assert all(row["mutations"] == [] for row in plan["issues"])


def test_plan_serialization_overrun_emits_the_scrubbed_timeout_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A deadline crossed by JSON rendering cannot be reported as a successful plan."""
    output_started = False
    original_dumps = issue_audit_core.json.dumps

    def dumps(value: object, *args: Any, **kwargs: Any) -> str:
        nonlocal output_started
        rendered = original_dumps(value, *args, **kwargs)
        if kwargs.get("indent") == 2:
            output_started = True
        return rendered

    monkeypatch.setattr(issue_audit_core.json, "dumps", dumps)
    monkeypatch.setattr(
        issue_audit_core,
        "_deadline_expired",
        lambda _deadline: output_started,
    )

    def discover(_repo: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "repo": "ll7/robot_sf_ll7",
            "issues": [_issue(201, body="## Definition of Done\n- [ ] implement the change")],
            "open_prs": [],
            "merged_prs": [],
            "labels": ["state:ready"],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "inventory": {},
        }

    monkeypatch.setattr(issue_audit_core, "discover_inventory", discover)
    output = tmp_path / "issue-audit-plan.json"

    result = main(["plan", "--max-wall-seconds", "30", "--output", str(output)])

    assert result == 2
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["classification_status"]["status"] == "timed_out"
    assert plan["mutations"] == []
    assert all(row["mutations"] == [] for row in plan["issues"])


def test_build_plan_indexes_merged_pr_references_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-issue classification consumes the shared merged-PR index."""
    issue = _issue(203, body="Completion condition: merged PR #901")
    merged_pr = {
        "number": 901,
        "title": "Implement audit support",
        "body": "",
        "head_ref": "fix/issue-203-audit",
        "merged_at": "2026-09-03T12:00:00Z",
        "linked_issue_numbers": [203],
    }

    def fail_if_scanned(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("build_audit_plan should use its merged-PR index")

    monkeypatch.setattr(issue_audit_core, "_merged_records", fail_if_scanned)

    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [issue],
            "open_prs": [],
            "merged_prs": [merged_pr],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": [],
            "inventory": {},
        }
    )

    assert plan["classification_status"]["status"] == "complete"
    assert plan["issues"][0]["closure_evidence"]["merged_prs"] == [901]


def test_plan_cli_shares_one_deadline_between_discovery_and_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI applies one aggregate deadline to both expensive phases."""
    observed: list[float | None] = []

    def discover(_repo: str, **kwargs: object) -> dict[str, object]:
        observed.append(kwargs["deadline"] if isinstance(kwargs["deadline"], float) else None)
        return {
            "repo": "ll7/robot_sf_ll7",
            "issues": [],
            "open_prs": [],
            "merged_prs": [],
            "labels": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "inventory": {},
        }

    original_build = issue_audit_core.build_audit_plan

    def build(inventory: dict[str, object], **kwargs: object) -> dict[str, object]:
        observed.append(kwargs["deadline"] if isinstance(kwargs["deadline"], float) else None)
        return original_build(inventory, **kwargs)

    monkeypatch.setattr(issue_audit_core, "discover_inventory", discover)
    monkeypatch.setattr(issue_audit_core, "build_audit_plan", build)
    output = tmp_path / "issue-audit-plan.json"

    result = main(["plan", "--max-wall-seconds", "30", "--output", str(output)])

    assert result == 0
    assert len(observed) == 2
    assert observed[0] is not None
    assert observed[0] == observed[1]


def test_plan_cli_accepts_a_separate_closed_pr_page_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Operators can raise closed-history coverage without widening every REST source."""
    observed: dict[str, object] = {}

    def discover(_repo: str, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "repo": "ll7/robot_sf_ll7",
            "issues": [],
            "open_prs": [],
            "merged_prs": [],
            "labels": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "inventory": {},
        }

    monkeypatch.setattr(issue_audit_core, "discover_inventory", discover)
    output = tmp_path / "issue-audit-plan.json"

    result = main(
        [
            "plan",
            "--max-pages",
            "2",
            "--max-closed-pr-pages",
            "47",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert observed["max_pages"] == 2
    assert observed["max_closed_pr_pages"] == 47


def test_decision_queue_is_machine_readable_and_project_free() -> None:
    """Pending decisions expose evidence while keeping Project #5 out of scope."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(
                    105,
                    labels=["decision-required"],
                    body="Maintainer decision required: choose option A or option B.",
                )
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["decision-required", "state:blocked"],
            "inventory": {},
        }
    )

    assert plan["schema"] == "issue_audit_plan.v1"
    assert plan["project5"] == {"writes": False, "owner": "gh-issue-sequencer"}
    assert len(plan["pending_decisions"]) == 1
    pending = plan["pending_decisions"][0]
    assert pending["issue"] == "#105"
    assert pending["decision_required"] is True
    assert pending["question_source"] == "issue body/comments"
    assert pending["blocking_evidence"].startswith(
        "decision-required label present; issue text: Maintainer decision"
    )
    assert pending["safe_mutations_applied"] == []


def test_decision_envelope_is_sorted_bound_to_plan_and_source_backed() -> None:
    """Envelopes are deterministic, digest-bound, and limited to source-backed options."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(
                    205,
                    labels=["decision-required"],
                    body="Maintainer decision required.\n(A) Keep the issue open.\n(B) Close it.",
                ),
                _issue(
                    204,
                    labels=["decision-required"],
                    body="Maintainer decision required.\n(A) Use the native path.\n(B) Use the adapter.",
                ),
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["decision-required"],
            "inventory": {},
        }
    )

    assert plan["plan_digest"] == compute_plan_digest(plan)
    selected = select_next_pending_decision(plan)
    assert selected is not None
    assert selected[0:2] == (0, 2)
    assert selected[2]["issue"] == "#204"
    scoped = select_next_pending_decision(plan, issue_scope=[205])
    assert scoped is not None
    assert scoped[0:2] == (0, 1)
    assert scoped[2]["issue"] == "#205"

    envelope = build_decision_envelope(plan)
    assert envelope is not None
    assert envelope["schema"] == "issue_decision_envelope.v1"
    assert envelope["status"] == "ready"
    assert envelope["queue"] == {
        "position": 1,
        "total": 2,
        "remaining_after": 1,
        "ordering": "issue_number_ascending",
    }
    assert envelope["issue"]["labels"] == ["decision-required"]
    assert [option["token"] for option in envelope["decision"]["documented_options"]] == [
        "A",
        "B",
    ]
    assert envelope["decision"]["evidence_sources"][0]["kind"] == "body"
    assert envelope["answer_contract"]["format"] == "#204: <option-token>"
    assert envelope["verification"]["project5_writes"] is False

    assert parse_decision_answer(envelope, "#204: A") == {"issue": "#204", "option": "A"}
    with pytest.raises(ValueError, match="not documented"):
        parse_decision_answer(envelope, "#204: C")


def test_documented_options_require_explicit_markers() -> None:
    """Bare list prose is not promoted to a maintainer choice token."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(
                    209,
                    labels=["decision-required"],
                    body=(
                        "Maintainer decision required.\n"
                        "- A: ordinary prose, not an option\n"
                        "B) ordinary prose, not an option\n"
                        "I. Introduction\n"
                        "Option A: use the native path.\n"
                        "Option B: use the adapter."
                    ),
                )
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["decision-required"],
            "inventory": {},
        }
    )

    envelope = build_decision_envelope(plan)
    assert envelope is not None
    assert envelope["status"] == "ready"
    assert [option["token"] for option in envelope["decision"]["documented_options"]] == [
        "A",
        "B",
    ]


def test_decision_envelope_rejects_stale_plan_and_live_issue_state() -> None:
    """Plan and live-label changes invalidate an envelope before answer application."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(
                    206,
                    labels=["decision-required"],
                    body="Maintainer decision required.\n(A) Keep open.\n(B) Close.",
                )
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["decision-required"],
            "inventory": {},
        }
    )
    envelope = build_decision_envelope(plan)
    assert envelope is not None
    assert validate_decision_envelope(
        envelope, plan=plan, live_issue=_issue(206, labels=["decision-required"])
    )["ok"]

    changed = json.loads(json.dumps(plan))
    changed["issues"][0]["title"] = "Changed after envelope"
    with pytest.raises(ValueError, match="stale"):
        build_decision_envelope(plan, expected_plan_digest=compute_plan_digest(changed))

    changed_issue = _issue(206, labels=["state:blocked", "decision-required"])
    validation = validate_decision_envelope(envelope, plan=plan, live_issue=changed_issue)
    assert validation["ok"] is False
    assert any("labels changed" in error for error in validation["errors"])


def test_decision_envelope_marks_undocumented_choices_and_incomplete_inventory() -> None:
    """Missing choices or inventory evidence fail closed instead of yielding policy."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(207, labels=["decision-required"], body="Maintainer decision required.")
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["decision-required"],
            "inventory": {"issues": {"errors": ["partial page"]}},
        }
    )
    envelope = build_decision_envelope(plan)
    assert envelope is not None
    assert envelope["status"] == "blocked_incomplete_inventory"
    assert envelope["answer_contract"]["allowed_tokens"] == []
    assert envelope["inventory_errors"]


def test_envelope_cli_emits_machine_readable_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The envelope CLI emits the versioned machine-readable handoff."""
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                _issue(
                    208,
                    labels=["decision-required"],
                    body="Maintainer decision required.\n(A) Keep open.\n(B) Close.",
                )
            ],
            "open_prs": [],
            "merged_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": ["decision-required"],
            "inventory": {},
        }
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    assert main(["envelope", str(plan_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "issue_decision_envelope.v1"
    assert payload["issue"]["number"] == 208


def test_closure_requires_documented_completion_condition() -> None:
    """Merged work closes an issue only with an explicit completion condition."""
    merged = [{"number": 901, "title": "Fixes #104", "merged_at": "2026-08-11T10:00:00Z"}]
    issue = _issue(104, body="## Definition of Done\n- [ ] verify the change")

    pending = closure_evidence(issue, merged_prs=merged, open_issue_numbers={104})
    assert pending["eligible"] is False
    assert "completion condition" in pending["reason"]

    issue["body"] = "Completion condition: merged PR #901"
    proven = closure_evidence(issue, merged_prs=merged, open_issue_numbers={104})
    assert proven["eligible"] is True
    assert proven["merged_prs"] == [901]


def test_parent_closure_requires_documented_condition_and_closed_children() -> None:
    """Parent closure requires its literal close condition and closed children."""
    merged = [{"number": 902, "title": "Fixes #105", "merged_at": "2026-08-11T10:00:00Z"}]
    issue = _issue(105, title="Parent roadmap", body="Child issue #106")

    assert closure_evidence(issue, merged_prs=merged, open_issue_numbers={105})["eligible"] is False

    issue["body"] = "Parent close condition: all linked children closed\nChild issue #106"
    assert closure_evidence(issue, merged_prs=merged, open_issue_numbers={105})["eligible"] is True


def test_label_endpoint_uri_escapes_colon() -> None:
    """Label deletion encodes colon-bearing names for the REST endpoint."""
    assert label_api_path("ll7/robot_sf_ll7", 106, "state:running") == (
        "repos/ll7/robot_sf_ll7/issues/106/labels/state%3Arunning"
    )


def test_apply_uses_encoded_delete_and_reads_back() -> None:
    """Mutation application uses URI-safe writes and verifies the live readback."""
    calls: list[tuple[list[str], str | None]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append((args, input_text))
        if args[:3] == ["api", "-X", "DELETE"]:
            return subprocess.CompletedProcess(args, 0, "{}", "")
        if args[:3] == ["api", "-X", "POST"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:2] == ["api", "repos/ll7/robot_sf_ll7/issues/106"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "state": "open",
                        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                        "labels": [{"name": "state:running"}],
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {args}")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "remove_label",
                "issue": 106,
                "value": "state:ready",
                "reason": "active work selects running",
                "evidence": ["open PR #903"],
                "expected_issue": _expected_issue(),
            },
            {
                "operation": "add_label",
                "issue": 106,
                "value": "state:running",
                "reason": "active work observed",
                "evidence": ["open PR #903"],
                "expected_issue": _expected_issue(),
            },
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)
    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is True
    assert [args for args, _ in calls if args[:3] == ["api", "-X", "DELETE"]] == [
        ["api", "-X", "DELETE", "repos/ll7/robot_sf_ll7/issues/106/labels/state%3Aready"]
    ]
    assert len(result["readback"]) == 1
    readback = result["readback"][0]
    assert readback["issue"] == 106
    assert readback["ok"] is True
    assert readback["state"] == "open"
    assert readback["labels"] == ["state:running"]
    assert readback["verified"]["missing_additions"] == []
    assert readback["verified"]["missing_removals"] == []


def test_apply_treats_absent_label_delete_as_idempotent() -> None:
    """An already-absent label is not a failed mutation and is read back."""
    calls: list[tuple[list[str], str | None]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append((args, input_text))
        if args[:3] == ["api", "-X", "DELETE"]:
            return subprocess.CompletedProcess(
                args,
                1,
                "",
                "gh: Label does not exist (HTTP 404)",
            )
        if args[:2] == ["api", "repos/ll7/robot_sf_ll7/issues/107"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "state": "open",
                        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                        "labels": [{"name": "state:running"}],
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {args}")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "remove_label",
                "issue": 107,
                "value": "state:ready",
                "reason": "active work selects running",
                "evidence": ["open PR #904"],
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is True
    assert result["applied"] == []
    assert result["already_applied"][0]["skipped_reason"] == "already_absent"
    assert result["failures"] == []
    assert result["counts"] == {
        "planned": 1,
        "applied": 0,
        "already_applied": 1,
        "failed": 0,
        "stale_state_issues": 0,
        "skipped_stale_mutations": 0,
    }
    assert result["readback"][0]["verified"]["missing_removals"] == []
    assert len(calls) == 3


def test_apply_keeps_unrelated_label_404_as_failure() -> None:
    """An issue or endpoint 404 must not be mistaken for an absent label."""

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["api", "repos/ll7/robot_sf_ll7/issues/108"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "state": "open",
                        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                        "labels": [{"name": "state:ready"}],
                    }
                ),
                "",
            )
        assert args[:3] == ["api", "-X", "DELETE"]
        return subprocess.CompletedProcess(args, 1, "", "gh: Not Found (HTTP 404)")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "remove_label",
                "issue": 108,
                "value": "state:ready",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["already_applied"] == []
    assert len(result["failures"]) == 1
    assert result["counts"] == {
        "planned": 1,
        "applied": 0,
        "already_applied": 0,
        "failed": 1,
        "stale_state_issues": 0,
        "skipped_stale_mutations": 0,
    }


@pytest.mark.parametrize(
    "observed_issue",
    [
        {"state": "closed", "updated_at": "2026-08-23T00:01:00Z"},
        {"state": "open", "updated_at": "2026-08-23T00:01:00Z"},
    ],
)
def test_apply_skips_entire_issue_batch_when_state_or_version_is_stale(
    observed_issue: dict[str, str],
) -> None:
    """A stale live issue produces zero writes for every mutation in its batch."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert args == ["api", "repos/ll7/robot_sf_ll7/issues/109"]
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps({**observed_issue, "labels": [{"name": "state:ready"}]}),
            "",
        )

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "remove_label",
                "issue": 109,
                "value": "state:ready",
                "expected_issue": _expected_issue(),
            },
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            },
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert result["already_applied"] == []
    assert result["stale_states"] == [
        {
            "issue": 109,
            "disposition": "stale_state",
            "expected_issue": _expected_issue(),
            "observed_issue": observed_issue,
            "skipped_mutations": 2,
        }
    ]
    assert result["failures"] == result["stale_states"]
    assert result["counts"] == {
        "planned": 2,
        "applied": 0,
        "already_applied": 0,
        "failed": 1,
        "stale_state_issues": 1,
        "skipped_stale_mutations": 2,
    }
    assert calls == [["api", "repos/ll7/robot_sf_ll7/issues/109"]]


def _expected_issue_with_labels(labels: list[str], state: str = "open") -> dict[str, object]:
    """Return a label-carrying plan-time snapshot for drift-tolerance tests."""
    return {**_expected_issue(state), "labels": sorted(labels)}


def test_apply_proceeds_on_timestamp_only_drift_with_matching_labels() -> None:
    """A comment-only updated_at bump must not block an unchanged label repair (issue #8295)."""
    calls: list[list[str]] = []
    live_labels = [{"name": "state:ready"}]

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["api", "repos/ll7/robot_sf_ll7/issues/109"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "state": "open",
                        # Periodic automation advanced the clock without touching labels.
                        "updated_at": "2026-08-23T00:05:00Z",
                        "labels": list(live_labels),
                    }
                ),
                "",
            )
        assert args[:3] == ["api", "-X", "POST"]
        live_labels.append({"name": "state:running"})
        return subprocess.CompletedProcess(args, 0, "{}", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue_with_labels(["state:ready"]),
            },
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is True
    assert result["stale_states"] == []
    assert result["failures"] == []
    assert len(result["applied"]) == 1
    assert result["timestamp_drift_bypassed"] == [
        {
            "issue": 109,
            "expected_updated_at": EXPECTED_ISSUE_UPDATED_AT,
            "observed_updated_at": "2026-08-23T00:05:00Z",
        }
    ]
    assert result["counts"]["applied"] == 1
    assert result["counts"]["stale_state_issues"] == 0


def test_apply_blocks_on_label_drift_with_retry_handoff() -> None:
    """A concurrent label change stays fail-closed stale with a regenerate handoff."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert args == ["api", "repos/ll7/robot_sf_ll7/issues/109"]
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {
                    "state": "open",
                    "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                    "labels": [{"name": "state:ready"}, {"name": "needs-triage"}],
                }
            ),
            "",
        )

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "remove_label",
                "issue": 109,
                "value": "state:ready",
                "expected_issue": _expected_issue_with_labels(["state:ready"]),
            },
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert calls == [["api", "repos/ll7/robot_sf_ll7/issues/109"]]
    (stale,) = result["stale_states"]
    assert stale["disposition"] == "stale_state"
    assert stale["drift_kind"] == "semantic"
    assert stale["expected_issue"]["labels"] == ["state:ready"]
    assert stale["observed_issue"]["labels"] == ["needs-triage", "state:ready"]
    assert stale["retry"]["action"] == "regenerate_plan"
    assert result["timestamp_drift_bypassed"] == []


def test_apply_blocks_on_state_drift_with_labels() -> None:
    """A concurrent state change stays stale even when labels match."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert args == ["api", "repos/ll7/robot_sf_ll7/issues/109"]
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {
                    "state": "closed",
                    "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                    "labels": [{"name": "state:ready"}],
                }
            ),
            "",
        )

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "remove_label",
                "issue": 109,
                "value": "state:ready",
                "expected_issue": _expected_issue_with_labels(["state:ready"]),
            },
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert calls == [["api", "repos/ll7/robot_sf_ll7/issues/109"]]
    (stale,) = result["stale_states"]
    assert stale["drift_kind"] == "semantic"


def test_apply_rejects_malformed_expected_labels_before_reads() -> None:
    """A non-list label snapshot is a plan defect, not a live-state question."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError("invalid plan must fail before any REST call")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": {**_expected_issue(), "labels": "state:ready"},
            }
        ],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert calls == []
    assert any("string list" in str(failure) for failure in result["failures"])


def test_apply_rejects_missing_state_version_precondition_before_reads() -> None:
    """A hand-built plan cannot omit the plan-time state/version binding."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError("invalid plan must fail before any REST call")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [{"operation": "add_label", "issue": 109, "value": "state:running"}],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert "preconditions" in result["reason"]
    assert result["applied"] == []
    assert result["counts"] == {
        "planned": 1,
        "applied": 0,
        "already_applied": 0,
        "failed": 1,
        "stale_state_issues": 0,
        "skipped_stale_mutations": 0,
    }
    assert calls == []


@pytest.mark.parametrize(
    "bad_mutation",
    [
        "not-an-object",
        {"operation": "add_label", "issue": "bad", "value": "state:ready"},
        {
            "operation": "add_label",
            "issue": True,
            "value": "state:ready",
            "expected_issue": _expected_issue(),
        },
        {
            "operation": "add_label",
            "issue": 1.9,
            "value": "state:ready",
            "expected_issue": _expected_issue(),
        },
        {
            "operation": "add_label",
            "issue": "001",
            "value": "state:ready",
            "expected_issue": _expected_issue(),
        },
        {
            "operation": "unsupported",
            "issue": 109,
            "expected_issue": _expected_issue(),
        },
        {
            "operation": "add_label",
            "issue": 109,
            "value": "",
            "expected_issue": _expected_issue(),
        },
        {
            "operation": "close_issue",
            "issue": 109,
            "value": "closed",
            "expected_issue": _expected_issue(),
        },
    ],
)
def test_apply_rejects_mixed_invalid_and_valid_plan_before_any_rest_call(
    bad_mutation: object,
) -> None:
    """One malformed row prevents a sibling valid row from writing."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError("mixed invalid plan must fail before any REST call")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            bad_mutation,
            {
                "operation": "add_label",
                "issue": 110,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            },
        ],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert result["counts"]["planned"] == 2
    assert result["counts"]["applied"] == 0
    assert result["counts"]["failed"] == 1
    assert calls == []


@pytest.mark.parametrize(
    ("plan", "max_mutations", "reason", "planned"),
    [
        ({"schema": "wrong", "mutations": []}, 10, "expected issue_audit_plan.v1", 0),
        ({"schema": "issue_audit_plan.v1", "mutations": {}}, 10, "must be a list", 0),
        (
            {
                "schema": "issue_audit_plan.v1",
                "mutations": [],
                "truncation_or_errors": 1,
            },
            10,
            "truncation_or_errors must be a list",
            0,
        ),
        (
            {
                "schema": "issue_audit_plan.v1",
                "mutations": [
                    {"operation": "close_issue", "issue": 1, "value": None},
                    {"operation": "close_issue", "issue": 2, "value": None},
                ],
            },
            1,
            "exceeds mutation budget",
            2,
        ),
    ],
)
def test_apply_contract_refusals_return_stable_counts_without_rest_calls(
    plan: dict[str, object],
    max_mutations: int,
    reason: str,
    planned: int,
) -> None:
    """Top-level plan contract failures use the stable no-write result shape."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise AssertionError("contract refusal must happen before any REST call")

    result = apply_mutations(plan, max_mutations=max_mutations, runner=runner)

    assert result["ok"] is False
    assert reason in result["reason"]
    assert result["applied"] == []
    assert result["counts"] == {
        "planned": planned,
        "applied": 0,
        "already_applied": 0,
        "failed": 1,
        "stale_state_issues": 0,
        "skipped_stale_mutations": 0,
    }
    assert calls == []


def test_apply_readback_rejects_state_change_after_label_write() -> None:
    """Readback verifies issue state in addition to the requested label effect."""
    issue_reads = 0

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        nonlocal issue_reads
        if args[:3] == ["api", "-X", "POST"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args == ["api", "repos/ll7/robot_sf_ll7/issues/109"]:
            issue_reads += 1
            state = "open" if issue_reads == 1 else "closed"
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "state": state,
                        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                        "labels": [{"name": "state:running"}],
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {args}")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"][0]["value"] == "state:running"
    assert result["readback"][0]["ok"] is False
    assert result["readback"][0]["verified"]["expected_state"] == "open"
    assert result["readback"][0]["verified"]["state_matches"] is False


def test_incomplete_plan_fails_closed_before_mutation() -> None:
    """Incomplete plans are rejected before any mutation can be attempted."""
    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [],
        "truncation_or_errors": ["issues"],
    }
    plan["plan_digest"] = compute_plan_digest(plan)
    result = apply_mutations(plan)

    assert result["ok"] is False
    assert result["applied"] == []


def test_apply_rejects_stale_plan_digest_before_mutation() -> None:
    """Edited plan files cannot reach the GitHub mutation runner."""
    calls: list[tuple[list[str], str | None]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append((args, input_text))
        raise AssertionError("stale plan must be rejected before invoking the runner")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)
    plan["mutations"].append(
        {
            "operation": "add_label",
            "issue": 110,
            "value": "state:blocked",
        }
    )

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert "stale" in result["reason"]
    assert calls == []


def test_apply_rejects_unreasoned_blocked_label_before_mutation() -> None:
    """A hand-edited plan cannot bypass the blocked-label reason guard."""
    calls: list[tuple[list[str], str | None]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append((args, input_text))
        raise AssertionError("unreasoned blocked-label plan must be rejected before writes")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 124,
                "value": "state:blocked",
                "reason": "prose-only blocker",
                "evidence": ["issue text says blocked"],
            }
        ],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert "unreasoned blocked-label" in result["reason"]
    assert calls == []


def test_apply_accepts_reasoned_blocked_label() -> None:
    """A planner-bound reason permits the blocked-label write and readback."""
    calls: list[tuple[list[str], str | None]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append((args, input_text))
        if args[:3] == ["api", "-X", "POST"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:2] == ["api", "repos/ll7/robot_sf_ll7/issues/126"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "state": "open",
                        "updated_at": EXPECTED_ISSUE_UPDATED_AT,
                        "labels": [{"name": "state:blocked"}],
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {args}")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 126,
                "value": "state:blocked",
                "reason": "record a proven blocker",
                "evidence": ["blocked evidence"],
                "blocked_reason": ["Blocked-by reference present: Blocked-by: #902"],
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)

    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is True
    assert result["readback"][0]["verified"]["missing_additions"] == []
    assert len(calls) == 3


def test_pending_queue_can_record_readback_confirmed_safe_mutations() -> None:
    """The queue carries autonomous mutations confirmed by the apply readback."""
    queue = build_pending_decision_queue(
        {
            "pending_decisions": [
                {
                    "issue": "#107",
                    "decision_required": True,
                    "question_source": "issue body/comments",
                    "blocking_evidence": "maintainer choice",
                    "safe_mutations_applied": [],
                }
            ]
        },
        applied_mutations=[
            {
                "operation": "add_label",
                "issue": 107,
                "value": "state:blocked",
                "reason": "proven blocker",
            }
        ],
    )

    assert queue[0]["safe_mutations_applied"][0]["value"] == "state:blocked"


def test_inventory_quota_exhausted_skips_closed_and_timeline_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core quota exhaustion preflight skips closed-PR and timeline pagination."""
    observed_calls: list[str] = []

    monkeypatch.setattr(
        issue_audit_core,
        "discover_open_issues",
        lambda *args, **kwargs: ([_issue(101)], {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_pull_requests",
        lambda _repo, *, state, max_pages, runner, request_budget: (
            observed_calls.append(f"prs_{state}") or ([], {"truncated": False, "errors": []})
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_issue_timeline_merged_prs",
        lambda *args, **kwargs: (
            observed_calls.append("timeline") or ([], {"available": True, "errors": []})
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_core_quota",
        lambda *args, **kwargs: (
            issue_audit_core.RateLimitSnapshot(
                status="ok",
                graphql_remaining=500,
                graphql_reset_at=1_800_000_000,
                core_remaining=0,
                core_reset_at=1_800_000_100,
            ),
            {
                "available": False,
                "status": "exhausted",
                "core_remaining": 0,
                "core_reset_at": 1_800_000_100,
                "reset_in_seconds": 100,
                "retry_after_utc": "2027-01-15T08:01:40Z",
                "retry_command": "uv run python scripts/dev/issue_audit_core.py plan --repo ll7/robot_sf_ll7",
                "next_action": "wait_for_reset_and_retry",
                "handoff": "GitHub core REST quota exhausted...",
                "reason": "GitHub core REST quota exhausted (0 remaining <= safety margin 10)",
                "errors": ["GitHub core REST quota exhausted (0 remaining)"],
                "quota_exhausted": True,
            },
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_repository_labels",
        lambda *args, **kwargs: (set(), {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    inventory = issue_audit_core.discover_inventory("ll7/robot_sf_ll7")

    # The quota preflight runs before every high-volume REST collection.
    assert observed_calls == []
    assert inventory["issues"] == []
    assert inventory["inventory"]["closed_prs"]["quota_exhausted"] is True
    assert inventory["inventory"]["closure_coverage"]["complete_for_open_issues"] is False
    assert inventory["inventory"]["closure_coverage"]["mode"] == "incomplete_quota_exhausted"

    # Now verify build_audit_plan fail-closed mutation and uncertainty behavior
    plan = build_audit_plan(inventory)
    assert "core_quota" in plan["inventory_uncertainties"]
    assert "core_quota" in plan["truncation_or_errors"]
    assert plan["mutations"] == []
    assert plan["quota"]["status"] == "exhausted"
    assert plan["quota"]["next_action"] == "wait_for_reset_and_retry"
    # Ensure no readiness or blocked label write is planned on any issue
    for issue_entry in plan["issues"]:
        assert issue_entry["mutations"] == []


def test_inventory_bounds_requests_when_core_quota_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low core budget bounds closed-PR pagination and skips timeline when budget runs out."""
    observed_closed_pages: list[int] = []
    timeline_called = False

    monkeypatch.setattr(
        issue_audit_core,
        "discover_open_issues",
        lambda *args, **kwargs: ([_issue(101)], {"truncated": False, "errors": []}),
    )

    def mock_prs(
        _repo: str,
        *,
        state: str,
        max_pages: int,
        runner: object,
        request_budget: object,
    ) -> tuple:
        if state == "closed":
            observed_closed_pages.append(max_pages)
            # Simulates reading all budgeted pages, returning truncated
            return [], {"truncated": True, "pages_read": max_pages, "errors": []}
        return [], {"truncated": False, "pages_read": 1, "errors": []}

    monkeypatch.setattr(issue_audit_core, "discover_pull_requests", mock_prs)

    def mock_timeline(*args, **kwargs) -> tuple:
        nonlocal timeline_called
        timeline_called = True
        return [], {"available": True, "errors": []}

    monkeypatch.setattr(issue_audit_core, "discover_issue_timeline_merged_prs", mock_timeline)
    monkeypatch.setattr(
        issue_audit_core,
        "discover_core_quota",
        lambda *args, **kwargs: (
            issue_audit_core.RateLimitSnapshot(
                status="ok",
                graphql_remaining=500,
                graphql_reset_at=1_800_000_000,
                core_remaining=13,
                core_reset_at=1_800_000_100,
            ),
            {
                "available": True,
                "status": "ok",
                "core_remaining": 13,
                "core_reset_at": 1_800_000_100,
                "available_budget": 3,  # 13 - 10 = 3
                "min_core_remaining": 10,
                "retry_command": "cmd",
                "next_action": "none",
                "reason": "sufficient core quota available",
                "errors": [],
                "quota_exhausted": False,
                "quota_uncertain": False,
                "budget_exhausted": False,
            },
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_repository_labels",
        lambda *args, **kwargs: (set(), {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    inventory = issue_audit_core.discover_inventory("ll7/robot_sf_ll7", max_closed_pr_pages=50)

    # Closed PRs budget was capped at available_budget (3), not 50
    assert observed_closed_pages == [3]
    # Because 3 pages were consumed, remaining budget is 0, so timeline was skipped
    assert timeline_called is False
    assert inventory["inventory"]["issue_timeline_merged_prs"]["budget_exhausted"] is True
    assert inventory["inventory"]["issue_timeline_merged_prs"].get("quota_exhausted") is not True
    assert inventory["inventory"]["closure_coverage"]["complete_for_open_issues"] is False


def test_timeline_pagination_breaks_early_on_403_rate_limit() -> None:
    """Timeline discovery bails out immediately on rate-limit error without querying all issues."""
    queried_issues: list[int] = []

    def runner(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        for part in args:
            if "issues/" in part:
                num = int(part.split("issues/")[1].split("/")[0])
                queried_issues.append(num)
                if num == 2:
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        "gh: HTTP 403: API rate limit exceeded for user (rate limit exceeded)",
                    )
                return subprocess.CompletedProcess(args, 0, "[]")
        return subprocess.CompletedProcess(args, 0, "[]")

    _prs, meta = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [1, 2, 3, 4, 5],
        runner=runner,
    )

    # Stops immediately after issue 2 hits rate limit; 3, 4, 5 are never queried
    assert queried_issues == [1, 2]
    assert meta["truncated"] is True
    assert meta["available"] is False
    assert meta["quota_exhausted"] is True
    assert meta["rate_limited"] is True


def test_recovered_core_quota_enables_complete_audit_without_leaked_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subsequent audit with recovered quota permits normal completion and zero residual uncertainty."""
    monkeypatch.setattr(
        issue_audit_core,
        "discover_open_issues",
        lambda *args, **kwargs: (
            [_issue(101, body="Completion condition: merged PR #999")],
            {"truncated": False, "errors": []},
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_pull_requests",
        lambda _repo, *, state, max_pages, runner, request_budget: (
            ([], {"truncated": False, "errors": []})
            if state == "open"
            else (
                [
                    {
                        "number": 999,
                        "title": "Fix #101",
                        "body": "Closes #101",
                        "state": "closed",
                        "html_url": "https://github.com/ll7/robot_sf_ll7/pull/999",
                        "merged_at": "2026-08-20T10:00:00Z",
                        "head_ref": "",
                    }
                ],
                {"truncated": False, "errors": [], "pages_read": 1},
            )
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_core_quota",
        lambda *args, **kwargs: (
            issue_audit_core.RateLimitSnapshot(
                status="ok",
                graphql_remaining=500,
                graphql_reset_at=1_800_000_000,
                core_remaining=500,
                core_reset_at=1_800_000_100,
            ),
            {
                "available": True,
                "status": "ok",
                "core_remaining": 500,
                "core_reset_at": 1_800_000_100,
                "available_budget": 490,
                "min_core_remaining": 10,
                "retry_command": "cmd",
                "next_action": "none",
                "reason": "sufficient core quota available",
                "errors": [],
                "quota_exhausted": False,
                "quota_uncertain": False,
                "budget_exhausted": False,
            },
        ),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_repository_labels",
        lambda *args, **kwargs: (set(), {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    inventory = issue_audit_core.discover_inventory("ll7/robot_sf_ll7")
    assert inventory["inventory"]["closure_coverage"]["complete_for_open_issues"] is True
    assert inventory["inventory"]["closure_coverage"]["mode"] == "global_closed_prs"

    plan = build_audit_plan(inventory)
    # Recovered quota leaves no stale uncertainty or truncation
    assert "core_quota" not in plan["inventory_uncertainties"]
    assert "core_quota" not in plan["truncation_or_errors"]
    assert plan["truncation_or_errors"] == []
    assert plan["quota"]["status"] == "ok"
    assert plan["quota"]["next_action"] == "none"


def test_main_plan_cli_reports_concise_quota_warning_and_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The plan CLI prints a concise quota warning to stderr and exits 2 when quota is exhausted."""
    plan_file = tmp_path / "plan.json"
    exhausted_inventory = {
        "repo": "ll7/robot_sf_ll7",
        "issues": [_issue(101)],
        "open_prs": [],
        "merged_prs": [],
        "labels": [],
        "claims": {},
        "worktrees": [],
        "jobs": [],
        "quota": {
            "status": "exhausted",
            "quota_exhausted": True,
            "core_remaining": 0,
            "reset_in_seconds": 120,
            "retry_after_utc": "2026-09-02T16:00:00Z",
            "retry_command": "uv run python scripts/dev/issue_audit_core.py plan --mode autonomous",
            "reason": "GitHub core REST quota exhausted (0 remaining <= safety margin 10)",
        },
        "inventory": {
            "quota": {
                "status": "exhausted",
                "quota_exhausted": True,
                "core_remaining": 0,
                "errors": ["quota exhausted"],
            },
            "closure_coverage": {"complete_for_open_issues": False},
        },
    }
    monkeypatch.setattr(
        issue_audit_core, "discover_inventory", lambda *args, **kwargs: exhausted_inventory
    )

    code = main(
        [
            "plan",
            "--repo",
            "ll7/robot_sf_ll7",
            "--output",
            str(plan_file),
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "issue-audit: GitHub core REST quota exhausted" in captured.err
    assert "Reset at 2026-09-02T16:00:00Z in ~120s" in captured.err
    assert (
        "Next action: uv run python scripts/dev/issue_audit_core.py plan --mode autonomous"
        in captured.err
    )

    # File was written with structured quota information and zero mutations
    saved_plan = json.loads(plan_file.read_text(encoding="utf-8"))
    assert saved_plan["quota"]["status"] == "exhausted"
    assert saved_plan["mutations"] == []
    assert "core_quota" in saved_plan["inventory_uncertainties"]
    assert "core_quota" in saved_plan["truncation_or_errors"]


def test_timeline_max_requests_counts_failed_requests_against_budget() -> None:
    """A failed timeline request with finite max_requests must not query the next issue.

    Regression for issue #8509: pages_read was used instead of requests_attempted,
    allowing repeated failures to bypass the budget.
    """
    queried_issues: list[int] = []

    def runner(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        for part in args:
            if "issues/" in part and "/timeline" in part:
                num = int(part.split("issues/")[1].split("/")[0])
                queried_issues.append(num)
                if num == 1:
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        "gh: HTTP 500: timeline request failed",
                    )
                return subprocess.CompletedProcess(args, 0, "[]")
        return subprocess.CompletedProcess(args, 0, "[]")

    _prs, meta = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [1, 2, 3],
        max_requests=1,
        runner=runner,
    )

    # With max_requests=1, only issue 1 should be attempted; the failed request
    # must count against the budget so issue 2 is never queried.
    assert queried_issues == [1]
    assert meta["truncated"] is True
    assert meta["requests_attempted"] == 1
    assert meta["pages_read"] == 0


def test_timeline_max_requests_caps_full_page_pagination() -> None:
    """A full page must not allow timeline pagination past the direct request limit."""
    queried_endpoints: list[str] = []

    def runner(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        endpoint = next(part for part in args if "issues/" in part and "/timeline" in part)
        queried_endpoints.append(endpoint)
        if "page=1" in endpoint:
            return subprocess.CompletedProcess(
                args,
                0,
                "[" + ",".join('{"event":"other"}' for _ in range(100)) + "]",
            )
        return subprocess.CompletedProcess(args, 0, "[]")

    _prs, meta = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [1, 2],
        max_pages=3,
        max_requests=1,
        runner=runner,
    )

    assert queried_endpoints == ["repos/ll7/robot_sf_ll7/issues/1/timeline?per_page=100&page=1"]
    assert meta["requests_attempted"] == 1
    assert meta["pages_read"] == 1
    assert meta["truncated"] is True
    assert meta["errors"]


def test_timeline_max_requests_is_global_across_issue_pages() -> None:
    """A short first issue must not reset the request limit for later issues."""
    queried_endpoints: list[str] = []

    def runner(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        endpoint = next(part for part in args if "issues/" in part and "/timeline" in part)
        queried_endpoints.append(endpoint)
        if "issues/1/" in endpoint:
            return subprocess.CompletedProcess(args, 0, "[]")
        return subprocess.CompletedProcess(
            args,
            0,
            "[" + ",".join('{"event":"other"}' for _ in range(100)) + "]",
        )

    _prs, meta = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [1, 2],
        max_pages=3,
        max_requests=2,
        runner=runner,
    )

    assert queried_endpoints == [
        "repos/ll7/robot_sf_ll7/issues/1/timeline?per_page=100&page=1",
        "repos/ll7/robot_sf_ll7/issues/2/timeline?per_page=100&page=1",
    ]
    assert meta["requests_attempted"] == 2
    assert meta["pages_read"] == 2
    assert meta["truncated"] is True
    assert meta["errors"]


def test_timeline_direct_limit_marks_exhausted_shared_budget() -> None:
    """A full final page must publish shared-budget exhaustion at the direct limit."""
    request_budget = issue_audit_core._RestRequestBudget.from_available(2)

    def runner(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        endpoint = next(part for part in args if "issues/" in part and "/timeline" in part)
        if "issues/1/" in endpoint:
            return subprocess.CompletedProcess(args, 0, "[]")
        return subprocess.CompletedProcess(
            args,
            0,
            "[" + ",".join('{"event":"other"}' for _ in range(100)) + "]",
        )

    _prs, meta = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [1, 2],
        max_pages=3,
        max_requests=request_budget.remaining,
        runner=runner,
        request_budget=request_budget,
    )

    assert meta["requests_attempted"] == 2
    assert meta["pages_read"] == 2
    assert meta["truncated"] is True
    assert meta["request_limit_exhausted"] is True
    assert meta["budget_exhausted"] is True
    assert request_budget.remaining == 0
    assert request_budget.budget_exhausted is True


def test_timeline_success_path_counts_attempts_consistently() -> None:
    """Successful timeline requests also count against the budget consistently."""
    queried_issues: list[int] = []

    def runner(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        for part in args:
            if "issues/" in part and "/timeline" in part:
                num = int(part.split("issues/")[1].split("/")[0])
                queried_issues.append(num)
                return subprocess.CompletedProcess(args, 0, "[]")
        return subprocess.CompletedProcess(args, 0, "[]")

    _prs, meta = discover_issue_timeline_merged_prs(
        "ll7/robot_sf_ll7",
        [1, 2, 3],
        max_requests=2,
        runner=runner,
    )

    # With max_requests=2 and successful requests, exactly two issues are queried.
    assert queried_issues == [1, 2]
    assert meta["requests_attempted"] == 2
    assert meta["pages_read"] == 2
    assert meta["request_limit_exhausted"] is True


def test_plan_cli_retry_command_preserves_non_default_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The quota retry command preserves every bounded plan CLI option and shell-quotes values.

    Regression for issue #8509: the retry command was repo-only, losing
    --remote, --include-comments, --max-pages, --max-wall-seconds, etc.
    """
    observed_retry_command: str | None = None

    def mock_discover_inventory(
        repo: str, *, retry_command: str | None = None, **kwargs: object
    ) -> dict[str, object]:
        nonlocal observed_retry_command
        observed_retry_command = retry_command
        return {
            "repo": repo,
            "issues": [],
            "open_prs": [],
            "merged_prs": [],
            "labels": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "quota": {
                "status": "exhausted",
                "quota_exhausted": True,
                "core_remaining": 0,
                "core_reset_at": 1_800_000_100,
                "reset_in_seconds": 120,
                "retry_after_utc": "2027-01-15T08:01:40Z",
                "retry_command": "fallback",
                "reason": "GitHub core REST quota exhausted",
                "errors": ["quota exhausted"],
            },
            "inventory": {
                "quota": {
                    "status": "exhausted",
                    "quota_exhausted": True,
                    "errors": ["quota exhausted"],
                },
                "closure_coverage": {"complete_for_open_issues": False},
            },
        }

    monkeypatch.setattr(issue_audit_core, "discover_inventory", mock_discover_inventory)
    output = tmp_path / "plan with spaces.json"
    repo = "owner/repo; touch pwned"

    code = main(
        [
            "plan",
            "--repo",
            repo,
            "--remote",
            "upstream",
            "--mode",
            "interactive",
            "--include-comments",
            "--max-pages",
            "3",
            "--max-closed-pr-pages",
            "20",
            "--max-comment-pages",
            "2",
            "--max-mutations",
            "7",
            "--max-wall-seconds",
            "60",
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert observed_retry_command is not None
    assert shlex.split(observed_retry_command) == [
        "uv",
        "run",
        "python",
        "scripts/dev/issue_audit_core.py",
        "plan",
        "--repo",
        repo,
        "--remote",
        "upstream",
        "--mode",
        "interactive",
        "--max-pages",
        "3",
        "--max-closed-pr-pages",
        "20",
        "--include-comments",
        "--max-comment-pages",
        "2",
        "--max-mutations",
        "7",
        "--max-wall-seconds",
        "60.0",
        "--output",
        str(output),
    ]
    assert output.exists()


def test_inventory_retry_command_preserves_inventory_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct inventory callers receive a retry handoff with their bounded options intact."""
    observed_retry_command: str | None = None

    def mock_discover_core_quota(
        repo: str, *, runner: object = None, retry_command: str | None = None
    ) -> tuple:
        nonlocal observed_retry_command
        observed_retry_command = retry_command
        return (
            issue_audit_core.RateLimitSnapshot(
                status="failed",
                error="quota preflight unavailable",
            ),
            {
                "available": False,
                "status": "failed",
                "retry_command": retry_command,
                "next_action": "retry_preflight",
                "reason": "quota preflight unavailable",
                "errors": ["quota preflight unavailable"],
                "quota_exhausted": False,
                "quota_uncertain": True,
            },
        )

    monkeypatch.setattr(issue_audit_core, "discover_core_quota", mock_discover_core_quota)
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    issue_audit_core.discover_inventory(
        "ll7/robot_sf_ll7",
        remote="upstream",
        max_pages=3,
        max_closed_pr_pages=20,
        include_comments=True,
        max_comment_pages=2,
        max_wall_seconds=60,
    )

    assert observed_retry_command is not None
    assert shlex.split(observed_retry_command) == [
        "uv",
        "run",
        "python",
        "scripts/dev/issue_audit_core.py",
        "plan",
        "--repo",
        "ll7/robot_sf_ll7",
        "--remote",
        "upstream",
        "--mode",
        "autonomous",
        "--max-pages",
        "3",
        "--max-closed-pr-pages",
        "20",
        "--include-comments",
        "--max-comment-pages",
        "2",
        "--max-mutations",
        str(issue_audit_core.DEFAULT_MAX_MUTATIONS),
        "--max-wall-seconds",
        "60",
    ]


def test_inventory_uses_explicit_retry_command_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers that already have a complete retry command keep it unchanged."""
    observed_retry_command: str | None = None

    def mock_discover_core_quota(
        repo: str, *, runner: object = None, retry_command: str | None = None
    ) -> tuple:
        nonlocal observed_retry_command
        observed_retry_command = retry_command
        return _healthy_quota_meta()

    monkeypatch.setattr(issue_audit_core, "discover_core_quota", mock_discover_core_quota)
    monkeypatch.setattr(
        issue_audit_core,
        "discover_open_issues",
        lambda *args, **kwargs: ([], {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_pull_requests",
        lambda *args, **kwargs: ([], {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_repository_labels",
        lambda *args, **kwargs: (set(), {"truncated": False, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_claims",
        lambda *args, **kwargs: ({}, {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_worktrees",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )
    monkeypatch.setattr(
        issue_audit_core,
        "discover_jobs",
        lambda *args, **kwargs: ([], {"available": True, "errors": []}),
    )

    custom_retry = "retry issue-audit plan --repo owner/repo --remote upstream"
    issue_audit_core.discover_inventory(
        "ll7/robot_sf_ll7",
        retry_command=custom_retry,
    )

    assert observed_retry_command == custom_retry


def test_apply_rejects_missing_source_sha() -> None:
    """A plan missing source_sha is rejected before any mutations are executed."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)
    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert "plan is missing source_sha" in result["reason"]
    assert calls == []


def test_apply_rejects_source_sha_mismatch() -> None:
    """A plan generated from a different commit SHA cannot be applied."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)
    plan["source_sha"] = "0" * 40
    plan["plan_digest"] = compute_plan_digest(plan)

    result = apply_mutations(
        plan,
        runner=runner,
        apply_source_sha="1" * 40,
    )
    assert result["ok"] is False
    assert result["applied"] == []
    assert "source SHA" in result["reason"]
    assert "does not match" in result["reason"]
    assert calls == []


def test_apply_rejects_classifier_digest_mismatch() -> None:
    """A plan generated from a different classifier digest cannot be applied."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)
    plan["classifier_digest"] = "a" * 64
    plan["plan_digest"] = compute_plan_digest(plan)

    result = apply_mutations(
        plan,
        runner=runner,
        apply_source_sha=plan["source_sha"],
        apply_classifier_digest="b" * 64,
    )
    assert result["ok"] is False
    assert result["applied"] == []
    assert "classifier digest" in result["reason"]
    assert calls == []


def test_apply_rejects_read_only_diagnostic_plan() -> None:
    """Mutations cannot be applied from a read-only diagnostic plan."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "diagnostic_mode": True,
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)
    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert "read-only diagnostic" in result["reason"]
    assert calls == []


def test_apply_rejects_stale_origin_main_revision() -> None:
    """Apply with freshness check refuses when the revision does not contain origin/main."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    def fake_command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        if "merge-base" in args:
            return subprocess.CompletedProcess(args, 1, "", "not ancestor")
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, "a" * 40 + "\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 109,
                "value": "state:running",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    _attach_valid_provenance(plan)
    result = apply_mutations(
        plan,
        runner=runner,
        check_freshness=True,
        command_runner=fake_command_runner,
        apply_source_sha=plan["source_sha"],
        apply_classifier_digest=plan["classifier_digest"],
    )

    assert result["ok"] is False
    assert "does not contain current origin/main" in result["reason"]
    assert calls == []


def test_plan_read_only_diagnostic_mode_suppresses_mutations() -> None:
    """Diagnostic mode sets diagnostic_mode flag and clears all mutations."""
    inventory = {
        "repo": "ll7/robot_sf_ll7",
        "issues": [_issue(101, labels=[], body="## Definition of Done\n- [ ] verify")],
        "labels": ["state:ready"],
    }
    plan = build_audit_plan(
        inventory,
        read_only_diagnostic=True,
    )
    assert plan["diagnostic_mode"] is True
    assert plan["mutations"] == []
    assert plan["classification_status"]["mutations_suppressed"] is True
    assert "diagnostic" in str(plan["classification_status"]["reason"])


def test_plan_rejects_stale_revision_in_autonomous_mode(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The autonomous plan CLI fails closed when executing revision does not contain origin/main."""
    monkeypatch.setattr(
        issue_audit_core,
        "check_origin_main_freshness",
        lambda *a, **k: (False, "head123", "main456", "stale revision"),
    )
    code = main(["plan", "--mode", "autonomous", "--repo", "ll7/robot_sf_ll7"])
    assert code == 2
    captured = capsys.readouterr()
    assert "does not contain current origin/main" in captured.err
    assert "git fetch origin main && git merge origin/main" in captured.err
    assert "--read-only-diagnostic" in captured.err


def test_stale_producer_cannot_write_labels() -> None:
    """A plan generated with pre-fix logic or stale producer identity is rejected at apply."""
    calls: list[list[str]] = []

    def runner(args: list[str], input_text: str | None) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    plan = {
        "schema": "issue_audit_plan.v1",
        "repo": "ll7/robot_sf_ll7",
        "mutations": [
            {
                "operation": "add_label",
                "issue": 7409,
                "value": "decision-required",
                "reason": "stale prompt interpretation",
                "expected_issue": _expected_issue(),
            }
        ],
        "truncation_or_errors": [],
    }
    plan["plan_digest"] = compute_plan_digest(plan)
    result = apply_mutations(plan, runner=runner)

    assert result["ok"] is False
    assert result["applied"] == []
    assert calls == []


@pytest.mark.parametrize(
    "issue_num",
    [
        3207,
        3287,
        6155,
        7290,
        7382,
        7383,
        7384,
        7385,
        7386,
        7387,
        7388,
        7409,
        7411,
        7412,
        7457,
        7980,
        8021,
        8064,
        8076,
        8172,
        8173,
    ],
)
def test_ruled_issues_produce_no_decision_required_mutations(issue_num: int) -> None:
    """All 21 previously affected issues produce no decision-required mutation after ruling."""
    issue = _issue(
        issue_num,
        body="Owner decision required: maintainer choice on strategy.",
    )
    issue["comments"] = [
        {
            "body": f"ll7/robot_sf_ll7#{issue_num}: ruling-action-token",
            "created_at": "2026-08-20T10:00:00Z",
        }
    ]
    classification = classify_issue(
        issue,
        available_labels={"state:ready", "decision-required"},
    )
    assert classification.decision_required is False
    assert not any(
        m["operation"] == "add_label" and m["value"] == "decision-required"
        for m in classification.mutations
    )
