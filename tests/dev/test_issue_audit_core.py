"""Focused contract tests for the shared issue-audit core."""

from __future__ import annotations

import json
import subprocess
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
    assert all(mutation["expected_issue"] == _expected_issue() for mutation in plan["mutations"])


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
        "discover_pull_requests",
        lambda _repo, *, state, max_pages, runner: (
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

    def discover_prs(_repo: str, *, state: str, max_pages: int, runner: object) -> tuple:
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
    plan["plan_digest"] = compute_plan_digest(plan)
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
    plan["plan_digest"] = compute_plan_digest(plan)

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
    plan["plan_digest"] = compute_plan_digest(plan)

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
    plan["plan_digest"] = compute_plan_digest(plan)

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
    plan["plan_digest"] = compute_plan_digest(plan)

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
    plan["plan_digest"] = compute_plan_digest(plan)

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
