"""Focused contract tests for the shared issue-audit core."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from scripts.dev.issue_audit_core import (
    apply_mutations,
    build_audit_plan,
    build_decision_envelope,
    build_pending_decision_queue,
    classify_issue,
    closure_evidence,
    compute_plan_digest,
    discover_issue_comments,
    label_api_path,
    main,
    parse_decision_answer,
    select_next_pending_decision,
    validate_decision_envelope,
)


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
        "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "author": "maintainer",
        "labels": labels or [],
        "body": body,
        "comments": [],
    }


def test_state_contradiction_prefers_observed_active_work() -> None:
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


def test_stale_running_state_is_preserved_and_not_promoted_to_ready() -> None:
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


def test_state_qualifiers_are_preserved_during_execution_state_cleanup() -> None:
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


def test_missing_optional_job_visibility_preserves_slurm_issue_without_blocker_claim() -> None:
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
    proven = classify_issue(
        _issue(
            112,
            body="Rights: license missing pending permission review.\n"
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
            body="This issue is blocked until the required checkpoint is staged.",
        ),
        available_labels={"state:blocked", "state:blocked-external-input"},
    )
    assert current.classification == "blocked"
    assert {mutation["value"] for mutation in current.mutations} == {"state:blocked-external-input"}


def test_decision_detection_distinguishes_resolved_records_from_open_gates() -> None:
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


def test_blocker_replaces_single_stale_execution_state() -> None:
    classification = classify_issue(
        _issue(
            118,
            labels=["state:running"],
            body="This issue is blocked until the required dataset is staged.",
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


def test_type_mirror_requires_complete_valid_archetype_metadata() -> None:
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


def test_decision_queue_is_machine_readable_and_project_free() -> None:
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


def test_decision_envelope_rejects_stale_plan_and_live_issue_state() -> None:
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


def test_envelope_cli_emits_machine_readable_payload(tmp_path: Any, capsys: Any) -> None:
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
    merged = [{"number": 902, "title": "Fixes #105", "merged_at": "2026-08-11T10:00:00Z"}]
    issue = _issue(105, title="Parent roadmap", body="Child issue #106")

    assert closure_evidence(issue, merged_prs=merged, open_issue_numbers={105})["eligible"] is False

    issue["body"] = "Parent close condition: all linked children closed\nChild issue #106"
    assert closure_evidence(issue, merged_prs=merged, open_issue_numbers={105})["eligible"] is True


def test_label_endpoint_uri_escapes_colon() -> None:
    assert label_api_path("ll7/robot_sf_ll7", 106, "state:running") == (
        "repos/ll7/robot_sf_ll7/issues/106/labels/state%3Arunning"
    )


def test_apply_uses_encoded_delete_and_reads_back() -> None:
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
                json.dumps({"state": "open", "labels": [{"name": "state:running"}]}),
                "",
            )
        raise AssertionError(f"unexpected command: {args}")

    result = apply_mutations(
        {
            "schema": "issue_audit_plan.v1",
            "repo": "ll7/robot_sf_ll7",
            "mutations": [
                {
                    "operation": "remove_label",
                    "issue": 106,
                    "value": "state:ready",
                    "reason": "active work selects running",
                    "evidence": ["open PR #903"],
                },
                {
                    "operation": "add_label",
                    "issue": 106,
                    "value": "state:running",
                    "reason": "active work observed",
                    "evidence": ["open PR #903"],
                },
            ],
            "truncation_or_errors": [],
        },
        runner=runner,
    )

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


def test_incomplete_plan_fails_closed_before_mutation() -> None:
    result = apply_mutations(
        {
            "schema": "issue_audit_plan.v1",
            "repo": "ll7/robot_sf_ll7",
            "mutations": [],
            "truncation_or_errors": ["issues"],
        }
    )

    assert result["ok"] is False
    assert result["applied"] == []


def test_pending_queue_can_record_readback_confirmed_safe_mutations() -> None:
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
