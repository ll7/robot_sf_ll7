"""Tests for PR deferred-work follow-up readiness checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import check_pr_followups
from scripts.dev.check_pr_followups import (
    analyze_body,
    analyze_body_quality,
    analyze_domain_approval,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_pr_followups.py"


def _body(*, deferred: str, issues: str = "") -> str:
    return f"""## Summary
Example PR.

## Follow-Up Issues
- Deferred work: {deferred}
- Issues opened for follow-up: {issues}
"""


def _domain_body(
    *,
    evidence_tier: str = "targeted smoke",
    result_classification: str = "blocker-resolution",
    domain_section: str = "",
) -> str:
    return f"""## Research Result Guidance
- Evidence tier: {evidence_tier}
- Result classification: {result_classification}

{domain_section}
    """


def _approved_domain_section(*, status: str = "approved", note: str = "maintainer review") -> str:
    return f"""## Domain-Aware Approval
- Required for this PR: yes - evidence-validity-sensitive result classification
- Domains reviewed: evidence classification, figure eligibility
- Status: {status}
- Approver/review source or waiver: {note}
- Validity checklist:
  - Target claim/hypothesis: issue claim boundary stays diagnostic-only
  - Comparator or split/evidence validity: existing targeted smoke proof only
  - Fallback/degraded exclusions: fallback/degraded evidence remains excluded
  - Claim boundary: no paper-facing benchmark claim
  - Implementation integrity vs experimental validity: tests prove gate behavior, not result validity
"""


def _well_formed_support_body() -> str:
    return """## Summary
Adds a small workflow guard for PR body quality.

## Research Result Guidance
- Evidence tier: NA - workflow guard only
- Result classification: NA - no research claim

## Follow-Up Issues
- Deferred work: none
- Issues opened for follow-up: none
"""


def test_analyze_body_passes_when_no_deferred_work_is_declared() -> None:
    """Empty or none deferred-work values are accepted."""
    report = analyze_body(_body(deferred="none"), source="fixture")

    assert report.status == "ok"
    assert report.deferred_work == ""
    assert report.message == "No deferred work declared."


@pytest.mark.parametrize(
    "files",
    [
        ("robot_sf/benchmark/camera_ready/_summaries.py",),
        ("robot_sf/benchmark/camera_ready/_route_clearance.py",),
    ],
)
def test_body_quality_rejects_empty_body_for_substantive_historical_prs(
    files: tuple[str, ...],
) -> None:
    """PRs #3414/#3415 changed benchmark code with an empty description."""
    report = analyze_body_quality(
        "",
        source="fixture",
        changed_files=files,
        require_substantive_body=True,
    )

    assert report.status == "empty_body"
    assert report.substantive_files == files


def test_body_quality_rejects_coderabbit_only_body_for_substantive_pr() -> None:
    """PR #3416 used only generated CodeRabbit release notes for benchmark code."""
    body = """
<!-- This is an auto-generated comment: release notes by coderabbit.ai -->

## Summary by CodeRabbit

* **Refactor**
  * Reorganized internal benchmark reporting code to improve maintainability.

<!-- end of auto-generated comment: release notes by coderabbit.ai -->
"""

    report = analyze_body_quality(
        body,
        source="fixture",
        changed_files=("robot_sf/benchmark/camera_ready/_reporting.py",),
        require_substantive_body=True,
    )

    assert report.status == "bot_only_body"


def test_body_quality_strips_whole_generated_blocks_before_accepting_human_text() -> None:
    """Generated release-note blocks should not mask whether human body text exists."""
    body = """
<!-- This is an auto-generated comment: release notes by coderabbit.ai -->

## Summary by CodeRabbit

* **Refactor**
  * Reorganized internal benchmark reporting code.

<!-- end of auto-generated comment: release notes by coderabbit.ai -->

## Summary
Human-authored rationale for the workflow change.
"""

    report = analyze_body_quality(
        body,
        source="fixture",
        changed_files=("scripts/dev/check_pr_followups.py",),
        require_substantive_body=True,
    )

    assert report.status == "ok"


def test_body_quality_allows_docs_only_empty_body() -> None:
    """Documentation-only PRs do not need the substantive source/config body gate."""
    report = analyze_body_quality(
        "",
        source="fixture",
        changed_files=("docs/context/example.md",),
        require_substantive_body=True,
    )

    assert report.status == "ok"
    assert report.substantive_files == ()


def test_body_quality_does_not_treat_path_lookalikes_as_template_only() -> None:
    """Path exemptions require directory boundaries, not arbitrary string prefixes."""
    report = analyze_body_quality(
        "",
        source="fixture",
        changed_files=(".github/PULL_REQUEST_TEMPLATE_extra.py", "docs_extra/change.py"),
        require_substantive_body=True,
    )

    assert report.status == "empty_body"
    assert report.substantive_files == (
        ".github/PULL_REQUEST_TEMPLATE_extra.py",
        "docs_extra/change.py",
    )


def test_body_quality_accepts_human_body_for_substantive_changes() -> None:
    """A human-authored body satisfies the source/configuration body gate."""
    report = analyze_body_quality(
        _well_formed_support_body(),
        source="fixture",
        changed_files=("scripts/dev/check_pr_followups.py",),
        require_substantive_body=True,
    )

    assert report.status == "ok"
    assert report.substantive_files == ("scripts/dev/check_pr_followups.py",)


def test_domain_approval_not_required_for_na_or_docs_only_research_fields() -> None:
    """Routine docs/support PRs can opt out without domain-review friction."""
    report = analyze_domain_approval(
        _domain_body(evidence_tier="docs-only", result_classification="NA"),
        source="fixture",
    )

    assert report.status == "ok"
    assert report.sensitive_terms == ()


@pytest.mark.parametrize(
    ("evidence_tier", "result_classification"),
    [
        ("docs-only (support helper)", "NA"),
        ("na, support-only workflow", "not applicable (workflow bugfix)"),
    ],
)
def test_domain_approval_not_required_allows_punctuated_opt_out_reasons(
    evidence_tier: str,
    result_classification: str,
) -> None:
    """Docs/NA opt-out fields may include a short reason after punctuation."""
    report = analyze_domain_approval(
        _domain_body(evidence_tier=evidence_tier, result_classification=result_classification),
        source="fixture",
    )

    assert report.status == "ok"
    assert report.sensitive_terms == ()


def test_domain_approval_required_for_non_na_research_fields() -> None:
    """Evidence-result PR bodies need the domain approval section."""
    report = analyze_domain_approval(_domain_body(), source="fixture")

    assert report.status == "missing_domain_approval"
    assert "Evidence tier: targeted smoke" in report.sensitive_terms
    assert "Result classification: blocker-resolution" in report.sensitive_terms


@pytest.mark.parametrize(
    ("pr_number", "body"),
    [
        (
            3449,
            """## Summary
Closes #2916. Implements a bounded forecast-risk gate, evidence_tier: stress,
claim_boundary: diagnostic_only, not paper-grade.

## Validation
Verdict: continue.
""",
        ),
        (
            3450,
            """## Summary
Closes #2546. A bounded diagnostic-only experiment testing ScenarioBelief uncertainty,
evidence_tier: stress, claim_boundary: diagnostic_only, not paper-grade.

## Validation
Finding: continue.
""",
        ),
    ],
)
def test_domain_approval_rejects_historical_evidence_sensitive_missing_contracts(
    pr_number: int,
    body: str,
) -> None:
    """PRs #3449/#3450 carried evidence-sensitive text without the structured approval section."""
    del pr_number
    report = analyze_domain_approval(body, source="fixture")

    assert report.status == "missing_domain_approval"


def test_domain_approval_accepts_approved_review_source() -> None:
    """Approved domain-review metadata satisfies the evidence-validity gate."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section=_approved_domain_section(
                note="maintainer review of claim boundary and fallback exclusions"
            )
        ),
        source="fixture",
    )

    assert report.status == "ok"
    assert "maintainer review" in report.approval_note


def test_domain_approval_accepts_none_in_validity_checklist_field() -> None:
    """A checklist answer can honestly say no exclusion or claim applies."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section=_approved_domain_section().replace(
                "fallback/degraded evidence remains excluded",
                "none",
            )
        ),
        source="fixture",
    )

    assert report.status == "ok"
    assert report.checklist_errors == ()


def test_domain_approval_accepts_explicit_maintainer_waiver() -> None:
    """A maintainer waiver is explicit enough to avoid pretending approval happened."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section=_approved_domain_section(
                status="waived",
                note="maintainer waiver; diagnostic-only docs update",
            )
        ),
        source="fixture",
    )

    assert report.status == "ok"
    assert "waiver" in report.approval_note


def test_domain_approval_rejects_pending_or_placeholder_status() -> None:
    """Pending domain review keeps a sensitive PR out of final readiness."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section="""## Domain-Aware Approval
- Required for this PR: yes
- Domains reviewed: experimental comparison
- Status: pending
- Approver/review source or waiver: waiting for domain reviewer
- Validity checklist:
  - Target claim/hypothesis: issue claim boundary stays diagnostic-only
  - Comparator or split/evidence validity: existing targeted smoke proof only
  - Fallback/degraded exclusions: fallback/degraded evidence remains excluded
  - Claim boundary: no paper-facing benchmark claim
  - Implementation integrity vs experimental validity: tests prove gate behavior, not result validity
"""
        ),
        source="fixture",
    )

    assert report.status == "pending_domain_approval"


def test_domain_approval_placeholder_message_names_concrete_domain_values() -> None:
    """Template option text should fail with a fix that avoids another body-check retry."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section="""## Domain-Aware Approval
- Required for this PR: yes
- Domains reviewed: evidence classification / experimental comparison / figure eligibility / benchmark interpretation / paper-facing claims / NA
- Status: approved
- Approver/review source or waiver: maintainer review
- Validity checklist:
  - Target claim/hypothesis: issue claim boundary stays diagnostic-only
  - Comparator or split/evidence validity: existing targeted smoke proof only
  - Fallback/degraded exclusions: fallback/degraded evidence remains excluded
  - Claim boundary: no paper-facing benchmark claim
  - Implementation integrity vs experimental validity: tests prove gate behavior, not result validity
"""
        ),
        source="fixture",
    )

    assert report.status == "missing_domain_approval_note"
    assert "slash-separated template option text" in report.message
    assert "comma-separated domains" in report.message


def test_domain_approval_rejects_not_approved_status() -> None:
    """A negated approval phrase cannot satisfy the final-readiness gate."""
    report = analyze_domain_approval(
        _domain_body(domain_section=_approved_domain_section(status="not approved")),
        source="fixture",
    )

    assert report.status == "invalid_domain_approval_status"


def test_domain_approval_requires_validity_checklist_fields() -> None:
    """Approval metadata must include the evidence-validity checklist."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section="""## Domain-Aware Approval
- Required for this PR: yes - evidence-validity-sensitive result classification
- Domains reviewed: experimental comparison
- Status: approved
- Approver/review source or waiver: maintainer review
- Validity checklist:
  - Target claim/hypothesis: issue claim boundary stays diagnostic-only
  - Comparator or split/evidence validity:
  - Fallback/degraded exclusions: fallback/degraded evidence remains excluded
  - Claim boundary: no paper-facing benchmark claim
  - Implementation integrity vs experimental validity: tests prove gate behavior, not result validity
"""
        ),
        source="fixture",
    )

    assert report.status == "incomplete_domain_approval"
    assert "Comparator or split/evidence validity" in report.checklist_errors


def test_domain_approval_rejects_not_required_for_sensitive_result() -> None:
    """Non-NA evidence fields cannot be paired with a not-required approval claim."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section="""## Domain-Aware Approval
- Required for this PR: no - ordinary implementation
- Domains reviewed: NA
- Status: not required
- Approver/review source or waiver: NA
"""
        ),
        source="fixture",
    )

    assert report.status == "domain_approval_required"


def test_domain_approval_required_value_does_not_treat_nominal_as_no() -> None:
    """Required explanations starting with nominal/narrowed stay in the required path."""
    report = analyze_domain_approval(
        _domain_body(
            domain_section=_approved_domain_section().replace(
                "yes - evidence-validity-sensitive result classification",
                "nominal benchmark change",
            )
        ),
        source="fixture",
    )

    assert report.status == "ok"
    assert report.required == "nominal benchmark change"


def test_analyze_body_requires_issue_when_deferred_work_is_declared() -> None:
    """Deferred work without a disposition fails closed."""
    report = analyze_body(_body(deferred="Add a broader benchmark sweep."), source="fixture")

    assert report.status == "missing_followup"
    assert "without a linked issue" in report.message


def test_analyze_body_accepts_linked_followup_issue() -> None:
    """Linked follow-up issues satisfy declared deferred work."""
    report = analyze_body(
        _body(deferred="Run the broader benchmark sweep.", issues="#2966"),
        source="fixture",
    )

    assert report.status == "ok"
    assert report.linked_issues == ("#2966",)


def test_analyze_body_accepts_explicit_no_issue_reason() -> None:
    """An explicit NA reason can replace a follow-up issue."""
    report = analyze_body(
        _body(
            deferred="No additional work beyond reviewer verification.",
            issues="NA - reviewer verification only",
        ),
        source="fixture",
    )

    assert report.status == "ok"
    assert report.explicit_no_issue_reason == "NA - reviewer verification only"


@pytest.mark.parametrize("issues", ["none", "no", "NA", "not applicable"])
def test_analyze_body_rejects_bare_no_followup_disposition(issues: str) -> None:
    """Bare NA/none/no values do not explain why deferred work needs no issue."""
    report = analyze_body(
        _body(
            deferred="No additional work beyond reviewer verification.",
            issues=issues,
        ),
        source="fixture",
    )

    assert report.status == "missing_followup"
    assert "without a linked issue" in report.message


def test_analyze_body_detects_residual_scope_outside_followup_section() -> None:
    """Closing PRs cannot hide residual work in other sections."""
    body = """## Summary
Closes `#2993`.

## Validation / Proof
This is diagnostic-only and leaves the broader vector-env matrix as remaining work.

## Follow-Up Issues
- Deferred work: none
- Issues opened for follow-up: NA - no follow-up needed
"""

    report = analyze_body(body, source="fixture")

    assert report.status == "residual_scope_without_followup"
    assert "declaring residual scope" in report.message


def test_analyze_body_allows_residual_scope_with_linked_followup_issue() -> None:
    """Residual-scope closure is allowed when an open follow-up is declared."""
    body = """## Summary
Closes #2993.

## Validation / Proof
This is diagnostic-only and leaves the broader vector-env matrix as remaining work.

## Follow-Up Issues
- Deferred work: Broader vector-env matrix remains.
- Issues opened for follow-up: #3165
"""

    report = analyze_body(body, source="fixture")

    assert report.status == "ok"
    assert report.linked_issues == ("#3165",)


def test_analyze_body_allows_residual_scope_with_explicit_maintainer_waiver() -> None:
    """Maintainer waiver language is the explicit escape hatch."""
    body = """## Summary
Closes #2993.

## Validation / Proof
This is diagnostic-only and leaves remaining work.

## Follow-Up Issues
- Deferred work: Remaining work is intentionally waived.
- Issues opened for follow-up: NA - maintainer waiver for this closure boundary
"""

    report = analyze_body(body, source="fixture")

    assert report.status == "ok"
    assert "waiver" in report.explicit_no_issue_reason


def test_analyze_body_rejects_closed_followup_issue(monkeypatch) -> None:
    """Open-state verification rejects linked issues that are not open."""

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout="CLOSED\n", stderr="")

    monkeypatch.setattr(check_pr_followups.subprocess, "run", fake_run)

    report = analyze_body(
        _body(deferred="Run the broader benchmark sweep.", issues="#2966"),
        source="fixture",
        require_open_issues=True,
    )

    assert report.status == "issue_state_error"
    assert "#2966: state is CLOSED" in report.issue_state_errors[0]


def test_analyze_body_rejects_unverifiable_issue_when_gh_is_missing(monkeypatch) -> None:
    """Open-state verification reports a compact error when gh is unavailable."""

    def fake_run(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("gh")

    monkeypatch.setattr(check_pr_followups.subprocess, "run", fake_run)

    report = analyze_body(
        _body(deferred="Run the broader benchmark sweep.", issues="#2966"),
        source="fixture",
        require_open_issues=True,
    )

    assert report.status == "issue_state_error"
    assert "#2966: unable to verify open state (gh CLI not found)" in report.issue_state_errors


def test_analyze_body_collects_multiline_deferred_work() -> None:
    """Continuation lines belong to the deferred-work value until the next field."""
    body = """## Follow-Up Issues
- Deferred work:
  - Run the broader benchmark sweep.
  - Promote durable evidence.
- Issues opened for follow-up: #2966
"""

    report = analyze_body(body, source="fixture")

    assert report.status == "ok"
    assert "Run the broader benchmark sweep" in report.deferred_work
    assert "Promote durable evidence" in report.deferred_work
    assert report.linked_issues == ("#2966",)


def test_cli_reads_github_pull_request_event(tmp_path: Path, monkeypatch) -> None:
    """The CLI can read pull_request.body from a GitHub event payload."""
    monkeypatch.delenv("PR_READY_PR_BODY_FILE", raising=False)
    monkeypatch.delenv("PR_READY_REQUIRE_OPEN_FOLLOWUP_ISSUES", raising=False)
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": _body(
                        deferred="Run release readiness dashboard.",
                        issues="https://github.com/ll7/robot_sf_ll7/issues/2965",
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--github-event-path", str(event_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "status=ok" in result.stdout
    assert "#2965" in result.stdout


def test_cli_fails_for_deferred_work_without_disposition(tmp_path: Path) -> None:
    """The CLI exits nonzero for deferred work without an issue or NA reason."""
    body_path = tmp_path / "body.md"
    body_path.write_text(_body(deferred="Open the remaining benchmark issues."), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--body-file", str(body_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "status=missing_followup" in result.stderr


def test_cli_rejects_empty_substantive_pr_body_from_event(tmp_path: Path) -> None:
    """The live PR workflow can fail closed on empty substantive PR descriptions."""
    event_path = tmp_path / "event.json"
    files_path = tmp_path / "files.txt"
    event_path.write_text(json.dumps({"pull_request": {"body": ""}}), encoding="utf-8")
    files_path.write_text("robot_sf/benchmark/camera_ready/_summaries.py\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--github-event-path",
            str(event_path),
            "--changed-files-file",
            str(files_path),
            "--require-substantive-body",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "status=empty_body" in result.stderr


def test_cli_event_body_overrides_inherited_pr_ready_body_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit event bodies should not be masked by PR readiness environment variables."""
    inherited_body = tmp_path / "inherited.md"
    event_path = tmp_path / "event.json"
    files_path = tmp_path / "files.txt"
    inherited_body.write_text(_well_formed_support_body(), encoding="utf-8")
    event_path.write_text(json.dumps({"pull_request": {"body": ""}}), encoding="utf-8")
    files_path.write_text("robot_sf/benchmark/camera_ready/_summaries.py\n", encoding="utf-8")
    monkeypatch.setenv("PR_READY_PR_BODY_FILE", str(inherited_body))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--github-event-path",
            str(event_path),
            "--changed-files-file",
            str(files_path),
            "--require-substantive-body",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "status=empty_body" in result.stderr
    assert str(event_path) in result.stderr


def test_cli_accepts_well_formed_substantive_pr_body_with_changed_files(tmp_path: Path) -> None:
    """Changed-file awareness does not block well-formed workflow PR bodies."""
    event_path = tmp_path / "event.json"
    files_path = tmp_path / "files.txt"
    event_path.write_text(
        json.dumps({"pull_request": {"body": _well_formed_support_body()}}),
        encoding="utf-8",
    )
    files_path.write_text("scripts/dev/check_pr_followups.py\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--github-event-path",
            str(event_path),
            "--changed-files-file",
            str(files_path),
            "--require-substantive-body",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "status=ok" in result.stdout


def test_cli_fails_for_sensitive_pr_without_domain_approval(tmp_path: Path) -> None:
    """The CLI blocks evidence-validity PR bodies that omit domain-aware approval."""
    body_path = tmp_path / "body.md"
    body_path.write_text(
        """## Research Result Guidance
- Evidence tier: targeted smoke
- Result classification: blocker-resolution

## Follow-Up Issues
- Deferred work: none
- Issues opened for follow-up: none
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--body-file", str(body_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "status=missing_domain_approval" in result.stderr


def test_cli_require_body_fails_closed_without_pr_body_source(monkeypatch) -> None:
    """Final readiness should not silently skip missing PR body input."""
    monkeypatch.delenv("PR_READY_PR_BODY_FILE", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--require-body"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "status=missing_body" in result.stdout
