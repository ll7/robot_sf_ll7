"""Tests for PR deferred-work follow-up readiness checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import check_pr_followups
from scripts.dev.check_pr_followups import analyze_body, analyze_domain_approval

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


def test_analyze_body_passes_when_no_deferred_work_is_declared() -> None:
    """Empty or none deferred-work values are accepted."""
    report = analyze_body(_body(deferred="none"), source="fixture")

    assert report.status == "ok"
    assert report.deferred_work == ""
    assert report.message == "No deferred work declared."


def test_domain_approval_not_required_for_na_or_docs_only_research_fields() -> None:
    """Routine docs/support PRs can opt out without domain-review friction."""
    report = analyze_domain_approval(
        _domain_body(evidence_tier="docs-only", result_classification="NA"),
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
