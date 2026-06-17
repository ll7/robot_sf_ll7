"""Tests for PR deferred-work follow-up readiness checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.dev import check_pr_followups
from scripts.dev.check_pr_followups import analyze_body

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_pr_followups.py"


def _body(*, deferred: str, issues: str = "") -> str:
    return f"""## Summary
Example PR.

## Follow-Up Issues
- Deferred work: {deferred}
- Issues opened for follow-up: {issues}
"""


def test_analyze_body_passes_when_no_deferred_work_is_declared() -> None:
    """Empty or none deferred-work values are accepted."""
    report = analyze_body(_body(deferred="none"), source="fixture")

    assert report.status == "ok"
    assert report.deferred_work == ""
    assert report.message == "No deferred work declared."


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
