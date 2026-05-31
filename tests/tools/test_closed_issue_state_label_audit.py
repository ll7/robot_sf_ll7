"""Tests for the closed issue state-label hygiene audit."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from scripts.tools import closed_issue_state_label_audit as audit


def test_query_plan_checks_closed_issues_for_each_live_state_label() -> None:
    """The live GitHub query plan should be closed-issue and label-specific."""
    plan = audit.build_query_plan(repo="owner/repo", labels=("state:ready", "state:running"))

    assert plan == (
        (
            "gh",
            "issue",
            "list",
            "--repo",
            "owner/repo",
            "--state",
            "closed",
            "--label",
            "state:ready",
            "--json",
            "number,title,labels,url",
            "--limit",
            "1000",
        ),
        (
            "gh",
            "issue",
            "list",
            "--repo",
            "owner/repo",
            "--state",
            "closed",
            "--label",
            "state:running",
            "--json",
            "number,title,labels,url",
            "--limit",
            "1000",
        ),
    )


def test_audit_report_merges_duplicate_issues_and_counts_stale_labels() -> None:
    """One closed issue can carry more than one stale live-state label."""
    payloads: dict[str, list[dict[str, Any]]] = {
        "state:ready": [
            {
                "number": 10,
                "title": "Ready but closed",
                "url": "https://example.test/10",
                "labels": [{"name": "state:ready"}, {"name": "workflow"}],
            },
            {
                "number": 11,
                "title": "Ready and running but closed",
                "url": "https://example.test/11",
                "labels": [{"name": "state:ready"}, {"name": "state:running"}],
            },
        ],
        "state:running": [
            {
                "number": 11,
                "title": "Ready and running but closed",
                "url": "https://example.test/11",
                "labels": [{"name": "state:ready"}, {"name": "state:running"}],
            }
        ],
        "state:blocked": [],
    }

    def fake_runner(command: tuple[str, ...]) -> list[dict[str, Any]]:
        label = command[command.index("--label") + 1]
        return payloads[label]

    report = audit.audit_closed_issue_state_labels(runner=fake_runner)

    assert report.schema == "closed_issue_state_label_audit.v1"
    assert report.read_only is True
    assert report.stale_issue_count == 2
    assert report.stale_label_counts == {
        "state:ready": 2,
        "state:running": 1,
        "state:blocked": 0,
    }
    assert [issue.number for issue in report.stale_issues] == [10, 11]
    assert report.stale_issues[1].stale_labels == ("state:ready", "state:running")
    assert report.mutation_plan == ()


def test_cli_exits_nonzero_and_emits_machine_readable_failure_summary(monkeypatch, capsys) -> None:
    """Stale labels should fail the command while still printing JSON."""

    def fake_runner(command: tuple[str, ...]) -> list[dict[str, Any]]:
        label = command[command.index("--label") + 1]
        if label != "state:blocked":
            return []
        return [
            {
                "number": 12,
                "title": "Blocked but closed",
                "url": "https://example.test/12",
                "labels": [{"name": "state:blocked"}],
            }
        ]

    monkeypatch.setattr(audit, "_run_gh_json", fake_runner)

    exit_code = audit.main(["--repo", "owner/repo"])

    assert exit_code == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "closed_issue_state_label_audit.v1"
    assert summary["stale_issue_count"] == 1
    assert summary["stale_label_counts"]["state:blocked"] == 1
    assert summary["mutation_plan"] == []


def test_run_gh_json_reports_missing_gh(monkeypatch) -> None:
    """A missing GitHub CLI should fail with an actionable message."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    try:
        audit._run_gh_json(("gh", "issue", "list"))
    except RuntimeError as err:
        assert "GitHub CLI 'gh' was not found" in str(err)
    else:
        raise AssertionError("expected missing gh to raise RuntimeError")


def test_run_gh_json_reports_captured_stderr(monkeypatch) -> None:
    """Captured gh stderr should be preserved when a command fails."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=("gh", "issue", "list"),
            stderr="authentication required",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    try:
        audit._run_gh_json(("gh", "issue", "list"))
    except RuntimeError as err:
        message = str(err)
        assert "GitHub CLI command failed" in message
        assert "authentication required" in message
    else:
        raise AssertionError("expected failing gh command to raise RuntimeError")


def test_run_gh_json_reports_invalid_json(monkeypatch) -> None:
    """Non-JSON gh output should fail before callers inspect the payload shape."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=("gh",), returncode=0, stdout="not-json")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    try:
        audit._run_gh_json(("gh", "issue", "list"))
    except RuntimeError as err:
        assert "Failed to parse GitHub CLI JSON output" in str(err)
    else:
        raise AssertionError("expected invalid JSON to raise RuntimeError")
