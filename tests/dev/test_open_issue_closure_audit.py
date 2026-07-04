"""Tests read-only open-issue closure audit helpers."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import open_issue_closure_audit


def _issue(number: int, title: str, *, state: str = "open") -> dict[str, object]:
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "state": state,
    }


def _pr(number: int, title: str, *, issue_number: int, state: str = "merged") -> dict[str, object]:
    del issue_number
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "state": state,
        "closedAt": "2026-07-04T11:00:00Z",
    }


def test_collect_candidates_reports_open_issues_with_merged_title_linked_prs() -> None:
    """Open issues with merged title-linked PRs become review candidates."""
    candidates = open_issue_closure_audit.collect_candidates(
        [_issue(12, "add checker"), _issue(13, "no coverage yet")],
        {
            12: [_pr(99, "fix #12 checker contract", issue_number=12)],
            13: [_pr(100, "fix #130 unrelated", issue_number=13)],
        },
    )

    assert [candidate.number for candidate in candidates] == [12]
    assert candidates[0].classification == "closure_review_required"
    assert candidates[0].title_linked_prs[0].number == 99
    assert "acceptance_criteria" in candidates[0].recommended_action


def test_collect_candidates_treats_null_fields_as_empty_not_none_string() -> None:
    """Explicit JSON nulls must not coerce to the literal string ``"None"``."""
    issue_row: dict[str, object] = {
        "number": 20,
        "title": None,
        "url": "https://github.com/ll7/robot_sf_ll7/issues/20",
        "state": "open",
    }
    pr_row: dict[str, object] = {
        "number": 200,
        "title": "fix #20 handler",
        "url": "https://github.com/ll7/robot_sf_ll7/pull/200",
        "state": "merged",
        "closedAt": "2026-07-04T11:00:00Z",
        "mergedAt": None,
    }

    candidates = open_issue_closure_audit.collect_candidates([issue_row], {20: [pr_row]})

    assert len(candidates) == 1
    assert candidates[0].title == ""
    assert candidates[0].title_linked_prs[0].merged_at == "2026-07-04T11:00:00Z"


def test_collect_candidates_classifies_parent_roadmap_without_closure() -> None:
    """Parent or roadmap issues get ledger-update guidance, not closure guidance."""
    candidates = open_issue_closure_audit.collect_candidates(
        [_issue(3481, "roadmap: multi-slice parent")],
        {3481: [_pr(4000, "issue #3481 slice one", issue_number=3481)]},
    )

    assert candidates[0].classification == "parent_or_roadmap"
    assert candidates[0].recommended_action == (
        "update_status_ledger_with_merged_slices_and_remaining_work"
    )


def test_collect_candidates_ignores_closed_issues_pr_rows_and_unlinked_titles() -> None:
    """The audit stays issue-specific and requires explicit title linkage."""
    rows = [
        _issue(12, "closed issue", state="closed"),
        {
            "number": 14,
            "title": "pull request shaped row",
            "url": "https://github.com/ll7/robot_sf_ll7/pull/14",
            "state": "open",
        },
        _issue(15, "open issue"),
    ]
    pr_rows_by_issue = {
        12: [_pr(91, "fix #12", issue_number=12)],
        14: [_pr(92, "fix #14", issue_number=14)],
        15: [_pr(93, "fix #150 not fifteen", issue_number=15)],
    }

    assert open_issue_closure_audit.collect_candidates(rows, pr_rows_by_issue) == []


def test_build_report_emits_read_only_failure_summary() -> None:
    """Candidate reports expose stable counts and no-write guarantees."""
    candidates = open_issue_closure_audit.collect_candidates(
        [_issue(12, "tracking parent issue"), _issue(13, "simple issue")],
        {
            12: [_pr(91, "fix #12", issue_number=12)],
            13: [_pr(92, "fix #13", issue_number=13)],
        },
    )
    report = open_issue_closure_audit.build_report(repo="ll7/robot_sf_ll7", candidates=candidates)

    assert report["schema"] == "open_issue_closure_audit.v1"
    assert report["ok"] is False
    assert report["read_only"] is True
    assert report["issue_writes"] is False
    assert report["project_writes"] is False
    assert report["candidate_count"] == 2
    assert report["parent_or_roadmap_count"] == 1
    assert report["closure_review_count"] == 1
    assert report["failure_summary"]["reason"] == "open_issues_with_merged_title_linked_prs"


def test_build_commands_are_read_only_github_searches() -> None:
    """GitHub commands search only; they never comment, close, edit, or touch projects."""
    open_command = open_issue_closure_audit.build_open_issues_command(
        repo="ll7/robot_sf_ll7", limit=200
    )
    pr_command = open_issue_closure_audit.build_merged_prs_command(
        repo="ll7/robot_sf_ll7", issue_number=4437, limit=20
    )

    assert open_command[:3] == ["gh", "search", "issues"]
    assert "--state" in open_command
    assert open_command[open_command.index("--state") + 1] == "open"
    assert pr_command[:3] == ["gh", "search", "prs"]
    assert "4437 in:title" in pr_command
    assert "--merged" in pr_command
    for command in (open_command, pr_command):
        assert "comment" not in command
        assert "close" not in command
        assert "edit" not in command
        assert "--project" not in command


def test_main_outputs_candidates_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI returns 1 when closure-review candidates exist."""

    def fake_open_rows(*, repo: str, limit: int) -> list[dict[str, object]]:
        assert repo == "ll7/robot_sf_ll7"
        assert limit == 10
        return [_issue(12, "simple issue")]

    def fake_pr_rows(
        *, repo: str, issue_numbers: list[int], limit_per_issue: int
    ) -> dict[int, list[dict[str, object]]]:
        assert repo == "ll7/robot_sf_ll7"
        assert issue_numbers == [12]
        assert limit_per_issue == 5
        return {12: [_pr(91, "issue #12 done", issue_number=12)]}

    monkeypatch.setattr(open_issue_closure_audit, "fetch_open_issue_rows", fake_open_rows)
    monkeypatch.setattr(open_issue_closure_audit, "fetch_merged_pr_rows_by_issue", fake_pr_rows)

    exit_code = open_issue_closure_audit.main(
        ["--repo", "ll7/robot_sf_ll7", "--issue-limit", "10", "--pr-limit-per-issue", "5"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["number"] == 12


def test_run_search_command_reports_missing_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing GitHub CLI produces an actionable runtime error."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GitHub CLI 'gh' was not found"):
        open_issue_closure_audit._run_search_command(["gh", "search", "issues"])


def test_run_search_command_reports_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed GitHub CLI output is diagnosed before aggregation."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=("gh",), returncode=0, stdout="{not-json")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="Invalid JSON"):
        open_issue_closure_audit._run_search_command(["gh", "search", "issues"])
