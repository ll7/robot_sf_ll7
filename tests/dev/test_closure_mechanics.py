"""Tests for closure mechanics script."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import closure_mechanics


def _audit_report(
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a minimal audit report for testing."""
    return {
        "schema": "open_issue_closure_audit.v1",
        "ok": False,
        "read_only": True,
        "repo": "ll7/robot_sf_ll7",
        "candidate_count": len(candidates or []),
        "candidates": candidates or [],
    }


def _candidate(
    number: int,
    title: str,
    *,
    classification: str = "closure_review_required",
    prs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a minimal audit candidate."""
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "classification": classification,
        "recommended_action": "read_acceptance_criteria_then_close_if_fully_covered_else_comment_residual_checklist",
        "title_linked_prs": prs or [],
    }


def _pr_entry(number: int, title: str) -> dict[str, object]:
    """Build a minimal PR entry."""
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "merged_at": "2026-07-04T11:00:00Z",
    }


def test_classify_and_build_actions_residual_for_closure_review() -> None:
    """Closure-review candidates get residual (not closure) by default."""
    report = _audit_report([_candidate(12, "add checker", prs=[_pr_entry(99, "fix #12 checker")])])
    actions = closure_mechanics.classify_and_build_actions(report)

    assert len(actions) == 1
    assert actions[0].issue_number == 12
    assert actions[0].action_type == "residual"
    assert actions[0].should_close is False
    assert actions[0].pr_numbers == (99,)
    assert "#99" in actions[0].comment_body
    assert "Residual checklist" in actions[0].comment_body


def test_classify_and_build_actions_parent_ledger() -> None:
    """Parent/roadmap candidates get a ledger comment, never close."""
    report = _audit_report(
        [
            _candidate(
                3481,
                "roadmap: multi-slice parent",
                classification="parent_or_roadmap",
                prs=[_pr_entry(4000, "issue #3481 slice one")],
            )
        ]
    )
    actions = closure_mechanics.classify_and_build_actions(report)

    assert len(actions) == 1
    assert actions[0].action_type == "parent_ledger"
    assert actions[0].should_close is False
    assert "#4000" in actions[0].comment_body
    assert "Keeping this parent issue open" in actions[0].comment_body


def test_classify_and_build_actions_empty_report() -> None:
    """Empty audit report produces no actions."""
    report = _audit_report([])
    assert closure_mechanics.classify_and_build_actions(report) == []


def test_classify_and_build_actions_skips_candidates_without_number() -> None:
    """Candidates missing a valid number are silently skipped."""
    report = _audit_report([{"title": "no number", "classification": "closure_review_required"}])
    assert closure_mechanics.classify_and_build_actions(report) == []


def test_build_comment_command_format() -> None:
    """Comment command targets the correct issue number and repo."""
    cmd = closure_mechanics.build_comment_command(
        issue_number=42, body="hello", repo="ll7/robot_sf_ll7"
    )
    assert cmd == [
        "gh",
        "issue",
        "comment",
        "42",
        "--repo",
        "ll7/robot_sf_ll7",
        "--body",
        "hello",
    ]


def test_build_close_command_format() -> None:
    """Close command uses completed reason and targets the repo."""
    cmd = closure_mechanics.build_close_command(issue_number=42, repo="ll7/robot_sf_ll7")
    assert cmd == [
        "gh",
        "issue",
        "close",
        "42",
        "--repo",
        "ll7/robot_sf_ll7",
        "--reason",
        "completed",
    ]


def test_execute_actions_dry_run_preview() -> None:
    """Dry-run mode previews actions without executing them."""
    report = _audit_report([_candidate(12, "add checker", prs=[_pr_entry(99, "fix #12")])])
    actions = closure_mechanics.classify_and_build_actions(report)
    result = closure_mechanics.execute_actions(actions, repo="ll7/robot_sf_ll7", dry_run=True)

    assert result["dry_run"] is True
    assert result["action_count"] == 1
    assert result["results"][0]["dry_run"] is True
    assert result["results"][0]["comment_posted"] is False
    assert result["results"][0]["closed"] is False
    assert result["results"][0]["error"] is None


def test_execute_actions_apply_calls_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply mode calls gh to post comments."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(closure_mechanics.subprocess, "run", fake_run)

    report = _audit_report([_candidate(12, "add checker", prs=[_pr_entry(99, "fix #12")])])
    actions = closure_mechanics.classify_and_build_actions(report)
    result = closure_mechanics.execute_actions(actions, repo="ll7/robot_sf_ll7", dry_run=False)

    assert result["dry_run"] is False
    assert result["results"][0]["comment_posted"] is True
    assert len(calls) == 1
    assert "comment" in calls[0]
    assert "12" in calls[0]
    # execute_actions must route through build_comment_command so the executed
    # command is repo-scoped (single source of truth; guards the --repo drift).
    body = calls[0][calls[0].index("--body") + 1]
    assert calls[0] == closure_mechanics.build_comment_command(
        issue_number=12, body=body, repo="ll7/robot_sf_ll7"
    )
    assert calls[0][calls[0].index("--repo") + 1] == "ll7/robot_sf_ll7"


def test_execute_actions_close_path_is_repo_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply+close routes both gh commands through the repo-scoped builders."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(closure_mechanics.subprocess, "run", fake_run)

    report = _audit_report([_candidate(12, "add checker", prs=[_pr_entry(99, "fix #12")])])
    actions = closure_mechanics.classify_and_build_actions(report, close_issues={12})
    result = closure_mechanics.execute_actions(actions, repo="ll7/robot_sf_ll7", dry_run=False)

    assert result["results"][0]["comment_posted"] is True
    assert result["results"][0]["closed"] is True
    # One comment call + one close call, both repo-scoped via the builders.
    assert len(calls) == 2
    assert calls[1] == closure_mechanics.build_close_command(
        issue_number=12, repo="ll7/robot_sf_ll7"
    )
    for cmd in calls:
        assert cmd[cmd.index("--repo") + 1] == "ll7/robot_sf_ll7"


def test_main_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main reads audit report from stdin in dry-run mode."""
    report = _audit_report([_candidate(15, "simple fix", prs=[_pr_entry(88, "fix #15")])])
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(report)))
    exit_code = closure_mechanics.main([])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["action_count"] == 1
    assert payload["results"][0]["issue_number"] == 15


def test_main_rejects_invalid_schema(capsys: pytest.CaptureFixture[str]) -> None:
    """Main rejects reports with wrong schema."""
    import io

    bad_report = {"schema": "wrong.v1", "candidates": []}

    monkeypatch_instance = pytest.MonkeyPatch()
    monkeypatch_instance.setattr("sys.stdin", io.StringIO(json.dumps(bad_report)))
    try:
        exit_code = closure_mechanics.main([])
    finally:
        monkeypatch_instance.undo()

    assert exit_code == 2


def test_main_reads_report_file(tmp_path: object) -> None:
    """Main reads audit report from file when --report-file is given."""
    import pathlib

    report = _audit_report([_candidate(20, "file test", prs=[_pr_entry(77, "fix #20")])])
    report_path = pathlib.Path(str(tmp_path)) / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    # Use dry-run to avoid subprocess calls.
    import io

    monkeypatch_instance = pytest.MonkeyPatch()
    monkeypatch_instance.setattr("sys.stdin", io.StringIO(""))
    try:
        # We can't easily capture stdout here with the file path approach,
        # so just verify it doesn't crash.
        exit_code = closure_mechanics.main(["--report-file", str(report_path)])
    finally:
        monkeypatch_instance.undo()

    assert exit_code == 0


def test_residual_comment_contains_dispatch_note() -> None:
    """Residual comments include the dispatch-note guidance."""
    report = _audit_report([_candidate(50, "partial fix", prs=[_pr_entry(200, "fix #50 part 1")])])
    actions = closure_mechanics.classify_and_build_actions(report)

    assert "Dispatch note" in actions[0].comment_body
    assert "already-merged scope should not be repeated" in actions[0].comment_body


def test_parent_ledger_lists_completed_slices() -> None:
    """Parent ledger comments list each merged PR as a completed slice."""
    report = _audit_report(
        [
            _candidate(
                100,
                "epic: multi-slice tracking",
                classification="parent_or_roadmap",
                prs=[
                    _pr_entry(500, "issue #100 slice A"),
                    _pr_entry(501, "issue #100 slice B"),
                ],
            )
        ]
    )
    actions = closure_mechanics.classify_and_build_actions(report)
    body = actions[0].comment_body

    assert "#500" in body
    assert "#501" in body
    assert "slice A" in body
    assert "slice B" in body


def test_classify_and_build_actions_with_close_issues_override() -> None:
    """Issues specified in close_issues override are marked for closure."""
    report = _audit_report(
        [
            _candidate(12, "add checker", prs=[_pr_entry(99, "fix #12")]),
            _candidate(
                15,
                "epic roadmap",
                classification="parent_or_roadmap",
                prs=[_pr_entry(101, "pr #15")],
            ),
        ]
    )
    actions = closure_mechanics.classify_and_build_actions(report, close_issues={12, 15})

    assert len(actions) == 2
    assert actions[0].issue_number == 12
    assert actions[0].action_type == "close_fully_covered"
    assert actions[0].should_close is True
    assert "#99" in actions[0].comment_body
    assert "Closing as completed" in actions[0].comment_body

    assert actions[1].issue_number == 15
    assert actions[1].action_type == "close_fully_covered"
    assert actions[1].should_close is True
    assert "#101" in actions[1].comment_body
    assert "Closing as completed" in actions[1].comment_body


def test_main_parses_close_issues_cli_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Main parses --close-issues CLI parameter and applies it."""
    report = _audit_report([_candidate(12, "add checker", prs=[_pr_entry(99, "fix #12")])])
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(report)))
    exit_code = closure_mechanics.main(["--close-issues", "12,#15,invalid"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action_count"] == 1
    # Check that issue 12 action is close_fully_covered
    assert payload["results"][0]["issue_number"] == 12
    # In dry-run, should_close is True for the action, let's verify classification in actions
    # Wait, the main outputs results which have "should_close" and "action_type"
    assert payload["results"][0]["action_type"] == "close_fully_covered"
    assert payload["results"][0]["should_close"] is True


def test_build_final_summary_comment_counts_completed_actions() -> None:
    """Final #4437 summary counts only completed action results."""
    result = {
        "results": [
            {
                "issue_number": 12,
                "action_type": "close_fully_covered",
                "closed": True,
                "comment_posted": True,
                "error": None,
            },
            {
                "issue_number": 13,
                "action_type": "residual",
                "closed": False,
                "comment_posted": True,
                "error": None,
            },
            {
                "issue_number": 14,
                "action_type": "parent_ledger",
                "closed": False,
                "comment_posted": True,
                "error": None,
            },
            {
                "issue_number": 15,
                "action_type": "residual",
                "closed": False,
                "comment_posted": False,
                "error": "permission denied",
            },
        ]
    }

    body = closure_mechanics.build_final_summary_comment(result)

    assert "- closed as fully covered: 1" in body
    assert "- residual annotated, kept open: 1" in body
    assert "- parent ledgers updated, kept open: 1" in body
    assert "- no action / skipped / failed: 1" in body
    assert "- #12" in body
    assert "- #13" in body
    assert "- #14" in body
    assert "- #15 residual not completed: permission denied" in body
    assert "No code, queue edits, new issues" in body


def test_main_writes_summary_comment_file(
    tmp_path: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI writes final summary Markdown from dry-run result without posting it."""
    import pathlib

    report = _audit_report([_candidate(20, "file test", prs=[_pr_entry(77, "fix #20")])])
    report_path = pathlib.Path(str(tmp_path)) / "report.json"
    summary_path = pathlib.Path(str(tmp_path)) / "summary.md"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    exit_code = closure_mechanics.main(
        ["--report-file", str(report_path), "--summary-comment-file", str(summary_path)]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = summary_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert payload["summary_comment_file"] == str(summary_path)
    assert "- closed as fully covered: 0" in summary
    assert "- no action / skipped / failed: 1" in summary
    assert "- #20 residual not completed" in summary
