"""Tests for closed-issue state-label hygiene helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.dev import closed_state_label_hygiene


def _mock_current_rest_issue(
    monkeypatch: pytest.MonkeyPatch,
    *,
    labels: list[str],
    title: str = "current issue",
    number: int = 12,
) -> None:
    """Install a current closed-issue REST response for reconciliation tests."""

    monkeypatch.setattr(
        "scripts.dev.gh_issue_rest.fetch_issue",
        lambda requested_number, **kwargs: {
            "number": requested_number,
            "status": "ok",
            "title": title,
            "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
            "state": "CLOSED",
            "labels": labels,
        },
    )


def test_collect_stale_issues_aggregates_closed_issue_state_labels() -> None:
    """Closed issues should be reported once with all stale live state labels."""
    rows_by_label = {
        "state:ready": [
            {
                "number": 12,
                "title": "done but still queued",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                "state": "closed",
                "labels": [{"name": "state:ready"}, {"name": "workflow"}],
            },
            {
                "number": 13,
                "title": "open issue should not count",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/13",
                "state": "open",
                "labels": [{"name": "state:ready"}],
            },
        ],
        "state:blocked": [
            {
                "number": 12,
                "title": "done but still queued",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                "state": "closed",
                "labels": [{"name": "state:ready"}, {"name": "state:blocked"}],
            }
        ],
    }

    stale = closed_state_label_hygiene.collect_stale_issues(rows_by_label)

    assert [issue.number for issue in stale] == [12]
    assert stale[0].stale_labels == ("state:blocked", "state:ready")


def test_collect_stale_issues_ignores_pull_request_rows() -> None:
    """The guard is issue-specific even if a caller supplies PR-shaped search rows."""
    rows_by_label = {
        "state:ready": [
            {
                "number": 12,
                "title": "closed PR with a state label",
                "url": "https://github.com/ll7/robot_sf_ll7/pull/12",
                "state": "closed",
                "labels": [{"name": "state:ready"}],
            }
        ],
    }

    assert closed_state_label_hygiene.collect_stale_issues(rows_by_label) == []


def test_reconcile_stale_issues_suppresses_search_index_lag() -> None:
    """A removed REST label should override the stale label still returned by search."""
    candidates = [_stale(12, ("state:ready",))]

    stale = closed_state_label_hygiene.reconcile_stale_issues(
        repo="ll7/robot_sf_ll7",
        candidates=candidates,
        fetch_issue=lambda number, **kwargs: {
            "number": number,
            "status": "ok",
            "title": "done and cleaned up",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
            "state": "CLOSED",
            "labels": ["priority:4", "technical-debt"],
        },
    )

    assert stale == []


def test_reconcile_stale_issues_preserves_current_rest_live_labels() -> None:
    """A genuinely stale live label in the current REST record remains a failure."""
    candidates = [_stale(12, ("state:ready", "state:running"))]

    stale = closed_state_label_hygiene.reconcile_stale_issues(
        repo="ll7/robot_sf_ll7",
        candidates=candidates,
        fetch_issue=lambda number, **kwargs: {
            "number": number,
            "status": "ok",
            "title": "current REST title",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
            "state": "CLOSED",
            "labels": ["priority:4", "state:running"],
        },
    )

    assert len(stale) == 1
    assert stale[0].title == "current REST title"
    assert stale[0].stale_labels == ("state:running",)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/ll7/robot_sf_ll7/pull/12", True),
        ("https://github.com/ll7/robot_sf_ll7/issues/12?next=/pull/12", False),
        ("https://github.com/ll7/robot_sf_ll7/issues/pull", False),
        ("https://github.com/ll7/robot_sf_ll7/pull/not-a-number", False),
        ("https://github.com/ll7/robot_sf_ll7/pulls/12", False),
        (None, False),
    ],
)
def test_is_pull_request_url_requires_canonical_pull_path(url: object, expected: bool) -> None:
    """PR detection should not treat arbitrary '/pull/' substrings as pull requests."""
    assert closed_state_label_hygiene._is_pull_request_url(url) is expected


def test_build_report_emits_machine_readable_failure_summary() -> None:
    """Reports should expose a stable failure summary when stale labels exist."""
    stale = [
        closed_state_label_hygiene.StaleIssue(
            number=12,
            title="done but still queued",
            url="https://github.com/ll7/robot_sf_ll7/issues/12",
            state="closed",
            stale_labels=("state:ready",),
        )
    ]

    report = closed_state_label_hygiene.build_report(
        repo="ll7/robot_sf_ll7",
        checked_labels=("state:ready", "state:running", "state:blocked"),
        stale_issues=stale,
    )

    assert report["schema"] == "closed_state_label_hygiene.v1"
    assert report["ok"] is False
    assert report["read_only"] is True
    assert report["project_writes"] is False
    assert report["stale_count"] == 1
    assert report["issues"][0]["stale_labels"] == ["state:ready"]


def test_build_search_command_uses_read_only_closed_issue_search() -> None:
    """The GitHub command should only search issues and avoid Project writes."""
    command = closed_state_label_hygiene.build_search_command(
        repo="ll7/robot_sf_ll7",
        label="state:ready",
        limit=200,
    )

    assert command[:3] == ["gh", "search", "issues"]
    assert "--state" in command
    assert command[command.index("--state") + 1] == "closed"
    assert "--label" in command
    assert command[command.index("--label") + 1] == "state:ready"
    assert "url" in command[command.index("--json") + 1].split(",")
    assert "isPullRequest" not in command[command.index("--json") + 1].split(",")
    assert "--project" not in command
    assert "edit" not in command


def test_closure_workflow_routes_state_label_io_through_rest_helpers() -> None:
    """The closure Action must not use native issue label commands."""
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "strip-closed-state-labels.yml"
    ).read_text(encoding="utf-8")
    code = "\n".join(line for line in workflow.splitlines() if not line.lstrip().startswith("#"))

    assert "python -m scripts.dev.gh_issue_rest view" in code
    assert "python -m scripts.dev.gh_pr_label_rest list" in code
    assert "python -m scripts.dev.gh_pr_label_rest remove" in code
    assert "if ! live_labels_output=$(" in code
    assert "mapfile -t LIVE_LABELS < <(" not in code
    assert "gh issue view" not in code
    assert "gh issue edit" not in code


def test_main_returns_nonzero_json_summary_without_live_github(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should be testable with an injected fetcher and emit JSON on failure."""

    def fake_fetch(
        *,
        repo: str,
        labels: tuple[str, ...],
        limit: int,
    ) -> dict[str, list[dict[str, object]]]:
        assert repo == "ll7/robot_sf_ll7"
        assert labels == ("state:ready", "state:running", "state:blocked")
        assert limit == 1000
        return {
            "state:ready": [
                {
                    "number": 12,
                    "title": "done but still queued",
                    "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    "state": "closed",
                    "labels": [{"name": "state:ready"}],
                }
            ]
        }

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fake_fetch)
    _mock_current_rest_issue(
        monkeypatch,
        labels=["state:ready"],
        title="done but still queued",
    )

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["stale_count"] == 1
    assert payload["issues"][0]["number"] == 12


def test_main_suppresses_search_label_absent_from_current_rest_issue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search-index lag should preserve the successful read-only JSON and exit contracts."""
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label",
        lambda **kwargs: {
            "state:ready": [
                {
                    "number": 12,
                    "title": "stale search row",
                    "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    "state": "closed",
                    "labels": [{"name": "state:ready"}],
                }
            ]
        },
    )
    _mock_current_rest_issue(monkeypatch, labels=["priority:4", "technical-debt"])

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema"] == "closed_state_label_hygiene.v1"
    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["stale_count"] == 0
    assert payload["issues"] == []


def test_run_search_command_reports_missing_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing gh should produce an actionable runtime error."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GitHub CLI 'gh' was not found"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def test_run_search_command_preserves_captured_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured gh stderr should appear in the machine-readable error path."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=("gh", "search", "issues"),
            stderr="authentication required",
        )

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="authentication required"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def test_run_search_command_reports_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed gh output should be diagnosed before row filtering."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=("gh",), returncode=0, stdout="not-json")

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to parse GitHub CLI JSON output"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def _rest_page(rows: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    """Build a successful mocked ``gh api`` page."""
    return subprocess.CompletedProcess(args=("gh", "api"), returncode=0, stdout=json.dumps(rows))


def _rest_issue(number: int, title: str, labels: list[str]) -> dict[str, object]:
    """Build a raw REST issue row."""
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "state": "closed",
        "labels": [{"name": label} for label in labels],
    }


def test_fetch_closed_issues_by_label_rest_paginates_until_partial_page() -> None:
    """REST discovery reads bounded pages and emits existing candidate row shape."""
    seen_paths: list[str] = []

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        seen_paths.append(path)
        if "page=1" in path:
            return _rest_page(
                [
                    _rest_issue(12, "done but still queued", ["state:ready"]),
                    _rest_issue(13, "also queued", ["state:ready"]),
                ]
            )
        if "page=2" in path:
            return _rest_page([_rest_issue(14, "last queued", ["state:ready"])])
        raise AssertionError(f"unexpected REST path: {path}")

    result = closed_state_label_hygiene.fetch_closed_issues_by_label_rest(
        repo="ll7/robot_sf_ll7",
        labels=("state:ready",),
        max_pages=3,
        per_page=2,
        gh_api=fake_gh_api,
    )

    assert [row["number"] for row in result.rows_by_label["state:ready"]] == [12, 13, 14]
    assert result.source == "rest"
    assert result.truncations == [
        {
            "label": "state:ready",
            "truncated": False,
            "row_count": 3,
            "limit": 6,
            "pages_read": 2,
            "per_page": 2,
            "page_budget": 3,
            "source": "rest",
            "note": "",
        }
    ]
    assert seen_paths == [
        "repos/ll7/robot_sf_ll7/issues?state=closed&labels=state%3Aready&per_page=2&page=1",
        "repos/ll7/robot_sf_ll7/issues?state=closed&labels=state%3Aready&per_page=2&page=2",
    ]


def test_fetch_closed_issues_by_label_rest_marks_page_budget_exhaustion() -> None:
    """A full final REST page is marked as partial inventory."""

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        page = 1 if "page=1" in path else 2
        offset = 10 * page
        return _rest_page(
            [
                _rest_issue(offset + 1, "queued a", ["state:ready"]),
                _rest_issue(offset + 2, "queued b", ["state:ready"]),
            ]
        )

    result = closed_state_label_hygiene.fetch_closed_issues_by_label_rest(
        repo="ll7/robot_sf_ll7",
        labels=("state:ready",),
        max_pages=2,
        per_page=2,
        gh_api=fake_gh_api,
    )

    marker = result.truncations[0]
    assert marker["truncated"] is True
    assert marker["row_count"] == 4
    assert marker["pages_read"] == 2
    assert "raise --max-rest-pages" in marker["note"]


def test_discover_closed_issues_falls_back_to_rest_when_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphQL/search exhaustion should not block REST candidate discovery."""

    def fail_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise RuntimeError("GitHub CLI command failed (gh search issues): GraphQL: API rate limit")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fail_search)
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label_rest",
        lambda **kwargs: closed_state_label_hygiene.CandidateDiscoveryResult(
            rows_by_label={
                "state:ready": [
                    {
                        "number": 12,
                        "title": "done but still queued",
                        "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                        "state": "closed",
                        "labels": [{"name": "state:ready"}],
                    }
                ]
            },
            truncations=[],
            source="rest",
        ),
    )

    result = closed_state_label_hygiene.discover_closed_issues_by_label(
        repo="ll7/robot_sf_ll7",
        labels=("state:ready",),
        limit=1000,
        max_rest_pages=2,
    )

    assert result.source == "rest"
    assert result.rows_by_label["state:ready"][0]["number"] == 12


def test_discover_closed_issues_keeps_non_graphql_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authentication and repository errors must not be hidden by REST fallback."""
    search_error = RuntimeError("GitHub CLI command failed: authentication required")

    def fail_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise search_error

    def fail_rest(**kwargs: object) -> closed_state_label_hygiene.CandidateDiscoveryResult:
        raise AssertionError("REST fallback must not mask non-GraphQL failures")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fail_search)
    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label_rest", fail_rest)

    with pytest.raises(RuntimeError, match="authentication required"):
        closed_state_label_hygiene.discover_closed_issues_by_label(
            repo="ll7/robot_sf_ll7",
            labels=("state:ready",),
            limit=1000,
        )


def test_main_rest_fallback_truncation_exits_nonzero_without_silent_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partial REST fallback inventory must not be treated as a complete clean audit."""

    def fail_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise RuntimeError("GitHub CLI command failed (gh search issues): GraphQL: API rate limit")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fail_search)
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label_rest",
        lambda **kwargs: closed_state_label_hygiene.CandidateDiscoveryResult(
            rows_by_label={"state:ready": []},
            truncations=[
                {
                    "label": "state:ready",
                    "truncated": True,
                    "row_count": 100,
                    "limit": 100,
                    "pages_read": 1,
                    "per_page": 100,
                    "page_budget": 1,
                    "source": "rest",
                    "note": "closed-issue REST label inventory may be partial",
                }
            ],
            source="rest",
        ),
    )

    exit_code = closed_state_label_hygiene.main(
        ["--repo", "ll7/robot_sf_ll7", "--label", "state:ready", "--max-rest-pages", "1"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is True
    assert payload["stale_count"] == 0
    assert payload["candidate_discovery_source"] == "rest"
    assert payload["truncated_any"] is True


def test_main_skips_fix_when_candidate_discovery_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partial inventory must not mutate any issue through ``--fix``."""
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("GraphQL: rate limit")),
    )
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label_rest",
        lambda **kwargs: closed_state_label_hygiene.CandidateDiscoveryResult(
            rows_by_label={
                "state:ready": [
                    {
                        "number": 12,
                        "title": "stale candidate",
                        "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                        "state": "closed",
                        "labels": [{"name": "state:ready"}],
                    }
                ]
            },
            truncations=[{"label": "state:ready", "truncated": True}],
            source="rest",
        ),
    )
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "reconcile_stale_issues",
        lambda **kwargs: [_stale(12, ("state:ready",))],
    )

    def fail_if_mutated(**kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("partial discovery must not enter fix mode")

    monkeypatch.setattr(closed_state_label_hygiene, "fix_stale_issues", fail_if_mutated)

    exit_code = closed_state_label_hygiene.main(
        ["--repo", "ll7/robot_sf_ll7", "--label", "state:ready", "--fix"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["fix_applied"] is False
    assert payload["fix_skipped"] == "candidate discovery was truncated"
    assert payload["truncated_any"] is True


def test_main_rejects_nonpositive_rest_page_budget(capsys: pytest.CaptureFixture[str]) -> None:
    """REST fallback bounds should be validated before candidate discovery."""
    exit_code = closed_state_label_hygiene.main(["--max-rest-pages", "0"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert "--max-rest-pages must be >= 1" in payload["error"]


def _stale(number: int, labels: tuple[str, ...]) -> closed_state_label_hygiene.StaleIssue:
    """Build a StaleIssue fixture for fix-mode tests."""
    return closed_state_label_hygiene.StaleIssue(
        number=number,
        title=f"issue {number}",
        url=f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        state="closed",
        stale_labels=labels,
    )


def test_fix_stale_issues_removes_only_live_state_labels_from_closed_issues() -> None:
    """Fix mode strips the live state labels after re-confirming each issue is closed."""
    removed: list[tuple[int, str]] = []
    confirmed: list[int] = []

    def fake_confirm(*, repo: str, number: int) -> bool:
        confirmed.append(number)
        return True

    def fake_remove(number: int, label: str) -> None:
        removed.append((number, label))

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[_stale(12, ("state:blocked", "state:ready"))],
        confirm_closed=fake_confirm,
        remove_label=fake_remove,
    )

    assert confirmed == [12]
    assert removed == [(12, "state:blocked"), (12, "state:ready")]
    assert actions == [
        {"number": 12, "skipped": False, "removed_labels": ["state:blocked", "state:ready"]}
    ]


def test_fix_stale_issues_does_not_touch_issue_that_is_not_closed() -> None:
    """Read-then-write guard: an issue reported as not-closed must not be edited."""
    removed: list[tuple[int, str]] = []

    def fake_confirm(*, repo: str, number: int) -> bool:
        return False

    def fake_remove(number: int, label: str) -> None:
        removed.append((number, label))

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[_stale(12, ("state:ready",))],
        confirm_closed=fake_confirm,
        remove_label=fake_remove,
    )

    assert removed == []
    assert actions == [
        {"number": 12, "skipped": True, "reason": "not_closed", "removed_labels": []}
    ]


def test_fix_stale_issues_is_a_no_op_when_there_are_no_stale_issues() -> None:
    """Fix mode performs no reads or writes when nothing is stale."""

    def fail_confirm(*, repo: str, number: int) -> bool:  # pragma: no cover - must not run
        raise AssertionError("confirm should not be called when there are no stale issues")

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[],
        confirm_closed=fail_confirm,
        remove_label=lambda number, label: None,
    )

    assert actions == []


def test_fix_stale_issues_only_removes_documented_label_set() -> None:
    """Only labels in LIVE_STATE_LABELS are removed even if other labels slip in."""
    removed: list[tuple[int, str]] = []

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[_stale(12, ("state:ready", "workflow", "priority:high"))],
        confirm_closed=lambda *, repo, number: True,
        remove_label=lambda number, label: removed.append((number, label)),
    )

    assert removed == [(12, "state:ready")]
    assert actions[0]["removed_labels"] == ["state:ready"]
    # The fix set is exactly the single-source-of-truth live label tuple.
    assert all(label in closed_state_label_hygiene.LIVE_STATE_LABELS for _, label in removed)


def test_confirm_issue_closed_reads_state_via_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """confirm_issue_closed returns True only for closed, non-PR issues."""

    def fake_fetch_issue(number: int, **kwargs: object) -> dict:
        return {
            "number": number,
            "status": "ok",
            "state": "CLOSED",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
        }

    monkeypatch.setattr("scripts.dev.gh_issue_rest.fetch_issue", fake_fetch_issue)

    assert closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)


def test_confirm_issue_closed_is_false_for_open_or_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open issues and pull requests must fail the read-then-write guard."""

    def fake_fetch_issue_open(number: int, **kwargs: object) -> dict:
        return {
            "number": number,
            "status": "ok",
            "state": "OPEN",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
        }

    monkeypatch.setattr("scripts.dev.gh_issue_rest.fetch_issue", fake_fetch_issue_open)
    assert not closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)

    def fake_fetch_issue_pr(number: int, **kwargs: object) -> dict:
        return {
            "number": number,
            "status": "ok",
            "state": "CLOSED",
            "url": "https://github.com/ll7/robot_sf_ll7/pull/12",
        }

    monkeypatch.setattr("scripts.dev.gh_issue_rest.fetch_issue", fake_fetch_issue_pr)
    assert not closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)


def test_remove_label_command_targets_only_one_label() -> None:
    """The command invokes the verified REST label helper for one named label."""
    command = closed_state_label_hygiene.build_remove_label_command(
        repo="ll7/robot_sf_ll7", number=12, label="state:ready"
    )
    assert command[:6] == [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.dev.gh_pr_label_rest",
        "remove",
    ]
    assert command[command.index("--label") + 1] == "state:ready"
    assert command[command.index("--repo") + 1] == "ll7/robot_sf_ll7"


def test_main_fix_mode_strips_labels_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --fix removes labels via mocked gh and reports success without network access."""

    def fake_fetch(
        *, repo: str, labels: tuple[str, ...], limit: int
    ) -> dict[str, list[dict[str, object]]]:
        return {
            "state:ready": [
                {
                    "number": 12,
                    "title": "done but still queued",
                    "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    "state": "closed",
                    "labels": [{"name": "state:ready"}],
                }
            ]
        }

    edits: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        edits.append(command)
        return subprocess.CompletedProcess(args=tuple(command), returncode=0, stdout="")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fake_fetch)
    _mock_current_rest_issue(monkeypatch, labels=["state:ready"])
    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["fix_applied"] is True
    assert payload["read_only"] is False
    assert payload["fix_actions"][0]["removed_labels"] == ["state:ready"]
    assert any(
        cmd[:6] == ["uv", "run", "python", "-m", "scripts.dev.gh_pr_label_rest", "remove"]
        for cmd in edits
    )


def test_fix_stale_issues_fails_closed_when_rest_helper_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified REST helper error must stop fix mode instead of reporting success."""

    def fail_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr='{"error": "HTTP 403: forbidden", "status": "error"}',
        )

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="HTTP 403: forbidden"):
        closed_state_label_hygiene.fix_stale_issues(
            repo="ll7/robot_sf_ll7",
            stale_issues=[_stale(12, ("state:ready",))],
            confirm_closed=lambda *, repo, number: True,
        )


def test_main_fix_mode_rejects_labels_outside_live_state_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fix mode must fail closed before searching or editing an arbitrary label."""

    def fail_fetch(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise AssertionError("GitHub search must not run for an unsupported fix label")

    def fail_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("GitHub edit must not run for an unsupported fix label")

    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label",
        fail_fetch,
    )
    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fail_run)

    exit_code = closed_state_label_hygiene.main(["--label", "workflow", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["read_only"] is False
    assert "--fix only supports live state labels" in payload["error"]
