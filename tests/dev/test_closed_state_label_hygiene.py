"""Tests for closed-issue state-label hygiene helpers."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import closed_state_label_hygiene


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

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["stale_count"] == 1
    assert payload["issues"][0]["number"] == 12


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


def test_confirm_issue_closed_reads_state_via_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """confirm_issue_closed returns True only for closed, non-PR issues."""
    captured: dict[str, list[str]] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(
            args=tuple(command),
            returncode=0,
            stdout=(
                '{"number": 12, "state": "CLOSED", '
                '"url": "https://github.com/ll7/robot_sf_ll7/issues/12"}'
            ),
        )

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    assert closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)
    assert captured["command"][:3] == ["gh", "issue", "view"]


def test_confirm_issue_closed_is_false_for_open_or_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open issues and pull requests must fail the read-then-write guard."""

    def fake_run_open(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=tuple(command),
            returncode=0,
            stdout='{"state": "OPEN", "url": "https://github.com/ll7/robot_sf_ll7/issues/12"}',
        )

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run_open)
    assert not closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)

    def fake_run_pr(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=tuple(command),
            returncode=0,
            stdout='{"state": "CLOSED", "url": "https://github.com/ll7/robot_sf_ll7/pull/12"}',
        )

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run_pr)
    assert not closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)


def test_remove_label_command_targets_only_one_label() -> None:
    """The remove command edits a single issue and removes exactly one named label."""
    command = closed_state_label_hygiene.build_remove_label_command(
        repo="ll7/robot_sf_ll7", number=12, label="state:ready"
    )
    assert command[:3] == ["gh", "issue", "edit"]
    assert command[command.index("--remove-label") + 1] == "state:ready"


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
        if command[:3] == ["gh", "issue", "view"]:
            return subprocess.CompletedProcess(
                args=tuple(command),
                returncode=0,
                stdout='{"state": "CLOSED", "url": "https://github.com/ll7/robot_sf_ll7/issues/12"}',
            )
        edits.append(command)
        return subprocess.CompletedProcess(args=tuple(command), returncode=0, stdout="")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fake_fetch)
    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["fix_applied"] is True
    assert payload["read_only"] is False
    assert payload["fix_actions"][0]["removed_labels"] == ["state:ready"]
    assert any(cmd[:3] == ["gh", "issue", "edit"] for cmd in edits)
