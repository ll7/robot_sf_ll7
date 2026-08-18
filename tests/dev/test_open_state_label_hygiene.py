"""Tests for the report-only open-issue stale-state-label guard."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import open_state_label_hygiene


def _issue_row(
    number: int,
    *,
    labels: list[str],
    state: str = "open",
    resource: str = "issues",
) -> dict[str, object]:
    """Build a mixed GitHub issue/PR row for discovery tests."""
    return {
        "number": number,
        "title": f"issue {number}",
        "html_url": f"https://github.com/ll7/robot_sf_ll7/{resource}/{number}",
        "state": state,
        "labels": [{"name": label} for label in labels],
    }


def _result(payload: object, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Build a deterministic REST response for injected runners."""
    return subprocess.CompletedProcess(
        args=("gh", "api"),
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr="" if returncode == 0 else "request failed",
    )


def test_collect_active_issues_deduplicates_labels_and_excludes_prs() -> None:
    """Only open issue rows with active labels should enter the timeline scan."""
    rows = {
        "state:ready": [
            _issue_row(12, labels=["state:ready", "state:working"]),
            _issue_row(99, labels=["state:ready"], resource="pull"),
        ],
        "state:working": [_issue_row(12, labels=["state:ready", "state:working"])],
    }

    issues = open_state_label_hygiene.collect_active_issues(rows)

    assert [(issue.number, issue.active_labels) for issue in issues] == [
        (12, ("state:ready", "state:working"))
    ]


def test_reconcile_active_issues_uses_current_state_and_labels() -> None:
    """Search/index rows cannot keep a closed or relabeled issue in the report."""
    candidates = [
        open_state_label_hygiene.ActiveIssue(
            number=12,
            title="stale search row",
            url="https://github.com/ll7/robot_sf_ll7/issues/12",
            state="open",
            active_labels=("state:ready",),
        ),
        open_state_label_hygiene.ActiveIssue(
            number=13,
            title="current row",
            url="https://github.com/ll7/robot_sf_ll7/issues/13",
            state="open",
            active_labels=("state:running",),
        ),
    ]

    def runner(path: str) -> subprocess.CompletedProcess[str]:
        if path.endswith("/12"):
            return _result(_issue_row(12, labels=["state:ready"], state="closed"))
        return _result(_issue_row(13, labels=["state:running"]))

    reconciled = open_state_label_hygiene.reconcile_active_issues(
        repo="ll7/robot_sf_ll7",
        candidates=candidates,
        runner=runner,
    )

    assert [issue.number for issue in reconciled] == [13]


def test_timeline_candidate_requires_merged_pr_reference() -> None:
    """Unmerged or non-PR timeline references must not become stale candidates."""
    unmerged = {
        "event": "cross-referenced",
        "source": {
            "issue": {
                "number": 901,
                "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                "pull_request": {},
            }
        },
    }
    merged = {
        "event": "cross-referenced",
        "created_at": "2026-08-18T10:00:00Z",
        "source": {
            "issue": {
                "number": 901,
                "title": "Fix #12",
                "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                "pull_request": {"merged_at": "2026-08-18T09:00:00Z"},
            }
        },
    }

    assert open_state_label_hygiene._timeline_candidate(issue_number=12, event=unmerged) is None
    assert open_state_label_hygiene._timeline_candidate(issue_number=12, event=merged) == (
        901,
        "Fix #12",
        "https://github.com/ll7/robot_sf_ll7/pull/901",
        "2026-08-18T10:00:00Z",
    )
    assert (
        open_state_label_hygiene._timeline_candidate(
            issue_number=12,
            event=merged,
            repo="other/repository",
        )
        is None
    )


def test_discover_merged_references_verifies_merge_commit() -> None:
    """A timeline reference is reportable only after the current PR has a merge SHA."""
    issue = open_state_label_hygiene.ActiveIssue(
        number=12,
        title="stale issue",
        url="https://github.com/ll7/robot_sf_ll7/issues/12",
        state="open",
        active_labels=("state:ready",),
    )

    seen_paths: list[str] = []

    def runner(path: str) -> subprocess.CompletedProcess[str]:
        seen_paths.append(path)
        if path.startswith("repos/ll7/robot_sf_ll7/issues/12/timeline"):
            return _result(
                [
                    {
                        "event": "cross-referenced",
                        "created_at": "2026-08-18T10:00:00Z",
                        "source": {
                            "issue": {
                                "number": 901,
                                "title": "Fix #12",
                                "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                                "pull_request": {"merged_at": "2026-08-18T09:00:00Z"},
                            }
                        },
                    }
                ]
            )
        if path == "repos/ll7/robot_sf_ll7/pulls/901":
            return _result(
                {
                    "number": 901,
                    "title": "Fix #12",
                    "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                    "merged_at": "2026-08-18T09:00:00Z",
                    "merged": True,
                    "merge_commit_sha": "abc123",
                }
            )
        raise AssertionError(f"unexpected path: {path}")

    references, metadata = open_state_label_hygiene.discover_merged_references(
        repo="ll7/robot_sf_ll7",
        issues=[issue],
        runner=runner,
    )

    assert metadata["complete_for_open_issues"] is True
    assert references[12][0].number == 901
    assert references[12][0].merge_commit_sha == "abc123"
    assert references[12][0].coverage_source == "issue_timeline_merged_pr"
    assert seen_paths[0] == ("repos/ll7/robot_sf_ll7/issues/12/timeline?per_page=100&page=1")


def test_discover_merged_references_fails_closed_on_timeline_error() -> None:
    """An unavailable timeline is not equivalent to a clean open-issue queue."""
    issue = open_state_label_hygiene.ActiveIssue(
        number=12,
        title="stale issue",
        url="https://github.com/ll7/robot_sf_ll7/issues/12",
        state="open",
        active_labels=("state:ready",),
    )

    def runner(path: str) -> subprocess.CompletedProcess[str]:
        return _result([], returncode=1)

    references, metadata = open_state_label_hygiene.discover_merged_references(
        repo="ll7/robot_sf_ll7",
        issues=[issue],
        runner=runner,
    )

    assert references == {}
    assert metadata["complete_for_open_issues"] is False
    assert metadata["errors"]


def test_discover_merged_references_ignores_still_open_pr() -> None:
    """A current open PR is not merged coverage even if a timeline reference exists."""
    issue = open_state_label_hygiene.ActiveIssue(
        number=12,
        title="stale issue",
        url="https://github.com/ll7/robot_sf_ll7/issues/12",
        state="open",
        active_labels=("state:ready",),
    )

    def runner(path: str) -> subprocess.CompletedProcess[str]:
        if path.startswith("repos/ll7/robot_sf_ll7/issues/12/timeline"):
            return _result(
                [
                    {
                        "event": "referenced",
                        "created_at": "2026-08-18T10:00:00Z",
                        "source": {
                            "issue": {
                                "number": 901,
                                "title": "Fix #12",
                                "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                                "pull_request": {"merged_at": "2026-08-18T09:00:00Z"},
                            }
                        },
                    }
                ]
            )
        if path == "repos/ll7/robot_sf_ll7/pulls/901":
            return _result(
                {
                    "number": 901,
                    "title": "Fix #12",
                    "state": "open",
                    "html_url": "https://github.com/ll7/robot_sf_ll7/pull/901",
                    "merged": False,
                    "merged_at": None,
                    "merge_commit_sha": None,
                }
            )
        raise AssertionError(f"unexpected path: {path}")

    references, metadata = open_state_label_hygiene.discover_merged_references(
        repo="ll7/robot_sf_ll7",
        issues=[issue],
        runner=runner,
    )

    assert references == {}
    assert metadata["complete_for_open_issues"] is True


def test_build_report_requires_complete_coverage_and_never_authorizes_writes() -> None:
    """The report distinguishes findings from incomplete evidence and stays read-only."""
    issue = open_state_label_hygiene.ActiveIssue(
        number=12,
        title="stale issue",
        url="https://github.com/ll7/robot_sf_ll7/issues/12",
        state="open",
        active_labels=("state:ready", "state:working"),
    )
    merged = open_state_label_hygiene.MergedPullRequest(
        issue_number=12,
        number=901,
        title="Fix #12",
        url="https://github.com/ll7/robot_sf_ll7/pull/901",
        merged_at="2026-08-18T09:00:00Z",
        merge_commit_sha="abc123",
        coverage_source="issue_timeline_merged_pr",
        timeline_event_created_at="2026-08-18T10:00:00Z",
    )

    report = open_state_label_hygiene.build_report(
        repo="ll7/robot_sf_ll7",
        checked_labels=open_state_label_hygiene.OPEN_ACTIVE_STATE_LABELS,
        issues=[issue],
        references_by_issue={12: (merged,)},
        discovery_metadata=[],
        coverage_metadata={
            "complete_for_open_issues": True,
            "truncated": False,
            "errors": [],
            "timelines": [],
        },
    )

    assert report["ok"] is False
    assert report["candidate_count"] == 1
    assert report["read_only"] is True
    assert report["issue_writes"] is False
    assert report["project_writes"] is False
    assert report["issues"][0]["recommended_action"] == "verify_exact_fix_then_close_or_relabel"


def test_main_clean_report_uses_injected_discovery(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A complete empty inventory returns zero without requiring GitHub access."""
    monkeypatch.setattr(
        open_state_label_hygiene,
        "fetch_open_issues_by_label",
        lambda **kwargs: ({label: [] for label in kwargs["labels"]}, []),
    )

    exit_code = open_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["candidate_count"] == 0
    assert payload["complete_for_open_issues"] is True


def test_main_rejects_non_state_label(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI must not be repurposed as a broad arbitrary-label scan."""
    exit_code = open_state_label_hygiene.main(["--label", "priority:1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert "active labels" in payload["error"]


def test_main_rejects_state_label_outside_open_active_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The open guard must not silently expand to blocked or unrelated state labels."""
    exit_code = open_state_label_hygiene.main(["--label", "state:blocked"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["checked_labels"] == ["state:blocked"]
