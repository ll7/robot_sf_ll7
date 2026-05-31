"""Tests for closed-issue state-label hygiene helpers."""

from __future__ import annotations

import json

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
                "isPullRequest": True,
                "labels": [{"name": "state:ready"}],
            }
        ],
    }

    assert closed_state_label_hygiene.collect_stale_issues(rows_by_label) == []


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
    assert "isPullRequest" in command[command.index("--json") + 1].split(",")
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
