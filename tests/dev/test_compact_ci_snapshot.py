"""Tests for compact CI snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev import compact_ci_snapshot as snapshot


def test_module_entrypoint_is_discoverable_from_repo_root() -> None:
    """The documented package invocation must work without caller PYTHONPATH setup."""
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.dev.compact_ci_snapshot", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    skill_text = (repo_root / ".agents/skills/goal-autopilot/SKILL.md").read_text(encoding="utf-8")
    assert "uv run python -m scripts.dev.compact_ci_snapshot" in skill_text


def test_build_check_summary_exposes_bounded_job_name_sets() -> None:
    """CI summaries should expose targeted names without raw logs."""
    summary = snapshot._build_check_summary(
        [
            {"name": "fast-feedback", "status": "completed", "conclusion": "success"},
            {"name": "examples-smoke", "status": "in_progress", "conclusion": ""},
            {"name": "lint", "status": "completed", "conclusion": "failure"},
        ]
    )

    assert summary.overall == "failure"
    assert summary.superseded == 0
    assert summary.failed_names == ["lint"]
    assert summary.pending_names == ["examples-smoke"]
    assert summary.success_names == ["fast-feedback"]
    assert summary.by_conclusion == {"failure": 1, "pending": 1, "success": 1}


def test_fetch_pr_snapshot_reports_freshness_and_next_action() -> None:
    """PR snapshots should carry expected-head freshness and next useful action."""
    pr_payload = {
        "number": 2712,
        "title": "compact CI state",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "headRefName": "issue-2712",
        "headRefOid": "abc123",
        "statusCheckRollup": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ],
    }
    with patch("scripts.dev.compact_ci_snapshot._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        result = snapshot._fetch_pr_snapshot(
            2712,
            repo="ll7/robot_sf_ll7",
            expected_head_sha="abc123",
        )

    assert result.freshness_key == "pr-2712:abc123"
    assert result.head_matches_expected is True
    assert result.next_action == "review_merge_readiness"
    assert result.checks is not None
    assert result.checks.success_names == ["ci"]


@pytest.mark.parametrize(
    ("replacement_status", "replacement_conclusion", "expected_overall"),
    [
        ("completed", "success", "success"),
        ("in_progress", "", "pending"),
    ],
)
def test_build_check_summary_suppresses_superseded_cancelled_reruns(
    replacement_status: str,
    replacement_conclusion: str,
    expected_overall: str,
) -> None:
    """A newer same-workflow run replaces an older cancellation in the compact view."""
    summary = snapshot._build_check_summary(
        [
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-08-14T01:00:00Z",
            },
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "status": replacement_status,
                "conclusion": replacement_conclusion,
                "startedAt": "2026-08-14T01:05:00Z",
            },
        ]
    )

    assert summary.overall == expected_overall
    assert summary.superseded == 1
    assert summary.total == 1
    assert summary.failed_names == []


def test_build_check_summary_keeps_current_cancellation_fail_closed() -> None:
    """A cancellation without a newer same-workflow replacement remains a failure."""
    summary = snapshot._build_check_summary(
        [
            {
                "__typename": "CheckRun",
                "name": "pr-body-contracts",
                "workflowName": "PR body contracts",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-08-14T01:00:00Z",
            }
        ]
    )

    assert summary.overall == "failure"
    assert summary.superseded == 0
    assert summary.failed_names == ["pr-body-contracts"]


def test_fetch_pr_snapshot_marks_stale_expected_head() -> None:
    """A changed PR head should route the parent to refresh instead of waiting."""
    pr_payload = {
        "number": 2712,
        "title": "compact CI state",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "headRefName": "issue-2712",
        "headRefOid": "new-sha",
        "statusCheckRollup": [
            {"name": "ci", "status": "in_progress", "conclusion": ""},
        ],
    }
    with patch("scripts.dev.compact_ci_snapshot._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        result = snapshot._fetch_pr_snapshot(
            2712,
            repo="ll7/robot_sf_ll7",
            expected_head_sha="old-sha",
        )

    assert result.head_matches_expected is False
    assert result.next_action == "refresh_snapshot_expected_head_changed"
