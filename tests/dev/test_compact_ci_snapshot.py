"""Tests for compact CI snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev import compact_ci_snapshot as snapshot


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
