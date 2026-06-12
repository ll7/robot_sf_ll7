"""Tests for compact PR queue snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev.snapshot_pr_queue import main, snapshot_prs


def test_snapshot_prs_emits_headline_state() -> None:
    """PR snapshots should summarize CI/review state without raw rollups."""
    pr_payload = {
        "number": 2679,
        "title": "compact PR state",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2679",
        "labels": [{"name": "merge-ready"}],
        "headRefName": "feature",
        "headRefOid": "abc123",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
            {"name": "lint", "status": "completed", "conclusion": "success"},
        ],
        "reviews": [{"state": "APPROVED"}, {"state": "COMMENTED"}],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2679], repo="ll7/robot_sf_ll7")

    pr = payload["prs"][0]
    assert payload["schema"] == "pr_queue_snapshot.v1"
    assert pr["number"] == 2679
    assert pr["head_sha"] == "abc123"
    assert pr["checks"]["overall"] == "success"
    assert pr["checks"]["names"] == ["ci", "lint"]
    assert pr["reviews"] == {"APPROVED": 1, "COMMENTED": 1}
    assert pr["next_action"] == "merge_readiness_local_check"


def test_snapshot_prs_pending_next_action() -> None:
    """Pending checks should route the parent toward a monitor instead of polling."""
    pr_payload = {
        "number": 2680,
        "title": "pending PR",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "def456",
        "mergeable": "UNKNOWN",
        "statusCheckRollup": [{"name": "ci", "status": "in_progress", "conclusion": ""}],
        "reviews": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2680], repo="ll7/robot_sf_ll7")

    assert payload["prs"][0]["checks"]["overall"] == "pending"
    assert payload["prs"][0]["next_action"] == "await_ci_or_start_read_only_monitor"


def test_main_requires_pr_number(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI should fail compactly when no PRs are provided."""
    rc = main(["--json"])
    assert rc == 1
    assert "at least one PR" in capsys.readouterr().err
