"""Tests for compact issue-batch snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev.snapshot_issue_batch import (
    expand_issue_numbers,
    main,
    snapshot_blocked_external_issues,
    snapshot_claimable_issues,
    snapshot_issues,
)


def test_expand_issue_numbers_treats_two_values_as_range() -> None:
    """Two ascending values should support the concise batch command."""
    assert expand_issue_numbers([2665, 2667], expand_range=True) == [2665, 2666, 2667]
    assert expand_issue_numbers([2665, 2667], expand_range=False) == [2665, 2667]


def test_snapshot_issues_emits_compact_fields() -> None:
    """Snapshot output should include excerpts and claim state without raw bodies."""
    body = " ".join(["detail"] * 100)
    issue_payload = {
        "number": 2665,
        "title": "workflow: compact issue snapshot",
        "body": body,
        "state": "OPEN",
        "url": "https://github.test/issues/2665",
        "labels": [{"name": "workflow"}, {"name": "enhancement"}],
        "assignees": [{"login": "alice"}],
    }
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(issue_payload),
            stderr="",
        )
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": False,
                "claim_ref": "agent-claims/issue-2665",
                "sha": None,
            }
            payload = snapshot_issues(
                [2665], repo="ll7/robot_sf_ll7", body_limit=20, remote="origin"
            )

    issue = payload["issues"][0]
    assert payload["schema"] == "issue_batch_snapshot.v1"
    assert issue["number"] == 2665
    assert issue["labels"] == ["enhancement", "workflow"]
    assert issue["assignees"] == ["alice"]
    assert issue["body_excerpt"] == "detail detail detail"
    assert issue["body_truncated"] is True
    assert issue["claim"]["claimed"] is False
    assert issue["linked_prs"] == []
    assert "goal_driven_agent_loops" in issue["recommended_context_pack"]


def test_main_returns_error_when_any_issue_fails(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI should keep a JSON payload even when one issue cannot be fetched."""
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
        rc = main(["1", "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["status"] == "error"


def test_snapshot_issues_can_write_context_capsules(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Optional capsules should seed workers without broad rediscovery."""
    issue_payload = {
        "number": 2666,
        "title": "docs: claim boundary first",
        "body": "short body",
        "state": "OPEN",
        "url": "https://github.test/issues/2666",
        "labels": [{"name": "docs"}],
        "assignees": [],
    }
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(issue_payload),
            stderr="",
        )
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": True,
                "claim_ref": "agent-claims/issue-2666",
                "sha": "abc123",
            }
            payload = snapshot_issues(
                [2666],
                repo="ll7/robot_sf_ll7",
                body_limit=300,
                remote="origin",
                capsule_dir=str(tmp_path),
            )

    path = tmp_path / "issue_2666_context_capsule.json"
    assert payload["issues"][0]["context_capsule_path"] == str(path)
    capsule = json.loads(path.read_text())
    assert capsule["schema"] == "issue_context_capsule.v1"
    assert capsule["issue"]["number"] == 2666
    assert capsule["claim"]["claimed"] is True
    assert capsule["files_to_read"] == ["docs/context/INDEX.md"]


def test_snapshot_claimable_issues_includes_classification_without_body() -> None:
    """Claimable discovery should return compact entries without raw body data."""
    issue_list = [
        {
            "number": 2667,
            "title": "claimable issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2667",
            "labels": [{"name": "workflow"}],
            "assignees": [],
            "body": "secret body that should not appear",
        },
        {
            "number": 2668,
            "title": "blocked issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2668",
            "labels": [{"name": "state:blocked"}],
            "assignees": [],
            "body": "another secret body that should not appear",
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.side_effect = [
                {"ok": True, "claimed": False, "claim_ref": "agent-claims/issue-2667", "sha": None},
                {"ok": True, "claimed": False, "claim_ref": "agent-claims/issue-2668", "sha": None},
            ]
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=2,
            )

    assert payload["mode"] == "claimable"
    assert payload["issues"][0]["classification"] == "claimable"
    assert payload["issues"][0]["body_excerpt"] == ""
    assert payload["issues"][0]["body_truncated"] is False
    assert payload["issues"][1]["classification"] == "blocked_label"
    assert "reason" in payload["issues"][1]


def test_snapshot_claimable_issues_excludes_blocked_external_by_default() -> None:
    """Default claim routing should quarantine external-data blockers."""
    issue_list = [
        {
            "number": 2962,
            "title": "workflow issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2962",
            "labels": [{"name": "workflow"}],
            "assignees": [],
        },
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [{"name": "resource:external-data"}, {"name": "state:blocked"}],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": False,
                "claim_ref": "agent-claims/issue",
                "sha": None,
            }
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=2,
            )

    assert [issue["number"] for issue in payload["issues"]] == [2962]
    assert payload["excluded_counts"] == {"blocked_external": 1}
    assert payload["include_blocked_external"] is False


def test_snapshot_claimable_issues_can_include_blocked_external() -> None:
    """Explicit routing should expose quarantined external blockers."""
    issue_list = [
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [{"name": "resource:external-data"}, {"name": "state:blocked"}],
            "assignees": [],
        }
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": False,
                "claim_ref": "agent-claims/issue-2415",
                "sha": None,
            }
            payload = snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7",
                remote="origin",
                body_limit=150,
                limit=1,
                include_blocked_external=True,
            )

    assert payload["include_blocked_external"] is True
    assert payload["issues"][0]["classification"] == "blocked_external"
    assert payload["excluded_counts"] == {"blocked_external": 1}


def test_snapshot_blocked_external_issues_writes_human_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Blocked external report should provide one action and monthly review date per row."""
    from datetime import UTC, datetime

    report_path = tmp_path / "blocked-assets.md"
    issue_list = [
        {
            "number": 2415,
            "title": "data: stage external asset",
            "state": "OPEN",
            "url": "https://github.test/issues/2415",
            "labels": [
                {"name": "resource:external-data"},
                {"name": "state:blocked"},
                {"name": "state:ready"},
            ],
            "assignees": [],
        },
        {
            "number": 2962,
            "title": "workflow: executable now",
            "state": "OPEN",
            "url": "https://github.test/issues/2962",
            "labels": [{"name": "resource:external-data"}, {"name": "state:ready"}],
            "assignees": [],
        },
    ]

    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        payload = snapshot_blocked_external_issues(
            repo="ll7/robot_sf_ll7",
            report_path=str(report_path),
            limit=10,
            now=datetime(2026, 6, 15, tzinfo=UTC),
        )

    assert payload["schema"] == "blocked_external_assets_report.v1"
    assert payload["recommended_state_label"] == "state:blocked-external-input"
    assert payload["row_count"] == 1
    row = payload["rows"][0]
    assert row["number"] == 2415
    assert row["human_action"].count(".") == 1
    assert row["monthly_review_date"] == "2026-07-01"
    assert "add `state:blocked-external-input`" in row["label_recommendation"]
    assert "remove `state:ready`" in row["label_recommendation"]
    assert "#2415 data: stage external asset" in report_path.read_text(encoding="utf-8")


def test_main_claimable_mode_can_be_called_without_issue_numbers() -> None:  # type: ignore[no-untyped-def]
    """CLI should run compact claim discovery when issue numbers are intentionally omitted."""
    issue_list = [
        {
            "number": 2669,
            "title": "no-arg mode issue",
            "state": "OPEN",
            "url": "https://github.test/issues/2669",
            "labels": [],
            "assignees": [],
        }
    ]
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(issue_list), stderr="")
        with patch("scripts.dev.snapshot_issue_batch.status_issue") as claim:
            claim.return_value = {
                "ok": True,
                "claimed": False,
                "claim_ref": "agent-claims/issue-2669",
                "sha": None,
            }
            rc = main(["--claimable", "--json", "--limit", "1"])

    assert rc == 0
