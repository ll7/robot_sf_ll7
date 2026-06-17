"""Tests for compact PR queue snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev.snapshot_pr_queue import (
    COMMENT_BODY_LIMIT,
    main,
    snapshot_active_prs,
    snapshot_prs,
    write_raw_review_comments_artifact,
)


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
        "comments": [
            {
                "author": {"login": "reviewer"},
                "createdAt": "2026-06-01T00:00:00Z",
                "body": "A short review note.",
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-06-01T01:00:00Z",
                "body": "Another short note.",
            },
        ],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2679], repo="ll7/robot_sf_ll7", expected_head_sha="abc123")

    pr = payload["prs"][0]
    assert payload["schema"] == "pr_queue_snapshot.v1"
    assert payload["route_health_overview"]["healthy"] == 1
    assert pr["number"] == 2679
    assert pr["head_sha"] == "abc123"
    assert pr["checks"]["overall"] == "success"
    assert pr["checks"]["names"] == ["ci", "lint"]
    assert pr["preflight"]["status"] == "healthy"
    assert pr["preflight"]["head_sha_matches_expected"] is True
    assert pr["reviews"] == {"APPROVED": 1, "COMMENTED": 1}
    assert pr["review_snapshot"]["total"] == 2
    assert len(pr["comment_snapshot"]["latest"]) == 2
    assert pr["comment_snapshot"]["contains_more"] is False
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
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2680], repo="ll7/robot_sf_ll7")

    pr = payload["prs"][0]
    assert pr["checks"]["overall"] == "pending"
    assert pr["checks"]["pending"] == [
        {"name": "ci", "status": "in_progress", "conclusion": "pending", "details_url": ""}
    ]
    assert pr["next_action"] == "await_ci_or_start_read_only_monitor"
    assert payload["route_health_overview"]["healthy"] == 1


def test_snapshot_prs_details_url_none_falls_back_without_none_string() -> None:
    """Explicit null detailsUrl should not become the literal string None."""
    pr_payload = {
        "number": 2682,
        "title": "pending PR",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "def457",
        "mergeable": "UNKNOWN",
        "statusCheckRollup": [
            {
                "name": "ci",
                "status": "completed",
                "conclusion": "failure",
                "detailsUrl": None,
                "targetUrl": "",
            }
        ],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2682], repo="ll7/robot_sf_ll7")

    pr = payload["prs"][0]
    assert pr["checks"]["failed"] == [
        {"name": "ci", "status": "completed", "conclusion": "failure", "details_url": ""}
    ]


def test_snapshot_prs_stale_if_head_sha_mismatch() -> None:
    """Expected-head-sha mismatch should mark lane stale and request refresh."""
    pr_payload = {
        "number": 2690,
        "title": "stale head PR",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "current",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2690], repo="ll7/robot_sf_ll7", expected_head_sha="expected")

    pr = payload["prs"][0]
    assert pr["preflight"]["status"] == "stale"
    assert pr["preflight"]["head_sha_matches_expected"] is False
    assert pr["next_action"] == "invalidate_stale_lane"
    assert payload["route_health_overview"]["stale"] == 1


def test_main_includes_compact_comment_review_evidence() -> None:
    """Comment and review bodies should be compacted to bounded excerpts."""
    long_body = "x" * (COMMENT_BODY_LIMIT + 50)
    pr_payload = {
        "number": 2691,
        "title": "noisy PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2691",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "beef00",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        "reviews": [
            {
                "state": "COMMENTED",
                "author": {"login": "r1"},
                "body": long_body,
                "submittedAt": "2026-06-01T00:01:00Z",
            },
            {
                "state": "COMMENTED",
                "author": {"login": "r2"},
                "body": "short",
                "submittedAt": "2026-06-01T00:00:00Z",
            },
        ],
        "comments": [
            {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:00:00Z", "body": long_body},
        ],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_main:
        mock_main.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([2691], repo="ll7/robot_sf_ll7", expected_head_sha="beef00")

    pr = payload["prs"][0]
    assert len(pr["comment_snapshot"]["latest"]) == 1
    assert len(pr["comment_snapshot"]["latest"][0]["body_excerpt"]) <= COMMENT_BODY_LIMIT
    assert pr["comment_snapshot"]["latest"][0]["body_excerpt"].endswith("...")
    assert pr["review_snapshot"]["latest"][0]["state"] == "COMMENTED"


def test_snapshot_prs_can_include_bounded_review_threads() -> None:
    """Review-thread mode should omit diff hunks and bound comment bodies."""
    long_body = "review body " * 40
    pr_payload = {
        "number": 2692,
        "title": "threaded PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2692",
        "labels": [{"name": "priority: high"}],
        "headRefName": "feature",
        "headRefOid": "abc999",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        "reviews": [],
        "comments": [],
    }
    thread_payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "totalCount": 1,
                        "nodes": [
                            {
                                "id": "thread-1",
                                "isResolved": False,
                                "path": "scripts/dev/example.py",
                                "line": 42,
                                "comments": {
                                    "totalCount": 1,
                                    "nodes": [
                                        {
                                            "author": {"login": "reviewer"},
                                            "body": long_body,
                                            "createdAt": "2026-06-01T00:00:00Z",
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                }
            }
        }
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr=""),
            MagicMock(returncode=0, stdout=json.dumps(thread_payload), stderr=""),
        ]
        payload = snapshot_prs([2692], repo="ll7/robot_sf_ll7", include_review_threads=True)

    snapshot = payload["prs"][0]["review_thread_snapshot"]
    thread = snapshot["threads"][0]
    comment = thread["comments"][0]
    assert snapshot["status"] == "ok"
    assert snapshot["unresolved"] == 1
    assert thread["diff_hunk_omitted"] is True
    assert len(comment["body_excerpt"]) <= COMMENT_BODY_LIMIT
    assert comment["body_omitted"] is True


def test_snapshot_prs_handles_null_review_thread_graphql_fields() -> None:
    """Null GraphQL response layers should not crash compact snapshots."""
    pr_payload = {
        "number": 2695,
        "title": "null thread PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2695",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc995",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        "reviews": [],
        "comments": [],
    }
    thread_payload = {"data": {"repository": {"pullRequest": None}}}
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr=""),
            MagicMock(returncode=0, stdout=json.dumps(thread_payload), stderr=""),
        ]
        payload = snapshot_prs([2695], repo="ll7/robot_sf_ll7", include_review_threads=True)

    snapshot = payload["prs"][0]["review_thread_snapshot"]
    assert snapshot["status"] == "ok"
    assert snapshot["total"] == 0
    assert snapshot["threads"] == []


def test_raw_review_comments_artifact_writes_full_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Raw review comments are opt-in and written to an artifact path."""
    artifact = tmp_path / "raw-review-comments.json"
    raw_comments = [
        {
            "id": 1,
            "body": "full body",
            "diff_hunk": "@@ -1 +1 @@\n-old\n+new",
        }
    ]
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(raw_comments), stderr="")
        result = write_raw_review_comments_artifact([2693], repo="ll7/robot_sf_ll7", path=artifact)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert result == payload
    assert payload["prs"]["2693"]["status"] == "ok"
    assert payload["prs"]["2693"]["contains_raw_diff_hunks"] is True
    assert payload["prs"]["2693"]["comments"][0]["diff_hunk"].startswith("@@")


def test_raw_review_comments_artifact_rejects_invalid_repo(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Invalid repo names should write an error artifact without calling gh."""
    artifact = tmp_path / "raw-review-comments.json"
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        result = write_raw_review_comments_artifact([2696], repo="robot_sf_ll7", path=artifact)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert result == payload
    assert payload["prs"]["2696"] == {
        "status": "error",
        "error": "repo_owner_missing",
    }
    mock_gh.assert_not_called()


def test_main_raw_review_artifact_keeps_hunks_out_of_stdout(
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """The CLI should report only the artifact path, not raw diff hunks."""
    artifact = tmp_path / "raw-review-comments.json"
    pr_payload = {
        "number": 2694,
        "title": "artifact PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2694",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc444",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [],
        "reviews": [],
        "comments": [],
    }
    raw_comments = [{"id": 1, "body": "full body", "diff_hunk": "@@ raw hunk"}]
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr=""),
            MagicMock(returncode=0, stdout=json.dumps(raw_comments), stderr=""),
        ]
        rc = main(
            [
                "--prs",
                "2694",
                "--raw-review-comments-artifact",
                str(artifact),
                "--json",
            ]
        )

    stdout = capsys.readouterr().out
    assert rc == 0
    assert str(artifact) in stdout
    assert "@@ raw hunk" not in stdout
    assert "@@ raw hunk" in artifact.read_text(encoding="utf-8")


def test_main_raw_review_artifact_failure_sets_nonzero_exit(
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """Raw artifact failures should fail automation instead of only writing artifact errors."""
    artifact = tmp_path / "raw-review-comments.json"
    pr_payload = {
        "number": 2697,
        "title": "artifact failure PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2697",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc997",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr=""),
            MagicMock(returncode=1, stdout="", stderr="not found"),
        ]
        rc = main(
            [
                "--prs",
                "2697",
                "--raw-review-comments-artifact",
                str(artifact),
                "--json",
            ]
        )

    stdout = capsys.readouterr().out
    output_payload = json.loads(stdout)
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert rc == 1
    assert output_payload["raw_review_comments_artifact_status"] == "error"
    assert artifact_payload["prs"]["2697"] == {"status": "error", "error": "not found"}


def test_main_requires_pr_number(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI should fail compactly when no PRs are provided."""
    rc = main(["--json"])
    assert rc == 1
    assert "at least one PR" in capsys.readouterr().err


def test_main_rejects_expected_head_sha_for_batch(capsys) -> None:  # type: ignore[no-untyped-def]
    """Expected-head guards should not be broadcast across batch snapshots."""
    rc = main(["--prs", "1", "2", "--expected-head-sha", "abc123", "--json"])
    assert rc == 1
    assert "--expected-head-sha requires exactly one PR" in capsys.readouterr().err


def test_main_rejects_active_review_thread_mode(capsys) -> None:  # type: ignore[no-untyped-def]
    """Review-thread mode should stay explicit instead of broad active discovery."""
    rc = main(["--active", "--review-threads", "--json"])
    assert rc == 1
    assert "--review-threads is only supported" in capsys.readouterr().err


def test_main_active_mode_discovers_open_prs() -> None:
    """Active-mode queue snapshot should emit compact PR attention entries."""
    pr_payload = [
        {
            "number": 2681,
            "title": "active PR",
            "state": "OPEN",
            "isDraft": False,
            "url": "https://github.test/pull/2681",
            "labels": [{"name": "merge-ready"}],
            "headRefName": "feature",
            "headRefOid": "cafe00",
            "mergeable": "MERGEABLE",
            "statusCheckRollup": [{"name": "ci", "status": "in_progress", "conclusion": ""}],
            "reviews": [],
            "comments": [],
        },
        {
            "number": 2682,
            "title": "draft PR",
            "state": "OPEN",
            "isDraft": True,
            "url": "https://github.test/pull/2682",
            "labels": [],
            "headRefName": "feat2",
            "headRefOid": "cafe01",
            "mergeable": "UNKNOWN",
            "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "failure"}],
            "reviews": [{"state": "APPROVED"}],
            "comments": [],
        },
    ]
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh_active:
        mock_gh_active.return_value = MagicMock(
            returncode=0, stdout=json.dumps(pr_payload), stderr=""
        )
        active_queue = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=2)

    assert active_queue["mode"] == "active"
    assert len(active_queue["prs"]) == 2
    assert active_queue["prs"][0]["next_action"] == "await_ci_or_start_read_only_monitor"
    assert active_queue["prs"][0]["attention"] == "ci_pending"
    assert active_queue["prs"][1]["attention"] == "preflight_attention"
    assert active_queue["route_health_overview"]["healthy"] == 1
    assert active_queue["route_health_overview"]["blocked"] == 1
    assert active_queue["prs"][1]["preflight"]["status"] == "blocked"

    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_main:
        mock_main.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        rc = main(["--active", "--json", "--limit", "2"])
    assert rc == 0
