"""Tests for compact PR queue snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.pr_metadata import metadata_digest, metadata_trailer
from scripts.dev.snapshot_pr_queue import (
    COMMENT_BODY_LIMIT,
    _pr_payload_from_dict,
    main,
    snapshot_active_prs,
    snapshot_prs,
    write_raw_review_comments_artifact,
)


@pytest.fixture(autouse=True)
def _default_fresh_base_freshness():  # type: ignore[no-untyped-def]
    """Existing queue tests use a fresh PR base unless they override the freshness source."""
    with (
        patch("scripts.dev.snapshot_pr_queue._fetch_current_main_sha", return_value="main-sha"),
        patch("scripts.dev.snapshot_pr_queue._fetch_pr_base_sha", return_value="main-sha"),
    ):
        yield


def _base_freshness_pr(*, number: int = 7021) -> dict[str, object]:
    return {
        "number": number,
        "title": "base freshness PR",
        "state": "OPEN",
        "isDraft": False,
        "url": f"https://github.test/pull/{number}",
        "labels": [{"name": "merge-ready"}],
        "headRefName": "feature",
        "headRefOid": "head-sha",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
        "reviews": [],
        "comments": [],
    }


def test_base_freshness_fresh_preserves_merge_ready_action() -> None:
    """A fresh PR base should expose provenance without changing merge-ready routing."""
    pr = _pr_payload_from_dict(
        _base_freshness_pr(),
        base_sha="main-sha",
        current_main_sha="main-sha",
        default_number=7021,
        expected_head_sha="head-sha",
    )

    assert pr["base_freshness"] == {
        "base_sha": "main-sha",
        "current_main_sha": "main-sha",
        "verdict": "fresh",
        "action": "continue_queue_routing",
        "reason": "PR base SHA matches current main",
    }
    assert pr["preflight"]["status"] == "healthy"
    assert pr["next_action"] == "merge_readiness_local_check"


def test_base_freshness_stale_blocks_merge_ready_action() -> None:
    """A stale PR base must route to branch refresh before review or merge readiness."""
    pr = _pr_payload_from_dict(
        _base_freshness_pr(),
        base_sha="old-base",
        current_main_sha="main-sha",
        default_number=7021,
        expected_head_sha="head-sha",
    )

    assert pr["base_freshness"]["base_sha"] == "old-base"
    assert pr["base_freshness"]["current_main_sha"] == "main-sha"
    assert pr["base_freshness"]["verdict"] == "stale"
    assert pr["base_freshness"]["action"] == "refresh_pr_base_before_review_or_merge"
    assert pr["preflight"]["status"] == "stale"
    assert "base_sha_stale" in pr["preflight"]["reasons"]
    assert pr["next_action"] == "refresh_pr_base_before_review_or_merge"


def test_base_freshness_missing_base_blocks_merge_ready_action() -> None:
    """Missing PR base provenance is unverifiable and must fail closed."""
    pr = _pr_payload_from_dict(
        _base_freshness_pr(),
        base_sha="",
        current_main_sha="main-sha",
        default_number=7021,
        expected_head_sha="head-sha",
    )

    assert pr["base_freshness"]["base_sha"] is None
    assert pr["base_freshness"]["current_main_sha"] == "main-sha"
    assert pr["base_freshness"]["verdict"] == "missing-base"
    assert pr["base_freshness"]["action"] == "verify_pr_base_before_queue_routing"
    assert pr["preflight"]["status"] == "blocked"
    assert "base_sha_missing" in pr["preflight"]["reasons"]
    assert pr["next_action"] == "verify_pr_base_before_queue_routing"


def test_base_freshness_unavailable_current_main_blocks_merge_ready_action() -> None:
    """Unavailable current-main provenance is unverifiable and must fail closed."""
    pr = _pr_payload_from_dict(
        _base_freshness_pr(),
        base_sha="base-sha",
        current_main_sha="",
        default_number=7021,
        expected_head_sha="head-sha",
    )

    assert pr["base_freshness"]["base_sha"] == "base-sha"
    assert pr["base_freshness"]["current_main_sha"] is None
    assert pr["base_freshness"]["verdict"] == "unavailable-current-main"
    assert pr["base_freshness"]["action"] == "refresh_current_main_before_queue_routing"
    assert pr["preflight"]["status"] == "blocked"
    assert "current_main_sha_unavailable" in pr["preflight"]["reasons"]
    assert pr["next_action"] == "refresh_current_main_before_queue_routing"


@pytest.mark.parametrize(
    ("label", "next_owner_or_gate"),
    [
        ("blocked", "blocker_owner_or_maintainer"),
        ("decision-required", "maintainer_decision_or_approval"),
        ("evidence:blocked", "evidence_or_domain_approval"),
        ("state:blocked", "blocker_owner_or_maintainer"),
        ("state:blocked-external-input", "external_input_owner_or_staging_gate"),
        ("state:hold", "maintainer_decision_or_approval"),
    ],
)
def test_explicit_blocker_overrides_green_merge_ready_routing(
    label: str, next_owner_or_gate: str
) -> None:
    """Explicit stop-state labels must wait for their owner or approval gate."""
    pr_data = _base_freshness_pr()
    pr_data["labels"] = [{"name": "merge-ready"}, {"name": label}]

    pr = _pr_payload_from_dict(
        pr_data,
        base_sha="main-sha",
        current_main_sha="main-sha",
        default_number=7021,
        expected_head_sha="head-sha",
    )

    blocked_state = pr["preflight"]["blocked_state"]
    assert blocked_state["status"] == "blocked"
    assert blocked_state["labels"] == [label]
    assert blocked_state["reasons"] == [f"explicit_blocked:{label}"]
    assert blocked_state["next_owner_or_gate"] == next_owner_or_gate
    assert pr["preflight"]["status"] == "blocked"
    assert pr["next_action"] == "await_blocker_owner_or_approval"
    assert pr["attention"] == "blocked_attention"


def test_explicit_blocker_precedes_stale_base_refresh_hint() -> None:
    """A blocked PR remains owner-gated even when its base also needs refresh."""
    pr_data = _base_freshness_pr()
    pr_data["labels"] = [{"name": "state:blocked"}, {"name": "merge-ready"}]

    pr = _pr_payload_from_dict(
        pr_data,
        base_sha="old-base",
        current_main_sha="main-sha",
        default_number=7021,
        expected_head_sha="head-sha",
    )

    assert pr["preflight"]["status"] == "blocked"
    assert "base_sha_stale" in pr["preflight"]["reasons"]
    assert "explicit_blocked:state:blocked" in pr["preflight"]["reasons"]
    assert pr["next_action"] == "await_blocker_owner_or_approval"


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
    assert payload["schema"] == "pr_queue_snapshot.v2"
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


def test_snapshot_prs_suppresses_superseded_cancelled_run() -> None:
    """An older cancelled run must not override a newer pending replacement."""
    pr_payload = {
        "number": 5869,
        "title": "superseded cancelled PR",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc111",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-07-01T00:00:00Z",
            },
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "in_progress",
                "conclusion": "",
                "startedAt": "2026-07-01T00:05:00Z",
            },
        ],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([5869], repo="ll7/robot_sf_ll7", expected_head_sha="abc111")

    pr = payload["prs"][0]
    assert pr["checks"]["overall"] == "pending"
    assert pr["checks"]["superseded"] == 1
    assert pr["checks"]["total"] == 1


def test_snapshot_prs_suppresses_superseded_cancelled_for_newer_success() -> None:
    """A newer successful replacement should suppress an older cancellation."""
    pr_payload = {
        "number": 5869,
        "title": "superseded cancelled then success",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc222",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-07-01T00:00:00Z",
            },
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "completed",
                "conclusion": "success",
                "startedAt": "2026-07-01T00:05:00Z",
            },
        ],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([5869], repo="ll7/robot_sf_ll7", expected_head_sha="abc222")

    pr = payload["prs"][0]
    assert pr["checks"]["overall"] == "success"
    assert pr["checks"]["superseded"] == 1
    assert pr["checks"]["total"] == 1


def test_snapshot_prs_current_cancellation_stays_fail_closed() -> None:
    """A current, non-superseded cancellation must remain a failure."""
    pr_payload = {
        "number": 5869,
        "title": "current cancellation PR",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc333",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-07-01T00:00:00Z",
            }
        ],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([5869], repo="ll7/robot_sf_ll7", expected_head_sha="abc333")

    pr = payload["prs"][0]
    assert pr["checks"]["overall"] == "failure"
    assert pr["checks"]["superseded"] == 0


def test_snapshot_prs_ignores_non_mapping_rollup_entries_before_deduplication() -> None:
    """Malformed rollup entries must not crash the latest-run filter."""
    pr_payload = {
        "number": 5869,
        "title": "mixed malformed rollup",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc-malformed",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            None,
            "not-a-check",
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([5869], repo="ll7/robot_sf_ll7", expected_head_sha="abc-malformed")

    checks = payload["prs"][0]["checks"]
    assert checks["overall"] == "success"
    assert checks["total"] == 1
    assert checks["superseded"] == 0
    assert checks["names"] == ["ci"]


def test_snapshot_prs_keeps_independent_legacy_status_entries() -> None:
    """Legacy statuses and timestamp-less runs stay independently classified."""
    pr_payload = {
        "number": 5869,
        "title": "mixed legacy and actions runs",
        "state": "OPEN",
        "isDraft": False,
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc444",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            {
                "name": "legacy-status",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "completed",
                "conclusion": "cancelled",
                "startedAt": "2026-07-01T00:00:00Z",
            },
            {
                "__typename": "CheckRun",
                "name": "ci",
                "workflowName": "ci",
                "status": "completed",
                "conclusion": "success",
                "startedAt": "2026-07-01T00:05:00Z",
            },
        ],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([5869], repo="ll7/robot_sf_ll7", expected_head_sha="abc444")

    pr = payload["prs"][0]
    assert pr["checks"]["overall"] == "success"
    assert pr["checks"]["superseded"] == 1
    assert pr["checks"]["total"] == 2
    assert pr["checks"]["names"] == ["ci", "legacy-status"]


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


def test_main_output_writes_snapshot_without_changing_stdout(
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    """The --output path should receive the same compact payload printed by the CLI."""
    artifact = tmp_path / "nested" / "snapshot.json"
    pr_payload = {
        "number": 2698,
        "title": "output PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2698",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc998",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [],
        "reviews": [],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        rc = main(["--prs", "2698", "--output", str(artifact), "--json"])

    stdout_payload = json.loads(capsys.readouterr().out)
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert rc == 0
    assert artifact_payload == stdout_payload
    assert artifact.read_text(encoding="utf-8").endswith("\n")


def test_main_output_write_failure_returns_controlled_error(capsys) -> None:  # type: ignore[no-untyped-def]
    """An unwritable output path should fail without printing a traceback or payload."""
    pr_payload = {
        "number": 2699,
        "title": "output error PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/2699",
        "labels": [],
        "headRefName": "feature",
        "headRefOid": "abc999",
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [],
        "reviews": [],
        "comments": [],
    }
    with (
        patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh,
        patch(
            "scripts.dev.snapshot_pr_queue.write_snapshot_artifact",
            side_effect=OSError("disk full"),
        ),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        rc = main(["--prs", "2699", "--output", "snapshot.json", "--json"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert captured.err == "snapshot output write failed: disk full\n"


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


def test_snapshot_prs_extracts_gate_verdicts_from_long_bodies() -> None:
    """Gate verdict trailers past 180 chars must be extracted into gate_verdicts before excerpt truncation."""
    sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
    long_prefix = "Detailed review feedback paragraph line. " * 6  # > 200 chars
    digest = metadata_digest("long body review PR", "")
    long_review_body = (
        f"{long_prefix}\n\ngate-verdict: accepted @ {sha}\n\n{metadata_trailer(digest)}"
    )

    pr_payload = {
        "number": 6130,
        "title": "long body review PR",
        "state": "OPEN",
        "isDraft": False,
        "url": "https://github.test/pull/6130",
        "labels": [{"name": "merge-ready"}],
        "headRefName": "feature",
        "headRefOid": sha,
        "mergeable": "MERGEABLE",
        "statusCheckRollup": [
            {"name": "ci", "status": "completed", "conclusion": "success"},
        ],
        "reviews": [
            {
                "state": "APPROVED",
                "author": {"login": "reviewer"},
                "authorAssociation": "OWNER",
                "body": long_review_body,
                "submittedAt": "2026-07-22T20:00:00Z",
            }
        ],
        "comments": [],
    }
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([6130], repo="ll7/robot_sf_ll7")

    pr = payload["prs"][0]
    assert pr["gate_verdicts"] == [f"gate-verdict: accepted @ {sha}"]
    assert pr["metadata_digest"] == digest
    assert pr["metadata_verdicts"] == [metadata_trailer(digest)]
    excerpt = pr["review_snapshot"]["latest"][0]["body_excerpt"]
    assert len(excerpt) <= 180
    assert excerpt.endswith("...")
    assert f"gate-verdict: accepted @ {sha}" not in excerpt


# Issue #6564: GraphQL quota exhaustion REST fallback tests (deterministic, no live GitHub).

from scripts.dev.snapshot_pr_queue import (  # noqa: E402
    _is_graphql_quota_error,
    _review_thread_snapshot,
    fetch_pr,
)

QUOTA_STDERR = "GraphQL: API rate limit already exceeded."


def _resp(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_is_graphql_quota_error_detection() -> None:
    assert _is_graphql_quota_error(QUOTA_STDERR)
    assert _is_graphql_quota_error("server error: API rate limit exceeded")
    assert not _is_graphql_quota_error("merge conflict")
    assert not _is_graphql_quota_error("")


def test_fetch_pr_falls_back_to_rest_on_graphql_quota() -> None:
    """A GraphQL quota failure switches to REST and marks the snapshot fail-closed."""
    pull = {
        "number": 42,
        "title": "demo",
        "state": "OPEN",
        "draft": False,
        "labels": [{"name": "merge-ready"}],
        "html_url": "https://x/42",
        "head": {"ref": "fix", "sha": "abc"},
        "base": {"sha": "main-sha"},
        "mergeable_state": "clean",
    }
    reviews = [{"state": "APPROVED", "author_association": "OWNER", "body": "lgtm"}]
    comments = [{"author_association": "MEMBER", "body": "nudge"}]
    check_runs = {
        "check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "success", "details_url": "u"},
        ]
    }
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        side_effect=[
            _resp(returncode=1, stderr=QUOTA_STDERR),  # gh pr view -> quota
            _resp(stdout=json.dumps(pull)),  # REST pulls/42
            _resp(stdout=json.dumps(reviews)),  # REST pulls/42/reviews
            _resp(stdout=json.dumps(comments)),  # REST issues/42/comments
            _resp(stdout=json.dumps(check_runs)),  # REST commits/abc/check-runs
        ],
    ):
        payload = fetch_pr(42, repo="ll7/robot_sf_ll7", expected_head_sha="abc")
    assert payload["status"] == "ok"
    assert payload["data_source"] == "rest_fallback_graphql_quota"
    assert payload["review_threads"] == "unknown_graphql_quota"
    assert payload["review_threads_admission"] == "fail_closed_unknown"
    assert payload["head_sha"] == "abc"
    assert payload["preflight"]["head_sha_matches_expected"] is True
    assert payload["title"] == "demo"
    assert "merge-ready" in payload["labels"]
    assert payload["checks"]["overall"] == "success"


def test_fetch_pr_rest_reports_head_mismatch_without_mixing_commits() -> None:
    """The REST fallback binds checks to the REST head sha and reports a mismatch as stale."""
    pull = {
        "number": 7,
        "title": "t",
        "state": "OPEN",
        "draft": False,
        "labels": [],
        "head": {"ref": "b", "sha": "resthead"},
        "base": {"sha": "main-sha"},
        "mergeable_state": "clean",
    }
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        side_effect=[
            _resp(returncode=1, stderr=QUOTA_STDERR),
            _resp(stdout=json.dumps(pull)),
            _resp(stdout="[]"),  # reviews
            _resp(stdout="[]"),  # comments
            _resp(stdout=json.dumps({"check_runs": []})),  # check-runs at resthead
        ],
    ):
        payload = fetch_pr(7, repo="ll7/robot_sf_ll7", expected_head_sha="differenthead")
    assert payload["status"] == "ok"
    assert payload["preflight"]["status"] == "stale"
    assert "head_sha_mismatch" in payload["preflight"]["reasons"]
    assert payload["preflight"]["head_sha_matches_expected"] is False


def test_fetch_pr_non_quota_error_still_returns_generic_error() -> None:
    """Non-quota gh failures are unchanged (no REST attempt)."""
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        return_value=_resp(returncode=1, stderr="could not find PR"),
    ):
        payload = fetch_pr(99, repo="ll7/robot_sf_ll7")
    assert payload["status"] == "error"
    assert "error_kind" not in payload
    assert "could not find PR" in payload["error"]


def test_review_thread_snapshot_reports_unknown_graphql_quota() -> None:
    """Review threads are GraphQL-only; under quota they are unknown (fail-closed)."""
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
    ):
        snap = _review_thread_snapshot(42, repo="ll7/robot_sf_ll7")
    assert snap["status"] == "unknown_graphql_quota"
    assert snap["unresolved"] is None
    assert "merge-ready" in snap["guidance"]


def test_fetch_pr_rest_rest_fallback_failure_is_labeled() -> None:
    """If REST also fails under quota, the error is labeled graphql_quota_exhausted."""
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        side_effect=[
            _resp(returncode=1, stderr=QUOTA_STDERR),  # gh pr view -> quota
            _resp(returncode=1, stderr="not found"),  # REST pulls -> fail
        ],
    ):
        payload = fetch_pr(5, repo="ll7/robot_sf_ll7")
    assert payload["status"] == "error"
    assert payload["error_kind"] == "graphql_quota_exhausted"
