"""Tests for compact PR queue snapshots."""

from __future__ import annotations

import json
import shlex
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.pr_metadata import metadata_digest, metadata_trailer
from scripts.dev.snapshot_pr_queue import (
    COMMENT_BODY_LIMIT,
    _pr_payload_from_dict,
    _project_review_thread_state,
    _refresh_route_hint,
    _rest_open_pr_list,
    _rest_paginated_check_runs,
    _review_thread_snapshot,
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


def _route_cancellation_pr(
    *,
    head_sha: str = "route-head",
    workflow_name: str = "Route external review bots",
    include_success: bool = True,
) -> dict[str, object]:
    """Return a compact PR fixture with the documented routing cancellation."""
    rollup: list[dict[str, object]] = [
        {
            "__typename": "CheckRun",
            "name": "ci",
            "status": "completed",
            "conclusion": "success",
            "startedAt": "2026-09-05T18:00:00Z",
        },
        {
            "__typename": "CheckRun",
            "name": "route-coderabbit",
            "workflowName": workflow_name,
            "status": "completed",
            "conclusion": "cancelled",
            "startedAt": "2026-09-05T18:39:49Z",
            "completedAt": "2026-09-05T18:39:49Z",
            "detailsUrl": ("https://github.test/actions/runs/33984757657/job/101356131316"),
        },
    ]
    if include_success:
        rollup.insert(
            1,
            {
                "__typename": "CheckRun",
                "name": "route-coderabbit",
                "workflowName": workflow_name,
                "status": "completed",
                "conclusion": "success",
                "startedAt": "2026-09-05T18:36:48Z",
                "completedAt": "2026-09-05T18:36:52Z",
                "detailsUrl": ("https://github.test/actions/runs/33984605002/job/101355720321"),
            },
        )
    return {
        "number": 8517,
        "title": "route cancellation fixture",
        "state": "OPEN",
        "isDraft": True,
        "labels": [{"name": "state:blocked"}],
        "headRefName": "feature",
        "headRefOid": head_sha,
        "mergeable": "MERGEABLE",
        "statusCheckRollup": rollup,
        "reviews": [],
        "comments": [],
    }


def _route_cancellation_rest_payloads(
    *,
    head_sha: str = "route-head",
    annotation_message: str = (
        "Canceling since a higher priority waiting request for review-bot-routing-8471 exists"
    ),
    cancelled_head_sha: str | None = None,
) -> dict[str, object]:
    """Return exact-head REST identity and annotation fixtures for route supersession."""
    cancelled_head_sha = cancelled_head_sha or head_sha
    return {
        "actions/runs/33984757657": {
            "id": 33984757657,
            "name": "Route external review bots",
            "head_sha": cancelled_head_sha,
            "conclusion": "cancelled",
            "run_started_at": "2026-09-05T18:39:48Z",
            "updated_at": "2026-09-05T18:39:50Z",
        },
        "check-runs/101356131316": {
            "id": 101356131316,
            "name": "route-coderabbit",
            "head_sha": cancelled_head_sha,
            "status": "completed",
            "conclusion": "cancelled",
            "started_at": "2026-09-05T18:39:49Z",
            "completed_at": "2026-09-05T18:39:49Z",
            "details_url": ("https://github.test/actions/runs/33984757657/job/101356131316"),
        },
        "check-runs/101356131316/annotations": [{"message": annotation_message}],
        "actions/runs/33984605002": {
            "id": 33984605002,
            "name": "Route external review bots",
            "head_sha": head_sha,
            "conclusion": "success",
            "run_started_at": "2026-09-05T18:36:48Z",
            "updated_at": "2026-09-05T18:36:52Z",
        },
        "check-runs/101355720321": {
            "id": 101355720321,
            "name": "route-coderabbit",
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-09-05T18:36:48Z",
            "completed_at": "2026-09-05T18:36:52Z",
            "details_url": "https://github.test/actions/runs/33984605002/job/101355720321",
        },
    }


def test_snapshot_prs_classifies_proven_superseded_route_cancellation() -> None:
    """A known higher-priority route cancellation must not override same-head success."""
    pr_payload = _route_cancellation_pr()
    rest_payloads = _route_cancellation_rest_payloads()

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        if path not in rest_payloads:
            raise AssertionError(f"unexpected REST path: {path}")
        return rest_payloads[path]

    with (
        patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh,
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([8517], repo="ll7/robot_sf_ll7", expected_head_sha="route-head")

    checks = payload["prs"][0]["checks"]
    assert checks["overall"] == "success"
    assert checks["failed"] == []
    assert checks["superseded"] == 1
    assert checks["superseded_cancellations"][0]["run_id"] == 33984757657
    assert checks["superseded_cancellations"][0]["replacement"]["run_id"] == 33984605002


def test_snapshot_prs_keeps_route_cancellation_without_exact_marker() -> None:
    """A cancellation with an unknown annotation remains a failure."""
    pr_payload = _route_cancellation_pr()
    rest_payloads = _route_cancellation_rest_payloads(annotation_message="manual cancellation")

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        return rest_payloads[path]

    with (
        patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh,
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([8517], repo="ll7/robot_sf_ll7", expected_head_sha="route-head")

    checks = payload["prs"][0]["checks"]
    assert checks["overall"] == "failure"
    assert checks["failed"][0]["conclusion"] == "cancelled"
    assert "superseded_cancellations" not in checks


def test_snapshot_prs_keeps_route_cancellation_on_head_mismatch() -> None:
    """A matching annotation cannot suppress a cancellation from another head."""
    pr_payload = _route_cancellation_pr()
    rest_payloads = _route_cancellation_rest_payloads(cancelled_head_sha="other-head")

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        return rest_payloads[path]

    with (
        patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh,
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([8517], repo="ll7/robot_sf_ll7", expected_head_sha="route-head")

    checks = payload["prs"][0]["checks"]
    assert checks["overall"] == "failure"
    assert "superseded_cancellations" not in checks


def test_snapshot_prs_keeps_route_cancellation_when_rest_replacement_is_not_earlier() -> None:
    """A REST replacement that starts after cancellation cannot suppress it."""
    pr_payload = _route_cancellation_pr()
    pr_payload["statusCheckRollup"][1]["startedAt"] = ""
    rest_payloads = _route_cancellation_rest_payloads()
    rest_payloads["actions/runs/33984605002"]["run_started_at"] = "2026-09-05T18:45:00Z"
    rest_payloads["check-runs/101355720321"]["started_at"] = "2026-09-05T18:45:00Z"

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        return rest_payloads[path]

    with (
        patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh,
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([8517], repo="ll7/robot_sf_ll7", expected_head_sha="route-head")

    checks = payload["prs"][0]["checks"]
    assert checks["overall"] == "failure"
    assert checks["failed"][0]["conclusion"] == "cancelled"
    assert "superseded_cancellations" not in checks


def test_snapshot_prs_keeps_route_cancellation_when_rest_ids_do_not_match_urls() -> None:
    """REST records with a mismatched returned ID cannot suppress a cancellation."""
    pr_payload = _route_cancellation_pr()
    rest_payloads = _route_cancellation_rest_payloads()
    rest_payloads["check-runs/101356131316"]["id"] = 999

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        return rest_payloads[path]

    with (
        patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh,
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(pr_payload), stderr="")
        payload = snapshot_prs([8517], repo="ll7/robot_sf_ll7", expected_head_sha="route-head")

    checks = payload["prs"][0]["checks"]
    assert checks["overall"] == "failure"
    assert "superseded_cancellations" not in checks


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


def test_snapshot_preserves_merged_at_for_external_merge_classification() -> None:
    """A closed REST/GraphQL row with mergedAt remains distinguishable downstream."""
    pr_data = _base_freshness_pr(number=7571)
    pr_data["state"] = "CLOSED"
    pr_data["mergedAt"] = "2026-08-18T15:40:53Z"

    pr = _pr_payload_from_dict(
        pr_data,
        base_sha="main-sha",
        current_main_sha="main-sha",
        default_number=7571,
        expected_head_sha="head-sha",
    )

    assert pr["state"] == "CLOSED"
    assert pr["merged_at"] == "2026-08-18T15:40:53Z"


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


def test_snapshot_preserves_trusted_exact_head_base_policy() -> None:
    """Queue policy can consume the selector recorded in trusted review evidence."""
    pr_data = _base_freshness_pr()
    head_sha = "a" * 40
    pr_data["headRefOid"] = head_sha
    pr_data["reviews"] = [
        {
            "state": "COMMENTED",
            "authorAssociation": "OWNER",
            "body": f"base-policy: ordinary-cas @ {head_sha}",
        }
    ]

    pr = _pr_payload_from_dict(
        pr_data,
        base_sha="old-base",
        current_main_sha="main-sha",
        default_number=7021,
        expected_head_sha=head_sha,
    )

    assert pr["base_policy"] == [f"base-policy: ordinary-cas @ {head_sha}"]


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
                        "pageInfo": {"hasNextPage": False},
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
    assert snapshot["status"] == "incomplete"
    assert snapshot["unresolved"] is None
    assert "refusing a thread-free result" in snapshot["error"]


def test_snapshot_prs_rejects_non_boolean_review_thread_resolution() -> None:
    """Malformed GraphQL resolution flags remain unevaluated instead of coercing truthiness."""
    pr_payload = _base_freshness_pr(number=2697)
    thread_payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "totalCount": 1,
                        "pageInfo": {"hasNextPage": False},
                        "nodes": [{"id": "thread-1", "isResolved": "false"}],
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
        payload = snapshot_prs([2697], repo="ll7/robot_sf_ll7", include_review_threads=True)

    pr = payload["prs"][0]
    assert pr["review_thread_snapshot"]["status"] == "incomplete"
    assert pr["review_thread_snapshot"]["unresolved"] is None
    assert pr["review_threads_admission"] == "not_evaluated"
    assert pr["preflight"]["review_threads_admission"] == "not_evaluated"
    assert pr["next_action"] == "inspect_blocking_preflight"


def test_review_thread_projection_preserves_stale_base_precedence() -> None:
    """Unknown nested evidence must not hide the more specific stale-base action."""
    pr = _pr_payload_from_dict(
        _base_freshness_pr(number=2698),
        base_sha="old-base",
        current_main_sha="main-sha",
        default_number=2698,
        expected_head_sha="head-sha",
    )

    _project_review_thread_state(pr, "unknown_graphql_quota")
    _refresh_route_hint(pr)

    assert pr["preflight"]["status"] == "stale"
    assert pr["next_action"] == "refresh_pr_base_before_review_or_merge"
    assert pr["attention"] == "stale_attention"


def test_review_thread_projection_preserves_explicit_blocker_precedence() -> None:
    """An explicit policy blocker remains owner-gated when thread evidence is unknown."""
    pr_data = _base_freshness_pr(number=2699)
    pr_data["labels"] = [{"name": "state:blocked"}]
    pr = _pr_payload_from_dict(
        pr_data,
        base_sha="main-sha",
        current_main_sha="main-sha",
        default_number=2699,
        expected_head_sha="head-sha",
    )

    _project_review_thread_state(pr, "incomplete")
    _refresh_route_hint(pr)

    assert pr["preflight"]["status"] == "blocked"
    assert pr["next_action"] == "await_blocker_owner_or_approval"
    assert pr["attention"] == "blocked_attention"


def test_successful_review_thread_refresh_clears_prior_unknown_projection() -> None:
    """A later complete thread read removes only the stale fallback projection."""
    pr = _pr_payload_from_dict(
        _base_freshness_pr(number=2700),
        base_sha="main-sha",
        current_main_sha="main-sha",
        default_number=2700,
        expected_head_sha="head-sha",
    )
    _project_review_thread_state(pr, "unknown_graphql_quota")
    assert pr["preflight"]["status"] == "blocked"

    _project_review_thread_state(pr, "ok")
    _refresh_route_hint(pr)

    assert "review_threads" not in pr
    assert "review_threads_admission" not in pr
    assert "review_threads" not in pr["preflight"]
    assert "review_threads_admission" not in pr["preflight"]
    assert not any(
        str(reason).startswith("review_threads_") for reason in pr["preflight"]["reasons"]
    )
    assert pr["preflight"]["status"] == "healthy"
    assert pr["next_action"] == "merge_readiness_local_check"


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

from scripts.dev.snapshot_pr_queue import _is_graphql_quota_error, fetch_pr  # noqa: E402

QUOTA_STDERR = "GraphQL: API rate limit already exceeded."


def _resp(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_is_graphql_quota_error_detection() -> None:
    assert _is_graphql_quota_error(QUOTA_STDERR)
    assert _is_graphql_quota_error("server error: API rate limit exceeded")
    assert not _is_graphql_quota_error("merge conflict")
    assert not _is_graphql_quota_error("")


@pytest.mark.parametrize(
    ("page_two_size", "expected_truncated"),
    [(50, False), (51, True), (100, True)],
)
def test_rest_open_pr_list_paginates_and_reports_overfetch_truncation(
    page_two_size: int, expected_truncated: bool
) -> None:
    """REST active discovery distinguishes a complete short page from discarded overfetch."""
    page_one = [{"number": number} for number in range(100)]
    page_two = [{"number": number} for number in range(page_two_size)]

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        if path.endswith("page=1"):
            return page_one
        if path.endswith("page=2"):
            return page_two
        raise AssertionError(f"unexpected REST path: {path}")

    with patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get) as mock_rest:
        rows, truncated = _rest_open_pr_list(repo="ll7/robot_sf_ll7", limit=150)  # type: ignore[misc]

    assert len(rows) == 150
    assert truncated is expected_truncated
    assert mock_rest.call_count == 2
    assert mock_rest.call_args_list[0].args[0] == "pulls?state=open&per_page=100&page=1"
    assert mock_rest.call_args_list[1].args[0] == "pulls?state=open&per_page=100&page=2"


def test_rest_check_runs_rejects_short_page_before_total_count() -> None:
    """An inconsistent REST count must not be accepted as a complete check snapshot."""
    with patch(
        "scripts.dev.snapshot_pr_queue._rest_api_get",
        return_value={"total_count": 2, "check_runs": []},
    ):
        rows, status = _rest_paginated_check_runs("head-sha", repo="ll7/robot_sf_ll7")

    assert rows == []
    assert status == "truncated"


def test_rest_check_runs_rejects_contradictory_total_count() -> None:
    """REST totals that cannot contain the returned rows must fail closed."""
    with patch(
        "scripts.dev.snapshot_pr_queue._rest_api_get",
        return_value={"total_count": 1, "check_runs": [{"name": "ci"}, {"name": "ci-2"}]},
    ):
        rows, status = _rest_paginated_check_runs("head-sha", repo="ll7/robot_sf_ll7")

    assert rows == []
    assert status == "error"


def test_rest_open_pr_list_has_a_page_budget() -> None:
    """An extreme active-list limit cannot turn REST fallback into unbounded requests."""
    page = [{"number": number} for number in range(100)]

    with (
        patch("scripts.dev.snapshot_pr_queue.REST_ACTIVE_MAX_PAGES", 2),
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", return_value=page) as mock_rest,
    ):
        rows, truncated = _rest_open_pr_list(repo="ll7/robot_sf_ll7", limit=250)  # type: ignore[misc]

    assert len(rows) == 200
    assert truncated is True
    assert mock_rest.call_count == 2


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
    assert payload["expected_head_sha"] == "abc"
    assert payload["preflight"]["status"] == "blocked"
    assert "review_threads_unknown_graphql_quota" in payload["preflight"]["reasons"]
    assert payload["next_action"] == "inspect_blocking_preflight"
    assert payload["rest_enrichment"] == {
        "reviews": "ok",
        "comments": "ok",
        "checks": "ok",
    }
    assert payload["title"] == "demo"
    assert "merge-ready" in payload["labels"]
    assert payload["checks"]["overall"] == "success"


@pytest.mark.parametrize("returncode", [0, 1])
def test_snapshot_active_prs_falls_back_to_bounded_rest_and_enriches_rows(
    returncode: int,
) -> None:
    """Active discovery should use bounded REST and preserve fail-closed row evidence."""
    rest_rows = [
        {"number": 42, "head": {"sha": "head-42"}},
        {"number": 43, "head": {"sha": "head-43"}},
    ]

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del timeout
        assert repo == "ll7/robot_sf_ll7"
        if path == "branches/main":
            return {"commit": {"sha": "main-sha"}}
        if path == "pulls?state=open&per_page=2&page=1":
            return rest_rows
        if path.startswith("pulls/") and path.count("/") == 1:
            number = int(path.split("/", 1)[1])
            return {
                "number": number,
                "title": f"REST PR {number}",
                "state": "open",
                "draft": False,
                "labels": [{"name": "merge-ready"}],
                "html_url": f"https://github.test/pull/{number}",
                "head": {"ref": f"feature-{number}", "sha": f"head-{number}"},
                "base": {"sha": "main-sha"},
                "mergeable_state": "clean",
            }
        endpoint = path.split("?", 1)[0]
        if endpoint.endswith("/reviews") or endpoint.endswith("/comments"):
            return []
        if endpoint.startswith("commits/") and endpoint.endswith("/check-runs"):
            return {
                "check_runs": [
                    {
                        "name": "ci",
                        "status": "completed",
                        "conclusion": "success",
                        "details_url": "https://github.test/check/1",
                    }
                ]
            }
        raise AssertionError(f"unexpected REST path: {path}")

    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=returncode, stdout=QUOTA_STDERR),
        ),
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get) as mock_rest,
    ):
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=2)

    assert payload["data_source"] == "rest_fallback_graphql_quota"
    assert payload["route_evidence_only"] is True
    assert payload["review_threads"] == "unknown_graphql_quota"
    assert payload["review_threads_admission"] == "fail_closed_unknown"
    assert payload["truncated"] is True
    assert "REST open-PR list may be capped" in payload["truncation_note"]
    assert [pr["number"] for pr in payload["prs"]] == [42, 43]
    for pr in payload["prs"]:
        assert pr["data_source"] == "rest_fallback_graphql_quota"
        assert pr["review_threads"] == "unknown_graphql_quota"
        assert pr["review_threads_admission"] == "fail_closed_unknown"
        assert pr["head_sha"].startswith("head-")
        assert pr["base_freshness"]["verdict"] == "fresh"
        assert pr["checks"]["overall"] == "success"
        assert pr["preflight"]["status"] == "blocked"
        assert "review_threads_unknown_graphql_quota" in pr["preflight"]["reasons"]
        assert pr["next_action"] == "inspect_blocking_preflight"
        assert pr["preflight"]["head_sha_matches_expected"] is True
    mock_rest.assert_any_call("pulls?state=open&per_page=2&page=1", repo="ll7/robot_sf_ll7")


def test_snapshot_active_prs_rest_list_failure_has_no_fabricated_rows() -> None:
    """A failed REST list remains a compact error rather than partial PR data."""
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", return_value=None),
    ):
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=20)

    assert payload["data_source"] == "rest_fallback_graphql_quota"
    assert payload["truncated"] is False
    assert payload["prs"] == [
        {
            "status": "error",
            "error_kind": "graphql_quota_exhausted",
            "error": "GraphQL quota exhausted and REST open-PR list fallback failed",
        }
    ]


@pytest.mark.parametrize(
    ("row", "expected_error"),
    [
        (
            {"number": 0, "head": {"sha": "head-sha"}},
            "REST active PR list contained a row without an integer number",
        ),
        (
            {"number": 42},
            "REST active PR list contained a row without a non-empty head SHA",
        ),
    ],
)
def test_snapshot_active_prs_malformed_rest_row_preserves_provenance(
    row: dict[str, object], expected_error: str
) -> None:
    """Malformed REST inventory must fail closed without dropping its route evidence."""

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del repo, timeout
        if path == "branches/main":
            return {"commit": {"sha": "main-sha"}}
        if path == "pulls?state=open&per_page=20&page=1":
            return [row]
        raise AssertionError(f"unexpected REST path: {path}")

    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=20)

    assert payload["data_source"] == "rest_fallback_graphql_quota"
    assert payload["route_evidence_only"] is True
    assert payload["review_threads_admission"] == "fail_closed_unknown"
    assert payload["prs"] == [
        {
            "status": "error",
            "error_kind": "rest_payload_malformed",
            "error": expected_error,
        }
    ]


def test_fetch_pr_rest_paginates_enrichment_and_preserves_provenance() -> None:
    """REST reviews, comments, and checks must not silently stop at GitHub's page size."""
    pull = {
        "number": 42,
        "title": "paged enrichment",
        "state": "OPEN",
        "draft": False,
        "labels": [],
        "html_url": "https://x/42",
        "head": {"ref": "fix", "sha": "rest-head"},
        "base": {"sha": "main-sha"},
        "mergeable_state": "clean",
    }
    reviews_page_one = [
        {
            "state": "APPROVED",
            "author_association": "OWNER",
            "user": {"login": "reviewer"},
            "submitted_at": "2026-07-01T00:00:00Z",
            "body": "approved",
        }
        for _ in range(100)
    ]
    reviews_page_two = [
        {
            "state": "COMMENTED",
            "author_association": "MEMBER",
            "user": {"login": "reviewer-final"},
            "submitted_at": "2026-07-02T00:00:00Z",
            "body": "follow-up",
        }
    ]
    comments_page_one = [
        {
            "author_association": "MEMBER",
            "user": {"login": "commenter"},
            "created_at": "2026-07-01T00:00:00Z",
            "body": "note",
        }
        for _ in range(100)
    ]
    comments_page_two = [
        {
            "author_association": "OWNER",
            "user": {"login": "commenter-final"},
            "created_at": "2026-07-02T00:00:00Z",
            "body": "latest note",
        }
    ]
    checks_page_one = [
        {
            "name": f"ci-{index}",
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-07-01T00:00:00Z",
        }
        for index in range(100)
    ]
    checks_page_two = [
        {
            "name": "ci-final",
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-07-02T00:00:00Z",
        }
    ]

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del repo, timeout
        if path == "pulls/42":
            return pull
        if path == "pulls/42/reviews?per_page=100&page=1":
            return reviews_page_one
        if path == "pulls/42/reviews?per_page=100&page=2":
            return reviews_page_two
        if path == "issues/42/comments?per_page=100&page=1":
            return comments_page_one
        if path == "issues/42/comments?per_page=100&page=2":
            return comments_page_two
        if path == "commits/rest-head/check-runs?per_page=100&page=1":
            return {"total_count": 101, "check_runs": checks_page_one}
        if path == "commits/rest-head/check-runs?per_page=100&page=2":
            return {"total_count": 101, "check_runs": checks_page_two}
        raise AssertionError(f"unexpected REST path: {path}")

    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get) as mock_rest,
    ):
        payload = fetch_pr(
            42,
            repo="ll7/robot_sf_ll7",
            current_main_sha="main-sha",
            expected_head_sha="rest-head",
        )

    assert payload["rest_enrichment"] == {
        "reviews": "ok",
        "comments": "ok",
        "checks": "ok",
    }
    assert payload["review_snapshot"]["total"] == 101
    assert payload["review_snapshot"]["latest"][0]["author"] == "reviewer-final"
    assert payload["review_snapshot"]["latest"][0]["submitted_at"] == "2026-07-02T00:00:00Z"
    assert payload["comment_snapshot"]["total"] == 101
    assert payload["comment_snapshot"]["latest"][0]["author"] == "commenter-final"
    assert payload["checks"]["total"] == 101
    assert payload["expected_head_sha"] == "rest-head"
    assert mock_rest.call_count == 7


def test_fetch_pr_rest_fails_closed_on_unusable_enrichment_page() -> None:
    """A failed REST enrichment endpoint remains visible and cannot look complete."""
    pull = {
        "number": 42,
        "title": "incomplete enrichment",
        "state": "OPEN",
        "draft": False,
        "labels": [],
        "head": {"ref": "fix", "sha": "rest-head"},
        "base": {"sha": "main-sha"},
        "mergeable_state": "clean",
    }

    def rest_get(path: str, *, repo: str, timeout: int = 45):  # type: ignore[no-untyped-def]
        del repo, timeout
        if path == "pulls/42":
            return pull
        if path.startswith("pulls/42/reviews?"):
            return None
        if path.startswith("issues/42/comments?"):
            return []
        if path.startswith("commits/rest-head/check-runs?"):
            return None
        raise AssertionError(f"unexpected REST path: {path}")

    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch("scripts.dev.snapshot_pr_queue._rest_api_get", side_effect=rest_get),
    ):
        payload = fetch_pr(42, repo="ll7/robot_sf_ll7", current_main_sha="main-sha")

    assert payload["rest_enrichment"] == {
        "reviews": "error",
        "comments": "ok",
        "checks": "error",
    }
    assert "rest_reviews_error" in payload["preflight"]["reasons"]
    assert "rest_checks_error" in payload["preflight"]["reasons"]
    assert payload["checks"]["overall"] == "pending"
    assert payload["preflight"]["status"] == "blocked"


def test_fetch_pr_rest_suppresses_duplicate_cancelled_run_with_actions_metadata() -> None:
    """REST fallback should ignore a cancelled rerun once Actions identifies the workflow."""
    pull = {
        "number": 42,
        "title": "duplicate rerun",
        "state": "OPEN",
        "draft": False,
        "labels": [],
        "html_url": "https://x/42",
        "head": {"ref": "fix", "sha": "abc"},
        "base": {"sha": "main-sha"},
        "mergeable_state": "clean",
    }
    check_runs = {
        "check_runs": [
            {
                "name": "pr-body-contracts",
                "status": "completed",
                "conclusion": "cancelled",
                "started_at": "2026-07-01T00:00:00Z",
                "details_url": "https://x/actions/runs/101/job/1",
            },
            {
                "name": "pr-body-contracts",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-07-01T00:05:00Z",
                "details_url": "https://x/actions/runs/102/job/2",
            },
        ]
    }
    actions_run = {"workflow_id": 300804702, "name": "PR body contracts"}
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        side_effect=[
            _resp(returncode=1, stderr=QUOTA_STDERR),  # gh pr view -> quota
            _resp(stdout=json.dumps(pull)),  # REST pulls/42
            _resp(stdout="[]"),  # REST pulls/42/reviews
            _resp(stdout="[]"),  # REST issues/42/comments
            _resp(stdout=json.dumps(check_runs)),  # REST commits/abc/check-runs
            _resp(stdout=json.dumps(actions_run)),  # REST actions/runs/101
            _resp(stdout=json.dumps(actions_run)),  # REST actions/runs/102
        ],
    ):
        payload = fetch_pr(42, repo="ll7/robot_sf_ll7", expected_head_sha="abc")

    assert payload["checks"]["overall"] == "success"
    assert payload["checks"]["superseded"] == 1
    assert payload["checks"]["total"] == 1
    assert payload["checks"]["failed"] == []


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
    assert "review_threads_unknown_graphql_quota" in payload["preflight"]["reasons"]
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
    handoff = {
        "quota_reset_at": 1_800_000_000,
        "reset_in_seconds": 42,
        "retry_after_utc": "2027-01-01T00:00:00Z",
        "retry_command": "uv run python -m scripts.dev.snapshot_pr_queue 42 --review-threads --json --repo ll7/robot_sf_ll7",
        "handoff": "GraphQL quota exhausted; quota resets at 2027-01-01T00:00:00Z (in ~42s).",
    }
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch(
            "scripts.dev.snapshot_pr_queue.quota_reset_handoff",
            return_value=handoff,
        ) as mock_handoff,
    ):
        snap = _review_thread_snapshot(42, repo="ll7/robot_sf_ll7")
    mock_handoff.assert_called_once()
    assert snap["status"] == "unknown_graphql_quota"
    assert snap["unresolved"] is None
    assert "merge-ready" in snap["guidance"]
    assert "2027-01-01T00:00:00Z" in snap["guidance"]
    assert snap["quota_reset_at"] == 1_800_000_000
    assert snap["reset_in_seconds"] == 42
    assert snap["retry_after_utc"] == "2027-01-01T00:00:00Z"
    assert snap["retry_command"].endswith("--repo ll7/robot_sf_ll7")
    assert "Never admit" in snap["guidance"]


def test_review_thread_snapshot_quota_handoff_unknown_reset_stays_fail_closed() -> None:
    """An unavailable reset read still yields a bounded retry handoff, never approval."""
    handoff = {
        "quota_reset_at": None,
        "reset_in_seconds": None,
        "retry_after_utc": None,
        "retry_command": "uv run python -m scripts.dev.snapshot_pr_queue 7 --review-threads --json --repo o/r",
        "handoff": "GraphQL quota exhausted; the quota reset time is unavailable.",
    }
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch(
            "scripts.dev.snapshot_pr_queue.quota_reset_handoff",
            return_value=handoff,
        ),
    ):
        snap = _review_thread_snapshot(7, repo="o/r")
    assert snap["status"] == "unknown_graphql_quota"
    assert snap["quota_reset_at"] is None
    assert snap["retry_after_utc"] is None
    assert "--review-threads" in snap["retry_command"]
    assert "Never admit" in snap["guidance"]


@pytest.mark.parametrize("returncode", [0, 1])
def test_review_thread_snapshot_uses_quota_retry_classification_from_stdout(
    returncode: int,
) -> None:
    """A quota diagnostic on stdout still gets the reset-aware fail-closed handoff."""
    handoff = {
        "quota_reset_at": None,
        "reset_in_seconds": None,
        "retry_after_utc": None,
        "retry_command": "retry",
        "handoff": "Never admit merge-ready from unknown thread state.",
    }
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=returncode, stdout=QUOTA_STDERR),
        ),
        patch(
            "scripts.dev.snapshot_pr_queue.quota_reset_handoff",
            return_value=handoff,
        ) as mock_handoff,
    ):
        snap = _review_thread_snapshot(42, repo="ll7/robot_sf_ll7")

    assert snap["status"] == "unknown_graphql_quota"
    mock_handoff.assert_called_once()


def test_review_thread_snapshot_ignores_rate_limit_text_in_success_payload() -> None:
    """A valid GraphQL payload containing rate-limit words is ordinary evidence."""
    thread_payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "totalCount": 1,
                        "pageInfo": {"hasNextPage": False},
                        "nodes": [
                            {
                                "id": "thread-1",
                                "isResolved": True,
                                "path": "README.md",
                                "line": 1,
                                "comments": {
                                    "totalCount": 1,
                                    "nodes": [
                                        {
                                            "author": {"login": "reviewer"},
                                            "body": "Document the API rate limit.",
                                            "createdAt": "2026-09-03T00:00:00Z",
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
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        return_value=_resp(returncode=0, stdout=json.dumps(thread_payload)),
    ):
        snap = _review_thread_snapshot(42, repo="ll7/robot_sf_ll7")

    assert snap["status"] == "ok"
    assert snap["unresolved"] == 0


def test_review_thread_snapshot_quotes_repo_in_retry_command() -> None:
    """Copy-paste retry guidance must not turn a repository value into shell syntax."""
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            return_value=_resp(returncode=1, stderr=QUOTA_STDERR),
        ),
        patch(
            "scripts.dev.snapshot_pr_queue.quota_reset_handoff",
            side_effect=lambda *, retry_command: {
                "quota_reset_at": None,
                "reset_in_seconds": None,
                "retry_after_utc": None,
                "retry_command": retry_command,
                "handoff": "Never admit merge-ready from unknown thread state.",
            },
        ),
    ):
        snap = _review_thread_snapshot(42, repo="owner/repo; touch pwned")

    assert shlex.split(snap["retry_command"]) == [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.dev.snapshot_pr_queue",
        "42",
        "--review-threads",
        "--json",
        "--repo",
        "owner/repo; touch pwned",
    ]


def test_snapshot_prs_projects_unknown_review_threads_to_outer_fail_closed_route() -> None:
    """Nested unknown thread evidence must block the outer route and action."""
    pr_payload = _base_freshness_pr(number=8336)
    pr_payload["headRefOid"] = "abc123"
    handoff = {
        "quota_reset_at": None,
        "reset_in_seconds": None,
        "retry_after_utc": None,
        "retry_command": "retry",
        "handoff": "Never admit merge-ready from unknown thread state.",
    }
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            side_effect=[
                _resp(stdout=json.dumps(pr_payload)),
                _resp(returncode=1, stdout=QUOTA_STDERR),
            ],
        ),
        patch(
            "scripts.dev.snapshot_pr_queue.quota_reset_handoff",
            return_value=handoff,
        ),
    ):
        payload = snapshot_prs(
            [8336],
            repo="ll7/robot_sf_ll7",
            expected_head_sha="abc123",
            include_review_threads=True,
        )

    pr = payload["prs"][0]
    assert pr["review_thread_snapshot"]["status"] == "unknown_graphql_quota"
    assert pr["review_threads"] == "unknown_graphql_quota"
    assert pr["review_threads_admission"] == "fail_closed_unknown"
    assert pr["preflight"]["status"] == "blocked"
    assert pr["preflight"]["review_threads"] == "unknown_graphql_quota"
    assert pr["preflight"]["review_threads_admission"] == "fail_closed_unknown"
    assert "review_threads_unknown_graphql_quota" in pr["preflight"]["reasons"]
    assert pr["next_action"] == "inspect_blocking_preflight"
    assert pr["attention"] == "preflight_attention"
    assert payload["route_health_overview"] == {
        "healthy": 0,
        "stale": 0,
        "blocked": 1,
        "unknown": 0,
    }


def test_review_thread_snapshot_reports_exhausted_graphql_transient() -> None:
    """Persistent transient GraphQL failure leaves review-thread evidence unknown."""
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            side_effect=[_resp(returncode=1, stderr="HTTP 503 Service Unavailable")] * 3,
        ),
        patch("scripts.dev.github_graphql_retry.time.sleep", lambda _seconds: None),
    ):
        snap = _review_thread_snapshot(42, repo="ll7/robot_sf_ll7")

    assert snap["status"] == "unknown_graphql_transient"
    assert snap["unresolved"] is None
    assert "after 3 attempts" in snap["retry_diagnostic"]
    assert "Never admit" in snap["guidance"]


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


def test_fetch_pr_transient_rest_fallback_failure_is_not_evaluated() -> None:
    """A transient REST fallback failure remains unevaluated, not quota-specific."""
    with (
        patch(
            "scripts.dev.snapshot_pr_queue._gh",
            side_effect=[
                _resp(returncode=1, stderr="HTTP 503 Service Unavailable"),
                _resp(returncode=1, stderr="HTTP 503 Service Unavailable"),
                _resp(returncode=1, stderr="HTTP 503 Service Unavailable"),
                _resp(returncode=1, stderr="not found"),
            ],
        ),
        patch("scripts.dev.github_graphql_retry.time.sleep", lambda _seconds: None),
    ):
        payload = fetch_pr(5, repo="ll7/robot_sf_ll7")

    assert payload["status"] == "error"
    assert payload["error_kind"] == "graphql_transient_exhausted"
    assert payload["data_source"] == "rest_fallback_graphql_transient"
    assert payload["review_threads"] == "unknown_graphql_transient"
    assert payload["review_threads_admission"] == "not_evaluated"


@pytest.mark.parametrize("returncode", [0, 1])
def test_fetch_pr_classifies_stdout_quota_as_graphql_quota_fallback(returncode: int) -> None:
    """Quota text from the preceding ``gh pr view`` stdout keeps quota semantics."""
    with patch(
        "scripts.dev.snapshot_pr_queue._gh",
        side_effect=[
            _resp(returncode=returncode, stdout=QUOTA_STDERR),  # gh pr view -> quota on stdout
            _resp(returncode=1, stderr="not found"),  # REST pulls -> fail
        ],
    ):
        payload = fetch_pr(5, repo="ll7/robot_sf_ll7")

    assert payload["status"] == "error"
    assert payload["error_kind"] == "graphql_quota_exhausted"
    assert payload["data_source"] == "rest_fallback_graphql_quota"
    assert payload["review_threads"] == "unknown_graphql_quota"
