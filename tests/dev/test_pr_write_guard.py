"""Tests for exact-head PR write freshness guards (issue #7571)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.dev.pr_write_guard import guard_pr_write

HEAD_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
BASE_SHA = "b1c2d3e4f5061728394a5b6c7d8e9f0011121314"
OTHER_SHA = "deadbeef00000000000000000000000000000001"


def _proc(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Build a fake ``gh api`` response."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)


def _pr_payload(
    *,
    state: str = "open",
    head_sha: str = HEAD_SHA,
    base_sha: str | None = BASE_SHA,
    merged_at: object = None,
    author_login: str | None = None,
) -> str:
    """Build a compact pull-request REST payload."""
    payload: dict[str, object] = {"state": state, "head": {"sha": head_sha}, "merged_at": merged_at}
    if base_sha is not None:
        payload["base"] = {"sha": base_sha}
    if author_login is not None:
        payload["user"] = {"login": author_login}
    return json.dumps(payload)


def test_matching_open_head_is_admitted() -> None:
    """A matching open head may proceed to the guarded write."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload()),
    ) as mock_get:
        result = guard_pr_write(
            7571,
            repo="ll7/robot_sf_ll7",
            expected_head_sha=HEAD_SHA,
            operation="commented_review",
        )

    assert result == {
        "status": "ok",
        "number": 7571,
        "repo": "ll7/robot_sf_ll7",
        "operation": "commented_review",
        "expected_head_sha": HEAD_SHA,
        "expected_base_sha": "",
        "observed_state": "OPEN",
        "observed_head_sha": HEAD_SHA,
        "observed_base_sha": BASE_SHA,
        "merged_at": None,
    }
    mock_get.assert_called_once_with("repos/ll7/robot_sf_ll7/pulls/7571")


def test_matching_open_head_and_base_are_admitted() -> None:
    """A supplied expected base must match the live base before any write."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload()),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            expected_base_sha=BASE_SHA,
            operation="merge_ready_label",
        )

    assert result["status"] == "ok"
    assert result["observed_base_sha"] == BASE_SHA


def test_include_author_returns_observed_pr_author() -> None:
    """An author-aware guard exposes the live PR author for policy checks."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload(author_login="Maintainer")),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            operation="review",
            include_author=True,
        )

    assert result["status"] == "ok"
    assert result["observed_author_login"] == "Maintainer"


def test_include_author_requires_live_pr_author() -> None:
    """An author-aware guard fails closed when the PR author is unavailable."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload()),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            operation="review",
            include_author=True,
        )

    assert result == {"status": "error", "error": "PR write-state payload has no author login"}


def test_base_movement_skips_write() -> None:
    """A live base different from the expected base is a stale write skip."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload(base_sha=OTHER_SHA)),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            expected_base_sha=BASE_SHA,
            operation="merge_ready_label",
        )

    assert result["status"] == "review_skipped_stale_state"
    assert result["reason"] == "base_sha_changed"
    assert result["observed_base_sha"] == OTHER_SHA


def test_missing_live_base_fails_closed_when_base_expected() -> None:
    """A missing live base cannot authorize a write when a base is required."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload(base_sha=None)),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            expected_base_sha=BASE_SHA,
            operation="merge_ready_label",
        )

    assert result["status"] == "error"
    assert "no base SHA" in result["error"]


def test_invalid_expected_base_fails_closed_without_read() -> None:
    """Only a full expected base SHA can authorize a write-state read."""
    with patch("scripts.dev.pr_write_guard._gh_api_get") as mock_get:
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            expected_base_sha=BASE_SHA[:12],
            operation="merge_ready_label",
        )

    assert result["status"] == "error"
    assert "full 40-character SHA" in result["error"]
    mock_get.assert_not_called()


def test_closed_or_merged_pr_skips_write() -> None:
    """A lifecycle transition is a structured stale-write skip."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload(state="closed", merged_at="2026-08-18T15:40:53Z")),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            operation="merge_ready_label",
        )

    assert result["status"] == "review_skipped_stale_state"
    assert result["reason"] == "pr_not_open"
    assert result["observed_state"] == "CLOSED"
    assert result["observed_head_sha"] == HEAD_SHA
    assert result["merged_at"] == "2026-08-18T15:40:53Z"


def test_head_movement_skips_write() -> None:
    """An open PR whose head moved must not receive the old write."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(stdout=_pr_payload(head_sha=OTHER_SHA)),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            operation="commented_review",
        )

    assert result["status"] == "review_skipped_stale_state"
    assert result["reason"] == "head_sha_changed"
    assert result["observed_state"] == "OPEN"
    assert result["expected_head_sha"] == HEAD_SHA
    assert result["observed_head_sha"] == OTHER_SHA


def test_missing_or_abbreviated_expected_head_fails_closed_without_read() -> None:
    """Only a full expected SHA can authorize a write-state read."""
    with patch("scripts.dev.pr_write_guard._gh_api_get") as mock_get:
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA[:12],
            operation="commented_review",
        )

    assert result["status"] == "error"
    assert "full 40-character SHA" in result["error"]
    mock_get.assert_not_called()


def test_transport_failure_fails_closed() -> None:
    """An unreadable PR state is an error, never permission to write."""
    with patch(
        "scripts.dev.pr_write_guard._gh_api_get",
        return_value=_proc(returncode=1, stderr="HTTP 403: forbidden"),
    ):
        result = guard_pr_write(
            7571,
            expected_head_sha=HEAD_SHA,
            operation="commented_review",
        )

    assert result["status"] == "error"
    assert "HTTP 403" in result["error"]
