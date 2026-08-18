"""Offline tests for the guarded REST PR review writer (issue #7571)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from scripts.dev.gh_pr_review_rest import main, post_review

if TYPE_CHECKING:
    from pathlib import Path

HEAD_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"


def _proc(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Build a fake ``gh api`` response."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)


def _write_body(tmp_path: Path, body: str = "Exact-head review evidence.") -> Path:
    """Write a review body fixture without passing Markdown through a shell."""
    path = tmp_path / "review.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_post_review_binds_rest_payload_to_expected_head(tmp_path: Path) -> None:
    """The review POST carries the captured head and body-file contents."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok"},
        ) as mock_guard,
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(
                stdout=json.dumps(
                    {"id": 7, "commit_id": HEAD_SHA, "html_url": "https://example.test/review/7"}
                )
            ),
        ) as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            event="COMMENT",
            repo="ll7/robot_sf_ll7",
        )

    assert result == {
        "status": "ok",
        "number": 7571,
        "repo": "ll7/robot_sf_ll7",
        "event": "COMMENT",
        "head_sha": HEAD_SHA,
        "review_id": 7,
        "url": "https://example.test/review/7",
    }
    mock_guard.assert_called_once_with(
        7571,
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD_SHA,
        operation="commented_review",
    )
    mock_post.assert_called_once_with(
        "repos/ll7/robot_sf_ll7/pulls/7571/reviews",
        {"body": "Exact-head review evidence.", "event": "COMMENT", "commit_id": HEAD_SHA},
    )


def test_stale_guard_skips_review_post(tmp_path: Path) -> None:
    """A stale-state result is returned without publishing a review."""
    body_file = _write_body(tmp_path)
    stale = {
        "status": "review_skipped_stale_state",
        "reason": "pr_not_open",
        "observed_state": "MERGED",
        "observed_head_sha": HEAD_SHA,
        "merged_at": "2026-08-18T15:40:53Z",
    }
    with (
        patch("scripts.dev.gh_pr_review_rest.guard_pr_write", return_value=stale),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result == stale
    mock_post.assert_not_called()


def test_empty_body_fails_before_guard(tmp_path: Path) -> None:
    """An empty body file cannot create a review or state read."""
    body_file = _write_body(tmp_path, "\n")
    with patch("scripts.dev.gh_pr_review_rest.guard_pr_write") as mock_guard:
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert "must not be empty" in result["error"]
    mock_guard.assert_not_called()


def test_cli_maps_stale_skip_to_exit_two(tmp_path: Path, capsys) -> None:
    """Automation can distinguish a safe stale skip from a transport error."""
    body_file = _write_body(tmp_path)
    stale = {"status": "review_skipped_stale_state", "reason": "head_sha_changed"}
    with patch("scripts.dev.gh_pr_review_rest.post_review", return_value=stale):
        rc = main(
            [
                "7571",
                "--body-file",
                str(body_file),
                "--expected-head-sha",
                HEAD_SHA,
            ]
        )

    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err) == stale
