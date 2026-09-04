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
BASE_SHA = "f0e1d2c3b4a5968778695a4b3c2d1e0f00112233"


def _proc(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Build a fake ``gh api`` response."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)


def _actor_payload(login: str = "reviewer") -> str:
    """Build a compact authenticated-user REST payload."""
    return json.dumps({"login": login})


def _write_body(
    tmp_path: Path,
    body: str = f"Exact-head review evidence for {HEAD_SHA}.",
) -> Path:
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
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
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
        {
            "body": f"Exact-head review evidence for {HEAD_SHA}.",
            "event": "COMMENT",
            "commit_id": HEAD_SHA,
        },
    )


def test_self_authored_request_changes_returns_explicit_comment_guidance(
    tmp_path: Path,
) -> None:
    """A self-authored request changes review never silently becomes a comment."""
    body_file = _write_body(tmp_path)
    guard = {
        "status": "ok",
        "number": 7571,
        "repo": "ll7/robot_sf_ll7",
        "operation": "review",
        "expected_head_sha": HEAD_SHA,
        "expected_base_sha": "",
        "observed_state": "OPEN",
        "observed_head_sha": HEAD_SHA,
        "observed_base_sha": BASE_SHA,
        "merged_at": None,
        "observed_author_login": "Maintainer",
    }
    with (
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(stdout=_actor_payload("maintainer")),
        ) as mock_actor,
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value=guard,
        ) as mock_guard,
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            event="REQUEST_CHANGES",
            repo="ll7/robot_sf_ll7",
        )

    assert result == {
        **guard,
        "status": "review_skipped_self_authored",
        "reason": "self_authored_request_changes_forbidden",
        "fallback_event": "COMMENT",
        "automatic_fallback": False,
        "body_preserved": True,
        "authenticated_actor_login": "maintainer",
    }
    mock_actor.assert_called_once_with("user")
    mock_guard.assert_called_once_with(
        7571,
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD_SHA,
        operation="review",
        include_author=True,
    )
    mock_post.assert_not_called()


def test_independent_actor_can_publish_request_changes(tmp_path: Path) -> None:
    """An independent authenticated actor retains the REQUEST_CHANGES event."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(stdout=_actor_payload("reviewer")),
        ),
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={
                "status": "ok",
                "observed_base_sha": BASE_SHA,
                "observed_author_login": "author",
            },
        ) as mock_guard,
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(
                stdout=json.dumps(
                    {"id": 9, "commit_id": HEAD_SHA, "html_url": "https://example.test/review/9"}
                )
            ),
        ) as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            event="REQUEST_CHANGES",
        )

    assert result["status"] == "ok"
    assert result["event"] == "REQUEST_CHANGES"
    mock_guard.assert_called_once_with(
        7571,
        repo="ll7/robot_sf_ll7",
        expected_head_sha=HEAD_SHA,
        operation="review",
        include_author=True,
    )
    mock_post.assert_called_once_with(
        "repos/ll7/robot_sf_ll7/pulls/7571/reviews",
        {
            "body": f"Exact-head review evidence for {HEAD_SHA}.",
            "event": "REQUEST_CHANGES",
            "commit_id": HEAD_SHA,
        },
    )


def test_explicit_comment_fallback_preserves_blocking_marker(tmp_path: Path) -> None:
    """An explicit comment route carries its blocking marker without actor preflight."""
    body = f"Blocking finding.\n\ngate-verdict: rejected @ {HEAD_SHA}\n"
    body_file = _write_body(tmp_path, body)
    with (
        patch("scripts.dev.gh_pr_review_rest._gh_api_get") as mock_actor,
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(
                stdout=json.dumps(
                    {"id": 10, "commit_id": HEAD_SHA, "html_url": "https://example.test/review/10"}
                )
            ),
        ) as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            event="COMMENT",
        )

    assert result["status"] == "ok"
    mock_actor.assert_not_called()
    assert mock_post.call_args.args[1]["body"] == body
    assert mock_post.call_args.args[1]["event"] == "COMMENT"


def test_publication_failure_is_reported_without_success(tmp_path: Path) -> None:
    """A failed review POST remains an error after all preflight checks pass."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(returncode=1, stderr="HTTP 422: review rejected"),
        ) as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert "HTTP 422" in result["error"]
    mock_post.assert_called_once()


def test_actor_read_failure_skips_guard_and_review_post(tmp_path: Path) -> None:
    """An uncertain authenticated actor cannot authorize a requested-changes write."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(returncode=1, stderr="HTTP 401: bad credentials"),
        ),
        patch("scripts.dev.gh_pr_review_rest.guard_pr_write") as mock_guard,
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            event="REQUEST_CHANGES",
        )

    assert result["status"] == "error"
    assert "HTTP 401" in result["error"]
    mock_guard.assert_not_called()
    mock_post.assert_not_called()


def test_missing_head_sha_in_body_fails_before_post(tmp_path: Path) -> None:
    """A review body lacking the expected head SHA fails closed before POST."""
    body_file = _write_body(tmp_path, "Review body citing no head SHA at all.")
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert "does not cite the expected head SHA" in result["error"]
    mock_post.assert_not_called()


def test_declared_base_without_live_base_fails_before_post(tmp_path: Path) -> None:
    """A declared base cannot publish when the live guard result lacks a base SHA."""
    body_file = _write_body(
        tmp_path,
        f"Review for head {HEAD_SHA}\nBase reviewed: {BASE_SHA}",
    )
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok"},
        ),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert "live base SHA is unavailable" in result["error"]
    mock_post.assert_not_called()


def test_mismatched_declared_base_in_body_fails_before_post(tmp_path: Path) -> None:
    """A review body declaring a different base SHA fails closed before POST."""
    other_base = "0000111122223333444455556666777788889999"
    body_file = _write_body(
        tmp_path,
        f"Review for head {HEAD_SHA}\nBase reviewed: {other_base}",
    )
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert "declares base SHA" in result["error"]
    assert "does not match live base" in result["error"]
    mock_post.assert_not_called()


def test_matching_declared_base_in_body_succeeds(tmp_path: Path) -> None:
    """A review body declaring the matching base SHA succeeds."""
    body_file = _write_body(
        tmp_path,
        f"Review for head {HEAD_SHA}\nBase reviewed: {BASE_SHA}",
    )
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(
                stdout=json.dumps(
                    {"id": 8, "commit_id": HEAD_SHA, "html_url": "https://example.test/review/8"}
                )
            ),
        ) as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "ok"
    assert result["review_id"] == 8
    mock_post.assert_called_once_with(
        "repos/ll7/robot_sf_ll7/pulls/7571/reviews",
        {
            "body": f"Review for head {HEAD_SHA}\nBase reviewed: {BASE_SHA}",
            "event": "COMMENT",
            "commit_id": HEAD_SHA,
        },
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


def test_cli_maps_self_authored_skip_to_exit_two(tmp_path: Path, capsys) -> None:
    """Automation can distinguish explicit self-authored guidance from failure."""
    body_file = _write_body(tmp_path)
    guidance = {
        "status": "review_skipped_self_authored",
        "fallback_event": "COMMENT",
        "automatic_fallback": False,
    }
    with patch("scripts.dev.gh_pr_review_rest.post_review", return_value=guidance):
        rc = main(
            [
                "7571",
                "--body-file",
                str(body_file),
                "--expected-head-sha",
                HEAD_SHA,
                "--event",
                "REQUEST_CHANGES",
            ]
        )

    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err) == guidance
