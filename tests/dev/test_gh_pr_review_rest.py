"""Offline tests for the guarded REST PR review writer (issue #7571)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dev.gh_pr_review_rest import main, post_review
from scripts.dev.pr_metadata import metadata_digest, metadata_trailer

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
        patch("scripts.dev.gh_pr_review_rest._gh_api_get") as mock_metadata,
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
    mock_metadata.assert_not_called()
    mock_post.assert_called_once_with(
        "repos/ll7/robot_sf_ll7/pulls/7571/reviews",
        {
            "body": f"Exact-head review evidence for {HEAD_SHA}.",
            "event": "COMMENT",
            "commit_id": HEAD_SHA,
        },
    )


@pytest.mark.parametrize("suffix", ["", "\n", "\n\n"])
def test_expected_metadata_digest_preserves_terminal_newlines(tmp_path: Path, suffix: str) -> None:
    """The optional live-metadata guard hashes the exact body string, including newlines."""
    body = f"Exact-head review evidence for {HEAD_SHA}.{suffix}"
    body_file = _write_body(tmp_path, body)
    expected_digest = metadata_digest("PR title", body)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(stdout=json.dumps({"title": "PR title", "body": body})),
        ) as mock_metadata,
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(
                stdout=json.dumps(
                    {"id": 11, "commit_id": HEAD_SHA, "html_url": "https://example.test/review/11"}
                )
            ),
        ) as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            repo="ll7/robot_sf_ll7",
            expected_metadata_digest=expected_digest,
        )

    assert result["status"] == "ok"
    assert result["review_id"] == 11
    mock_metadata.assert_called_once_with("repos/ll7/robot_sf_ll7/pulls/7571")
    mock_post.assert_called_once()


def test_mismatched_review_body_metadata_trailer_fails_closed(tmp_path: Path) -> None:
    """A stale review carrier cannot be published with a matching live PR digest."""
    live_body = f"Final PR body for {HEAD_SHA}.\n\n"
    expected_digest = metadata_digest("PR title", live_body)
    stale_digest = "b" * 64
    body_file = _write_body(
        tmp_path,
        f"Exact-head review evidence for {HEAD_SHA}.\n{metadata_trailer(stale_digest)}",
    )
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(stdout=json.dumps({"title": "PR title", "body": live_body})),
        ),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            expected_metadata_digest=expected_digest,
        )

    assert result["status"] == "error"
    assert stale_digest in result["error"]
    assert result["expected_metadata_digest"] == expected_digest
    assert result["observed_review_body_metadata_digests"] == [stale_digest]
    mock_post.assert_not_called()


def test_metadata_digest_mismatch_skips_review_before_post(tmp_path: Path) -> None:
    """A newline-stripped digest is rejected before a review can be created."""
    body = f"Exact-head review evidence for {HEAD_SHA}.\n\n"
    body_file = _write_body(tmp_path, body)
    stripped_digest = metadata_digest("PR title", body.rstrip("\n"))
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(stdout=json.dumps({"title": "PR title", "body": body})),
        ),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            expected_metadata_digest=stripped_digest,
        )

    assert result["status"] == "review_skipped_stale_state"
    assert result["reason"] == "metadata_digest_changed"
    assert result["expected_metadata_digest"] == stripped_digest
    assert result["observed_metadata_digest"] == metadata_digest("PR title", body)
    mock_post.assert_not_called()


def test_metadata_digest_read_failure_skips_review_post(tmp_path: Path) -> None:
    """An uncertain live title/body read fails closed before review publication."""
    body_file = _write_body(tmp_path, f"Exact-head review evidence for {HEAD_SHA}.")
    expected_digest = metadata_digest("PR title", body_file.read_text(encoding="utf-8"))
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_get",
            return_value=_proc(returncode=1, stderr="HTTP 503: unavailable"),
        ),
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            expected_metadata_digest=expected_digest,
        )

    assert result["status"] == "error"
    assert "HTTP 503" in result["error"]
    mock_post.assert_not_called()


def test_invalid_metadata_digest_fails_before_guard(tmp_path: Path) -> None:
    """Malformed metadata guards cannot reach the PR state reader or POST."""
    body_file = _write_body(tmp_path)
    with (
        patch("scripts.dev.gh_pr_review_rest.guard_pr_write") as mock_guard,
        patch("scripts.dev.gh_pr_review_rest._gh_api_post") as mock_post,
    ):
        result = post_review(
            7571,
            body_file,
            expected_head_sha=HEAD_SHA,
            expected_metadata_digest="not-a-digest",
        )

    assert result == {
        "status": "error",
        "error": "expected_metadata_digest must be a full 64-character hexadecimal digest",
    }
    mock_guard.assert_not_called()
    mock_post.assert_not_called()


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


def test_publication_failure_reports_structured_endpoint_evidence(tmp_path: Path) -> None:
    """A rejected review POST names the endpoint, status, and explicit fallback handoff."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(
                returncode=1,
                stderr="gh: Validation Failed (HTTP 422)",
                stdout='{"message":"Validation Failed","errors":[]}',
            ),
        ) as mock_post,
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert result["error_class"] == "review_publication_failed"
    assert result["endpoint"] == "repos/ll7/robot_sf_ll7/pulls/7571/reviews"
    assert result["http_status"] == 422
    assert result["expected_head_sha"] == HEAD_SHA
    assert "Validation Failed" in result["response_body_excerpt"]
    assert "HTTP 422" in result["stderr_excerpt"]
    assert result["fallback_recommended"] is True
    assert result["fallback"] == {
        "kind": "top_level_comment",
        "endpoint": "repos/ll7/robot_sf_ll7/issues/7571/comments",
        "event": "COMMENT",
        "automatic": False,
        "authoritative_for_review": False,
        "idempotency_marker": "<!-- exact-head-review-fallback:v1 -->",
        "expected_head_sha": HEAD_SHA,
        "cli_flag": "--fallback-comment",
    }
    mock_post.assert_called_once()


def test_empty_review_response_is_classified_unavailable(tmp_path: Path) -> None:
    """An empty 200 response still names the endpoint and recommends the fallback."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(stdout=""),
        ),
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert result["http_status"] is None
    assert result["fallback_recommended"] is True


def test_permission_failure_does_not_recommend_fallback(tmp_path: Path) -> None:
    """A 403 is an authorization failure, not an endpoint-availability fallback case."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(returncode=1, stderr="gh: Forbidden (HTTP 403)"),
        ),
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA)

    assert result["status"] == "error"
    assert result["http_status"] == 403
    assert result["fallback_recommended"] is False


def test_fallback_comment_is_explicit_and_marked(tmp_path: Path) -> None:
    """Only --fallback-comment records the fallback, and it carries the head marker."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            side_effect=[
                _proc(returncode=1, stderr="gh: Server Error (HTTP 502)"),
                _proc(stdout=json.dumps({"id": 4242, "html_url": "https://example.test/c/4242"})),
            ],
        ) as mock_post,
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_comments_get",
            return_value=_proc(stdout="[]"),
        ),
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA, fallback_comment=True)

    assert result["status"] == "error"
    assert result["fallback_result"]["status"] == "review_fallback_comment_recorded"
    assert result["fallback_result"]["authoritative_for_review"] is False
    assert result["fallback_result"]["duplicate_prevented"] is False
    assert result["fallback_result"]["comment_id"] == 4242
    assert mock_post.call_count == 2
    endpoint, payload = mock_post.call_args.args
    assert endpoint == "repos/ll7/robot_sf_ll7/issues/7571/comments"
    assert payload["body"].startswith(f"<!-- exact-head-review-fallback:v1 --> head: {HEAD_SHA}")
    assert "Exact-head review evidence" in payload["body"]
    assert "event" not in payload


def test_fallback_comment_is_idempotent_per_head(tmp_path: Path) -> None:
    """An existing marked fallback for the same head is reused without a new comment."""
    body_file = _write_body(tmp_path)
    existing = json.dumps(
        [
            {
                "id": 99,
                "html_url": "https://example.test/c/99",
                "body": f"<!-- exact-head-review-fallback:v1 --> head: {HEAD_SHA}\n\nold",
            }
        ]
    )
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(returncode=1, stderr="gh: Validation Failed (HTTP 422)"),
        ) as mock_post,
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_comments_get",
            return_value=_proc(stdout=existing),
        ),
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA, fallback_comment=True)

    assert result["fallback_result"]["status"] == "review_fallback_comment_recorded"
    assert result["fallback_result"]["duplicate_prevented"] is True
    assert result["fallback_result"]["comment_id"] == 99
    mock_post.assert_called_once()


def test_fallback_comment_read_failure_fails_closed(tmp_path: Path) -> None:
    """An unreadable comment list blocks the fallback instead of risking a duplicate."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(returncode=1, stderr="gh: Validation Failed (HTTP 422)"),
        ) as mock_post,
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_comments_get",
            return_value=_proc(returncode=1, stderr="gh: Server Error (HTTP 502)"),
        ),
    ):
        result = post_review(7571, body_file, expected_head_sha=HEAD_SHA, fallback_comment=True)

    assert result["fallback_result"]["error_class"] == "fallback_read_failed"
    mock_post.assert_called_once()


def test_cli_fallback_flag_maps_to_exit_two(tmp_path: Path, capsys) -> None:
    """The CLI records the fallback and exits 2, never reporting a published review."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            side_effect=[
                _proc(returncode=1, stderr="gh: Server Error (HTTP 503)"),
                _proc(stdout=json.dumps({"id": 7, "html_url": "https://example.test/c/7"})),
            ],
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_comments_get",
            return_value=_proc(stdout="[]"),
        ),
    ):
        exit_code = main(
            [
                "7571",
                "--repo",
                "ll7/robot_sf_ll7",
                "--body-file",
                str(body_file),
                "--expected-head-sha",
                HEAD_SHA,
                "--fallback-comment",
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.err)
    assert payload["status"] == "error"
    assert payload["fallback_result"]["status"] == "review_fallback_comment_recorded"


def test_cli_reports_structured_failure_without_fallback(tmp_path: Path, capsys) -> None:
    """Without the flag the CLI still exits 1 and never posts a comment."""
    body_file = _write_body(tmp_path)
    with (
        patch(
            "scripts.dev.gh_pr_review_rest.guard_pr_write",
            return_value={"status": "ok", "observed_base_sha": BASE_SHA},
        ),
        patch(
            "scripts.dev.gh_pr_review_rest._gh_api_post",
            return_value=_proc(returncode=1, stderr="gh: Validation Failed (HTTP 422)"),
        ) as mock_post,
    ):
        exit_code = main(
            [
                "7571",
                "--repo",
                "ll7/robot_sf_ll7",
                "--body-file",
                str(body_file),
                "--expected-head-sha",
                HEAD_SHA,
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.err)
    assert payload["http_status"] == 422
    assert payload["fallback_recommended"] is True
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


def test_cli_forwards_expected_metadata_digest(tmp_path: Path) -> None:
    """The CLI passes the optional exact metadata digest to the publisher."""
    body_file = _write_body(tmp_path)
    expected_digest = "a" * 64
    with patch("scripts.dev.gh_pr_review_rest.post_review", return_value={"status": "ok"}) as mock:
        assert (
            main(
                [
                    "7571",
                    "--body-file",
                    str(body_file),
                    "--expected-head-sha",
                    HEAD_SHA,
                    "--expected-metadata-digest",
                    expected_digest,
                ]
            )
            == 0
        )

    assert mock.call_args.kwargs["expected_metadata_digest"] == expected_digest


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


def test_direct_invocation_without_ambient_pythonpath(tmp_path: Path) -> None:
    """Direct CLI invocation must succeed from an arbitrary working directory without PYTHONPATH."""
    entrypoint = Path(__file__).resolve().parents[2] / "scripts/dev/gh_pr_review_rest.py"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [sys.executable, "-B", "-I", str(entrypoint), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
