"""Tests for the GitHub issue-with-comments helper (issues #5021 and #5092).

These tests mock ``gh api`` so they run offline and never hit the network. They
cover the success path, comment pagination, field normalization (the drop-in
contract for ``gh issue view --json`` consumers), fail-closed error handling,
and the CLI render modes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.gh_issue_rest import (
    COMMENTS_PAGE_SIZE,
    PROJECT_CARDS_ERROR_MARKER,
    fetch_comments,
    fetch_issue,
    fetch_issue_with_comments,
    main,
    read_complete_issue_thread,
    render_issue_plain,
)


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _raw_issue(*, number: int = 5021, state: str = "open") -> dict:
    """Return a raw REST issue payload with REST-native field shapes."""
    return {
        "number": number,
        "title": "friction: gh issue view fails on deprecated classic-project field",
        "body": "## What I was doing\n\nReading the live issue",
        "state": state,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "url": f"graphql://issues/{number}",
        "user": {"login": "ll7"},
        "author_association": "OWNER",
        "labels": [{"name": "workflow"}, {"name": "bug"}],
        "assignees": [{"login": "alice"}, {"login": "bob"}],
        "created_at": "2026-07-10T08:45:56Z",
        "updated_at": "2026-07-10T09:00:00Z",
    }


def _raw_comment(*, cid: int = 1, login: str = "ll7") -> dict:
    """Return a raw REST comment payload with REST-native field shapes."""
    return {
        "id": cid,
        "body": "## Implementation plan",
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/5021#issuecomment-{cid}",
        "url": f"graphql://issues/comments/{cid}",
        "user": {"login": login},
        "author_association": "OWNER",
        "created_at": "2026-07-10T11:12:48Z",
        "updated_at": "2026-07-10T11:12:48Z",
    }


def test_fetch_issue_normalizes_rest_fields_to_gh_json_shape() -> None:
    """REST ``state``/``html_url`` should be normalized to gh ``--json`` shape."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps(_raw_issue(state="open")))
        payload = fetch_issue(5021)
    assert payload["status"] == "ok"
    assert payload["number"] == 5021
    # state uppercased to match gh issue view --json
    assert payload["state"] == "OPEN"
    # url equals html_url, not the graphql url
    assert payload["url"] == "https://github.com/ll7/robot_sf_ll7/issues/5021"
    # labels/assignees flattened and sorted
    assert payload["labels"] == ["bug", "workflow"]
    assert payload["assignees"] == ["alice", "bob"]
    assert payload["user"] == "ll7"
    assert payload["author_association"] == "OWNER"


def test_fetch_issue_fails_closed_on_nonzero_exit() -> None:
    """A nonzero gh api exit should surface as a clear error payload, not raise."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(returncode=1, stderr="GraphQL: deprecation error")
        payload = fetch_issue(5021)
    assert payload["status"] == "error"
    assert "deprecation error" in payload["error"]
    assert payload["number"] == 5021


def test_fetch_issue_fails_closed_on_invalid_json() -> None:
    """Malformed JSON should fail closed with a helpful snippet."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout="not-json{")
        payload = fetch_issue(5021)
    assert payload["status"] == "error"
    assert "invalid JSON" in payload["error"]


def test_fetch_comments_paginates_until_short_page() -> None:
    """Comments should paginate by page size and stop on a short final page."""
    full_page = [_raw_comment(cid=i) for i in range(COMMENTS_PAGE_SIZE)]
    short_page = [_raw_comment(cid=COMMENTS_PAGE_SIZE + 1)]
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(full_page)),
            _proc(stdout=json.dumps(short_page)),
        ]
        result = fetch_comments(5021, max_pages=5)
    assert result["status"] == "ok"
    assert len(result["comments"]) == COMMENTS_PAGE_SIZE + 1
    # normalization applied
    assert result["comments"][0]["user"] == "ll7"
    assert result["comments"][0]["url"].startswith("https://github.com/")
    # only two pages fetched (stopped early on short page)
    assert mock_api.call_count == 2


def test_fetch_comments_fails_closed_when_page_budget_exceeded() -> None:
    """A full last page must fail closed rather than silently truncate."""
    full_page = [_raw_comment(cid=i) for i in range(COMMENTS_PAGE_SIZE)]
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps(full_page))
        result = fetch_comments(5021, max_pages=1)
    assert result["status"] == "error"
    assert "more than 100 comments" in result["error"]


def test_fetch_comments_rejects_invalid_max_pages() -> None:
    """A non-positive page budget should fail closed without any API call."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        result = fetch_comments(5021, max_pages=0)
    assert result["status"] == "error"
    assert "max_pages" in result["error"]
    mock_api.assert_not_called()


def test_fetch_issue_with_comments_combines_issue_and_thread() -> None:
    """The combined helper should attach a normalized comments list."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_issue())),
            _proc(stdout=json.dumps([_raw_comment(cid=10, login="reviewer")])),
        ]
        payload = fetch_issue_with_comments(5021)
    assert payload["status"] == "ok"
    assert payload["state"] == "OPEN"
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["user"] == "reviewer"
    # issue endpoint read first, then comments endpoint
    assert mock_api.call_count == 2
    assert "issues/5021" in mock_api.call_args_list[0].args[0]
    assert "issues/5021/comments" in mock_api.call_args_list[1].args[0]


def test_fetch_issue_with_comments_propagates_issue_error() -> None:
    """If the issue read fails, the combined helper must not fetch comments."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(returncode=1, stderr="not found")
        payload = fetch_issue_with_comments(999)
    assert payload["status"] == "error"
    assert "not found" in payload["error"]
    assert mock_api.call_count == 1


def test_complete_thread_uses_native_output_when_available() -> None:
    """The concise CLI path should remain the fast path when it succeeds."""
    with (
        patch("scripts.dev.gh_issue_rest._gh_issue_view") as mock_view,
        patch("scripts.dev.gh_issue_rest.fetch_issue_with_comments") as mock_rest,
    ):
        mock_view.return_value = _proc(stdout="native thread\n")
        result = read_complete_issue_thread(5092)
    assert result == {
        "number": 5092,
        "status": "ok",
        "source": "gh_issue_view",
        "text": "native thread\n",
    }
    mock_rest.assert_not_called()


def test_complete_thread_falls_back_to_rest_and_preserves_comment_order() -> None:
    """The known projectCards failure should use the complete REST thread in API order."""
    comments = [
        {**_raw_comment(cid=10, login="first"), "body": "first comment"},
        {**_raw_comment(cid=20, login="second"), "body": "second comment"},
    ]
    with (
        patch("scripts.dev.gh_issue_rest._gh_issue_view") as mock_view,
        patch("scripts.dev.gh_issue_rest._gh_api") as mock_api,
    ):
        mock_view.return_value = _proc(
            returncode=1,
            stderr=f"GraphQL: Projects (classic) is deprecated ({PROJECT_CARDS_ERROR_MARKER})",
        )
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_issue(number=5092))),
            _proc(stdout=json.dumps(comments)),
        ]
        result = read_complete_issue_thread(5092, max_comment_pages=7)
    assert result["status"] == "ok"
    assert result["source"] == "rest_fallback"
    assert result["text"].index("first comment") < result["text"].index("second comment")
    assert [call.args[0] for call in mock_api.call_args_list] == [
        "repos/ll7/robot_sf_ll7/issues/5092",
        "repos/ll7/robot_sf_ll7/issues/5092/comments?per_page=100&page=1",
    ]


def test_complete_thread_does_not_mask_unrelated_native_failure() -> None:
    """Authentication and other native errors must fail closed without REST retry."""
    with (
        patch("scripts.dev.gh_issue_rest._gh_issue_view") as mock_view,
        patch("scripts.dev.gh_issue_rest.fetch_issue_with_comments") as mock_rest,
    ):
        mock_view.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
        result = read_complete_issue_thread(5092)
    assert result["status"] == "error"
    assert result["source"] == "gh_issue_view"
    assert "Bad credentials" in result["error"]
    mock_rest.assert_not_called()


def test_complete_thread_reports_rest_fallback_failure() -> None:
    """A failed fallback must report both the triggering path and REST error."""
    with (
        patch("scripts.dev.gh_issue_rest._gh_issue_view") as mock_view,
        patch("scripts.dev.gh_issue_rest.fetch_issue_with_comments") as mock_rest,
    ):
        mock_view.return_value = _proc(returncode=1, stderr=PROJECT_CARDS_ERROR_MARKER)
        mock_rest.return_value = {"status": "error", "error": "comments page 2 failed"}
        result = read_complete_issue_thread(5092)
    assert result["status"] == "error"
    assert result["source"] == "rest_fallback"
    assert PROJECT_CARDS_ERROR_MARKER in result["error"]
    assert "comments page 2 failed" in result["error"]


def test_render_issue_plain_resembles_gh_issue_view_comments() -> None:
    """Plain rendering should expose title, state, url, body, and the thread."""
    payload = {
        "number": 5021,
        "title": "friction",
        "state": "OPEN",
        "url": "https://github.com/ll7/robot_sf_ll7/issues/5021",
        "author_association": "OWNER",
        "user": "ll7",
        "labels": ["workflow"],
        "body": "## What I was doing",
        "comments": [
            {
                "id": 1,
                "user": "ll7",
                "author_association": "OWNER",
                "created_at": "2026-07-10T11:12:48Z",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/5021#issuecomment-1",
                "body": "## Implementation plan",
            }
        ],
    }
    text = render_issue_plain(payload)
    assert "title:\tfriction" in text
    assert "state:\tOPEN" in text
    assert "url:\thttps://github.com/ll7/robot_sf_ll7/issues/5021" in text
    assert "## What I was doing" in text
    assert "ll7 (OWNER) commented on 2026-07-10T11:12:48Z" in text
    assert "## Implementation plan" in text


def test_cli_view_plain_outputs_thread(capsys: pytest.CaptureFixture[str]) -> None:
    """``view --plain`` should print the human-readable thread and exit 0."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_issue())),
            _proc(stdout=json.dumps([_raw_comment(cid=7)])),
        ]
        rc = main(["view", "5021", "--repo", "ll7/robot_sf_ll7", "--plain", "--comments"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "friction" in captured.out
    assert "## Implementation plan" in captured.out


def test_cli_thread_outputs_rest_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    """``thread`` should expose fallback output without adding source metadata."""
    with patch("scripts.dev.gh_issue_rest.read_complete_issue_thread") as mock_read:
        mock_read.return_value = {
            "status": "ok",
            "source": "rest_fallback",
            "text": "complete fallback thread\n",
        }
        rc = main(["thread", "5092", "--repo", "ll7/robot_sf_ll7"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "complete fallback thread\n"
    mock_read.assert_called_once_with(
        5092,
        repo="ll7/robot_sf_ll7",
        max_comment_pages=10,
    )


def test_cli_view_json_without_comments_omits_thread(capsys: pytest.CaptureFixture[str]) -> None:
    """Without ``--comments`` the JSON output must not include a comments key."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_issue())),
            _proc(stdout=json.dumps([_raw_comment(cid=7)])),
        ]
        rc = main(["view", "5021", "--json", "number", "title", "state"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"number", "title", "state"}
    assert payload["state"] == "OPEN"


def test_cli_view_json_comments_returns_thread(capsys: pytest.CaptureFixture[str]) -> None:
    """``--json comments`` must return the thread, not an empty object (gate #5049 fix)."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_issue())),
            _proc(stdout=json.dumps([_raw_comment(cid=7)])),
        ]
        rc = main(["view", "5021", "--json", "comments"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"comments"}
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["id"] == 7


def test_fetch_issue_maps_null_body_to_empty_string() -> None:
    """A REST ``null`` body must normalize to ``""``, never the string ``"None"`` (gate #5049)."""
    raw = _raw_issue()
    raw["body"] = None
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps(raw))
        payload = fetch_issue(5021)
    assert payload["status"] == "ok"
    assert payload["body"] == ""


def test_fetch_issue_fails_closed_when_gh_cli_missing() -> None:
    """A missing gh CLI must return an error payload, not raise (gate #5049 fix)."""
    with patch("scripts.dev.gh_issue_rest.subprocess.run", side_effect=FileNotFoundError("gh")):
        payload = fetch_issue(5021)
    assert payload["status"] == "error"
    assert "gh CLI not found" in payload["error"]


def test_cli_view_fails_closed_on_rest_error(capsys: pytest.CaptureFixture[str]) -> None:
    """A REST failure should print the error to stderr and exit nonzero."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(
            returncode=1, stderr="GraphQL: Projects (classic) is being deprecated"
        )
        rc = main(["view", "5021", "--comments"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "deprecated" in captured.err


def test_cli_view_rejects_unknown_json_field(capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown --json fields should fail fast with exit code 2, not emit partial JSON."""
    with patch("scripts.dev.gh_issue_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_issue())),
            _proc(stdout=json.dumps([])),
        ]
        rc = main(["view", "5021", "--json", "bogus"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown field" in captured.err
