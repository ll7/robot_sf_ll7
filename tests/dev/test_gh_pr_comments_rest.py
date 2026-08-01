"""Offline tests for the REST-only PR conversation-comment reader (issue #6496).

These tests mock ``gh api`` so they run offline and never hit the network. They
cover the REST endpoint contract, field normalization, comment pagination, the
regression check that no Projects Classic ``projectCards`` field is ever
requested, and fail-closed behavior on nonzero exit, invalid JSON, and
non-list payloads.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from scripts.dev.gh_pr_comments_rest import (
    COMMENTS_PAGE_SIZE,
    fetch_pr_comments,
    fetch_pr_header,
    fetch_pr_with_comments,
    main,
    render_pr_comments_plain,
)


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _raw_pr_header(*, number: int = 6454, state: str = "open") -> dict:
    """Return a raw REST issue-shaped payload for a PR header."""
    return {
        "number": number,
        "title": "fix(dev): reconcile closed-state label search lag",
        "body": "## Summary\n\nMerged PR body",
        "state": state,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "url": f"graphql://issues/{number}",
        "user": {"login": "ll7"},
        "pull_request": {"url": f"graphql://pulls/{number}"},
    }


def _raw_comment(*, cid: int = 1, login: str = "reviewer", body: str = "lgtm") -> dict:
    """Return a raw REST comment payload with REST-native field shapes."""
    return {
        "id": cid,
        "body": body,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/pull/6454#issuecomment-{cid}",
        "url": f"graphql://issues/comments/{cid}",
        "user": {"login": login},
        "author_association": "MEMBER",
        "created_at": "2026-07-30T10:00:00Z",
        "updated_at": "2026-07-30T10:05:00Z",
    }


def test_fetch_pr_header_uses_rest_issues_endpoint_and_normalizes() -> None:
    """Header read hits ``repos/{repo}/issues/{n}`` and uppercases state."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps(_raw_pr_header(state="open")))
        payload = fetch_pr_header(6454, repo="ll7/robot_sf_ll7")
    assert payload["status"] == "ok"
    # state uppercased, url equals html_url (not the graphql url)
    assert payload["state"] == "OPEN"
    assert payload["url"] == "https://github.com/ll7/robot_sf_ll7/pull/6454"
    assert payload["title"] == "fix(dev): reconcile closed-state label search lag"
    assert mock_api.call_args.args[0] == "repos/ll7/robot_sf_ll7/issues/6454"


def test_fetch_pr_header_rejects_an_issue_payload() -> None:
    """The PR reader must not silently read comments from a non-PR issue."""
    raw_header = _raw_pr_header()
    raw_header.pop("pull_request")
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps(raw_header))
        result = fetch_pr_header(6454)
    assert result["status"] == "error"
    assert "not a pull request" in result["error"]


def test_fetch_pr_header_rejects_non_positive_number() -> None:
    """Invalid PR numbers must fail before making a REST request."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        result = fetch_pr_header(0)
    assert result["status"] == "error"
    assert "must be positive" in result["error"]
    mock_api.assert_not_called()


def test_fetch_pr_comments_uses_rest_issues_comments_endpoint() -> None:
    """Comments read must use ``repos/{repo}/issues/{n}/comments`` for PRs."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps([_raw_comment(cid=7)]))
        result = fetch_pr_comments(6454, repo="ll7/robot_sf_ll7")
    assert result["status"] == "ok"
    assert len(result["comments"]) == 1
    assert mock_api.call_args.args[0] == (
        f"repos/ll7/robot_sf_ll7/issues/6454/comments?per_page={COMMENTS_PAGE_SIZE}&page=1"
    )


def test_fetch_pr_comments_normalizes_rest_fields() -> None:
    """Comment fields are normalized to the stable machine shape."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps([_raw_comment(cid=7, login="ll7")]))
        result = fetch_pr_comments(6454)
    comment = result["comments"][0]
    assert comment["id"] == 7
    assert comment["user"] == "ll7"
    assert comment["author_association"] == "MEMBER"
    # url equals html_url, not the graphql url
    assert comment["url"] == "https://github.com/ll7/robot_sf_ll7/pull/6454#issuecomment-7"
    assert comment["body"] == "lgtm"


def test_rest_paths_do_not_request_deprecated_project_cards_field() -> None:
    """Regression check for issue #6496: no REST path may include ``projectCards``.

    ``gh pr view --comments`` fails on some GitHub CLI versions because the
    underlying GraphQL query requests the deprecated
    ``repository.pullRequest.projectCards`` field. The pure-REST helpers bypass
    this by calling ``/repos/{repo}/issues/{n}`` and
    ``/repos/{repo}/issues/{n}/comments``. This captures every ``_gh_api`` path
    used by a full PR-with-comments read and asserts none reference the
    deprecated field, so a future refactor cannot silently reintroduce a
    GraphQL projectCards dependency.
    """
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_pr_header())),
            _proc(stdout=json.dumps([_raw_comment()])),
        ]
        result = fetch_pr_with_comments(6454)
    assert result["status"] == "ok"
    assert mock_api.call_count == 2
    for call in mock_api.call_args_list:
        path = call.args[0]
        assert "projectCards" not in path, (
            f"PR comment REST path contains deprecated field: {path!r}"
        )
    # And neither the header nor the comments endpoint is a pulls/ GraphQL path.
    assert mock_api.call_args_list[0].args[0] == "repos/ll7/robot_sf_ll7/issues/6454"
    assert mock_api.call_args_list[1].args[0] == (
        f"repos/ll7/robot_sf_ll7/issues/6454/comments?per_page={COMMENTS_PAGE_SIZE}&page=1"
    )


def test_fetch_pr_with_comments_combines_header_and_thread() -> None:
    """The combined helper attaches a normalized comments list to the header."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_pr_header(number=6454, state="open"))),
            _proc(stdout=json.dumps([_raw_comment(cid=10, login="reviewer")])),
        ]
        payload = fetch_pr_with_comments(6454)
    assert payload["status"] == "ok"
    assert payload["number"] == 6454
    assert payload["state"] == "OPEN"
    assert payload["url"] == "https://github.com/ll7/robot_sf_ll7/pull/6454"
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["user"] == "reviewer"
    assert mock_api.call_count == 2


def test_fetch_pr_with_comments_propagates_header_error() -> None:
    """If the header read fails, the combined helper must not fetch comments."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(returncode=1, stderr="HTTP 404: Not Found")
        payload = fetch_pr_with_comments(999)
    assert payload["status"] == "error"
    assert "Not Found" in payload["error"]
    assert payload["number"] == 999
    assert mock_api.call_count == 1


def test_fetch_pr_with_comments_propagates_comments_error() -> None:
    """If the comments read fails, the combined helper must propagate the error."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_pr_header())),
            _proc(returncode=1, stderr="rate limit exceeded"),
        ]
        payload = fetch_pr_with_comments(6454)
    assert payload["status"] == "error"
    assert "rate limit exceeded" in payload["error"]


def test_fetch_pr_comments_paginates_until_short_page() -> None:
    """Comments should paginate by page size and stop on a short final page."""
    full_page = [_raw_comment(cid=i) for i in range(1, COMMENTS_PAGE_SIZE + 1)]
    short_page = [_raw_comment(cid=COMMENTS_PAGE_SIZE + 1)]
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(full_page)),
            _proc(stdout=json.dumps(short_page)),
        ]
        result = fetch_pr_comments(6454, max_pages=5)
    assert result["status"] == "ok"
    assert len(result["comments"]) == COMMENTS_PAGE_SIZE + 1
    # only two pages fetched (stopped early on short page)
    assert mock_api.call_count == 2


def test_fetch_pr_comments_fails_closed_when_page_budget_exceeded() -> None:
    """A full last page must fail closed rather than silently truncate."""
    full_page = [_raw_comment(cid=i) for i in range(1, COMMENTS_PAGE_SIZE + 1)]
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps(full_page))
        result = fetch_pr_comments(6454, max_pages=1)
    assert result["status"] == "error"
    assert "more than 100 comments" in result["error"]


def test_fetch_pr_comments_fails_closed_on_nonzero_exit() -> None:
    """A nonzero gh api exit must surface as a clear error payload, not raise."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(
            returncode=1,
            stderr="GraphQL: repository.pullRequest.projectCards deprecation",
        )
        result = fetch_pr_comments(6454)
    assert result["status"] == "error"
    assert "projectCards" in result["error"]


def test_fetch_pr_comments_fails_closed_on_invalid_json() -> None:
    """Malformed JSON must fail closed with a helpful snippet."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout="not-json{")
        result = fetch_pr_comments(6454)
    assert result["status"] == "error"
    assert "invalid JSON" in result["error"]


def test_fetch_pr_comments_fails_closed_on_non_list_payload() -> None:
    """A comments endpoint returning a JSON object (not a list) must fail closed."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps({"unexpected": "object"}))
        result = fetch_pr_comments(6454)
    assert result["status"] == "error"
    assert "was not a list" in result["error"]


def test_fetch_pr_comments_fails_closed_on_malformed_entry() -> None:
    """A malformed comment entry must not be silently dropped from the thread."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(stdout=json.dumps([{"id": "not-a-number"}]))
        result = fetch_pr_comments(6454)
    assert result["status"] == "error"
    assert "is malformed" in result["error"]


def test_fetch_pr_comments_rejects_non_positive_number() -> None:
    """Zero or negative numbers must be rejected without any API call."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        result = fetch_pr_comments(0)
    assert result["status"] == "error"
    assert "must be positive" in result["error"]
    mock_api.assert_not_called()


def test_fetch_pr_comments_rejects_invalid_max_pages() -> None:
    """A non-positive page budget must fail closed without any API call."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        result = fetch_pr_comments(6454, max_pages=0)
    assert result["status"] == "error"
    assert "max_pages" in result["error"]
    mock_api.assert_not_called()


def test_fetch_pr_comments_fails_closed_on_timeout() -> None:
    """A timeout must remain a structured error rather than escaping."""
    with patch("scripts.dev.gh_pr_comments_rest.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30)
        result = fetch_pr_comments(6454)
    assert result["status"] == "error"
    assert "timed out" in result["error"]


def test_fetch_pr_comments_fails_closed_when_gh_cli_missing() -> None:
    """A missing gh CLI must return an error payload, not raise."""
    with patch(
        "scripts.dev.gh_pr_comments_rest.subprocess.run", side_effect=FileNotFoundError("gh")
    ):
        result = fetch_pr_comments(6454)
    assert result["status"] == "error"
    assert "gh CLI not found" in result["error"]


def test_render_pr_comments_plain_resembles_gh_pr_view_comments() -> None:
    """Plain rendering exposes the PR header and the conversation thread."""
    payload = {
        "number": 6454,
        "title": "fix(dev): reconcile",
        "state": "MERGED",
        "url": "https://github.com/ll7/robot_sf_ll7/pull/6454",
        "comments": [
            {
                "id": 1,
                "user": "reviewer",
                "author_association": "MEMBER",
                "created_at": "2026-07-30T10:00:00Z",
                "url": "https://github.com/ll7/robot_sf_ll7/pull/6454#issuecomment-1",
                "body": "lgtm",
            }
        ],
    }
    text = render_pr_comments_plain(payload)
    assert "title:\tfix(dev): reconcile" in text
    assert "state:\tMERGED" in text
    assert "url:\thttps://github.com/ll7/robot_sf_ll7/pull/6454" in text
    assert "reviewer (MEMBER) commented on 2026-07-30T10:00:00Z" in text
    assert "lgtm" in text


def test_cli_outputs_json_by_default(capsys) -> None:
    """The CLI prints compact JSON with header + comments and exits 0."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_pr_header())),
            _proc(stdout=json.dumps([_raw_comment(cid=7)])),
        ]
        rc = main(["6454", "--repo", "ll7/robot_sf_ll7"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["number"] == 6454
    assert len(payload["comments"]) == 1


def test_cli_plain_outputs_thread(capsys) -> None:
    """``--plain`` prints a human-readable conversation thread and exits 0."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.side_effect = [
            _proc(stdout=json.dumps(_raw_pr_header())),
            _proc(stdout=json.dumps([_raw_comment(cid=7, body="## Plan")])),
        ]
        rc = main(["6454", "--repo", "ll7/robot_sf_ll7", "--plain"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "fix(dev): reconcile" in captured.out
    assert "## Plan" in captured.out


def test_cli_fails_closed_on_rest_error(capsys) -> None:
    """A REST failure prints JSON to stderr and exits nonzero."""
    with patch("scripts.dev.gh_pr_comments_rest._gh_api") as mock_api:
        mock_api.return_value = _proc(
            returncode=1, stderr="GraphQL: repository.pullRequest.projectCards"
        )
        rc = main(["6454"])
    captured = capsys.readouterr()
    assert rc == 1
    payload = json.loads(captured.err)
    assert payload["status"] == "error"
    assert "projectCards" in payload["error"]
