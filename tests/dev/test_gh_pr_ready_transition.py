"""Offline tests for the explicit exact-head draft-to-ready transition."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from scripts.dev.gh_pr_ready_transition import (
    PR_READY_MUTATION,
    main,
    parse_included_http_response,
    transition_pr_to_ready,
)
from scripts.dev.pr_write_guard import PRWriteGuardOptions

NUMBER = 5220
REPO = "ll7/robot_sf_ll7"
HEAD_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
NODE_ID = "PR_kwDO_test"
NOW = 1_700_000_000.0


def _proc(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Build a fake gh process result."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)


def _http_response(
    status: int,
    headers: dict[str, str],
    payload: object | str,
) -> str:
    """Build the HTTP/2 framing emitted by ``gh api --include``."""
    body = payload if isinstance(payload, str) else json.dumps(payload)
    lines = [f"HTTP/2.0 {status} Fixture"]
    lines.extend(f"{key}: {value}" for key, value in headers.items())
    return "\r\n".join([*lines, "", body])


def _guard(*, draft: bool, head_sha: str = HEAD_SHA, state: str = "OPEN") -> dict[str, object]:
    """Build a successful shared write-guard snapshot."""
    return {
        "status": "ok",
        "observed_state": state,
        "observed_head_sha": head_sha,
        "observed_draft": draft,
        "observed_node_id": NODE_ID,
    }


def _accepted_mutation() -> subprocess.CompletedProcess[str]:
    """Return the documented GraphQL success shape with included headers."""
    return _proc(
        stdout=_http_response(
            200,
            {"Content-Type": "application/json", "X-Ratelimit-Remaining": "4999"},
            {"data": {"markPullRequestReadyForReview": {"pullRequest": {"id": NODE_ID}}}},
        )
    )


def test_http2_include_parser_reads_status_headers_and_body() -> None:
    """HTTP/2.0 status framing and case-insensitive headers are parsed."""
    parsed = parse_included_http_response(
        "HTTP/1.1 100 Continue\r\n\r\n"
        'HTTP/2.0 200 OK\r\nX-Ratelimit-Remaining: 0\r\n\r\n{"errors": []}'
    )

    assert parsed is not None
    assert parsed.status_code == 200
    assert parsed.headers["x-ratelimit-remaining"] == ("0",)
    assert json.loads(parsed.body) == {"errors": []}


@pytest.mark.parametrize("stdout", ["", "200 OK\n\n{}", "HTTP/2 200 OK\nBad Header\n\n{}"])
def test_http_include_parser_rejects_incomplete_framing(stdout: str) -> None:
    """Malformed or absent status/header framing cannot authorize a result."""
    assert parse_included_http_response(stdout) is None


def test_success_uses_node_id_graphql_mutation_then_rest_confirms() -> None:
    """The exact REST head is preflighted and read back after one GraphQL mutation."""
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=False)],
        ) as guard,
        patch(
            "scripts.dev.gh_pr_ready_transition._gh_api_post",
            return_value=_accepted_mutation(),
        ) as post,
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "ready_confirmed"
    assert result["mutation_outcome"] == "accepted"
    assert result["observed_head_sha"] == HEAD_SHA
    assert result["observed_draft"] is False
    assert guard.call_count == 2
    assert guard.call_args_list[0].kwargs == {
        "repo": REPO,
        "expected_head_sha": HEAD_SHA,
        "operation": "mark_ready_transition",
        "options": PRWriteGuardOptions(include_node_id=True, include_draft=True),
    }
    assert guard.call_args_list[1].kwargs == {
        "repo": REPO,
        "expected_head_sha": HEAD_SHA,
        "operation": "mark_ready_readback",
        "options": PRWriteGuardOptions(include_draft=True),
    }
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "graphql"
    assert args[1] == {
        "query": PR_READY_MUTATION,
        "variables": {"input": {"pullRequestId": NODE_ID}},
    }
    assert kwargs == {"extra_args": ["--include"], "timeout": 20}


def test_matching_open_ready_pr_is_an_explicit_noop() -> None:
    """A matching ready PR does not receive another mutation."""
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            return_value=_guard(draft=False),
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post") as post,
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "already_ready"
    post.assert_not_called()


def test_success_with_zero_remaining_is_not_called_rate_limited() -> None:
    """Exhausting the last primary point after success is not a rate failure."""
    final_point = _proc(
        stdout=_http_response(
            200,
            {"X-Ratelimit-Remaining": "0"},
            {"data": {"markPullRequestReadyForReview": {"pullRequest": {"id": NODE_ID}}}},
        )
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=False)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=final_point),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "ready_confirmed"
    assert result["mutation_outcome"] == "accepted"


@pytest.mark.parametrize(
    ("guard_result", "expected_reason"),
    (
        (
            {"status": "review_skipped_stale_state", "reason": "head_sha_changed"},
            "head_sha_changed",
        ),
        (
            {"status": "review_skipped_stale_state", "reason": "pr_not_open"},
            "pr_not_open",
        ),
        ({"status": "error", "error": "private API detail"}, "preflight_unavailable"),
    ),
)
def test_stale_closed_or_unreadable_pr_is_not_mutated(
    guard_result: dict[str, object], expected_reason: str
) -> None:
    """Only a readable open PR with the expected head reaches the mutation."""
    with (
        patch("scripts.dev.gh_pr_ready_transition.guard_pr_write", return_value=guard_result),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post") as post,
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "blocked"
    assert result["reason"] == expected_reason
    assert "private API detail" not in json.dumps(result)
    post.assert_not_called()


def test_primary_rate_limit_uses_mutation_headers_not_rate_limit_summary() -> None:
    """A 200 GraphQL quota error uses its own headers despite a contradictory summary."""
    rate_response = _proc(
        stdout=_http_response(
            200,
            {"X-Ratelimit-Remaining": "0", "X-Ratelimit-Reset": str(int(NOW + 123))},
            {"errors": [{"message": "API rate limit exceeded"}]},
        )
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ) as guard,
        patch(
            "scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response
        ) as post,
        patch("scripts.dev.gh_pr_ready_transition.time.time", return_value=NOW),
        patch(
            "scripts.dev.github_quota.fetch_graphql_reset_at", return_value=int(NOW + 9999)
        ) as summary,
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "blocked"
    assert result["reason"] == "rate_limited"
    assert result["rate_limit_kind"] == "primary"
    assert result["cooldown_seconds"] == 123
    assert result["cooldown_source"] == "X-RateLimit-Reset"
    assert result["resume_command"] == (
        f"scripts/dev/gh_pr_ready_transition.py {NUMBER} "
        f"--expected-head-sha {HEAD_SHA} --repo {REPO}"
    )
    assert result["ui_url"] == f"https://github.com/{REPO}/pull/{NUMBER}"
    assert guard.call_count == 2
    post.assert_called_once()
    summary.assert_not_called()


def test_secondary_rate_limit_uses_retry_after_header_without_retrying() -> None:
    """Secondary-limit cooldowns come from the mutation response and are manual."""
    rate_response = _proc(
        returncode=1,
        stderr="private transport diagnostic",
        stdout=_http_response(
            200,
            {"Retry-After": "17", "X-Ratelimit-Remaining": "25"},
            {"errors": [{"message": "You hit a secondary rate limit."}]},
        ),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch(
            "scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response
        ) as post,
        patch("scripts.dev.gh_pr_ready_transition.time.time", return_value=NOW),
        patch("scripts.dev.gh_pr_ready_transition.time.sleep") as sleep,
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "blocked"
    assert result["reason"] == "rate_limited"
    assert result["rate_limit_kind"] == "secondary"
    assert result["cooldown_seconds"] == 17
    assert result["cooldown_source"] == "Retry-After"
    post.assert_called_once()
    sleep.assert_not_called()


def test_http_429_is_secondary_throttling_without_rate_limit_text() -> None:
    """HTTP 429 itself is sufficient evidence of secondary throttling."""
    rate_response = _proc(
        returncode=1,
        stdout=_http_response(429, {}, {"message": "Request throttled"}),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "blocked"
    assert result["reason"] == "rate_limited"
    assert result["mutation_outcome"] == "rate_limited"
    assert result["rate_limit_kind"] == "secondary"
    assert result["cooldown_basis"] == "conservative_minimum"


@pytest.mark.parametrize(
    ("headers", "payload", "expected_cooldown", "expected_source"),
    (
        (
            {"Retry-After": "23", "X-Ratelimit-Remaining": "0"},
            {"errors": [{"message": "Please retry later"}]},
            23,
            "Retry-After",
        ),
        (
            {},
            {"errors": [{"message": "You hit a secondary rate limit."}]},
            60,
            "conservative_minimum",
        ),
    ),
)
def test_http_403_secondary_retry_after_or_message_is_classified(
    headers: dict[str, str],
    payload: object,
    expected_cooldown: int,
    expected_source: str,
) -> None:
    """A 403 with Retry-After or an explicit secondary marker is throttling."""
    rate_response = _proc(
        returncode=1,
        stdout=_http_response(403, headers, payload),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response),
        patch("scripts.dev.gh_pr_ready_transition.time.time", return_value=NOW),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "blocked"
    assert result["reason"] == "rate_limited"
    assert result["rate_limit_kind"] == "secondary"
    assert result["cooldown_seconds"] == expected_cooldown
    assert result["cooldown_source"] == expected_source


def test_permission_http_403_without_rate_evidence_is_not_rate_limited() -> None:
    """A normal permission response remains an HTTP/transport failure."""
    permission_response = _proc(
        returncode=1,
        stdout=_http_response(
            403,
            {"Content-Type": "application/json"},
            {"message": "Resource not accessible by integration"},
        ),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch(
            "scripts.dev.gh_pr_ready_transition._gh_api_post",
            return_value=permission_response,
        ),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["status"] == "blocked"
    assert result["reason"] == "still_draft"
    assert result["mutation_outcome"] == "transport_error"
    assert "rate_limit_kind" not in result


def test_explicit_secondary_marker_wins_when_primary_remaining_is_zero() -> None:
    """An explicit secondary message takes precedence over exhausted quota headers."""
    rate_response = _proc(
        returncode=1,
        stdout=_http_response(
            200,
            {"X-Ratelimit-Remaining": "0", "X-Ratelimit-Reset": str(int(NOW + 31))},
            {"errors": [{"message": "You hit a secondary rate limit."}]},
        ),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response),
        patch("scripts.dev.gh_pr_ready_transition.time.time", return_value=NOW),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["reason"] == "rate_limited"
    assert result["rate_limit_kind"] == "secondary"
    assert result["cooldown_seconds"] == 31
    assert result["cooldown_source"] == "X-RateLimit-Reset"


def test_zero_remaining_with_graphql_errors_is_primary_without_secondary_evidence() -> None:
    """The exhausted primary quota fallback remains available without a marker."""
    rate_response = _proc(
        returncode=1,
        stdout=_http_response(
            200,
            {"X-Ratelimit-Remaining": "0"},
            {"errors": [{"message": "Request could not be completed"}]},
        ),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response),
        patch("scripts.dev.gh_pr_ready_transition.time.time", return_value=NOW),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["reason"] == "rate_limited"
    assert result["rate_limit_kind"] == "primary"
    assert result["cooldown_basis"] == "conservative_minimum"


def test_rate_limit_without_cooldown_uses_conservative_manual_wait() -> None:
    """A secondary-limit response without usable headers advises a 60-second wait."""
    rate_response = _proc(
        stdout=_http_response(
            200,
            {},
            {"errors": [{"message": "Secondary rate limit was exceeded."}]},
        )
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=rate_response),
        patch("scripts.dev.gh_pr_ready_transition.time.time", return_value=NOW),
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert result["rate_limit_kind"] == "secondary"
    assert result["cooldown_seconds"] == 60
    assert result["cooldown_basis"] == "conservative_minimum"


@pytest.mark.parametrize(
    ("mutation", "readback", "expected_reason"),
    (
        (_proc(returncode=124), _guard(draft=True), "still_draft"),
        (
            _proc(
                returncode=1,
                stdout=_http_response(502, {}, "upstream unavailable"),
                stderr="credential-like ghp_fixture_secret",
            ),
            _guard(draft=False),
            "ready_confirmed",
        ),
        (
            _accepted_mutation(),
            {"status": "error", "error": "private readback detail"},
            "readback_unavailable",
        ),
        (
            _accepted_mutation(),
            {"status": "review_skipped_stale_state", "reason": "head_sha_changed"},
            "head_sha_changed",
        ),
    ),
)
def test_mutation_attempt_always_reads_back_even_when_outcome_is_ambiguous(
    mutation: subprocess.CompletedProcess[str],
    readback: dict[str, object],
    expected_reason: str,
) -> None:
    """Timeouts and nonzero responses are followed by exact-head REST readback."""
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), readback],
        ) as guard,
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=mutation) as post,
    ):
        result = transition_pr_to_ready(NUMBER, repo=REPO, expected_head_sha=HEAD_SHA)

    assert guard.call_count == 2
    post.assert_called_once()
    if expected_reason == "ready_confirmed":
        assert result["status"] == "ready_confirmed"
    else:
        assert result["status"] == "blocked"
        assert result["reason"] == expected_reason


def test_cli_output_never_echoes_credentials_or_raw_graphql_errors(capsys) -> None:
    """Structured CLI receipts omit stderr and untrusted GraphQL error text."""
    secret = "ghp_fixture_secret"
    mutation = _proc(
        returncode=1,
        stderr=f"authorization diagnostic contains {secret}",
        stdout=_http_response(
            200,
            {},
            {"errors": [{"message": f"mutation rejected; credential {secret}"}]},
        ),
    )
    with (
        patch(
            "scripts.dev.gh_pr_ready_transition.guard_pr_write",
            side_effect=[_guard(draft=True), _guard(draft=True)],
        ),
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post", return_value=mutation),
    ):
        exit_code = main([str(NUMBER), "--repo", REPO, "--expected-head-sha", HEAD_SHA])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert secret not in captured.out
    assert secret not in captured.err
    receipt = json.loads(captured.err)
    assert receipt["status"] == "blocked"
    assert receipt["mutation_outcome"] == "transport_error"


@pytest.mark.parametrize(
    ("number", "repo", "expected_head_sha", "expected_error"),
    (
        (0, REPO, HEAD_SHA, "PR number must be a positive integer"),
        (
            NUMBER,
            "ll7/robot_sf_ll7;echo bad",
            HEAD_SHA,
            "repo must have owner/repo form using letters, digits, dot, underscore, or hyphen",
        ),
        (NUMBER, REPO, "abc", "expected_head_sha must be a full 40-character SHA"),
    ),
)
def test_invalid_request_fails_before_any_transport(
    number: int, repo: str, expected_head_sha: str, expected_error: str
) -> None:
    """Request validation occurs before any REST or GraphQL call."""
    with (
        patch("scripts.dev.gh_pr_ready_transition.guard_pr_write") as guard,
        patch("scripts.dev.gh_pr_ready_transition._gh_api_post") as post,
    ):
        result = transition_pr_to_ready(
            number,
            repo=repo,
            expected_head_sha=expected_head_sha,
        )

    assert result == {"status": "error", "error": expected_error}
    guard.assert_not_called()
    post.assert_not_called()
