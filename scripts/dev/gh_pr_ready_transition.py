#!/usr/bin/env python3
"""Explicitly mark one exact-head draft PR ready, then verify it through REST.

This command is a one-shot state transition. It never retries or sleeps, and a
successful GraphQL response is not sufficient: the same open head must read
back from the REST API with ``draft == false`` before the command reports
``ready_confirmed``. On an ambiguous result, inspect the PR and invoke the
printed command again only when another explicit attempt is appropriate.

Usage::

    scripts/dev/gh_pr_ready_transition.py 5220 \\
        --expected-head-sha <full-head-sha> --repo ll7/robot_sf_ll7
"""

from __future__ import annotations

import argparse
import email.utils
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Direct execution must resolve this checkout's REST transport and guards.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev._gh_rest import gh_api_post as _gh_api_post
from scripts.dev.pr_write_guard import (
    DEFAULT_REPO,
    FULL_SHA_RE,
    PRWriteGuardOptions,
    guard_pr_write,
    pr_write_lock,
)

PR_READY_MUTATION = """mutation MarkPullRequestReadyForReview($input: MarkPullRequestReadyForReviewInput!) {
  markPullRequestReadyForReview(input: $input) {
    pullRequest { id }
  }
}"""
MUTATION_TIMEOUT_SECONDS = 20
CONSERVATIVE_COOLDOWN_SECONDS = 60
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HTTP_STATUS_RE = re.compile(r"^HTTP/\d+(?:\.\d+)?[ \t]+(?P<status>\d{3})(?:[ \t].*)?$", re.I)
_HEADER_RE = re.compile(r"^([!#$%&'*+.^_`|~0-9A-Za-z-]+):[ \t]*(.*)$")
_SECONDARY_RATE_MARKERS = (
    "secondary rate limit",
    "secondary-rate-limit",
    "abuse detection",
)
_RATE_MARKERS = (
    "rate limit",
    "rate-limit",
    "rate_limit",
    "too many requests",
)
_MUTATION_FIELD = "markPullRequestReadyForReview"


@dataclass(frozen=True)
class IncludedHTTPResponse:
    """One final response parsed from ``gh api --include`` output."""

    status_code: int
    headers: dict[str, tuple[str, ...]]
    body: str


def _parse_header_frame(
    lines: list[str], cursor: int
) -> tuple[int, dict[str, tuple[str, ...]], int] | None:
    """Parse one HTTP status line and its complete header block."""
    if cursor >= len(lines):
        return None
    status_match = _HTTP_STATUS_RE.fullmatch(lines[cursor].strip())
    if status_match is None:
        return None
    status_code = int(status_match.group("status"))
    cursor += 1
    collected: dict[str, list[str]] = {}
    while cursor < len(lines) and lines[cursor].strip():
        header_match = _HEADER_RE.fullmatch(lines[cursor])
        if header_match is None:
            return None
        name = header_match.group(1).lower()
        value = header_match.group(2).strip()
        if any(ord(character) < 32 and character != "\t" for character in value):
            return None
        collected.setdefault(name, []).append(value)
        cursor += 1
    if cursor >= len(lines):
        return None
    return status_code, {name: tuple(values) for name, values in collected.items()}, cursor + 1


def parse_included_http_response(stdout: object) -> IncludedHTTPResponse | None:
    """Parse status, headers, and body from ``gh api --include`` output.

    GitHub CLI emits status lines such as ``HTTP/2.0 200 OK``. The parser also
    accepts multiple header-only response frames, while requiring a complete
    status/header separator before exposing a body to JSON parsing.
    """
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    lines = stdout.splitlines()
    cursor = 0
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    while cursor < len(lines):
        frame = _parse_header_frame(lines, cursor)
        if frame is None:
            return None
        status_code, headers, cursor = frame
        next_status = cursor
        while next_status < len(lines) and not lines[next_status].strip():
            next_status += 1
        if next_status < len(lines) and _HTTP_STATUS_RE.fullmatch(lines[next_status].strip()):
            cursor = next_status
            continue
        return IncludedHTTPResponse(status_code, headers, "\n".join(lines[cursor:]).strip())
    return None


def _single_header(response: IncludedHTTPResponse, name: str) -> str | None:
    """Return one unambiguous response-header value."""
    values = response.headers.get(name.lower(), ())
    if not values or len(set(values)) != 1:
        return None
    return values[0]


def _bounded_integer(raw: str | None) -> int | None:
    """Parse a non-negative decimal header value without accepting booleans."""
    if raw is None or re.fullmatch(r"\d{1,20}", raw) is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _retry_after_seconds(raw: str | None, *, now: float) -> int | None:
    """Parse numeric or HTTP-date ``Retry-After`` into a non-negative delay."""
    numeric = _bounded_integer(raw)
    if numeric is not None:
        return numeric
    if raw is None:
        return None
    try:
        retry_at = email.utils.parsedate_to_datetime(raw)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = math.ceil(retry_at.timestamp() - now)
    except (OverflowError, OSError, TypeError, ValueError, IndexError):
        return None
    return max(0, seconds)


def _has_graphql_errors(body: str) -> bool:
    """Return whether a JSON GraphQL envelope contains at least one error."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    errors = payload.get("errors") if isinstance(payload, dict) else None
    return isinstance(errors, list) and bool(errors)


def _rate_limit_kind(
    response: IncludedHTTPResponse,
    *,
    body_lower: str,
    remaining: int | None,
    retry_after_raw: str | None,
    has_graphql_errors: bool,
) -> str | None:
    """Classify primary/secondary rate evidence from one mutation response."""
    has_rate_message = any(marker in body_lower for marker in _RATE_MARKERS)
    has_secondary_message = any(marker in body_lower for marker in _SECONDARY_RATE_MARKERS)
    if has_secondary_message:
        return "secondary"
    if "primary rate limit" in body_lower or "api rate limit exceeded" in body_lower:
        return "primary"
    if response.status_code == 429:
        return "secondary"
    if response.status_code == 403 and retry_after_raw is not None:
        return "secondary"
    if response.status_code == 403 and has_rate_message:
        return "secondary"
    if retry_after_raw is not None and has_rate_message:
        return "secondary"
    if remaining == 0 and has_graphql_errors:
        return "primary"
    return None


def _rate_limit_evidence(
    response: IncludedHTTPResponse | None,
    *,
    now: float,
) -> dict[str, Any] | None:
    """Classify rate-limit errors using only this mutation response."""
    if response is None:
        return None
    body_lower = response.body.lower()
    remaining = _bounded_integer(_single_header(response, "x-ratelimit-remaining"))
    retry_after_raw = _single_header(response, "retry-after")
    kind = _rate_limit_kind(
        response,
        body_lower=body_lower,
        remaining=remaining,
        retry_after_raw=retry_after_raw,
        has_graphql_errors=_has_graphql_errors(response.body),
    )
    if kind is None:
        return None
    retry_after = _retry_after_seconds(retry_after_raw, now=now)
    reset_at = _bounded_integer(_single_header(response, "x-ratelimit-reset"))
    cooldowns: list[tuple[int, str]] = []
    if retry_after is not None:
        cooldowns.append((retry_after, "Retry-After"))
    if reset_at is not None:
        try:
            if not math.isfinite(now):
                raise ValueError("invalid clock")
            cooldowns.append((max(0, math.ceil(reset_at - now)), "X-RateLimit-Reset"))
        except (OverflowError, TypeError, ValueError):
            pass
    if cooldowns:
        cooldown_seconds = max(seconds for seconds, _source in cooldowns)
        sources = "+".join(source for seconds, source in cooldowns if seconds == cooldown_seconds)
        cooldown_basis = "observed_header"
    else:
        cooldown_seconds = CONSERVATIVE_COOLDOWN_SECONDS
        cooldown_basis = "conservative_minimum"
        sources = "conservative_minimum"
    evidence: dict[str, Any] = {
        "rate_limit_kind": kind,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_basis": cooldown_basis,
        "cooldown_source": sources,
    }
    if reset_at is not None:
        evidence["rate_limit_reset_at"] = reset_at
    if retry_after is not None:
        evidence["retry_after_seconds"] = retry_after
    return evidence


def _graphql_outcome(response: IncludedHTTPResponse, expected_node_id: str) -> str:
    """Parse one GraphQL result without returning untrusted error messages."""
    try:
        payload = json.loads(response.body)
    except (json.JSONDecodeError, TypeError):
        return "response_unparseable"
    if not isinstance(payload, dict):
        return "response_unparseable"
    errors = payload.get("errors", [])
    if not isinstance(errors, list) or any(
        not isinstance(error, dict) or not isinstance(error.get("message"), str) for error in errors
    ):
        return "response_unparseable"
    if errors:
        return "graphql_error"
    data = payload.get("data")
    mutation = data.get(_MUTATION_FIELD) if isinstance(data, dict) else None
    pull_request = mutation.get("pullRequest") if isinstance(mutation, dict) else None
    observed_node_id = pull_request.get("id") if isinstance(pull_request, dict) else None
    if observed_node_id == expected_node_id:
        return "accepted"
    return "response_unconfirmed"


def _mutation_outcome(
    completed: subprocess.CompletedProcess[str],
    *,
    expected_node_id: str,
    now: float,
) -> dict[str, Any]:
    """Return a safe summary of the mutation response without exposing its body."""
    response = parse_included_http_response(getattr(completed, "stdout", ""))
    rate_limit = _rate_limit_evidence(response, now=now)
    if rate_limit is not None:
        return {"mutation_outcome": "rate_limited", **rate_limit}
    returncode = getattr(completed, "returncode", 1)
    if returncode == 124:
        return {"mutation_outcome": "transport_timeout"}
    if returncode != 0:
        return {"mutation_outcome": "transport_error"}
    if response is None:
        return {"mutation_outcome": "response_unparseable"}
    if response.status_code < 200 or response.status_code >= 300:
        return {"mutation_outcome": "http_error", "http_status": response.status_code}
    return {
        "mutation_outcome": _graphql_outcome(response, expected_node_id),
        "http_status": response.status_code,
    }


def _valid_request(number: object, repo: object, expected_head_sha: object) -> str | None:
    """Validate public CLI inputs before the first GitHub request."""
    if type(number) is not int or number < 1:
        return "PR number must be a positive integer"
    if not isinstance(repo, str) or _REPO_RE.fullmatch(repo) is None:
        return "repo must have owner/repo form using letters, digits, dot, underscore, or hyphen"
    if not isinstance(expected_head_sha, str) or FULL_SHA_RE.fullmatch(expected_head_sha) is None:
        return "expected_head_sha must be a full 40-character SHA"
    return None


def _resume_fields(number: int, repo: str, expected_head_sha: str) -> dict[str, str]:
    """Return credential-free manual recovery instructions for this exact PR."""
    return {
        "resume_command": (
            f"scripts/dev/gh_pr_ready_transition.py {number} --expected-head-sha "
            f"{expected_head_sha} --repo {repo}"
        ),
        "ui_url": f"https://github.com/{repo}/pull/{number}",
    }


def _public_guard_fields(guard: dict[str, Any]) -> dict[str, Any]:
    """Copy only safe state evidence from the shared write guard."""
    fields = ("observed_state", "observed_head_sha", "observed_draft")
    return {field: guard[field] for field in fields if field in guard}


def _preflight_blocked(
    guard: dict[str, Any], *, number: int, repo: str, expected_head_sha: str
) -> dict[str, Any]:
    """Map an untrusted preflight result to a safe, bounded receipt."""
    if guard.get("status") == "review_skipped_stale_state":
        reason = guard.get("reason")
        if reason not in {"pr_not_open", "head_sha_changed", "base_sha_changed"}:
            reason = "stale_state"
    else:
        reason = "preflight_unavailable"
    return {
        "status": "blocked",
        "reason": reason,
        "number": number,
        "repo": repo,
        "expected_head_sha": expected_head_sha,
        **_public_guard_fields(guard),
        "message": "PR state could not be safely admitted for a ready transition.",
        **_resume_fields(number, repo, expected_head_sha),
    }


def _finish_after_mutation(
    *,
    number: int,
    repo: str,
    expected_head_sha: str,
    mutation: dict[str, Any],
    readback: dict[str, Any],
) -> dict[str, Any]:
    """Confirm the exact open head after every mutation attempt."""
    common: dict[str, Any] = {
        "number": number,
        "repo": repo,
        "expected_head_sha": expected_head_sha,
        **_public_guard_fields(readback),
        **mutation,
    }
    if readback.get("status") == "ok" and readback.get("observed_draft") is False:
        return {"status": "ready_confirmed", **common}
    if readback.get("status") == "ok" and readback.get("observed_draft") is True:
        reason = "still_draft"
    elif readback.get("status") == "review_skipped_stale_state":
        reason = readback.get("reason")
        if reason not in {"pr_not_open", "head_sha_changed", "base_sha_changed"}:
            reason = "stale_state"
    else:
        reason = "readback_unavailable"
    if mutation.get("mutation_outcome") == "rate_limited":
        reason = "rate_limited"
    return {
        "status": "blocked",
        "reason": reason,
        "message": "The requested ready state was not confirmed for the expected open head.",
        **common,
        **_resume_fields(number, repo, expected_head_sha),
    }


def transition_pr_to_ready(
    number: int,
    *,
    expected_head_sha: str,
    repo: str = DEFAULT_REPO,
) -> dict[str, Any]:
    """Mark a matching draft PR ready once, then confirm its exact head through REST."""
    error = _valid_request(number, repo, expected_head_sha)
    if error is not None:
        return {"status": "error", "error": error}
    try:
        lock_context = pr_write_lock(repo, number)
        lock_context.__enter__()
    except RuntimeError:
        return {
            "status": "blocked",
            "reason": "write_lock_unavailable",
            "number": number,
            "repo": repo,
            "expected_head_sha": expected_head_sha,
            "message": "Could not acquire the local PR write lock; no mutation was attempted.",
            **_resume_fields(number, repo, expected_head_sha),
        }
    try:
        preflight = guard_pr_write(
            number,
            repo=repo,
            expected_head_sha=expected_head_sha,
            operation="mark_ready_transition",
            options=PRWriteGuardOptions(include_node_id=True, include_draft=True),
        )
        if preflight.get("status") != "ok":
            return _preflight_blocked(
                preflight,
                number=number,
                repo=repo,
                expected_head_sha=expected_head_sha,
            )
        if preflight.get("observed_draft") is False:
            return {
                "status": "already_ready",
                "number": number,
                "repo": repo,
                "expected_head_sha": expected_head_sha,
                **_public_guard_fields(preflight),
                "message": "The matching open PR is already ready for review; no mutation was sent.",
            }
        node_id = preflight.get("observed_node_id")
        if not isinstance(node_id, str) or not node_id:
            return _preflight_blocked(
                {"status": "error"},
                number=number,
                repo=repo,
                expected_head_sha=expected_head_sha,
            )

        try:
            completed = _gh_api_post(
                "graphql",
                {
                    "query": PR_READY_MUTATION,
                    "variables": {"input": {"pullRequestId": node_id}},
                },
                extra_args=["--include"],
                timeout=MUTATION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            completed = subprocess.CompletedProcess(
                ["gh", "api", "graphql", "--include"],
                124,
                stdout="",
                stderr="",
            )
        mutation = _mutation_outcome(completed, expected_node_id=node_id, now=time.time())
        readback = guard_pr_write(
            number,
            repo=repo,
            expected_head_sha=expected_head_sha,
            operation="mark_ready_readback",
            options=PRWriteGuardOptions(include_draft=True),
        )
        return _finish_after_mutation(
            number=number,
            repo=repo,
            expected_head_sha=expected_head_sha,
            mutation=mutation,
            readback=readback,
        )
    finally:
        lock_context.__exit__(None, None, None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Positive pull-request number to mark ready.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/repo to update.")
    parser.add_argument(
        "--expected-head-sha",
        required=True,
        help="Full 40-character PR head SHA captured by the review lane.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the one-shot transition and emit one credential-free JSON result."""
    args = _build_parser().parse_args(argv)
    result = transition_pr_to_ready(
        args.number,
        expected_head_sha=args.expected_head_sha,
        repo=args.repo,
    )
    successful = result.get("status") in {"ready_confirmed", "already_ready"}
    print(json.dumps(result, sort_keys=True), file=sys.stdout if successful else sys.stderr)
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
