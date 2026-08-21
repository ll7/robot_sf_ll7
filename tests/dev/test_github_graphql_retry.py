"""Deterministic tests for bounded GraphQL-backed GitHub CLI retries."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.dev.github_graphql_retry import (
    is_quota_exhausted,
    run_with_retry,
    transient_http_status,
)


def _response(*, status: int | None = None, stdout: str = "") -> MagicMock:
    """Build a small CompletedProcess-shaped response for the retry helper."""
    return MagicMock(
        returncode=0 if status is None else 1,
        stdout=stdout,
        stderr="" if status is None else f"gh: GraphQL request failed (HTTP {status})",
    )


def test_transient_failure_recovers_with_bounded_backoff() -> None:
    """A transient 503 is retried and a later success is returned."""
    responses = [_response(status=503), _response(status=503), _response(stdout="ok")]
    sleeps: list[float] = []

    outcome = run_with_retry(
        MagicMock(side_effect=responses),
        ["api", "graphql"],
        backoff_base_seconds=1,
        sleep=sleeps.append,
    )

    assert outcome.result.stdout == "ok"
    assert outcome.attempts == 3
    assert outcome.retryable_failure is False
    assert outcome.exhausted is False
    assert sleeps == [1, 2]


def test_persistent_transient_failure_is_explicitly_exhausted() -> None:
    """A persistent transient outage never becomes a successful snapshot."""
    outcome = run_with_retry(
        MagicMock(side_effect=[_response(status=503)] * 3),
        ["api", "graphql"],
        sleep=lambda _seconds: None,
    )

    assert outcome.result.returncode == 1
    assert outcome.attempts == 3
    assert outcome.retryable_failure is True
    assert outcome.exhausted is True
    assert "after 3 attempts" in outcome.terminal_diagnostic
    assert "HTTP 503" in outcome.terminal_diagnostic


def test_non_transient_failure_is_not_retried() -> None:
    """Authentication and other permanent failures remain immediate errors."""
    runner = MagicMock(return_value=_response(status=404))

    outcome = run_with_retry(runner, ["api", "graphql"], sleep=lambda _seconds: None)

    assert runner.call_count == 1
    assert outcome.exhausted is False
    assert outcome.retryable_failure is False
    assert transient_http_status("HTTP 503 Service Unavailable") == 503
    assert transient_http_status("HTTP 404 Not Found") is None


def test_is_quota_exhausted_classifies_graphql_rate_limit() -> None:
    """The exact issue #7705 diagnostic is classified as quota exhaustion."""
    result = MagicMock(
        returncode=1,
        stdout="",
        stderr="gh: GraphQL: API rate limit already exceeded (403)\n",
    )
    assert is_quota_exhausted(result) is True


def test_is_quota_exhausted_rejects_unrelated_failures() -> None:
    """Non-quota errors, including plain 429 and 404, are not quota exhaustion."""
    assert is_quota_exhausted(_response(status=429)) is False
    assert is_quota_exhausted(_response(status=404)) is False
    quota_less = MagicMock(returncode=1, stdout="", stderr="gh: internal server error")
    assert is_quota_exhausted(quota_less) is False


def test_quota_exhaustion_is_not_retried_and_is_flagged() -> None:
    """Quota exhaustion returns immediately with no futile retry backoff."""
    runner = MagicMock(
        return_value=MagicMock(
            returncode=1,
            stdout="",
            stderr="gh: GraphQL: API rate limit already exceeded (403)",
        )
    )
    sleeps: list[float] = []

    outcome = run_with_retry(
        runner,
        ["api", "graphql"],
        sleep=sleeps.append,
    )

    assert runner.call_count == 1
    assert sleeps == []
    assert outcome.quota_exhausted is True
    assert outcome.exhausted is True
    assert outcome.retryable_failure is False
    assert "quota exhausted" in outcome.terminal_diagnostic
