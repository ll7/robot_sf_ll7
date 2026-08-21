#!/usr/bin/env python3
"""Bounded retries for ``gh`` commands whose data path is GraphQL.

GitHub's GraphQL endpoint can return transient HTTP failures while REST remains
available.  Callers use this helper for read-only queue evidence and must keep
their existing fail-closed behavior when the bounded retry budget is exhausted.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Sequence

TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
_HTTP_STATUS_RE = re.compile(r"\b(?:HTTP\s*)?(429|500|502|503|504)\b", re.IGNORECASE)


def is_quota_exhausted(result: subprocess.CompletedProcess[Any]) -> bool:
    """Return whether a failed gh diagnostic reports GraphQL quota exhaustion.

    GitHub returns ``GraphQL: API rate limit already exceeded`` (or a related
    rate-limit message) when the authenticated user's GraphQL quota is spent.
    Unlike a transient 429/5xx this is not retryable within a short backoff
    window, so callers should distinguish it instead of burning the remaining
    REST-equivalent budget on identical retries. The classifier is
    message-based to stay robust across gh versions.

    Returns:
        bool: True when the diagnostic names a GraphQL or GitHub API rate limit.
    """
    text = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    if "rate limit" not in text and "rate_limit" not in text:
        return False
    return (
        "graphql" in text
        or "api rate limit" in text
        or "too many requests" in text
        or "secondary rate limit" in text
    )


@dataclass(frozen=True, slots=True)
class GraphQLRetryOutcome:
    """Result and bounded-retry evidence for one read-only ``gh`` command."""

    result: subprocess.CompletedProcess[Any]
    attempts: int
    retryable_failure: bool
    exhausted: bool
    http_status: int | None
    quota_exhausted: bool = False

    @property
    def terminal_diagnostic(self) -> str:
        """Return a concise diagnostic suitable for fail-closed queue output."""
        if self.quota_exhausted:
            detail = str(self.result.stderr or self.result.stdout or "").strip()
            if detail:
                detail = detail.splitlines()[-1][:300]
            suffix = f": {detail}" if detail else ""
            return f"GitHub GraphQL quota exhausted{suffix}"
        if not self.exhausted:
            return ""
        status = f"HTTP {self.http_status}" if self.http_status is not None else "transient HTTP"
        detail = str(self.result.stderr or self.result.stdout or "").strip()
        if detail:
            detail = detail.splitlines()[-1][:300]
        suffix = f": {detail}" if detail else ""
        return f"transient GitHub GraphQL failure after {self.attempts} attempts ({status}){suffix}"


def transient_http_status(output: str) -> int | None:
    """Return the first recognized transient HTTP status in a CLI diagnostic."""
    match = _HTTP_STATUS_RE.search(output or "")
    return int(match.group(1)) if match else None


def is_transient_failure(result: subprocess.CompletedProcess[Any]) -> bool:
    """Return whether a failed command reports a retryable HTTP response."""
    if result.returncode == 0:
        return False
    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    return transient_http_status(output) in TRANSIENT_HTTP_STATUSES


def run_with_retry(
    runner: Callable[..., subprocess.CompletedProcess[Any]],
    args: Sequence[str],
    *,
    timeout: int = 45,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> GraphQLRetryOutcome:
    """Run a GraphQL-backed ``gh`` read with bounded transient retries.

    Non-transient failures return immediately.  A persistent 429/5xx response
    returns the final failed result with ``exhausted=True``; callers must not
    convert that result into a successful or complete evidence snapshot.
    """
    attempts_limit = max(1, int(max_attempts))
    backoff = max(0.0, float(backoff_base_seconds))
    sleep_fn = sleep or time.sleep
    last_result: subprocess.CompletedProcess[Any] | None = None
    last_status: int | None = None

    for attempt in range(1, attempts_limit + 1):
        result = runner(list(args), timeout=timeout)
        last_result = result
        last_status = transient_http_status(f"{result.stderr or ''}\n{result.stdout or ''}")
        if is_quota_exhausted(result):
            return GraphQLRetryOutcome(
                result=result,
                attempts=attempt,
                retryable_failure=False,
                exhausted=True,
                http_status=last_status,
                quota_exhausted=True,
            )
        retryable = is_transient_failure(result)
        if not retryable or attempt == attempts_limit:
            return GraphQLRetryOutcome(
                result=result,
                attempts=attempt,
                retryable_failure=retryable,
                exhausted=retryable and attempt == attempts_limit,
                http_status=last_status,
            )
        sleep_fn(backoff * (2 ** (attempt - 1)))

    # The loop always returns, but keep a defensive failure for type-checkers
    # and future changes to the attempt arithmetic.
    assert last_result is not None
    return GraphQLRetryOutcome(
        result=last_result,
        attempts=attempts_limit,
        retryable_failure=True,
        exhausted=True,
        http_status=last_status,
    )
