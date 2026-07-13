"""Shared configuration and error-signature helpers for benchmark circuit breakers."""

from __future__ import annotations

from typing import Any

DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 10
_CIRCUIT_BREAKER_MSG_PREFIX_LEN = 200


def normalize_circuit_breaker_threshold(value: int | None) -> int:
    """Normalize and validate the optional circuit-breaker threshold.

    ``None`` selects the default, ``0`` explicitly disables the breaker, and
    negative or non-integer values are rejected before any benchmark work runs.

    Returns:
        The validated non-negative threshold.
    """
    if value is None:
        return DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("circuit_breaker_threshold must be an integer or None")
    if value < 0:
        raise ValueError("circuit_breaker_threshold must be non-negative")
    return value


def error_signature(exc: Exception) -> tuple[str, str]:
    """Return a stable exception type/message-prefix signature."""
    msg = str(exc).replace("\r\n", "\n").replace("\r", "\n").strip()
    return (type(exc).__name__, msg[:_CIRCUIT_BREAKER_MSG_PREFIX_LEN])


def build_abort_metadata(
    *,
    signature: tuple[str, str],
    consecutive_failures: int,
    first_fail_index: int,
    episodes_completed_before_onset: int,
    total_jobs: int,
) -> dict[str, Any]:
    """Build the common circuit-breaker abort payload.

    Returns:
        Structured metadata describing the aborted arm and projected savings.
    """
    projected_remaining = total_jobs - first_fail_index - consecutive_failures + 1
    return {
        "status": "aborted_systematic_failure",
        "signature": {
            "type": signature[0],
            "message_prefix": signature[1],
        },
        "consecutive_failures": consecutive_failures,
        "first_fail_index": first_fail_index,
        "episodes_completed_before_onset": episodes_completed_before_onset,
        "projected_episodes_saved": projected_remaining,
        "projected_walltime_saved_hint": f"~{projected_remaining} episodes not attempted",
    }
