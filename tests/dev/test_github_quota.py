"""Tests for fail-closed GitHub quota preflight decisions."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from scripts.dev.github_quota import (
    RateLimitSnapshot,
    fetch_graphql_reset_at,
    graphql_budget_decision,
    parse_rate_limit_payload,
    quota_reset_handoff,
)


def _payload(*, graphql: int = 500, core: int = 500) -> dict:
    """Return a representative ``gh api rate_limit`` payload."""
    return {
        "resources": {
            "graphql": {"remaining": graphql, "reset": 1_800_000_000},
            "core": {"remaining": core, "reset": 1_800_000_001},
        }
    }


def test_parse_rate_limit_payload_preserves_resources() -> None:
    """A valid REST response is normalized without changing quota values."""
    snapshot = parse_rate_limit_payload(_payload(graphql=321, core=654))

    assert snapshot.status == "ok"
    assert snapshot.graphql_remaining == 321
    assert snapshot.core_remaining == 654
    assert snapshot.as_dict()["graphql_reset_at"] == 1_800_000_000


def test_parse_rate_limit_payload_fails_closed_on_shape_drift() -> None:
    """Malformed quota data cannot authorize a potentially expensive lookup."""
    snapshot = parse_rate_limit_payload({"resources": {"graphql": {}}})

    assert snapshot.status == "malformed"
    assert snapshot.graphql_remaining is None


def test_graphql_budget_decision_blocks_before_safety_margin() -> None:
    """A near-limit lookup is blocked before any caller can perform a mutation."""
    decision = graphql_budget_decision(
        RateLimitSnapshot(
            status="ok",
            graphql_remaining=105,
            graphql_reset_at=1_800_000_000,
            core_remaining=500,
            core_reset_at=1_800_000_001,
        ),
        expected_graphql_requests=6,
        min_graphql_remaining=100,
    )

    assert decision["status"] == "quota_blocked"
    assert decision["reason"] == "graphql_budget_below_safety_threshold"
    assert decision["writes_performed"] is False
    assert decision["resume_after"] == 1_800_000_000


def test_graphql_budget_decision_allows_bounded_budget() -> None:
    """A bounded request estimate may proceed while retaining the configured margin."""
    decision = graphql_budget_decision(
        parse_rate_limit_payload(_payload(graphql=500, core=500)),
        expected_graphql_requests=6,
        min_graphql_remaining=100,
        expected_core_requests=1,
    )

    assert decision["status"] == "ok"
    assert decision["graphql_remaining_after_budget"] == 494
    assert decision["core_remaining_after_budget"] == 499


def test_graphql_budget_decision_rejects_negative_estimates() -> None:
    """Callers cannot bypass the guard with a nonsensical negative request budget."""
    with pytest.raises(ValueError, match="non-negative"):
        graphql_budget_decision(
            parse_rate_limit_payload(_payload()),
            expected_graphql_requests=-1,
        )


def _rate_limit_completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Build a stubbed `gh api rate_limit` result."""
    return subprocess.CompletedProcess(
        args=["gh", "api", "rate_limit"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def test_fetch_graphql_reset_at_returns_reset_epoch() -> None:
    """The reset diagnostic reads the REST reset epoch without spending GraphQL."""
    with patch(
        "scripts.dev.github_quota.subprocess.run",
        return_value=_rate_limit_completed(stdout=json.dumps(_payload())),
    ) as mock_run:
        assert fetch_graphql_reset_at() == 1_800_000_000
    assert mock_run.call_args.args[0][:3] == ["gh", "api", "rate_limit"]


def test_fetch_graphql_reset_at_fails_closed_on_transport_or_shape() -> None:
    """Any rate-limit failure yields an unknown reset, never a fabricated epoch."""
    with patch(
        "scripts.dev.github_quota.subprocess.run",
        return_value=_rate_limit_completed(returncode=1),
    ):
        assert fetch_graphql_reset_at() is None
    with patch(
        "scripts.dev.github_quota.subprocess.run",
        return_value=_rate_limit_completed(stdout="not json"),
    ):
        assert fetch_graphql_reset_at() is None
    with patch(
        "scripts.dev.github_quota.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["gh"], timeout=30),
    ):
        assert fetch_graphql_reset_at() is None


def test_quota_reset_handoff_names_reset_and_retry_command() -> None:
    """A known reset produces a bounded retry handoff, never an approval (issue #8282)."""
    handoff = quota_reset_handoff(
        retry_command="uv run python -m scripts.dev.snapshot_pr_queue 42 --review-threads --json",
        now=1_799_999_900.1,
        reset_at=1_800_000_000,
    )

    assert handoff["quota_reset_at"] == 1_800_000_000
    assert handoff["reset_in_seconds"] == 100
    assert handoff["retry_after_utc"] == "2027-01-15T08:00:00Z"
    assert "snapshot_pr_queue 42" in handoff["retry_command"]
    assert "Never admit" in handoff["handoff"]


def test_quota_reset_handoff_unknown_reset_stays_fail_closed() -> None:
    """An unavailable reset still yields a retry handoff without thread approval."""
    with patch(
        "scripts.dev.github_quota.fetch_graphql_reset_at",
        return_value=None,
    ):
        handoff = quota_reset_handoff(retry_command="retry-cmd")

    assert handoff["quota_reset_at"] is None
    assert handoff["retry_after_utc"] is None
    assert handoff["retry_command"] == "retry-cmd"
    assert "reset time is unavailable" in handoff["handoff"]
    assert "Never admit" in handoff["handoff"]


@pytest.mark.parametrize("reset_at", [10**100, -1, 1.5])
def test_quota_reset_handoff_malformed_or_overflow_epoch_stays_unknown(
    reset_at: int | float,
) -> None:
    """Malformed or unrepresentable reset epochs cannot become retry evidence."""
    handoff = quota_reset_handoff(retry_command="retry-cmd", now=0, reset_at=reset_at)

    assert handoff["quota_reset_at"] is None
    assert handoff["reset_in_seconds"] is None
    assert handoff["retry_after_utc"] is None
    assert handoff["retry_command"] == "retry-cmd"
    assert "reset time is unavailable" in handoff["handoff"]
    assert "Never admit" in handoff["handoff"]
