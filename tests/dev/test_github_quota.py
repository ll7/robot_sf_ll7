"""Tests for fail-closed GitHub quota preflight decisions."""

from __future__ import annotations

import pytest

from scripts.dev.github_quota import (
    RateLimitSnapshot,
    graphql_budget_decision,
    parse_rate_limit_payload,
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
