"""Small, fail-closed helpers for GitHub REST and GraphQL quota checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_CORE_SAFETY_THRESHOLD = 10
DEFAULT_GRAPHQL_SAFETY_THRESHOLD = 100


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    """Normalized rate-limit values needed by bounded GitHub workflows."""

    status: str
    graphql_remaining: int | None = None
    graphql_reset_at: int | None = None
    core_remaining: int | None = None
    core_reset_at: int | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible snapshot."""
        return asdict(self)


def _resource(payload: Any, name: str) -> dict[str, Any] | None:
    """Return one rate-limit resource when the response has the expected shape."""
    if not isinstance(payload, dict):
        return None
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        return None
    resource = resources.get(name)
    return resource if isinstance(resource, dict) else None


def _integer(resource: dict[str, Any], name: str) -> int | None:
    """Parse a non-negative integer field without accepting booleans."""
    value = resource.get(name)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def parse_rate_limit_payload(payload: Any) -> RateLimitSnapshot:
    """Normalize a ``gh api rate_limit`` JSON response, failing closed on drift."""
    graphql = _resource(payload, "graphql")
    core = _resource(payload, "core")
    if graphql is None or core is None:
        return RateLimitSnapshot(
            status="malformed",
            error="rate_limit response is missing graphql or core resources",
        )

    graphql_remaining = _integer(graphql, "remaining")
    graphql_reset_at = _integer(graphql, "reset")
    core_remaining = _integer(core, "remaining")
    core_reset_at = _integer(core, "reset")
    if None in (graphql_remaining, graphql_reset_at, core_remaining, core_reset_at):
        return RateLimitSnapshot(
            status="malformed",
            error="rate_limit resources have missing or invalid integer fields",
        )
    return RateLimitSnapshot(
        status="ok",
        graphql_remaining=graphql_remaining,
        graphql_reset_at=graphql_reset_at,
        core_remaining=core_remaining,
        core_reset_at=core_reset_at,
    )


def graphql_budget_decision(
    snapshot: RateLimitSnapshot,
    *,
    expected_graphql_requests: int,
    min_graphql_remaining: int = DEFAULT_GRAPHQL_SAFETY_THRESHOLD,
    expected_core_requests: int = 0,
    min_core_remaining: int = DEFAULT_CORE_SAFETY_THRESHOLD,
) -> dict[str, Any]:
    """Decide whether a bounded operation may spend its estimated API budget.

    The decision is deliberately conservative: an unavailable or malformed quota response is
    blocked, and the safety margin remains after the estimated request budget. Callers may use the
    returned reset values to resume later; no retry loop is implied.
    """
    if expected_graphql_requests < 0 or expected_core_requests < 0:
        raise ValueError("expected request counts must be non-negative")
    if min_graphql_remaining < 0 or min_core_remaining < 0:
        raise ValueError("quota thresholds must be non-negative")

    decision: dict[str, Any] = {
        "status": "ok",
        "resource": "graphql",
        "graphql_remaining": snapshot.graphql_remaining,
        "graphql_reset_at": snapshot.graphql_reset_at,
        "core_remaining": snapshot.core_remaining,
        "core_reset_at": snapshot.core_reset_at,
        "expected_graphql_requests": expected_graphql_requests,
        "expected_core_requests": expected_core_requests,
        "min_graphql_remaining": min_graphql_remaining,
        "min_core_remaining": min_core_remaining,
        "writes_performed": False,
        "resume_after": None,
        "reason": "budget_available",
        "message": "estimated API budget fits below the configured safety margins",
    }
    if snapshot.status != "ok":
        decision.update(
            status="quota_blocked",
            reason=f"rate_limit_{snapshot.status}",
            message=snapshot.error or "rate_limit response unavailable",
            resume_after=snapshot.graphql_reset_at or snapshot.core_reset_at,
        )
        return decision

    assert snapshot.graphql_remaining is not None
    assert snapshot.core_remaining is not None
    graphql_after = snapshot.graphql_remaining - expected_graphql_requests
    core_after = snapshot.core_remaining - expected_core_requests
    decision["graphql_remaining_after_budget"] = graphql_after
    decision["core_remaining_after_budget"] = core_after
    if graphql_after < min_graphql_remaining:
        decision.update(
            status="quota_blocked",
            reason="graphql_budget_below_safety_threshold",
            message=(
                "estimated GraphQL requests would cross the configured safety threshold; "
                "resume after the GraphQL quota reset"
            ),
            resume_after=snapshot.graphql_reset_at,
        )
        return decision
    if core_after < min_core_remaining:
        decision.update(
            status="quota_blocked",
            resource="core",
            reason="core_budget_below_safety_threshold",
            message=(
                "estimated REST requests would cross the configured safety threshold; "
                "resume after the core quota reset"
            ),
            resume_after=snapshot.core_reset_at,
        )
    return decision
