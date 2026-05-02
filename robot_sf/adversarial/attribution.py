"""Failure-attribution helpers for adversarial search bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureAttribution:
    """Small, replay-bundle-friendly failure attribution payload."""

    status: str
    primary_failure: str | None
    reasons: list[str]
    details: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable attribution payload."""
        return {
            "status": self.status,
            "primary_failure": self.primary_failure,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


def attribution_from_episode_record(record: dict[str, Any]) -> FailureAttribution:
    """Build a conservative attribution summary from a benchmark episode record."""
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    termination = str(record.get("termination_reason", record.get("status", "unknown")))
    reasons: list[str] = []
    primary: str | None = None
    collision = bool(outcome.get("collision")) or bool(outcome.get("collision_event"))
    timeout = bool(outcome.get("timeout")) or bool(outcome.get("timeout_event"))
    if collision:
        primary = "collision"
        reasons.append("episode outcome reports a collision")
    elif timeout:
        primary = "timeout"
        reasons.append("episode timed out before route completion")
    elif not bool(outcome.get("route_complete")):
        primary = "incomplete"
        reasons.append("episode did not report route completion")
    else:
        primary = "success"
        reasons.append("episode completed without an attributed failure")
    return FailureAttribution(
        status="attributed",
        primary_failure=primary,
        reasons=reasons,
        details={
            "termination_reason": termination,
            "status": record.get("status"),
            "outcome": outcome,
        },
    )


def attribution_from_error(error: str) -> FailureAttribution:
    """Build an attribution payload for an evaluation error."""
    return FailureAttribution(
        status="evaluation_failed",
        primary_failure="evaluation_error",
        reasons=[error],
        details={"error": error},
    )
