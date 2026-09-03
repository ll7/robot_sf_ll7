"""Small validation and canonicalization helpers for prediction contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import rfc8785

FORBIDDEN_EVIDENCE_SOURCE_NAMES = frozenset(
    {
        "scenario_assigned_route",
        "assigned_route",
        "true_goal",
        "goal_truth",
        "waypoint_truth",
        "future_trajectory",
        "simulator_goal",
        "simulator_route",
    }
)
FORBIDDEN_EVIDENCE_SOURCE_TOKENS = frozenset(
    {
        "oracle",
        "simulator",
        "true_goal",
        "route_truth",
        "waypoint_truth",
        "force_component",
    }
)


def is_forbidden_evidence_source(value: str) -> bool:
    """Return whether one source label is reserved for privileged evidence."""
    normalized = value.strip().lower()
    return normalized in FORBIDDEN_EVIDENCE_SOURCE_NAMES or any(
        token in normalized for token in FORBIDDEN_EVIDENCE_SOURCE_TOKENS
    )


def require_text(value: Any, field_name: str) -> str:
    """Return non-empty text, rejecting values that would be ambiguous in JSON."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def require_finite(value: Any, field_name: str) -> float:
    """Return a finite real as ``float`` while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def require_non_negative(value: Any, field_name: str) -> float:
    """Return a finite non-negative real."""
    numeric = require_finite(value, field_name)
    if numeric < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return numeric


def require_probability(value: Any, field_name: str) -> float:
    """Return a finite probability in the closed unit interval."""
    numeric = require_finite(value, field_name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return numeric


def require_step_index(value: Any, field_name: str) -> int:
    """Return a non-negative Python integer step index."""
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def require_xy(value: Any, field_name: str) -> tuple[float, float]:
    """Return exactly two finite coordinates."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    return (
        require_finite(value[0], f"{field_name}[0]"),
        require_finite(value[1], f"{field_name}[1]"),
    )


def require_covariance(
    value: Any, field_name: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a finite, symmetric, positive-definite two-dimensional covariance."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(
            isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 2
            for row in value
        )
    ):
        raise ValueError(f"{field_name} must be a 2x2 matrix")
    a = require_finite(value[0][0], f"{field_name}[0][0]")
    b = require_finite(value[0][1], f"{field_name}[0][1]")
    c = require_finite(value[1][0], f"{field_name}[1][0]")
    d = require_finite(value[1][1], f"{field_name}[1][1]")
    if not math.isclose(b, c, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field_name} must be symmetric")
    if a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
        raise ValueError(f"{field_name} must be positive-definite")
    return ((a, b), (c, d))


def require_digest(value: Any, field_name: str) -> str:
    """Return a lowercase SHA-256 digest."""
    text = require_text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    """Reject unknown external-record keys so version drift fails closed."""
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unexpected key(s): {', '.join(unknown)}")


def canonical_json(value: Any) -> str:
    """Serialize an I-JSON value with RFC 8785 canonical bytes.

    Returns:
        Canonical JSON text without insignificant whitespace.
    """
    try:
        return rfc8785.dumps(value).decode("utf-8")
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise ValueError(f"value cannot be canonicalized with RFC 8785: {exc}") from exc


def stable_digest(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON bytes."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a JSON-safe configuration mapping for contract provenance.

    Returns:
        Lowercase SHA-256 digest of the canonical configuration bytes.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return stable_digest(dict(config))


__all__ = [
    "FORBIDDEN_EVIDENCE_SOURCE_NAMES",
    "FORBIDDEN_EVIDENCE_SOURCE_TOKENS",
    "canonical_json",
    "is_forbidden_evidence_source",
    "reject_unknown_keys",
    "require_covariance",
    "require_digest",
    "require_finite",
    "require_non_negative",
    "require_probability",
    "require_step_index",
    "require_text",
    "require_xy",
    "stable_config_hash",
    "stable_digest",
]
