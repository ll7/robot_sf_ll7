"""Profile payload normalization helpers for map-based benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from robot_sf.benchmark.latency_stress import LatencyStressProfile, load_latency_stress_profile
from robot_sf.benchmark.synthetic_actuation import (
    SyntheticActuationProfile,
    validate_actuation_profile_claim_boundary,
    validate_synthetic_actuation_profile,
)


def _optional_profile_mapping(payload: dict[str, Any], key: str) -> Mapping[str, Any] | None:
    """Return an optional nested profile metadata mapping, rejecting malformed values."""
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"synthetic_actuation_profile.{key} must be a mapping when provided")
    return value


def load_synthetic_actuation_profile(payload: Any) -> SyntheticActuationProfile | None:
    """Normalize optional synthetic-actuation payloads into the typed profile contract.

    Returns:
        A validated profile, or ``None`` when the payload is absent.
    """
    if payload is None:
        return None
    if isinstance(payload, SyntheticActuationProfile):
        validate_synthetic_actuation_profile(payload)
        return payload
    if not isinstance(payload, dict):
        raise TypeError("synthetic_actuation_profile must be a mapping when provided")
    validate_actuation_profile_claim_boundary(payload)
    claim_scope = str(payload.get("claim_scope", "synthetic-only")).strip() or "synthetic-only"
    if claim_scope != "synthetic-only":
        raise ValueError("synthetic_actuation_profile.claim_scope must be 'synthetic-only'")
    profile = SyntheticActuationProfile(
        name=str(payload.get("name", "")),
        profile_version=str(payload.get("profile_version", "v0")),
        claim_scope=claim_scope,
        claim_boundary=str(payload.get("claim_boundary", "")),
        max_linear_accel_m_s2=float(payload.get("max_linear_accel_m_s2")),
        max_linear_decel_m_s2=float(payload.get("max_linear_decel_m_s2")),
        max_yaw_rate_rad_s=float(payload.get("max_yaw_rate_rad_s")),
        max_angular_accel_rad_s2=float(payload.get("max_angular_accel_rad_s2")),
        latency_mode=str(payload.get("latency_mode", "")),
        update_mode=str(payload.get("update_mode", "")),
        variability_distribution=_optional_profile_mapping(payload, "variability_distribution"),
        variability_sample=_optional_profile_mapping(payload, "variability_sample"),
    )
    validate_synthetic_actuation_profile(profile)
    return profile


def load_latency_profile(payload: Any) -> LatencyStressProfile | None:
    """Normalize optional latency-stress payloads into the typed profile contract.

    Returns:
        A validated profile, or ``None`` when the payload is absent.
    """
    return load_latency_stress_profile(payload)
