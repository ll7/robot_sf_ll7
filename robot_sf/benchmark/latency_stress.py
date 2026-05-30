"""Latency-stress preflight contract helpers for learned-policy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_PLANNER_UPDATE_MODES = {"every-step", "hold-last"}
_DEFAULT_NON_SUCCESS_STATUSES = (
    "fallback",
    "degraded",
    "timeout",
    "not_available",
    "failed",
)


@dataclass(frozen=True)
class LatencyStressProfile:
    """Serializable preflight-only latency stress profile for benchmark diagnostics."""

    name: str
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    planner_update_mode: str = "every-step"
    planner_update_period_steps: int = 1
    inference_timeout_ms: float | None = None
    profile_version: str = "v0"
    claim_scope: str = "synthetic-only"
    non_success_statuses: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_NON_SUCCESS_STATUSES
    )

    def to_metadata(self, *, dt: float | None = None) -> dict[str, Any]:
        """Return a JSON-safe metadata payload.

        ``dt`` is optional so config/preflight paths can emit a stable contract before a
        runner-specific step size is known.
        """
        validate_latency_stress_profile(self)
        observation_delay_ms = None
        action_delay_ms = None
        planner_update_interval_ms = None
        if dt is not None and float(dt) > 0.0:
            step_ms = float(dt) * 1000.0
            observation_delay_ms = float(self.observation_delay_steps * step_ms)
            action_delay_ms = float(self.action_delay_steps * step_ms)
            planner_update_interval_ms = float(self.planner_update_period_steps * step_ms)
        return {
            "schema_version": "latency-stress-profile.v1",
            "name": self.name,
            "profile_version": self.profile_version,
            "claim_scope": self.claim_scope,
            "observation_delay_steps": int(self.observation_delay_steps),
            "observation_delay_ms": observation_delay_ms,
            "action_delay_steps": int(self.action_delay_steps),
            "action_delay_ms": action_delay_ms,
            "planner_update_mode": self.planner_update_mode,
            "planner_update_period_steps": int(self.planner_update_period_steps),
            "planner_update_interval_ms": planner_update_interval_ms,
            "inference_timeout_ms": (
                float(self.inference_timeout_ms) if self.inference_timeout_ms is not None else None
            ),
            "non_success_statuses": list(self.non_success_statuses),
            "contract_scope": "preflight-and-provenance-only",
        }


def known_planner_update_modes() -> tuple[str, ...]:
    """Return supported planner update modes."""
    return tuple(sorted(_PLANNER_UPDATE_MODES))


def load_latency_stress_profile(payload: Any) -> LatencyStressProfile | None:
    """Normalize optional latency-stress payloads into the typed profile contract.

    Returns:
        A validated profile, or ``None`` when the payload is absent.
    """
    if payload is None:
        return None
    if isinstance(payload, LatencyStressProfile):
        validate_latency_stress_profile(payload)
        return payload
    if not isinstance(payload, dict):
        raise TypeError("latency_stress_profile must be a mapping when provided")
    non_success_raw = payload.get("non_success_statuses", _DEFAULT_NON_SUCCESS_STATUSES)
    if isinstance(non_success_raw, (str, int, float)):
        non_success_statuses = (str(non_success_raw).strip(),)
    elif isinstance(non_success_raw, (list, tuple)):
        non_success_statuses = tuple(str(value).strip() for value in non_success_raw)
    else:
        raise TypeError("latency_stress_profile.non_success_statuses must be a list")
    profile = LatencyStressProfile(
        name=str(payload.get("name", "")).strip(),
        profile_version=str(payload.get("profile_version", "v0")).strip() or "v0",
        claim_scope=str(payload.get("claim_scope", "synthetic-only")).strip() or "synthetic-only",
        observation_delay_steps=int(payload.get("observation_delay_steps", 0)),
        action_delay_steps=int(payload.get("action_delay_steps", 0)),
        planner_update_mode=(
            str(payload.get("planner_update_mode", "every-step")).strip().lower() or "every-step"
        ),
        planner_update_period_steps=int(payload.get("planner_update_period_steps", 1)),
        inference_timeout_ms=(
            float(payload["inference_timeout_ms"])
            if payload.get("inference_timeout_ms") is not None
            else None
        ),
        non_success_statuses=non_success_statuses,
    )
    validate_latency_stress_profile(profile)
    return profile


def validate_latency_stress_profile(profile: LatencyStressProfile) -> None:  # noqa: C901
    """Validate that one latency-stress profile is usable as a diagnostic contract."""
    if not profile.name.strip():
        raise ValueError("latency_stress_profile.name must be non-empty")
    if not profile.profile_version.strip():
        raise ValueError("latency_stress_profile.profile_version must be non-empty")
    if profile.claim_scope.strip() != "synthetic-only":
        raise ValueError("latency_stress_profile.claim_scope must be 'synthetic-only'")
    if profile.observation_delay_steps < 0:
        raise ValueError("latency_stress_profile.observation_delay_steps must be >= 0")
    if profile.action_delay_steps < 0:
        raise ValueError("latency_stress_profile.action_delay_steps must be >= 0")
    if profile.planner_update_mode not in _PLANNER_UPDATE_MODES:
        known = ", ".join(known_planner_update_modes())
        raise ValueError(
            "Unsupported latency_stress_profile.planner_update_mode "
            f"'{profile.planner_update_mode}'. Expected one of: {known}"
        )
    if profile.planner_update_mode == "every-step" and profile.planner_update_period_steps != 1:
        raise ValueError(
            "latency_stress_profile.planner_update_period_steps must be 1 for every-step"
        )
    if profile.planner_update_mode == "hold-last" and profile.planner_update_period_steps < 2:
        raise ValueError(
            "latency_stress_profile.planner_update_period_steps must be >= 2 for hold-last"
        )
    if profile.inference_timeout_ms is not None and profile.inference_timeout_ms <= 0.0:
        raise ValueError("latency_stress_profile.inference_timeout_ms must be > 0 when set")
    normalized_statuses = tuple(
        str(value).strip() for value in profile.non_success_statuses if str(value).strip()
    )
    if not normalized_statuses:
        raise ValueError("latency_stress_profile.non_success_statuses must be non-empty")
    required = set(_DEFAULT_NON_SUCCESS_STATUSES)
    if not required.issubset(normalized_statuses):
        missing = ", ".join(sorted(required.difference(normalized_statuses)))
        raise ValueError(
            "latency_stress_profile.non_success_statuses must include fallback, degraded, "
            f"timeout, not_available, and failed; missing: {missing}"
        )


def not_available_latency_metrics() -> dict[str, str]:
    """Return explicit placeholders for latency metrics not measured by this preflight contract."""
    return {
        "observation_age_steps": "not_available",
        "observation_age_ms": "not_available",
        "held_action_ratio": "not_available",
        "planner_update_interval_steps": "not_available",
        "planner_update_interval_ms": "not_available",
        "inference_timeout_count": "not_available",
        "inference_fallback_count": "not_available",
        "synthetic_actuation_delay_steps": "not_available",
    }
