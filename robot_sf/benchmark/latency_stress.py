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


def _optional_text(payload: dict[str, Any], key: str, *, default: str) -> str:
    """Return a normalized optional string field.

    ``None`` means absent for optional fields. Non-string values fail before they can be silently
    coerced into misleading contract metadata.
    """
    raw = payload.get(key, default)
    if raw is None:
        raw = default
    if not isinstance(raw, str):
        raise TypeError(f"latency_stress_profile.{key} must be a string")
    return raw.strip()


def _optional_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    """Return a normalized optional integer field."""
    raw = payload.get(key, default)
    if raw is None:
        raw = default
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"latency_stress_profile.{key} must be an integer")
    return int(raw)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    """Return a normalized optional float field."""
    raw = payload.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"latency_stress_profile.{key} must be a number")
    return float(raw)


def _normalize_non_success_statuses(raw: Any) -> tuple[str, ...]:
    """Normalize non-success status labels without coercing invalid entries.

    Returns:
        Normalized status labels.
    """
    if raw is None:
        return _DEFAULT_NON_SUCCESS_STATUSES
    if isinstance(raw, str):
        return (raw.strip(),)
    if isinstance(raw, (list, tuple)):
        statuses: list[str] = []
        for value in raw:
            if not isinstance(value, str):
                raise TypeError(
                    "latency_stress_profile.non_success_statuses entries must be strings"
                )
            statuses.append(value.strip())
        return tuple(statuses)
    raise TypeError("latency_stress_profile.non_success_statuses must be a list")


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
    profile = LatencyStressProfile(
        name=_optional_text(payload, "name", default=""),
        profile_version=_optional_text(payload, "profile_version", default="v0") or "v0",
        claim_scope=_optional_text(payload, "claim_scope", default="synthetic-only")
        or "synthetic-only",
        observation_delay_steps=_optional_int(payload, "observation_delay_steps", default=0),
        action_delay_steps=_optional_int(payload, "action_delay_steps", default=0),
        planner_update_mode=(
            _optional_text(payload, "planner_update_mode", default="every-step").lower()
            or "every-step"
        ),
        planner_update_period_steps=_optional_int(
            payload,
            "planner_update_period_steps",
            default=1,
        ),
        inference_timeout_ms=_optional_float(payload, "inference_timeout_ms"),
        non_success_statuses=_normalize_non_success_statuses(
            payload.get("non_success_statuses", _DEFAULT_NON_SUCCESS_STATUSES)
        ),
    )
    validate_latency_stress_profile(profile)
    return profile


def validate_latency_stress_profile(profile: LatencyStressProfile) -> None:  # noqa: C901, PLR0912
    """Validate that one latency-stress profile is usable as a diagnostic contract."""
    if not isinstance(profile.name, str):
        raise TypeError("latency_stress_profile.name must be a string")
    if not profile.name.strip():
        raise ValueError("latency_stress_profile.name must be non-empty")
    if not isinstance(profile.profile_version, str):
        raise TypeError("latency_stress_profile.profile_version must be a string")
    if not profile.profile_version.strip():
        raise ValueError("latency_stress_profile.profile_version must be non-empty")
    if not isinstance(profile.claim_scope, str):
        raise TypeError("latency_stress_profile.claim_scope must be a string")
    if profile.claim_scope.strip() != "synthetic-only":
        raise ValueError("latency_stress_profile.claim_scope must be 'synthetic-only'")
    if isinstance(profile.observation_delay_steps, bool) or not isinstance(
        profile.observation_delay_steps,
        int,
    ):
        raise TypeError("latency_stress_profile.observation_delay_steps must be an integer")
    if profile.observation_delay_steps < 0:
        raise ValueError("latency_stress_profile.observation_delay_steps must be >= 0")
    if isinstance(profile.action_delay_steps, bool) or not isinstance(
        profile.action_delay_steps,
        int,
    ):
        raise TypeError("latency_stress_profile.action_delay_steps must be an integer")
    if profile.action_delay_steps < 0:
        raise ValueError("latency_stress_profile.action_delay_steps must be >= 0")
    if not isinstance(profile.planner_update_mode, str):
        raise TypeError("latency_stress_profile.planner_update_mode must be a string")
    if profile.planner_update_mode not in _PLANNER_UPDATE_MODES:
        known = ", ".join(known_planner_update_modes())
        raise ValueError(
            "Unsupported latency_stress_profile.planner_update_mode "
            f"'{profile.planner_update_mode}'. Expected one of: {known}"
        )
    if isinstance(profile.planner_update_period_steps, bool) or not isinstance(
        profile.planner_update_period_steps,
        int,
    ):
        raise TypeError("latency_stress_profile.planner_update_period_steps must be an integer")
    if profile.planner_update_mode == "every-step" and profile.planner_update_period_steps != 1:
        raise ValueError(
            "latency_stress_profile.planner_update_period_steps must be 1 for every-step"
        )
    if profile.planner_update_mode == "hold-last" and profile.planner_update_period_steps < 2:
        raise ValueError(
            "latency_stress_profile.planner_update_period_steps must be >= 2 for hold-last"
        )
    if (
        profile.inference_timeout_ms is not None
        and (isinstance(profile.inference_timeout_ms, bool))
    ) or (
        profile.inference_timeout_ms is not None
        and not isinstance(profile.inference_timeout_ms, int | float)
    ):
        raise TypeError("latency_stress_profile.inference_timeout_ms must be a number")
    if profile.inference_timeout_ms is not None and profile.inference_timeout_ms <= 0.0:
        raise ValueError("latency_stress_profile.inference_timeout_ms must be > 0 when set")
    if isinstance(profile.non_success_statuses, str) or not isinstance(
        profile.non_success_statuses,
        tuple,
    ):
        raise TypeError("latency_stress_profile.non_success_statuses must be a tuple")
    if any(not isinstance(value, str) for value in profile.non_success_statuses):
        raise TypeError("latency_stress_profile.non_success_statuses entries must be strings")
    normalized_statuses = tuple(
        value.strip() for value in profile.non_success_statuses if value.strip()
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
