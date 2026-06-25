"""Planner-agnostic safety wrapper around the action interface (issue #3501).

The strongest available thesis result is causal: *the framework identifies a mitigation lever
and quantifies its effect*. This module provides that lever — a single, planner-agnostic safety
wrapper that post-processes any planner's commanded action through fixed, predeclared safety
stages:

1. **clearance / TTC monitor** — read the per-step safety context;
2. **speed cap near pedestrians** — clamp commanded speed within a caution radius;
3. **hard stop / yield veto** — zero commanded speed when time-to-collision or clearance is
   critical (turning is still permitted so the robot can yield).

The wrapper is **off by default** and opt-in per run, with **fixed, predeclared thresholds** (no
per-planner tuning), so a factorial ``planner × {wrapper off, wrapper on}`` ablation can quantify
its effect. This module is the pure per-step transform; the ablation campaign, live wiring into the
action adapters, and a stateful deadlock-recovery stage are deliberate follow-ups.

Thresholds are predeclared modeling choices, diagnostic until durable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAFETY_WRAPPER_SCHEMA = "safety_wrapper.v1"

# Intervention labels (stable vocabulary).
INTERVENTION_DISABLED = "disabled"
INTERVENTION_NONE = "none"
INTERVENTION_SPEED_CAP = "speed_cap"
INTERVENTION_HARD_STOP = "hard_stop"


@dataclass(frozen=True, slots=True)
class SafetyWrapperConfig:
    """Fixed, predeclared safety-wrapper thresholds (planner-agnostic).

    Attributes:
        enabled: Whether the wrapper is active. **Off by default** (opt-in per run).
        pedestrian_caution_radius_m: Within this distance, the speed cap applies.
        capped_speed_m_s: Defensive forward-speed ceiling near pedestrians.
        ttc_veto_threshold_s: Time-to-collision at/below which a hard stop is vetoed in.
        clearance_veto_m: Clearance at/below which a hard stop is vetoed in.
    """

    enabled: bool = False
    pedestrian_caution_radius_m: float = 2.0
    capped_speed_m_s: float = 0.5
    ttc_veto_threshold_s: float = 1.0
    clearance_veto_m: float = 0.3

    def __post_init__(self) -> None:
        """Validate that the thresholds are physically usable."""
        for name in (
            "pedestrian_caution_radius_m",
            "capped_speed_m_s",
            "ttc_veto_threshold_s",
            "clearance_veto_m",
        ):
            value = getattr(self, name)
            if not (value > 0.0):
                raise ValueError(f"SafetyWrapperConfig.{name} must be > 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class SafetyContext:
    """Per-step safety signals consumed by the wrapper.

    Attributes:
        min_pedestrian_distance_m: Distance to the nearest pedestrian (m).
        min_clearance_m: Minimum clearance to any obstacle (m).
        min_ttc_s: Minimum time-to-collision (s); ``None`` when undefined/closing-free.
    """

    min_pedestrian_distance_m: float
    min_clearance_m: float
    min_ttc_s: float | None = None


def apply_safety_wrapper(
    linear_velocity: float,
    angular_velocity: float,
    context: SafetyContext,
    config: SafetyWrapperConfig | None = None,
) -> dict[str, Any]:
    """Post-process a commanded action through the planner-agnostic safety stages.

    Stage precedence: hard stop/yield veto (TTC or clearance critical) overrides the speed cap,
    which overrides pass-through. Angular velocity is never reduced, so the robot can still turn
    to yield.

    Returns:
        dict[str, Any]: Versioned record with the corrected action and the intervention applied.
    """
    config = config or SafetyWrapperConfig()
    original = (float(linear_velocity), float(angular_velocity))

    if not config.enabled:
        return _record(config, context, original, original, INTERVENTION_DISABLED)

    ttc_critical = (
        context.min_ttc_s is not None and context.min_ttc_s <= config.ttc_veto_threshold_s
    )
    clearance_critical = context.min_clearance_m <= config.clearance_veto_m
    if ttc_critical or clearance_critical:
        corrected = (0.0, float(angular_velocity))
        return _record(config, context, original, corrected, INTERVENTION_HARD_STOP)

    near_pedestrian = context.min_pedestrian_distance_m <= config.pedestrian_caution_radius_m
    if near_pedestrian and float(linear_velocity) > config.capped_speed_m_s:
        corrected = (config.capped_speed_m_s, float(angular_velocity))
        return _record(config, context, original, corrected, INTERVENTION_SPEED_CAP)

    return _record(config, context, original, original, INTERVENTION_NONE)


def _record(
    config: SafetyWrapperConfig,
    context: SafetyContext,
    original: tuple[float, float],
    corrected: tuple[float, float],
    intervention: str,
) -> dict[str, Any]:
    """Build the versioned wrapper record.

    Returns:
        dict[str, Any]: The schema-tagged wrapper output record.
    """
    return {
        "schema_version": SAFETY_WRAPPER_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "enabled": config.enabled,
        "intervention": intervention,
        "intervened": intervention in {INTERVENTION_SPEED_CAP, INTERVENTION_HARD_STOP},
        "original_linear_velocity": original[0],
        "original_angular_velocity": original[1],
        "corrected_linear_velocity": corrected[0],
        "corrected_angular_velocity": corrected[1],
        "context": {
            "min_pedestrian_distance_m": context.min_pedestrian_distance_m,
            "min_clearance_m": context.min_clearance_m,
            "min_ttc_s": context.min_ttc_s,
        },
    }


__all__ = [
    "INTERVENTION_DISABLED",
    "INTERVENTION_HARD_STOP",
    "INTERVENTION_NONE",
    "INTERVENTION_SPEED_CAP",
    "SAFETY_WRAPPER_SCHEMA",
    "SafetyContext",
    "SafetyWrapperConfig",
    "apply_safety_wrapper",
]
