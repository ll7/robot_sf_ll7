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
its effect. :func:`apply_safety_wrapper` is the pure per-step transform.

The fourth predeclared stage, **deadlock recovery**, is stateful (it needs a run of frozen steps
to detect a stall), so it lives in :class:`DeadlockRecoveryMonitor` — an opt-in, disabled-by-default
monitor that composes *around* the per-step transform. It only ever overrides angular velocity to
rotate free of a freeze; it never adds forward speed, so it cannot override the hard stop/yield veto
or worsen a collision. The monitor is wired into the benchmark runtime step loop via
:func:`robot_sf.benchmark.safety_wrapper_runtime.make_deadlock_recovery_monitor` (opt-in on the
wrapper_on arm); running the paired ablation campaign remains a deliberate follow-up.

Thresholds are predeclared modeling choices, diagnostic until durable evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SAFETY_WRAPPER_SCHEMA = "safety_wrapper.v1"
DEADLOCK_RECOVERY_SCHEMA = "safety_wrapper_deadlock_recovery.v1"

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
    # Cap on speed magnitude so a reversing robot (negative linear velocity, supported by
    # both drive models via allow_backwards) is also limited near pedestrians; the cap keeps
    # the original direction of travel.
    if near_pedestrian and abs(float(linear_velocity)) > config.capped_speed_m_s:
        corrected = (
            math.copysign(config.capped_speed_m_s, linear_velocity),
            float(angular_velocity),
        )
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


@dataclass(frozen=True, slots=True)
class DeadlockRecoveryConfig:
    """Fixed, predeclared deadlock-recovery thresholds (planner-agnostic).

    Deadlock recovery is the stateful fourth safety stage: it breaks the *frozen robot* failure
    mode (the planner keeps commanding near-zero forward speed while a nearby pedestrian or
    obstacle blocks progress, so the episode stalls without a collision) by permitting a bounded
    in-place rotation to search for a new heading. It is a modeling choice, predeclared and fixed
    (no per-planner tuning), and diagnostic until durable evidence.

    Attributes:
        enabled: Whether the monitor is active. **Off by default** (opt-in per run).
        patience_steps: Consecutive frozen steps required before recovery engages.
        recovery_steps: Maximum steps a single recovery maneuver persists before re-evaluating.
        recovery_angular_velocity_rad_s: In-place rotation magnitude applied during recovery.
        recovery_turn_sign: Fixed rotation direction (+1 or -1); predeclared, not tuned per run.
        frozen_speed_eps_m_s: Executed forward-speed magnitude at/below which a step counts frozen.
        hazard_proximity_m: A frozen step counts as a deadlock only when the nearest pedestrian is
            within this distance (so a legitimate goal-reached stop is not treated as a deadlock).
        hazard_clearance_m: Alternatively, a finite predicted clearance at/below this also marks the
            step as hazard-blocked (obstacle-driven freeze without a nearby pedestrian).
    """

    enabled: bool = False
    patience_steps: int = 20
    recovery_steps: int = 10
    recovery_angular_velocity_rad_s: float = 0.5
    recovery_turn_sign: int = 1
    frozen_speed_eps_m_s: float = 0.05
    hazard_proximity_m: float = 2.0
    hazard_clearance_m: float = 0.5

    def __post_init__(self) -> None:
        """Validate that the recovery thresholds are physically usable."""
        if self.patience_steps < 1:
            raise ValueError(
                f"DeadlockRecoveryConfig.patience_steps must be >= 1, got {self.patience_steps!r}"
            )
        if self.recovery_steps < 1:
            raise ValueError(
                f"DeadlockRecoveryConfig.recovery_steps must be >= 1, got {self.recovery_steps!r}"
            )
        if self.recovery_turn_sign not in (-1, 1):
            raise ValueError(
                "DeadlockRecoveryConfig.recovery_turn_sign must be -1 or 1, "
                f"got {self.recovery_turn_sign!r}"
            )
        if not (self.recovery_angular_velocity_rad_s > 0.0):
            raise ValueError(
                "DeadlockRecoveryConfig.recovery_angular_velocity_rad_s must be > 0, "
                f"got {self.recovery_angular_velocity_rad_s!r}"
            )
        for name in ("frozen_speed_eps_m_s", "hazard_proximity_m", "hazard_clearance_m"):
            value = getattr(self, name)
            if not (value >= 0.0):
                raise ValueError(f"DeadlockRecoveryConfig.{name} must be >= 0, got {value!r}")


def _is_hazard_blocked(context: SafetyContext, config: DeadlockRecoveryConfig) -> bool:
    """Return whether the step is blocked by a nearby pedestrian or obstacle.

    Used to distinguish a genuine deadlock (frozen while blocked) from a legitimate stop with
    clear surroundings (e.g. goal reached), which must not trigger recovery.
    """
    near_pedestrian = context.min_pedestrian_distance_m <= config.hazard_proximity_m
    near_obstacle = (
        math.isfinite(context.min_clearance_m)
        and context.min_clearance_m <= config.hazard_clearance_m
    )
    return bool(near_pedestrian or near_obstacle)


class DeadlockRecoveryMonitor:
    """Stateful, opt-in deadlock-recovery stage composed around the per-step transform.

    Call :meth:`step` once per simulation step with the *executed* (post-:func:`apply_safety_wrapper`)
    command and the same :class:`SafetyContext`. The monitor tracks a run of frozen steps and, once
    ``patience_steps`` is reached, overrides the angular velocity with a bounded in-place rotation
    for up to ``recovery_steps`` steps to break the stall. **Forward speed is passed through
    unchanged**, so any upstream hard stop/yield veto (which zeroes forward speed) is preserved and
    recovery can never inject forward motion into a hazard.

    The monitor is deterministic and holds only integer counters, so a fresh instance per episode
    yields reproducible behavior.
    """

    def __init__(self, config: DeadlockRecoveryConfig | None = None) -> None:
        """Initialize a monitor with predeclared thresholds and a cleared counter state."""
        self._config = config or DeadlockRecoveryConfig()
        self.reset()

    @property
    def config(self) -> DeadlockRecoveryConfig:
        """Return the predeclared recovery configuration."""
        return self._config

    def reset(self) -> None:
        """Clear the frozen-run and recovery counters (call once per episode)."""
        self._frozen_run = 0
        self._recovery_remaining = 0

    def step(
        self,
        corrected_linear_velocity: float,
        corrected_angular_velocity: float,
        context: SafetyContext,
    ) -> dict[str, Any]:
        """Advance the monitor one step and return the (possibly recovery-adjusted) command record.

        Args:
            corrected_linear_velocity: Forward speed already produced by the per-step transform.
            corrected_angular_velocity: Angular velocity already produced by the transform.
            context: The per-step safety signals (same context passed to the transform).

        Returns:
            dict[str, Any]: Versioned record with the deadlock state and the final command. Forward
            speed always equals ``corrected_linear_velocity``; angular velocity is overridden only
            while a recovery maneuver is active.
        """
        config = self._config
        original = (float(corrected_linear_velocity), float(corrected_angular_velocity))

        if not config.enabled:
            return self._record(config, original, original, frozen=False, recovery_applied=False)

        frozen = abs(
            float(corrected_linear_velocity)
        ) <= config.frozen_speed_eps_m_s and _is_hazard_blocked(context, config)
        if frozen:
            self._frozen_run += 1
        else:
            # Progress resumed (or surroundings cleared): the stall is over.
            self._frozen_run = 0
            self._recovery_remaining = 0

        # Engage a new maneuver once the frozen run crosses patience and none is in flight.
        if self._recovery_remaining <= 0 and self._frozen_run >= config.patience_steps:
            self._recovery_remaining = config.recovery_steps

        recovery_applied = frozen and self._recovery_remaining > 0
        if recovery_applied:
            self._recovery_remaining -= 1
            final = (
                float(corrected_linear_velocity),  # forward speed preserved; never increased
                config.recovery_turn_sign * config.recovery_angular_velocity_rad_s,
            )
        else:
            final = original
        record = self._record(
            config, original, final, frozen=frozen, recovery_applied=recovery_applied
        )
        # A maneuver that just used its last step re-arms the patience window: reset the frozen run
        # so a still-stuck robot pauses for one patience cycle (giving the planner a fresh chance)
        # before rotating again, rather than overriding angular velocity on every frozen step.
        if recovery_applied and self._recovery_remaining == 0:
            self._frozen_run = 0
        return record

    def _record(
        self,
        config: DeadlockRecoveryConfig,
        original: tuple[float, float],
        final: tuple[float, float],
        *,
        frozen: bool,
        recovery_applied: bool,
    ) -> dict[str, Any]:
        """Build the versioned deadlock-recovery record.

        Returns:
            dict[str, Any]: The schema-tagged deadlock-recovery output record.
        """
        return {
            "schema_version": DEADLOCK_RECOVERY_SCHEMA,
            "evidence_kind": "diagnostic_proxy",
            "enabled": config.enabled,
            "frozen": bool(frozen),
            "frozen_run": int(self._frozen_run),
            "deadlock_detected": bool(config.enabled and self._frozen_run >= config.patience_steps),
            "recovery_active": bool(recovery_applied),
            "recovery_steps_remaining": int(self._recovery_remaining),
            "input_linear_velocity": original[0],
            "input_angular_velocity": original[1],
            "final_linear_velocity": final[0],
            "final_angular_velocity": final[1],
        }


__all__ = [
    "DEADLOCK_RECOVERY_SCHEMA",
    "INTERVENTION_DISABLED",
    "INTERVENTION_HARD_STOP",
    "INTERVENTION_NONE",
    "INTERVENTION_SPEED_CAP",
    "SAFETY_WRAPPER_SCHEMA",
    "DeadlockRecoveryConfig",
    "DeadlockRecoveryMonitor",
    "SafetyContext",
    "SafetyWrapperConfig",
    "apply_safety_wrapper",
]
