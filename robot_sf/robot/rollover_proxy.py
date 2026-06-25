"""Internal-proxy lateral-stability (rollover) margin for narrow-track platforms.

Planar bicycle / differential-drive / holonomic models mask the *dynamic rollover*
risk of an asymmetric (triangular-support) narrow-track three-wheeled platform: a
maneuver certified "safe" under kinematics may demand a yaw rate that tips such a
platform. This module provides the closed-form proxy from issue #3479 so a
stability-margin signal and a ``ROLLOVER_CRITICAL`` classifier can be surfaced as
telemetry.

.. warning::

   **Internal proxy only — governance gate #2416 / #2417.** The geometry parameters
   here are explicit, documented, *non-hardware* proxy assumptions. They are NOT a
   hardware-calibrated AMV profile and must NOT be read as validated tip-over limits
   or used for paper-facing AMV safety claims. Those remain blocked until real-source
   provenance (#1585 / #2000) is accepted.

Model (issue #3479):

- lateral acceleration ``a_y ≈ v · ω``
- critical lateral acceleration ``a_y,crit = g · (t_w / (2 · h_c)) · (a / L)``
- stability margin ``= clamp(1 − |a_y| / a_y,crit, 0, 1)`` (1 = fully stable,
  0 = at/over the proxy tip-over threshold)

This module is pure and side-effect free; it does not alter planner, training, or
benchmark behavior. Wiring a terminal flag and reward penalty into the stepping
loop is intentionally a separate, opt-in follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GRAVITY_M_S2 = 9.81
PROXY_SCHEMA_VERSION = "rollover_proxy.v1"


@dataclass(frozen=True, slots=True)
class RolloverProxyParams:
    """Versioned, documented **non-hardware** geometry for the rollover proxy.

    Every value is an explicit internal-proxy assumption, not a measured hardware
    parameter (governance gate #2416 / #2417). Defaults describe a small, narrow-track
    three-wheeled platform purely so the proxy is exercisable; they carry no hardware
    authority.

    The default geometry is **aligned with the benchmark-surface source of truth**
    ``robot_sf.benchmark.metrics.evaluate_stability_margin`` (the reviewer-supplied TWV proxy:
    ``t_w=0.8``, ``L=1.2``, ``h_c=0.6``, ``a=0.5``) so the runtime diagnostic and the benchmark
    column ``rollover_min_stability_margin`` cannot diverge (issue #3587). The closed form here is
    identical to that function; ``test_rollover_proxy`` cross-checks numerical agreement.

    Attributes:
        track_width_m: Lateral wheel track ``t_w`` (m).
        cog_height_m: Centre-of-gravity height ``h_c`` (m).
        front_axle_to_cog_m: Longitudinal distance from front axle to CoG ``a`` (m).
        wheelbase_m: Wheelbase ``L`` (m).
        gravity_m_s2: Gravitational acceleration ``g`` (m/s^2).
        schema_version: Stable schema tag for reproducibility.
    """

    track_width_m: float = 0.80
    cog_height_m: float = 0.60
    front_axle_to_cog_m: float = 0.50
    wheelbase_m: float = 1.20
    gravity_m_s2: float = GRAVITY_M_S2
    schema_version: str = PROXY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate that the proxy geometry is physically usable."""
        positive = {
            "track_width_m": self.track_width_m,
            "cog_height_m": self.cog_height_m,
            "front_axle_to_cog_m": self.front_axle_to_cog_m,
            "wheelbase_m": self.wheelbase_m,
            "gravity_m_s2": self.gravity_m_s2,
        }
        for name, value in positive.items():
            if not (value > 0.0):
                raise ValueError(f"RolloverProxyParams.{name} must be > 0, got {value!r}")
        if self.front_axle_to_cog_m > self.wheelbase_m:
            raise ValueError(
                "front_axle_to_cog_m must not exceed wheelbase_m "
                f"({self.front_axle_to_cog_m} > {self.wheelbase_m})"
            )


def lateral_acceleration(linear_velocity: float, yaw_rate: float) -> float:
    """Return the proxy lateral acceleration ``a_y ≈ v · ω`` (m/s^2)."""
    return float(linear_velocity) * float(yaw_rate)


def critical_lateral_acceleration(params: RolloverProxyParams) -> float:
    """Return the proxy critical lateral acceleration ``a_y,crit`` (m/s^2)."""
    return (
        params.gravity_m_s2
        * (params.track_width_m / (2.0 * params.cog_height_m))
        * (params.front_axle_to_cog_m / params.wheelbase_m)
    )


def stability_margin(
    linear_velocity: float,
    yaw_rate: float,
    params: RolloverProxyParams | None = None,
) -> float:
    """Return the rollover stability margin in ``[0, 1]``.

    ``1`` means fully within the proxy tip-over threshold; ``0`` means the demanded
    lateral acceleration meets or exceeds the critical value.

    Returns:
        float: ``clamp(1 − |a_y| / a_y,crit, 0, 1)``.
    """
    params = params or RolloverProxyParams()
    a_y = abs(lateral_acceleration(linear_velocity, yaw_rate))
    a_y_crit = critical_lateral_acceleration(params)
    margin = 1.0 - (a_y / a_y_crit)
    return max(0.0, min(1.0, margin))


def is_rollover_critical(margin: float) -> bool:
    """Return whether a stability margin indicates a ``ROLLOVER_CRITICAL`` condition."""
    return margin <= 0.0


def rollover_proxy_telemetry(
    linear_velocity: float,
    yaw_rate: float,
    params: RolloverProxyParams | None = None,
) -> dict[str, Any]:
    """Return a compact, schema-tagged telemetry record for one step.

    This is diagnostic only: it reports the proxy signals without altering behavior.

    Returns:
        dict[str, Any]: Margin, lateral accelerations, critical flag, and provenance.
    """
    params = params or RolloverProxyParams()
    a_y = lateral_acceleration(linear_velocity, yaw_rate)
    a_y_crit = critical_lateral_acceleration(params)
    margin = stability_margin(linear_velocity, yaw_rate, params)
    return {
        "schema_version": params.schema_version,
        "proxy_kind": "internal_non_hardware",
        "linear_velocity": float(linear_velocity),
        "yaw_rate": float(yaw_rate),
        "lateral_acceleration": a_y,
        "critical_lateral_acceleration": a_y_crit,
        "stability_margin": margin,
        "rollover_critical": is_rollover_critical(margin),
    }


__all__ = [
    "GRAVITY_M_S2",
    "PROXY_SCHEMA_VERSION",
    "RolloverProxyParams",
    "critical_lateral_acceleration",
    "is_rollover_critical",
    "lateral_acceleration",
    "rollover_proxy_telemetry",
    "stability_margin",
]
