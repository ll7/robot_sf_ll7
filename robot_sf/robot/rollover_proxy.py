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

import warnings
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

    The geometry names are layout-neutral: ``two_wheel_axle_track_m`` is the track
    ``t_w`` of the axle with two wheels, and ``single_wheel_axle_to_cog_m`` is the
    distance ``a`` from the axle with one wheel to the centre of gravity.
    The single wheel may be at the front or at the rear. The deprecated aliases ``track_width_m``
    and ``front_axle_to_cog_m`` remain accepted for one release and emit a
    ``DeprecationWarning``.

    The default geometry is **aligned with the benchmark-surface source of truth**
    ``robot_sf.benchmark.metrics.evaluate_stability_margin`` (the reviewer-supplied TWV proxy:
    ``t_w=0.8``, ``L=1.2``, ``h_c=0.6``, ``a=0.5``) so the runtime diagnostic and the benchmark
    column ``rollover_min_stability_margin`` cannot diverge (issue #3587). The closed form here is
    identical to that function; ``test_rollover_proxy`` cross-checks numerical agreement.

    Attributes:
        two_wheel_axle_track_m: Track of the axle with two wheels ``t_w`` (m).
        cog_height_m: Centre-of-gravity height ``h_c`` (m).
        single_wheel_axle_to_cog_m: Distance from the axle with one wheel to CoG ``a`` (m).
        wheelbase_m: Wheelbase ``L`` (m).
        gravity_m_s2: Gravitational acceleration ``g`` (m/s^2).
        schema_version: Stable schema tag for reproducibility.
    """

    two_wheel_axle_track_m: float = 0.80
    cog_height_m: float = 0.60
    single_wheel_axle_to_cog_m: float = 0.50
    wheelbase_m: float = 1.20
    gravity_m_s2: float = GRAVITY_M_S2
    schema_version: str = PROXY_SCHEMA_VERSION

    def __init__(
        self,
        two_wheel_axle_track_m: float = 0.80,
        cog_height_m: float = 0.60,
        single_wheel_axle_to_cog_m: float = 0.50,
        wheelbase_m: float = 1.20,
        gravity_m_s2: float = GRAVITY_M_S2,
        schema_version: str = PROXY_SCHEMA_VERSION,
        **kwargs: Any,
    ) -> None:
        """Create proxy geometry, accepting deprecated aliases for one release."""
        if "track_width_m" in kwargs:
            warnings.warn(
                "track_width_m is deprecated; use two_wheel_axle_track_m instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            two_wheel_axle_track_m = kwargs.pop("track_width_m")
        if "front_axle_to_cog_m" in kwargs:
            warnings.warn(
                "front_axle_to_cog_m is deprecated; use single_wheel_axle_to_cog_m instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            single_wheel_axle_to_cog_m = kwargs.pop("front_axle_to_cog_m")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"RolloverProxyParams.__init__() got unexpected keyword(s): {unexpected}"
            )
        object.__setattr__(self, "two_wheel_axle_track_m", two_wheel_axle_track_m)
        object.__setattr__(self, "cog_height_m", cog_height_m)
        object.__setattr__(self, "single_wheel_axle_to_cog_m", single_wheel_axle_to_cog_m)
        object.__setattr__(self, "wheelbase_m", wheelbase_m)
        object.__setattr__(self, "gravity_m_s2", gravity_m_s2)
        object.__setattr__(self, "schema_version", schema_version)
        self.__post_init__()

    @property
    def track_width_m(self) -> float:
        """Deprecated alias for ``two_wheel_axle_track_m`` (kept for one release)."""
        warnings.warn(
            "track_width_m is deprecated; use two_wheel_axle_track_m instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.two_wheel_axle_track_m

    @property
    def front_axle_to_cog_m(self) -> float:
        """Deprecated alias for ``single_wheel_axle_to_cog_m`` (kept for one release)."""
        warnings.warn(
            "front_axle_to_cog_m is deprecated; use single_wheel_axle_to_cog_m instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.single_wheel_axle_to_cog_m

    def __post_init__(self) -> None:
        """Validate that the proxy geometry is physically usable."""
        positive = {
            "two_wheel_axle_track_m": self.two_wheel_axle_track_m,
            "cog_height_m": self.cog_height_m,
            "single_wheel_axle_to_cog_m": self.single_wheel_axle_to_cog_m,
            "wheelbase_m": self.wheelbase_m,
            "gravity_m_s2": self.gravity_m_s2,
        }
        for name, value in positive.items():
            if not (value > 0.0):
                raise ValueError(f"RolloverProxyParams.{name} must be > 0, got {value!r}")
        if self.single_wheel_axle_to_cog_m > self.wheelbase_m:
            raise ValueError(
                "single_wheel_axle_to_cog_m must not exceed wheelbase_m "
                f"({self.single_wheel_axle_to_cog_m} > {self.wheelbase_m})"
            )


def lateral_acceleration(linear_velocity: float, yaw_rate: float) -> float:
    """Return the proxy lateral acceleration ``a_y ≈ v · ω`` (m/s^2)."""
    return float(linear_velocity) * float(yaw_rate)


def critical_lateral_acceleration(params: RolloverProxyParams) -> float:
    """Return the proxy critical lateral acceleration ``a_y,crit`` (m/s^2)."""
    return (
        params.gravity_m_s2
        * (params.two_wheel_axle_track_m / (2.0 * params.cog_height_m))
        * (params.single_wheel_axle_to_cog_m / params.wheelbase_m)
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
