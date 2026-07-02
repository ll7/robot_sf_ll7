"""Horizon-dependent pedestrian uncertainty envelopes.

robot_sf normally treats a predicted pedestrian as a deterministic point with a
fixed physical radius when a planner checks clearance. That is over-optimistic at
longer prediction horizons: prediction error grows with the look-ahead time, so a
disc that is exactly the pedestrian's body radius understates how much room the
planner should keep at horizon step ``i``.

This module defines a deterministic scalar-radius envelope for pedestrian
obstacle representation. It is a planning-interface seam only; it does not
perform conformal calibration or provide a safety certificate. The default
policy is a linear inflation ``r_eff(i) = base_radius + alpha * i * dt`` where
``alpha`` (metres per second) is a tunable conservatism knob and ``dt`` is the
planning timestep. ``alpha == 0`` (or a disabled config) reproduces the current
deterministic behaviour exactly, so the abstraction is opt-in and regression
safe.

The inflation is expressed as a plain ``Callable[[int], float]`` so a future
slice can swap the linear policy for a calibrated conformal bound (see issue
#4138) without changing this dataclass or any planner clearance query.
:class:`ConformalInflationPolicy` documents that seam as an explicit,
unimplemented stub.

.. admonition:: Claim boundary
   :class: note

   Providing this abstraction is **not** evidence that inflating the obstacle
   radius improves safety, success, or any benchmark metric. It only gives the
   planner a structured, tunable seam for conservatism. ``alpha`` is a heuristic
   conservatism knob, not a certified coverage bound. Any planning-benefit or
   calibration claim requires separate benchmark evidence per the project's
   maintainer values.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Horizon-indexed spatial inflation function. Maps a non-negative prediction
# horizon step index to an additive radius (metres).
type SpatialInflationPolicy = Callable[[int], float]

# Machine-readable schema version for the diagnostics/provenance payload.
ENVELOPE_SCHEMA_VERSION = "pedestrian_uncertainty_envelope.v1"

# Default conservatism rate in metres of added radius per second of look-ahead.
# Matches the issue #4141 design default; it is a heuristic, not a calibrated
# bound.
DEFAULT_ALPHA_MPS = 0.1


@runtime_checkable
class ConformalInflationPolicy(Protocol):
    """Stub interface for a future conformal-prediction inflation policy.

    This protocol documents the upgrade seam described in issue #4141 and
    tracked by issue #4138. A calibrated implementation would derive the added
    radius ``r_conf(i)`` from trajectory-log residuals with distribution-free
    coverage, replacing :func:`linear_inflation_policy` without changing
    :class:`PedestrianUncertaintyEnvelope` or any planner clearance query.

    No calibrated implementation ships in this slice; conforming objects are
    simply any callable matching the :data:`SpatialInflationPolicy` signature.
    The protocol exists so callers can type-annotate the seam and so a future
    slice has a named contract to satisfy.
    """

    def __call__(self, horizon_step: int) -> float:
        """Return the additive radius (metres) for the given horizon step."""
        ...


@dataclass(frozen=True, slots=True)
class PedestrianUncertaintyEnvelope:
    """Pedestrian obstacle represented by position plus horizon-dependent radius.

    Attributes:
        position: Nominal pedestrian position ``(x, y)`` in world coordinates.
        base_radius: Physical/nominal obstacle radius in metres; must be finite
            and non-negative.
        spatial_inflation: Callable mapping a non-negative horizon step index to
            an additive radius (metres). ``spatial_inflation(0)`` must be finite
            and non-negative.
    """

    position: tuple[float, float]
    base_radius: float
    spatial_inflation: SpatialInflationPolicy

    def __post_init__(self) -> None:
        """Validate scalar fields and the policy's step-0 value."""
        position = np.asarray(self.position, dtype=float).reshape(-1)
        if position.size != 2 or not np.all(np.isfinite(position)):
            raise ValueError(
                "PedestrianUncertaintyEnvelope.position must contain two finite floats"
            )
        if not math.isfinite(float(self.base_radius)) or self.base_radius < 0.0:
            raise ValueError("PedestrianUncertaintyEnvelope.base_radius must be finite and >= 0")
        inflation_0 = float(self.spatial_inflation(0))
        if not math.isfinite(inflation_0) or inflation_0 < 0.0:
            raise ValueError("spatial_inflation(0) must be finite and >= 0")

    def effective_radius(self, horizon_step: int) -> float:
        """Return base radius plus non-negative inflation for a horizon step.

        Args:
            horizon_step: Non-negative prediction horizon step index. Step ``0``
                returns ``base_radius`` for the standard linear policy.

        Returns:
            ``base_radius + spatial_inflation(horizon_step)`` in metres.

        Raises:
            ValueError: If ``horizon_step`` is negative or the policy returns a
                non-finite/negative inflation.
        """
        step = int(horizon_step)
        if step < 0:
            raise ValueError("horizon_step must be >= 0")
        inflation = float(self.spatial_inflation(step))
        if not math.isfinite(inflation) or inflation < 0.0:
            raise ValueError("spatial_inflation(horizon_step) must be finite and >= 0")
        return float(self.base_radius + inflation)


def linear_inflation_policy(alpha: float, dt: float) -> SpatialInflationPolicy:
    """Return the default linear horizon-step inflation ``alpha * horizon_step * dt``.

    Step ``0`` always returns ``0.0`` so the effective radius at the current
    timestep equals ``base_radius``, preserving deterministic behaviour when the
    envelope is first applied.

    Args:
        alpha: Additive radius growth per second of prediction horizon (m/s).
            ``alpha == 0.0`` yields a zero-inflation (deterministic) policy.
        dt: Planning timestep in seconds; must be strictly positive.

    Returns:
        A ``SpatialInflationPolicy`` mapping ``horizon_step`` to added radius.

    Raises:
        ValueError: If ``alpha`` is negative/non-finite, ``dt`` is not strictly
            positive/finite, or a negative horizon step is queried.
    """
    alpha_f = float(alpha)
    dt_f = float(dt)
    if not math.isfinite(alpha_f) or alpha_f < 0.0:
        raise ValueError("alpha must be finite and >= 0")
    if not math.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError("dt must be finite and > 0")

    def _inflation(horizon_step: int) -> float:
        step = int(horizon_step)
        if step < 0:
            raise ValueError("horizon_step must be >= 0")
        return alpha_f * float(step) * dt_f

    return _inflation


def envelope_from_position(
    position: tuple[float, float],
    base_radius: float,
    *,
    alpha: float = DEFAULT_ALPHA_MPS,
    dt: float,
) -> PedestrianUncertaintyEnvelope:
    """Build an envelope at ``position`` using the default linear policy.

    Args:
        position: Nominal pedestrian position ``(x, y)`` in world coordinates.
        base_radius: Physical/nominal obstacle radius in metres.
        alpha: Conservatism rate in metres per second of look-ahead.
        dt: Planning timestep in seconds; must be strictly positive.

    Returns:
        A ``PedestrianUncertaintyEnvelope`` with a linear inflation policy.
    """
    return PedestrianUncertaintyEnvelope(
        position=position,
        base_radius=float(base_radius),
        spatial_inflation=linear_inflation_policy(alpha, dt),
    )


def effective_pedestrian_radius(
    *,
    base_radius: float,
    horizon_step: int,
    alpha: float,
    dt: float,
    enabled: bool,
) -> float:
    """Return the horizon-dependent effective pedestrian radius for clearance.

    This is the planner-agnostic substitution point: a clearance query passes its
    nominal ``base_radius`` and the current ``horizon_step`` and receives the
    inflated radius. When ``enabled`` is ``False`` the nominal radius is returned
    unchanged, so a disabled envelope (or ``alpha == 0.0``) is a bit-for-bit
    no-op.

    Args:
        base_radius: Nominal pedestrian obstacle radius in metres.
        horizon_step: Non-negative prediction horizon step index.
        alpha: Conservatism rate in metres per second of look-ahead.
        dt: Planning timestep in seconds; must be strictly positive when enabled.
        enabled: Whether the envelope is applied at all.

    Returns:
        The (possibly inflated) effective radius in metres.
    """
    if not enabled:
        return float(base_radius)
    return PedestrianUncertaintyEnvelope(
        position=(0.0, 0.0),
        base_radius=float(base_radius),
        spatial_inflation=linear_inflation_policy(alpha, dt),
    ).effective_radius(horizon_step)


def envelope_diagnostics(*, enabled: bool, alpha: float, dt: float) -> dict[str, Any]:
    """Build the provenance/diagnostics payload for a planner's envelope settings.

    Args:
        enabled: Whether the envelope is applied.
        alpha: Conservatism rate in metres per second of look-ahead.
        dt: Planning timestep in seconds.

    Returns:
        A JSON-serialisable payload recording the schema version, policy id, the
        settings, and an explicit claim boundary so downstream provenance never
        over-claims a calibration or safety guarantee.
    """
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "policy": "linear" if (enabled and float(alpha) > 0.0) else "deterministic",
        "alpha_mps": float(alpha),
        "dt": float(dt),
        "claim_boundary": (
            "deterministic scalar-radius planning envelope only; "
            "not conformal calibration or safety certification"
        ),
    }
