"""Tracking-precision drift mask and a precision-to-speed operational contract.

Line-of-sight raycasting is otherwise treated as an uncorrupted truth stream with
zero tracking decay. A defensible pre-deployment safety case must instead show the
planner holds a separation buffer that absorbs *tracking localization error*. This
module provides the tracking-precision lever from issue #3480:

- a stochastic **tracking-drift mask** that perturbs tracked actor coordinates by a
  target MOTP-like precision (MOTP = 0 ⇒ pass-through ground truth), and
- a precision-to-speed **operational contract** that drops the planner max-speed cap
  to a defensive ceiling once tracking precision crosses a threshold ``T_u``.

.. warning::

   The Gaussian drift is an explicit **proxy**, not a measured or hardware-calibrated
   sensor-noise model, and these signals are not paper-facing perception-robustness
   evidence. Reuse durable runtime evidence and the observation-noise portfolio
   (#2777 / #2927) before any such claim.

MOTP convention: MOTP is the mean Euclidean position error of tracked actors. For an
isotropic 2D Gaussian with per-axis standard deviation ``σ``, the mean Euclidean error
is ``σ·√(π/2)`` (the Rayleigh mean). To hit a target ``MOTP = m`` we therefore use
``σ = m / √(π/2)``, so the realized mean drift matches the requested precision.

This module is pure and side-effect free; it does not alter the live perception, action,
or benchmark loop. Wiring the mask and speed-cap into the stepping loop is a deliberate
opt-in follow-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

TRACKING_PRECISION_SCHEMA = "tracking_precision_contract.v1"
# Rayleigh mean factor: mean Euclidean error = per-axis sigma * sqrt(pi/2).
_RAYLEIGH_MEAN_FACTOR = math.sqrt(math.pi / 2.0)


def drift_std_for_motp(motp_m: float) -> float:
    """Return the per-axis Gaussian std that realizes a target MOTP (mean error).

    Args:
        motp_m: Target mean Euclidean tracking error in metres (>= 0).

    Returns:
        float: Per-axis standard deviation ``σ = MOTP / √(π/2)``.
    """
    if motp_m < 0.0:
        raise ValueError(f"motp_m must be >= 0, got {motp_m!r}")
    return float(motp_m) / _RAYLEIGH_MEAN_FACTOR


def apply_tracking_drift(
    positions: NDArray[np.floating] | object,
    motp_m: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return tracked positions perturbed by a MOTP-parameterized Gaussian drift.

    ``motp_m == 0`` is an exact pass-through (a copy of the input). Otherwise each
    coordinate gets independent zero-mean Gaussian noise with std
    :func:`drift_std_for_motp`.

    Returns:
        NDArray[np.float64]: Perturbed positions with the same shape as the input.
    """
    arr = np.asarray(positions, dtype=np.float64)
    if motp_m == 0.0:
        return arr.copy()
    std = drift_std_for_motp(motp_m)
    return arr + rng.normal(0.0, std, size=arr.shape)


def minimum_separation(
    observed_positions: NDArray[np.floating] | object,
    reference_point: NDArray[np.floating] | object,
) -> float:
    """Return the minimum distance from ``reference_point`` to any observed actor.

    Computed on the (possibly corrupted) observation vector, not ground truth.

    Returns:
        float: Minimum Euclidean distance, or ``inf`` when no actors are observed.
    """
    actors = np.asarray(observed_positions, dtype=np.float64).reshape(-1, 2)
    if actors.shape[0] == 0:
        return float("inf")
    ref = np.asarray(reference_point, dtype=np.float64).reshape(2)
    return float(np.min(np.linalg.norm(actors - ref, axis=1)))


@dataclass(frozen=True, slots=True)
class TrackingPrecisionContract:
    """Precision-to-speed operational contract (non-hardware proxy thresholds).

    Attributes:
        precision_threshold_m: ``T_u`` — MOTP at/above which the defensive cap applies.
        default_speed_cap: Max-speed cap under good tracking precision (m/s).
        defensive_speed_cap: Reduced cap once precision is degraded (m/s).
        schema_version: Stable schema tag for reproducibility.
    """

    precision_threshold_m: float = 2.5
    default_speed_cap: float = 2.0
    defensive_speed_cap: float = 0.5
    schema_version: str = TRACKING_PRECISION_SCHEMA

    def __post_init__(self) -> None:
        """Validate that the contract is well-formed."""
        if not (self.precision_threshold_m > 0.0):
            raise ValueError("precision_threshold_m must be > 0")
        if not (self.default_speed_cap > 0.0) or not (self.defensive_speed_cap > 0.0):
            raise ValueError("speed caps must be > 0")
        if self.defensive_speed_cap > self.default_speed_cap:
            raise ValueError("defensive_speed_cap must not exceed default_speed_cap")


def speed_cap_for_precision(
    motp_m: float, contract: TrackingPrecisionContract | None = None
) -> float:
    """Return the contracted max-speed cap for an observed tracking precision."""
    contract = contract or TrackingPrecisionContract()
    if motp_m < 0.0:
        raise ValueError(f"motp_m must be >= 0, got {motp_m!r}")
    if motp_m >= contract.precision_threshold_m:
        return contract.defensive_speed_cap
    return contract.default_speed_cap


def is_contract_honored(
    applied_speed_cap: float,
    motp_m: float,
    contract: TrackingPrecisionContract | None = None,
) -> bool:
    """Return whether an applied speed cap respects the contracted ceiling."""
    contract = contract or TrackingPrecisionContract()
    expected = speed_cap_for_precision(motp_m, contract)
    return applied_speed_cap <= expected + 1e-9


def tracking_precision_telemetry(
    motp_m: float,
    applied_speed_cap: float,
    contract: TrackingPrecisionContract | None = None,
) -> dict[str, Any]:
    """Return a compact, schema-tagged telemetry record (diagnostic only).

    Returns:
        dict[str, Any]: Precision, drift std, contracted/applied caps, and honored flag.
    """
    contract = contract or TrackingPrecisionContract()
    expected = speed_cap_for_precision(motp_m, contract)
    return {
        "schema_version": contract.schema_version,
        "proxy_kind": "internal_non_hardware",
        "motp_m": float(motp_m),
        "drift_std_m": drift_std_for_motp(motp_m),
        "precision_threshold_m": contract.precision_threshold_m,
        "contracted_speed_cap": expected,
        "applied_speed_cap": float(applied_speed_cap),
        "contract_honored": is_contract_honored(applied_speed_cap, motp_m, contract),
        "defensive_regime": motp_m >= contract.precision_threshold_m,
    }


__all__ = [
    "TRACKING_PRECISION_SCHEMA",
    "TrackingPrecisionContract",
    "apply_tracking_drift",
    "drift_std_for_motp",
    "is_contract_honored",
    "minimum_separation",
    "speed_cap_for_precision",
    "tracking_precision_telemetry",
]
