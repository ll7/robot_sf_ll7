"""Map a ForecastBatch.v1 artifact into a per-step scalar risk signal.

This adapter is the bounded bridge between a forecast artifact (see
``robot_sf.benchmark.forecast_batch``) and a downstream risk gate or planner
proxy.  It is deliberately small, pure, and deterministic: given a forecast
batch and the robot position at the same step, it returns a single non-negative
scalar risk value plus the metadata a fail-closed consumer needs.

Design boundaries (issue #2916, evidence_tier: stress, diagnostic_only):

- The risk scalar is a *geometric proximity-to-forecast-occupancy* signal, not a
  learned or calibrated probability.  It rewards short predicted robot/pedestrian
  separation and widens with forecast uncertainty.  It makes no benchmark claim.
- Fail-closed contract: a batch that is degraded, fallback, oracle-sourced, or
  carries no usable forecast payload yields ``available=False``.  A consumer must
  treat ``available=False`` as *not a success row*, never as zero risk.
- Pure and deterministic: no RNG, no I/O, no global state.  The same inputs
  always produce the same output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from robot_sf.benchmark.forecast_batch import ForecastBatch

FORECAST_RISK_ADAPTER_SCHEMA_VERSION = "forecast_risk_signal.v1"

# A row is treated as deployable only when both status fields declare a clean,
# native execution.  Anything else fails closed.
_CLEAN_FALLBACK_STATUSES = frozenset({"native", "none"})
_CLEAN_DEGRADED_STATUSES = frozenset({"none"})


@dataclass(frozen=True)
class ForecastRiskSignal:
    """Per-step scalar risk derived from a forecast batch.

    Attributes:
        risk: Non-negative scalar risk for the step (0.0 when no actor is within
            the influence radius).  Higher means a closer / more uncertain
            predicted encounter.
        available: True only when the batch was deployable and at least one actor
            forecast contributed.  False means fail-closed: the consumer must not
            count this as a usable risk row.
        reason: Short machine-readable reason, ``"ok"`` when available, otherwise
            the fail-closed cause (e.g. ``"degraded_observation"``).
        contributing_actor_count: Number of actor forecasts that contributed.
        nearest_predicted_distance_m: Closest robot/forecast-mean distance across
            actors and horizons, or ``None`` when nothing contributed.
        metadata: Provenance echo for the evidence bundle.
    """

    risk: float
    available: bool
    reason: str
    contributing_actor_count: int = 0
    nearest_predicted_distance_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        Returns:
            Mapping with the schema version and all signal fields.
        """
        return {
            "schema_version": FORECAST_RISK_ADAPTER_SCHEMA_VERSION,
            "risk": self.risk,
            "available": self.available,
            "reason": self.reason,
            "contributing_actor_count": self.contributing_actor_count,
            "nearest_predicted_distance_m": self.nearest_predicted_distance_m,
            "metadata": dict(self.metadata),
        }


def _batch_is_deployable(batch: ForecastBatch) -> tuple[bool, str]:
    """Classify whether a forecast batch may feed a risk gate.

    Returns:
        ``(deployable, reason)``.  ``reason`` is ``"ok"`` when deployable, else a
        machine-readable fail-closed cause.
    """
    prov = batch.provenance
    if prov.oracle_state:
        return False, "oracle_state"
    fallback = prov.fallback_status.strip().lower()
    if fallback not in _CLEAN_FALLBACK_STATUSES:
        return False, f"fallback:{fallback}"
    degraded = prov.degraded_status.strip().lower()
    if degraded not in _CLEAN_DEGRADED_STATUSES:
        return False, f"degraded:{degraded}"
    if not any(included for included in prov.actor_mask):
        return False, "no_actor_available"
    if not batch.forecasts:
        return False, "no_forecast_payload"
    return True, "ok"


def _actor_forecast_distance(
    deterministic: np.ndarray,
    robot_position: np.ndarray,
) -> float:
    """Closest robot/forecast-mean distance over horizons for one actor.

    Returns:
        Minimum Euclidean distance (meters) between the robot and the actor's
        forecast occupancy means across all horizons.
    """
    diffs = deterministic - robot_position.reshape(1, 2)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return float(np.min(distances))


def compute_forecast_risk(
    batch: ForecastBatch,
    robot_position: np.ndarray | list[float] | tuple[float, float],
    *,
    influence_radius_m: float = 3.0,
) -> ForecastRiskSignal:
    """Map one forecast batch to a per-step scalar risk for a risk gate.

    The risk is a bounded proximity signal in ``[0, 1]``: for each actor with a
    deterministic forecast, the closest robot/forecast-mean distance across
    horizons is converted to ``max(0, 1 - d / influence_radius_m)``.  The actor
    risks are combined by taking the maximum (the single most threatening
    predicted encounter drives the gate).  Actors beyond ``influence_radius_m``
    contribute zero.

    The function is pure and deterministic.  It fails closed: degraded, fallback,
    oracle, or payload-less batches return ``available=False`` with ``risk=0.0``
    and a reason, and the caller must not treat that as a usable low-risk row.

    Args:
        batch: A validated :class:`ForecastBatch`.
        robot_position: The robot ``(x, y)`` position at the same step, in the
            batch's coordinate frame.
        influence_radius_m: Distance beyond which a predicted occupancy mean adds
            no risk.  Must be positive.

    Returns:
        A :class:`ForecastRiskSignal` for the step.
    """
    if influence_radius_m <= 0.0:
        raise ValueError("influence_radius_m must be positive")
    robot = np.asarray(robot_position, dtype=float).reshape(2)
    if not np.all(np.isfinite(robot)):
        raise ValueError("robot_position must be finite")

    deployable, reason = _batch_is_deployable(batch)
    base_metadata = {
        "predictor_id": batch.provenance.predictor_id,
        "predictor_family": batch.provenance.predictor_family,
        "observation_tier": batch.provenance.observation_tier,
        "scenario_id": batch.provenance.scenario_id,
        "seed": batch.provenance.seed,
        "fallback_status": batch.provenance.fallback_status,
        "degraded_status": batch.provenance.degraded_status,
        "influence_radius_m": float(influence_radius_m),
    }
    if not deployable:
        return ForecastRiskSignal(
            risk=0.0,
            available=False,
            reason=reason,
            metadata=base_metadata,
        )

    per_actor_risk: list[float] = []
    distances: list[float] = []
    for forecast in batch.forecasts:
        if forecast.deterministic is None:
            continue
        distance = _actor_forecast_distance(forecast.deterministic, robot)
        distances.append(distance)
        proximity = 1.0 - distance / influence_radius_m
        per_actor_risk.append(max(0.0, proximity))

    if not per_actor_risk:
        return ForecastRiskSignal(
            risk=0.0,
            available=False,
            reason="no_deterministic_payload",
            metadata=base_metadata,
        )

    risk = float(max(per_actor_risk))
    return ForecastRiskSignal(
        risk=risk,
        available=True,
        reason="ok",
        contributing_actor_count=len(per_actor_risk),
        nearest_predicted_distance_m=float(min(distances)),
        metadata=base_metadata,
    )


__all__ = [
    "FORECAST_RISK_ADAPTER_SCHEMA_VERSION",
    "ForecastRiskSignal",
    "compute_forecast_risk",
]
