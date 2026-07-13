"""Versioned schema for the action-conditioned online collision-risk API (issue #5444).

This module defines the *contract* returned by the estimators in
:mod:`robot_sf.research.collision_risk.estimators`. It keeps three fundamentally
different notions of "risk" explicitly separate, because conflating them is the
most common way an online risk surface becomes misleading:

- **model probabilities** -- e.g. the constant-velocity Monte Carlo joint
  contact probability and its first-passage / hazard decomposition. These are
  estimates from an explicit, declared stochastic forecast model and are only as
  good as that model;
- **deterministic warnings** -- time-to-collision (TTC), velocity-obstacle (VO)
  membership, and reachable-set flags. These are *not* probabilities and are
  labelled non-probabilistic;
- **formal / hard guards** -- authoritative collision guards elsewhere in the
  stack. This API never emits a ``safe`` verdict and never lets a low probability
  stand in for a hard guard. See :data:`GUARD_AUTHORITY_NOTE`.

.. admonition:: Claim boundary
   :class: note

   Producing this schema and its baselines is *API + fixture evidence*, not a
   calibrated benchmark risk claim. No ``safe`` label may be inferred from a low
   contact probability, and hard guards remain authoritative.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

RISK_SCHEMA_VERSION = "action_conditioned_collision_risk.v1"

# Label attached to the deterministic block so downstream consumers can never
# mistake a TTC/VO/reachability field for a probability.
DETERMINISTIC_FIELD_LABEL = "deterministic_non_probabilistic"

# The API intentionally refuses to emit a boolean ``safe`` verdict. Low
# probability is not safety; hard guards elsewhere remain authoritative.
GUARD_AUTHORITY_NOTE = (
    "Hard collision guards remain authoritative. This surface emits contact "
    "probabilities and deterministic warnings only; it never emits a 'safe' "
    "label and a low probability must not be treated as a safety guarantee."
)

_PROB_TOL = 1e-6


class RiskSchemaError(ValueError):
    """Raised when a risk estimate violates its versioned schema contract."""


def _finite(value: float) -> bool:
    """Return True when ``value`` is a finite real number."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _json_safe(value: object) -> object:
    """Convert values to strict JSON-compatible primitives (NaN/inf -> None).

    Returns:
        A structure containing only JSON-safe primitives.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class DeterministicRiskFields:
    """Non-probabilistic deterministic warnings for a candidate action.

    These fields are computed from the *noise-free* constant-velocity
    extrapolation and exact footprint geometry. They are deterministic warnings,
    not probabilities, and are labelled :data:`DETERMINISTIC_FIELD_LABEL`.

    Attributes:
        min_clearance_m: Minimum robot/actor footprint clearance over the horizon
            (centre distance minus summed radii). Negative means overlap.
        min_clearance_step: Horizon step index at which the minimum occurs.
        ttc_s: Time-to-collision in seconds for the noise-free rollout, or
            ``inf`` when the footprints never touch within the horizon.
        contact_certain: True when the noise-free rollout makes footprints touch.
        first_contact_step: First horizon step of footprint contact, or ``-1``.
        velocity_obstacle_flags: Per-actor flag; True when the robot's initial
            commanded velocity lies in that actor's velocity obstacle within the
            horizon (constant-velocity cone test).
        reachable_actor_flags: Per-actor conservative reachable-set warning; True
            when the actor is within the robot's maximum reachable distance over
            the horizon. Over-approximate by construction.
    """

    min_clearance_m: float
    min_clearance_step: int
    ttc_s: float
    contact_certain: bool
    first_contact_step: int
    velocity_obstacle_flags: tuple[bool, ...] = field(default_factory=tuple)
    reachable_actor_flags: tuple[bool, ...] = field(default_factory=tuple)
    label: str = DETERMINISTIC_FIELD_LABEL

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the deterministic fields."""
        return _json_safe(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class PerActorContribution:
    """Per-actor marginal contact contribution.

    The marginal is ``P(contact with this actor by horizon)`` estimated from the
    same Monte Carlo samples as the joint probability. Marginals are *not*
    independent and must not be summed as if they were: the joint probability of
    any contact is generally smaller than their sum because samples can contact
    several actors at once. :attr:`overlap_note` records this explicitly.

    Attributes:
        actor_id: Identifier of the pedestrian/actor.
        marginal_contact_probability: ``P(contact with this actor by horizon)``.
        first_contact_step_mode: Most frequent first-contact step for this actor
            across contacting samples, or ``-1`` when the actor never contacts.
    """

    actor_id: int
    marginal_contact_probability: float
    first_contact_step_mode: int
    overlap_note: str = "marginal; not independent -- do not sum marginals as a joint probability"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the per-actor contribution."""
        return _json_safe(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class UncertaintyState:
    """Monte Carlo uncertainty, out-of-distribution, and abstention state.

    Attributes:
        n_samples: Number of Monte Carlo samples used.
        mc_standard_error: Standard error of the joint probability estimate.
        ci95_halfwidth: Half-width of the 95% Wald interval on the joint estimate.
        abstained: True when the estimate should not be trusted as-is (either the
            interval is too wide or an out-of-distribution input was detected).
        abstention_reasons: Human-readable reasons for abstention (empty when not
            abstaining).
        ood_actor_flags: Per-actor out-of-distribution flag (e.g. speed exceeds
            the declared model validity range).
    """

    n_samples: int
    mc_standard_error: float
    ci95_halfwidth: float
    abstained: bool
    abstention_reasons: tuple[str, ...] = field(default_factory=tuple)
    ood_actor_flags: tuple[bool, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the uncertainty state."""
        return _json_safe(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class RiskProvenance:
    """Estimator, forecast, geometry, horizon, action, and config provenance.

    Attributes:
        estimator_id: Identifier of the risk estimator.
        forecast_model: Identifier of the pedestrian forecast model.
        geometry_version: Identifier of the footprint/geometry convention.
        horizon_steps: Number of horizon steps ``H``.
        horizon_s: Horizon length in seconds (``H * dt``).
        dt_s: Timestep in seconds.
        action_id: Identifier of the scored candidate action.
        action_representation: How the action is represented (e.g. ``waypoints``).
        config_hash: Stable hash of the estimator configuration.
        seed: Monte Carlo seed used.
        schema_version: Version tag of this schema.
    """

    estimator_id: str
    forecast_model: str
    geometry_version: str
    horizon_steps: int
    horizon_s: float
    dt_s: float
    action_id: str
    action_representation: str
    config_hash: str
    seed: int
    schema_version: str = RISK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the provenance."""
        return _json_safe(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class LatencySummary:
    """Latency summary for one or more risk estimates against a deadline.

    Attributes:
        p50_ms: Median call latency in milliseconds.
        p95_ms: 95th-percentile call latency in milliseconds.
        p99_ms: 99th-percentile call latency in milliseconds.
        n_calls: Number of timed calls aggregated.
        deadline_ms: Control deadline the estimator is measured against.
        deadline_misses: Number of calls whose latency exceeded the deadline.
        classification: ``"online"`` when ``p95_ms <= deadline_ms``, else
            ``"offline_only"``.
    """

    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_calls: int
    deadline_ms: float
    deadline_misses: int
    classification: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the latency summary."""
        return _json_safe(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class ActionConditionedRiskEstimate:
    """Versioned action-conditioned collision-risk estimate for one candidate action.

    Estimand::

        P(contact in (t, t+H] | history h_t, candidate action u[t:t+H],
          prediction model M, geometry/uncertainty version V)

    The object keeps model probabilities, deterministic warnings, and hard-guard
    authority explicitly separate. It never emits a ``safe`` verdict.

    Attributes:
        joint_contact_probability: ``P(any contact by horizon)`` (Monte Carlo).
        no_contact_probability: ``1 - joint_contact_probability``.
        per_actor: Per-actor marginal contributions (not independent).
        union_bound_probability: ``min(1, sum of marginals)`` -- an upper bound.
        independence_approx_probability: ``1 - prod(1 - marginal)`` -- an
            intentionally *invalid* independence approximation, kept for
            comparison only.
        first_passage_distribution: ``P(first contact at step i)`` per horizon
            step; sums to ``joint_contact_probability``.
        binned_hazard: Discrete hazard per horizon step,
            ``P(first contact at i | survived to i)``.
        deterministic: Non-probabilistic deterministic warning fields.
        uncertainty: Monte Carlo / OOD / abstention state.
        provenance: Estimator, forecast, geometry, horizon, action, config tags.
        latency: Latency summary for this call (or an aggregate).
    """

    joint_contact_probability: float
    no_contact_probability: float
    per_actor: tuple[PerActorContribution, ...]
    union_bound_probability: float
    independence_approx_probability: float
    first_passage_distribution: tuple[float, ...]
    binned_hazard: tuple[float, ...]
    deterministic: DeterministicRiskFields
    uncertainty: UncertaintyState
    provenance: RiskProvenance
    latency: LatencySummary | None = None
    schema_version: str = RISK_SCHEMA_VERSION
    guard_authority_note: str = GUARD_AUTHORITY_NOTE

    def to_dict(self) -> dict[str, object]:
        """Return a strictly JSON-safe mapping of the whole estimate."""
        return {
            "schema_version": self.schema_version,
            "joint_contact_probability": _json_safe(self.joint_contact_probability),
            "no_contact_probability": _json_safe(self.no_contact_probability),
            "per_actor": [contribution.to_dict() for contribution in self.per_actor],
            "union_bound_probability": _json_safe(self.union_bound_probability),
            "independence_approx_probability": _json_safe(self.independence_approx_probability),
            "first_passage_distribution": _json_safe(list(self.first_passage_distribution)),
            "binned_hazard": _json_safe(list(self.binned_hazard)),
            "deterministic": self.deterministic.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "provenance": self.provenance.to_dict(),
            "latency": self.latency.to_dict() if self.latency is not None else None,
            "guard_authority_note": self.guard_authority_note,
        }

    def validate(self) -> ActionConditionedRiskEstimate:
        """Validate the versioned schema contract, returning ``self`` on success.

        Checks finite probability ranges, monotonic cumulative first-passage,
        first-passage/joint consistency, horizon field agreement, and provenance
        completeness. Raises :class:`RiskSchemaError` on the first violation.

        Returns:
            The validated estimate (for fluent use).
        """
        if self.schema_version != RISK_SCHEMA_VERSION:
            raise RiskSchemaError(
                f"schema_version must be {RISK_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )

        for name, prob in (
            ("joint_contact_probability", self.joint_contact_probability),
            ("no_contact_probability", self.no_contact_probability),
            ("union_bound_probability", self.union_bound_probability),
            ("independence_approx_probability", self.independence_approx_probability),
        ):
            if not _finite(prob) or not (0.0 - _PROB_TOL <= prob <= 1.0 + _PROB_TOL):
                raise RiskSchemaError(
                    f"{name} must be a finite probability in [0, 1], got {prob!r}"
                )

        if abs(self.joint_contact_probability + self.no_contact_probability - 1.0) > 1e-6:
            raise RiskSchemaError("joint_contact_probability + no_contact_probability must equal 1")

        self._validate_first_passage()
        self._validate_marginals()
        self._validate_deterministic()
        self._validate_provenance()
        return self

    def _validate_first_passage(self) -> None:
        """Validate first-passage non-negativity, monotonic CDF, and joint sum."""
        cumulative = 0.0
        for step, value in enumerate(self.first_passage_distribution):
            if not _finite(value) or value < -_PROB_TOL:
                raise RiskSchemaError(
                    f"first_passage_distribution[{step}] must be a finite non-negative"
                    f" probability, got {value!r}"
                )
            cumulative += value
            if cumulative > 1.0 + _PROB_TOL:
                raise RiskSchemaError("cumulative first-passage probability exceeds 1")
        if abs(cumulative - self.joint_contact_probability) > 1e-6:
            raise RiskSchemaError(
                "first_passage_distribution must sum to joint_contact_probability "
                f"({cumulative!r} vs {self.joint_contact_probability!r})"
            )
        if len(self.binned_hazard) != len(self.first_passage_distribution):
            raise RiskSchemaError("binned_hazard and first_passage_distribution length mismatch")
        for step, hazard in enumerate(self.binned_hazard):
            if not _finite(hazard) or not (0.0 - _PROB_TOL <= hazard <= 1.0 + _PROB_TOL):
                raise RiskSchemaError(
                    f"binned_hazard[{step}] must be a finite probability in [0, 1], got {hazard!r}"
                )

    def _validate_marginals(self) -> None:
        """Validate per-actor marginals are finite probabilities."""
        for contribution in self.per_actor:
            prob = contribution.marginal_contact_probability
            if not _finite(prob) or not (0.0 - _PROB_TOL <= prob <= 1.0 + _PROB_TOL):
                raise RiskSchemaError(
                    f"actor {contribution.actor_id} marginal must be a finite probability, "
                    f"got {prob!r}"
                )

    def _validate_deterministic(self) -> None:
        """Validate deterministic monotone/finite fields and label."""
        det = self.deterministic
        if det.label != DETERMINISTIC_FIELD_LABEL:
            raise RiskSchemaError("deterministic block must carry the non-probabilistic label")
        # +inf means no actors were present within the horizon.
        if not (det.min_clearance_m == float("inf") or _finite(det.min_clearance_m)):
            raise RiskSchemaError("min_clearance_m must be finite or +inf")
        if not (det.ttc_s == float("inf") or (_finite(det.ttc_s) and det.ttc_s >= 0.0)):
            raise RiskSchemaError("ttc_s must be non-negative or +inf")
        if len(det.velocity_obstacle_flags) != len(self.per_actor):
            raise RiskSchemaError("velocity_obstacle_flags length must match per_actor")
        if len(det.reachable_actor_flags) != len(self.per_actor):
            raise RiskSchemaError("reachable_actor_flags length must match per_actor")

    def _validate_provenance(self) -> None:
        """Validate provenance completeness and horizon-field agreement."""
        prov = self.provenance
        required = (
            prov.estimator_id,
            prov.forecast_model,
            prov.geometry_version,
            prov.action_id,
            prov.action_representation,
            prov.config_hash,
        )
        if any(not str(field_value).strip() for field_value in required):
            raise RiskSchemaError("provenance identifier fields must be non-empty")
        if prov.horizon_steps <= 0:
            raise RiskSchemaError("horizon_steps must be positive")
        if len(self.first_passage_distribution) != prov.horizon_steps:
            raise RiskSchemaError(
                "first_passage_distribution length must equal provenance.horizon_steps"
            )
        if not _finite(prov.dt_s) or prov.dt_s <= 0.0:
            raise RiskSchemaError("dt_s must be positive")
        if abs(prov.horizon_s - prov.horizon_steps * prov.dt_s) > 1e-6:
            raise RiskSchemaError("horizon_s must equal horizon_steps * dt_s")


def latency_summary_from_samples(
    samples_ms: list[float] | tuple[float, ...],
    *,
    deadline_ms: float,
) -> LatencySummary:
    """Build a :class:`LatencySummary` from per-call latencies.

    Args:
        samples_ms: One or more per-call latencies in milliseconds.
        deadline_ms: Control deadline to classify against.

    Returns:
        Aggregated latency summary with p50/p95/p99, deadline misses, and an
        ``online`` / ``offline_only`` classification based on p95.
    """
    if not samples_ms:
        raise ValueError("samples_ms must contain at least one latency")
    ordered = sorted(float(value) for value in samples_ms)

    def _percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = fraction * (len(ordered) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return ordered[low]
        weight = rank - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    p95 = _percentile(0.95)
    misses = sum(1 for value in ordered if value > deadline_ms)
    classification = "online" if p95 <= deadline_ms else "offline_only"
    return LatencySummary(
        p50_ms=_percentile(0.5),
        p95_ms=p95,
        p99_ms=_percentile(0.99),
        n_calls=len(ordered),
        deadline_ms=float(deadline_ms),
        deadline_misses=misses,
        classification=classification,
    )


def estimate_from_dict(payload: Mapping[str, object]) -> dict[str, object]:  # pragma: no cover
    """Round-trip helper placeholder kept intentionally minimal.

    The canonical direction is object -> dict via :meth:`ActionConditionedRiskEstimate.to_dict`.
    A dict -> object loader is deferred until a concrete consumer needs it.
    """
    raise NotImplementedError("dict -> estimate loading is not part of the v1 contract")


__all__ = [
    "DETERMINISTIC_FIELD_LABEL",
    "GUARD_AUTHORITY_NOTE",
    "RISK_SCHEMA_VERSION",
    "ActionConditionedRiskEstimate",
    "DeterministicRiskFields",
    "LatencySummary",
    "PerActorContribution",
    "RiskProvenance",
    "RiskSchemaError",
    "UncertaintyState",
    "latency_summary_from_samples",
]
