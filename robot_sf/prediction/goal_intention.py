"""Bayesian pedestrian goal-intention inference.

This module provides an interpretable, CPU-only posterior over explicit
candidate goal points from observed pedestrian motion. It is a planner input
interface, not a calibrated human-intention model or benchmark claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from robot_sf.prediction._contract_utils import (
    require_finite,
    require_non_negative,
    require_probability,
    require_step_index,
    require_text,
    stable_config_hash,
)
from robot_sf.prediction.goal_belief_contract import (
    ActorObservationStep,
    CensoringState,
    CoordinateFrame,
    GoalBeliefMode,
    GoalBeliefObservation,
    GoalBeliefSource,
    GoalBeliefV1,
    GoalCandidateKind,
    GoalCandidateProbability,
    ObservationMask,
)


class GoalCandidateRole(StrEnum):
    """Semantic role of a public candidate supplied to actor inference."""

    ACTIVE_WAYPOINT = "active_waypoint"
    FINAL_DESTINATION = "final_destination"
    ROUTE_ENDPOINT = "route_endpoint"
    OPEN_RAY = "open_ray"
    UNKNOWN = "unknown"


class GoalCandidateAvailability(StrEnum):
    """Availability state for a candidate or candidate provider."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GoalCandidate:
    """A versioned public candidate with explicit provenance and geometry semantics."""

    id: str
    position: tuple[float, float] | None
    source: str
    role: GoalCandidateRole = GoalCandidateRole.FINAL_DESTINATION
    route_signature: str | None = None
    availability: GoalCandidateAvailability = GoalCandidateAvailability.AVAILABLE
    prior_weight: float | None = None
    coordinate_frame: CoordinateFrame = CoordinateFrame.GLOBAL_XY

    def __post_init__(self) -> None:
        """Reject ambiguous identity, frame, provenance, and non-finite geometry."""

        object.__setattr__(self, "id", require_text(self.id, "candidate.id"))
        object.__setattr__(self, "source", require_text(self.source, "candidate.source"))
        if not isinstance(self.role, GoalCandidateRole):
            raise TypeError("candidate.role must be GoalCandidateRole")
        if not isinstance(self.availability, GoalCandidateAvailability):
            raise TypeError("candidate.availability must be GoalCandidateAvailability")
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("candidate.coordinate_frame must be CoordinateFrame")
        if self.coordinate_frame is not CoordinateFrame.GLOBAL_XY:
            raise ValueError("candidate.coordinate_frame must be global_xy")
        if self.position is None:
            if self.availability is GoalCandidateAvailability.AVAILABLE and self.role not in {
                GoalCandidateRole.OPEN_RAY,
                GoalCandidateRole.UNKNOWN,
            }:
                raise ValueError("available point candidates require a position")
        else:
            object.__setattr__(self, "position", _finite_xy(self.position, "candidate.position"))
        if self.route_signature is not None:
            object.__setattr__(
                self,
                "route_signature",
                require_text(self.route_signature, "candidate.route_signature"),
            )
        if self.prior_weight is not None:
            object.__setattr__(
                self,
                "prior_weight",
                require_non_negative(self.prior_weight, "candidate.prior_weight"),
            )

    @property
    def candidate_id(self) -> str:
        """Return the stable ID under the naming used by the belief contract."""

        return self.id

    @property
    def kind(self) -> GoalCandidateKind:
        """Map the richer candidate role to the v1 probability role."""

        if self.role is GoalCandidateRole.FINAL_DESTINATION:
            return GoalCandidateKind.FINAL_DESTINATION
        return GoalCandidateKind.ACTIVE_WAYPOINT


# Backwards-compatible name used by the original point-only helper.
CandidateGoal = GoalCandidate

_POINT_CANDIDATE_ROLES = frozenset(
    {
        GoalCandidateRole.ACTIVE_WAYPOINT,
        GoalCandidateRole.FINAL_DESTINATION,
        GoalCandidateRole.ROUTE_ENDPOINT,
    }
)


@dataclass(frozen=True, slots=True)
class GoalCandidateSet:
    """Strict public candidate collection consumed by the actor-only estimator."""

    candidates: tuple[GoalCandidate, ...] = ()
    source: str = "public_candidates"
    coordinate_frame: CoordinateFrame = CoordinateFrame.GLOBAL_XY
    availability: GoalCandidateAvailability = GoalCandidateAvailability.AVAILABLE

    def __post_init__(self) -> None:
        """Validate candidate identity, provider provenance, and frame consistency."""

        object.__setattr__(self, "source", require_text(self.source, "candidate_set.source"))
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("candidate_set.coordinate_frame must be CoordinateFrame")
        if self.coordinate_frame is not CoordinateFrame.GLOBAL_XY:
            raise ValueError("candidate_set.coordinate_frame must be global_xy")
        if not isinstance(self.availability, GoalCandidateAvailability):
            raise TypeError("candidate_set.availability must be GoalCandidateAvailability")
        values = tuple(self.candidates)
        if any(type(candidate) is not GoalCandidate for candidate in values):
            raise TypeError("candidate_set.candidates must contain GoalCandidate values")
        ids = [candidate.id for candidate in values]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_set candidate IDs must be unique")
        if any(candidate.coordinate_frame is not self.coordinate_frame for candidate in values):
            raise ValueError("candidate frames must match candidate_set.coordinate_frame")
        object.__setattr__(self, "candidates", values)

    @classmethod
    def from_points(
        cls,
        points: Mapping[str, tuple[float, float] | Sequence[float]],
        *,
        source: str,
        role: GoalCandidateRole = GoalCandidateRole.FINAL_DESTINATION,
    ) -> GoalCandidateSet:
        """Build a candidate set from explicit global point annotations.

        Returns:
            Candidate set with finite global point candidates.
        """

        return cls(
            candidates=tuple(
                GoalCandidate(
                    id=str(candidate_id),
                    position=_finite_xy(point, "candidate position"),
                    source=source,
                    role=role,
                )
                for candidate_id, point in points.items()
            ),
            source=source,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, metadata-safe candidate-set representation."""

        return {
            "source": self.source,
            "coordinate_frame": self.coordinate_frame.value,
            "availability": self.availability.value,
            "candidates": [
                {
                    "id": candidate.id,
                    "position": list(candidate.position) if candidate.position else None,
                    "source": candidate.source,
                    "role": candidate.role.value,
                    "route_signature": candidate.route_signature,
                    "availability": candidate.availability.value,
                    "prior_weight": candidate.prior_weight,
                    "coordinate_frame": candidate.coordinate_frame.value,
                }
                for candidate in sorted(self.candidates, key=lambda item: item.id)
            ],
        }


@dataclass(frozen=True, slots=True)
class HeadingGoalPosteriorConfig:
    """Configuration for the one-frame, observation-only heading posterior."""

    heading_kappa: float = 4.0
    velocity_min_mps: float = 0.05
    prior_floor: float = 1e-6
    unknown_prior_probability: float = 0.1
    unknown_likelihood: float = 1.0
    stationary_prior_policy: str = "preserve_prior"

    def __post_init__(self) -> None:
        """Validate numeric configuration values."""

        require_non_negative(self.heading_kappa, "heading_kappa")
        require_non_negative(self.velocity_min_mps, "velocity_min_mps")
        if require_finite(self.prior_floor, "prior_floor") <= 0.0:
            raise ValueError("prior_floor must be finite and > 0")
        require_probability(self.unknown_prior_probability, "unknown_prior_probability")
        if require_finite(self.unknown_likelihood, "unknown_likelihood") <= 0.0:
            raise ValueError("unknown_likelihood must be finite and > 0")
        if self.stationary_prior_policy not in {"preserve_prior", "uniform"}:
            raise ValueError("stationary_prior_policy must be preserve_prior or uniform")

    def to_dict(self) -> dict[str, object]:
        """Return every parameter participating in posterior provenance."""

        return {
            "heading_kappa": self.heading_kappa,
            "velocity_min_mps": self.velocity_min_mps,
            "prior_floor": self.prior_floor,
            "unknown_prior_probability": self.unknown_prior_probability,
            "unknown_likelihood": self.unknown_likelihood,
            "stationary_prior_policy": self.stationary_prior_policy,
        }

    @property
    def config_hash(self) -> str:
        """Stable full digest for contract provenance."""

        return stable_config_hash(self.to_dict())


# Preserve the original public name while exposing the research-specific name.
GoalPosteriorConfig = HeadingGoalPosteriorConfig


@dataclass(frozen=True, slots=True)
class GoalIntentionPosterior:
    """Normalized goal posterior and provenance for one pedestrian."""

    pedestrian_id: str
    probabilities: dict[str, float]
    candidate_goals: tuple[CandidateGoal, ...]
    candidate_source: str
    config_hash: str
    blocker: str | None = None

    @property
    def top_goal_id(self) -> str | None:
        """Return the maximum-probability goal ID, or ``None`` when unavailable."""

        if not self.probabilities:
            return None
        return min(self.probabilities, key=lambda goal_id: (-self.probabilities[goal_id], goal_id))

    @property
    def top_goal_confidence(self) -> float | None:
        """Return the maximum posterior probability, or ``None`` when unavailable."""

        top_goal_id = self.top_goal_id
        if top_goal_id is None:
            return None
        return self.probabilities[top_goal_id]

    @property
    def entropy(self) -> float:
        """Return Shannon entropy over the candidate-only posterior."""

        return _entropy((*self.probabilities.values(),))

    def as_planner_summary(self) -> dict[str, object]:
        """Return a JSON-serializable planner observation metadata summary."""

        return {
            "pedestrian_id": self.pedestrian_id,
            "candidate_source": self.candidate_source,
            "config_hash": self.config_hash,
            "top_goal_id": self.top_goal_id,
            "top_goal_confidence": self.top_goal_confidence,
            "entropy": self.entropy,
            "unknown_candidate_probability": 0.0,
            "source": GoalBeliefSource.OBSERVATION_ONLY.value,
            "probabilities": dict(self.probabilities),
            "blocker": self.blocker,
        }


def candidate_goals_from_points(
    points: Mapping[str, tuple[float, float] | Sequence[float]],
    *,
    source: str,
) -> tuple[CandidateGoal, ...]:
    """Build candidate goals from explicit map or scenario annotations.

    Returns:
        Candidate goals preserving input insertion order.
    """

    if not source:
        raise ValueError("source must be non-empty")
    return tuple(
        CandidateGoal(
            id=str(goal_id), position=_finite_xy(position, "candidate position"), source=source
        )
        for goal_id, position in points.items()
    )


def update_goal_posterior(
    *,
    pedestrian_id: str,
    candidate_goals: Sequence[CandidateGoal],
    observed_position: tuple[float, float] | Sequence[float],
    observed_velocity: tuple[float, float] | Sequence[float],
    prior: Mapping[str, float] | None = None,
    config: GoalPosteriorConfig | None = None,
) -> GoalIntentionPosterior:
    """Update the posterior over candidate goals from observed velocity heading.

    The likelihood is proportional to ``exp(kappa * cos(theta))``, where
    ``theta`` is the angle between observed pedestrian velocity and the vector
    from the observed position to each candidate goal. Slow or stationary
    observations return the normalized prior unchanged and record a blocker
    instead of producing NaN likelihoods.

    Returns:
        Normalized posterior and planner-facing provenance for one pedestrian.
    """

    cfg = config or GoalPosteriorConfig()
    goals = _validate_candidate_goals(candidate_goals)
    position = _finite_xy(observed_position, "observed_position")
    velocity = _finite_xy(observed_velocity, "observed_velocity")
    normalized_prior = _normalize_prior(goals, prior, cfg)
    candidate_source = _candidate_source(goals)

    speed = math.hypot(*velocity)
    if speed <= 0.0 or speed < cfg.velocity_min_mps:
        return GoalIntentionPosterior(
            pedestrian_id=pedestrian_id,
            probabilities=normalized_prior,
            candidate_goals=goals,
            candidate_source=candidate_source,
            config_hash=cfg.config_hash,
            blocker="stationary_below_velocity_min_mps",
        )

    velocity_unit = (velocity[0] / speed, velocity[1] / speed)
    alignments: dict[str, float] = {}
    for goal in goals:
        if goal.position is None:
            raise ValueError("candidate goal position must be available")
        to_goal = (goal.position[0] - position[0], goal.position[1] - position[1])
        distance = math.hypot(*to_goal)
        if distance == 0.0:
            alignment = 1.0
        else:
            alignment = (velocity_unit[0] * to_goal[0] + velocity_unit[1] * to_goal[1]) / distance
        alignment = max(-1.0, min(1.0, alignment))
        alignments[goal.id] = alignment

    max_exponent = max(cfg.heading_kappa * alignment for alignment in alignments.values())
    weighted: dict[str, float] = {}
    for goal in goals:
        likelihood = math.exp(cfg.heading_kappa * alignments[goal.id] - max_exponent)
        if not math.isfinite(likelihood):
            raise ValueError("goal likelihood must be finite")
        weighted[goal.id] = normalized_prior[goal.id] * likelihood

    return GoalIntentionPosterior(
        pedestrian_id=pedestrian_id,
        probabilities=_normalize_weights(weighted, cfg.prior_floor),
        candidate_goals=goals,
        candidate_source=candidate_source,
        config_hash=cfg.config_hash,
    )


def update_heading_goal_posterior(  # noqa: PLR0913
    *,
    track_id: str,
    observed_position_global: tuple[float, float] | Sequence[float],
    observed_velocity_global: tuple[float, float] | Sequence[float],
    candidate_set: GoalCandidateSet,
    prior: Mapping[str, float] | None = None,
    config: HeadingGoalPosteriorConfig | None = None,
    timestamp_s: float = 0.0,
    step_index: int = 0,
    tracking_epoch_id: str = "heading-posterior-v1",
) -> GoalBeliefV1:
    """Return an ``H=1`` actor belief from public tracking and candidates only.

    This interface deliberately has no simulator-state or ``goals`` parameter.  It
    consumes global-frame current position/velocity and an explicit candidate
    provider.  Point candidates use only circular heading alignment; unsupported or
    unavailable candidate hypotheses contribute to the unknown mass.
    """

    if type(candidate_set) is not GoalCandidateSet:
        raise TypeError("candidate_set must be a GoalCandidateSet")
    cfg = config or HeadingGoalPosteriorConfig()
    if type(cfg) is not HeadingGoalPosteriorConfig:
        raise TypeError("config must be a HeadingGoalPosteriorConfig")
    track = require_text(track_id, "track_id")
    position = _finite_xy(observed_position_global, "observed_position_global")
    velocity = _finite_xy(observed_velocity_global, "observed_velocity_global")
    timestamp = require_finite(timestamp_s, "timestamp_s")
    step = require_step_index(step_index, "step_index")
    epoch = require_text(tracking_epoch_id, "tracking_epoch_id")

    candidate_ids = {candidate.id for candidate in candidate_set.candidates}
    _validate_actor_prior_keys(prior, candidate_ids)
    point_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in candidate_set.candidates
                if candidate.availability is GoalCandidateAvailability.AVAILABLE
                and candidate.position is not None
                and candidate.role in _POINT_CANDIDATE_ROLES
            ),
            key=lambda candidate: candidate.id,
        )
    )
    blockers: list[str] = [
        "arrival_probability_unestimated",
        "change_probability_unestimated",
    ]
    if any(
        candidate.availability is not GoalCandidateAvailability.AVAILABLE
        for candidate in candidate_set.candidates
    ):
        blockers.append("candidate_unavailable_unknown")
    if any(
        candidate.availability is GoalCandidateAvailability.AVAILABLE
        and (candidate.position is None or candidate.role not in _POINT_CANDIDATE_ROLES)
        for candidate in candidate_set.candidates
    ):
        blockers.append("non_point_candidate_unknown")

    if candidate_set.availability is not GoalCandidateAvailability.AVAILABLE:
        blockers.append(f"candidate_provider_{candidate_set.availability.value}")
        return _actor_goal_belief(
            track_id=track,
            tracking_epoch_id=epoch,
            timestamp_s=timestamp,
            step_index=step,
            position=position,
            velocity=velocity,
            config_hash=cfg.config_hash,
            candidate_probabilities=(),
            unknown_candidate_probability=1.0,
            mode=GoalBeliefMode.UNAVAILABLE,
            blockers=blockers,
        )

    if not point_candidates:
        blockers.append("no_point_candidates")
        return _actor_goal_belief(
            track_id=track,
            tracking_epoch_id=epoch,
            timestamp_s=timestamp,
            step_index=step,
            position=position,
            velocity=velocity,
            config_hash=cfg.config_hash,
            candidate_probabilities=(),
            unknown_candidate_probability=1.0,
            mode=GoalBeliefMode.UNAVAILABLE,
            blockers=blockers,
        )

    candidate_prior, unknown_prior = _actor_initial_prior(point_candidates, prior, cfg)
    speed = math.hypot(*velocity)
    if speed < cfg.velocity_min_mps:
        blockers.append("stationary_below_velocity_min_mps")
        candidate_probabilities, unknown_probability = _actor_stationary_posterior(
            point_candidates,
            candidate_prior,
            unknown_prior,
            cfg,
        )
    else:
        candidate_probabilities, unknown_probability = _actor_heading_posterior(
            point_candidates,
            position,
            velocity,
            candidate_prior,
            unknown_prior,
            cfg,
        )

    if unknown_probability >= max(
        (candidate.probability for candidate in candidate_probabilities),
        default=0.0,
    ):
        blockers.append("unknown_hypothesis_dominant")
    return _actor_goal_belief(
        track_id=track,
        tracking_epoch_id=epoch,
        timestamp_s=timestamp,
        step_index=step,
        position=position,
        velocity=velocity,
        config_hash=cfg.config_hash,
        candidate_probabilities=candidate_probabilities,
        unknown_candidate_probability=unknown_probability,
        mode=GoalBeliefMode.CENSORED,
        blockers=blockers,
    )


def _actor_goal_belief(  # noqa: PLR0913
    *,
    track_id: str,
    tracking_epoch_id: str,
    timestamp_s: float,
    step_index: int,
    position: tuple[float, float],
    velocity: tuple[float, float],
    config_hash: str,
    candidate_probabilities: Sequence[GoalCandidateProbability],
    unknown_candidate_probability: float,
    mode: GoalBeliefMode,
    blockers: Sequence[str],
) -> GoalBeliefV1:
    """Build a contract value while keeping actor construction narrowly typed.

    Returns:
        Observation-only v1 goal belief.
    """

    observation = GoalBeliefObservation(
        track_id=track_id,
        tracking_epoch_id=tracking_epoch_id,
        timestamp_s=timestamp_s,
        step_index=step_index,
        config_hash=config_hash,
        history_steps=(
            ActorObservationStep(
                timestamp_s=timestamp_s,
                step_index=step_index,
                position_xy=position,
                velocity_xy=velocity,
                mask=ObservationMask.OBSERVED,
            ),
        ),
        coordinate_frame=CoordinateFrame.GLOBAL_XY,
        candidate_probabilities=tuple(candidate_probabilities),
        unknown_candidate_probability=unknown_candidate_probability,
        mode=mode,
        censoring_state=CensoringState.UNKNOWN,
        blockers=tuple(dict.fromkeys(blockers)),
    )
    return GoalBeliefV1.from_observation(observation)


def _validate_actor_prior_keys(
    prior: Mapping[str, float] | None,
    candidate_ids: set[str],
) -> None:
    """Reject prior keys that cannot be tied to the declared candidate set."""

    if prior is None:
        return
    if isinstance(prior, (str, bytes)) or not isinstance(prior, Mapping):
        raise TypeError("prior must be a mapping from candidate ID to weight")
    for candidate_id in prior:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("prior candidate IDs must be non-empty text")
    unknown_ids = set(prior) - candidate_ids
    if unknown_ids:
        raise ValueError(f"prior contains unknown candidate IDs: {sorted(unknown_ids)}")


def _actor_initial_prior(
    candidates: Sequence[GoalCandidate],
    prior: Mapping[str, float] | None,
    config: HeadingGoalPosteriorConfig,
) -> tuple[dict[str, float], float]:
    """Return candidate and unknown prior masses before heading evidence."""

    unknown_probability = config.unknown_prior_probability
    candidate_mass = 1.0 - unknown_probability
    if candidate_mass == 0.0:
        return {candidate.id: 0.0 for candidate in candidates}, 1.0
    weights: dict[str, float] = {}
    for candidate in candidates:
        if prior is not None and candidate.id in prior:
            raw_weight = prior[candidate.id]
        elif candidate.prior_weight is not None:
            raw_weight = candidate.prior_weight
        else:
            raw_weight = 1.0
        weight = require_non_negative(raw_weight, f"prior[{candidate.id}]")
        weights[candidate.id] = max(weight, config.prior_floor)
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("actor prior normalization failed")
    return (
        {candidate_id: candidate_mass * weight / total for candidate_id, weight in weights.items()},
        unknown_probability,
    )


def _actor_stationary_posterior(
    candidates: Sequence[GoalCandidate],
    candidate_prior: Mapping[str, float],
    unknown_prior: float,
    config: HeadingGoalPosteriorConfig,
) -> tuple[tuple[GoalCandidateProbability, ...], float]:
    """Preserve or diffuse the prior when heading evidence is unavailable.

    Returns:
        Candidate probabilities and the unchanged unknown mass.
    """

    if config.stationary_prior_policy == "uniform":
        candidate_probability = (1.0 - unknown_prior) / len(candidates)
        probabilities = {candidate.id: candidate_probability for candidate in candidates}
    else:
        probabilities = dict(candidate_prior)
    return _candidate_probabilities(candidates, probabilities), unknown_prior


def _actor_heading_posterior(
    candidates: Sequence[GoalCandidate],
    position: tuple[float, float],
    velocity: tuple[float, float],
    candidate_prior: Mapping[str, float],
    unknown_prior: float,
    config: HeadingGoalPosteriorConfig,
) -> tuple[tuple[GoalCandidateProbability, ...], float]:
    """Apply circular heading likelihoods with stable log-sum-exp normalization.

    Returns:
        Stable candidate probabilities and posterior unknown mass.
    """

    speed = math.hypot(*velocity)
    velocity_unit = (velocity[0] / speed, velocity[1] / speed)
    log_weights: dict[str, float] = {}
    for candidate in candidates:
        if candidate.position is None:
            raise ValueError("candidate position must be available for heading inference")
        to_candidate = (
            candidate.position[0] - position[0],
            candidate.position[1] - position[1],
        )
        distance = math.hypot(*to_candidate)
        alignment = (
            1.0
            if distance == 0.0
            else (velocity_unit[0] * to_candidate[0] + velocity_unit[1] * to_candidate[1])
            / distance
        )
        alignment = max(-1.0, min(1.0, alignment))
        prior_mass = candidate_prior[candidate.id]
        if prior_mass > 0.0:
            log_weights[candidate.id] = math.log(prior_mass) + config.heading_kappa * alignment
    if unknown_prior > 0.0:
        log_weights["__unknown__"] = math.log(unknown_prior) + math.log(config.unknown_likelihood)
    normalized = _normalize_log_weights(log_weights)
    return (
        _candidate_probabilities(
            candidates,
            {candidate.id: normalized.get(candidate.id, 0.0) for candidate in candidates},
        ),
        normalized.get("__unknown__", 0.0),
    )


def _candidate_probabilities(
    candidates: Sequence[GoalCandidate], probabilities: Mapping[str, float]
) -> tuple[GoalCandidateProbability, ...]:
    """Convert stable-ID probability values to the versioned contract type.

    Returns:
        Contract candidate-probability rows in stable ID order.
    """

    return tuple(
        GoalCandidateProbability(
            candidate_id=candidate.id,
            kind=candidate.kind,
            probability=probabilities.get(candidate.id, 0.0),
        )
        for candidate in candidates
    )


def _normalize_log_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize finite log weights without exponent overflow or underflow bias.

    Returns:
        Normalized weights with the same keys.
    """

    if not weights:
        raise ValueError("log-weight normalization requires at least one value")
    maximum = max(weights.values())
    shifted = {key: math.exp(value - maximum) for key, value in weights.items()}
    total = sum(shifted.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("log-weight normalization failed")
    return {key: value / total for key, value in shifted.items()}


def _entropy(probabilities: Sequence[float]) -> float:
    """Return finite Shannon entropy for a normalized or empty distribution."""

    entropy = 0.0
    for probability in probabilities:
        if probability < 0.0 or not math.isfinite(probability):
            raise ValueError("entropy requires finite non-negative probabilities")
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def planner_goal_posterior_channel(
    posteriors: Sequence[GoalIntentionPosterior],
    *,
    enabled: bool,
    source: GoalBeliefSource | str = GoalBeliefSource.OBSERVATION_ONLY,
    oracle_only: bool = False,
    status: str = "ok",
) -> dict[str, object]:
    """Return the optional planner observation channel payload.

    Returns:
        JSON-serializable observation-channel payload. When disabled, the
        posterior map is intentionally empty.
    """

    if not enabled:
        return {"enabled": False, "pedestrian_goal_posteriors": {}}
    try:
        source_value = GoalBeliefSource(source).value
    except (TypeError, ValueError) as exc:
        raise ValueError("source must be a GoalBeliefSource value") from exc
    if not isinstance(oracle_only, bool):
        raise TypeError("oracle_only must be a bool")
    status_value = require_text(status, "status")
    return {
        "enabled": True,
        "status": status_value,
        "source": source_value,
        "oracle_only": oracle_only,
        "pedestrian_goal_posteriors": {
            posterior.pedestrian_id: {
                **posterior.as_planner_summary(),
                "source": source_value,
                "oracle_only": oracle_only,
            }
            for posterior in posteriors
        },
    }


def planner_goal_posterior_channel_unavailable(
    *,
    enabled: bool,
    blocker: str = "candidate_provider_not_configured",
) -> dict[str, object]:
    """Return an explicit actor-channel absence when no public provider is configured."""

    if not enabled:
        return planner_goal_posterior_channel((), enabled=False)
    return {
        "enabled": True,
        "status": "unavailable",
        "source": GoalBeliefSource.OBSERVATION_ONLY.value,
        "oracle_only": False,
        "blocker": require_text(blocker, "blocker"),
        "pedestrian_goal_posteriors": {},
    }


def planner_goal_posterior_channel_from_beliefs(
    beliefs: Sequence[GoalBeliefV1],
    *,
    enabled: bool,
    actor_only: bool = True,
) -> dict[str, object]:
    """Adapt typed beliefs to the legacy planner metadata channel.

    ``actor_only=True`` is the safe default and rejects oracle upper-bound values
    before they reach a planner configured for actor evaluation.

    Returns:
        JSON-shaped planner metadata channel.
    """

    if not enabled:
        return planner_goal_posterior_channel((), enabled=False)
    if not isinstance(actor_only, bool):
        raise TypeError("actor_only must be a bool")
    values = tuple(beliefs)
    if any(type(belief) is not GoalBeliefV1 for belief in values):
        raise TypeError("beliefs must contain GoalBeliefV1 values")
    if actor_only and any(
        belief.source is not GoalBeliefSource.OBSERVATION_ONLY for belief in values
    ):
        raise ValueError("actor-only planner channel rejects oracle source records")
    source_values = {belief.source.value for belief in values}
    source = next(iter(source_values)) if len(source_values) == 1 else "mixed"
    if source == GoalBeliefSource.OBSERVATION_ONLY.value:
        oracle_only = False
    else:
        oracle_only = True
    summaries: dict[str, dict[str, object]] = {}
    for belief in values:
        probabilities = {
            candidate.candidate_id: candidate.probability
            for candidate in belief.candidate_probabilities
        }
        top_goal_id = min(
            probabilities,
            key=lambda candidate_id: (-probabilities[candidate_id], candidate_id),
            default=None,
        )
        summaries[belief.track_id] = {
            "pedestrian_id": belief.track_id,
            "candidate_source": "typed_goal_belief",
            "config_hash": belief.config_hash,
            "top_goal_id": top_goal_id,
            "top_goal_confidence": (
                probabilities[top_goal_id] if top_goal_id is not None else None
            ),
            "entropy": belief.entropy,
            "unknown_candidate_probability": belief.unknown_candidate_probability,
            "source": belief.source.value,
            "oracle_only": oracle_only,
            "probabilities": probabilities,
            "blocker": belief.blockers[0] if belief.blockers else None,
            "blockers": list(belief.blockers),
        }
    return {
        "enabled": True,
        "status": "ok",
        "source": source,
        "oracle_only": oracle_only,
        "pedestrian_goal_posteriors": summaries,
    }


def planner_oracle_goal_posterior_channel_from_state(
    *,
    enabled: bool,
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    goals: Sequence[Sequence[float]],
    pedestrian_ids: Sequence[str] | None = None,
    config: GoalPosteriorConfig | None = None,
    candidate_source: str = "oracle_true_goal_identity",
) -> dict[str, object]:
    """Build an explicitly labelled simulator-state upper-bound channel.

    This helper is reserved for upper-bound evaluation and compatibility smoke
    tests.  It consumes true simulator goals by design and therefore must never
    be used as the actor-side inference path.

    Returns:
        JSON-serializable planner metadata channel.
    """

    if not enabled:
        return planner_goal_posterior_channel((), enabled=False)

    if len(positions) != len(velocities) or len(positions) != len(goals):
        raise ValueError("positions, velocities, and goals must have the same length")
    if pedestrian_ids is not None and len(pedestrian_ids) != len(positions):
        raise ValueError("pedestrian_ids length must match positions")

    posteriors: list[GoalIntentionPosterior] = []
    for index, (position, velocity, goal) in enumerate(
        zip(positions, velocities, goals, strict=True)
    ):
        pedestrian_id = pedestrian_ids[index] if pedestrian_ids is not None else f"ped_{index}"
        candidate_goals = candidate_goals_from_points(
            {f"{pedestrian_id}_route_goal": goal},
            source=candidate_source,
        )
        posteriors.append(
            update_goal_posterior(
                pedestrian_id=pedestrian_id,
                candidate_goals=candidate_goals,
                observed_position=position,
                observed_velocity=velocity,
                config=config,
            )
        )

    return planner_goal_posterior_channel(
        posteriors,
        enabled=True,
        source=GoalBeliefSource.SIMULATOR_UPPER_BOUND,
        oracle_only=True,
        status="oracle_upper_bound",
    )


def planner_goal_posterior_channel_from_state(
    *,
    enabled: bool,
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    goals: Sequence[Sequence[float]],
    pedestrian_ids: Sequence[str] | None = None,
    config: GoalPosteriorConfig | None = None,
    candidate_source: str = "oracle_true_goal_identity",
) -> dict[str, object]:
    """Deprecated compatibility wrapper for the explicitly named oracle helper.

    The name is retained for callers of the issue #4164 smoke API, but every
    enabled result is marked ``source=simulator_upper_bound`` and
    ``oracle_only=true``.  Actor code must use :func:`update_heading_goal_posterior`.

    Returns:
        JSON-shaped, explicitly oracle-labelled planner channel.
    """

    return planner_oracle_goal_posterior_channel_from_state(
        enabled=enabled,
        positions=positions,
        velocities=velocities,
        goals=goals,
        pedestrian_ids=pedestrian_ids,
        config=config,
        candidate_source=candidate_source,
    )


def _finite_xy(
    value: tuple[float, float] | Sequence[float], field_name: str
) -> tuple[float, float]:
    """Validate and return a two-element tuple of finite floats.

    Returns:
        Tuple of (x, y) validated finite floats.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must contain exactly two numeric values")
    try:
        size = len(value)
        first = value[0]
        second = value[1]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"{field_name} must contain exactly two values") from exc
    if size != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    return (
        require_finite(first, f"{field_name}[0]"),
        require_finite(second, f"{field_name}[1]"),
    )


def _validate_candidate_goals(
    candidate_goals: Sequence[CandidateGoal],
) -> tuple[CandidateGoal, ...]:
    """Validate candidate goals for non-empty ids, unique ids, and finite positions.

    Returns:
        Validated tuple of candidate goals.
    """
    goals = tuple(candidate_goals)
    if not goals:
        raise ValueError("candidate_goals must be non-empty")

    seen: set[str] = set()
    for goal in goals:
        if type(goal) is not GoalCandidate:
            raise TypeError("candidate_goals must contain GoalCandidate values")
        if not goal.id:
            raise ValueError("candidate goal id must be non-empty")
        if goal.id in seen:
            raise ValueError(f"duplicate candidate goal id: {goal.id}")
        seen.add(goal.id)
        if goal.position is None:
            raise ValueError("candidate goal position must be available")
        _finite_xy(goal.position, "candidate goal position")
        if not goal.source:
            raise ValueError("candidate goal source must be non-empty")
    return goals


def _candidate_source(goals: Sequence[CandidateGoal]) -> str:
    """Return the single source name when all goals share it, otherwise 'mixed'."""
    sources = {goal.source for goal in goals}
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _normalize_prior(
    goals: Sequence[CandidateGoal],
    prior: Mapping[str, float] | None,
    config: GoalPosteriorConfig,
) -> dict[str, float]:
    """Normalize prior weights, defaulting to uniform when prior is None.

    Returns:
        Dict mapping goal id to normalized prior weight.
    """
    if prior is None:
        return {goal.id: 1.0 / len(goals) for goal in goals}

    goal_ids = {goal.id for goal in goals}
    unknown_ids = set(prior) - goal_ids
    if unknown_ids:
        raise ValueError(f"prior contains unknown candidate goal ids: {sorted(unknown_ids)}")

    weights: dict[str, float] = {}
    for goal in goals:
        value = require_non_negative(prior.get(goal.id, config.prior_floor), f"prior[{goal.id}]")
        weights[goal.id] = max(value, config.prior_floor)
    return _normalize_weights(weights, config.prior_floor)


def _normalize_weights(weights: Mapping[str, float], prior_floor: float) -> dict[str, float]:
    """Normalize weight dict to a probability distribution with a prior floor.

    Returns:
        Dict mapping goal id to normalized probability.
    """
    floored = {
        goal_id: max(require_non_negative(weight, f"weights[{goal_id}]"), prior_floor)
        for goal_id, weight in weights.items()
    }
    total = sum(floored.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("posterior normalization failed")
    probabilities = {goal_id: weight / total for goal_id, weight in floored.items()}
    if not all(math.isfinite(probability) for probability in probabilities.values()):
        raise ValueError("posterior probabilities must be finite")
    return probabilities
