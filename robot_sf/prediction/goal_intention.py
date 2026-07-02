"""Bayesian pedestrian goal-intention inference.

This module provides an interpretable, CPU-only posterior over explicit
candidate goal points from observed pedestrian motion. It is a planner input
interface, not a calibrated human-intention model or benchmark claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class CandidateGoal:
    """A candidate pedestrian goal point in world coordinates."""

    id: str
    position: tuple[float, float]
    source: str


@dataclass(frozen=True, slots=True)
class GoalPosteriorConfig:
    """Configuration for heading-based Bayesian goal updates."""

    heading_kappa: float = 4.0
    velocity_min_mps: float = 0.05
    prior_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Validate numeric configuration values."""

        if not math.isfinite(self.heading_kappa) or self.heading_kappa < 0.0:
            raise ValueError("heading_kappa must be finite and >= 0")
        if not math.isfinite(self.velocity_min_mps) or self.velocity_min_mps < 0.0:
            raise ValueError("velocity_min_mps must be finite and >= 0")
        if not math.isfinite(self.prior_floor) or self.prior_floor <= 0.0:
            raise ValueError("prior_floor must be finite and > 0")

    @property
    def config_hash(self) -> str:
        """Stable short hash for metadata provenance."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


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
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def top_goal_confidence(self) -> float | None:
        """Return the maximum posterior probability, or ``None`` when unavailable."""

        top_goal_id = self.top_goal_id
        if top_goal_id is None:
            return None
        return self.probabilities[top_goal_id]

    def as_planner_summary(self) -> dict[str, object]:
        """Return a JSON-serializable planner observation metadata summary."""

        return {
            "pedestrian_id": self.pedestrian_id,
            "candidate_source": self.candidate_source,
            "config_hash": self.config_hash,
            "top_goal_id": self.top_goal_id,
            "top_goal_confidence": self.top_goal_confidence,
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

    weighted: dict[str, float] = {}
    velocity_unit = (velocity[0] / speed, velocity[1] / speed)
    for goal in goals:
        to_goal = (goal.position[0] - position[0], goal.position[1] - position[1])
        distance = math.hypot(*to_goal)
        if distance == 0.0:
            alignment = 1.0
        else:
            alignment = (velocity_unit[0] * to_goal[0] + velocity_unit[1] * to_goal[1]) / distance
            alignment = max(-1.0, min(1.0, alignment))
        try:
            likelihood = math.exp(cfg.heading_kappa * alignment)
        except OverflowError as exc:
            raise ValueError("goal likelihood must be finite") from exc
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


def planner_goal_posterior_channel(
    posteriors: Sequence[GoalIntentionPosterior],
    *,
    enabled: bool,
) -> dict[str, object]:
    """Return the optional planner observation channel payload.

    Returns:
        JSON-serializable observation-channel payload. When disabled, the
        posterior map is intentionally empty.
    """

    if not enabled:
        return {"enabled": False, "pedestrian_goal_posteriors": {}}
    return {
        "enabled": True,
        "pedestrian_goal_posteriors": {
            posterior.pedestrian_id: posterior.as_planner_summary() for posterior in posteriors
        },
    }


def _finite_xy(
    value: tuple[float, float] | Sequence[float], field_name: str
) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{field_name} must contain finite floats")
    return (x, y)


def _validate_candidate_goals(
    candidate_goals: Sequence[CandidateGoal],
) -> tuple[CandidateGoal, ...]:
    goals = tuple(candidate_goals)
    if not goals:
        raise ValueError("candidate_goals must be non-empty")

    seen: set[str] = set()
    for goal in goals:
        if not goal.id:
            raise ValueError("candidate goal id must be non-empty")
        if goal.id in seen:
            raise ValueError(f"duplicate candidate goal id: {goal.id}")
        seen.add(goal.id)
        _finite_xy(goal.position, "candidate goal position")
        if not goal.source:
            raise ValueError("candidate goal source must be non-empty")
    return goals


def _candidate_source(goals: Sequence[CandidateGoal]) -> str:
    sources = {goal.source for goal in goals}
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _normalize_prior(
    goals: Sequence[CandidateGoal],
    prior: Mapping[str, float] | None,
    config: GoalPosteriorConfig,
) -> dict[str, float]:
    if prior is None:
        return {goal.id: 1.0 / len(goals) for goal in goals}

    goal_ids = {goal.id for goal in goals}
    unknown_ids = set(prior) - goal_ids
    if unknown_ids:
        raise ValueError(f"prior contains unknown candidate goal ids: {sorted(unknown_ids)}")

    weights: dict[str, float] = {}
    for goal in goals:
        value = float(prior.get(goal.id, config.prior_floor))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("prior weights must be finite and >= 0")
        weights[goal.id] = max(value, config.prior_floor)
    return _normalize_weights(weights, config.prior_floor)


def _normalize_weights(weights: Mapping[str, float], prior_floor: float) -> dict[str, float]:
    floored = {goal_id: max(float(weight), prior_floor) for goal_id, weight in weights.items()}
    total = sum(floored.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("posterior normalization failed")
    probabilities = {goal_id: weight / total for goal_id, weight in floored.items()}
    if not all(math.isfinite(probability) for probability in probabilities.values()):
        raise ValueError("posterior probabilities must be finite")
    return probabilities
