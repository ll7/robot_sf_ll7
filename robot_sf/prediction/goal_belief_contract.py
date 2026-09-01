"""Versioned actor-side goal-belief values for pedestrian research.

The contract is intentionally observation-only.  Oracle transition records live in
``oracle_transition_trace`` and cannot be passed to :meth:`GoalBeliefV1.from_observation`.
This keeps later estimator work honest about which values were available at a decision point.
"""

# The package supports Python 3.11, while the repository-wide Ruff target is
# Python 3.12. Keep the TypeVar spelling below until the minimum is raised.
# ruff: noqa: UP047

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any, TypeVar

from robot_sf.prediction._contract_utils import (
    canonical_json,
    reject_unknown_keys,
    require_covariance,
    require_digest,
    require_finite,
    require_probability,
    require_step_index,
    require_text,
    require_xy,
    stable_config_hash,
    stable_digest,
)

GOAL_BELIEF_SCHEMA_VERSION = "goal_belief.v1"
HISTORY_ORDER = "oldest_to_newest"
ACTOR_UNITS: dict[str, str] = {
    "position": "m",
    "velocity": "m/s",
    "force": "m/s^2",
    "force_covariance": "(m/s^2)^2",
    "time": "s",
    "angle": "rad",
}

ACTOR_FORBIDDEN_KEYS = frozenset(
    {
        "goal_before_behavior",
        "goal_after_behavior",
        "route_truth",
        "waypoint_truth",
        "simulator_pedestrian_id",
        "oracle_speed_cap_active",
        "speed_cap_truth",
        "uncapped_velocity_xy",
    }
)

EnumT = TypeVar("EnumT", bound=StrEnum)


class GoalBeliefSource(StrEnum):
    """Provenance of a belief; upper bounds are never actor observations."""

    OBSERVATION_ONLY = "observation_only"
    ORACLE_UPPER_BOUND = "oracle_upper_bound"
    SIMULATOR_UPPER_BOUND = "simulator_upper_bound"


class CoordinateFrame(StrEnum):
    """Coordinate frame used by positions, velocities, and force estimates."""

    GLOBAL_XY = "global_xy"


class GoalBeliefMode(StrEnum):
    """Availability state of a goal belief."""

    NOMINAL = "nominal"
    UNAVAILABLE = "unavailable"
    CENSORED = "censored"


class CensoringState(StrEnum):
    """Whether sensing or force saturation limits the payload."""

    NONE = "none"
    SATURATED = "saturated"
    CENSORED = "censored"
    UNKNOWN = "unknown"


class ActorSpeedCapStatus(StrEnum):
    """Actor-visible uncertainty about whether a hidden speed cap affected data."""

    CLEAR = "clear"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"


class ObservationMask(StrEnum):
    """Availability of one history row; slot position is never identity."""

    OBSERVED = "observed"
    INVISIBLE = "invisible"
    PADDED = "padded"


class GoalCandidateKind(StrEnum):
    """Semantic role of a candidate probability."""

    ACTIVE_WAYPOINT = "active_waypoint"
    FINAL_DESTINATION = "final_destination"


def _parse_enum(enum_type: type[EnumT], value: Any, field_name: str) -> EnumT:
    """Parse an external enum value with a field-specific failure.

    Returns:
        The parsed enum member.
    """
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Require a mapping for a nested external record.

    Returns:
        The validated mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _validate_history(
    history_steps: Sequence[ActorObservationStep],
    *,
    timestamp_s: float,
    step_index: int,
) -> tuple[ActorObservationStep, ...]:
    """Validate oldest-to-newest history and its decision-point endpoint.

    Returns:
        An immutable history tuple.
    """
    history = tuple(history_steps)
    if any(type(item) is not ActorObservationStep for item in history):
        raise TypeError("history_steps must contain ActorObservationStep values")
    for previous, current in pairwise(history):
        if current.step_index <= previous.step_index:
            raise ValueError("history_steps must use strictly increasing step_index values")
        if current.timestamp_s < previous.timestamp_s:
            raise ValueError("history_steps must use non-decreasing timestamp_s values")
    if history:
        latest = history[-1]
        if latest.step_index != step_index:
            raise ValueError("latest history step must match step_index")
        if not math.isclose(latest.timestamp_s, timestamp_s, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("latest history timestamp must match timestamp_s")
    return history


def _validate_candidates(
    candidates: Sequence[GoalCandidateProbability],
    unknown_candidate_probability: float,
) -> tuple[tuple[GoalCandidateProbability, ...], float]:
    """Validate candidate identity and normalization including unknown mass.

    Returns:
        Canonically ordered candidates and the validated unknown mass.
    """
    candidate_values = tuple(candidates)
    if any(type(item) is not GoalCandidateProbability for item in candidate_values):
        raise TypeError("candidate_probabilities must contain GoalCandidateProbability values")
    ids: set[str] = set()
    for candidate in candidate_values:
        if candidate.candidate_id in ids:
            raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
        ids.add(candidate.candidate_id)
    unknown = require_probability(unknown_candidate_probability, "unknown_candidate_probability")
    total = unknown + sum(candidate.probability for candidate in candidate_values)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("candidate probabilities plus unknown mass must sum to 1")
    return tuple(
        sorted(candidate_values, key=lambda item: (item.kind.value, item.candidate_id))
    ), unknown


def _validate_actor_values(  # noqa: C901, PLR0912, PLR0913
    *,
    timestamp_s: float,
    step_index: int,
    track_id: str,
    tracking_epoch_id: str,
    coordinate_frame: CoordinateFrame,
    history_steps: Sequence[ActorObservationStep],
    force_estimate: ForceEstimate2D | None,
    desired_velocity_xy: Sequence[float] | None,
    desired_direction_rad: float | None,
    candidate_probabilities: Sequence[GoalCandidateProbability],
    unknown_candidate_probability: float,
    arrival_probability: float,
    change_probability: float,
    mode: GoalBeliefMode,
    track_confidence: float | None,
    censoring_state: CensoringState,
    speed_cap_status: ActorSpeedCapStatus,
    blockers: Sequence[str],
    reset_provenance: str | None,
    config_hash: str,
) -> tuple[tuple[ActorObservationStep, ...], tuple[GoalCandidateProbability, ...], tuple[str, ...]]:
    """Validate fields shared by typed actor observations and serialized beliefs.

    Returns:
        Normalized history, candidates, and blocker tuples.
    """
    timestamp = require_finite(timestamp_s, "timestamp_s")
    step = require_step_index(step_index, "step_index")
    del timestamp, step
    require_text(track_id, "track_id")
    require_text(tracking_epoch_id, "tracking_epoch_id")
    if not isinstance(coordinate_frame, CoordinateFrame):
        raise TypeError("coordinate_frame must be CoordinateFrame")
    if coordinate_frame is not CoordinateFrame.GLOBAL_XY:
        raise ValueError("unsupported coordinate_frame")
    history = _validate_history(history_steps, timestamp_s=timestamp_s, step_index=step_index)
    if force_estimate is not None and type(force_estimate) is not ForceEstimate2D:
        raise TypeError("force_estimate must be ForceEstimate2D or None")
    if desired_velocity_xy is not None:
        desired_velocity = require_xy(desired_velocity_xy, "desired_velocity_xy")
        del desired_velocity
    if desired_direction_rad is not None:
        direction = require_finite(desired_direction_rad, "desired_direction_rad")
        if not -math.pi <= direction <= math.pi:
            raise ValueError("desired_direction_rad must be between -pi and pi")
    candidates, unknown = _validate_candidates(
        candidate_probabilities, unknown_candidate_probability
    )
    del unknown
    require_probability(arrival_probability, "arrival_probability")
    require_probability(change_probability, "change_probability")
    if not isinstance(mode, GoalBeliefMode):
        raise TypeError("mode must be GoalBeliefMode")
    if track_confidence is not None:
        require_probability(track_confidence, "track_confidence")
    if not isinstance(censoring_state, CensoringState):
        raise TypeError("censoring_state must be CensoringState")
    if not isinstance(speed_cap_status, ActorSpeedCapStatus):
        raise TypeError("speed_cap_status must be ActorSpeedCapStatus")
    blocker_values = tuple(require_text(blocker, "blockers[]") for blocker in blockers)
    if len(set(blocker_values)) != len(blocker_values):
        raise ValueError("blockers must be unique")
    blocker_values = tuple(sorted(blocker_values))
    if reset_provenance is not None:
        require_text(reset_provenance, "reset_provenance")
    require_digest(config_hash, "config_hash")
    if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
        raise TypeError("blockers must be an array")

    if mode is GoalBeliefMode.UNAVAILABLE:
        if not blocker_values:
            raise ValueError("unavailable beliefs must name at least one blocker")
        if force_estimate is not None or desired_velocity_xy is not None:
            raise ValueError("unavailable beliefs must not carry force or desired velocity")
        if track_confidence is not None:
            raise ValueError("unavailable beliefs must not carry track confidence")
        if candidates or not math.isclose(unknown_candidate_probability, 1.0):
            raise ValueError("unavailable beliefs must assign all candidate mass to unknown")
    elif mode is GoalBeliefMode.NOMINAL:
        if force_estimate is None or desired_velocity_xy is None or track_confidence is None:
            raise ValueError("nominal beliefs require force, desired velocity, and confidence")
    elif censoring_state is CensoringState.NONE:
        raise ValueError("censored beliefs must declare a censoring state")

    return history, candidates, blocker_values


@dataclass(frozen=True, slots=True)
class ForceEstimate2D:
    """Finite two-dimensional force estimate and strictly positive covariance."""

    mean_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        """Normalize and validate the estimate without admitting NaN values."""
        object.__setattr__(self, "mean_xy", require_xy(self.mean_xy, "force.mean_xy"))
        object.__setattr__(
            self,
            "covariance_xy",
            require_covariance(self.covariance_xy, "force.covariance_xy"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the actor-safe JSON shape."""
        return {
            "mean_xy": list(self.mean_xy),
            "covariance_xy": [list(row) for row in self.covariance_xy],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ForceEstimate2D:
        """Parse a strict force estimate record.

        Returns:
            A validated force estimate.
        """
        reject_unknown_keys(value, {"mean_xy", "covariance_xy"}, "force_estimate")
        if set(value) != {"mean_xy", "covariance_xy"}:
            raise ValueError("force_estimate must contain mean_xy and covariance_xy")
        return cls(
            mean_xy=require_xy(value["mean_xy"], "force_estimate.mean_xy"),
            covariance_xy=require_covariance(
                value["covariance_xy"], "force_estimate.covariance_xy"
            ),
        )


@dataclass(frozen=True, slots=True)
class ActorObservationStep:
    """One observation-history row with explicit visibility/padding semantics."""

    timestamp_s: float
    step_index: int
    position_xy: tuple[float, float] | None
    velocity_xy: tuple[float, float] | None
    mask: ObservationMask = ObservationMask.OBSERVED

    def __post_init__(self) -> None:
        """Reject future-like or ambiguous history values at construction time."""
        object.__setattr__(self, "timestamp_s", require_finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "step_index", require_step_index(self.step_index, "step_index"))
        if not isinstance(self.mask, ObservationMask):
            raise TypeError("mask must be ObservationMask")
        position = None if self.position_xy is None else require_xy(self.position_xy, "position_xy")
        velocity = None if self.velocity_xy is None else require_xy(self.velocity_xy, "velocity_xy")
        if self.mask is ObservationMask.OBSERVED and (position is None or velocity is None):
            raise ValueError("observed history rows require position and velocity")
        if self.mask is not ObservationMask.OBSERVED and (
            position is not None or velocity is not None
        ):
            raise ValueError("invisible and padded history rows must omit position and velocity")
        object.__setattr__(self, "position_xy", position)
        object.__setattr__(self, "velocity_xy", velocity)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe history row."""
        return {
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "position_xy": list(self.position_xy) if self.position_xy is not None else None,
            "velocity_xy": list(self.velocity_xy) if self.velocity_xy is not None else None,
            "mask": self.mask.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActorObservationStep:
        """Parse a strict history row.

        Returns:
            A validated actor history row.
        """
        allowed = {"timestamp_s", "step_index", "position_xy", "velocity_xy", "mask"}
        reject_unknown_keys(value, allowed, "history_step")
        if set(value) != allowed:
            raise ValueError("history_step is missing a required field")
        return cls(
            timestamp_s=value["timestamp_s"],
            step_index=value["step_index"],
            position_xy=None
            if value["position_xy"] is None
            else require_xy(value["position_xy"], "history_step.position_xy"),
            velocity_xy=None
            if value["velocity_xy"] is None
            else require_xy(value["velocity_xy"], "history_step.velocity_xy"),
            mask=_parse_enum(ObservationMask, value["mask"], "history_step.mask"),
        )


@dataclass(frozen=True, slots=True)
class GoalCandidateProbability:
    """Probability mass for one active-waypoint or final-destination candidate."""

    candidate_id: str
    kind: GoalCandidateKind
    probability: float

    def __post_init__(self) -> None:
        """Validate candidate identity and probability mass."""
        object.__setattr__(self, "candidate_id", require_text(self.candidate_id, "candidate_id"))
        if not isinstance(self.kind, GoalCandidateKind):
            raise TypeError("kind must be GoalCandidateKind")
        object.__setattr__(
            self, "probability", require_probability(self.probability, "probability")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe candidate probability."""
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoalCandidateProbability:
        """Parse a strict candidate probability.

        Returns:
            A validated candidate probability.
        """
        allowed = {"candidate_id", "kind", "probability"}
        reject_unknown_keys(value, allowed, "candidate_probability")
        if set(value) != allowed:
            raise ValueError("candidate_probability is missing a required field")
        return cls(
            candidate_id=value["candidate_id"],
            kind=_parse_enum(GoalCandidateKind, value["kind"], "candidate_probability.kind"),
            probability=value["probability"],
        )


@dataclass(frozen=True, slots=True)
class GoalBeliefObservation:
    """Narrow actor observation/history input accepted by ``GoalBeliefV1``."""

    track_id: str
    tracking_epoch_id: str
    timestamp_s: float
    step_index: int
    config_hash: str
    history_steps: tuple[ActorObservationStep, ...] = ()
    coordinate_frame: CoordinateFrame = CoordinateFrame.GLOBAL_XY
    force_estimate: ForceEstimate2D | None = None
    desired_velocity_xy: tuple[float, float] | None = None
    desired_direction_rad: float | None = None
    candidate_probabilities: tuple[GoalCandidateProbability, ...] = ()
    unknown_candidate_probability: float = 1.0
    arrival_probability: float = 0.0
    change_probability: float = 0.0
    mode: GoalBeliefMode = GoalBeliefMode.NOMINAL
    track_confidence: float | None = None
    censoring_state: CensoringState = CensoringState.NONE
    speed_cap_status: ActorSpeedCapStatus = ActorSpeedCapStatus.UNKNOWN
    blockers: tuple[str, ...] = ()
    reset_provenance: str | None = None

    def __post_init__(self) -> None:
        """Validate the typed observation before it can become an actor belief."""
        history, candidates, blockers = _validate_actor_values(
            timestamp_s=self.timestamp_s,
            step_index=self.step_index,
            track_id=self.track_id,
            tracking_epoch_id=self.tracking_epoch_id,
            coordinate_frame=self.coordinate_frame,
            history_steps=self.history_steps,
            force_estimate=self.force_estimate,
            desired_velocity_xy=self.desired_velocity_xy,
            desired_direction_rad=self.desired_direction_rad,
            candidate_probabilities=self.candidate_probabilities,
            unknown_candidate_probability=self.unknown_candidate_probability,
            arrival_probability=self.arrival_probability,
            change_probability=self.change_probability,
            mode=self.mode,
            track_confidence=self.track_confidence,
            censoring_state=self.censoring_state,
            speed_cap_status=self.speed_cap_status,
            blockers=self.blockers,
            reset_provenance=self.reset_provenance,
            config_hash=self.config_hash,
        )
        object.__setattr__(self, "timestamp_s", require_finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "step_index", require_step_index(self.step_index, "step_index"))
        object.__setattr__(self, "track_id", require_text(self.track_id, "track_id"))
        object.__setattr__(
            self, "tracking_epoch_id", require_text(self.tracking_epoch_id, "tracking_epoch_id")
        )
        object.__setattr__(self, "config_hash", require_digest(self.config_hash, "config_hash"))
        object.__setattr__(self, "history_steps", history)
        object.__setattr__(self, "candidate_probabilities", candidates)
        object.__setattr__(self, "blockers", blockers)
        if self.desired_velocity_xy is not None:
            object.__setattr__(
                self,
                "desired_velocity_xy",
                require_xy(self.desired_velocity_xy, "desired_velocity_xy"),
            )


@dataclass(frozen=True, slots=True)
class GoalBeliefV1:
    """Immutable actor-side goal belief with no oracle-only fields."""

    timestamp_s: float
    step_index: int
    track_id: str
    tracking_epoch_id: str
    source: GoalBeliefSource
    coordinate_frame: CoordinateFrame
    history_steps: tuple[ActorObservationStep, ...]
    force_estimate: ForceEstimate2D | None
    desired_velocity_xy: tuple[float, float] | None
    desired_direction_rad: float | None
    candidate_probabilities: tuple[GoalCandidateProbability, ...]
    unknown_candidate_probability: float
    arrival_probability: float
    change_probability: float
    mode: GoalBeliefMode
    track_confidence: float | None
    censoring_state: CensoringState
    speed_cap_status: ActorSpeedCapStatus
    blockers: tuple[str, ...]
    reset_provenance: str | None
    config_hash: str
    schema_version: str = field(default=GOAL_BELIEF_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Validate and normalize the complete actor contract."""
        if self.schema_version != GOAL_BELIEF_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {GOAL_BELIEF_SCHEMA_VERSION}")
        history, candidates, blockers = _validate_actor_values(
            timestamp_s=self.timestamp_s,
            step_index=self.step_index,
            track_id=self.track_id,
            tracking_epoch_id=self.tracking_epoch_id,
            coordinate_frame=self.coordinate_frame,
            history_steps=self.history_steps,
            force_estimate=self.force_estimate,
            desired_velocity_xy=self.desired_velocity_xy,
            desired_direction_rad=self.desired_direction_rad,
            candidate_probabilities=self.candidate_probabilities,
            unknown_candidate_probability=self.unknown_candidate_probability,
            arrival_probability=self.arrival_probability,
            change_probability=self.change_probability,
            mode=self.mode,
            track_confidence=self.track_confidence,
            censoring_state=self.censoring_state,
            speed_cap_status=self.speed_cap_status,
            blockers=self.blockers,
            reset_provenance=self.reset_provenance,
            config_hash=self.config_hash,
        )
        if not isinstance(self.source, GoalBeliefSource):
            raise TypeError("source must be GoalBeliefSource")
        object.__setattr__(self, "timestamp_s", require_finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "step_index", require_step_index(self.step_index, "step_index"))
        object.__setattr__(self, "track_id", require_text(self.track_id, "track_id"))
        object.__setattr__(
            self, "tracking_epoch_id", require_text(self.tracking_epoch_id, "tracking_epoch_id")
        )
        object.__setattr__(self, "history_steps", history)
        object.__setattr__(self, "candidate_probabilities", candidates)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "config_hash", require_digest(self.config_hash, "config_hash"))
        if self.desired_velocity_xy is not None:
            object.__setattr__(
                self,
                "desired_velocity_xy",
                require_xy(self.desired_velocity_xy, "desired_velocity_xy"),
            )

    @classmethod
    def from_observation(
        cls,
        observation: GoalBeliefObservation,
        *,
        current_timestamp_s: float | None = None,
        current_step_index: int | None = None,
    ) -> GoalBeliefV1:
        """Construct an actor belief from a narrowed, non-privileged observation type.

        Returns:
            An observation-only goal belief at the requested decision point.
        """
        if type(observation) is not GoalBeliefObservation:
            raise TypeError("actor construction requires a GoalBeliefObservation value")
        decision_time = (
            observation.timestamp_s
            if current_timestamp_s is None
            else require_finite(current_timestamp_s, "current_timestamp_s")
        )
        decision_step = (
            observation.step_index
            if current_step_index is None
            else require_step_index(current_step_index, "current_step_index")
        )
        if observation.timestamp_s > decision_time + 1e-12:
            raise ValueError("observation timestamp is newer than the decision point")
        if observation.step_index > decision_step:
            raise ValueError("observation step is newer than the decision point")
        if any(
            step.timestamp_s > decision_time + 1e-12 or step.step_index > decision_step
            for step in observation.history_steps
        ):
            raise ValueError("history contains a value newer than the decision point")
        return cls(
            timestamp_s=observation.timestamp_s,
            step_index=observation.step_index,
            track_id=observation.track_id,
            tracking_epoch_id=observation.tracking_epoch_id,
            source=GoalBeliefSource.OBSERVATION_ONLY,
            coordinate_frame=observation.coordinate_frame,
            history_steps=observation.history_steps,
            force_estimate=observation.force_estimate,
            desired_velocity_xy=observation.desired_velocity_xy,
            desired_direction_rad=observation.desired_direction_rad,
            candidate_probabilities=observation.candidate_probabilities,
            unknown_candidate_probability=observation.unknown_candidate_probability,
            arrival_probability=observation.arrival_probability,
            change_probability=observation.change_probability,
            mode=observation.mode,
            track_confidence=observation.track_confidence,
            censoring_state=observation.censoring_state,
            speed_cap_status=observation.speed_cap_status,
            blockers=observation.blockers,
            reset_provenance=observation.reset_provenance,
            config_hash=observation.config_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the actor-only JSON payload; oracle fields are not representable here."""
        return {
            "schema_version": self.schema_version,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "track_id": self.track_id,
            "tracking_epoch_id": self.tracking_epoch_id,
            "source": self.source.value,
            "coordinate_frame": self.coordinate_frame.value,
            "units": dict(ACTOR_UNITS),
            "history_order": HISTORY_ORDER,
            "history_steps": [step.to_dict() for step in self.history_steps],
            "force_estimate": self.force_estimate.to_dict() if self.force_estimate else None,
            "desired_velocity_xy": list(self.desired_velocity_xy)
            if self.desired_velocity_xy is not None
            else None,
            "desired_direction_rad": self.desired_direction_rad,
            "candidate_probabilities": [
                candidate.to_dict() for candidate in self.candidate_probabilities
            ],
            "unknown_candidate_probability": self.unknown_candidate_probability,
            "arrival_probability": self.arrival_probability,
            "change_probability": self.change_probability,
            "mode": self.mode.value,
            "track_confidence": self.track_confidence,
            "censoring_state": self.censoring_state.value,
            "speed_cap_status": self.speed_cap_status.value,
            "blockers": list(self.blockers),
            "reset_provenance": self.reset_provenance,
            "config_hash": self.config_hash,
        }

    def to_actor_model_features(self) -> dict[str, Any]:
        """Return model features only when the belief is observation-derived."""
        if self.source is not GoalBeliefSource.OBSERVATION_ONLY:
            raise ValueError("actor model features require source=observation_only")
        payload = self.to_dict()
        return {
            key: payload[key]
            for key in (
                "track_id",
                "tracking_epoch_id",
                "coordinate_frame",
                "units",
                "history_order",
                "history_steps",
                "force_estimate",
                "desired_velocity_xy",
                "desired_direction_rad",
                "candidate_probabilities",
                "unknown_candidate_probability",
                "arrival_probability",
                "change_probability",
                "mode",
                "track_confidence",
                "censoring_state",
                "speed_cap_status",
            )
        }

    def to_model_features(self) -> dict[str, Any]:
        """Return observation-only features through the legacy method name."""
        return self.to_actor_model_features()

    def to_json(self) -> str:
        """Return RFC 8785 canonical actor JSON."""
        return canonical_json(self.to_dict())

    @property
    def content_digest(self) -> str:
        """Return the deterministic SHA-256 digest of the actor payload."""
        return stable_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoalBeliefV1:
        """Parse a strict actor payload and reject unknown versioned keys.

        Returns:
            A validated actor-side goal belief.
        """
        allowed = {
            "schema_version",
            "timestamp_s",
            "step_index",
            "track_id",
            "tracking_epoch_id",
            "source",
            "coordinate_frame",
            "units",
            "history_order",
            "history_steps",
            "force_estimate",
            "desired_velocity_xy",
            "desired_direction_rad",
            "candidate_probabilities",
            "unknown_candidate_probability",
            "arrival_probability",
            "change_probability",
            "mode",
            "track_confidence",
            "censoring_state",
            "speed_cap_status",
            "blockers",
            "reset_provenance",
            "config_hash",
        }
        reject_unknown_keys(value, allowed, "goal_belief")
        if set(value) != allowed:
            raise ValueError("goal_belief is missing a required field")
        if value["units"] != ACTOR_UNITS:
            raise ValueError("goal_belief.units does not match the v1 unit contract")
        if value["history_order"] != HISTORY_ORDER:
            raise ValueError("goal_belief.history_order must be oldest_to_newest")
        history_raw = value["history_steps"]
        candidates_raw = value["candidate_probabilities"]
        if not isinstance(history_raw, Sequence) or isinstance(history_raw, (str, bytes)):
            raise TypeError("goal_belief.history_steps must be an array")
        if not isinstance(candidates_raw, Sequence) or isinstance(candidates_raw, (str, bytes)):
            raise TypeError("goal_belief.candidate_probabilities must be an array")
        blockers_raw = value["blockers"]
        if not isinstance(blockers_raw, Sequence) or isinstance(blockers_raw, (str, bytes)):
            raise TypeError("goal_belief.blockers must be an array")
        history = tuple(
            ActorObservationStep.from_dict(_mapping(item, "history_step")) for item in history_raw
        )
        candidates = tuple(
            GoalCandidateProbability.from_dict(_mapping(item, "candidate_probability"))
            for item in candidates_raw
        )
        force_value = value["force_estimate"]
        force = (
            None
            if force_value is None
            else ForceEstimate2D.from_dict(_mapping(force_value, "force_estimate"))
        )
        desired_velocity_value = value["desired_velocity_xy"]
        return cls(
            schema_version=value["schema_version"],
            timestamp_s=value["timestamp_s"],
            step_index=value["step_index"],
            track_id=value["track_id"],
            tracking_epoch_id=value["tracking_epoch_id"],
            source=_parse_enum(GoalBeliefSource, value["source"], "source"),
            coordinate_frame=_parse_enum(
                CoordinateFrame, value["coordinate_frame"], "coordinate_frame"
            ),
            history_steps=history,
            force_estimate=force,
            desired_velocity_xy=None
            if desired_velocity_value is None
            else require_xy(desired_velocity_value, "desired_velocity_xy"),
            desired_direction_rad=value["desired_direction_rad"],
            candidate_probabilities=candidates,
            unknown_candidate_probability=value["unknown_candidate_probability"],
            arrival_probability=value["arrival_probability"],
            change_probability=value["change_probability"],
            mode=_parse_enum(GoalBeliefMode, value["mode"], "mode"),
            track_confidence=value["track_confidence"],
            censoring_state=_parse_enum(
                CensoringState, value["censoring_state"], "censoring_state"
            ),
            speed_cap_status=_parse_enum(
                ActorSpeedCapStatus, value["speed_cap_status"], "speed_cap_status"
            ),
            blockers=tuple(blockers_raw),
            reset_provenance=value["reset_provenance"],
            config_hash=value["config_hash"],
        )


__all__ = [
    "ACTOR_FORBIDDEN_KEYS",
    "ACTOR_UNITS",
    "GOAL_BELIEF_SCHEMA_VERSION",
    "HISTORY_ORDER",
    "ActorObservationStep",
    "ActorSpeedCapStatus",
    "CensoringState",
    "CoordinateFrame",
    "ForceEstimate2D",
    "GoalBeliefMode",
    "GoalBeliefObservation",
    "GoalBeliefSource",
    "GoalBeliefV1",
    "GoalCandidateKind",
    "GoalCandidateProbability",
    "ObservationMask",
    "stable_config_hash",
]
