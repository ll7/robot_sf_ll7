"""Additive actor-only hierarchical GoalBeliefV1 contract.

This module defines the contract slice for issue #8075.  It stores a destination
posterior and an independently normalized conditional waypoint posterior without
changing the flat ``GoalBeliefV1`` wire format.  Temporal updates, candidate
generation, lifecycle transitions, and planner integration deliberately remain
outside this contract owner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robot_sf.prediction._contract_utils import (
    canonical_json,
    is_forbidden_evidence_source,
    reject_unknown_keys,
    require_digest,
    require_finite,
    require_non_negative,
    require_probability,
    require_step_index,
    require_text,
    stable_digest,
)
from robot_sf.prediction.goal_belief_contract import (
    ActorSpeedCapStatus,
    CensoringState,
    CoordinateFrame,
    GoalBeliefMode,
    GoalBeliefSource,
    GoalBeliefV1,
    GoalCandidateKind,
    GoalCandidateProbability,
)

HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION = "hierarchical_goal_posterior.v1"
HIERARCHICAL_PROJECTION_LEVELS = ("active_waypoint", "final_destination")
_ACTOR_EVIDENCE_SOURCES = frozenset({"upstream_selected"})
_UNKNOWN_CANDIDATE_ID = "unknown"


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    """Return a non-string sequence as an immutable tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be an array")
    return tuple(value)


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a mapping for one external record.

    Returns:
        The validated mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _validate_probability_vector(
    values: Sequence[HierarchicalProbability],
    unknown: float,
    field_name: str,
) -> tuple[tuple[HierarchicalProbability, ...], float]:
    """Validate identity, finiteness, and normalization for one level.

    Returns:
        The candidates in deterministic identifier order and the canonical unknown mass.
    """
    vector = tuple(values)
    if any(type(value) is not HierarchicalProbability for value in vector):
        raise TypeError(f"{field_name} must contain HierarchicalProbability values")
    identifiers = [value.candidate_id for value in vector]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{field_name} contains duplicate candidate IDs")
    unknown_probability = require_probability(unknown, f"{field_name}.unknown_probability")
    total = math.fsum((unknown_probability, *(value.probability for value in vector)))
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{field_name} probabilities plus unknown mass must sum to 1")
    if total == 1.0:
        return tuple(sorted(vector, key=lambda value: value.candidate_id)), unknown_probability
    normalized = tuple(
        HierarchicalProbability(value.candidate_id, value.probability / total) for value in vector
    )
    return (
        tuple(sorted(normalized, key=lambda value: value.candidate_id)),
        unknown_probability / total,
    )


def _normalize_parent_map(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Normalize a waypoint-to-destination parent mapping.

    Returns:
        The parent pairs in deterministic waypoint-identifier order.
    """
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    else:
        pairs = _as_sequence(value, "waypoint_parent_destination")
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pair in pairs:
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise TypeError(
                "waypoint_parent_destination entries must be [waypoint_id, destination_id]"
            )
        waypoint_id = require_text(pair[0], "waypoint_parent_destination.waypoint_id")
        destination_id = require_text(pair[1], "waypoint_parent_destination.destination_id")
        if waypoint_id in seen:
            raise ValueError(f"duplicate waypoint parent mapping: {waypoint_id}")
        seen.add(waypoint_id)
        normalized.append((waypoint_id, destination_id))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class HierarchicalProbability:
    """Probability mass for one stable destination or waypoint identifier."""

    candidate_id: str
    probability: float

    def __post_init__(self) -> None:
        """Validate identity and finite probability mass."""
        object.__setattr__(self, "candidate_id", require_text(self.candidate_id, "candidate_id"))
        if self.candidate_id.strip().lower() == _UNKNOWN_CANDIDATE_ID:
            raise ValueError(
                "candidate_id 'unknown' is reserved for the explicit unknown probability mass"
            )
        object.__setattr__(
            self,
            "probability",
            require_probability(self.probability, "probability"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON shape."""
        return {"candidate_id": self.candidate_id, "probability": self.probability}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HierarchicalProbability:
        """Parse one strict probability entry.

        Returns:
            A validated probability entry.
        """
        value = _as_mapping(value, "hierarchical_probability")
        allowed = {"candidate_id", "probability"}
        reject_unknown_keys(value, allowed, "hierarchical_probability")
        if set(value) != allowed:
            raise ValueError("hierarchical_probability is missing a required field")
        return cls(candidate_id=value["candidate_id"], probability=value["probability"])


@dataclass(frozen=True, slots=True)
class HierarchicalWaypointConditionalV1:
    """Independently normalized waypoint posterior for one destination."""

    destination_id: str
    waypoint_probabilities: Sequence[HierarchicalProbability] = ()
    unknown_waypoint_probability: float = 1.0

    def __post_init__(self) -> None:
        """Validate the conditional vector and canonicalize its order."""
        object.__setattr__(
            self,
            "destination_id",
            require_text(self.destination_id, "destination_id"),
        )
        probabilities, unknown_probability = _validate_probability_vector(
            _as_sequence(self.waypoint_probabilities, "waypoint_probabilities"),
            self.unknown_waypoint_probability,
            f"waypoint_conditionals[{self.destination_id}]",
        )
        object.__setattr__(self, "waypoint_probabilities", probabilities)
        object.__setattr__(
            self,
            "unknown_waypoint_probability",
            require_probability(
                unknown_probability,
                "unknown_waypoint_probability",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON shape."""
        return {
            "destination_id": self.destination_id,
            "waypoint_probabilities": [
                probability.to_dict() for probability in self.waypoint_probabilities
            ],
            "unknown_waypoint_probability": self.unknown_waypoint_probability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HierarchicalWaypointConditionalV1:
        """Parse one strict conditional vector.

        Returns:
            A validated conditional waypoint distribution.
        """
        value = _as_mapping(value, "waypoint_conditional")
        allowed = {
            "destination_id",
            "waypoint_probabilities",
            "unknown_waypoint_probability",
        }
        reject_unknown_keys(value, allowed, "waypoint_conditional")
        if set(value) != allowed:
            raise ValueError("waypoint_conditional is missing a required field")
        probabilities = tuple(
            HierarchicalProbability.from_dict(item)
            for item in _as_sequence(value["waypoint_probabilities"], "waypoint_probabilities")
        )
        return cls(
            destination_id=value["destination_id"],
            waypoint_probabilities=probabilities,
            unknown_waypoint_probability=value["unknown_waypoint_probability"],
        )


@dataclass(frozen=True, slots=True)
class HierarchicalGoalPosteriorV1:
    """Versioned actor-only destination/conditional-waypoint posterior.

    The two levels are normalized separately.  A caller must explicitly choose a
    level when requesting the legacy ``GoalBeliefV1`` projection.
    """

    track_id: str
    tracking_epoch_id: str
    timestamp_s: float
    step_index: int
    destination_probabilities: Sequence[HierarchicalProbability]
    unknown_destination_probability: float
    waypoint_conditionals: Sequence[HierarchicalWaypointConditionalV1]
    config_hash: str
    candidate_set_digest: str
    waypoint_parent_destination: Mapping[str, str] | Sequence[tuple[str, str]] = field(
        default_factory=dict
    )
    evidence_source: str = "upstream_selected"
    innovation: float = 0.0
    blockers: Sequence[str] = ()
    schema_version: str = HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:  # noqa: C901
        """Validate the additive hierarchy contract and canonicalize containers."""
        if self.schema_version != HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION:
            raise ValueError("schema_version must be " + HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION)
        object.__setattr__(self, "track_id", require_text(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "tracking_epoch_id",
            require_text(self.tracking_epoch_id, "tracking_epoch_id"),
        )
        timestamp = require_finite(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(
            self,
            "step_index",
            require_step_index(self.step_index, "step_index"),
        )
        object.__setattr__(self, "config_hash", require_digest(self.config_hash, "config_hash"))
        object.__setattr__(
            self,
            "candidate_set_digest",
            require_digest(self.candidate_set_digest, "candidate_set_digest"),
        )
        evidence_source = require_text(self.evidence_source, "evidence_source").strip().lower()
        if (
            is_forbidden_evidence_source(evidence_source)
            or evidence_source not in _ACTOR_EVIDENCE_SOURCES
        ):
            allowed = ", ".join(sorted(_ACTOR_EVIDENCE_SOURCES))
            raise ValueError(f"evidence_source must be an actor-safe source: {allowed}")
        object.__setattr__(self, "evidence_source", evidence_source)
        object.__setattr__(
            self,
            "innovation",
            require_non_negative(self.innovation, "innovation"),
        )

        blocker_values = tuple(
            require_text(blocker, "blockers[]")
            for blocker in _as_sequence(self.blockers, "blockers")
        )
        if len(blocker_values) != len(set(blocker_values)):
            raise ValueError("blockers must be unique")
        object.__setattr__(self, "blockers", tuple(sorted(blocker_values)))

        destinations, unknown_destination_probability = _validate_probability_vector(
            _as_sequence(self.destination_probabilities, "destination_probabilities"),
            self.unknown_destination_probability,
            "destination_probabilities",
        )
        object.__setattr__(self, "destination_probabilities", destinations)
        object.__setattr__(
            self,
            "unknown_destination_probability",
            require_probability(
                unknown_destination_probability,
                "unknown_destination_probability",
            ),
        )

        conditionals = _as_sequence(self.waypoint_conditionals, "waypoint_conditionals")
        if any(type(value) is not HierarchicalWaypointConditionalV1 for value in conditionals):
            raise TypeError(
                "waypoint_conditionals must contain HierarchicalWaypointConditionalV1 values"
            )
        conditional_ids = [value.destination_id for value in conditionals]
        if len(conditional_ids) != len(set(conditional_ids)):
            raise ValueError("waypoint_conditionals contains duplicate destination IDs")
        destination_ids = {value.candidate_id for value in destinations}
        if set(conditional_ids) != destination_ids:
            raise ValueError(
                "waypoint_conditionals must name exactly the destination candidate IDs"
            )
        conditionals = tuple(sorted(conditionals, key=lambda value: value.destination_id))
        object.__setattr__(self, "waypoint_conditionals", conditionals)

        parents = _normalize_parent_map(self.waypoint_parent_destination)
        parent_by_waypoint = dict(parents)
        waypoint_ids: set[str] = set()
        for conditional in conditionals:
            for waypoint in conditional.waypoint_probabilities:
                if waypoint.candidate_id in waypoint_ids:
                    raise ValueError(f"duplicate waypoint ID: {waypoint.candidate_id}")
                waypoint_ids.add(waypoint.candidate_id)
                if parent_by_waypoint.get(waypoint.candidate_id) != conditional.destination_id:
                    raise ValueError(
                        f"waypoint {waypoint.candidate_id} has a missing or incorrect parent"
                    )
        if set(parent_by_waypoint) != waypoint_ids:
            raise ValueError("waypoint_parent_destination must cover every waypoint exactly")
        object.__setattr__(self, "waypoint_parent_destination", parents)

    @property
    def state_digest(self) -> str:
        """Return the digest of state fields without the self-referential digest."""
        return stable_digest(self._payload_without_state_digest())

    @property
    def content_digest(self) -> str:
        """Return the digest of the complete canonical payload."""
        return stable_digest(self.to_dict())

    def _payload_without_state_digest(self) -> dict[str, Any]:
        """Build the canonical payload used to derive ``state_digest``.

        Returns:
            The canonical state fields excluding the derived digest.
        """
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "tracking_epoch_id": self.tracking_epoch_id,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "destination_probabilities": [
                value.to_dict() for value in self.destination_probabilities
            ],
            "unknown_destination_probability": self.unknown_destination_probability,
            "waypoint_conditionals": [value.to_dict() for value in self.waypoint_conditionals],
            "waypoint_parent_destination": dict(self.waypoint_parent_destination),
            "evidence_source": self.evidence_source,
            "innovation": self.innovation,
            "blockers": list(self.blockers),
            "config_hash": self.config_hash,
            "candidate_set_digest": self.candidate_set_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a strict actor-only JSON payload."""
        payload = self._payload_without_state_digest()
        payload["state_digest"] = self.state_digest
        return payload

    def to_json(self) -> str:
        """Return RFC 8785 canonical JSON."""
        return canonical_json(self.to_dict())

    def active_waypoint_marginal(
        self,
    ) -> tuple[tuple[HierarchicalProbability, ...], float]:
        """Derive the active-waypoint marginal and its aggregate unknown mass.

        Returns:
            The marginal waypoint probabilities and combined unknown mass.
        """
        conditional_by_destination = {
            value.destination_id: value for value in self.waypoint_conditionals
        }
        # Keep every product as a separate term so cardinality-dependent
        # accumulation uses compensated summation instead of order-sensitive
        # incremental addition.
        masses: dict[str, list[float]] = {}
        unknown_terms = [self.unknown_destination_probability]
        for destination in self.destination_probabilities:
            conditional = conditional_by_destination[destination.candidate_id]
            for waypoint in conditional.waypoint_probabilities:
                masses.setdefault(waypoint.candidate_id, []).append(
                    destination.probability * waypoint.probability
                )
            unknown_terms.append(destination.probability * conditional.unknown_waypoint_probability)

        # Provider-bound vectors are independently normalized within a small
        # tolerance. Normalize the derived vector once to absorb their
        # floating-point drift while preserving its relative masses.
        canonical_masses = {
            candidate_id: math.fsum(contributions)
            for candidate_id, contributions in sorted(masses.items())
        }
        unknown = math.fsum(unknown_terms)
        total = math.fsum((*canonical_masses.values(), unknown))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("active waypoint marginal has an invalid probability total")
        scale = 1.0 / total
        probabilities = tuple(
            HierarchicalProbability(candidate_id, min(1.0, probability * scale))
            for candidate_id, probability in canonical_masses.items()
        )
        return probabilities, min(1.0, unknown * scale)

    def to_goal_belief_v1(self, level: str) -> GoalBeliefV1:
        """Project exactly one hierarchy level into the unchanged flat v1 type.

        Returns:
            A flat actor-only belief for the explicitly requested level.
        """
        if level not in HIERARCHICAL_PROJECTION_LEVELS:
            raise ValueError("level must be one of: " + ", ".join(HIERARCHICAL_PROJECTION_LEVELS))

        if level == "final_destination":
            values = self.destination_probabilities
            unknown = self.unknown_destination_probability
            kind = GoalCandidateKind.FINAL_DESTINATION
        else:
            values, unknown = self.active_waypoint_marginal()
            kind = GoalCandidateKind.ACTIVE_WAYPOINT
        candidates = tuple(
            GoalCandidateProbability(value.candidate_id, kind, value.probability)
            for value in values
        )
        blockers = tuple(
            sorted(
                set(self.blockers)
                | {
                    "arrival_probability_unestimated",
                    "change_probability_unestimated",
                    "hierarchical_projection",
                }
            )
        )
        return GoalBeliefV1(
            timestamp_s=self.timestamp_s,
            step_index=self.step_index,
            track_id=self.track_id,
            tracking_epoch_id=self.tracking_epoch_id,
            source=GoalBeliefSource.OBSERVATION_ONLY,
            coordinate_frame=CoordinateFrame.GLOBAL_XY,
            history_steps=(),
            force_estimate=None,
            desired_velocity_xy=None,
            desired_direction_rad=None,
            candidate_probabilities=candidates,
            unknown_candidate_probability=unknown,
            arrival_probability=0.0,
            # ``innovation`` is a non-negative diagnostic magnitude (for example,
            # an NIS); Slice A does not define a calibrated change probability.
            change_probability=0.0,
            mode=GoalBeliefMode.CENSORED,
            track_confidence=None,
            censoring_state=CensoringState.CENSORED,
            speed_cap_status=ActorSpeedCapStatus.UNKNOWN,
            blockers=blockers,
            reset_provenance=None,
            config_hash=self.config_hash,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HierarchicalGoalPosteriorV1:
        """Parse and verify a strict hierarchical actor payload.

        Returns:
            A validated hierarchical actor posterior.
        """
        value = _as_mapping(value, "hierarchical_goal_posterior")
        allowed = {
            "schema_version",
            "track_id",
            "tracking_epoch_id",
            "timestamp_s",
            "step_index",
            "destination_probabilities",
            "unknown_destination_probability",
            "waypoint_conditionals",
            "waypoint_parent_destination",
            "evidence_source",
            "innovation",
            "blockers",
            "config_hash",
            "candidate_set_digest",
            "state_digest",
        }
        reject_unknown_keys(value, allowed, "hierarchical_goal_posterior")
        if set(value) != allowed:
            raise ValueError("hierarchical_goal_posterior is missing a required field")
        destinations = tuple(
            HierarchicalProbability.from_dict(_as_mapping(item, "hierarchical_probability"))
            for item in _as_sequence(
                value["destination_probabilities"],
                "destination_probabilities",
            )
        )
        conditionals = tuple(
            HierarchicalWaypointConditionalV1.from_dict(_as_mapping(item, "waypoint_conditional"))
            for item in _as_sequence(value["waypoint_conditionals"], "waypoint_conditionals")
        )
        posterior = cls(
            schema_version=value["schema_version"],
            track_id=value["track_id"],
            tracking_epoch_id=value["tracking_epoch_id"],
            timestamp_s=value["timestamp_s"],
            step_index=value["step_index"],
            destination_probabilities=destinations,
            unknown_destination_probability=value["unknown_destination_probability"],
            waypoint_conditionals=conditionals,
            waypoint_parent_destination=value["waypoint_parent_destination"],
            evidence_source=value["evidence_source"],
            innovation=value["innovation"],
            blockers=value["blockers"],
            config_hash=value["config_hash"],
            candidate_set_digest=value["candidate_set_digest"],
        )
        if value["state_digest"] != posterior.state_digest:
            raise ValueError("state_digest does not match the canonical hierarchical state")
        return posterior


__all__ = [
    "HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION",
    "HIERARCHICAL_PROJECTION_LEVELS",
    "HierarchicalGoalPosteriorV1",
    "HierarchicalProbability",
    "HierarchicalWaypointConditionalV1",
]
