"""Causal inverse pedestrian goal-force estimation.

This module is the first executable rule-based slice of issue #8072.  It turns
public, global-frame pedestrian tracks into a bounded estimate of the desired
force using one, two, or three causal history rows.  The estimator is deliberately
separate from simulator state and from :mod:`oracle_transition_trace`:

* actor mode accepts only typed observations and explicitly supplied public force
  contributions;
* missing or unsupported contributors increase uncertainty and are reported;
* oracle transition traces are accepted only by the separate upper-bound method;
* a goal belief is emitted only from the actor-safe observation contract.

The output is an implementation and smoke-evidence surface, not a calibrated
behavior model or a benchmark result.  In particular, a partial observation is
never silently relabelled as an exact inverse.
"""

# This module keeps the complete validation and estimate assembly in one
# contract owner. The structural suppressions do not disable value or type
# checks.
# ruff: noqa: C901, DOC201, PLC0415, PLR0912, PLR0913, PLR0915

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from robot_sf.prediction._contract_utils import (
    reject_unknown_keys,
    require_covariance,
    require_digest,
    require_finite,
    require_non_negative,
    require_probability,
    require_step_index,
    require_text,
    require_xy,
    stable_digest,
)
from robot_sf.prediction.goal_belief_contract import (
    ActorObservationStep,
    ActorSpeedCapStatus,
    CensoringState,
    CoordinateFrame,
    ForceEstimate2D,
    GoalBeliefMode,
    GoalBeliefObservation,
    GoalBeliefV1,
    GoalCandidateProbability,
    ObservationMask,
)
from robot_sf.prediction.goal_intention import (
    GoalCandidateSet,
    HeadingGoalPosteriorConfig,
    update_heading_goal_posterior,
)

if TYPE_CHECKING:
    from robot_sf.prediction.goal_candidate_provider import GoalCandidateGenerationResult
    from robot_sf.prediction.oracle_transition_trace import OracleTransitionTraceV1
    from robot_sf.sensor.pedestrian_tracking import PedestrianTrack, PedestrianTrackingResult


GOAL_FORCE_INVERSE_SCHEMA_VERSION = "goal_force_inverse.v1"
GOAL_FORCE_INVERSE_CLAIM_BOUNDARY = "implementation_integrity_smoke"
HISTORY_LENGTHS = (1, 2, 3)
DEFAULT_EXPECTED_FORCE_COMPONENT_TYPES = (
    "social",
    "obstacle",
    "pedestrian_robot",
    "group",
    "adversarial",
)

Matrix2 = tuple[tuple[float, float], tuple[float, float]]
Vector2 = tuple[float, float]


class GoalForceEstimatorMode(StrEnum):
    """Input authority selected by a goal-force estimator configuration."""

    ACTOR_OBSERVATION_ONLY = "actor_observation_only"
    ORACLE_COMPONENT_UPPER_BOUND = "oracle_component_upper_bound"


class GoalForceInformationMode(StrEnum):
    """Evidence state of one inverse-force estimate."""

    HEADING_BASELINE = "heading_baseline"
    OBSERVATION_RECONSTRUCTED = "observation_reconstructed"
    PARTIAL_OBSERVATION = "partial_observation"
    ORACLE_COMPONENT_UPPER_BOUND = "oracle_component_upper_bound"
    UNAVAILABLE = "unavailable"


# Short public spelling for callers that use the issue's terminology.
GoalForceMode = GoalForceInformationMode


class ObservableForceComponentType(StrEnum):
    """Known non-goal force families in the #8065 registry."""

    SOCIAL = "social"
    PEDESTRIAN_PEDESTRIAN = "social"
    OBSTACLE = "obstacle"
    STATIC_OBSTACLE = "obstacle"
    PEDESTRIAN_ROBOT = "pedestrian_robot"
    GROUP = "group"
    ADVERSARIAL = "adversarial"


def _vector_add(left: Vector2, right: Vector2) -> Vector2:
    """Add two finite two-dimensional vectors."""

    return (left[0] + right[0], left[1] + right[1])


def _vector_scale(value: Vector2, factor: float) -> Vector2:
    """Scale a two-dimensional vector by a finite scalar."""

    return (value[0] * factor, value[1] * factor)


def _vector_subtract(left: Vector2, right: Vector2) -> Vector2:
    """Subtract two finite two-dimensional vectors."""

    return (left[0] - right[0], left[1] - right[1])


def _vector_norm(value: Vector2) -> float:
    """Return a numerically stable Euclidean norm."""

    return math.hypot(value[0], value[1])


def _vector_dot(left: Vector2, right: Vector2) -> float:
    """Return a two-dimensional dot product."""

    return left[0] * right[0] + left[1] * right[1]


def _zero_covariance() -> Matrix2:
    """Return a two-dimensional zero covariance."""

    return ((0.0, 0.0), (0.0, 0.0))


def _diagonal_covariance(value: float) -> Matrix2:
    """Return an isotropic covariance with a variance value."""

    variance = require_non_negative(value, "variance")
    return ((variance, 0.0), (0.0, variance))


def _covariance_add(*values: Matrix2) -> Matrix2:
    """Add two-dimensional covariance matrices component-wise."""

    return (
        (sum(value[0][0] for value in values), sum(value[0][1] for value in values)),
        (sum(value[1][0] for value in values), sum(value[1][1] for value in values)),
    )


def _covariance_scale(value: Matrix2, factor: float) -> Matrix2:
    """Scale a covariance by a finite non-negative factor."""

    scale = require_non_negative(factor, "covariance_scale")
    return tuple(tuple(entry * scale for entry in row) for row in value)  # type: ignore[return-value]


def _project_psd(
    value: Any,
    field_name: str,
    *,
    floor: float = 0.0,
    ceiling: float = math.inf,
) -> Matrix2:
    """Validate and project a symmetric 2x2 matrix onto a bounded PSD cone.

    Tracker covariances are allowed to be semidefinite.  The wire-level
    ``ForceEstimate2D`` contract is stricter and receives a positive-definite
    matrix after all uncertainty terms have been accumulated.
    """

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(
            isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 2
            for row in value
        )
    ):
        raise ValueError(f"{field_name} must be a 2x2 matrix")
    lower = require_non_negative(floor, f"{field_name}.floor")
    upper = (
        ceiling
        if isinstance(ceiling, float) and math.isinf(ceiling) and ceiling > 0.0
        else require_finite(ceiling, f"{field_name}.ceiling")
    )
    if upper < lower:
        raise ValueError(f"{field_name}.ceiling must be at least floor")
    a = require_finite(value[0][0], f"{field_name}[0][0]")
    b = require_finite(value[0][1], f"{field_name}[0][1]")
    c = require_finite(value[1][0], f"{field_name}[1][0]")
    d = require_finite(value[1][1], f"{field_name}[1][1]")
    off_diagonal = (b + c) / 2.0
    half_difference = (a - d) / 2.0
    radius = math.hypot(half_difference, off_diagonal)
    largest = (a + d) / 2.0 + radius
    smallest = (a + d) / 2.0 - radius
    if smallest < -1e-8:
        raise ValueError(f"{field_name} must be positive semidefinite")
    largest = min(max(largest, lower), upper)
    smallest = min(max(smallest, lower), upper)
    angle = 0.5 * math.atan2(2.0 * off_diagonal, a - d)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    projected_a = largest * cosine * cosine + smallest * sine * sine
    projected_d = largest * sine * sine + smallest * cosine * cosine
    projected_b = (largest - smallest) * cosine * sine
    return ((projected_a, projected_b), (projected_b, projected_d))


def _bounded_probability(value: float) -> float:
    """Clamp a computed probability after validating finite arithmetic."""

    numeric = require_finite(value, "computed_probability")
    return min(1.0, max(0.0, numeric))


def _sigmoid(value: float) -> float:
    """Evaluate a stable logistic function."""

    if value >= 0.0:
        exponent = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def _enum_value(enum_type: type[StrEnum], value: Any, field_name: str) -> StrEnum:
    """Parse a strict string enum with a useful field name."""

    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _normalize_blockers(values: Sequence[str]) -> tuple[str, ...]:
    """Return unique, stable blocker text."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("blockers must be a sequence")
    normalized = tuple(require_text(value, "blockers[]") for value in values)
    return tuple(sorted(set(normalized)))


@dataclass(frozen=True, slots=True)
class GoalForceInverseConfig:
    """Strict immutable configuration for causal inverse-force estimation."""

    enabled: bool = False
    history_length: int = 3
    mode: GoalForceEstimatorMode | str = GoalForceEstimatorMode.ACTOR_OBSERVATION_ONLY
    acceleration_estimator: str = "causal_linear_fit"
    min_dt_s: float = 1e-4
    max_dt_s: float = 2.0
    max_history_gap_s: float = 2.0
    speed_min_mps: float = 0.05
    direction_min_speed_mps: float = 0.05
    max_speed_tolerance_mps: float = 0.05
    relaxation_time_s: float = 0.5
    desired_force_factor: float = 1.0
    preferred_speed_mps: float | None = 1.3
    preferred_speed_std_mps: float = 0.25
    acceleration_noise_mps2: float = 0.25
    tracking_covariance_scale: float = 1.0
    known_force_variance_floor: float = 1e-8
    unmodeled_force_variance: float = 4.0
    parameter_variance: float = 0.25
    model_mismatch_variance: float = 0.25
    saturation_variance: float = 4.0
    covariance_floor: float = 1e-8
    covariance_ceiling: float = 1e6
    missing_force_policy: str = "partial"
    saturation_policy: str = "censor"
    expected_force_component_types: tuple[str, ...] = DEFAULT_EXPECTED_FORCE_COMPONENT_TYPES
    arrival_speed_threshold_mps: float = 0.25
    braking_acceleration_threshold_mps2: float = 0.1
    braking_probability_scale_mps2: float = 0.25
    heading_kappa: float = 4.0
    unknown_prior_probability: float = 0.1

    def __post_init__(self) -> None:
        """Validate every parameter participating in output provenance."""

        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")
        if type(self.history_length) is not int or self.history_length not in HISTORY_LENGTHS:
            raise ValueError("history_length must be one of 1, 2, or 3")
        object.__setattr__(
            self,
            "mode",
            _enum_value(GoalForceEstimatorMode, self.mode, "mode"),
        )
        if self.acceleration_estimator not in {"finite_difference", "causal_linear_fit"}:
            raise ValueError(
                "acceleration_estimator must be finite_difference or causal_linear_fit"
            )
        for field_name in ("min_dt_s", "max_dt_s", "max_history_gap_s"):
            value = require_finite(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if self.max_dt_s < self.min_dt_s:
            raise ValueError("max_dt_s must be at least min_dt_s")
        if self.max_history_gap_s < self.min_dt_s:
            raise ValueError("max_history_gap_s must be at least min_dt_s")
        for field_name in (
            "speed_min_mps",
            "direction_min_speed_mps",
            "max_speed_tolerance_mps",
            "preferred_speed_std_mps",
            "acceleration_noise_mps2",
            "tracking_covariance_scale",
            "known_force_variance_floor",
            "unmodeled_force_variance",
            "parameter_variance",
            "model_mismatch_variance",
            "saturation_variance",
            "covariance_floor",
            "covariance_ceiling",
            "arrival_speed_threshold_mps",
            "braking_acceleration_threshold_mps2",
            "heading_kappa",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_negative(getattr(self, field_name), field_name),
            )
        for field_name in (
            "relaxation_time_s",
            "desired_force_factor",
            "braking_probability_scale_mps2",
        ):
            value = require_finite(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if self.covariance_ceiling < self.covariance_floor:
            raise ValueError("covariance_ceiling must be at least covariance_floor")
        if self.preferred_speed_mps is not None:
            object.__setattr__(
                self,
                "preferred_speed_mps",
                require_non_negative(self.preferred_speed_mps, "preferred_speed_mps"),
            )
        require_probability(self.unknown_prior_probability, "unknown_prior_probability")
        for field_name in ("missing_force_policy", "saturation_policy"):
            text_value = require_text(getattr(self, field_name), field_name)
            if field_name == "missing_force_policy" and text_value not in {
                "partial",
                "unavailable",
            }:
                raise ValueError("missing_force_policy must be partial or unavailable")
            if field_name == "saturation_policy" and text_value not in {"censor", "unavailable"}:
                raise ValueError("saturation_policy must be censor or unavailable")
            object.__setattr__(self, field_name, text_value)
        component_types = tuple(
            require_text(value, "expected_force_component_types[]")
            for value in self.expected_force_component_types
        )
        if len(component_types) != len(set(component_types)):
            raise ValueError("expected_force_component_types must be unique")
        object.__setattr__(self, "expected_force_component_types", component_types)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GoalForceInverseConfig:
        """Build a configuration from a strict mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("config must be a mapping")
        raw = dict(value)
        schema_version = raw.pop("schema_version", None)
        if schema_version is not None and schema_version != GOAL_FORCE_INVERSE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {GOAL_FORCE_INVERSE_SCHEMA_VERSION!r}")
        allowed = {item.name for item in fields(cls)}
        reject_unknown_keys(raw, allowed, "goal_force_inverse_config")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe configuration provenance."""

        mode = self.mode.value if isinstance(self.mode, GoalForceEstimatorMode) else self.mode
        return {
            "schema_version": GOAL_FORCE_INVERSE_SCHEMA_VERSION,
            "enabled": self.enabled,
            "history_length": self.history_length,
            "mode": mode,
            "acceleration_estimator": self.acceleration_estimator,
            "min_dt_s": self.min_dt_s,
            "max_dt_s": self.max_dt_s,
            "max_history_gap_s": self.max_history_gap_s,
            "speed_min_mps": self.speed_min_mps,
            "direction_min_speed_mps": self.direction_min_speed_mps,
            "max_speed_tolerance_mps": self.max_speed_tolerance_mps,
            "relaxation_time_s": self.relaxation_time_s,
            "desired_force_factor": self.desired_force_factor,
            "preferred_speed_mps": self.preferred_speed_mps,
            "preferred_speed_std_mps": self.preferred_speed_std_mps,
            "acceleration_noise_mps2": self.acceleration_noise_mps2,
            "tracking_covariance_scale": self.tracking_covariance_scale,
            "known_force_variance_floor": self.known_force_variance_floor,
            "unmodeled_force_variance": self.unmodeled_force_variance,
            "parameter_variance": self.parameter_variance,
            "model_mismatch_variance": self.model_mismatch_variance,
            "saturation_variance": self.saturation_variance,
            "covariance_floor": self.covariance_floor,
            "covariance_ceiling": self.covariance_ceiling,
            "missing_force_policy": self.missing_force_policy,
            "saturation_policy": self.saturation_policy,
            "expected_force_component_types": list(self.expected_force_component_types),
            "arrival_speed_threshold_mps": self.arrival_speed_threshold_mps,
            "braking_acceleration_threshold_mps2": self.braking_acceleration_threshold_mps2,
            "braking_probability_scale_mps2": self.braking_probability_scale_mps2,
            "heading_kappa": self.heading_kappa,
            "unknown_prior_probability": self.unknown_prior_probability,
        }

    @property
    def config_hash(self) -> str:
        """Return a deterministic SHA-256 configuration digest."""

        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class GoalForceObservation:
    """One actor-visible, global-frame observation-history row."""

    track_id: str
    tracking_epoch_id: str
    timestamp_s: float
    step_index: int
    position_xy: Vector2 | None
    velocity_xy: Vector2 | None
    velocity_covariance_xy: Matrix2 = field(default_factory=_zero_covariance)
    position_covariance_xy: Matrix2 = field(default_factory=_zero_covariance)
    confidence: float = 1.0
    mask: ObservationMask = ObservationMask.OBSERVED
    status: str = "confirmed"
    blockers: tuple[str, ...] = ()
    coordinate_frame: CoordinateFrame = CoordinateFrame.GLOBAL_XY

    def __post_init__(self) -> None:
        """Reject future, non-global, non-finite, or ambiguous rows."""

        object.__setattr__(self, "track_id", require_text(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "tracking_epoch_id",
            require_text(self.tracking_epoch_id, "tracking_epoch_id"),
        )
        object.__setattr__(self, "timestamp_s", require_finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "step_index", require_step_index(self.step_index, "step_index"))
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("coordinate_frame must be CoordinateFrame")
        if self.coordinate_frame is not CoordinateFrame.GLOBAL_XY:
            raise ValueError("coordinate_frame must be global_xy")
        if not isinstance(self.mask, ObservationMask):
            raise TypeError("mask must be ObservationMask")
        position = None if self.position_xy is None else require_xy(self.position_xy, "position_xy")
        velocity = None if self.velocity_xy is None else require_xy(self.velocity_xy, "velocity_xy")
        if self.mask is ObservationMask.OBSERVED and (position is None or velocity is None):
            raise ValueError("observed rows require position and velocity")
        if self.mask is not ObservationMask.OBSERVED and (
            position is not None or velocity is not None
        ):
            raise ValueError("invisible and padded rows must omit position and velocity")
        object.__setattr__(self, "position_xy", position)
        object.__setattr__(self, "velocity_xy", velocity)
        object.__setattr__(
            self,
            "velocity_covariance_xy",
            _project_psd(self.velocity_covariance_xy, "velocity_covariance_xy"),
        )
        object.__setattr__(
            self,
            "position_covariance_xy",
            _project_psd(self.position_covariance_xy, "position_covariance_xy"),
        )
        object.__setattr__(self, "confidence", require_probability(self.confidence, "confidence"))
        object.__setattr__(self, "status", require_text(self.status, "status"))
        object.__setattr__(self, "blockers", _normalize_blockers(self.blockers))

    @classmethod
    def from_track(
        cls,
        track: PedestrianTrack,
        *,
        tracking_epoch_id: str,
        reset_provenance: str | None = None,
    ) -> GoalForceObservation:
        """Project one public tracker result without importing simulator state."""

        if not hasattr(track, "track_id") or not hasattr(track, "history_valid_mask"):
            raise TypeError("track must be a public PedestrianTrack value")
        status = str(getattr(track.status, "value", track.status))
        blockers = list(track.blockers)
        if reset_provenance is not None:
            blockers.append(f"reset:{require_text(reset_provenance, 'reset_provenance')}")
        history_mask = track.history_valid_mask
        if len(history_mask) == 0:
            raise ValueError("track history must contain at least one row")
        visible = bool(history_mask[-1]) and status not in {"lost", "retired"}
        if "velocity_unavailable" in blockers:
            visible = False
        return cls(
            track_id=f"track-{track.track_id}",
            tracking_epoch_id=tracking_epoch_id,
            timestamp_s=track.timestamp_s,
            step_index=track.step_index,
            position_xy=(float(track.position_global_xy[0]), float(track.position_global_xy[1]))
            if visible
            else None,
            velocity_xy=(float(track.velocity_global_xy[0]), float(track.velocity_global_xy[1]))
            if visible
            else None,
            velocity_covariance_xy=_project_psd(
                track.velocity_covariance,
                "track.velocity_covariance",
            ),
            position_covariance_xy=_project_psd(
                track.position_covariance,
                "track.position_covariance",
            ),
            confidence=track.association_confidence,
            mask=ObservationMask.OBSERVED if visible else ObservationMask.INVISIBLE,
            status=status,
            blockers=tuple(blockers),
        )

    def to_actor_step(self) -> ActorObservationStep:
        """Convert the row to the shared actor belief history contract."""

        return ActorObservationStep(
            timestamp_s=self.timestamp_s,
            step_index=self.step_index,
            position_xy=self.position_xy,
            velocity_xy=self.velocity_xy,
            mask=self.mask,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic diagnostic row."""

        return {
            "track_id": self.track_id,
            "tracking_epoch_id": self.tracking_epoch_id,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "position_xy": list(self.position_xy) if self.position_xy is not None else None,
            "velocity_xy": list(self.velocity_xy) if self.velocity_xy is not None else None,
            "velocity_covariance_xy": [list(row) for row in self.velocity_covariance_xy],
            "position_covariance_xy": [list(row) for row in self.position_covariance_xy],
            "confidence": self.confidence,
            "mask": self.mask.value,
            "status": self.status,
            "blockers": list(self.blockers),
            "coordinate_frame": self.coordinate_frame.value,
        }


@dataclass(frozen=True, slots=True)
class ObservableForceComponent:
    """One public non-goal force contribution or explicit unavailable slot.

    A zero vector is meaningful: it records that a component family was checked
    and contributed no force.  ``force_xy=None`` means the family was not
    reconstructed and therefore increases the unmodelled-force covariance.
    """

    component_id: str
    component_type: ObservableForceComponentType | str
    force_xy: Vector2 | None = None
    covariance_xy: Matrix2 = field(default_factory=_zero_covariance)
    enabled: bool = True
    actor_observable: bool = True
    source_entity: str | None = None
    unavailable_reason: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate public component identity and availability semantics."""

        object.__setattr__(self, "component_id", require_text(self.component_id, "component_id"))
        component_type = str(getattr(self.component_type, "value", self.component_type)).strip()
        object.__setattr__(self, "component_type", require_text(component_type, "component_type"))
        if self.force_xy is not None:
            object.__setattr__(self, "force_xy", require_xy(self.force_xy, "force_xy"))
            if self.unavailable_reason is not None:
                raise ValueError("available components must omit unavailable_reason")
        elif self.unavailable_reason is None:
            raise ValueError("unavailable components must name unavailable_reason")
        else:
            object.__setattr__(
                self,
                "unavailable_reason",
                require_text(self.unavailable_reason, "unavailable_reason"),
            )
        if type(self.enabled) is not bool or type(self.actor_observable) is not bool:
            raise TypeError("enabled and actor_observable must be bool")
        if self.source_entity is not None:
            object.__setattr__(
                self, "source_entity", require_text(self.source_entity, "source_entity")
            )
        if self.config_hash is not None:
            object.__setattr__(self, "config_hash", require_digest(self.config_hash, "config_hash"))
        object.__setattr__(
            self,
            "covariance_xy",
            _project_psd(self.covariance_xy, "covariance_xy"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic actor-side component record."""

        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "force_xy": list(self.force_xy) if self.force_xy is not None else None,
            "covariance_xy": [list(row) for row in self.covariance_xy],
            "enabled": self.enabled,
            "actor_observable": self.actor_observable,
            "source_entity": self.source_entity,
            "unavailable_reason": self.unavailable_reason,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class ForceComponentDiagnostic:
    """Source-level inclusion or exclusion reason for one force component."""

    component_id: str
    component_type: str
    status: str
    reason: str | None = None
    source_entity: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate diagnostic text."""

        for field_name in ("component_id", "component_type", "status"):
            object.__setattr__(
                self, field_name, require_text(getattr(self, field_name), field_name)
            )
        if self.reason is not None:
            object.__setattr__(self, "reason", require_text(self.reason, "reason"))
        if self.source_entity is not None:
            object.__setattr__(
                self, "source_entity", require_text(self.source_entity, "source_entity")
            )
        if self.config_hash is not None:
            object.__setattr__(self, "config_hash", require_digest(self.config_hash, "config_hash"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic."""

        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "status": self.status,
            "reason": self.reason,
            "source_entity": self.source_entity,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class CovarianceTerm:
    """One named term in the inverse-force uncertainty decomposition."""

    name: str
    covariance_xy: Matrix2

    def __post_init__(self) -> None:
        """Validate finite PSD covariance terms."""

        object.__setattr__(self, "name", require_text(self.name, "name"))
        object.__setattr__(self, "covariance_xy", _project_psd(self.covariance_xy, "covariance_xy"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe covariance term."""

        return {"name": self.name, "covariance_xy": [list(row) for row in self.covariance_xy]}


@dataclass(frozen=True, slots=True)
class ObservableForceReconstruction:
    """Public force sum and explicit missing-component accounting."""

    total_force_xy: Vector2 | None
    covariance_xy: Matrix2
    mode: GoalForceInformationMode
    included_component_ids: tuple[str, ...] = ()
    omitted_component_ids: tuple[str, ...] = ()
    missing_component_types: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    diagnostics: tuple[ForceComponentDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Validate stable component diagnostics."""

        if self.total_force_xy is not None:
            object.__setattr__(
                self, "total_force_xy", require_xy(self.total_force_xy, "total_force_xy")
            )
        object.__setattr__(self, "covariance_xy", _project_psd(self.covariance_xy, "covariance_xy"))
        if not isinstance(self.mode, GoalForceInformationMode):
            raise TypeError("mode must be GoalForceInformationMode")
        for field_name in (
            "included_component_ids",
            "omitted_component_ids",
            "missing_component_types",
        ):
            values = tuple(
                require_text(value, f"{field_name}[]") for value in getattr(self, field_name)
            )
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
            object.__setattr__(self, field_name, tuple(sorted(values)))
        diagnostics = tuple(self.diagnostics)
        if any(type(value) is not ForceComponentDiagnostic for value in diagnostics):
            raise TypeError("diagnostics must contain ForceComponentDiagnostic values")
        object.__setattr__(
            self, "diagnostics", tuple(sorted(diagnostics, key=lambda item: item.component_id))
        )
        object.__setattr__(self, "blockers", _normalize_blockers(self.blockers))

    def to_dict(self) -> dict[str, Any]:
        """Return a compact reconstruction receipt."""

        return {
            "total_force_xy": list(self.total_force_xy)
            if self.total_force_xy is not None
            else None,
            "covariance_xy": [list(row) for row in self.covariance_xy],
            "mode": self.mode.value,
            "included_component_ids": list(self.included_component_ids),
            "omitted_component_ids": list(self.omitted_component_ids),
            "missing_component_types": list(self.missing_component_types),
            "blockers": list(self.blockers),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


def reconstruct_observable_force(
    components: Sequence[ObservableForceComponent],
    *,
    expected_component_types: Sequence[str] = DEFAULT_EXPECTED_FORCE_COMPONENT_TYPES,
) -> ObservableForceReconstruction:
    """Sum only explicitly actor-observable force contributions.

    A caller must provide a zero-valued record when it knows a force family is
    absent.  Merely omitting a family is treated as missing information, which
    prevents the common but unsafe ``missing == zero`` shortcut.
    """

    values = tuple(components)
    if any(type(value) is not ObservableForceComponent for value in values):
        raise TypeError("components must contain ObservableForceComponent values")
    expected = tuple(
        require_text(value, "expected_component_types[]") for value in expected_component_types
    )
    if len(expected) != len(set(expected)):
        raise ValueError("expected_component_types must be unique")
    identifiers = tuple(value.component_id for value in values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("components must have unique component_id values")
    included: list[str] = []
    omitted: list[str] = []
    diagnostics: list[ForceComponentDiagnostic] = []
    blockers: list[str] = []
    total = (0.0, 0.0)
    covariance = _zero_covariance()
    seen_types: set[str] = set()
    for component in values:
        if component.component_type not in expected:
            omitted.append(component.component_id)
            blockers.append(f"unsupported_force_component:{component.component_type}")
            diagnostics.append(
                ForceComponentDiagnostic(
                    component.component_id,
                    component.component_type,
                    "unsupported",
                    "component type is outside the configured reconstruction roster",
                    component.source_entity,
                    component.config_hash,
                )
            )
        elif not component.actor_observable:
            omitted.append(component.component_id)
            blockers.append(f"force_component_not_actor_observable:{component.component_id}")
            diagnostics.append(
                ForceComponentDiagnostic(
                    component.component_id,
                    component.component_type,
                    "hidden",
                    "component is not actor-observable",
                    component.source_entity,
                    component.config_hash,
                )
            )
        elif not component.enabled:
            omitted.append(component.component_id)
            blockers.append(f"force_component_disabled:{component.component_id}")
            diagnostics.append(
                ForceComponentDiagnostic(
                    component.component_id,
                    component.component_type,
                    "disabled",
                    "component was disabled at the observation boundary",
                    component.source_entity,
                    component.config_hash,
                )
            )
        elif component.force_xy is None:
            omitted.append(component.component_id)
            blockers.append(f"force_component_unavailable:{component.component_id}")
            diagnostics.append(
                ForceComponentDiagnostic(
                    component.component_id,
                    component.component_type,
                    "unavailable",
                    component.unavailable_reason,
                    component.source_entity,
                    component.config_hash,
                )
            )
        else:
            included.append(component.component_id)
            seen_types.add(component.component_type)
            total = _vector_add(total, component.force_xy)
            covariance = _covariance_add(covariance, component.covariance_xy)
            diagnostics.append(
                ForceComponentDiagnostic(
                    component.component_id,
                    component.component_type,
                    "included",
                    None,
                    component.source_entity,
                    component.config_hash,
                )
            )
    missing_types = tuple(sorted(set(expected) - seen_types))
    blockers.extend(f"force_component_missing:{component_type}" for component_type in missing_types)
    if missing_types:
        blockers.append("non_goal_force_components_incomplete")
    complete = not blockers
    return ObservableForceReconstruction(
        total_force_xy=total if included else None,
        covariance_xy=covariance,
        mode=(
            GoalForceInformationMode.OBSERVATION_RECONSTRUCTED
            if complete
            else GoalForceInformationMode.PARTIAL_OBSERVATION
        ),
        included_component_ids=tuple(included),
        omitted_component_ids=tuple(omitted),
        missing_component_types=missing_types,
        blockers=tuple(blockers),
        diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True, slots=True)
class GoalForceEstimate:
    """Typed inverse-force result and optional actor-safe goal belief."""

    track_id: str
    tracking_epoch_id: str
    timestamp_s: float
    step_index: int
    history_length: int
    history_steps: tuple[ActorObservationStep, ...]
    mode: GoalForceInformationMode
    estimator_variant: str
    acceleration_xy: Vector2 | None
    acceleration_covariance_xy: Matrix2
    force_estimate: ForceEstimate2D | None
    desired_velocity_xy: Vector2 | None
    desired_direction_rad: float | None
    preferred_speed_mps: float | None
    relaxation_time_s: float | None
    desired_force_factor: float | None
    arrival_probability: float
    braking_probability: float
    change_probability: float
    speed_cap_status: ActorSpeedCapStatus
    censoring_state: CensoringState
    blockers: tuple[str, ...]
    covariance_terms: tuple[CovarianceTerm, ...]
    component_diagnostics: tuple[ForceComponentDiagnostic, ...]
    reconstruction: ObservableForceReconstruction | None
    config_hash: str
    belief: GoalBeliefV1 | None = None

    def __post_init__(self) -> None:
        """Validate finite estimate fields and actor/oracle separation."""

        object.__setattr__(self, "track_id", require_text(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "tracking_epoch_id",
            require_text(self.tracking_epoch_id, "tracking_epoch_id"),
        )
        object.__setattr__(self, "timestamp_s", require_finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "step_index", require_step_index(self.step_index, "step_index"))
        if type(self.history_length) is not int or self.history_length not in HISTORY_LENGTHS:
            raise ValueError("history_length must be one of 1, 2, or 3")
        history = tuple(self.history_steps)
        if any(type(value) is not ActorObservationStep for value in history):
            raise TypeError("history_steps must contain ActorObservationStep values")
        object.__setattr__(self, "history_steps", history)
        if not isinstance(self.mode, GoalForceInformationMode):
            raise TypeError("mode must be GoalForceInformationMode")
        object.__setattr__(
            self, "estimator_variant", require_text(self.estimator_variant, "estimator_variant")
        )
        if self.acceleration_xy is not None:
            object.__setattr__(
                self, "acceleration_xy", require_xy(self.acceleration_xy, "acceleration_xy")
            )
        object.__setattr__(
            self,
            "acceleration_covariance_xy",
            _project_psd(self.acceleration_covariance_xy, "acceleration_covariance_xy"),
        )
        if self.force_estimate is not None and type(self.force_estimate) is not ForceEstimate2D:
            raise TypeError("force_estimate must be ForceEstimate2D or None")
        if self.desired_velocity_xy is not None:
            object.__setattr__(
                self,
                "desired_velocity_xy",
                require_xy(self.desired_velocity_xy, "desired_velocity_xy"),
            )
        for field_name in ("preferred_speed_mps", "relaxation_time_s", "desired_force_factor"):
            value = getattr(self, field_name)
            if value is not None:
                numeric = require_non_negative(value, field_name)
                if field_name != "preferred_speed_mps" and numeric <= 0.0:
                    raise ValueError(f"{field_name} must be positive")
                object.__setattr__(self, field_name, numeric)
        if self.desired_direction_rad is not None:
            direction = require_finite(self.desired_direction_rad, "desired_direction_rad")
            if not -math.pi <= direction <= math.pi:
                raise ValueError("desired_direction_rad must be between -pi and pi")
            object.__setattr__(self, "desired_direction_rad", direction)
        for field_name in (
            "arrival_probability",
            "braking_probability",
            "change_probability",
        ):
            object.__setattr__(
                self,
                field_name,
                require_probability(getattr(self, field_name), field_name),
            )
        if not isinstance(self.speed_cap_status, ActorSpeedCapStatus):
            raise TypeError("speed_cap_status must be ActorSpeedCapStatus")
        if not isinstance(self.censoring_state, CensoringState):
            raise TypeError("censoring_state must be CensoringState")
        object.__setattr__(self, "blockers", _normalize_blockers(self.blockers))
        terms = tuple(self.covariance_terms)
        if any(type(value) is not CovarianceTerm for value in terms):
            raise TypeError("covariance_terms must contain CovarianceTerm values")
        if len({term.name for term in terms}) != len(terms):
            raise ValueError("covariance_terms must have unique names")
        object.__setattr__(
            self, "covariance_terms", tuple(sorted(terms, key=lambda item: item.name))
        )
        diagnostics = tuple(self.component_diagnostics)
        if any(type(value) is not ForceComponentDiagnostic for value in diagnostics):
            raise TypeError("component_diagnostics must contain ForceComponentDiagnostic values")
        object.__setattr__(
            self,
            "component_diagnostics",
            tuple(sorted(diagnostics, key=lambda item: item.component_id)),
        )
        if (
            self.reconstruction is not None
            and type(self.reconstruction) is not ObservableForceReconstruction
        ):
            raise TypeError("reconstruction must be ObservableForceReconstruction or None")
        object.__setattr__(self, "config_hash", require_digest(self.config_hash, "config_hash"))
        if self.belief is not None and type(self.belief) is not GoalBeliefV1:
            raise TypeError("belief must be GoalBeliefV1 or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic diagnostic payload."""

        return {
            "schema_version": GOAL_FORCE_INVERSE_SCHEMA_VERSION,
            "claim_boundary": GOAL_FORCE_INVERSE_CLAIM_BOUNDARY,
            "track_id": self.track_id,
            "tracking_epoch_id": self.tracking_epoch_id,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "history_length": self.history_length,
            "history_steps": [step.to_dict() for step in self.history_steps],
            "mode": self.mode.value,
            "estimator_variant": self.estimator_variant,
            "acceleration_xy": list(self.acceleration_xy)
            if self.acceleration_xy is not None
            else None,
            "acceleration_covariance_xy": [list(row) for row in self.acceleration_covariance_xy],
            "force_estimate": self.force_estimate.to_dict() if self.force_estimate else None,
            "desired_velocity_xy": list(self.desired_velocity_xy)
            if self.desired_velocity_xy is not None
            else None,
            "inferred_preferred_speed_mps": self.inferred_preferred_speed_mps,
            "desired_direction_rad": self.desired_direction_rad,
            "preferred_speed_mps": self.preferred_speed_mps,
            "relaxation_time_s": self.relaxation_time_s,
            "desired_force_factor": self.desired_force_factor,
            "arrival_probability": self.arrival_probability,
            "braking_probability": self.braking_probability,
            "change_probability": self.change_probability,
            "speed_cap_status": self.speed_cap_status.value,
            "censoring_state": self.censoring_state.value,
            "blockers": list(self.blockers),
            "covariance_terms": [term.to_dict() for term in self.covariance_terms],
            "component_diagnostics": [item.to_dict() for item in self.component_diagnostics],
            "reconstruction": self.reconstruction.to_dict() if self.reconstruction else None,
            "config_hash": self.config_hash,
            "belief": self.belief.to_dict() if self.belief is not None else None,
        }

    @property
    def content_digest(self) -> str:
        """Return a deterministic digest of the estimate payload."""

        return stable_digest(self.to_dict())

    @property
    def inferred_preferred_speed_mps(self) -> float | None:
        """Return the speed implied by the reconstructed desired velocity."""

        return None if self.desired_velocity_xy is None else _vector_norm(self.desired_velocity_xy)

    def to_actor_model_features(self) -> dict[str, Any]:
        """Return actor features only when the estimate has an actor belief."""

        if (
            self.belief is None
            or self.mode is GoalForceInformationMode.ORACLE_COMPONENT_UPPER_BOUND
        ):
            raise ValueError("oracle or unavailable estimates do not expose actor model features")
        return self.belief.to_actor_model_features()


def _history_steps(history: Sequence[GoalForceObservation]) -> tuple[ActorObservationStep, ...]:
    """Project typed observations to the actor history contract."""

    return tuple(value.to_actor_step() for value in history)


def _validate_history(history: Sequence[GoalForceObservation]) -> tuple[GoalForceObservation, ...]:
    """Validate identity and oldest-to-newest timing before estimation."""

    values = tuple(history)
    if not values:
        raise ValueError("history must contain at least one GoalForceObservation")
    if any(type(value) is not GoalForceObservation for value in values):
        raise TypeError("history must contain GoalForceObservation values")
    first = values[0]
    for value in values[1:]:
        if value.track_id != first.track_id:
            raise ValueError("history must contain one track_id")
        if value.tracking_epoch_id != first.tracking_epoch_id:
            raise ValueError("history must contain one tracking_epoch_id")
    for previous, current in pairwise(values):
        if current.step_index <= previous.step_index:
            raise ValueError("history step_index must be strictly increasing")
        if current.timestamp_s <= previous.timestamp_s:
            raise ValueError("history timestamp_s must be strictly increasing")
    return values


def _heading(value: Vector2, minimum_speed: float) -> float | None:
    """Return a stable heading or ``None`` below the direction threshold."""

    if _vector_norm(value) < minimum_speed:
        return None
    return math.atan2(value[1], value[0])


def _candidate_probabilities(
    *,
    position: Vector2,
    velocity: Vector2,
    candidate_set: GoalCandidateSet | None,
    prior: Mapping[str, float] | None,
    config: GoalForceInverseConfig,
    track_id: str,
    epoch: str,
    timestamp_s: float,
    step_index: int,
) -> tuple[tuple[GoalCandidateProbability, ...], float, tuple[str, ...]]:
    """Use #8068's heading posterior as the candidate-provider integration seam."""

    if candidate_set is None:
        return (), 1.0, ("candidate_provider_not_configured",)
    if type(candidate_set) is not GoalCandidateSet:
        raise TypeError("candidate_set must be GoalCandidateSet or None")
    posterior = update_heading_goal_posterior(
        track_id=track_id,
        observed_position_global=position,
        observed_velocity_global=velocity,
        candidate_set=candidate_set,
        prior=prior,
        config=HeadingGoalPosteriorConfig(
            heading_kappa=config.heading_kappa,
            unknown_prior_probability=config.unknown_prior_probability,
        ),
        timestamp_s=timestamp_s,
        step_index=step_index,
        tracking_epoch_id=epoch,
    )
    belief = posterior
    # The heading helper's contract is the canonical candidate conversion.  It
    # may return an unavailable typed belief when the provider has no point
    # candidates; preserve its explicit blocker and unknown mass.
    return belief.candidate_probabilities, belief.unknown_candidate_probability, belief.blockers


def _normalize_candidate_set(
    value: GoalCandidateSet | GoalCandidateGenerationResult | None,
) -> GoalCandidateSet | None:
    """Accept either #8073's result envelope or its actor-safe candidate set."""

    if value is None or type(value) is GoalCandidateSet:
        return value
    candidate_set = getattr(value, "candidate_set", None)
    if type(candidate_set) is GoalCandidateSet:
        return candidate_set
    raise TypeError(
        "candidate_set must be GoalCandidateSet, GoalCandidateGenerationResult, or None"
    )


class GoalForceInverseEstimator:
    """Estimate desired pedestrian force from causal actor-visible history."""

    __slots__ = ("config",)

    def __init__(self, config: GoalForceInverseConfig | Mapping[str, Any] | None = None) -> None:
        """Create a bounded estimator; the default configuration is off."""

        if config is None:
            config = GoalForceInverseConfig()
        elif isinstance(config, Mapping):
            config = GoalForceInverseConfig.from_mapping(config)
        if type(config) is not GoalForceInverseConfig:
            raise TypeError("config must be GoalForceInverseConfig or a mapping")
        self.config = config

    def estimate(
        self,
        history: Sequence[GoalForceObservation],
        *,
        known_force_components: Sequence[ObservableForceComponent] = (),
        expected_force_component_types: Sequence[str] | None = None,
        preferred_speed_mps: float | None = None,
        relaxation_time_s: float | None = None,
        desired_force_factor: float | None = None,
        max_speed_mps: float | None = None,
        candidate_set: GoalCandidateSet | GoalCandidateGenerationResult | None = None,
        prior: Mapping[str, float] | None = None,
    ) -> GoalForceEstimate:
        """Estimate one actor-safe inverse force from oldest-to-newest history.

        ``known_force_components`` is a public reconstruction projection.  It is
        intentionally not the #8065 ``ForceComponents`` oracle record; passing
        an oracle object or trace here fails closed at the boundary.
        """

        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise TypeError("history must be a sequence of GoalForceObservation values")
        values = _validate_history(history)
        if self.config.mode is GoalForceEstimatorMode.ORACLE_COMPONENT_UPPER_BOUND:
            raise ValueError("oracle mode requires estimate_from_oracle_trace")
        if isinstance(known_force_components, (str, bytes)) or not isinstance(
            known_force_components, Sequence
        ):
            raise TypeError("known_force_components must be a sequence")
        if any(
            type(component) is not ObservableForceComponent for component in known_force_components
        ):
            raise TypeError("known_force_components must contain ObservableForceComponent values")
        candidate_set = _normalize_candidate_set(candidate_set)
        latest = values[-1]
        selected = values[-self.config.history_length :]
        blockers = list(latest.blockers)
        blockers.extend(
            blocker for value in selected for blocker in value.blockers if blocker not in blockers
        )
        if not self.config.enabled:
            blockers.append("estimator_disabled")
            return self._unavailable(values, blockers, estimator_variant="disabled")
        if len(values) < self.config.history_length:
            blockers.append(f"history_length_{self.config.history_length}_required")
            return self._unavailable(values, blockers, estimator_variant="insufficient_history")
        if latest.mask is not ObservationMask.OBSERVED:
            blockers.append("latest_observation_unavailable")
            return self._unavailable(
                values, blockers, estimator_variant="latest_observation_unavailable"
            )
        if latest.status in {"lost", "retired"}:
            blockers.append(f"track_{latest.status}")
            return self._unavailable(
                values, blockers, estimator_variant="track_lifecycle_unavailable"
            )
        for previous, current in pairwise(selected):
            delta_t = current.timestamp_s - previous.timestamp_s
            if delta_t < self.config.min_dt_s:
                blockers.append("non_positive_or_tiny_dt")
            elif delta_t > self.config.max_dt_s or delta_t > self.config.max_history_gap_s:
                blockers.append("history_gap_exceeds_configured_limit")
        if any(value.mask is not ObservationMask.OBSERVED for value in selected):
            blockers.append("observation_history_incomplete")
        if blockers and any(
            blocker
            in {
                "non_positive_or_tiny_dt",
                "history_gap_exceeds_configured_limit",
                "observation_history_incomplete",
            }
            for blocker in blockers
        ):
            return self._unavailable(selected, blockers, estimator_variant="invalid_history")
        if latest.position_xy is None or latest.velocity_xy is None:
            blockers.append("latest_state_unavailable")
            return self._unavailable(
                selected, blockers, estimator_variant="latest_state_unavailable"
            )
        max_speed = (
            None if max_speed_mps is None else require_non_negative(max_speed_mps, "max_speed_mps")
        )
        preferred_speed = self._resolve_preferred_speed(
            preferred_speed_mps, max_speed, latest.velocity_xy
        )
        tau = self._resolve_positive(
            relaxation_time_s, self.config.relaxation_time_s, "relaxation_time_s"
        )
        factor = self._resolve_positive(
            desired_force_factor,
            self.config.desired_force_factor,
            "desired_force_factor",
        )
        cap_status = self._infer_speed_cap(latest.velocity_xy, max_speed)
        if cap_status is ActorSpeedCapStatus.UNKNOWN:
            blockers.append("speed_cap_status_unknown")
        elif cap_status is ActorSpeedCapStatus.POSSIBLE:
            blockers.append("speed_cap_may_censor_transition")
        candidate_velocity = latest.velocity_xy
        if self.config.history_length == 1:
            direction = _heading(candidate_velocity, self.config.direction_min_speed_mps)
            blockers.extend(("force_requires_two_frames", "arrival_probability_unestimated"))
            candidate_probs, unknown, candidate_blockers = _candidate_probabilities(
                position=latest.position_xy,
                velocity=candidate_velocity,
                candidate_set=candidate_set,
                prior=prior,
                config=self.config,
                track_id=latest.track_id,
                epoch=latest.tracking_epoch_id,
                timestamp_s=latest.timestamp_s,
                step_index=latest.step_index,
            )
            blockers.extend(candidate_blockers)
            belief = self._build_belief(
                selected,
                mode=GoalBeliefMode.CENSORED,
                force_estimate=None,
                desired_velocity=None,
                desired_direction=direction,
                candidate_probabilities=candidate_probs,
                unknown_candidate_probability=unknown,
                arrival_probability=0.0,
                change_probability=0.0,
                track_confidence=latest.confidence,
                censoring_state=CensoringState.UNKNOWN,
                speed_cap_status=cap_status,
                blockers=tuple(blockers),
            )
            covariance_terms = (
                CovarianceTerm(
                    "unavailable_force", _diagonal_covariance(self.config.unmodeled_force_variance)
                ),
            )
            return GoalForceEstimate(
                track_id=latest.track_id,
                tracking_epoch_id=latest.tracking_epoch_id,
                timestamp_s=latest.timestamp_s,
                step_index=latest.step_index,
                history_length=self.config.history_length,
                history_steps=_history_steps(selected),
                mode=GoalForceInformationMode.HEADING_BASELINE,
                estimator_variant="h1_heading_baseline",
                acceleration_xy=None,
                acceleration_covariance_xy=_project_psd(
                    _diagonal_covariance(self.config.unmodeled_force_variance),
                    "acceleration_covariance_xy",
                    floor=self.config.covariance_floor,
                    ceiling=self.config.covariance_ceiling,
                ),
                force_estimate=None,
                desired_velocity_xy=None,
                desired_direction_rad=direction,
                preferred_speed_mps=preferred_speed,
                relaxation_time_s=tau,
                desired_force_factor=factor,
                arrival_probability=0.0,
                braking_probability=0.0,
                change_probability=0.0,
                speed_cap_status=cap_status,
                censoring_state=CensoringState.UNKNOWN,
                blockers=tuple(blockers),
                covariance_terms=covariance_terms,
                component_diagnostics=(),
                reconstruction=None,
                config_hash=self.config.config_hash,
                belief=belief,
            )

        acceleration, acceleration_covariance, estimator_variant, residual_covariance = (
            self._estimate_acceleration(selected)
        )
        reconstruction = reconstruct_observable_force(
            known_force_components,
            expected_component_types=(
                self.config.expected_force_component_types
                if expected_force_component_types is None
                else expected_force_component_types
            ),
        )
        blockers.extend(reconstruction.blockers)
        if reconstruction.mode is GoalForceInformationMode.PARTIAL_OBSERVATION:
            if self.config.missing_force_policy == "unavailable":
                blockers.append("missing_force_policy_unavailable")
                return self._unavailable(
                    selected,
                    blockers,
                    estimator_variant="missing_force_components",
                    reconstruction=reconstruction,
                )
        known_force = reconstruction.total_force_xy or (0.0, 0.0)
        goal_force = _vector_subtract(acceleration, known_force)
        known_covariance = reconstruction.covariance_xy
        acceleration_term = CovarianceTerm("acceleration", acceleration_covariance)
        known_force_term = CovarianceTerm("known_force", known_covariance)
        tracking_term = CovarianceTerm(
            "tracking",
            self._tracking_covariance(selected),
        )
        unmodeled_count = len(reconstruction.missing_component_types) + len(
            reconstruction.omitted_component_ids
        )
        unmodeled_variance = (
            self.config.unmodeled_force_variance * max(1, unmodeled_count)
            if reconstruction.mode is GoalForceInformationMode.PARTIAL_OBSERVATION
            else 0.0
        )
        unmodeled_term = CovarianceTerm(
            "unmodeled_force",
            _diagonal_covariance(unmodeled_variance),
        )
        parameter_variance = self.config.parameter_variance
        if preferred_speed_mps is None:
            parameter_variance += self.config.preferred_speed_std_mps**2
            blockers.append("preferred_speed_prior_used")
        parameter_term = CovarianceTerm("parameter", _diagonal_covariance(parameter_variance))
        model_mismatch_term = CovarianceTerm(
            "model_mismatch",
            _covariance_add(
                _diagonal_covariance(self.config.model_mismatch_variance),
                residual_covariance,
            ),
        )
        terms = [
            acceleration_term,
            known_force_term,
            tracking_term,
            unmodeled_term,
            parameter_term,
            model_mismatch_term,
        ]
        if cap_status is not ActorSpeedCapStatus.CLEAR:
            terms.append(
                CovarianceTerm("saturation", _diagonal_covariance(self.config.saturation_variance))
            )
            if self.config.saturation_policy == "unavailable":
                blockers.append("saturation_policy_unavailable")
                return self._unavailable(
                    selected,
                    blockers,
                    estimator_variant="speed_cap_uncertain",
                    reconstruction=reconstruction,
                )
        force_covariance = _project_psd(
            _covariance_add(*(term.covariance_xy for term in terms)),
            "force_covariance",
            floor=self.config.covariance_floor,
            ceiling=self.config.covariance_ceiling,
        )
        force_estimate = ForceEstimate2D(
            mean_xy=goal_force,
            covariance_xy=require_covariance(force_covariance, "force_covariance"),
        )
        transition_start_velocity = selected[-2].velocity_xy
        assert transition_start_velocity is not None
        desired_velocity_raw = _vector_add(
            transition_start_velocity,
            _vector_scale(goal_force, tau / factor),
        )
        braking_probability, arrival_probability = self._arrival_and_braking(
            selected,
            goal_force,
        )
        desired_velocity = desired_velocity_raw
        if (
            braking_probability >= 0.5
            and _vector_norm(latest.velocity_xy) >= self.config.direction_min_speed_mps
        ):
            unit = _vector_scale(latest.velocity_xy, 1.0 / _vector_norm(latest.velocity_xy))
            longitudinal = _vector_dot(desired_velocity_raw, unit)
            if longitudinal < 0.0:
                desired_velocity = _vector_scale(unit, max(0.0, longitudinal))
                blockers.append("braking_direction_preserved")
        desired_direction = _heading(desired_velocity, self.config.direction_min_speed_mps)
        if desired_direction is None:
            desired_direction = _heading(latest.velocity_xy, self.config.direction_min_speed_mps)
        if arrival_probability > 0.0:
            blockers.append("arrival_probability_is_local_braking_proxy")
        blockers.append("change_probability_unestimated")
        candidate_velocity = desired_velocity
        candidate_probs, unknown, candidate_blockers = _candidate_probabilities(
            position=latest.position_xy,
            velocity=candidate_velocity,
            candidate_set=candidate_set,
            prior=prior,
            config=self.config,
            track_id=latest.track_id,
            epoch=latest.tracking_epoch_id,
            timestamp_s=latest.timestamp_s,
            step_index=latest.step_index,
        )
        blockers.extend(candidate_blockers)
        information_mode = reconstruction.mode
        belief_mode = (
            GoalBeliefMode.NOMINAL
            if information_mode is GoalForceInformationMode.OBSERVATION_RECONSTRUCTED
            and cap_status is ActorSpeedCapStatus.CLEAR
            else GoalBeliefMode.CENSORED
        )
        censoring_state = (
            CensoringState.NONE
            if belief_mode is GoalBeliefMode.NOMINAL
            else CensoringState.SATURATED
            if cap_status is ActorSpeedCapStatus.POSSIBLE
            else CensoringState.UNKNOWN
        )
        belief = self._build_belief(
            selected,
            mode=belief_mode,
            force_estimate=force_estimate,
            desired_velocity=desired_velocity,
            desired_direction=desired_direction,
            candidate_probabilities=candidate_probs,
            unknown_candidate_probability=unknown,
            arrival_probability=arrival_probability,
            change_probability=0.0,
            track_confidence=latest.confidence,
            censoring_state=censoring_state,
            speed_cap_status=cap_status,
            blockers=blockers,
        )
        return GoalForceEstimate(
            track_id=latest.track_id,
            tracking_epoch_id=latest.tracking_epoch_id,
            timestamp_s=latest.timestamp_s,
            step_index=latest.step_index,
            history_length=self.config.history_length,
            history_steps=_history_steps(selected),
            mode=information_mode,
            estimator_variant=estimator_variant,
            acceleration_xy=acceleration,
            acceleration_covariance_xy=_project_psd(
                acceleration_covariance,
                "acceleration_covariance_xy",
                floor=self.config.covariance_floor,
                ceiling=self.config.covariance_ceiling,
            ),
            force_estimate=force_estimate,
            desired_velocity_xy=desired_velocity,
            desired_direction_rad=desired_direction,
            preferred_speed_mps=preferred_speed,
            relaxation_time_s=tau,
            desired_force_factor=factor,
            arrival_probability=arrival_probability,
            braking_probability=braking_probability,
            change_probability=0.0,
            speed_cap_status=cap_status,
            censoring_state=censoring_state,
            blockers=tuple(blockers),
            covariance_terms=tuple(terms),
            component_diagnostics=reconstruction.diagnostics,
            reconstruction=reconstruction,
            config_hash=self.config.config_hash,
            belief=belief,
        )

    def estimate_from_track(
        self,
        track: PedestrianTrack,
        *,
        tracking_epoch_id: str,
        **kwargs: Any,
    ) -> GoalForceEstimate:
        """Estimate from one public track; one row yields the H=1 baseline."""

        observation = GoalForceObservation.from_track(
            track,
            tracking_epoch_id=tracking_epoch_id,
        )
        return self.estimate((observation,), **kwargs)

    def estimate_from_oracle_trace(self, trace: OracleTransitionTraceV1) -> GoalForceEstimate:
        """Return an evaluator-only upper bound from one exact oracle trace.

        This method never constructs a ``GoalBeliefV1``.  Ineligible traces are
        represented as finite unavailable diagnostics so a caller cannot mistake
        a controller jump, cap, or missing force stage for an exact label.
        """

        from robot_sf.prediction.oracle_transition_trace import OracleTransitionTraceV1

        if type(trace) is not OracleTransitionTraceV1:
            raise TypeError("trace must be OracleTransitionTraceV1")
        pre_force = trace.post_behavior_pre_force
        post = trace.post_integration
        dt = post.timestamp_s - pre_force.timestamp_s
        identity = trace.actor_track_id or trace.simulator_pedestrian_id
        epoch = trace.actor_tracking_epoch_id or "oracle"
        history = (
            GoalForceObservation(
                track_id=identity,
                tracking_epoch_id=epoch,
                timestamp_s=pre_force.timestamp_s,
                step_index=pre_force.step_index,
                position_xy=pre_force.position_xy,
                velocity_xy=pre_force.velocity_xy,
            ),
            GoalForceObservation(
                track_id=identity,
                tracking_epoch_id=epoch,
                timestamp_s=post.timestamp_s,
                step_index=post.step_index,
                position_xy=post.position_xy,
                velocity_xy=post.velocity_xy,
            ),
        )
        history_steps = _history_steps(history)
        blockers = [f"oracle_trace:{reason.value}" for reason in trace.exact_inverse_reasons]
        if not trace.exact_inverse_eligible:
            blockers.append("oracle_trace_not_exact_inverse_eligible")
            return GoalForceEstimate(
                track_id=identity,
                tracking_epoch_id=epoch,
                timestamp_s=post.timestamp_s,
                step_index=post.step_index,
                history_length=2,
                history_steps=history_steps,
                mode=GoalForceInformationMode.UNAVAILABLE,
                estimator_variant="oracle_exact_inverse_rejected",
                acceleration_xy=None,
                acceleration_covariance_xy=_project_psd(
                    _diagonal_covariance(self.config.unmodeled_force_variance),
                    "acceleration_covariance_xy",
                    floor=self.config.covariance_floor,
                    ceiling=self.config.covariance_ceiling,
                ),
                force_estimate=None,
                desired_velocity_xy=None,
                desired_direction_rad=None,
                preferred_speed_mps=trace.dynamics.preferred_speed_mps,
                relaxation_time_s=trace.dynamics.relaxation_time_s,
                desired_force_factor=trace.dynamics.desired_force_factor,
                arrival_probability=0.0,
                braking_probability=0.0,
                change_probability=0.0,
                speed_cap_status=ActorSpeedCapStatus.UNKNOWN,
                censoring_state=CensoringState.UNKNOWN,
                blockers=tuple(blockers),
                covariance_terms=(
                    CovarianceTerm(
                        "unavailable_force",
                        _diagonal_covariance(self.config.unmodeled_force_variance),
                    ),
                ),
                component_diagnostics=tuple(
                    ForceComponentDiagnostic(
                        item.component_id,
                        item.component_type,
                        "oracle_not_eligible",
                        item.unavailable_reason,
                        item.source_entity,
                        item.config_hash,
                    )
                    for item in trace.force_components.component_records
                ),
                reconstruction=None,
                config_hash=self.config.config_hash,
            )
        if dt < self.config.min_dt_s or dt > self.config.max_dt_s:
            blockers.append("oracle_transition_dt_outside_config")
        goal_force = trace.force_components.goal_force_xy
        if goal_force is None:
            blockers.append("oracle_goal_force_stage_unavailable")
        if blockers:
            return GoalForceEstimate(
                track_id=identity,
                tracking_epoch_id=epoch,
                timestamp_s=post.timestamp_s,
                step_index=post.step_index,
                history_length=2,
                history_steps=history_steps,
                mode=GoalForceInformationMode.UNAVAILABLE,
                estimator_variant="oracle_exact_inverse_unavailable",
                acceleration_xy=None,
                acceleration_covariance_xy=_project_psd(
                    _diagonal_covariance(self.config.unmodeled_force_variance),
                    "acceleration_covariance_xy",
                    floor=self.config.covariance_floor,
                    ceiling=self.config.covariance_ceiling,
                ),
                force_estimate=None,
                desired_velocity_xy=None,
                desired_direction_rad=None,
                preferred_speed_mps=trace.dynamics.preferred_speed_mps,
                relaxation_time_s=trace.dynamics.relaxation_time_s,
                desired_force_factor=trace.dynamics.desired_force_factor,
                arrival_probability=0.0,
                braking_probability=0.0,
                change_probability=0.0,
                speed_cap_status=ActorSpeedCapStatus.UNKNOWN,
                censoring_state=CensoringState.UNKNOWN,
                blockers=tuple(blockers),
                covariance_terms=(
                    CovarianceTerm(
                        "unavailable_force",
                        _diagonal_covariance(self.config.unmodeled_force_variance),
                    ),
                ),
                component_diagnostics=(),
                reconstruction=None,
                config_hash=self.config.config_hash,
            )
        assert goal_force is not None
        acceleration = _vector_scale(
            _vector_subtract(post.velocity_xy, pre_force.velocity_xy),
            1.0 / dt,
        )
        force_covariance = _project_psd(
            _diagonal_covariance(
                max(self.config.covariance_floor, self.config.known_force_variance_floor)
            ),
            "oracle_force_covariance",
            floor=self.config.covariance_floor,
            ceiling=self.config.covariance_ceiling,
        )
        force_estimate = ForceEstimate2D(
            mean_xy=goal_force,
            covariance_xy=require_covariance(force_covariance, "oracle_force_covariance"),
        )
        desired_velocity = None
        desired_direction = None
        tau = trace.dynamics.relaxation_time_s
        factor = trace.dynamics.desired_force_factor
        if tau is not None and factor is not None and factor > 0.0:
            desired_velocity = _vector_add(
                pre_force.velocity_xy,
                _vector_scale(goal_force, tau / factor),
            )
            desired_direction = _heading(desired_velocity, self.config.direction_min_speed_mps)
        else:
            blockers.append("oracle_dynamics_parameters_unavailable")
        cap_status = {
            "not_applied": ActorSpeedCapStatus.CLEAR,
            "applied": ActorSpeedCapStatus.POSSIBLE,
            "unknown": ActorSpeedCapStatus.UNKNOWN,
        }[trace.speed_cap.status.value]
        censoring = (
            CensoringState.NONE
            if cap_status is ActorSpeedCapStatus.CLEAR
            else CensoringState.SATURATED
            if cap_status is ActorSpeedCapStatus.POSSIBLE
            else CensoringState.UNKNOWN
        )
        return GoalForceEstimate(
            track_id=identity,
            tracking_epoch_id=epoch,
            timestamp_s=post.timestamp_s,
            step_index=post.step_index,
            history_length=2,
            history_steps=history_steps,
            mode=GoalForceInformationMode.ORACLE_COMPONENT_UPPER_BOUND,
            estimator_variant="oracle_component_upper_bound",
            acceleration_xy=acceleration,
            acceleration_covariance_xy=force_covariance,
            force_estimate=force_estimate,
            desired_velocity_xy=desired_velocity,
            desired_direction_rad=desired_direction,
            preferred_speed_mps=trace.dynamics.preferred_speed_mps,
            relaxation_time_s=tau,
            desired_force_factor=factor,
            arrival_probability=1.0 if trace.goal_change_kind.value == "arrival" else 0.0,
            braking_probability=0.0,
            change_probability=0.0,
            speed_cap_status=cap_status,
            censoring_state=censoring,
            blockers=tuple(blockers),
            covariance_terms=(
                CovarianceTerm("acceleration", force_covariance),
                CovarianceTerm("known_force", force_covariance),
            ),
            component_diagnostics=tuple(
                ForceComponentDiagnostic(
                    item.component_id,
                    item.component_type,
                    "oracle_included" if item.force_xy is not None else "oracle_unavailable",
                    item.unavailable_reason,
                    item.source_entity,
                    item.config_hash,
                )
                for item in trace.force_components.component_records
            ),
            reconstruction=None,
            config_hash=self.config.config_hash,
        )

    # Explicit alias used by evaluation scripts and issue text.
    estimate_oracle_component_upper_bound = estimate_from_oracle_trace

    def _resolve_positive(self, override: float | None, default: float, field_name: str) -> float:
        """Resolve a positive parameter override."""

        value = default if override is None else override
        numeric = require_finite(value, field_name)
        if numeric <= 0.0:
            raise ValueError(f"{field_name} must be positive")
        return numeric

    def _resolve_preferred_speed(
        self,
        override: float | None,
        max_speed: float | None,
        velocity: Vector2,
    ) -> float | None:
        """Resolve preferred speed from the public per-ped value or prior."""

        if override is not None:
            return require_non_negative(override, "preferred_speed_mps")
        if self.config.preferred_speed_mps is not None:
            return self.config.preferred_speed_mps
        if max_speed is not None:
            return max_speed
        return _vector_norm(velocity)

    def _infer_speed_cap(self, velocity: Vector2, max_speed: float | None) -> ActorSpeedCapStatus:
        """Infer cap uncertainty from actor-visible speed only."""

        if max_speed is None:
            return ActorSpeedCapStatus.UNKNOWN
        return (
            ActorSpeedCapStatus.POSSIBLE
            if _vector_norm(velocity) >= max(0.0, max_speed - self.config.max_speed_tolerance_mps)
            else ActorSpeedCapStatus.CLEAR
        )

    def _estimate_acceleration(
        self,
        history: Sequence[GoalForceObservation],
    ) -> tuple[Vector2, Matrix2, str, Matrix2]:
        """Estimate acceleration and propagated velocity-fit uncertainty."""

        if len(history) == 2:
            previous, current = history[-2:]
            dt = current.timestamp_s - previous.timestamp_s
            assert previous.velocity_xy is not None and current.velocity_xy is not None
            acceleration = _vector_scale(
                _vector_subtract(current.velocity_xy, previous.velocity_xy),
                1.0 / dt,
            )
            covariance = _covariance_scale(
                _covariance_add(previous.velocity_covariance_xy, current.velocity_covariance_xy),
                1.0 / (dt * dt),
            )
            covariance = _covariance_add(
                covariance,
                _diagonal_covariance(self.config.acceleration_noise_mps2**2),
            )
            return acceleration, covariance, "h2_finite_difference", _zero_covariance()
        if self.config.acceleration_estimator == "finite_difference":
            previous, middle, current = history[-3:]
            assert (
                previous.velocity_xy is not None
                and middle.velocity_xy is not None
                and current.velocity_xy is not None
            )
            first_dt = middle.timestamp_s - previous.timestamp_s
            second_dt = current.timestamp_s - middle.timestamp_s
            first = _vector_scale(
                _vector_subtract(middle.velocity_xy, previous.velocity_xy),
                1.0 / first_dt,
            )
            second = _vector_scale(
                _vector_subtract(current.velocity_xy, middle.velocity_xy),
                1.0 / second_dt,
            )
            acceleration = _vector_scale(_vector_add(first, second), 0.5)
            covariance = _covariance_add(
                _covariance_scale(
                    _covariance_add(previous.velocity_covariance_xy, middle.velocity_covariance_xy),
                    0.25 / (first_dt * first_dt),
                ),
                _covariance_scale(
                    _covariance_add(middle.velocity_covariance_xy, current.velocity_covariance_xy),
                    0.25 / (second_dt * second_dt),
                ),
                _diagonal_covariance(self.config.acceleration_noise_mps2**2),
            )
            residual = _vector_scale(_vector_subtract(first, second), 0.5)
            return (
                acceleration,
                covariance,
                "h3_causal_finite_difference_mean",
                (
                    (residual[0] * residual[0], residual[0] * residual[1]),
                    (residual[0] * residual[1], residual[1] * residual[1]),
                ),
            )
        points = history[-3:]
        velocities = tuple(point.velocity_xy for point in points)
        if any(velocity is None for velocity in velocities):
            raise ValueError("causal linear fit requires visible velocity observations")
        resolved_velocities: tuple[Vector2, ...] = tuple(
            velocity for velocity in velocities if velocity is not None
        )
        times = [point.timestamp_s for point in points]
        mean_time = sum(times) / len(times)
        centered = [time - mean_time for time in times]
        denominator = sum(value * value for value in centered)
        if denominator <= 0.0:
            raise ValueError("causal linear fit requires distinct timestamps")
        acceleration = (
            sum(
                value * velocity[0]
                for value, velocity in zip(centered, resolved_velocities, strict=True)
            )
            / denominator,
            sum(
                value * velocity[1]
                for value, velocity in zip(centered, resolved_velocities, strict=True)
            )
            / denominator,
        )
        covariance = _zero_covariance()
        for value, point in zip(centered, points, strict=True):
            weight = value / denominator
            covariance = _covariance_add(
                covariance,
                _covariance_scale(point.velocity_covariance_xy, weight * weight),
            )
        covariance = _covariance_add(
            covariance,
            _diagonal_covariance(self.config.acceleration_noise_mps2**2),
        )
        residual_covariance = _zero_covariance()
        for velocity, time in zip(resolved_velocities, times, strict=True):
            predicted = _vector_add(
                (
                    sum(item[0] for item in resolved_velocities) / 3.0,
                    sum(item[1] for item in resolved_velocities) / 3.0,
                ),
                _vector_scale(acceleration, time - mean_time),
            )
            residual = _vector_subtract(velocity, predicted)
            residual_covariance = _covariance_add(
                residual_covariance,
                (
                    (residual[0] * residual[0], residual[0] * residual[1]),
                    (residual[0] * residual[1], residual[1] * residual[1]),
                ),
            )
        residual_covariance = _covariance_scale(residual_covariance, 1.0 / 3.0)
        return acceleration, covariance, "h3_causal_linear_fit", residual_covariance

    def _tracking_covariance(self, history: Sequence[GoalForceObservation]) -> Matrix2:
        """Inflate uncertainty for low-confidence and lifecycle-disrupted tracks."""

        confidence = min(value.confidence for value in history)
        lifecycle_multiplier = 1.0
        for value in history:
            if value.status in {"lost", "reacquired"} or value.blockers:
                lifecycle_multiplier += 1.0
        covariance = _zero_covariance()
        for value in history:
            covariance = _covariance_add(covariance, value.velocity_covariance_xy)
        covariance = _covariance_scale(
            covariance,
            self.config.tracking_covariance_scale
            * lifecycle_multiplier
            * (1.0 + (1.0 - confidence) * 4.0)
            / max(1, len(history)),
        )
        confidence_variance = (
            self.config.tracking_covariance_scale
            * (1.0 - confidence) ** 2
            * max(self.config.acceleration_noise_mps2**2, self.config.covariance_floor)
        )
        return _covariance_add(covariance, _diagonal_covariance(confidence_variance))

    def _arrival_and_braking(
        self,
        history: Sequence[GoalForceObservation],
        goal_force: Vector2,
    ) -> tuple[float, float]:
        """Classify residual goal-force deceleration without treating it as reversal."""

        current = history[-1].velocity_xy
        assert current is not None
        speed = _vector_norm(current)
        if speed < self.config.speed_min_mps:
            return 0.0, _bounded_probability(
                1.0
                - speed / max(self.config.arrival_speed_threshold_mps, self.config.speed_min_mps)
            )
        unit = _vector_scale(current, 1.0 / speed)
        longitudinal_deceleration = -_vector_dot(goal_force, unit)
        score = _sigmoid(
            (longitudinal_deceleration - self.config.braking_acceleration_threshold_mps2)
            / self.config.braking_probability_scale_mps2
        )
        if len(history) >= 2:
            previous = history[-2].velocity_xy
            assert previous is not None
            if _vector_norm(current) < _vector_norm(previous):
                score = max(score, 0.5)
        arrival = score * max(
            0.25,
            min(
                1.0, self.config.arrival_speed_threshold_mps / max(speed, self.config.speed_min_mps)
            ),
        )
        return _bounded_probability(score), _bounded_probability(arrival)

    def _build_belief(
        self,
        history: Sequence[GoalForceObservation],
        *,
        mode: GoalBeliefMode,
        force_estimate: ForceEstimate2D | None,
        desired_velocity: Vector2 | None,
        desired_direction: float | None,
        candidate_probabilities: Sequence[GoalCandidateProbability],
        unknown_candidate_probability: float,
        arrival_probability: float,
        change_probability: float,
        track_confidence: float | None,
        censoring_state: CensoringState,
        speed_cap_status: ActorSpeedCapStatus,
        blockers: Sequence[str],
    ) -> GoalBeliefV1:
        """Construct the shared actor-only belief from estimator outputs."""

        latest = history[-1]
        normalized_blockers = _normalize_blockers(blockers)
        observation = GoalBeliefObservation(
            track_id=latest.track_id,
            tracking_epoch_id=latest.tracking_epoch_id,
            timestamp_s=latest.timestamp_s,
            step_index=latest.step_index,
            config_hash=self.config.config_hash,
            history_steps=_history_steps(history),
            coordinate_frame=CoordinateFrame.GLOBAL_XY,
            force_estimate=force_estimate,
            desired_velocity_xy=desired_velocity,
            desired_direction_rad=desired_direction,
            candidate_probabilities=tuple(candidate_probabilities),
            unknown_candidate_probability=unknown_candidate_probability,
            arrival_probability=arrival_probability,
            change_probability=change_probability,
            mode=mode,
            track_confidence=track_confidence,
            censoring_state=censoring_state,
            speed_cap_status=speed_cap_status,
            blockers=normalized_blockers,
        )
        return GoalBeliefV1.from_observation(observation)

    def _unavailable(
        self,
        history: Sequence[GoalForceObservation],
        blockers: Sequence[str],
        *,
        estimator_variant: str,
        reconstruction: ObservableForceReconstruction | None = None,
    ) -> GoalForceEstimate:
        """Return a finite unavailable result with all candidate mass unknown."""

        latest = history[-1]
        values = tuple(history)
        all_blockers = list(blockers)
        if not all_blockers:
            all_blockers.append("estimate_unavailable")
        belief = self._build_belief(
            values,
            mode=GoalBeliefMode.UNAVAILABLE,
            force_estimate=None,
            desired_velocity=None,
            desired_direction=None,
            candidate_probabilities=(),
            unknown_candidate_probability=1.0,
            arrival_probability=0.0,
            change_probability=0.0,
            track_confidence=None,
            censoring_state=CensoringState.UNKNOWN,
            speed_cap_status=ActorSpeedCapStatus.UNKNOWN,
            blockers=all_blockers,
        )
        unavailable_covariance = _project_psd(
            _diagonal_covariance(self.config.unmodeled_force_variance),
            "acceleration_covariance_xy",
            floor=self.config.covariance_floor,
            ceiling=self.config.covariance_ceiling,
        )
        return GoalForceEstimate(
            track_id=latest.track_id,
            tracking_epoch_id=latest.tracking_epoch_id,
            timestamp_s=latest.timestamp_s,
            step_index=latest.step_index,
            history_length=self.config.history_length,
            history_steps=_history_steps(values),
            mode=GoalForceInformationMode.UNAVAILABLE,
            estimator_variant=estimator_variant,
            acceleration_xy=None,
            acceleration_covariance_xy=unavailable_covariance,
            force_estimate=None,
            desired_velocity_xy=None,
            desired_direction_rad=None,
            preferred_speed_mps=None,
            relaxation_time_s=None,
            desired_force_factor=None,
            arrival_probability=0.0,
            braking_probability=0.0,
            change_probability=0.0,
            speed_cap_status=ActorSpeedCapStatus.UNKNOWN,
            censoring_state=CensoringState.UNKNOWN,
            blockers=tuple(all_blockers),
            covariance_terms=(CovarianceTerm("unavailable_force", unavailable_covariance),),
            component_diagnostics=() if reconstruction is None else reconstruction.diagnostics,
            reconstruction=reconstruction,
            config_hash=self.config.config_hash,
            belief=belief,
        )


class GoalForceTrackingAdapter:
    """Maintain observation history keyed by ``(epoch, track_id)``."""

    __slots__ = (
        "_epoch_index",
        "_histories",
        "_missed_tracks",
        "_reset_provenance",
        "config",
        "estimator",
    )

    def __init__(self, config: GoalForceInverseConfig | Mapping[str, Any] | None = None) -> None:
        """Create a stateful bridge from public tracker results."""

        self.estimator = GoalForceInverseEstimator(config)
        self.config = self.estimator.config
        self._epoch_index = 0
        self._reset_provenance: str | None = None
        self._histories: dict[tuple[str, str], list[GoalForceObservation]] = {}
        self._missed_tracks: set[tuple[str, str]] = set()

    @property
    def tracking_epoch_id(self) -> str:
        """Return the current episode-local identity epoch."""

        return str(self._epoch_index)

    def reset(self, reset_provenance: str) -> None:
        """Clear histories and advance the identity epoch."""

        self._epoch_index += 1
        self._reset_provenance = require_text(reset_provenance, "reset_provenance")
        self._histories.clear()
        self._missed_tracks.clear()

    def update(
        self,
        result: PedestrianTrackingResult,
        *,
        known_force_components_by_track: Mapping[str | int, Sequence[ObservableForceComponent]]
        | None = None,
        candidate_sets_by_track: Mapping[
            str | int, GoalCandidateSet | GoalCandidateGenerationResult
        ]
        | None = None,
        preferred_speeds_by_track: Mapping[str | int, float] | None = None,
        max_speeds_by_track: Mapping[str | int, float] | None = None,
        expected_force_component_types: Sequence[str] | None = None,
    ) -> tuple[GoalForceEstimate, ...]:
        """Update histories and estimate every public track in stable order."""

        from robot_sf.sensor.pedestrian_tracking import PedestrianTrackingResult

        if type(result) is not PedestrianTrackingResult:
            raise TypeError("result must be PedestrianTrackingResult")
        if not result.diagnostics.enabled:
            return ()
        force_mapping = known_force_components_by_track or {}
        candidate_mapping = candidate_sets_by_track or {}
        preferred_mapping = preferred_speeds_by_track or {}
        max_mapping = max_speeds_by_track or {}
        estimates: list[GoalForceEstimate] = []
        epoch = self.tracking_epoch_id
        for track in result.tracks:
            observation = GoalForceObservation.from_track(
                track,
                tracking_epoch_id=epoch,
                reset_provenance=self._reset_provenance,
            )
            key = (epoch, observation.track_id)
            prior_history = list(self._histories.get(key, ()))
            if observation.mask is ObservationMask.OBSERVED:
                if key in self._missed_tracks and prior_history:
                    observation = replace(
                        observation,
                        status="reacquired",
                        blockers=(*observation.blockers, "track_reacquired_after_gap"),
                    )
                prior_history.append(observation)
                self._histories[key] = prior_history[-self.config.history_length :]
                self._missed_tracks.discard(key)
                estimation_history = tuple(self._histories[key])
            else:
                self._missed_tracks.add(key)
                estimation_history = tuple(
                    (*prior_history, observation)[-self.config.history_length :]
                )
            components = _mapping_value(force_mapping, track.track_id, observation.track_id, ())
            candidate_set = _candidate_mapping_value(
                candidate_mapping,
                track.track_id,
                observation.track_id,
            )
            estimates.append(
                self.estimator.estimate(
                    estimation_history,
                    known_force_components=components,
                    expected_force_component_types=expected_force_component_types,
                    preferred_speed_mps=_mapping_scalar_value(
                        preferred_mapping,
                        track.track_id,
                        observation.track_id,
                    ),
                    max_speed_mps=_mapping_scalar_value(
                        max_mapping,
                        track.track_id,
                        observation.track_id,
                    ),
                    candidate_set=candidate_set,
                )
            )
        return tuple(sorted(estimates, key=lambda value: value.track_id))

    adapt = update


def _mapping_value(
    mapping: Mapping[str | int, Sequence[ObservableForceComponent]],
    numeric_id: int,
    text_id: str,
    default: Sequence[ObservableForceComponent],
) -> Sequence[ObservableForceComponent]:
    """Resolve an optional per-track component mapping."""

    return mapping.get(text_id, mapping.get(numeric_id, default))


def _mapping_scalar_value(
    mapping: Mapping[str | int, float],
    numeric_id: int,
    text_id: str,
) -> float | None:
    """Resolve an optional per-track scalar mapping."""

    value = mapping.get(text_id, mapping.get(numeric_id))
    return None if value is None else require_finite(value, f"mapping[{text_id}]")


def _candidate_mapping_value(
    mapping: Mapping[str | int, GoalCandidateSet | GoalCandidateGenerationResult],
    numeric_id: int,
    text_id: str,
) -> GoalCandidateSet | None:
    """Resolve a candidate set or a #8073 generation result."""

    value = mapping.get(text_id, mapping.get(numeric_id))
    if value is None:
        return None
    if type(value) is GoalCandidateSet:
        return value
    candidate_set = getattr(value, "candidate_set", None)
    if type(candidate_set) is GoalCandidateSet:
        return candidate_set
    raise TypeError(
        "candidate mapping values must be GoalCandidateSet or GoalCandidateGenerationResult"
    )


def estimate_oracle_component_upper_bound(
    trace: OracleTransitionTraceV1,
    *,
    config: GoalForceInverseConfig | Mapping[str, Any] | None = None,
) -> GoalForceEstimate:
    """Functional wrapper for the evaluator-only oracle upper bound."""

    return GoalForceInverseEstimator(config).estimate_from_oracle_trace(trace)


__all__ = [
    "DEFAULT_EXPECTED_FORCE_COMPONENT_TYPES",
    "GOAL_FORCE_INVERSE_CLAIM_BOUNDARY",
    "GOAL_FORCE_INVERSE_SCHEMA_VERSION",
    "CovarianceTerm",
    "ForceComponentDiagnostic",
    "GoalForceEstimate",
    "GoalForceEstimatorMode",
    "GoalForceInformationMode",
    "GoalForceInverseConfig",
    "GoalForceInverseEstimator",
    "GoalForceMode",
    "GoalForceObservation",
    "GoalForceTrackingAdapter",
    "ObservableForceComponent",
    "ObservableForceComponentType",
    "ObservableForceReconstruction",
    "estimate_oracle_component_upper_bound",
    "reconstruct_observable_force",
]
