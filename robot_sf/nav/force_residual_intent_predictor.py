"""Deterministic planner-visible force-residual pedestrian prediction.

The interaction terms here are a small analytic estimator, not a port of the
simulator's Social Force implementation. Its output is a set of inspectable
hypotheses, not recovered hidden intent or prediction-quality evidence. The
optional deceleration mode is disabled by default because no behavioral
evidence justifies enabling it for a general pedestrian population.

Call the predictor after the tracker updates its current-step
``PlannerTrackBelief`` records and before the planner consumes forecasts. Each
old mode is first scored against the current position-and-velocity observation;
only then is that observation used to update bounded acceleration history and
the next forecast. Inputs remain planner-visible and do not include simulator
oracles or future state.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from robot_sf.nav.predictive_types import MultimodalPrediction, PedestrianForecast, TrajectoryMode
from robot_sf.planner.scenario_belief_adapter import PlannerTrackBelief

_CV = "constant_velocity"
_RESIDUAL = "force_residual_desired_motion"
_YIELD = "yield_or_decelerate"
_MODE_IDS = (_CV, _RESIDUAL, _YIELD)
_SNAPSHOT_SCHEMA = "force-residual-predictor-state.v1"
_PREDICTION_SCHEMA = "force-residual-intent-predictor.v1"
_FORCE_MODEL_VERSION = "planner_visible_analytic_interactions.v1"


def _finite_xy(value: Sequence[float], name: str) -> tuple[float, float]:
    """Normalize one finite XY vector.

    Returns:
        An immutable pair of finite coordinates.
    """
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite XY pair")
    return float(array[0]), float(array[1])


def _finite_scalar(value: float, name: str, *, minimum: float | None = None) -> float:
    """Normalize a finite scalar and enforce an optional lower bound.

    Returns:
        The normalized finite scalar.
    """
    normalized = float(value)
    if not math.isfinite(normalized) or (minimum is not None and normalized < minimum):
        bound = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {bound}")
    return normalized


def _clip_vector(vector: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    """Bound vector magnitude and report whether clipping occurred.

    Returns:
        A bounded vector and a flag indicating whether it was clipped.
    """
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm == 0.0:
        return vector.copy(), False
    return vector * (maximum / norm), True


def _snapshot_covariance(value: Any, name: str) -> np.ndarray:
    """Validate a finite symmetric PSD covariance decoded from a snapshot.

    Returns:
        A validated covariance copy.
    """
    try:
        covariance = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"snapshot_invalid_covariance:{name}") from exc
    if (
        covariance.shape != (4, 4)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-8)
        or np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-7
    ):
        raise ValueError(f"snapshot_invalid_covariance:{name}")
    return covariance.copy()


@dataclass(frozen=True, slots=True)
class RobotKinematics:
    """Minimal current robot state visible to the planner-side predictor."""

    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    radius_m: float

    def __post_init__(self) -> None:
        """Normalize robot position, velocity, and radius."""
        object.__setattr__(self, "position_xy", _finite_xy(self.position_xy, "robot position"))
        object.__setattr__(self, "velocity_xy", _finite_xy(self.velocity_xy, "robot velocity"))
        object.__setattr__(
            self, "radius_m", _finite_scalar(self.radius_m, "robot radius", minimum=0.0)
        )


@dataclass(frozen=True, slots=True)
class ObstacleSegment:
    """One static line segment in the shared global XY frame."""

    obstacle_id: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    def __post_init__(self) -> None:
        """Normalize obstacle identity and endpoints."""
        obstacle_id = str(self.obstacle_id).strip()
        if not obstacle_id:
            raise ValueError("obstacle_id must be non-empty")
        object.__setattr__(self, "obstacle_id", obstacle_id)
        object.__setattr__(self, "start_xy", _finite_xy(self.start_xy, "obstacle start"))
        object.__setattr__(self, "end_xy", _finite_xy(self.end_xy, "obstacle end"))


@dataclass(frozen=True, slots=True)
class StaticGeometry:
    """Planner-visible static line segments; unavailable geometry fails closed."""

    obstacles: tuple[ObstacleSegment, ...] = field(default_factory=tuple)
    source: str = "planner_map"
    version: str = "static_geometry.v1"
    available: bool = True

    def __post_init__(self) -> None:
        """Normalize and sort static planner geometry for deterministic use."""
        obstacles = tuple(self.obstacles)
        if any(not isinstance(item, ObstacleSegment) for item in obstacles):
            raise TypeError("obstacles must contain ObstacleSegment values")
        ids = [item.obstacle_id for item in obstacles]
        if len(set(ids)) != len(ids):
            raise ValueError("obstacle_id values must be unique")
        object.__setattr__(
            self, "obstacles", tuple(sorted(obstacles, key=lambda item: item.obstacle_id))
        )
        for name in ("source", "version"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if type(self.available) is not bool:
            raise TypeError("available must be a bool")


@dataclass(frozen=True, slots=True)
class ForceResidualIntentConfig:
    """Frozen physical, history, posterior, and forecast bounds for one predictor."""

    history_steps: int = 6
    min_history_steps: int = 2
    acceleration_estimator: Literal["velocity_difference", "least_squares"] = "least_squares"
    relaxation_time_s: float = 0.6
    max_speed_mps: float = 2.5
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 2.0
    process_noise_position_m2_per_s: float = 0.02
    process_noise_velocity_m2_per_s: float = 0.08
    residual_model_noise_m2_per_s: float = 0.12
    staleness_inflation_m2_per_step: float = 0.04
    occlusion_inflation_m2: float = 0.08
    confidence_inflation_m2: float = 0.05
    mode_priors: tuple[tuple[str, float], ...] = (
        (_CV, 0.55),
        (_RESIDUAL, 0.40),
        (_YIELD, 0.05),
    )
    mode_transition_priors: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
        (_CV, ((_CV, 0.90), (_RESIDUAL, 0.09), (_YIELD, 0.01))),
        (_RESIDUAL, ((_CV, 0.10), (_RESIDUAL, 0.85), (_YIELD, 0.05))),
        (_YIELD, ((_CV, 0.15), (_RESIDUAL, 0.10), (_YIELD, 0.75))),
    )
    mode_probability_floor: float = 0.005
    likelihood_covariance_floor_m2: float = 1e-4
    likelihood_velocity_covariance_floor_m2_per_s2: float = 1e-3
    likelihood_collapse_threshold: float = -1e5
    heading_change_reset_rad: float = math.pi / 3.0
    heading_change_persistence_steps: int = 1
    stop_speed_threshold_mps: float = 0.05
    restart_speed_threshold_mps: float = 0.18
    max_coast_steps_for_intent: int = 3
    max_modes: int = 3
    prediction_horizon_steps: int = 8
    prediction_dt_s: float = 0.1
    max_update_dt_s: float = 10.0
    max_tracks: int = 128
    max_static_obstacles: int = 512
    yield_mode_enabled: bool = False
    yield_deceleration_mps2: float = 0.8
    pedestrian_interaction_strength_mps2: float = 1.2
    robot_interaction_strength_mps2: float = 1.5
    obstacle_interaction_strength_mps2: float = 1.2
    interaction_decay_length_m: float = 0.7
    max_interaction_acceleration_mps2: float = 3.0
    degenerate_distance_m: float = 1e-6
    force_model_version: str = _FORCE_MODEL_VERSION

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Reject unbounded histories, invalid physical limits, and malformed priors."""
        if type(self.history_steps) is not int or not 2 <= self.history_steps <= 256:
            raise ValueError("history_steps must be an integer in [2, 256]")
        if (
            type(self.min_history_steps) is not int
            or not 2 <= self.min_history_steps <= self.history_steps
        ):
            raise ValueError("min_history_steps must be between 2 and history_steps")
        if self.acceleration_estimator not in {"velocity_difference", "least_squares"}:
            raise ValueError("unsupported acceleration_estimator")
        if type(self.max_modes) is not int or self.max_modes not in {2, 3}:
            raise ValueError("max_modes must be 2 or 3")
        if self.yield_mode_enabled and self.max_modes < 3:
            raise ValueError("yield mode requires max_modes=3")
        for name in (
            "relaxation_time_s",
            "max_speed_mps",
            "max_acceleration_mps2",
            "max_deceleration_mps2",
            "likelihood_covariance_floor_m2",
            "likelihood_velocity_covariance_floor_m2_per_s2",
            "heading_change_reset_rad",
            "restart_speed_threshold_mps",
            "prediction_dt_s",
            "max_update_dt_s",
            "yield_deceleration_mps2",
            "interaction_decay_length_m",
            "max_interaction_acceleration_mps2",
            "degenerate_distance_m",
        ):
            value = _finite_scalar(getattr(self, name), name, minimum=0.0)
            if value == 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "process_noise_position_m2_per_s",
            "process_noise_velocity_m2_per_s",
            "residual_model_noise_m2_per_s",
            "staleness_inflation_m2_per_step",
            "occlusion_inflation_m2",
            "confidence_inflation_m2",
            "stop_speed_threshold_mps",
            "pedestrian_interaction_strength_mps2",
            "robot_interaction_strength_mps2",
            "obstacle_interaction_strength_mps2",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name, minimum=0.0),
            )
        if (
            type(self.prediction_horizon_steps) is not int
            or not 1 <= self.prediction_horizon_steps <= 1000
        ):
            raise ValueError("prediction_horizon_steps must be an integer in [1, 1000]")
        if type(self.max_tracks) is not int or not 1 <= self.max_tracks <= 512:
            raise ValueError("max_tracks must be an integer in [1, 512]")
        if type(self.max_static_obstacles) is not int or not 0 <= self.max_static_obstacles <= 4096:
            raise ValueError("max_static_obstacles must be an integer in [0, 4096]")
        if type(self.max_coast_steps_for_intent) is not int or self.max_coast_steps_for_intent < 0:
            raise ValueError("max_coast_steps_for_intent must be a non-negative integer")
        if (
            type(self.heading_change_persistence_steps) is not int
            or self.heading_change_persistence_steps < 1
        ):
            raise ValueError("heading_change_persistence_steps must be an integer >= 1")
        if self.restart_speed_threshold_mps <= self.stop_speed_threshold_mps:
            raise ValueError("restart speed threshold must exceed stop speed threshold")
        if not math.isfinite(self.likelihood_collapse_threshold):
            raise ValueError("likelihood_collapse_threshold must be finite")
        if not 0.0 < self.mode_probability_floor < 1.0 / self.max_modes:
            raise ValueError("mode_probability_floor must be in (0, 1/max_modes)")
        if self.heading_change_reset_rad > math.pi:
            raise ValueError("heading_change_reset_rad must be <= pi")
        priors = tuple((str(name), float(probability)) for name, probability in self.mode_priors)
        if len({name for name, _ in priors}) != len(priors) or {name for name, _ in priors} != set(
            _MODE_IDS
        ):
            raise ValueError("mode_priors must define each canonical mode exactly once")
        if any(not math.isfinite(probability) or probability < 0.0 for _, probability in priors):
            raise ValueError("mode priors must be finite and non-negative")
        if not math.isclose(sum(probability for _, probability in priors), 1.0, abs_tol=1e-8):
            raise ValueError("mode priors must sum to 1")
        if dict(priors)[_CV] <= 0.0 or dict(priors)[_RESIDUAL] <= 0.0:
            raise ValueError("CV and residual mode priors must be positive")
        if self.yield_mode_enabled and dict(priors)[_YIELD] <= 0.0:
            raise ValueError("enabled yield mode requires a positive prior")
        object.__setattr__(self, "mode_priors", tuple(sorted(priors)))
        transitions = self._validated_transition_priors(self.mode_transition_priors)
        object.__setattr__(self, "mode_transition_priors", transitions)
        version = str(self.force_model_version).strip()
        if not version:
            raise ValueError("force_model_version must be non-empty")
        object.__setattr__(self, "force_model_version", version)

    @staticmethod
    def _validated_transition_priors(
        values: tuple[tuple[str, tuple[tuple[str, float], ...]], ...],
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        """Validate complete row-stochastic transition priors.

        Returns:
            Canonically sorted source and destination rows.
        """
        rows: dict[str, tuple[tuple[str, float], ...]] = {}
        for source, destinations in values:
            source_id = str(source)
            row = tuple((str(mode), float(probability)) for mode, probability in destinations)
            destination_ids = {mode for mode, _ in row}
            if source_id in rows or destination_ids != set(_MODE_IDS) or len(row) != len(_MODE_IDS):
                raise ValueError("mode_transition_priors must define every canonical mode once")
            if any(not math.isfinite(probability) or probability < 0.0 for _, probability in row):
                raise ValueError("mode transition priors must be finite and non-negative")
            if not math.isclose(sum(probability for _, probability in row), 1.0, abs_tol=1e-8):
                raise ValueError("each mode transition prior row must sum to 1")
            rows[source_id] = tuple(sorted(row))
        if set(rows) != set(_MODE_IDS):
            raise ValueError("mode_transition_priors must define every canonical mode once")
        return tuple((source, rows[source]) for source in sorted(rows))

    @property
    def config_hash(self) -> str:
        """Return a deterministic SHA-256 hash over every validated setting."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def prior_for(self, mode_id: str) -> float:
        """Return the configured prior for one canonical mode."""
        return dict(self.mode_priors)[mode_id]

    def transition_for(self, source_mode: str, target_mode: str) -> float:
        """Return the configured transition probability between canonical modes."""
        rows = dict(self.mode_transition_priors)
        return dict(rows[source_mode])[target_mode]


@dataclass(frozen=True, slots=True)
class _Observation:
    step: int
    time_s: float
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    confidence: float
    age_steps: int
    missed_steps: int


@dataclass(slots=True)
class _TrackIntentState:
    track_id: int
    tracker_namespace: str
    reset_epoch: int
    lifecycle_token: str
    history: deque[_Observation]
    log_probabilities: dict[str, float]
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    confidence: float
    age_steps: int
    missed_steps: int
    visible: bool
    last_update_step: int
    last_visible_step: int | None
    desired_velocity: np.ndarray | None
    reset_count: int = 0
    last_reset_reason: str | None = None
    heading_change_count: int = 0
    interaction_terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    measured_acceleration: np.ndarray | None = None
    estimator_status: str = "insufficient_history"


class ForceResidualIntentPredictor:
    """Produce bounded CV/residual hypotheses from planner-visible track history.

    ``dt`` is the positive elapsed time since the preceding accepted update (or
    the first sample's time scale after reset). Pass ``simulation_timestamp_s``
    when the caller owns an absolute simulation clock; otherwise the shared
    prediction timestamp uses the shared ``-1.0`` unavailable sentinel and
    reset-relative elapsed time is available only in metadata.
    """

    def __init__(self, config: ForceResidualIntentConfig | None = None) -> None:
        """Create a stateful predictor with immutable validated configuration."""
        self.config = config or ForceResidualIntentConfig()
        self._states: dict[str, _TrackIntentState] = {}
        self._last_step: int | None = None
        self._elapsed_time_s = 0.0
        self._last_dt_s: float | None = None
        self._seed: int | None = None
        self._diagnostics: dict[str, Any] = self._base_diagnostics("uninitialized")

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """Return a JSON-safe copy of the latest bounded diagnostic record."""
        return MappingProxyType(json.loads(json.dumps(self._diagnostics, sort_keys=True)))

    def reset(self, *, seed: int | None = None) -> None:
        """Clear all histories and posteriors; the seed is provenance only."""
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        self._states.clear()
        self._last_step = None
        self._elapsed_time_s = 0.0
        self._last_dt_s = None
        self._seed = seed
        self._diagnostics = self._base_diagnostics("reset")
        self._diagnostics["reset_reason"] = "explicit_reset"

    def update_and_predict(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        step: int,
        dt: float,
        tracks: Mapping[int, PlannerTrackBelief],
        robot_state: RobotKinematics,
        static_geometry: StaticGeometry,
        simulation_timestamp_s: float | None = None,
    ) -> MultimodalPrediction:
        """Score current beliefs, update posteriors, and emit mode forecasts.

        Invoke once per accepted, strictly increasing belief step after the
        tracker update. Existing modes are scored from the prior state against
        current position and velocity before this observation updates the
        acceleration history; resulting forecasts are then available to the
        planner for that step.
        ``simulation_timestamp_s`` is an optional non-negative absolute clock
        value. When omitted, the shared prediction carries its ``-1.0`` unavailable
        timestamp sentinel while reset-relative elapsed time remains metadata.

        Returns:
            A multimodal forecast for the accepted step.
        """
        started = time.perf_counter()
        try:
            self._validate_step(step, dt)
            normalized_tracks = self._validate_tracks(step, tracks)
            absolute_timestamp = (
                None
                if simulation_timestamp_s is None
                else _finite_scalar(
                    simulation_timestamp_s,
                    "simulation_timestamp_s",
                    minimum=0.0,
                )
            )
            if not isinstance(robot_state, RobotKinematics):
                raise ValueError("invalid_robot_kinematics")
            if not isinstance(static_geometry, StaticGeometry):
                raise ValueError("invalid_static_geometry")
            if len(static_geometry.obstacles) > self.config.max_static_obstacles:
                raise ValueError("static_obstacle_limit_exceeded")
        except (TypeError, ValueError) as exc:
            reason = str(exc) or "invalid_input"
            self._diagnostics = self._base_diagnostics("rejected")
            self._diagnostics["reason"] = reason
            raise ValueError(reason) from exc

        if self._last_step is not None and step == self._last_step:
            self._reject("duplicate_step", step)
        if self._last_step is not None and step < self._last_step:
            self._reject("out_of_order_step", step)

        interval = float(dt)
        if self._last_step is None:
            current_time_s = 0.0
        else:
            current_time_s = self._elapsed_time_s + interval

        prior_keys_by_id = {state.track_id: key for key, state in self._states.items()}
        forecasts: dict[int, PedestrianForecast] = {}
        track_diagnostics: dict[str, Any] = {}
        reset_events: list[dict[str, Any]] = []
        for track_id, track in normalized_tracks:
            identity = track.lifecycle_token
            previous_key = prior_keys_by_id.get(track_id)
            generation_changed = previous_key is not None and previous_key != identity
            state = self._states.get(identity)
            if generation_changed:
                old_state = self._states.pop(previous_key, None)
                state = None
                inherited_reset_count = 0 if old_state is None else old_state.reset_count + 1
                reset_reason = "identity_generation_changed"
            else:
                inherited_reset_count = 0
                reset_reason = "initial_track" if state is None else None

            position, velocity, covariance = self._track_arrays(track)
            if state is not None:
                reset_reason = self._observable_reset_reason(
                    state,
                    track,
                    step=step,
                    position=position,
                    velocity=velocity,
                    interval_s=interval,
                )
                if reset_reason is not None:
                    inherited_reset_count = state.reset_count + 1
                    state = None

            prior_probabilities = (
                {}
                if state is None
                else self._transition_probabilities(self._probabilities(state.log_probabilities))
            )
            log_likelihoods: dict[str, float | None] = {}
            mode_prediction_details: dict[str, dict[str, Any]] = {}
            if state is not None and track.visibility:
                log_likelihoods, mode_prediction_details = self._score_previous_modes(
                    state,
                    position,
                    velocity,
                    covariance,
                    interval,
                )
                if log_likelihoods and all(
                    value is not None and value < self.config.likelihood_collapse_threshold
                    for value in log_likelihoods.values()
                ):
                    reset_reason = "likelihood_collapse"
                    inherited_reset_count = state.reset_count + 1
                    state = None
                    prior_probabilities = {}

            if state is None:
                history: deque[_Observation] = deque(maxlen=self.config.history_steps)
                if track.visibility:
                    history.append(
                        _Observation(
                            step=step,
                            time_s=current_time_s,
                            position=position.copy(),
                            velocity=velocity.copy(),
                            covariance=covariance.copy(),
                            confidence=track.confidence,
                            age_steps=track.age_steps,
                            missed_steps=track.missed_steps,
                        )
                    )
                state = _TrackIntentState(
                    track_id=track_id,
                    tracker_namespace=track.tracker_namespace,
                    reset_epoch=track.reset_epoch,
                    lifecycle_token=identity,
                    history=history,
                    log_probabilities={},
                    position=position.copy(),
                    velocity=velocity.copy(),
                    covariance=covariance.copy(),
                    confidence=track.confidence,
                    age_steps=track.age_steps,
                    missed_steps=track.missed_steps,
                    visible=track.visibility,
                    last_update_step=step,
                    last_visible_step=step
                    if track.visibility
                    else max(0, step - track.missed_steps),
                    desired_velocity=None,
                    reset_count=inherited_reset_count,
                    last_reset_reason=reset_reason,
                )
            else:
                state.position = position.copy()
                state.velocity = velocity.copy()
                state.covariance = covariance.copy()
                state.confidence = track.confidence
                state.age_steps = track.age_steps
                state.missed_steps = track.missed_steps
                state.visible = track.visibility
                state.last_update_step = step
                if track.visibility:
                    state.history.append(
                        _Observation(
                            step=step,
                            time_s=current_time_s,
                            position=position.copy(),
                            velocity=velocity.copy(),
                            covariance=covariance.copy(),
                            confidence=track.confidence,
                            age_steps=track.age_steps,
                            missed_steps=track.missed_steps,
                        )
                    )
                    state.last_visible_step = step

            terms = self._interaction_terms(track, normalized_tracks, robot_state, static_geometry)
            state.interaction_terms = terms
            state.measured_acceleration = None
            state.estimator_status = "insufficient_history"
            residual_available = False
            interaction_available = all(
                term["status"].startswith("available") for term in terms.values()
            )
            interaction_acceleration = (
                sum(
                    (np.asarray(term["acceleration_mps2"], dtype=float) for term in terms.values()),
                    start=np.zeros(2, dtype=float),
                )
                if interaction_available
                else None
            )
            measured_status = (
                "coasted_no_measurement" if not track.visibility else "insufficient_history"
            )
            residual_status = measured_status
            inferred_velocity = None
            acceleration_clipped: bool | None = None
            desired_speed_clipped: bool | None = None
            residual_acceleration: np.ndarray | None = None
            measured: np.ndarray | None = None
            if track.visibility and len(state.history) >= self.config.min_history_steps:
                measured, estimator_status = self._estimate_acceleration(state.history)
                state.measured_acceleration = measured
                measured_status = estimator_status
                state.estimator_status = estimator_status
                if measured is not None and interaction_acceleration is not None:
                    residual_acceleration, clipped = _clip_vector(
                        measured - interaction_acceleration,
                        self.config.max_acceleration_mps2,
                    )
                    desired_velocity = (
                        velocity + self.config.relaxation_time_s * residual_acceleration
                    )
                    desired_velocity, speed_clipped = _clip_vector(
                        desired_velocity,
                        self.config.max_speed_mps,
                    )
                    state.desired_velocity = desired_velocity
                    inferred_velocity = desired_velocity
                    residual_available = True
                    acceleration_clipped = clipped
                    desired_speed_clipped = speed_clipped
                    residual_status = "available"
                    state.estimator_status = "valid"
                else:
                    state.desired_velocity = None
                    if measured is not None:
                        state.estimator_status = "interaction_unavailable"
                        residual_status = "interaction_unavailable"
                    else:
                        residual_status = estimator_status
            elif not track.visibility and state.desired_velocity is not None:
                residual_available = True
                inferred_velocity = state.desired_velocity
                residual_status = "coasted_retained"

            if inferred_velocity is None and not track.visibility:
                inferred_velocity = state.desired_velocity
            observation_ages_steps = [max(0, step - item.step) for item in state.history]
            observation_ages_s = [max(0.0, current_time_s - item.time_s) for item in state.history]
            track_diagnostics[str(track_id)] = {
                "tracker_age_steps": track.age_steps,
                "observation_ages_steps": observation_ages_steps,
                "observation_ages_s": observation_ages_s,
                "measured_acceleration_mps2": None if measured is None else measured.tolist(),
                "measured_acceleration_status": measured_status,
                "interaction_acceleration_mps2": (
                    None if interaction_acceleration is None else interaction_acceleration.tolist()
                ),
                "interaction_acceleration_status": (
                    "available" if interaction_available else "unavailable"
                ),
                "residual_acceleration_mps2": (
                    None if residual_acceleration is None else residual_acceleration.tolist()
                ),
                "residual_acceleration_status": residual_status,
                "inferred_desired_velocity_mps": (
                    None if inferred_velocity is None else inferred_velocity.tolist()
                ),
                "inferred_desired_velocity_status": (
                    "valid"
                    if residual_available and track.visibility
                    else "coasted_retained"
                    if inferred_velocity is not None and not track.visibility
                    else "unavailable"
                ),
                "acceleration_clipped": acceleration_clipped,
                "desired_speed_clipped": desired_speed_clipped,
                "estimator_status": state.estimator_status,
            }

            active_modes = [_CV]
            if residual_available and self.config.max_modes >= 2:
                active_modes.append(_RESIDUAL)
            if self.config.yield_mode_enabled and self.config.max_modes >= 3:
                active_modes.append(_YIELD)

            updated_probabilities, posterior_diagnostics = self._update_posterior(
                state,
                active_modes,
                prior_probabilities=prior_probabilities,
                log_likelihoods=log_likelihoods,
                reset=state.last_reset_reason is not None
                and state.last_reset_reason == reset_reason,
                prediction_details=mode_prediction_details,
            )
            state.log_probabilities = {
                mode_id: math.log(probability)
                for mode_id, probability in updated_probabilities.items()
            }
            state.last_reset_reason = (
                reset_reason if reset_reason is not None else state.last_reset_reason
            )
            state.reset_count = max(state.reset_count, inherited_reset_count)

            inflated_covariance, inflation = self._inflate_input_covariance(track, covariance)
            forecast_diagnostics, forecast_modes = self._build_modes(
                track,
                position,
                velocity,
                inflated_covariance,
                updated_probabilities,
                state.desired_velocity if residual_available else None,
            )
            forecast_metadata = {
                "identity_token": track.lifecycle_token,
                "tracker_namespace": track.tracker_namespace,
                "reset_epoch": track.reset_epoch,
                "source": "force_residual_intent_predictor",
                "force_model": {
                    "source": "analytic_estimator",
                    "version": self.config.force_model_version,
                    "units": "acceleration_like_m/s^2",
                    "simulator_parity_claim": False,
                },
                "prediction_status": self._prediction_status(track, state),
                "reset_reason": state.last_reset_reason,
            }
            forecasts[track_id] = PedestrianForecast(
                pedestrian_id=track_id,
                modes=tuple(forecast_modes),
                existence_probability=track.existence_probability,
                source_confidence=track.confidence,
                age_steps=track.age_steps,
                metadata=forecast_metadata,
            )
            track_diagnostics[str(track_id)] = {
                **track_diagnostics.get(str(track_id), {}),
                "identity_token": track.lifecycle_token,
                "history_length": len(state.history),
                "last_visible_step": state.last_visible_step,
                "visible_now": track.visibility,
                "mode_probabilities": dict(sorted(updated_probabilities.items())),
                "mode_update": posterior_diagnostics,
                "forecast_modes": forecast_diagnostics,
                "interaction_terms": terms,
                "covariance_inflation_m2": inflation,
                "prediction_status": self._prediction_status(track, state),
                "reset_reason": state.last_reset_reason,
                "reset_count": state.reset_count,
                "force_model": {
                    "source": "analytic_estimator",
                    "version": self.config.force_model_version,
                    "units": "acceleration_like_m/s^2",
                },
            }
            if reset_reason is not None:
                reset_events.append(
                    {"track_id": track_id, "identity_token": identity, "reason": reset_reason}
                )
            self._states[identity] = state

        self._prune_missing_states(step, {track.lifecycle_token for _, track in normalized_tracks})
        if self._last_step is not None:
            self._elapsed_time_s = current_time_s
            self._last_dt_s = interval
        self._last_step = step
        prediction = MultimodalPrediction(
            forecasts=forecasts,
            prediction_horizon=self.config.prediction_horizon_steps * self.config.prediction_dt_s,
            prediction_dt=self.config.prediction_dt_s,
            timestamp=-1.0 if absolute_timestamp is None else absolute_timestamp,
            schema_version="multimodal-prediction.v1",
            metadata={
                "predictor_schema": _PREDICTION_SCHEMA,
                "config_hash": self.config.config_hash,
                "step": step,
                "time_semantics": "elapsed_since_predictor_reset_s",
                "elapsed_since_predictor_reset_s": self._elapsed_time_s,
                "simulation_timestamp_available": absolute_timestamp is not None,
                "force_model": {
                    "source": "analytic_estimator",
                    "version": self.config.force_model_version,
                    "units": "acceleration_like_m/s^2",
                    "simulator_parity_claim": False,
                },
            },
        )
        self._diagnostics = self._base_diagnostics("ok")
        self._diagnostics.update(
            {
                "step": step,
                "elapsed_time_s": self._elapsed_time_s,
                "track_count": len(normalized_tracks),
                "forecast_count": len(forecasts),
                "reset_events": reset_events,
                "tracks": dict(sorted(track_diagnostics.items())),
                "stage_duration_ms": (time.perf_counter() - started) * 1000.0,
                "bounded_history_steps": self.config.history_steps,
                "work_cap_status": "within_configured_bounds",
            }
        )
        return prediction

    def snapshot_state(self) -> dict[str, Any]:
        """Return a portable JSON-compatible snapshot bound to this config hash."""
        states = [self._state_to_dict(self._states[key]) for key in sorted(self._states)]
        return json.loads(
            json.dumps(
                {
                    "schema": _SNAPSHOT_SCHEMA,
                    "config_hash": self.config.config_hash,
                    "seed": self._seed,
                    "last_step": self._last_step,
                    "elapsed_time_s": self._elapsed_time_s,
                    "last_dt_s": self._last_dt_s,
                    "states": states,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a portable snapshot only when schema and configuration match."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot must be a mapping")
        if snapshot.get("schema") != _SNAPSHOT_SCHEMA:
            raise ValueError("snapshot_schema_mismatch")
        if snapshot.get("config_hash") != self.config.config_hash:
            raise ValueError("snapshot_config_hash_mismatch")
        last_step = snapshot.get("last_step")
        if last_step is not None and (type(last_step) is not int or last_step < 0):
            raise ValueError("snapshot_invalid_last_step")
        elapsed = _finite_scalar(
            snapshot.get("elapsed_time_s", 0.0), "snapshot elapsed time", minimum=0.0
        )
        last_dt_raw = snapshot.get("last_dt_s")
        last_dt = (
            None if last_dt_raw is None else _finite_scalar(last_dt_raw, "snapshot dt", minimum=0.0)
        )
        restored: dict[str, _TrackIntentState] = {}
        restored_track_ids: set[int] = set()
        for payload in snapshot.get("states", []):
            state = self._state_from_dict(payload)
            if state.lifecycle_token in restored:
                raise ValueError("snapshot_duplicate_identity")
            if state.track_id in restored_track_ids:
                raise ValueError("snapshot_duplicate_track_id")
            restored[state.lifecycle_token] = state
            restored_track_ids.add(state.track_id)
        self._states = restored
        self._last_step = last_step
        self._elapsed_time_s = elapsed
        self._last_dt_s = last_dt
        self._seed = snapshot.get("seed")
        self._diagnostics = self._base_diagnostics("restored")
        self._diagnostics["last_step"] = self._last_step

    def _base_diagnostics(self, status: str) -> dict[str, Any]:
        return {
            "schema": "force-residual-predictor-diagnostics.v1",
            "status": status,
            "config_hash": self.config.config_hash,
            "force_model": {
                "source": "analytic_estimator",
                "version": self.config.force_model_version,
                "units": "acceleration_like_m/s^2",
                "simulator_parity_claim": False,
            },
            "yield_mode_enabled": self.config.yield_mode_enabled,
        }

    def _reject(self, reason: str, step: int) -> None:
        self._diagnostics = self._base_diagnostics("rejected")
        self._diagnostics.update({"step": step, "reason": reason})
        raise ValueError(reason)

    def _validate_step(self, step: int, dt: float) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("invalid_step")
        try:
            interval = _finite_scalar(dt, "dt", minimum=0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_dt") from exc
        if interval <= 0.0 or interval > self.config.max_update_dt_s:
            raise ValueError("invalid_dt")

    def _validate_tracks(  # noqa: C901
        self,
        step: int,
        tracks: Mapping[int, PlannerTrackBelief],
    ) -> list[tuple[int, PlannerTrackBelief]]:
        if not isinstance(tracks, Mapping):
            raise ValueError("invalid_tracks_mapping")
        if len(tracks) > self.config.max_tracks:
            raise ValueError("track_limit_exceeded")
        normalized: list[tuple[int, PlannerTrackBelief]] = []
        tokens: set[str] = set()
        for key in sorted(tracks, key=lambda value: (type(value).__name__, repr(value))):
            track = tracks[key]
            if type(key) is not int or key < 1 or not isinstance(track, PlannerTrackBelief):
                raise ValueError("invalid_track_record")
            if track.track_id != key or track.belief_step != step:
                raise ValueError("track_identity_or_step_mismatch")
            if track.lifecycle_token in tokens:
                raise ValueError("duplicate_lifecycle_token")
            tokens.add(track.lifecycle_token)
            mean = np.asarray(track.mean_state, dtype=float)
            covariance = np.asarray(track.covariance, dtype=float)
            if mean.shape != (5,) or covariance.shape != (4, 4):
                raise ValueError("invalid_track_state_shape")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
                raise ValueError("non_finite_track_state")
            if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-8):
                raise ValueError("non_symmetric_track_covariance")
            if np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-7:
                raise ValueError("non_psd_track_covariance")
            if not 0.0 <= track.confidence <= 1.0:
                raise ValueError("invalid_track_confidence")
            normalized.append((key, track))
        return normalized

    def _track_arrays(self, track: PlannerTrackBelief) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.asarray(track.mean_state, dtype=float)
        return mean[:2].copy(), mean[2:4].copy(), np.asarray(track.covariance, dtype=float).copy()

    def _observable_reset_reason(
        self,
        state: _TrackIntentState,
        track: PlannerTrackBelief,
        *,
        step: int,
        position: np.ndarray,
        velocity: np.ndarray,
        interval_s: float,
    ) -> str | None:
        if state.last_visible_step is not None and track.visibility:
            if step - state.last_visible_step > self.config.max_coast_steps_for_intent:
                return "reappearance_after_gap"
        if self._last_dt_s is not None and interval_s > max(self._last_dt_s * 5.0, 0.5):
            return "timestep_discontinuity"
        old_speed = float(np.linalg.norm(state.velocity))
        new_speed = float(np.linalg.norm(velocity))
        if (
            old_speed <= self.config.stop_speed_threshold_mps
            and new_speed >= self.config.restart_speed_threshold_mps
        ):
            return "stop_restart"
        if (
            old_speed > self.config.stop_speed_threshold_mps
            and new_speed > self.config.stop_speed_threshold_mps
        ):
            cosine = float(np.dot(state.velocity, velocity)) / (old_speed * new_speed)
            angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))
            state.heading_change_count = (
                state.heading_change_count + 1
                if angle >= self.config.heading_change_reset_rad
                else 0
            )
            if state.heading_change_count >= self.config.heading_change_persistence_steps:
                return "heading_change"
        return None

    def _score_previous_modes(
        self,
        state: _TrackIntentState,
        observed_position: np.ndarray,
        observed_velocity: np.ndarray,
        observed_covariance: np.ndarray,
        dt: float,
    ) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
        """Score each old mode against the current position and velocity observation.

        The one-step prediction is produced from the prior state before the current
        observation is appended to history or used to refit residual acceleration.
        Position uses semi-implicit Euler, matching the forecast integrator.

        Returns:
            Per-mode log likelihoods and their four-state prediction diagnostics.
        """
        scores: dict[str, float | None] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for mode_id in sorted(state.log_probabilities):
            if mode_id == _RESIDUAL and state.desired_velocity is None:
                scores[mode_id] = None
                diagnostics[mode_id] = self._unscored_mode_diagnostic(
                    "residual_unavailable_before_observation"
                )
                continue
            predicted_position, predicted_velocity = self._one_step_mean(state, mode_id, dt)
            transition = self._mode_transition(
                dt,
                mode_id,
                state.velocity,
                state.desired_velocity,
            )
            predicted_covariance = transition @ state.covariance @ transition.T
            predicted_covariance += self._process_noise(dt, mode_id)
            predicted_mean = np.concatenate((predicted_position, predicted_velocity))
            observed_mean = np.concatenate((observed_position, observed_velocity))
            innovation_covariance = (
                predicted_covariance
                + observed_covariance
                + np.diag(
                    [
                        self.config.likelihood_covariance_floor_m2,
                        self.config.likelihood_covariance_floor_m2,
                        self.config.likelihood_velocity_covariance_floor_m2_per_s2,
                        self.config.likelihood_velocity_covariance_floor_m2_per_s2,
                    ]
                )
            )
            innovation = observed_mean - predicted_mean
            sign, logdet = np.linalg.slogdet(innovation_covariance)
            if sign <= 0 or not np.isfinite(logdet):
                scores[mode_id] = -1e300
                diagnostics[mode_id] = self._mode_prediction_diagnostic(
                    predicted_mean,
                    innovation,
                    innovation_covariance,
                    "invalid_innovation_covariance",
                )
                continue
            mahalanobis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
            score = -0.5 * (mahalanobis + logdet + 4.0 * math.log(2.0 * math.pi))
            score = score if math.isfinite(score) else -1e300
            scores[mode_id] = score
            diagnostics[mode_id] = self._mode_prediction_diagnostic(
                predicted_mean,
                innovation,
                innovation_covariance,
                "scored",
            )
        return scores, diagnostics

    @staticmethod
    def _unscored_mode_diagnostic(reason: str) -> dict[str, Any]:
        """Describe a mode without a valid previous one-step forecast.

        Returns:
            JSON-safe diagnostics with explicit unavailable prediction fields.
        """
        return {
            "one_step_predicted_mean": None,
            "one_step_predicted_state": None,
            "one_step_predicted_position": None,
            "one_step_predicted_velocity": None,
            "innovation": None,
            "position_innovation": None,
            "velocity_innovation": None,
            "innovation_covariance": None,
            "observation_dimension": 4,
            "prediction_status": reason,
        }

    @staticmethod
    def _mode_prediction_diagnostic(
        predicted_mean: np.ndarray,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
        status: str,
    ) -> dict[str, Any]:
        """Serialize one-step likelihood inputs for replay inspection.

        Returns:
            JSON-safe four-state prediction, innovation, and covariance fields.
        """
        predicted_position = predicted_mean[:2]
        predicted_velocity = predicted_mean[2:]
        return {
            "one_step_predicted_mean": predicted_mean.tolist(),
            "one_step_predicted_state": predicted_mean.tolist(),
            "one_step_predicted_position": predicted_position.tolist(),
            "one_step_predicted_velocity": predicted_velocity.tolist(),
            "innovation": innovation.tolist(),
            "position_innovation": innovation[:2].tolist(),
            "velocity_innovation": innovation[2:].tolist(),
            "innovation_covariance": innovation_covariance.tolist(),
            "observation_dimension": 4,
            "prediction_status": status,
        }

    def _one_step_mean(
        self,
        state: _TrackIntentState,
        mode_id: str,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict one step before observing current data.

        Returns:
            Predicted position and velocity.
        """
        velocity, _, _ = self._mode_step_velocity(
            mode_id,
            state.velocity,
            state.desired_velocity,
            dt,
        )
        position = state.position + dt * velocity
        return position, velocity

    def _update_posterior(
        self,
        state: _TrackIntentState,
        active_modes: list[str],
        *,
        prior_probabilities: dict[str, float],
        log_likelihoods: dict[str, float | None],
        reset: bool,
        prediction_details: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        logs: dict[str, float] = {}
        per_mode: dict[str, Any] = {}
        for mode_id in active_modes:
            previous = prior_probabilities.get(mode_id)
            prior = previous if previous is not None else self.config.prior_for(mode_id)
            likelihood = log_likelihoods.get(mode_id)
            if reset:
                likelihood_value = 0.0
                update_status = "reset_prior"
            elif not state.visible:
                likelihood_value = 0.0
                update_status = "coasted_no_measurement_update"
            elif likelihood is None:
                likelihood_value = 0.0
                update_status = (
                    "prior_only_new_mode" if previous is None else "prior_only_unscored_mode"
                )
            else:
                likelihood_value = likelihood
                update_status = "predict_then_update"
            logs[mode_id] = (
                math.log(max(prior, self.config.mode_probability_floor)) + likelihood_value
            )
            per_mode[mode_id] = {
                "prior_probability": float(prior),
                "log_likelihood": likelihood,
                "update_status": update_status,
                **prediction_details.get(
                    mode_id,
                    self._unscored_mode_diagnostic(
                        "no_measurement_to_score"
                        if not state.visible
                        else "no_previous_mode_prediction"
                    ),
                ),
            }
        probabilities = self._normalize_log_probabilities(logs)
        count = len(probabilities)
        floor = min(self.config.mode_probability_floor, 0.5 / count)
        floor_applied = {key: value < floor for key, value in probabilities.items()}
        probabilities = {key: max(value, floor) for key, value in probabilities.items()}
        total = sum(probabilities.values())
        probabilities = {key: value / total for key, value in probabilities.items()}
        for mode_id, probability in probabilities.items():
            per_mode[mode_id]["posterior_probability"] = float(probability)
            per_mode[mode_id]["probability_floor"] = floor
            per_mode[mode_id]["probability_floor_applied"] = floor_applied[mode_id]
            per_mode[mode_id]["state_smoothing_applied"] = False
        return probabilities, per_mode

    def _normalize_log_probabilities(self, values: Mapping[str, float]) -> dict[str, float]:
        maximum = max(values.values())
        if not math.isfinite(maximum):
            uniform = 1.0 / len(values)
            return dict.fromkeys(sorted(values), uniform)
        weights = {key: math.exp(value - maximum) for key, value in values.items()}
        total = sum(weights.values())
        return {key: weights[key] / total for key in sorted(weights)}

    def _probabilities(self, log_probabilities: Mapping[str, float]) -> dict[str, float]:
        """Convert stored log probabilities to a stable simplex.

        Returns:
            A normalized probability mapping.
        """
        if not log_probabilities:
            return {}
        return self._normalize_log_probabilities(log_probabilities)

    def _transition_probabilities(
        self, previous_probabilities: Mapping[str, float]
    ) -> dict[str, float]:
        """Apply the configured semantic mode transition matrix to prior mass.

        Returns:
            The normalized transitioned probability distribution.
        """
        if not previous_probabilities:
            return {}
        transitioned = {
            target: sum(
                probability * self.config.transition_for(source, target)
                for source, probability in previous_probabilities.items()
            )
            for target in _MODE_IDS
        }
        total = sum(transitioned.values())
        if total <= 0.0 or not math.isfinite(total):
            return {mode_id: self.config.prior_for(mode_id) for mode_id in _MODE_IDS}
        return {mode_id: transitioned[mode_id] / total for mode_id in _MODE_IDS}

    def _interaction_terms(
        self,
        track: PlannerTrackBelief,
        tracks: list[tuple[int, PlannerTrackBelief]],
        robot: RobotKinematics,
        geometry: StaticGeometry,
    ) -> dict[str, dict[str, Any]]:
        position = np.asarray(track.mean_state[:2], dtype=float)
        radius = float(track.mean_state[4])
        pedestrian_acceleration = np.zeros(2, dtype=float)
        pedestrian_count = 0
        degenerate = False
        for other_id, other in tracks:
            if other_id == track.track_id:
                continue
            other_position = np.asarray(other.mean_state[:2], dtype=float)
            vector = self._repulsion(
                position,
                other_position,
                radius + float(other.mean_state[4]),
                self.config.pedestrian_interaction_strength_mps2,
            )
            if vector is None:
                degenerate = True
                continue
            pedestrian_acceleration += vector
            pedestrian_count += 1
        pedestrian_acceleration, pedestrian_clipped = _clip_vector(
            pedestrian_acceleration, self.config.max_interaction_acceleration_mps2
        )
        terms = {
            "pedestrian": self._term_record(
                pedestrian_acceleration,
                "unavailable_degenerate_geometry"
                if degenerate
                else ("available" if pedestrian_count else "available_no_neighbors"),
                count=pedestrian_count,
                clipped=pedestrian_clipped,
            )
        }

        obstacle_acceleration = np.zeros(2, dtype=float)
        obstacle_count = 0
        obstacle_degenerate = False
        if not geometry.available:
            terms["obstacle"] = self._term_record(
                obstacle_acceleration, "unavailable_geometry", count=0, clipped=False
            )
        else:
            for obstacle in geometry.obstacles:
                nearest = self._nearest_segment_point(
                    position,
                    np.asarray(obstacle.start_xy, dtype=float),
                    np.asarray(obstacle.end_xy, dtype=float),
                )
                vector = self._repulsion(
                    position,
                    nearest,
                    radius,
                    self.config.obstacle_interaction_strength_mps2,
                )
                if vector is None:
                    obstacle_degenerate = True
                    continue
                obstacle_acceleration += vector
                obstacle_count += 1
            obstacle_acceleration, obstacle_clipped = _clip_vector(
                obstacle_acceleration, self.config.max_interaction_acceleration_mps2
            )
            terms["obstacle"] = self._term_record(
                obstacle_acceleration,
                "unavailable_degenerate_geometry"
                if obstacle_degenerate
                else ("available" if obstacle_count else "available_no_obstacles"),
                count=obstacle_count,
                clipped=obstacle_clipped,
            )

        robot_acceleration = self._repulsion(
            position,
            np.asarray(robot.position_xy, dtype=float),
            radius + robot.radius_m,
            self.config.robot_interaction_strength_mps2,
        )
        if robot_acceleration is None:
            terms["robot"] = self._term_record(
                np.zeros(2), "unavailable_degenerate_geometry", count=1, clipped=False
            )
        else:
            robot_acceleration, robot_clipped = _clip_vector(
                robot_acceleration, self.config.max_interaction_acceleration_mps2
            )
            terms["robot"] = self._term_record(
                robot_acceleration, "available", count=1, clipped=robot_clipped
            )
        return terms

    def _repulsion(
        self,
        pedestrian_position: np.ndarray,
        source_position: np.ndarray,
        combined_radius: float,
        strength: float,
    ) -> np.ndarray | None:
        away = pedestrian_position - source_position
        distance = float(np.linalg.norm(away))
        if distance <= self.config.degenerate_distance_m:
            return None
        surface_distance = max(0.0, distance - combined_radius)
        magnitude = strength * math.exp(
            -min(surface_distance / self.config.interaction_decay_length_m, 60.0)
        )
        return away / distance * magnitude

    @staticmethod
    def _nearest_segment_point(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Return the closest point on a segment, including point-like segments."""
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        if length_squared <= 1e-12:
            return start.copy()
        fraction = float(np.clip(np.dot(point - start, direction) / length_squared, 0.0, 1.0))
        return start + fraction * direction

    @staticmethod
    def _term_record(
        acceleration: np.ndarray,
        status: str,
        *,
        count: int,
        clipped: bool,
    ) -> dict[str, Any]:
        return {
            "acceleration_mps2": np.asarray(acceleration, dtype=float).tolist(),
            "status": status,
            "source_count": count,
            "clipped": clipped,
            "units": "acceleration_like_m/s^2",
        }

    def _estimate_acceleration(self, history: deque[_Observation]) -> tuple[np.ndarray | None, str]:
        samples = list(history)[-self.config.history_steps :]
        if len(samples) < self.config.min_history_steps:
            return None, "insufficient_history"
        times = np.asarray([sample.time_s for sample in samples], dtype=float)
        velocities = np.asarray([sample.velocity for sample in samples], dtype=float)
        if np.any(np.diff(times) <= 0.0) or not np.all(np.isfinite(times)):
            return None, "invalid_time_interval"
        if self.config.acceleration_estimator == "velocity_difference":
            acceleration = (velocities[-1] - velocities[-2]) / (times[-1] - times[-2])
        else:
            centered_time = times - times.mean()
            denominator = float(np.dot(centered_time, centered_time))
            if denominator <= 0.0:
                return None, "invalid_time_interval"
            acceleration = (centered_time[:, None] * velocities).sum(axis=0) / denominator
        if not np.all(np.isfinite(acceleration)):
            return None, "non_finite_acceleration"
        clipped, did_clip = _clip_vector(acceleration, self.config.max_acceleration_mps2)
        return clipped, "valid_clipped" if did_clip else "valid"

    def _inflate_input_covariance(
        self,
        track: PlannerTrackBelief,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Add conservative covariance for stale, occluded, and low-confidence inputs.

        Tracker age_steps represents track tenure and is not treated as observation
        staleness. The predictor uses missed_steps and visibility for staleness so
        tenure alone cannot inflate or deflate forecast covariance.

        Returns:
            Inflated positive-semidefinite covariance and component magnitudes.
        """
        result = covariance.copy()
        stale = self.config.staleness_inflation_m2_per_step * track.missed_steps
        occlusion = self.config.occlusion_inflation_m2 if not track.visibility else 0.0
        confidence = self.config.confidence_inflation_m2 * (1.0 - track.confidence)
        addition = stale + occlusion + confidence
        result[:2, :2] += np.eye(2) * addition
        result[2:, 2:] += np.eye(2) * (
            confidence + stale / max(self.config.prediction_horizon_steps, 1)
        )
        return self._symmetrize_psd(result), {
            "staleness": stale,
            "occlusion": occlusion,
            "confidence": confidence,
        }

    def _build_modes(
        self,
        track: PlannerTrackBelief,
        position: np.ndarray,
        velocity: np.ndarray,
        covariance: np.ndarray,
        probabilities: dict[str, float],
        desired_velocity: np.ndarray | None,
    ) -> tuple[dict[str, Any], list[TrajectoryMode]]:
        modes: list[TrajectoryMode] = []
        diagnostics: dict[str, Any] = {}
        for mode_id in sorted(probabilities):
            current_position = position.copy()
            current_velocity = velocity.copy()
            current_covariance = covariance.copy()
            means = np.zeros((self.config.prediction_horizon_steps, 2), dtype=np.float32)
            covariances = np.zeros((self.config.prediction_horizon_steps, 2, 2), dtype=np.float32)
            clipped_steps = 0
            for index in range(self.config.prediction_horizon_steps):
                pre_step_velocity = current_velocity.copy()
                next_velocity, speed_clipped, _ = self._mode_step_velocity(
                    mode_id,
                    pre_step_velocity,
                    desired_velocity,
                    self.config.prediction_dt_s,
                )
                clipped_steps += int(speed_clipped)
                current_position = current_position + self.config.prediction_dt_s * next_velocity
                transition = self._mode_transition(
                    self.config.prediction_dt_s,
                    mode_id,
                    pre_step_velocity,
                    desired_velocity,
                )
                current_covariance = transition @ current_covariance @ transition.T
                current_covariance += self._process_noise(self.config.prediction_dt_s, mode_id)
                current_covariance = self._symmetrize_psd(current_covariance)
                current_velocity = next_velocity
                means[index] = current_position.astype(np.float32)
                covariances[index] = current_covariance[:2, :2].astype(np.float32)
            mode_provenance = (
                f"analytic_estimator:{self.config.force_model_version}"
                if mode_id == _RESIDUAL
                else "planner_visible_kinematics.v1"
            )
            modes.append(
                TrajectoryMode(
                    mode_id=mode_id,
                    probability=probabilities[mode_id],
                    mean=means,
                    covariance=covariances,
                    intent={
                        _CV: "constant_velocity",
                        _RESIDUAL: "desired_motion",
                        _YIELD: "deceleration",
                    }[mode_id],
                    provenance=mode_provenance,
                    metadata={
                        "identity_token": track.lifecycle_token,
                        "force_model_source": "analytic_estimator"
                        if mode_id == _RESIDUAL
                        else "none",
                        "force_model_version": self.config.force_model_version
                        if mode_id == _RESIDUAL
                        else None,
                        "units": "global_xy_m",
                        "integration": "semi_implicit_euler",
                    },
                )
            )
            diagnostics[mode_id] = {
                "probability": probabilities[mode_id],
                "clipped_speed_steps": clipped_steps,
                "prediction_steps": self.config.prediction_horizon_steps,
            }
        return diagnostics, modes

    def _mode_acceleration(
        self,
        mode_id: str,
        velocity: np.ndarray,
        desired_velocity: np.ndarray | None,
    ) -> np.ndarray:
        if mode_id == _CV:
            return np.zeros(2, dtype=float)
        if mode_id == _RESIDUAL:
            if desired_velocity is None:
                return np.zeros(2, dtype=float)
            desired_acceleration = (desired_velocity - velocity) / self.config.relaxation_time_s
            limit = self.config.max_acceleration_mps2
            if float(np.dot(desired_acceleration, velocity)) < 0.0:
                limit = min(limit, self.config.max_deceleration_mps2)
            return _clip_vector(desired_acceleration, limit)[0]
        speed = float(np.linalg.norm(velocity))
        if speed <= 1e-12:
            return np.zeros(2, dtype=float)
        magnitude = min(self.config.yield_deceleration_mps2, self.config.max_deceleration_mps2)
        return -velocity / speed * magnitude

    def _mode_step_velocity(
        self,
        mode_id: str,
        velocity: np.ndarray,
        desired_velocity: np.ndarray | None,
        dt: float,
    ) -> tuple[np.ndarray, bool, bool]:
        """Apply one bounded mode velocity step.

        Returns:
            Bounded next velocity, speed-clipping status, and yield-stop status.
        """
        acceleration = self._mode_acceleration(mode_id, velocity, desired_velocity)
        raw_next_velocity = velocity + dt * acceleration
        next_velocity, clipped = _clip_vector(raw_next_velocity, self.config.max_speed_mps)
        speed = float(np.linalg.norm(velocity))
        yield_stop_speed = (
            min(
                self.config.yield_deceleration_mps2,
                self.config.max_deceleration_mps2,
            )
            * dt
        )
        stopped = (
            mode_id == _YIELD
            and speed > 1e-12
            and (
                float(np.dot(next_velocity, velocity)) <= 0.0
                or speed <= yield_stop_speed + 1e-12 * max(1.0, yield_stop_speed)
            )
        )
        if stopped:
            next_velocity = np.zeros(2, dtype=float)
        return next_velocity, clipped, stopped

    def _mode_transition(
        self,
        dt: float,
        mode_id: str,
        velocity: np.ndarray,
        desired_velocity: np.ndarray | None,
    ) -> np.ndarray:
        """Linearize semi-implicit mode dynamics in ``[x, y, vx, vy]``.

        Residual motion differentiates the relaxation acceleration with respect
        to current velocity. Yield motion differentiates directional deceleration;
        within the stopped region the local velocity derivative is zero, while
        at the exact deceleration threshold it retains radial velocity support
        from the moving side. Acceleration and speed clipping are included locally.

        Returns:
            The mode-specific 4x4 state transition Jacobian.
        """
        identity = np.eye(2, dtype=float)
        acceleration = self._mode_acceleration(mode_id, velocity, desired_velocity)
        raw_next_velocity = velocity + dt * acceleration
        _, _, stopped = self._mode_step_velocity(
            mode_id,
            velocity,
            desired_velocity,
            dt,
        )
        speed = float(np.linalg.norm(velocity))

        magnitude = min(
            self.config.yield_deceleration_mps2,
            self.config.max_deceleration_mps2,
        )
        yield_stop_speed = dt * magnitude
        at_yield_stop_boundary = (
            mode_id == _YIELD
            and speed > 1e-12
            and math.isclose(
                speed,
                yield_stop_speed,
                rel_tol=0.0,
                abs_tol=1e-9 * max(1.0, yield_stop_speed),
            )
        )
        if at_yield_stop_boundary:
            direction = velocity / speed
            velocity_jacobian = np.outer(direction, direction)
        elif stopped:
            velocity_jacobian = np.zeros((2, 2), dtype=float)
        elif mode_id == _RESIDUAL and desired_velocity is not None:
            raw_acceleration = (desired_velocity - velocity) / self.config.relaxation_time_s
            acceleration_norm = float(np.linalg.norm(raw_acceleration))
            acceleration_limit = self.config.max_acceleration_mps2
            if float(np.dot(raw_acceleration, velocity)) < 0.0:
                acceleration_limit = min(
                    acceleration_limit,
                    self.config.max_deceleration_mps2,
                )
            if acceleration_norm <= acceleration_limit or acceleration_norm <= 1e-12:
                velocity_jacobian = (1.0 - dt / self.config.relaxation_time_s) * identity
            else:
                direction = raw_acceleration / acceleration_norm
                radial_projection = identity - np.outer(direction, direction)
                velocity_jacobian = (
                    identity
                    - (
                        dt
                        * acceleration_limit
                        / (self.config.relaxation_time_s * acceleration_norm)
                    )
                    * radial_projection
                )
        elif mode_id == _YIELD and speed > 1e-12:
            direction = velocity / speed
            tangential_projection = identity - np.outer(direction, direction)
            velocity_jacobian = identity - (dt * magnitude / speed) * tangential_projection
        elif mode_id in {_CV, _RESIDUAL} or speed <= 1e-12:
            velocity_jacobian = identity
        else:
            velocity_jacobian = identity

        raw_speed = float(np.linalg.norm(raw_next_velocity))
        if not stopped and raw_speed > self.config.max_speed_mps:
            direction = raw_next_velocity / raw_speed
            clipping_jacobian = (self.config.max_speed_mps / raw_speed) * (
                identity - np.outer(direction, direction)
            )
            velocity_jacobian = clipping_jacobian @ velocity_jacobian

        transition = np.eye(4, dtype=float)
        transition[:2, 2:] = dt * velocity_jacobian
        transition[2:, 2:] = velocity_jacobian
        return transition

    def _process_noise(self, dt: float, mode_id: str) -> np.ndarray:
        """Return discrete process covariance for one mode step.

        Residual model noise represents an uncertain post-step velocity increment
        for mismatch in inferred desired velocity and analytic dynamics. Because
        semi-implicit position uses that same next velocity, its covariance is
        propagated through ``[dt I; I]`` into position, velocity, and their
        cross-covariance rather than added to velocity alone.
        """
        noise = np.diag(
            [
                self.config.process_noise_position_m2_per_s * dt,
                self.config.process_noise_position_m2_per_s * dt,
                self.config.process_noise_velocity_m2_per_s * dt,
                self.config.process_noise_velocity_m2_per_s * dt,
            ]
        )
        if mode_id == _RESIDUAL:
            velocity_increment_variance = self.config.residual_model_noise_m2_per_s * dt
            noise[:2, :2] += np.eye(2) * velocity_increment_variance * dt**2
            noise[:2, 2:] += np.eye(2) * velocity_increment_variance * dt
            noise[2:, :2] += np.eye(2) * velocity_increment_variance * dt
            noise[2:, 2:] += np.eye(2) * velocity_increment_variance
        return noise

    @staticmethod
    def _symmetrize_psd(covariance: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if eigenvalues.min(initial=0.0) < -1e-6:
            raise ValueError("propagated_covariance_not_psd")
        if eigenvalues.min(initial=0.0) < 0.0:
            symmetric += np.eye(symmetric.shape[0]) * -float(eigenvalues.min())
        return symmetric

    def _prediction_status(self, track: PlannerTrackBelief, state: _TrackIntentState) -> str:
        if not track.visibility:
            return "coasted"
        if state.estimator_status == "interaction_unavailable":
            return "interaction_unavailable_cv_fallback"
        if state.desired_velocity is None:
            return "insufficient_history"
        return "residual_available"

    def _prune_missing_states(self, step: int, current_tokens: set[str]) -> None:
        for identity, state in tuple(self._states.items()):
            if identity in current_tokens:
                continue
            last = (
                state.last_visible_step
                if state.last_visible_step is not None
                else state.last_update_step
            )
            if step - last > self.config.max_coast_steps_for_intent:
                del self._states[identity]

    def _state_to_dict(self, state: _TrackIntentState) -> dict[str, Any]:
        return {
            "track_id": state.track_id,
            "tracker_namespace": state.tracker_namespace,
            "reset_epoch": state.reset_epoch,
            "lifecycle_token": state.lifecycle_token,
            "history": [
                {
                    "step": item.step,
                    "time_s": item.time_s,
                    "position": item.position.tolist(),
                    "velocity": item.velocity.tolist(),
                    "covariance": item.covariance.tolist(),
                    "confidence": item.confidence,
                    "age_steps": item.age_steps,
                    "missed_steps": item.missed_steps,
                }
                for item in state.history
            ],
            "log_probabilities": dict(sorted(state.log_probabilities.items())),
            "position": state.position.tolist(),
            "velocity": state.velocity.tolist(),
            "covariance": state.covariance.tolist(),
            "confidence": state.confidence,
            "age_steps": state.age_steps,
            "missed_steps": state.missed_steps,
            "visible": state.visible,
            "last_update_step": state.last_update_step,
            "last_visible_step": state.last_visible_step,
            "desired_velocity": None
            if state.desired_velocity is None
            else state.desired_velocity.tolist(),
            "reset_count": state.reset_count,
            "last_reset_reason": state.last_reset_reason,
            "heading_change_count": state.heading_change_count,
            "interaction_terms": state.interaction_terms,
            "measured_acceleration": None
            if state.measured_acceleration is None
            else state.measured_acceleration.tolist(),
            "estimator_status": state.estimator_status,
        }

    def _state_from_dict(self, payload: Mapping[str, Any]) -> _TrackIntentState:
        if not isinstance(payload, Mapping):
            raise ValueError("snapshot_invalid_track_state")
        track_id = payload.get("track_id")
        namespace = payload.get("tracker_namespace")
        reset_epoch = payload.get("reset_epoch")
        identity = payload.get("lifecycle_token")
        if (
            type(track_id) is not int
            or track_id < 1
            or not isinstance(namespace, str)
            or not namespace.strip()
            or type(reset_epoch) is not int
            or reset_epoch < 0
            or not isinstance(identity, str)
        ):
            raise ValueError("snapshot_invalid_track_identity")
        expected_identity = json.dumps(
            [namespace, reset_epoch, track_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if identity != expected_identity:
            raise ValueError("snapshot_lifecycle_identity_mismatch")
        history: deque[_Observation] = deque(maxlen=self.config.history_steps)
        for item in payload.get("history", []):
            history.append(
                _Observation(
                    step=int(item["step"]),
                    time_s=_finite_scalar(item["time_s"], "snapshot history time", minimum=0.0),
                    position=np.asarray(_finite_xy(item["position"], "snapshot position")),
                    velocity=np.asarray(_finite_xy(item["velocity"], "snapshot velocity")),
                    covariance=_snapshot_covariance(item["covariance"], "history"),
                    confidence=float(item["confidence"]),
                    age_steps=int(item["age_steps"]),
                    missed_steps=int(item["missed_steps"]),
                )
            )
        state = _TrackIntentState(
            track_id=track_id,
            tracker_namespace=namespace,
            reset_epoch=reset_epoch,
            lifecycle_token=identity,
            history=history,
            log_probabilities={
                str(key): float(value) for key, value in payload["log_probabilities"].items()
            },
            position=np.asarray(_finite_xy(payload["position"], "snapshot position")),
            velocity=np.asarray(_finite_xy(payload["velocity"], "snapshot velocity")),
            covariance=_snapshot_covariance(payload["covariance"], "track"),
            confidence=float(payload["confidence"]),
            age_steps=int(payload["age_steps"]),
            missed_steps=int(payload["missed_steps"]),
            visible=bool(payload["visible"]),
            last_update_step=int(payload["last_update_step"]),
            last_visible_step=None
            if payload["last_visible_step"] is None
            else int(payload["last_visible_step"]),
            desired_velocity=None
            if payload["desired_velocity"] is None
            else np.asarray(_finite_xy(payload["desired_velocity"], "snapshot desired velocity")),
            reset_count=int(payload["reset_count"]),
            last_reset_reason=payload["last_reset_reason"],
            heading_change_count=int(payload["heading_change_count"]),
            interaction_terms=dict(payload["interaction_terms"]),
            measured_acceleration=None
            if payload["measured_acceleration"] is None
            else np.asarray(_finite_xy(payload["measured_acceleration"], "snapshot acceleration")),
            estimator_status=str(payload["estimator_status"]),
        )
        if (
            state.position.shape != (2,)
            or state.velocity.shape != (2,)
            or state.covariance.shape != (4, 4)
        ):
            raise ValueError("snapshot_invalid_track_shape")
        return state


__all__ = [
    "ForceResidualIntentConfig",
    "ForceResidualIntentPredictor",
    "ObstacleSegment",
    "RobotKinematics",
    "StaticGeometry",
]
