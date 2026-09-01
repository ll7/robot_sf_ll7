"""Observation-only bridge from pedestrian tracking results to goal beliefs.

The adapter is deliberately default-off and does not create a posterior.  It translates
observation-derived tracker output into the canonical actor-safe goal-belief contract while
making the missing candidate-provider boundary explicit.  Simulator state, route assignments,
true goals, and force truth are not accepted by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose
from typing import Any

from robot_sf.prediction._contract_utils import (
    require_digest,
    require_finite,
    require_step_index,
    require_text,
    stable_config_hash,
    stable_digest,
)
from robot_sf.prediction.goal_belief_contract import (
    ActorObservationStep,
    CensoringState,
    CoordinateFrame,
    GoalBeliefMode,
    GoalBeliefObservation,
    GoalBeliefV1,
    ObservationMask,
)
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianTrack,
    PedestrianTrackingResult,
    TrackStatus,
)

TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION = "tracker_goal_belief_adapter.v1"
TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY = "implementation_integrity_smoke"
TRACKER_GOAL_BELIEF_BLOCKER = "candidate_provider_not_configured"


@dataclass(frozen=True, slots=True)
class TrackerGoalBeliefAdapterConfig:
    """Default-off configuration for the tracker-to-prediction side channel."""

    enabled: bool = False

    def __post_init__(self) -> None:
        """Reject implicit truthy values at the feature boundary."""
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-safe configuration."""
        return {
            "schema_version": TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION,
            "enabled": self.enabled,
            "candidate_policy": "not_configured",
        }

    @property
    def config_hash(self) -> str:
        """Return the deterministic configuration digest."""
        return stable_config_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrackerGoalBeliefChannel:
    """Typed, serializable side-channel payload for one tracker update."""

    enabled: bool
    tracking_epoch_id: int
    timestamp_s: float | None
    step_index: int | None
    beliefs: tuple[GoalBeliefV1, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    schema_version: str = TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate channel lifecycle and deterministic belief ordering."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if type(self.tracking_epoch_id) is not int or self.tracking_epoch_id < 0:
            raise ValueError("tracking_epoch_id must be a non-negative integer")
        if self.schema_version != TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported tracker goal-belief adapter schema_version")
        require_digest(self.config_hash, "config_hash")
        beliefs = tuple(self.beliefs)
        _validate_beliefs(beliefs)
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        _validate_channel_timing(
            enabled=self.enabled,
            timestamp_s=self.timestamp_s,
            step_index=self.step_index,
            beliefs=beliefs,
        )
        object.__setattr__(self, "beliefs", beliefs)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe planner-side channel."""
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "tracking_epoch_id": self.tracking_epoch_id,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "config_hash": self.config_hash,
            "claim_boundary": TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY,
            "beliefs": {belief.track_id: belief.to_dict() for belief in self.beliefs},
            "diagnostics": dict(self.diagnostics),
        }

    @property
    def content_digest(self) -> str:
        """Return the digest of the canonical channel payload."""
        return stable_digest(self.to_dict())


class TrackerGoalBeliefAdapter:
    """Build an explicit observation-only prediction side channel from tracker results."""

    __slots__ = ("_reset_provenance", "_tracking_epoch_id", "config")

    def __init__(self, config: TrackerGoalBeliefAdapterConfig | None = None) -> None:
        """Create an adapter without retaining tracker or simulator state."""
        if config is None:
            config = TrackerGoalBeliefAdapterConfig()
        if type(config) is not TrackerGoalBeliefAdapterConfig:
            raise TypeError("config must be TrackerGoalBeliefAdapterConfig")
        self.config = config
        self._tracking_epoch_id = 0
        self._reset_provenance: str | None = None

    @property
    def tracking_epoch_id(self) -> int:
        """Return the current episode-local adapter epoch."""
        return self._tracking_epoch_id

    def reset(self, reset_provenance: str) -> None:
        """Start a new episode-local epoch and record its caller-owned reset token."""
        token = require_text(reset_provenance, "reset_provenance")
        self._tracking_epoch_id += 1
        self._reset_provenance = token

    def adapt(self, result: PedestrianTrackingResult) -> TrackerGoalBeliefChannel:
        """Translate one validated tracker result into an actor-safe channel.

        The disabled path validates only the result type and emits no timing, track, or belief
        payload.  The enabled path carries an explicit unavailable belief until a separate,
        observation-only candidate provider is configured.

        Returns:
            A default-off empty channel or an enabled channel of observation-only beliefs.
        """
        if type(result) is not PedestrianTrackingResult:
            raise TypeError("result must be PedestrianTrackingResult")
        if not self.config.enabled:
            return TrackerGoalBeliefChannel(
                enabled=False,
                tracking_epoch_id=self._tracking_epoch_id,
                timestamp_s=None,
                step_index=None,
                config_hash=self.config.config_hash,
                diagnostics={
                    "status": "disabled",
                    "track_count": 0,
                    "claim_boundary": TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY,
                },
            )
        if not result.diagnostics.enabled:
            return TrackerGoalBeliefChannel(
                enabled=True,
                tracking_epoch_id=self._tracking_epoch_id,
                timestamp_s=result.timestamp_s,
                step_index=result.step_index,
                config_hash=self.config.config_hash,
                diagnostics={
                    "status": "tracking_unavailable",
                    "track_count": 0,
                    "tracking_blockers": list(result.diagnostics.blockers),
                    "claim_boundary": TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY,
                },
            )
        track_records = []
        tracking_epoch_id = str(self._tracking_epoch_id)
        for track in result.tracks:
            _validate_track_alignment(track, result)
            history, history_projection = _track_history(track)
            belief = _belief_from_track(
                track,
                config_hash=self.config.config_hash,
                tracking_epoch_id=tracking_epoch_id,
                reset_provenance=self._reset_provenance,
                history=history,
                history_projection=history_projection,
            )
            track_records.append((track, belief, history_projection))
        track_records.sort(key=lambda record: record[1].track_id)
        beliefs = tuple(record[1] for record in track_records)
        statuses = [belief.mode.value for belief in beliefs]
        track_statuses = [_track_status(track) for track, _, _ in track_records]
        history_projections = [projection for _, _, projection in track_records]
        return TrackerGoalBeliefChannel(
            enabled=True,
            tracking_epoch_id=self._tracking_epoch_id,
            timestamp_s=result.timestamp_s,
            step_index=result.step_index,
            beliefs=beliefs,
            config_hash=self.config.config_hash,
            diagnostics={
                "status": "enabled",
                "track_count": len(beliefs),
                "unavailable_belief_count": statuses.count(GoalBeliefMode.UNAVAILABLE.value),
                "lost_track_count": track_statuses.count(TrackStatus.LOST),
                "retired_track_count": track_statuses.count(TrackStatus.RETIRED),
                "history_projection_counts": {
                    projection: history_projections.count(projection)
                    for projection in sorted(set(history_projections))
                },
                "track_diagnostics": {
                    belief.track_id: _track_diagnostics(track, history_projection)
                    for track, belief, history_projection in track_records
                },
                "tracking_blockers": list(result.diagnostics.blockers),
                "reset_provenance": self._reset_provenance,
                "candidate_policy": "not_configured",
                "claim_boundary": TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY,
            },
        )


def _current_history_row(track: PedestrianTrack) -> ActorObservationStep:
    """Build one conservative current row when tracker history is not reconstructable.

    Returns:
        The current observed or invisible history row.
    """
    if track.history_valid_mask.size == 0:
        raise ValueError("track history must contain at least one row")
    current_visible = bool(track.history_valid_mask[-1])
    if not current_visible or "velocity_unavailable" in track.blockers:
        return ActorObservationStep(
            timestamp_s=track.timestamp_s,
            step_index=track.step_index,
            position_xy=None,
            velocity_xy=None,
            mask=ObservationMask.INVISIBLE,
        )
    return ActorObservationStep(
        timestamp_s=track.timestamp_s,
        step_index=track.step_index,
        position_xy=(float(track.position_global_xy[0]), float(track.position_global_xy[1])),
        velocity_xy=(float(track.velocity_global_xy[0]), float(track.velocity_global_xy[1])),
        mask=ObservationMask.OBSERVED,
    )


def _track_history(track: PedestrianTrack) -> tuple[tuple[ActorObservationStep, ...], str]:
    """Expose only the current row until tracker velocity provenance is row-level.

    The tracker v1 output exposes one history-validity mask for both position and velocity, so it
    cannot prove that historical velocity values were observed rather than estimated or predicted.
    The adapter therefore emits a stateless current-decision-point projection until the tracker
    contract provides row-level velocity provenance.

    Returns:
        The actor history and a diagnostic describing the projection.
    """
    if "velocity_unavailable" in track.blockers:
        return ((_current_history_row(track),), "current_row_only_velocity_unavailable")
    return (
        (_current_history_row(track),),
        "current_row_only_tracker_v1_velocity_provenance_unavailable",
    )


def _belief_from_track(
    track: PedestrianTrack,
    *,
    config_hash: str,
    tracking_epoch_id: str,
    reset_provenance: str | None,
    history: tuple[ActorObservationStep, ...] | None = None,
    history_projection: str | None = None,
) -> GoalBeliefV1:
    """Create an unavailable actor belief from one observation-derived track.

    Returns:
        The canonical observation-only belief.
    """
    if history is None or history_projection is None:
        history, history_projection = _track_history(track)
    blockers = set(track.blockers)
    blockers.add(TRACKER_GOAL_BELIEF_BLOCKER)
    blockers.add(history_projection)
    status = _track_status(track)
    if status is TrackStatus.LOST:
        blockers.add("track_not_currently_visible")
    elif status is TrackStatus.RETIRED:
        blockers.add("track_retired")
    observation = GoalBeliefObservation(
        track_id=f"track-{track.track_id}",
        tracking_epoch_id=tracking_epoch_id,
        timestamp_s=track.timestamp_s,
        step_index=track.step_index,
        config_hash=config_hash,
        history_steps=history,
        coordinate_frame=CoordinateFrame.GLOBAL_XY,
        mode=GoalBeliefMode.UNAVAILABLE,
        censoring_state=CensoringState.UNKNOWN,
        blockers=tuple(sorted(blockers)),
        reset_provenance=reset_provenance,
    )
    return GoalBeliefV1.from_observation(observation)


def _validate_track_alignment(track: PedestrianTrack, result: PedestrianTrackingResult) -> None:
    """Reject a result whose per-track decision point differs from the batch point."""
    if track.step_index != result.step_index:
        raise ValueError("track step_index must match tracking result step_index")
    if not isclose(track.timestamp_s, result.timestamp_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("track timestamp_s must match tracking result timestamp_s")


def _validate_beliefs(beliefs: tuple[GoalBeliefV1, ...]) -> None:
    """Validate the channel's belief element type, order, and identity uniqueness."""
    if any(type(belief) is not GoalBeliefV1 for belief in beliefs):
        raise TypeError("beliefs must contain GoalBeliefV1 values")
    track_ids = tuple(belief.track_id for belief in beliefs)
    if track_ids != tuple(sorted(track_ids)):
        raise ValueError("beliefs must be sorted by track_id")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("belief track IDs must be unique")


def _validate_channel_timing(
    *,
    enabled: bool,
    timestamp_s: float | None,
    step_index: int | None,
    beliefs: tuple[GoalBeliefV1, ...],
) -> None:
    """Validate timing and payload presence for either channel mode."""
    if not enabled:
        if beliefs:
            raise ValueError("disabled channel must not carry beliefs")
        if timestamp_s is not None or step_index is not None:
            raise ValueError("disabled channel must not carry tracker timing")
        return
    if timestamp_s is None or step_index is None:
        raise ValueError("enabled channel requires tracker timing")
    require_finite(timestamp_s, "timestamp_s")
    require_step_index(step_index, "step_index")


def _track_diagnostics(track: PedestrianTrack, history_projection: str) -> dict[str, Any]:
    """Return actor-safe tracker uncertainty and lifecycle diagnostics.

    Returns:
        A JSON-safe per-track diagnostic mapping.
    """
    return {
        "status": _track_status(track).value,
        "association_confidence": track.association_confidence,
        "age_steps": track.age_steps,
        "visible_age_steps": track.visible_age_steps,
        "missed_steps": track.missed_steps,
        "position_covariance": track.position_covariance.tolist(),
        "velocity_covariance": track.velocity_covariance.tolist(),
        "history_capacity": int(track.history_valid_mask.shape[0]),
        "history_valid_count": int(track.history_valid_mask.sum()),
        "history_projection": history_projection,
        "blockers": list(track.blockers),
        "tracking_config_hash": track.config_hash,
    }


def _track_status(track: PedestrianTrack) -> TrackStatus:
    """Normalize the public track status union to its validated enum.

    Returns:
        The validated track lifecycle status.
    """
    if isinstance(track.status, TrackStatus):
        return track.status
    return TrackStatus(track.status)


__all__ = [
    "TRACKER_GOAL_BELIEF_ADAPTER_SCHEMA_VERSION",
    "TRACKER_GOAL_BELIEF_BLOCKER",
    "TRACKER_GOAL_BELIEF_CLAIM_BOUNDARY",
    "TrackerGoalBeliefAdapter",
    "TrackerGoalBeliefAdapterConfig",
    "TrackerGoalBeliefChannel",
]
