"""Planner-facing ScenarioBelief uncertainty projection helpers.

These helpers are diagnostic interface smoke, not benchmark evidence. They bridge
the uncertainty-preserving ScenarioBelief report into one planner-compatible observation shape
without changing legacy policy projections.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.sensor.pedestrian_tracking import (
    PedestrianTrack,
    PedestrianTrackingResult,
    TrackStatus,
)

if TYPE_CHECKING:
    from robot_sf.representation import ScenarioBelief

SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION = "scenario-belief-planner-projection.v1"
SUPPORTED_UNCERTAINTY_PLANNER_KEYS = frozenset({"stream_gap"})
IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION = "identity-safe-planner-input.v1"
SUPPORTED_IDENTITY_SAFE_PLANNER_NAMES = frozenset({"BeliefGuidedLocalPlanner"})
TRACK_EXISTENCE_PROBABILITY_SEMANTICS = "active_track_keep_alive_assumption"


class _AssociationLifecycleConflict(ValueError):
    """A current association contradicts the associated track's lifecycle state."""


def _validated_projected_arrays(
    mean_values: np.ndarray,
    covariance_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and defensively freeze projected state and covariance arrays.

    Returns:
        Owned, read-only mean-state and covariance arrays.
    """
    mean = np.array(mean_values, dtype=float, copy=True)
    covariance = np.array(covariance_values, dtype=float, copy=True)
    if mean.shape != (5,) or not np.all(np.isfinite(mean)):
        raise ValueError("mean_state must be a finite [x, y, vx, vy, radius] vector")
    if mean[4] < 0.0:
        raise ValueError("mean_state radius must be non-negative")
    if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite 4x4 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    mean.setflags(write=False)
    covariance.setflags(write=False)
    return mean, covariance


def _validate_projected_track_identity(track: PlannerTrackBelief) -> None:
    """Validate scalar identity and lifecycle fields on a projected track."""
    if type(track.track_id) is not int or track.track_id < 1:
        raise ValueError("track_id must be a positive integer")
    _validate_projected_track_timing(track)
    _validate_projected_track_strings(track)
    if type(track.visibility) is not bool:
        raise TypeError("visibility must be a bool")


def _validate_projected_track_timing(track: PlannerTrackBelief) -> None:
    """Validate integer step and epoch fields for one projected track."""
    if type(track.belief_step) is not int or track.belief_step < 0:
        raise ValueError("belief_step must be a non-negative integer")
    if (
        type(track.last_observed_step) is not int
        or not 0 <= track.last_observed_step <= track.belief_step
    ):
        raise ValueError("last_observed_step must be within the belief step")
    if type(track.age_steps) is not int or track.age_steps < 0:
        raise ValueError("age_steps must be a non-negative integer")
    if type(track.missed_steps) is not int or track.missed_steps < 0:
        raise ValueError("missed_steps must be a non-negative integer")
    if type(track.reset_epoch) is not int or track.reset_epoch < 0:
        raise ValueError("reset_epoch must be a non-negative integer")


def _validate_projected_track_strings(track: PlannerTrackBelief) -> None:
    """Validate human-readable identity and lifecycle labels."""
    for field_name in ("tracker_namespace", "lifecycle_token", "source", "status"):
        if (
            not isinstance(getattr(track, field_name), str)
            or not getattr(track, field_name).strip()
        ):
            raise ValueError(f"{field_name} must be non-empty text")
    active_statuses = {status.value for status in TrackStatus if status is not TrackStatus.RETIRED}
    if track.status not in active_statuses:
        raise ValueError("projected track status must be active or coasted")
    if track.lifecycle_token != _track_lifecycle_token(
        track.tracker_namespace, track.reset_epoch, track.track_id
    ):
        raise ValueError("lifecycle_token must encode namespace, reset_epoch, and track_id")


def _validated_probability(value: float, field_name: str) -> float:
    """Return a finite probability scalar."""
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return normalized


def _validated_existence_probability(
    value: float,
    *,
    calibrated: bool,
    semantics: str,
) -> float:
    """Validate the explicit non-calibrated keep-alive existence contract.

    Returns:
        The normalized existence probability when its semantics are valid.
    """
    normalized = _validated_probability(value, "existence_probability")
    if type(calibrated) is not bool or calibrated:
        raise ValueError("tracker existence probability must be marked uncalibrated")
    if semantics != TRACK_EXISTENCE_PROBABILITY_SEMANTICS or normalized != 1.0:
        raise ValueError("tracker existence probability must use the 1.0 keep-alive assumption")
    return normalized


@dataclass(frozen=True)
class ScenarioBeliefPlannerProjection:
    """ScenarioBelief observation plus explicit planner uncertainty compatibility status."""

    observation: dict[str, Any]
    compatibility: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlannerTrackBelief:
    """Immutable belief for one producer-owned numeric track identity.

    The tracker reports maintained hypotheses but no calibrated existence
    probability, so retained tracks use ``1.0`` as an explicit keep-alive value;
    association confidence is carried separately and never discounts existence.
    """

    track_id: int
    mean_state: np.ndarray
    covariance: np.ndarray
    confidence: float
    existence_probability: float
    existence_probability_calibrated: bool
    existence_probability_semantics: str
    visibility: bool
    age_steps: int
    source: str
    tracker_namespace: str
    reset_epoch: int
    lifecycle_token: str
    belief_step: int
    last_observed_step: int
    missed_steps: int
    status: str

    def __post_init__(self) -> None:
        """Copy and freeze arrays, then validate the complete track record."""
        _validate_projected_track_identity(self)
        mean, covariance = _validated_projected_arrays(self.mean_state, self.covariance)
        object.__setattr__(self, "mean_state", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self, "confidence", _validated_probability(self.confidence, "confidence")
        )
        object.__setattr__(
            self,
            "existence_probability",
            _validated_existence_probability(
                self.existence_probability,
                calibrated=self.existence_probability_calibrated,
                semantics=self.existence_probability_semantics,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe record."""
        return {
            "track_id": self.track_id,
            "mean_state": self.mean_state.tolist(),
            "covariance": self.covariance.tolist(),
            "confidence": self.confidence,
            "existence_probability": self.existence_probability,
            "existence_probability_calibrated": self.existence_probability_calibrated,
            "existence_probability_semantics": self.existence_probability_semantics,
            "visibility": self.visibility,
            "age_steps": self.age_steps,
            "source": self.source,
            "tracker_namespace": self.tracker_namespace,
            "reset_epoch": self.reset_epoch,
            "lifecycle_token": self.lifecycle_token,
            "belief_step": self.belief_step,
            "last_observed_step": self.last_observed_step,
            "missed_steps": self.missed_steps,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class IdentitySafeProjectionDiagnostics:
    """Versioned counters and lifecycle identifiers for one projection attempt."""

    status: str
    planner_name: str
    belief_step: int | None
    visible_track_count: int
    occluded_track_count: int
    stale_track_count: int
    retained_track_count: int
    retired_track_count: int
    dropped_track_count: int
    per_reason_drop_count: tuple[tuple[str, int], ...]
    fallback_reason: str | None
    ordered_track_ids: tuple[int, ...]
    retired_track_ids: tuple[int, ...]
    identity_lifecycle_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-safe projection diagnostics."""
        return {
            "schema_version": IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION,
            "status": self.status,
            "planner_name": self.planner_name,
            "belief_step": self.belief_step,
            "visible_track_count": self.visible_track_count,
            "occluded_track_count": self.occluded_track_count,
            "stale_track_count": self.stale_track_count,
            "retained_track_count": self.retained_track_count,
            "retired_track_count": self.retired_track_count,
            "dropped_track_count": self.dropped_track_count,
            "per_reason_drop_count": dict(self.per_reason_drop_count),
            "fallback_reason": self.fallback_reason,
            "ordered_track_ids": list(self.ordered_track_ids),
            "retired_track_ids": list(self.retired_track_ids),
            "identity_lifecycle_tokens": list(self.identity_lifecycle_tokens),
        }


@dataclass(frozen=True, slots=True)
class BeliefAwarePlannerInput:
    """Legacy observation plus an identity-keyed immutable track sidecar."""

    legacy_observation: Mapping[str, Any] | None
    tracks: Mapping[int, PlannerTrackBelief]
    belief_step: int | None
    schema_version: str
    diagnostics: IdentitySafeProjectionDiagnostics

    def __post_init__(self) -> None:
        """Defensively own mappings while preserving the legacy observation shape."""
        if self.schema_version != IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION:
            raise ValueError("unsupported identity-safe planner input schema_version")
        tracks = dict(self.tracks)
        if tuple(tracks) != tuple(sorted(tracks)):
            raise ValueError("tracks must be inserted in ascending track_id order")
        if any(type(key) is not int or value.track_id != key for key, value in tracks.items()):
            raise ValueError("track keys must match each embedded numeric track_id")
        if tracks and (
            type(self.belief_step) is not int
            or any(track.belief_step != self.belief_step for track in tracks.values())
        ):
            raise ValueError("every projected track must match the wrapper belief_step")
        object.__setattr__(self, "tracks", MappingProxyType(tracks))
        if self.legacy_observation is not None:
            if not isinstance(self.legacy_observation, Mapping):
                raise TypeError("legacy_observation must be a mapping or None")
            object.__setattr__(
                self, "legacy_observation", copy.deepcopy(dict(self.legacy_observation))
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe representation of the full input."""
        return {
            "schema_version": self.schema_version,
            "projection": self.diagnostics.to_dict(),
            "belief_step": self.belief_step,
            "legacy_observation": _json_safe(self.legacy_observation),
            "tracks": {str(track_id): track.to_dict() for track_id, track in self.tracks.items()},
        }

    def to_json(self) -> str:
        """Serialize deterministically, rejecting non-finite or unsupported values.

        Returns:
            A compact JSON string with stable key ordering.
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


def _json_safe(value: Any) -> Any:
    """Convert nested observation values to finite JSON-safe Python values.

    Returns:
        A recursively normalized value accepted by :func:`json.dumps`.
    """
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError("JSON-safe projection export rejects non-finite array values")
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) for key in value):
            raise TypeError("JSON-safe projection mapping keys must be strings or integers")
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON-safe projection export rejects NaN and Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON-safe projection value: {type(value).__name__}")


def _pedestrian_count(observation: dict[str, Any]) -> int | None:
    """Return the active pedestrian count from a SOCNAV_STRUCT-like observation."""
    pedestrians = observation.get("pedestrians")
    if not isinstance(pedestrians, dict):
        return None
    try:
        raw_count = np.asarray(pedestrians.get("count"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if raw_count.size == 0 or not np.isfinite(raw_count[0]):
        return None
    return max(0, int(raw_count[0]))


def _compatibility_payload(
    *,
    planner_key: str,
    status: str,
    reason: str | None = None,
    consumed_agent_count: int = 0,
) -> dict[str, Any]:
    """Return a deterministic planner-compatibility diagnostic payload."""
    payload: dict[str, Any] = {
        "schema_version": SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION,
        "planner_key": planner_key,
        "status": status,
        "uncertainty_consumed": status == "compatible",
        "consumed_agent_count": int(consumed_agent_count),
        "claim_boundary": "diagnostic_interface_smoke",
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def project_scenario_belief_for_planner(
    belief: ScenarioBelief,
    *,
    planner_key: str,
) -> ScenarioBeliefPlannerProjection:
    """Project ScenarioBelief into one planner observation with uncertainty compatibility status.

    Only ``stream_gap`` currently consumes the uncertainty sidecar under
    ``observation["pedestrians"]["uncertainty"]``. Unsupported planner keys fail closed by
    returning the legacy ``to_socnav_struct()`` observation without the sidecar and by recording
    an explicit unsupported status.

    Returns:
        ScenarioBeliefPlannerProjection: Observation plus diagnostic compatibility metadata.
    """
    observation = belief.to_socnav_struct()
    pedestrians = observation.get("pedestrians")
    if not isinstance(pedestrians, dict):
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="malformed_legacy_observation",
        )
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    if planner_key not in SUPPORTED_UNCERTAINTY_PLANNER_KEYS:
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="unsupported_uncertainty_planner",
        )
        pedestrians["uncertainty_compatibility"] = compatibility
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    count = _pedestrian_count(observation)
    if count is None:
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="malformed_pedestrian_count",
        )
        pedestrians["uncertainty_compatibility"] = compatibility
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    report = belief.to_uncertainty_report()
    rows = report.get("agents")
    if not isinstance(rows, list) or len(rows) < count:
        compatibility = _compatibility_payload(
            planner_key=planner_key,
            status="fail_closed",
            reason="malformed_uncertainty_report",
        )
        pedestrians["uncertainty_compatibility"] = compatibility
        return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)

    uncertainty_rows = [dict(row) for row in rows[:count]]
    pedestrians["uncertainty"] = uncertainty_rows
    compatibility = _compatibility_payload(
        planner_key=planner_key,
        status="compatible",
        consumed_agent_count=len(uncertainty_rows),
    )
    pedestrians["uncertainty_compatibility"] = compatibility
    return ScenarioBeliefPlannerProjection(observation=observation, compatibility=compatibility)


def project_identity_safe_scenario_belief(
    belief: ScenarioBelief | None,
    *,
    planner_name: str,
    belief_step: int,
    tracking_result: PedestrianTrackingResult | None,
    tracker_namespace: str | None,
    legacy_observation: Mapping[str, Any] | None = None,
) -> BeliefAwarePlannerInput:
    """Project maintained tracker tracks by identity, independently of legacy row order.

    ``belief_step`` is the caller's explicit projection step and must match the
    producer result's step. ``belief`` supplies only the legacy observation shape and
    configured pedestrian radius.
    Track means, covariances, age, status, and observed-now state come from the immutable
    ``PedestrianTrackingResult``. Current-result associations and the producer's
    ``observed_this_step`` marker identify visible tracks; a track's remembered
    ``last_observation_slot`` is deliberately ignored.

    Returns:
        A legacy observation plus immutable identity-keyed tracks and projection diagnostics.
    """
    planner_valid = isinstance(planner_name, str) and bool(planner_name.strip())
    resolved_name = planner_name if planner_valid else "<invalid>"
    valid_belief_step = type(belief_step) is int and belief_step >= 0
    resolved_belief_step = belief_step if valid_belief_step else None
    try:
        observation = _copy_legacy_observation(belief, legacy_observation)
    except (TypeError, ValueError, AttributeError):
        return _failed_identity_projection(
            None,
            resolved_name,
            "invalid",
            "invalid_legacy_observation",
            tracking_result,
            resolved_belief_step,
        )
    input_error = _identity_projection_input_error(
        belief,
        resolved_name,
        planner_valid,
        valid_belief_step,
        belief_step,
        tracking_result,
        tracker_namespace,
    )
    if input_error is not None:
        status, reason, count_source = input_error
        return _failed_identity_projection(
            observation,
            resolved_name,
            status,
            reason,
            count_source,
            resolved_belief_step,
        )
    try:
        radius = float(belief.pedestrian_radius)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("pedestrian radius must be finite and non-negative")
        summary = _project_tracking_result(
            tracking_result,
            tracker_namespace=tracker_namespace,
            radius=radius,
        )
    except _AssociationLifecycleConflict:
        return _failed_identity_projection(
            observation,
            resolved_name,
            "invalid",
            "association_lifecycle_conflict",
            tracking_result,
            resolved_belief_step,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return _failed_identity_projection(
            observation,
            resolved_name,
            "invalid",
            "malformed_tracking_metadata",
            tracking_result,
            resolved_belief_step,
        )
    status = "empty" if not tracking_result.tracks else "supported"
    return _assemble_identity_projection(observation, resolved_name, status, None, summary)


def _identity_projection_input_error(
    belief: ScenarioBelief | None,
    planner_name: str,
    planner_valid: bool,
    belief_step_valid: bool,
    belief_step: int,
    tracking_result: PedestrianTrackingResult | None,
    tracker_namespace: str | None,
) -> tuple[str, str, PedestrianTrackingResult | None] | None:
    """Return the first fail-closed reason for unsupported projection inputs.

    Returns:
        Status, reason, and any trustworthy producer result to count, or ``None``.
    """
    valid_result = isinstance(tracking_result, PedestrianTrackingResult)
    if not planner_valid:
        return "invalid", "invalid_planner_name", tracking_result if valid_result else None
    if not belief_step_valid:
        return "invalid", "invalid_belief_step", tracking_result if valid_result else None
    if belief is None:
        return "missing", "missing_scenario_belief", tracking_result if valid_result else None
    if not hasattr(belief, "pedestrian_radius"):
        return "invalid", "invalid_scenario_belief", tracking_result if valid_result else None
    if planner_name not in SUPPORTED_IDENTITY_SAFE_PLANNER_NAMES:
        return "unsupported", "unsupported_planner", tracking_result if valid_result else None
    if tracking_result is None:
        return "missing", "missing_tracking_result", None
    if not valid_result:
        return "invalid", "invalid_tracking_result", None
    return _identity_projection_lifecycle_error(tracking_result, belief_step, tracker_namespace)


def _identity_projection_lifecycle_error(
    tracking_result: PedestrianTrackingResult,
    belief_step: int,
    tracker_namespace: str | None,
) -> tuple[str, str, PedestrianTrackingResult] | None:
    """Validate producer availability and the projection's composite identity context.

    Returns:
        The fail-closed state, reason, and trustworthy tracking result, or ``None``.
    """
    if tracking_result.step_index != belief_step:
        return "invalid", "belief_step_mismatch", tracking_result
    if not tracking_result.diagnostics.enabled:
        return "unavailable", "tracking_disabled", tracking_result
    if not isinstance(tracker_namespace, str) or not tracker_namespace.strip():
        return "invalid", "missing_tracker_namespace", tracking_result
    if tracking_result.reset_epoch is None:
        return "invalid", "missing_reset_epoch", tracking_result
    return None


def _copy_legacy_observation(
    belief: ScenarioBelief | None,
    legacy_observation: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return an owned legacy observation without adding projection metadata."""
    if legacy_observation is not None:
        if not isinstance(legacy_observation, Mapping):
            raise TypeError("legacy_observation must be a mapping or None")
        return copy.deepcopy(dict(legacy_observation))
    if belief is None:
        return None
    projector = getattr(belief, "to_socnav_struct", None)
    if not callable(projector):
        raise TypeError("belief must provide to_socnav_struct()")
    return projector()


def _failed_identity_projection(
    observation: Mapping[str, Any] | None,
    planner_name: str,
    status: str,
    reason: str,
    tracking_result: PedestrianTrackingResult | None,
    belief_step: int | None,
) -> BeliefAwarePlannerInput:
    """Build a fail-closed legacy fallback and count unavailable track records.

    Returns:
        An identity-safe wrapper with no tracks and an explicit failure diagnostic.
    """
    tracks = tracking_result.tracks if isinstance(tracking_result, PedestrianTrackingResult) else ()
    retired_ids = tuple(
        track.track_id for track in tracks if TrackStatus(track.status) is TrackStatus.RETIRED
    )
    dropped_count = sum(TrackStatus(track.status) is not TrackStatus.RETIRED for track in tracks)
    summary: dict[str, Any] = {
        "belief_step": belief_step,
        "dropped_track_count": dropped_count,
        "retired_track_ids": retired_ids,
    }
    if dropped_count:
        summary["drop_reason"] = reason
    return _assemble_identity_projection(observation, planner_name, status, reason, summary)


def _assemble_identity_projection(
    observation: Mapping[str, Any] | None,
    planner_name: str,
    status: str,
    fallback_reason: str | None,
    summary: Mapping[str, Any],
) -> BeliefAwarePlannerInput:
    """Construct one typed result from projected tracks and compact counters.

    Returns:
        An immutable planner input with deterministic diagnostics.
    """
    tracks: dict[int, PlannerTrackBelief] = summary.get("tracks", {})
    dropped_count = int(summary.get("dropped_track_count", 0))
    drop_reason = summary.get("drop_reason")
    per_reason = ((drop_reason, dropped_count),) if drop_reason and dropped_count else ()
    diagnostics = IdentitySafeProjectionDiagnostics(
        status=status,
        planner_name=planner_name,
        belief_step=summary.get("belief_step"),
        visible_track_count=int(summary.get("visible_track_count", 0)),
        occluded_track_count=int(summary.get("occluded_track_count", 0)),
        stale_track_count=int(summary.get("stale_track_count", 0)),
        retained_track_count=len(tracks),
        retired_track_count=len(summary.get("retired_track_ids", ())),
        dropped_track_count=dropped_count,
        per_reason_drop_count=per_reason,
        fallback_reason=fallback_reason,
        ordered_track_ids=tuple(tracks),
        retired_track_ids=summary.get("retired_track_ids", ()),
        identity_lifecycle_tokens=summary.get("identity_lifecycle_tokens", ()),
    )
    return BeliefAwarePlannerInput(
        legacy_observation=observation,
        tracks=tracks,
        belief_step=diagnostics.belief_step,
        schema_version=IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION,
        diagnostics=diagnostics,
    )


def _project_tracking_result(
    tracking_result: PedestrianTrackingResult,
    *,
    tracker_namespace: str,
    radius: float,
) -> dict[str, Any]:
    """Build sorted track records from numeric IDs and current-result associations.

    Returns:
        Projected tracks and counts needed for deterministic diagnostics.
    """
    tracks_by_id = {track.track_id: track for track in tracking_result.tracks}
    if len(tracks_by_id) != len(tracking_result.tracks):
        raise ValueError("duplicate track_id values")
    associations = {association.track_id for association in tracking_result.associations}
    if not associations.issubset(tracks_by_id):
        raise ValueError("association references a missing track")
    for track_id in associations:
        associated_track = tracks_by_id[track_id]
        if (
            not associated_track.observed_this_step
            or TrackStatus(associated_track.status) in {TrackStatus.LOST, TrackStatus.RETIRED}
            or associated_track.missed_steps > 0
        ):
            raise _AssociationLifecycleConflict(
                "current association conflicts with the track lifecycle state"
            )
    projected: dict[int, PlannerTrackBelief] = {}
    retired_ids: list[int] = []
    visible_count = 0
    occluded_count = 0
    stale_count = 0
    tokens: list[str] = []
    for track_id in sorted(tracks_by_id):
        track = tracks_by_id[track_id]
        token = _track_lifecycle_token(tracker_namespace, tracking_result.reset_epoch, track_id)
        tokens.append(token)
        status = TrackStatus(track.status)
        if status is TrackStatus.RETIRED:
            retired_ids.append(track_id)
            continue
        observed_now = track.observed_this_step or track_id in associations
        projected[track_id] = _project_one_track(
            track,
            belief_step=tracking_result.step_index,
            tracker_namespace=tracker_namespace,
            reset_epoch=tracking_result.reset_epoch,
            radius=radius,
            observed_now=observed_now,
            lifecycle_token=token,
        )
        if observed_now:
            visible_count += 1
        elif "occluded" in track.blockers:
            occluded_count += 1
        if track.missed_steps > 0:
            stale_count += 1
    return {
        "tracks": projected,
        "belief_step": tracking_result.step_index,
        "visible_track_count": visible_count,
        "occluded_track_count": occluded_count,
        "stale_track_count": stale_count,
        "retired_track_ids": tuple(retired_ids),
        "identity_lifecycle_tokens": tuple(tokens),
    }


def _project_one_track(
    track: PedestrianTrack,
    *,
    belief_step: int,
    tracker_namespace: str,
    reset_epoch: int,
    radius: float,
    observed_now: bool,
    lifecycle_token: str,
) -> PlannerTrackBelief:
    """Convert one active or coasted record without reading observation slots.

    Returns:
        An immutable track belief with current association-based visibility.
    """
    if track.step_index != belief_step:
        raise ValueError("track step_index does not match tracking result")
    if track.missed_steps > track.age_steps:
        raise ValueError("track missed_steps exceeds age_steps")
    last_observed_step = belief_step - track.missed_steps
    if last_observed_step < 0:
        raise ValueError("track last_observed_step is outside the result step")
    covariance = np.zeros((4, 4), dtype=float)
    covariance[:2, :2] = track.position_covariance
    covariance[2:, 2:] = track.velocity_covariance
    mean_state = np.array([*track.position_global_xy, *track.velocity_global_xy, radius])
    return PlannerTrackBelief(
        track_id=track.track_id,
        mean_state=mean_state,
        covariance=covariance,
        confidence=track.association_confidence,
        existence_probability=1.0,
        existence_probability_calibrated=False,
        existence_probability_semantics=TRACK_EXISTENCE_PROBABILITY_SEMANTICS,
        visibility=observed_now,
        age_steps=track.age_steps,
        source="pedestrian_tracker",
        tracker_namespace=tracker_namespace,
        reset_epoch=reset_epoch,
        lifecycle_token=lifecycle_token,
        belief_step=belief_step,
        last_observed_step=last_observed_step,
        missed_steps=track.missed_steps,
        status=TrackStatus(track.status).value,
    )


def _track_lifecycle_token(namespace: str, reset_epoch: int, track_id: int) -> str:
    """Encode producer-owned identity tuple without delimiter ambiguity.

    Returns:
        A compact deterministic JSON token for the composite track identity.
    """
    return json.dumps(
        [namespace, reset_epoch, track_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = [
    "IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION",
    "SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION",
    "SUPPORTED_IDENTITY_SAFE_PLANNER_NAMES",
    "SUPPORTED_UNCERTAINTY_PLANNER_KEYS",
    "TRACK_EXISTENCE_PROBABILITY_SEMANTICS",
    "BeliefAwarePlannerInput",
    "IdentitySafeProjectionDiagnostics",
    "PlannerTrackBelief",
    "ScenarioBeliefPlannerProjection",
    "project_identity_safe_scenario_belief",
    "project_scenario_belief_for_planner",
]
