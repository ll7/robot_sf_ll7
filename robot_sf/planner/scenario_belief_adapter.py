"""Planner-facing ScenarioBelief uncertainty projection helpers.

These helpers are diagnostic interface smoke, not benchmark evidence. They bridge
the uncertainty-preserving ScenarioBelief report into planner-compatible observation shapes
without changing legacy policy projections. The typed seam is an entity-ID-keyed projection of
one ScenarioBelief snapshot; it does not establish cross-lifecycle identity continuity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.common.optional_import import try_import

if TYPE_CHECKING:
    from robot_sf.representation.scenario_belief import ScenarioBelief

SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION = "scenario-belief-planner-projection.v1"
SUPPORTED_UNCERTAINTY_PLANNER_KEYS = frozenset({"stream_gap"})
BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION = "belief-aware-planner-input.v1"
SUPPORTED_PROJECTION_TARGETS = frozenset({"BeliefGuidedLocalPlanner"})
# Keep the existing names discoverable for callers that use the initial adapter vocabulary.
SUPPORTED_BELIEF_AWARE_PLANNER_NAMES = SUPPORTED_PROJECTION_TARGETS
SUPPORTED_BELIEF_AWARE_PLANNER_KEYS = SUPPORTED_PROJECTION_TARGETS


def _load_scenario_belief_types() -> tuple[type[Any], type[Any]] | None:
    """Load canonical belief and visibility types only when the new seam is used.

    The existing adapter is imported by dependency-light legacy planner paths.  Keep the
    optional SciPy-backed ScenarioBelief representation out of that import path; callers that
    invoke the typed projection get an explicit unavailable/invalid fallback instead.

    Returns:
        The canonical ``ScenarioBelief`` and ``VisibilityState`` classes, or ``None`` when the
        optional representation dependencies are unavailable.
    """
    scenario_belief_module = try_import("robot_sf.representation.scenario_belief")
    if scenario_belief_module is None:
        return None
    return scenario_belief_module.ScenarioBelief, scenario_belief_module.VisibilityState


@dataclass(frozen=True)
class ScenarioBeliefPlannerProjection:
    """ScenarioBelief observation plus explicit planner uncertainty compatibility status."""

    observation: dict[str, Any]
    compatibility: dict[str, Any]


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


def _readonly_float_array(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Validate and own one finite floating-point array for a planner record.

    Returns:
        An owned, read-only float64 array with the requested shape.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must use a numeric dtype")
    try:
        owned = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.all(np.isfinite(owned)):
        raise ValueError(f"{name} must contain only finite values")
    owned.setflags(write=False)
    return owned


def _readonly_covariance(value: Any) -> np.ndarray:
    """Validate the adapter's 5D state covariance and return an owned copy.

    ``ScenarioBelief`` owns independent 2D position and velocity covariance
    matrices.  The planner state is ``[x, y, vx, vy, radius]``; the adapter
    embeds those two blocks and uses a deterministic zero-variance radius block
    because radius uncertainty and cross terms are unavailable as modelled at
    the ScenarioBelief boundary. The zero block is not evidence of measured
    zero radius uncertainty. A 4x4 block matrix is accepted for standalone
    typed-record construction and is normalized to the same 5x5 representation.

    Returns:
        An owned, read-only 5x5 positive-semidefinite covariance matrix.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("covariance must be a numeric array") from exc
    if array.shape == (4, 4):
        try:
            normalized = np.zeros((5, 5), dtype=np.float64)
            normalized[:4, :4] = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("covariance must be numeric") from exc
    elif array.shape == (5, 5):
        try:
            normalized = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("covariance must be numeric") from exc
    else:
        raise ValueError(f"covariance must have shape (5, 5) or (4, 4), got {array.shape}")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(normalized, normalized.T, atol=1e-8, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    if np.any(np.linalg.eigvalsh(normalized) < -1e-8):
        raise ValueError("covariance must be positive semidefinite")
    owned = np.array(normalized, dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return owned


def _validate_probability(name: str, value: Any) -> float:
    """Return a finite probability in the closed unit interval."""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite value in [0, 1]") from exc
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return normalized


def _validate_nonnegative_int(name: str, value: Any) -> int:
    """Return a non-negative integer without truncating fractional input."""
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _validate_track_id(value: Any) -> str | int:
    """Validate a snapshot-supplied string or integer track identifier.

    Current ``ScenarioBelief`` uses string entity IDs.  Integer IDs remain
    accepted for interoperability with the canonical prediction types, but are
    never synthesized from an observation-row position.

    Returns:
        The validated string or normalized built-in integer ID.
    """
    if isinstance(value, bool):
        raise ValueError("track_id must be a non-empty string or integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("track_id must be a non-empty string or integer")


def _track_sort_key(track_id: str | int) -> tuple[int, str | int]:
    """Return a deterministic ordering key for supported track-ID types."""
    if isinstance(track_id, int):
        return (0, track_id)
    return (1, track_id)


@dataclass(frozen=True)
class PlannerTrackBelief:
    """Immutable, entity-ID-keyed planner state from one belief snapshot.

    ``track_id`` is the entity identifier supplied by this ``ScenarioBelief``
    snapshot, not a visible-observation row number. No cross-lifecycle
    continuity is implied. The current representation supplies string IDs;
    integer IDs are retained only for standalone typed interoperability.
    ``covariance`` uses state order ``[x, y, vx, vy, radius]`` and is a 5x5
    owned, read-only array. The position/velocity blocks are adapter-projected;
    the radius block is zero because radius uncertainty is unavailable as
    modelled, not because it is known to be zero.

    The aggregate ``confidence`` is adapter-derived as the minimum of position
    and velocity confidence. Stateful consumers must reset at an externally
    supplied lifecycle boundary until the representation owner supplies a
    generation or retirement epoch.
    """

    track_id: str | int
    mean_state: np.ndarray
    covariance: np.ndarray
    confidence: float
    existence_probability: float
    visibility: bool
    age_steps: int
    source: str
    position_confidence: float | None = None
    velocity_confidence: float | None = None
    visibility_state: str | None = None

    def __post_init__(self) -> None:
        """Validate and defensively normalize all planner-track fields."""
        track_id = _validate_track_id(self.track_id)
        object.__setattr__(self, "track_id", track_id)
        mean_state = _readonly_float_array("mean_state", self.mean_state, shape=(5,))
        if mean_state[4] < 0.0:
            raise ValueError("mean_state radius must be finite and non-negative")
        object.__setattr__(self, "mean_state", mean_state)
        object.__setattr__(self, "covariance", _readonly_covariance(self.covariance))
        object.__setattr__(self, "confidence", _validate_probability("confidence", self.confidence))
        object.__setattr__(
            self,
            "existence_probability",
            _validate_probability("existence_probability", self.existence_probability),
        )
        if not isinstance(self.visibility, (bool, np.bool_)):
            raise ValueError("visibility must be a boolean")
        object.__setattr__(self, "visibility", bool(self.visibility))
        object.__setattr__(
            self, "age_steps", _validate_nonnegative_int("age_steps", self.age_steps)
        )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if self.position_confidence is not None:
            object.__setattr__(
                self,
                "position_confidence",
                _validate_probability("position_confidence", self.position_confidence),
            )
        if self.velocity_confidence is not None:
            object.__setattr__(
                self,
                "velocity_confidence",
                _validate_probability("velocity_confidence", self.velocity_confidence),
            )
        if self.visibility_state is not None and (
            not isinstance(self.visibility_state, str) or not self.visibility_state
        ):
            raise ValueError("visibility_state must be a non-empty string when provided")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe track mapping."""
        payload: dict[str, Any] = {
            "track_id": self.track_id,
            "mean_state": [float(value) for value in self.mean_state],
            "covariance": self.covariance.tolist(),
            "confidence": float(self.confidence),
            "existence_probability": float(self.existence_probability),
            "visibility": self.visibility,
            "age_steps": self.age_steps,
            "source": self.source,
        }
        if self.position_confidence is not None:
            payload["position_confidence"] = float(self.position_confidence)
        if self.velocity_confidence is not None:
            payload["velocity_confidence"] = float(self.velocity_confidence)
        if self.visibility_state is not None:
            payload["visibility_state"] = self.visibility_state
        return payload


def _copy_runtime_value(value: Any) -> Any:
    """Copy nested observation values while owning and freezing NumPy arrays.

    Returns:
        A recursively copied runtime value with independent read-only arrays.
    """
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        return {key: _copy_runtime_value(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_runtime_value(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_copy_runtime_value(nested) for nested in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _runtime_value_is_finite(value: Any) -> bool:
    """Return whether nested numeric runtime values are finite."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            return bool(np.all(np.isfinite(value)))
        return value.dtype.kind in "biu"
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.floating):
            return bool(np.isfinite(value))
        return True
    if isinstance(value, Mapping):
        return all(_runtime_value_is_finite(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return all(_runtime_value_is_finite(nested) for nested in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def _json_safe(value: Any) -> Any:
    """Convert nested runtime values to JSON primitives, rejecting non-finite data.

    Returns:
        A value composed only of JSON-compatible primitives and containers.
    """
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON export cannot contain NaN or Inf")
        return value
    return value


def _validate_planner_mapping(
    mapping: Mapping[str | int, PlannerTrackBelief],
) -> dict[str | int, PlannerTrackBelief]:
    """Validate a track mapping without allowing key/embedded-ID drift.

    Returns:
        A shallow copy of the validated mapping.
    """
    normalized: dict[str | int, PlannerTrackBelief] = {}
    for key, track in mapping.items():
        normalized_key = _validate_track_id(key)
        if not isinstance(track, PlannerTrackBelief):
            raise TypeError("tracks must contain PlannerTrackBelief values")
        if normalized_key != track.track_id:
            raise ValueError(
                f"track key mismatch: mapping key {normalized_key!r} != track_id {track.track_id!r}"
            )
        normalized[normalized_key] = track
    return {track_id: normalized[track_id] for track_id in sorted(normalized, key=_track_sort_key)}


@dataclass(frozen=True)
class BeliefAwarePlannerInput:
    """Versioned planner input preserving legacy observations and snapshot-keyed tracks."""

    legacy_observation: Mapping[str, Any]
    tracks: Mapping[str | int, PlannerTrackBelief]
    belief_step: int
    schema_version: str = BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate mappings and make caller-owned observation data independent."""
        if not isinstance(self.legacy_observation, Mapping):
            raise TypeError("legacy_observation must be a mapping")
        object.__setattr__(self, "legacy_observation", _copy_runtime_value(self.legacy_observation))
        if not isinstance(self.tracks, Mapping):
            raise TypeError("tracks must be a mapping")
        normalized_tracks = _validate_planner_mapping(self.tracks)
        object.__setattr__(self, "tracks", MappingProxyType(normalized_tracks))
        belief_step = _validate_nonnegative_int("belief_step", self.belief_step)
        object.__setattr__(self, "belief_step", belief_step)
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(
            self, "diagnostics", MappingProxyType(_copy_runtime_value(self.diagnostics))
        )

    @property
    def projection(self) -> Mapping[str, Any]:
        """Return the compact entity-ID-keyed snapshot projection diagnostics."""
        return self.diagnostics

    def ordered_track_ids(self) -> tuple[str | int, ...]:
        """Return track IDs in deterministic order."""
        return tuple(sorted(self.tracks, key=_track_sort_key))

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe input and projection diagnostics."""
        tracks: dict[str, Any] = {}
        for track_id in self.ordered_track_ids():
            serialized_id = str(track_id)
            if serialized_id in tracks:
                raise ValueError("track IDs collide after JSON object-key normalization")
            tracks[serialized_id] = self.tracks[track_id].to_dict()
        payload = {
            "schema_version": self.schema_version,
            "belief_step": self.belief_step,
            "legacy_observation": _json_safe(self.legacy_observation),
            "tracks": tracks,
            "diagnostics": _json_safe(self.diagnostics),
        }
        try:
            json.dumps(payload, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("belief-aware planner input is not JSON-safe") from exc
        return payload

    def to_json(self) -> str:
        """Return a stable compact JSON representation of this planner input."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))


def _planner_name(*, planner_name: str | None, planner_key: str | None) -> str:
    """Resolve the explicit planner-name spelling without permitting wildcards.

    Returns:
        The one non-empty planner name supplied by the caller.
    """
    if planner_name is not None and planner_key is not None and planner_name != planner_key:
        raise ValueError("planner_name and planner_key must match when both are supplied")
    resolved = planner_name if planner_name is not None else planner_key
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("planner_name must be a non-empty string")
    return resolved


def _belief_step(belief: Any) -> int:
    """Derive an integral step from canonical simulation time and timestep.

    Returns:
        A non-negative step aligned to the belief timestep.
    """
    try:
        sim_time_s = float(belief.sim_time_s)
        timestep_s = float(belief.timestep_s)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("belief time metadata is malformed") from exc
    if not np.isfinite(sim_time_s) or sim_time_s < 0.0:
        raise ValueError("belief sim_time_s must be finite and non-negative")
    if not np.isfinite(timestep_s) or timestep_s < 0.0:
        raise ValueError("belief timestep_s must be finite and non-negative")
    if timestep_s == 0.0:
        if sim_time_s == 0.0:
            return 0
        raise ValueError("belief timestep_s must be positive when sim_time_s is non-zero")
    ratio = sim_time_s / timestep_s
    rounded = round(ratio)
    if not np.isclose(ratio, rounded, atol=1e-6, rtol=0.0):
        raise ValueError("belief sim_time_s is not aligned to timestep_s")
    return rounded


def _age_steps(age_s: Any, timestep_s: Any) -> int:
    """Convert canonical observation age in seconds to conservative whole steps.

    Returns:
        A non-negative integer, rounded upward so age is never understated.
    """
    try:
        age = float(age_s)
        timestep = float(timestep_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("last_observed_age_s must be numeric") from exc
    if not np.isfinite(age) or age < 0.0:
        raise ValueError("last_observed_age_s must be finite and non-negative")
    if timestep <= 0.0:
        if age == 0.0:
            return 0
        raise ValueError("positive observation age requires a positive belief timestep")
    return max(0, int(np.ceil(age / timestep - 1e-9)))


def _planner_track_from_entity(
    agent: Any,
    *,
    timestep_s: float,
    visibility_type: type[Any],
) -> PlannerTrackBelief:
    """Build one planner track from public fields of one belief snapshot.

    Returns:
        An immutable planner track containing only public snapshot data.
    """

    if not isinstance(agent.entity_id, (str, int)) or isinstance(agent.entity_id, bool):
        raise ValueError("entity_id must be a non-empty string or integer")
    visibility_state = agent.visibility_state
    if not isinstance(visibility_state, visibility_type):
        raise ValueError("visibility_state is malformed")
    try:
        position = np.asarray(agent.position.mean_xy, dtype=np.float64).reshape(-1)
        velocity = np.asarray(agent.velocity.mean_xy, dtype=np.float64).reshape(-1)
        position_covariance = np.asarray(agent.position.covariance_xy, dtype=np.float64)
        velocity_covariance = np.asarray(agent.velocity.covariance_xy, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("entity state or covariance is malformed") from exc
    if position.shape != (2,) or velocity.shape != (2,):
        raise ValueError("entity position and velocity must have two coordinates")
    if position_covariance.shape != (2, 2) or velocity_covariance.shape != (2, 2):
        raise ValueError("entity position and velocity covariance must be 2x2")
    try:
        radius = float(agent.radius)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("entity radius is malformed") from exc
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("entity radius must be finite and non-negative")
    state = np.asarray([*position, *velocity, radius], dtype=np.float64)
    covariance = np.zeros((5, 5), dtype=np.float64)
    covariance[:2, :2] = position_covariance
    covariance[2:4, 2:4] = velocity_covariance
    position_confidence = _validate_probability("position confidence", agent.position.confidence)
    velocity_confidence = _validate_probability("velocity confidence", agent.velocity.confidence)
    confidence = min(position_confidence, velocity_confidence)
    existence_probability = _validate_probability(
        "existence_probability", agent.existence_probability
    )
    age_steps = _age_steps(agent.last_observed_age_s, timestep_s)
    source = getattr(agent.source, "adapter", None)
    if not isinstance(source, str) or not source:
        raise ValueError("entity source adapter must be a non-empty string")
    track_id = _validate_track_id(agent.entity_id)
    return PlannerTrackBelief(
        track_id=track_id,
        mean_state=state,
        covariance=covariance,
        confidence=confidence,
        existence_probability=existence_probability,
        visibility=visibility_state.value == "visible",
        age_steps=age_steps,
        source=source,
        position_confidence=position_confidence,
        velocity_confidence=velocity_confidence,
        visibility_state=visibility_state.value,
    )


def _safe_legacy_observation(belief: Any) -> tuple[dict[str, Any], str | None]:
    """Build a finite legacy fallback, or an empty non-authoritative mapping.

    Returns:
        A legacy observation and an optional fail-closed reason.
    """
    try:
        observation = belief.to_socnav_struct()
    except Exception:  # noqa: BLE001 - fail closed at the planner adapter boundary
        return {}, "legacy_observation_unavailable"
    if not isinstance(observation, Mapping) or not _runtime_value_is_finite(observation):
        return {}, "legacy_observation_nonfinite"
    return dict(observation), None


def _belief_projection_diagnostics(
    *,
    planner_name: str,
    status: str,
    belief_step: int,
    tracks: tuple[PlannerTrackBelief, ...] = (),
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build the required compact, deterministic belief-projection diagnostics.

    Returns:
        A JSON-safe diagnostic mapping with deterministic track ordering.
    """
    ordered_tracks = tuple(sorted(tracks, key=lambda track: _track_sort_key(track.track_id)))
    diagnostics: dict[str, Any] = {
        "schema_version": BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION,
        "status": status,
        "planner_name": planner_name,
        "projection_target": planner_name,
        "supported_projection_target": planner_name in SUPPORTED_PROJECTION_TARGETS,
        "belief_step": belief_step,
        "visible_track_count": sum(track.visibility for track in ordered_tracks),
        "occluded_track_count": sum(
            (
                track.visibility_state == "occluded"
                if track.visibility_state is not None
                else not track.visibility
            )
            for track in ordered_tracks
        ),
        "stale_track_count": sum(track.age_steps > 0 for track in ordered_tracks),
        "projected_track_count": len(ordered_tracks),
        "retired_track_count": None,
        "dropped_track_count": 0,
        "per_reason_drop_count": {},
        "fallback_reason": fallback_reason,
        "ordered_track_ids": [track.track_id for track in ordered_tracks],
        "uncertainty_semantics": {
            "source": "adapter_derived",
            "aggregate_confidence": "min(position_confidence, velocity_confidence)",
            "state_covariance": "position_velocity_blocks_plus_zero_radius_block",
            "radius_uncertainty": "unavailable_as_modelled",
        },
        "identity_lifecycle_status": "entity_id_only",
        "identity_generation_available": False,
        "identity_reuse_safe": False,
        "retirement_tracking": "unavailable_at_scenario_belief_boundary",
        "lifecycle_reset_required": True,
        "stateful_identity_admitted": False,
        "claim_boundary": "diagnostic_interface_smoke",
    }
    return diagnostics


def _build_belief_aware_input(
    *,
    belief: Any,
    planner_name: str,
    legacy_observation: Mapping[str, Any],
    belief_step: int,
    tracks: Mapping[str | int, PlannerTrackBelief],
    status: str,
    fallback_reason: str | None = None,
) -> BeliefAwarePlannerInput:
    """Construct one validated typed input and its diagnostics.

    Returns:
        A validated typed planner input.
    """
    del belief
    ordered_tracks = tuple(
        sorted(tracks.values(), key=lambda track: _track_sort_key(track.track_id))
    )
    diagnostics = _belief_projection_diagnostics(
        planner_name=planner_name,
        status=status,
        belief_step=belief_step,
        tracks=ordered_tracks,
        fallback_reason=fallback_reason,
    )
    return BeliefAwarePlannerInput(
        legacy_observation=legacy_observation,
        tracks=tracks,
        belief_step=belief_step,
        diagnostics=diagnostics,
    )


def project_belief_aware_planner_input(
    belief: ScenarioBelief | None,
    *,
    planner_name: str | None = None,
    planner_key: str | None = None,
) -> BeliefAwarePlannerInput:
    """Project one ScenarioBelief snapshot into an entity-ID-keyed planner input.

    ``BeliefGuidedLocalPlanner`` is the only currently supported projection
    target; this allow-list does not admit a planner implementation or stateful
    identity semantics. Missing belief, empty belief, unsupported target, and
    invalid belief are represented by distinct statuses. A valid projection
    retains every entity in ``ScenarioBelief.agents`` regardless of visibility,
    age, confidence, or existence. ``track_id`` is the identifier supplied by
    this snapshot, not a visible-row position, and no cross-lifecycle
    continuity or retirement policy is inferred here.

    This is an additive diagnostic seam.  It does not register a planner,
    alter a default roster, or change ``to_socnav_struct()``/the existing
    stream-gap adapter.

    Returns:
        A typed input with an explicit status and safe legacy fallback.
    """
    resolved_name = _planner_name(planner_name=planner_name, planner_key=planner_key)
    if belief is None:
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="no_belief",
            belief_step=0,
            fallback_reason="belief_not_supplied",
        )
        return BeliefAwarePlannerInput(
            legacy_observation={},
            tracks={},
            belief_step=0,
            diagnostics=diagnostics,
        )

    scenario_belief_types = _load_scenario_belief_types()
    if scenario_belief_types is None:
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="invalid_belief",
            belief_step=0,
            fallback_reason="scenario_belief_representation_unavailable",
        )
        return BeliefAwarePlannerInput(
            legacy_observation={},
            tracks={},
            belief_step=0,
            diagnostics=diagnostics,
        )
    scenario_belief_type, visibility_type = scenario_belief_types
    if not isinstance(belief, scenario_belief_type):
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="invalid_belief",
            belief_step=0,
            fallback_reason="belief_type_unsupported",
        )
        return BeliefAwarePlannerInput(
            legacy_observation={},
            tracks={},
            belief_step=0,
            diagnostics=diagnostics,
        )

    legacy_observation, legacy_reason = _safe_legacy_observation(belief)
    try:
        belief_step = _belief_step(belief)
    except ValueError as exc:
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="invalid_belief",
            belief_step=0,
            fallback_reason=str(exc),
        )
        return BeliefAwarePlannerInput(
            legacy_observation=legacy_observation,
            tracks={},
            belief_step=0,
            diagnostics=diagnostics,
        )

    if legacy_reason is not None:
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="invalid_belief",
            belief_step=belief_step,
            fallback_reason=legacy_reason,
        )
        return BeliefAwarePlannerInput(
            legacy_observation={},
            tracks={},
            belief_step=belief_step,
            diagnostics=diagnostics,
        )

    if resolved_name not in SUPPORTED_PROJECTION_TARGETS:
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="projection_target_not_supported",
            belief_step=belief_step,
            fallback_reason="projection_target_not_supported",
        )
        return BeliefAwarePlannerInput(
            legacy_observation=legacy_observation,
            tracks={},
            belief_step=belief_step,
            diagnostics=diagnostics,
        )

    try:
        seen_ids: set[str | int] = set()
        tracks = {}
        timestep_s = float(belief.timestep_s)
        for agent in belief.agents:
            track = _planner_track_from_entity(
                agent,
                timestep_s=timestep_s,
                visibility_type=visibility_type,
            )
            if track.track_id in seen_ids:
                raise ValueError(f"duplicate track_id {track.track_id!r}")
            seen_ids.add(track.track_id)
            tracks[track.track_id] = track
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        diagnostics = _belief_projection_diagnostics(
            planner_name=resolved_name,
            status="invalid_belief",
            belief_step=belief_step,
            fallback_reason=str(exc),
        )
        return BeliefAwarePlannerInput(
            legacy_observation=legacy_observation,
            tracks={},
            belief_step=belief_step,
            diagnostics=diagnostics,
        )

    status = "empty_belief" if not tracks else "projected"
    return _build_belief_aware_input(
        belief=belief,
        planner_name=resolved_name,
        legacy_observation=legacy_observation,
        belief_step=belief_step,
        tracks=tracks,
        status=status,
    )


def project_scenario_belief_for_belief_aware_planner(
    belief: ScenarioBelief | None,
    *,
    planner_name: str | None = None,
    planner_key: str | None = None,
) -> BeliefAwarePlannerInput:
    """Readable alias for :func:`project_belief_aware_planner_input`.

    Returns:
        The same typed input returned by the canonical projection helper.
    """
    return project_belief_aware_planner_input(
        belief,
        planner_name=planner_name,
        planner_key=planner_key,
    )


__all__ = [
    "BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION",
    "SCENARIO_BELIEF_PLANNER_PROJECTION_SCHEMA_VERSION",
    "SUPPORTED_BELIEF_AWARE_PLANNER_KEYS",
    "SUPPORTED_BELIEF_AWARE_PLANNER_NAMES",
    "SUPPORTED_PROJECTION_TARGETS",
    "SUPPORTED_UNCERTAINTY_PLANNER_KEYS",
    "BeliefAwarePlannerInput",
    "PlannerTrackBelief",
    "ScenarioBeliefPlannerProjection",
    "project_belief_aware_planner_input",
    "project_scenario_belief_for_belief_aware_planner",
    "project_scenario_belief_for_planner",
]
