"""Empirical pedestrian-model validation harness vs public trajectory datasets.

Plain-language summary: this is the trajectory-level *realism* harness requested
by issue #4975. It compares simulated pedestrian trajectories against real
reference tracks (parsed from staged public datasets such as ETH/UCY) using three
metric families, and emits a CI-friendly per-dataset *scorecard* artifact so a
force-model or parameter PR can show its realism delta. It is deliberately
distinct from the issue #3971 ``pedestrian_flow_validation`` harness, which only
runs *synthetic* no-robot fixtures and never compares against real tracks.

The three core metric families requested by #4975:

1. **Trajectory RMSE** against matched real tracks (``trajectory_rmse``). A
   simulated track and a matched real track are resampled onto a common time grid
   and compared position-by-position.
2. **Fundamental-diagram comparison** (``fundamental_diagram_comparison``):
   per-trace speed-vs-density summary distance between simulation and the real
   reference distribution, computed from the same kinematics the simulator emits.
3. **Lane-formation comparison** (``lane_formation_comparison``): emergent-pattern
   delta between the lateral-separation structure of opposite-moving pedestrians
   in the simulation versus the real reference.

Every metric is a pure function over numpy arrays, so its correctness is
provable on synthetic tracks with known ground truth (e.g. RMSE is exactly zero
for identical tracks, and scales linearly with a uniform positional offset). The
orchestrator :func:`run_realism_validation` fails closed when the real reference
data is not staged (it never presents a missing-data run as success evidence),
and falls back to a synthetic self-consistency check so CI still exercises the
metric math and the scorecard writer without license-gated bytes.

Claim boundary: this harness computes metric values and emits a scorecard. It
does not establish a calibrated realism threshold, a benchmark ranking, or a
paper-facing claim. When the real reference is absent, the scorecard is labeled
``not_available`` per the repository fail-closed contract.

The optional speed- and proxemic-distribution diagnostics are descriptive
empirical distances over the same crowd arrays. They are diagnostic-only and do
not change the core metric status or claim boundary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.data.external.eth_ucy import EthUcyDataError
from robot_sf.data.external.eth_ucy_trajectories import (
    load_provenance_validated_track_set,
)
from robot_sf.nav.map_config import SinglePedestrianDefinition

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.data.external.eth_ucy_trajectories import EthUcyTrackSet
    from robot_sf.data.external.sdd_trajectories import SddTrajectoryTrackSet

    TrackSet = EthUcyTrackSet | SddTrajectoryTrackSet

__all__ = [
    "INTERACTION_CLASSES",
    "REALISM_CLAIM_BOUNDARY",
    "REALISM_SCORECARD_SCHEMA_VERSION",
    "RECONSTRUCTION_CLAIM_BOUNDARY",
    "RECONSTRUCTION_SCHEMA_VERSION",
    "STATIC_OBSTACLE_SEMANTIC",
    "InteractionSegmentationConfig",
    "InteractionSegmentationResult",
    "InteractionWindow",
    "RealismCrowdInputs",
    "RealismEntryExitFlow",
    "RealismInteractionContext",
    "RealismMetricConfig",
    "RealismObstacle",
    "RealismReconstructionPlan",
    "RealismSceneGeometry",
    "RealismScorecard",
    "RealismStagedDatasetReference",
    "RealismTrackPair",
    "build_dataset_scorecard",
    "build_track_reconstruction_plan",
    "fundamental_diagram_comparison",
    "lane_formation_comparison",
    "lane_formation_score_curve",
    "match_tracks",
    "proxemic_distribution_distance",
    "render_scorecard_markdown",
    "resample_track",
    "run_realism_validation",
    "run_realism_validation_from_staged_dataset",
    "run_realism_validation_from_track_set",
    "segment_interactions",
    "speed_density_points",
    "speed_distribution_distance",
    "trajectory_rmse",
    "write_realism_scorecard",
]

REALISM_SCORECARD_SCHEMA_VERSION = "pedestrian_realism_validation.scorecard.v1"
REALISM_CLAIM_BOUNDARY = (
    "trajectory-level empirical realism metrics vs public trajectory datasets; "
    "no calibrated realism threshold, benchmark ranking, or paper-facing claim"
)
RECONSTRUCTION_SCHEMA_VERSION = "pedestrian_realism_validation.reconstruction.v1"
RECONSTRUCTION_CLAIM_BOUNDARY = (
    "trajectory-derived pedestrian replay seed plus validated caller-supplied static scene "
    "geometry; scene-faithful benchmark evidence remains unavailable without a simulator trace"
)
STATIC_OBSTACLE_SEMANTIC = "static_blocking"

INTERACTION_CLASSES: tuple[str, ...] = (
    "free_walking",
    "ped_ped_interaction",
    "obstacle_avoidance",
    "robot_approach",
    "crossing_conflict",
    "overtaking",
    "group",
)
_INTERACTION_LABEL_PRECEDENCE: tuple[str, ...] = (
    "robot_approach",
    "obstacle_avoidance",
    "crossing_conflict",
    "overtaking",
    "group",
    "ped_ped_interaction",
    "free_walking",
)

#: Status reported when the real reference data is not staged. Per the repository
#: fail-closed contract this is never treated as success evidence.
STATUS_NOT_AVAILABLE = "not_available"
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
EVIDENCE_STATUS_DIAGNOSTIC_ONLY = "diagnostic-only"


@dataclass(frozen=True)
class RealismMetricConfig:
    """Configuration for the realism metric computations.

    Attributes:
        resample_hz: Uniform time grid frequency used to align matched tracks.
            Higher values give finer RMSE resolution at higher compute cost.
        neighbor_radius_m: Radius for local-density (fundamental-diagram)
            estimation, in meters.
        movement_threshold_mps: Minimum along-axis speed for a pedestrian to count
            as moving in a direction (lane-formation grouping).
        max_rmse_cap_m: Sanity cap (meters) used only to flag a degenerate match;
            it never masks a computed value.
    """

    resample_hz: float = 10.0
    neighbor_radius_m: float = 1.0
    movement_threshold_mps: float = 0.05
    max_rmse_cap_m: float = 50.0

    def __post_init__(self) -> None:
        """Validate finite positive configuration."""

        for name, value, positive in (
            ("resample_hz", self.resample_hz, True),
            ("neighbor_radius_m", self.neighbor_radius_m, True),
            ("movement_threshold_mps", self.movement_threshold_mps, False),
            ("max_rmse_cap_m", self.max_rmse_cap_m, True),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if positive and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class RealismTrackPair:
    """One matched (simulation, real) track pair for RMSE comparison.

    Attributes:
        sim_time_s: Simulation sample times, shape ``(T,)``.
        sim_positions: Simulation positions, shape ``(T, 2)``.
        real_time_s: Real sample times, shape ``(T',)``.
        real_positions: Real positions, shape ``(T', 2)``.
    """

    sim_time_s: np.ndarray
    sim_positions: np.ndarray
    real_time_s: np.ndarray
    real_positions: np.ndarray


@dataclass(frozen=True)
class RealismEntryExitFlow:
    """Observed entry/exit timing and displacement for one real track.

    This record preserves the source track's temporal admission information for a replay
    planner. It is not a scene annotation: the entry and exit points are observed trajectory
    endpoints, and the flow direction is inferred from the dominant displacement axis.
    """

    pedestrian_id: int
    entry_time_s: float
    exit_time_s: float
    entry_position: tuple[float, float]
    exit_position: tuple[float, float]
    flow_direction: str

    @property
    def observed_duration_s(self) -> float:
        """Return the observed duration between the first and last samples."""

        return float(self.exit_time_s - self.entry_time_s)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for caller-owned decision packets."""

        return {
            "pedestrian_id": int(self.pedestrian_id),
            "entry_time_s": float(self.entry_time_s),
            "exit_time_s": float(self.exit_time_s),
            "entry_position": [float(value) for value in self.entry_position],
            "exit_position": [float(value) for value in self.exit_position],
            "flow_direction": self.flow_direction,
            "observed_duration_s": self.observed_duration_s,
        }


@dataclass(frozen=True)
class RealismObstacle:
    """One polygonal obstacle with an explicit collision semantic.

    The public reconstruction contract currently supports only static blocking obstacles. The
    polygon is kept in the in-memory plan for a future simulator adapter, while scorecard
    summaries expose only counts and semantic labels so geometry coordinates are not exported.
    """

    obstacle_id: str
    polygon_m: tuple[tuple[float, float], ...]
    semantic: str = STATIC_OBSTACLE_SEMANTIC

    def __post_init__(self) -> None:
        """Validate and normalize the obstacle geometry."""

        if not isinstance(self.obstacle_id, str) or not self.obstacle_id.strip():
            raise ValueError("obstacle_id must be a non-empty string")
        if self.semantic != STATIC_OBSTACLE_SEMANTIC:
            raise ValueError(
                f"unsupported obstacle semantic {self.semantic!r}; use {STATIC_OBSTACLE_SEMANTIC!r}"
            )
        try:
            polygon = np.asarray(self.polygon_m, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("polygon_m must be a finite polygon with shape (N, 2)") from exc
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise ValueError("polygon_m must have shape (N, 2) with at least three vertices")
        if not np.all(np.isfinite(polygon)):
            raise ValueError("polygon_m must contain only finite coordinates")
        if polygon.shape[0] > 3 and np.array_equal(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        if abs(_polygon_signed_area(polygon)) <= 1e-9:
            raise ValueError("polygon_m must enclose a non-zero area")
        object.__setattr__(
            self,
            "obstacle_id",
            self.obstacle_id.strip(),
        )
        object.__setattr__(
            self,
            "polygon_m",
            tuple((float(point[0]), float(point[1])) for point in polygon),
        )


@dataclass(frozen=True)
class RealismSceneGeometry:
    """Validated static scene bounds and blocking obstacles for replay seeding.

    This is an adapter input, not an ETH/UCY parser output: trajectory files do not contain
    static scene geometry. Callers must provide the bounds and obstacle polygons from a trusted
    scene source, and the reconstruction builder checks that observed tracks fit that contract.
    """

    bounds_m: tuple[tuple[float, float], tuple[float, float]]
    obstacles: tuple[RealismObstacle, ...] = ()
    source: str = "caller-supplied_scene_contract"

    def __post_init__(self) -> None:
        """Validate bounds, obstacle containment, and unique obstacle identifiers."""

        bounds = _normalize_scene_bounds(self.bounds_m)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty label")
        try:
            obstacles = tuple(self.obstacles)
        except TypeError as exc:
            raise ValueError("obstacles must be an iterable of RealismObstacle values") from exc
        if any(not isinstance(obstacle, RealismObstacle) for obstacle in obstacles):
            raise ValueError("obstacles must contain only RealismObstacle values")
        obstacle_ids = [obstacle.obstacle_id for obstacle in obstacles]
        if len(obstacle_ids) != len(set(obstacle_ids)):
            raise ValueError("obstacle_id values must be unique")
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        for obstacle in obstacles:
            polygon = np.asarray(obstacle.polygon_m, dtype=float)
            if np.any(polygon < lower) or np.any(polygon > upper):
                raise ValueError(
                    f"obstacle {obstacle.obstacle_id!r} must be contained within scene bounds"
                )
        object.__setattr__(self, "bounds_m", bounds)
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "source", self.source.strip())

    def summary_dict(self) -> dict[str, Any]:
        """Return content-light geometry metadata for reconstruction summaries."""

        return {
            "status": "validated",
            "source": self.source,
            "bounds_available": True,
            "obstacle_count": len(self.obstacles),
            "obstacle_semantics": sorted({obstacle.semantic for obstacle in self.obstacles}),
        }


@dataclass(frozen=True)
class RealismCrowdInputs:
    """Matched simulation/real crowd arrays for distribution-metric comparison.

    Bundles the ``(T, K, 2)`` position/velocity arrays the fundamental-diagram
    and lane-formation comparisons need, so the orchestrator signature stays
    small. The optional speed- and proxemic-distribution diagnostics consume the
    same arrays. Either both simulation arrays or both real arrays may be ``None``;
    the corresponding distribution metrics then remain absent (fail-closed).

    Attributes:
        sim_positions: Simulation positions shaped ``(T, K, 2)`` or ``None``.
        sim_velocities: Simulation velocities shaped ``(T, K, 2)`` or ``None``.
        real_positions: Real positions shaped ``(T', K', 2)`` or ``None``.
        real_velocities: Real velocities shaped ``(T', K', 2)`` or ``None``.
    """

    sim_positions: np.ndarray | None
    sim_velocities: np.ndarray | None
    real_positions: np.ndarray | None
    real_velocities: np.ndarray | None


@dataclass(frozen=True)
class RealismScorecard:
    """Per-dataset realism validation scorecard.

    Attributes:
        dataset_id: Reference dataset id (e.g. ``"eth-ucy/eth"``).
        status: ``"ok"``, ``"not_available"``, or ``"empty"``.
        metrics: JSON-safe metric family summaries.
        config: JSON-safe metric configuration used.
        reference_source: Provenance note for the real reference data.
        notes: Caveat and limitation notes.
        reconstruction: Content-light reconstruction readiness summary, when a parsed
            trajectory set was supplied. The serialized scorecard derives an explicit
            diagnostic-only or unavailable evidence boundary from ``status``.
    """

    dataset_id: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    reference_source: str = ""
    notes: list[str] = field(default_factory=list)
    reconstruction: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping representation of the scorecard."""

        return {
            "schema_version": REALISM_SCORECARD_SCHEMA_VERSION,
            "claim_boundary": REALISM_CLAIM_BOUNDARY,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "evidence_status": (
                EVIDENCE_STATUS_DIAGNOSTIC_ONLY
                if self.status == STATUS_OK
                else STATUS_NOT_AVAILABLE
            ),
            "metrics": self.metrics,
            "config": self.config,
            "reference_source": self.reference_source,
            "notes": list(self.notes),
            "reconstruction": self.reconstruction,
        }


@dataclass(frozen=True)
class RealismReconstructionPlan:
    """Trajectory-derived simulator seed inputs with explicit geometry status.

    The plan converts parsed real tracks into the repository's existing
    :class:`SinglePedestrianDefinition` input shape. Without ``scene_geometry``, it derives
    padded observation bounds and reports ``trajectory_bounds_only``. With validated
    ``scene_geometry``, the supplied bounds and static blocking obstacles are retained for a
    future simulator adapter. A non-empty plan remains ``partial`` until a time-faithful
    simulator trace is produced.
    """

    dataset_id: str
    split: str
    status: str
    geometry_status: str
    scene_bounds_m: tuple[tuple[float, float], tuple[float, float]] | None
    pedestrians: tuple[SinglePedestrianDefinition, ...]
    flow_axis: str | None
    flow_direction_counts: dict[str, int]
    total_sample_count: int
    blockers: tuple[str, ...]
    entry_exit_flows: tuple[RealismEntryExitFlow, ...] = ()
    timing_status: str = "unavailable"
    scene_geometry: RealismSceneGeometry | None = None

    def summary_dict(self) -> dict[str, Any]:
        """Return a content-light JSON-safe summary without trajectory coordinates."""

        return {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "claim_boundary": RECONSTRUCTION_CLAIM_BOUNDARY,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "status": self.status,
            "geometry_status": self.geometry_status,
            "flow_axis": self.flow_axis,
            "flow_direction_counts": dict(self.flow_direction_counts),
            "entry_exit_flow_count": len(self.entry_exit_flows),
            "timing_status": self.timing_status,
            "entry_exit_time_span_s": _entry_exit_time_span(self.entry_exit_flows),
            "pedestrian_count": len(self.pedestrians),
            "total_sample_count": self.total_sample_count,
            "blockers": list(self.blockers),
            "scene_geometry": (
                self.scene_geometry.summary_dict()
                if self.scene_geometry is not None
                else {
                    "status": "unavailable",
                    "bounds_available": False,
                    "obstacle_count": 0,
                    "obstacle_semantics": [],
                }
            ),
        }


@dataclass(frozen=True)
class RealismStagedDatasetReference:
    """Location and provenance pointer for one staged ETH/UCY split."""

    split: str
    root: Path | str | None = None
    provenance_manifest: Path | str | None = None
    scene_geometry: RealismSceneGeometry | None = None


@dataclass(frozen=True, slots=True)
class InteractionSegmentationConfig:
    """Declared geometric thresholds for interaction-conditioned windows.

    The segmenter is intentionally conservative.  A frame window is assigned one
    primary class using the fixed precedence in ``_INTERACTION_LABEL_PRECEDENCE``;
    the evidence fields preserve the participating track ids.  Missing robot or
    obstacle context never causes an inferred ``robot_approach`` or
    ``obstacle_avoidance`` label.
    """

    frame_window_s: float = 0.8
    frame_stride_s: float = 0.4
    minimum_speed_mps: float = 0.1
    ped_interaction_distance_m: float = 2.0
    crossing_distance_m: float = 2.0
    crossing_heading_min_deg: float = 45.0
    overtaking_distance_m: float = 2.0
    same_direction_cosine: float = 0.8
    overtaking_speed_delta_mps: float = 0.1
    group_distance_m: float = 1.5
    group_min_tracks: int = 3
    group_heading_cosine: float = 0.7
    obstacle_distance_m: float = 0.75
    obstacle_turn_angle_deg: float = 12.0
    robot_distance_m: float = 2.0
    robot_approach_min_speed_mps: float = 0.05

    def __post_init__(self) -> None:
        """Reject thresholds that would make the classifier ambiguous or non-finite."""

        for name in (
            "frame_window_s",
            "frame_stride_s",
            "ped_interaction_distance_m",
            "crossing_distance_m",
            "overtaking_distance_m",
            "group_distance_m",
            "obstacle_distance_m",
            "robot_distance_m",
        ):
            _require_positive_finite_float(getattr(self, name), name)
        for name in (
            "minimum_speed_mps",
            "overtaking_speed_delta_mps",
            "robot_approach_min_speed_mps",
        ):
            _require_non_negative_finite_float(getattr(self, name), name)
        for name in ("crossing_heading_min_deg", "obstacle_turn_angle_deg"):
            value = _require_finite_float(getattr(self, name), name)
            if not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be between 0 and 180 degrees")
        for name in ("same_direction_cosine", "group_heading_cosine"):
            value = _require_finite_float(getattr(self, name), name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        if (
            isinstance(self.group_min_tracks, bool)
            or int(self.group_min_tracks) != self.group_min_tracks
            or self.group_min_tracks < 2
        ):
            raise ValueError("group_min_tracks must be an integer >= 2")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> InteractionSegmentationConfig:
        """Build thresholds from a YAML/JSON mapping, rejecting unknown keys.

        Returns:
            Validated segmentation thresholds.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("segmentation must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"segmentation contains unsupported fields: {unknown}")
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, float | int]:
        """Return the threshold contract as JSON-safe scalar values."""

        return {
            name: (int(value) if name == "group_min_tracks" else float(value))
            for name, value in (
                (field, getattr(self, field)) for field in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class RealismInteractionContext:
    """Optional robot trajectory and trusted obstacle context for segmentation."""

    robot_time_s: np.ndarray | None = None
    robot_positions: np.ndarray | None = None
    scene_geometry: RealismSceneGeometry | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the optional robot trajectory arrays."""

        if (self.robot_time_s is None) != (self.robot_positions is None):
            raise ValueError("robot_time_s and robot_positions must be supplied together")
        if self.robot_time_s is None:
            return
        time_s = np.array(self.robot_time_s, dtype=float, copy=True).reshape(-1)
        positions = np.array(self.robot_positions, dtype=float, copy=True)
        if time_s.shape[0] < 2 or positions.shape != (time_s.shape[0], 2):
            raise ValueError("robot trajectory must contain matching time_s and (T, 2) positions")
        if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(positions)):
            raise ValueError("robot trajectory arrays must be finite")
        if not np.all(np.diff(time_s) > 0.0):
            raise ValueError("robot_time_s must be strictly increasing")
        time_s.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "robot_time_s", time_s)
        object.__setattr__(self, "robot_positions", positions)


@dataclass(frozen=True, slots=True)
class InteractionWindow:
    """One primary interaction label assigned to a time window."""

    scene_id: str
    start_time_s: float
    end_time_s: float
    label: str
    track_ids: tuple[int, ...]
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the serialized window contract."""

        if self.label not in INTERACTION_CLASSES:
            raise ValueError(f"unsupported interaction label {self.label!r}")
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not math.isfinite(self.start_time_s) or not math.isfinite(self.end_time_s):
            raise ValueError("interaction window times must be finite")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("interaction window end must be after start")
        if not self.track_ids:
            raise ValueError("interaction window must name at least one track")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "scene_id": self.scene_id,
            "start_time_s": float(self.start_time_s),
            "end_time_s": float(self.end_time_s),
            "label": self.label,
            "track_ids": list(self.track_ids),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class InteractionSegmentationResult:
    """Fail-closed segmentation result with explicit per-class denominators."""

    scene_id: str
    status: str
    windows: tuple[InteractionWindow, ...]
    counts: dict[str, int]
    config: dict[str, float | int]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the scorecard-facing JSON representation."""

        return {
            "schema_version": "interaction_conditioned_segmentation.v1",
            "claim_boundary": (
                "synthetic/real trajectory-window labels for descriptive realism stratification; "
                "no human-behavior or benchmark-ranking claim"
            ),
            "scene_id": self.scene_id,
            "status": self.status,
            "window_count": len(self.windows),
            "counts": {label: int(self.counts.get(label, 0)) for label in INTERACTION_CLASSES},
            "config": dict(self.config),
            "blockers": list(self.blockers),
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True, slots=True)
class _InteractionTrackState:
    """Interpolated state for one track inside a segmentation window."""

    track_id: int
    center_time_s: float
    position: np.ndarray
    velocity: np.ndarray
    start_position: np.ndarray | None
    end_position: np.ndarray | None


def segment_interactions(
    track_set: TrackSet | None,
    *,
    config: InteractionSegmentationConfig | None = None,
    context: RealismInteractionContext | None = None,
    scene_id: str | None = None,
) -> InteractionSegmentationResult:
    """Assign one conservative interaction label to each complete track window.

    The segmenter consumes any parsed ETH/UCY or SDD-like track set exposing a
    ``tracks`` sequence with ``pedestrian_id``, ``time_s``, and ``positions``.
    Pedestrian-only labels are inferred from geometry and finite differences;
    robot and obstacle labels require explicit caller-supplied context.  No
    external data is loaded and no missing context is treated as a positive event.

    Returns:
        A deterministic, denominator-aware segmentation result.  ``not_available``
        is returned for an absent track set and ``empty`` when no complete windows
        can be formed.
    """

    cfg = config or InteractionSegmentationConfig()
    resolved_scene_id = scene_id or _segmentation_scene_id(track_set)
    counts = dict.fromkeys(INTERACTION_CLASSES, 0)
    blockers = _segmentation_context_blockers(context)
    if track_set is None:
        return InteractionSegmentationResult(
            scene_id=resolved_scene_id,
            status=STATUS_NOT_AVAILABLE,
            windows=(),
            counts=counts,
            config=cfg.to_dict(),
            blockers=("real track set not provided", *blockers),
        )

    tracks = tuple(getattr(track_set, "tracks", ()))
    if not tracks:
        return InteractionSegmentationResult(
            scene_id=resolved_scene_id,
            status=STATUS_NOT_AVAILABLE,
            windows=(),
            counts=counts,
            config=cfg.to_dict(),
            blockers=("track set contains no parsed tracks", *blockers),
        )

    track_arrays = [_validated_segmentation_track(track) for track in tracks]
    global_start = min(float(times[0]) for times, _positions, _track_id in track_arrays)
    global_end = max(float(times[-1]) for times, _positions, _track_id in track_arrays)
    starts = _segmentation_window_starts(global_start, global_end, cfg)
    windows: list[InteractionWindow] = []
    for start_time_s in starts:
        end_time_s = start_time_s + cfg.frame_window_s
        center_time_s = start_time_s + 0.5 * cfg.frame_window_s
        states = [
            _interpolate_segmentation_state(
                times,
                positions,
                track_id,
                start_time_s=start_time_s,
                center_time_s=center_time_s,
                end_time_s=end_time_s,
            )
            for times, positions, track_id in track_arrays
        ]
        active_states = [state for state in states if state is not None]
        if not active_states:
            continue
        label, track_ids, evidence = _classify_interaction_window(
            active_states,
            config=cfg,
            context=context,
            horizon_s=cfg.frame_window_s,
        )
        window = InteractionWindow(
            scene_id=resolved_scene_id,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            label=label,
            track_ids=tuple(sorted(track_ids)),
            evidence=evidence,
        )
        windows.append(window)
        counts[label] += 1

    return InteractionSegmentationResult(
        scene_id=resolved_scene_id,
        status=STATUS_OK if windows else STATUS_EMPTY,
        windows=tuple(windows),
        counts=counts,
        config=cfg.to_dict(),
        blockers=tuple(blockers),
    )


def _segmentation_scene_id(track_set: TrackSet | None) -> str:
    """Return a stable scene id without reading trajectory content into output."""

    if track_set is None:
        return "realism/unknown"
    asset_id = str(getattr(track_set, "asset_id", "track-set"))
    scene = getattr(track_set, "scene", None)
    split = str(getattr(track_set, "split", "unknown"))
    return f"{asset_id}/{scene}/{split}" if scene else f"{asset_id}/{split}"


def _segmentation_context_blockers(
    context: RealismInteractionContext | None,
) -> list[str]:
    """Describe unavailable optional context without changing labels.

    Returns:
        Human-readable blockers for context-dependent labels.
    """

    blockers: list[str] = []
    if context is None or context.robot_positions is None:
        blockers.append("robot_approach requires a caller-supplied robot trajectory")
    if context is None or context.scene_geometry is None or not context.scene_geometry.obstacles:
        blockers.append("obstacle_avoidance requires caller-supplied static obstacle geometry")
    return blockers


def _validated_segmentation_track(
    track: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate the minimal common track interface used by the segmenter.

    Returns:
        Validated ``(time_s, positions, pedestrian_id)`` values.
    """

    track_id = getattr(track, "pedestrian_id", getattr(track, "track_id", None))
    if isinstance(track_id, bool) or track_id is None:
        raise ValueError("each segmentation track must expose an integer pedestrian_id")
    try:
        normalized_id = int(track_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("each segmentation track must expose an integer pedestrian_id") from exc
    if normalized_id != track_id:
        raise ValueError("each segmentation track must expose an integer pedestrian_id")
    times = np.asarray(getattr(track, "time_s", None), dtype=float).reshape(-1)
    positions = np.asarray(getattr(track, "positions", None), dtype=float)
    if times.shape[0] < 2 or positions.shape != (times.shape[0], 2):
        raise ValueError("segmentation tracks require time_s and positions with matching shape")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("segmentation tracks must contain finite values")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("segmentation track time_s must be strictly increasing")
    return times, positions, normalized_id


def _segmentation_window_starts(
    global_start: float,
    global_end: float,
    config: InteractionSegmentationConfig,
) -> np.ndarray:
    """Build complete, deterministic windows over the global track span.

    Returns:
        Window start times, or an empty array when no complete window exists.
    """

    if global_end - global_start < config.frame_window_s:
        return np.empty((0,), dtype=float)
    last_start = global_end - config.frame_window_s
    return np.arange(
        global_start,
        last_start + 0.5 * config.frame_stride_s,
        config.frame_stride_s,
        dtype=float,
    )


def _interpolate_segmentation_state(
    times: np.ndarray,
    positions: np.ndarray,
    track_id: int,
    *,
    start_time_s: float,
    center_time_s: float,
    end_time_s: float,
) -> _InteractionTrackState | None:
    """Interpolate one track when it is present at the window centre.

    Returns:
        The interpolated state, or ``None`` when the track is absent at centre time.
    """

    if center_time_s < times[0] or center_time_s > times[-1]:
        return None
    position = _interpolate_position(times, positions, center_time_s)
    velocity = _track_velocity_at(times, positions, center_time_s)
    start_position = (
        _interpolate_position(times, positions, start_time_s)
        if times[0] <= start_time_s <= times[-1]
        else None
    )
    end_position = (
        _interpolate_position(times, positions, end_time_s)
        if times[0] <= end_time_s <= times[-1]
        else None
    )
    return _InteractionTrackState(
        track_id=track_id,
        center_time_s=float(center_time_s),
        position=position,
        velocity=velocity,
        start_position=start_position,
        end_position=end_position,
    )


def _interpolate_position(times: np.ndarray, positions: np.ndarray, time_s: float) -> np.ndarray:
    """Linearly interpolate a two-dimensional position.

    Returns:
        The interpolated ``(x, y)`` position.
    """

    return np.asarray(
        [
            np.interp(time_s, times, positions[:, 0]),
            np.interp(time_s, times, positions[:, 1]),
        ],
        dtype=float,
    )


def _track_velocity_at(times: np.ndarray, positions: np.ndarray, time_s: float) -> np.ndarray:
    """Return the local finite-difference velocity at one track time."""

    right = int(np.searchsorted(times, time_s, side="right"))
    left = max(0, right - 1)
    right = min(right, len(times) - 1)
    if left == right:
        if left == 0:
            right = 1
        else:
            left = right - 1
    delta_t = float(times[right] - times[left])
    if delta_t <= 0.0:
        raise ValueError("segmentation track time_s must be strictly increasing")
    return np.asarray((positions[right] - positions[left]) / delta_t, dtype=float)


def _classify_interaction_window(
    states: Sequence[_InteractionTrackState],
    *,
    config: InteractionSegmentationConfig,
    context: RealismInteractionContext | None,
    horizon_s: float,
) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    """Apply the fixed primary-label precedence to one time window.

    Returns:
        The primary label, participating track ids, and evidence notes.
    """

    robot_ids = _robot_approach_ids(states, context, config=config, horizon_s=horizon_s)
    if robot_ids:
        return (
            "robot_approach",
            tuple(robot_ids),
            ("explicit robot trajectory approached pedestrian",),
        )

    obstacle_ids = _obstacle_avoidance_ids(states, context, config=config)
    if obstacle_ids:
        return (
            "obstacle_avoidance",
            tuple(obstacle_ids),
            ("turning trajectory near static obstacle",),
        )

    crossing_pair = _first_crossing_pair(states, config=config, horizon_s=horizon_s)
    if crossing_pair:
        return (
            "crossing_conflict",
            crossing_pair,
            ("opposing headings with predicted close approach",),
        )

    overtaking_pair = _first_overtaking_pair(states, config=config)
    if overtaking_pair:
        return (
            "overtaking",
            overtaking_pair,
            ("same-direction faster pedestrian behind slower pedestrian",),
        )

    group_ids = _first_group(states, config=config)
    if group_ids:
        return "group", tuple(group_ids), ("co-moving spatial cluster",)

    interaction_pair = _first_pedestrian_interaction_pair(states, config=config)
    if interaction_pair:
        return (
            "ped_ped_interaction",
            interaction_pair,
            ("close pedestrian pair with relative motion",),
        )

    return "free_walking", tuple(state.track_id for state in states), ()


def _pairwise_states(
    states: Sequence[_InteractionTrackState],
) -> Sequence[tuple[_InteractionTrackState, _InteractionTrackState]]:
    """Return unordered state pairs in deterministic track order."""

    ordered = sorted(states, key=lambda state: state.track_id)
    return tuple(
        (ordered[left], ordered[right])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    )


def _pair_geometry(
    first: _InteractionTrackState,
    second: _InteractionTrackState,
    *,
    horizon_s: float,
) -> tuple[float, float, float, float | None, float, float, float]:
    """Return distance, predicted minimum distance, time, heading cosine, and speeds."""

    relative_position = second.position - first.position
    relative_velocity = second.velocity - first.velocity
    velocity_norm_sq = float(np.dot(relative_velocity, relative_velocity))
    closest_time = 0.0
    if velocity_norm_sq > 1e-12:
        closest_time = float(
            np.clip(
                -np.dot(relative_position, relative_velocity) / velocity_norm_sq, 0.0, horizon_s
            )
        )
    closest_distance = float(np.linalg.norm(relative_position + closest_time * relative_velocity))
    distance = float(np.linalg.norm(relative_position))
    first_speed = float(np.linalg.norm(first.velocity))
    second_speed = float(np.linalg.norm(second.velocity))
    heading_cosine = _heading_cosine(first.velocity, second.velocity)
    closing_dot = float(np.dot(relative_position, relative_velocity))
    return (
        distance,
        closest_distance,
        closest_time,
        heading_cosine,
        first_speed,
        second_speed,
        closing_dot,
    )


def _heading_cosine(first_velocity: np.ndarray, second_velocity: np.ndarray) -> float | None:
    """Return cosine of the heading angle, or ``None`` for stationary tracks."""

    first_norm = float(np.linalg.norm(first_velocity))
    second_norm = float(np.linalg.norm(second_velocity))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return None
    return float(np.dot(first_velocity, second_velocity) / (first_norm * second_norm))


def _first_crossing_pair(
    states: Sequence[_InteractionTrackState],
    *,
    config: InteractionSegmentationConfig,
    horizon_s: float,
) -> tuple[int, int] | None:
    """Find the first opposing-heading pair with a predicted crossing conflict.

    Returns:
        The sorted pair of track ids, or ``None`` when no conflict is detected.
    """

    minimum_cosine = math.cos(math.radians(config.crossing_heading_min_deg))
    for first, second in _pairwise_states(states):
        distance, closest_distance, _time, heading_cosine, first_speed, second_speed, _closing = (
            _pair_geometry(first, second, horizon_s=horizon_s)
        )
        if (
            heading_cosine is not None
            and heading_cosine <= minimum_cosine
            and max(first_speed, second_speed) >= config.minimum_speed_mps
            and min(distance, closest_distance) <= config.crossing_distance_m
        ):
            return first.track_id, second.track_id
    return None


def _first_overtaking_pair(
    states: Sequence[_InteractionTrackState],
    *,
    config: InteractionSegmentationConfig,
) -> tuple[int, int] | None:
    """Find a same-direction pair where the faster track is behind.

    Returns:
        The pair of track ids, or ``None`` when no overtake is detected.
    """

    for first, second in _pairwise_states(states):
        distance, _closest, _time, heading_cosine, first_speed, second_speed, _closing = (
            _pair_geometry(first, second, horizon_s=0.0)
        )
        if heading_cosine is None or heading_cosine < config.same_direction_cosine:
            continue
        if distance > config.overtaking_distance_m:
            continue
        direction = first.velocity + second.velocity
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            continue
        direction /= direction_norm
        first_is_behind = float(np.dot(first.position - second.position, direction)) < 0.0
        second_is_behind = not first_is_behind
        if (first_is_behind and first_speed > second_speed + config.overtaking_speed_delta_mps) or (
            second_is_behind and second_speed > first_speed + config.overtaking_speed_delta_mps
        ):
            return first.track_id, second.track_id
    return None


def _first_group(
    states: Sequence[_InteractionTrackState],
    *,
    config: InteractionSegmentationConfig,
) -> tuple[int, ...] | None:
    """Find the largest deterministic co-moving spatial cluster.

    Returns:
        Sorted cluster track ids, or ``None`` when no qualifying group exists.
    """

    if len(states) < config.group_min_tracks:
        return None
    ordered = sorted(states, key=lambda state: state.track_id)
    adjacency = {state.track_id: set() for state in ordered}
    for first, second in _pairwise_states(ordered):
        distance = float(np.linalg.norm(first.position - second.position))
        heading_cosine = _heading_cosine(first.velocity, second.velocity)
        first_speed = float(np.linalg.norm(first.velocity))
        second_speed = float(np.linalg.norm(second.velocity))
        if (
            distance <= config.group_distance_m
            and heading_cosine is not None
            and heading_cosine >= config.group_heading_cosine
            and max(first_speed, second_speed) >= config.minimum_speed_mps
            and abs(first_speed - second_speed) <= max(0.5, 2.0 * config.overtaking_speed_delta_mps)
        ):
            adjacency[first.track_id].add(second.track_id)
            adjacency[second.track_id].add(first.track_id)
    components: list[tuple[int, ...]] = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        if len(component) >= config.group_min_tracks:
            components.append(tuple(sorted(component)))
    return max(
        components,
        key=lambda component: (len(component), tuple(-value for value in component)),
        default=None,
    )


def _first_pedestrian_interaction_pair(
    states: Sequence[_InteractionTrackState],
    *,
    config: InteractionSegmentationConfig,
) -> tuple[int, int] | None:
    """Find a close pair with enough relative motion to be an interaction.

    Returns:
        The pair of track ids, or ``None`` when no interaction is detected.
    """

    for first, second in _pairwise_states(states):
        distance, _closest, _time, heading_cosine, first_speed, second_speed, closing_dot = (
            _pair_geometry(first, second, horizon_s=0.0)
        )
        relative_speed = float(np.linalg.norm(first.velocity - second.velocity))
        headings_differ = (
            heading_cosine is not None and heading_cosine < config.same_direction_cosine
        )
        if (
            distance <= config.ped_interaction_distance_m
            and max(first_speed, second_speed) >= config.minimum_speed_mps
            and (
                closing_dot < 0.0
                or relative_speed >= config.overtaking_speed_delta_mps
                or headings_differ
            )
        ):
            return first.track_id, second.track_id
    return None


def _robot_approach_ids(
    states: Sequence[_InteractionTrackState],
    context: RealismInteractionContext | None,
    *,
    config: InteractionSegmentationConfig,
    horizon_s: float,
) -> tuple[int, ...]:
    """Return pedestrians approached by the explicit robot trajectory.

    Returns:
        Sorted pedestrian ids approached within the configured horizon.
    """

    if context is None or context.robot_time_s is None or context.robot_positions is None:
        return ()
    robot_time_s = context.robot_time_s
    robot_positions = context.robot_positions
    center_time_s = _state_center_time(states)
    if center_time_s < robot_time_s[0] or center_time_s > robot_time_s[-1]:
        return ()
    robot_position = _interpolate_position(robot_time_s, robot_positions, center_time_s)
    robot_velocity = _track_velocity_at(robot_time_s, robot_positions, center_time_s)
    ids: list[int] = []
    for state in states:
        relative_position = state.position - robot_position
        relative_velocity = state.velocity - robot_velocity
        distance = float(np.linalg.norm(relative_position))
        velocity_norm_sq = float(np.dot(relative_velocity, relative_velocity))
        closest_distance = distance
        if velocity_norm_sq > 1e-12:
            closest_time = float(
                np.clip(
                    -np.dot(relative_position, relative_velocity) / velocity_norm_sq, 0.0, horizon_s
                )
            )
            closest_distance = float(
                np.linalg.norm(relative_position + closest_time * relative_velocity)
            )
        approach_rate = -float(np.dot(relative_position, relative_velocity)) / max(distance, 1e-12)
        if (
            min(distance, closest_distance) <= config.robot_distance_m
            and approach_rate >= config.robot_approach_min_speed_mps
        ):
            ids.append(state.track_id)
    return tuple(sorted(ids))


def _state_center_time(states: Sequence[_InteractionTrackState]) -> float:
    """Return a representative centre time carried by the first state."""

    return float(states[0].center_time_s)


def _obstacle_avoidance_ids(
    states: Sequence[_InteractionTrackState],
    context: RealismInteractionContext | None,
    *,
    config: InteractionSegmentationConfig,
) -> tuple[int, ...]:
    """Return pedestrians turning near trusted static obstacles.

    Returns:
        Sorted pedestrian ids with a turning trajectory near an obstacle.
    """

    if context is None or context.scene_geometry is None or not context.scene_geometry.obstacles:
        return ()
    ids: list[int] = []
    polygons = [
        np.asarray(obstacle.polygon_m, dtype=float) for obstacle in context.scene_geometry.obstacles
    ]
    for state in states:
        if state.start_position is None or state.end_position is None:
            continue
        distance = min(_point_to_polygon_distance(state.position, polygon) for polygon in polygons)
        reference = state.end_position - state.start_position
        reference_norm = float(np.linalg.norm(reference))
        velocity_norm = float(np.linalg.norm(state.velocity))
        if (
            distance > config.obstacle_distance_m
            or reference_norm <= 1e-12
            or velocity_norm <= 1e-12
        ):
            continue
        cosine = float(np.dot(reference, state.velocity) / (reference_norm * velocity_norm))
        turn_angle_deg = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        if turn_angle_deg >= config.obstacle_turn_angle_deg:
            ids.append(state.track_id)
    return tuple(sorted(ids))


def _point_to_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    """Return the Euclidean distance from a point to a polygon boundary/interior."""

    if _point_inside_polygon(point, polygon):
        return 0.0
    distances = [
        _point_to_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    return float(min(distances))


def _point_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Use a deterministic ray-crossing test for a non-self-intersecting polygon.

    Returns:
        ``True`` when the point is inside the polygon.
    """

    x, y = float(point[0]), float(point[1])
    inside = False
    for index in range(len(polygon)):
        x_first, y_first = polygon[index]
        x_second, y_second = polygon[(index + 1) % len(polygon)]
        crosses = (y_first > y) != (y_second > y)
        if crosses:
            crossing_x = (x_second - x_first) * (y - y_first) / (y_second - y_first) + x_first
            if x < crossing_x:
                inside = not inside
    return inside


def _point_to_segment_distance(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the Euclidean point-to-segment distance."""

    segment = second - first
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - first))
    fraction = float(np.clip(np.dot(point - first, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + fraction * segment)))


# --------------------------------------------------------------------------- #
# 1. Trajectory RMSE
# --------------------------------------------------------------------------- #


def resample_track(
    time_s: np.ndarray,
    positions: np.ndarray,
    *,
    resample_hz: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a single track onto a uniform time grid by linear interpolation.

    Args:
        time_s: Strictly increasing sample times, shape ``(T,)``.
        positions: Positions, shape ``(T, 2)``.
        resample_hz: Target grid frequency in Hz.
        t_start: Grid start time. Defaults to the first sample time.
        t_end: Grid end time. Defaults to the last sample time.

    Returns:
        A ``(grid_time, grid_positions)`` tuple. ``grid_positions`` has shape
        ``(G, 2)`` where ``G >= 1``. When the overlap window is degenerate a
        single-sample grid is returned.

    Raises:
        ValueError: If ``time_s`` and ``positions`` are inconsistent, ``time_s``
            is not monotonically non-decreasing, or ``resample_hz`` is not
            positive.
    """

    time_arr = np.asarray(time_s, dtype=float).reshape(-1)
    pos_arr = np.asarray(positions, dtype=float)
    if pos_arr.ndim != 2 or pos_arr.shape[1] != 2:
        raise ValueError("positions must have shape (T, 2)")
    if time_arr.shape[0] != pos_arr.shape[0]:
        raise ValueError("time_s and positions must share the first dimension")
    if time_arr.shape[0] < 2:
        raise ValueError("at least two samples are required to resample")
    if resample_hz <= 0.0 or not math.isfinite(resample_hz):
        raise ValueError("resample_hz must be finite and positive")
    if np.any(np.diff(time_arr) < 0.0):
        raise ValueError("time_s must be monotonically non-decreasing")

    start = float(time_arr[0]) if t_start is None else float(t_start)
    end = float(time_arr[-1]) if t_end is None else float(t_end)
    if end <= start:
        # Degenerate overlap: return a single sample at the boundary midpoint.
        mid = 0.5 * (start + end)
        return np.asarray([mid]), np.atleast_2d(np.interp(mid, time_arr, pos_arr.T).T)
    step = 1.0 / resample_hz
    grid_time = np.arange(start, end + 0.5 * step, step)
    grid_x = np.interp(grid_time, time_arr, pos_arr[:, 0])
    grid_y = np.interp(grid_time, time_arr, pos_arr[:, 1])
    return grid_time, np.stack((grid_x, grid_y), axis=1)


def trajectory_rmse(pair: RealismTrackPair, *, config: RealismMetricConfig) -> dict[str, Any]:
    """Position RMSE between a matched simulation and real track.

    Both tracks are resampled onto the uniform grid spanning their *time overlap*
    window, then compared position-by-position. The reported value is the root
    mean square of Euclidean position errors in meters.

    Returns:
        A JSON-safe mapping with ``rmse_m``, ``sample_count``, ``overlap_s``, and
        ``status``. ``status`` is ``"empty"`` when the tracks share no time
        overlap or too few aligned samples.
    """

    sim_t = np.asarray(pair.sim_time_s, dtype=float)
    real_t = np.asarray(pair.real_time_s, dtype=float)
    if sim_t.shape[0] < 2 or real_t.shape[0] < 2:
        return _empty_metric("trajectory_rmse")
    overlap_start = float(max(sim_t[0], real_t[0]))
    overlap_end = float(min(sim_t[-1], real_t[-1]))
    if overlap_end - overlap_start <= config.resample_hz * 1e-9:
        return _empty_metric("trajectory_rmse")
    _sim_grid_t, sim_grid = resample_track(
        sim_t,
        np.asarray(pair.sim_positions, dtype=float),
        resample_hz=config.resample_hz,
        t_start=overlap_start,
        t_end=overlap_end,
    )
    _real_grid_t, real_grid = resample_track(
        real_t,
        np.asarray(pair.real_positions, dtype=float),
        resample_hz=config.resample_hz,
        t_start=overlap_start,
        t_end=overlap_end,
    )
    n = min(sim_grid.shape[0], real_grid.shape[0])
    if n < 2:
        return _empty_metric("trajectory_rmse")
    diff = sim_grid[:n] - real_grid[:n]
    errors = np.sqrt(np.sum(diff * diff, axis=1))
    rmse = float(np.sqrt(np.mean(errors * errors)))
    return {
        "metric_id": "trajectory_rmse",
        "rmse_m": rmse,
        "sample_count": int(n),
        "overlap_s": float(overlap_end - overlap_start),
        "status": STATUS_OK,
    }


def match_tracks(
    sim_tracks: Sequence[RealismTrackPair] | None,
) -> list[RealismTrackPair]:
    """Return the list of (sim, real) track pairs to score.

    Matching simulated pedestrians to real tracks is dataset- and
    scenario-specific; callers build pairs (e.g. by entry region or nearest
    seed). This helper normalizes the input and filters degenerate pairs so the
    RMSE aggregator only sees well-formed pairs.

    Returns:
        A list of :class:`RealismTrackPair` with at least two samples on each side.
    """

    pairs: list[RealismTrackPair] = []
    if not sim_tracks:
        return pairs
    for pair in sim_tracks:
        if np.asarray(pair.sim_time_s).shape[0] >= 2 and np.asarray(pair.real_time_s).shape[0] >= 2:
            pairs.append(pair)
    return pairs


# --------------------------------------------------------------------------- #
# 2. Fundamental-diagram comparison
# --------------------------------------------------------------------------- #


def speed_density_points(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    neighbor_radius_m: float,
) -> np.ndarray:
    """Return per-frame ``(density, speed)`` fundamental-diagram points.

    Args:
        positions: ``(T, K, 2)`` positions in meters.
        velocities: ``(T, K, 2)`` velocities in m/s.
        neighbor_radius_m: Radius for local-density estimation.

    Returns:
        An ``(N, 2)`` array of ``(local_density_ped_per_m2, speed_mps)`` samples
        over all frames and pedestrians. Density uses an area-normalized neighbor
        count within ``neighbor_radius_m``.

    Raises:
        ValueError: If the arrays have inconsistent shapes.
    """

    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("positions must have shape (T, K, 2)")
    if vel.shape != pos.shape:
        raise ValueError("velocities must have the same shape as positions")
    if not math.isfinite(neighbor_radius_m) or neighbor_radius_m <= 0.0:
        raise ValueError("neighbor_radius_m must be finite and positive")
    if pos.shape[0] == 0 or pos.shape[1] == 0:
        return np.empty((0, 2), dtype=float)
    speed = np.linalg.norm(vel, axis=2)
    density_area = math.pi * float(neighbor_radius_m) ** 2
    # A gridded real track set uses NaNs before a pedestrian enters and after it
    # leaves the scene.  Compute each frame over only the observed pedestrians;
    # otherwise one absent pedestrian makes every pairwise distance NaN and
    # silently reports zero density for the people that are present.
    density = np.full(pos.shape[:2], np.nan, dtype=float)
    for t in range(pos.shape[0]):
        frame = pos[t]
        observed = np.all(np.isfinite(frame), axis=1) & np.all(np.isfinite(vel[t]), axis=1)
        if not np.any(observed):
            continue
        observed_frame = frame[observed]
        # Pairwise distance matrix for this frame (K may be large for big scenes,
        # but this harness is for short validation fixtures).
        diff = observed_frame[:, None, :] - observed_frame[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        neighbors = np.count_nonzero(dist <= float(neighbor_radius_m), axis=1) - 1
        density[t, observed] = np.maximum(neighbors, 0) / density_area
    return np.stack((density.reshape(-1), speed.reshape(-1)), axis=1)


def fundamental_diagram_comparison(
    sim_points: np.ndarray,
    real_points: np.ndarray,
) -> dict[str, Any]:
    """Compare simulation and real fundamental-diagram point clouds.

    The comparison summarizes each distribution's mean speed and mean density,
    then reports the absolute delta of the mean speeds and the 1-Wasserstein-like
    scalar distance between the two speed marginals (mean absolute difference of
    sorted samples). This is a descriptive distribution distance, not a pass/fail
    threshold.

    Returns:
        A JSON-safe mapping with per-distribution means and the speed-distance
        metric. ``status`` is ``"empty"`` when either distribution has no samples.
    """

    sim = _finite_points(sim_points)
    real = _finite_points(real_points)
    if sim.shape[0] == 0 or real.shape[0] == 0:
        return _empty_metric("fundamental_diagram_comparison")
    sim_speed = sim[:, 1]
    real_speed = real[:, 1]
    speed_distance = _sorted_distance(sim_speed, real_speed)
    return {
        "metric_id": "fundamental_diagram_comparison",
        "sim": _density_speed_summary(sim),
        "real": _density_speed_summary(real),
        "mean_speed_delta_mps": float(abs(np.mean(sim_speed) - np.mean(real_speed))),
        "mean_density_delta_ped_per_m2": float(abs(np.mean(sim[:, 0]) - np.mean(real[:, 0]))),
        "speed_marginal_distance_mps": float(speed_distance),
        "sim_sample_count": int(sim.shape[0]),
        "real_sample_count": int(real.shape[0]),
        "status": STATUS_OK,
    }


# --------------------------------------------------------------------------- #
# 3. Lane-formation comparison
# --------------------------------------------------------------------------- #


def lane_formation_score_curve(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    movement_axis: int,
    lateral_axis: int,
    movement_threshold_mps: float,
) -> np.ndarray:
    """Return the per-frame lane-separation score curve for a crowd.

    For each frame, pedestrians are split into two groups by their sign of
    along-axis velocity (above ``movement_threshold_mps``). The score is the
    absolute difference of the two groups' mean lateral positions, normalized by
    the frame's lateral spread. Higher scores indicate clearer two-lane
    separation. Frames lacking both directions are skipped.

    Returns:
        A ``(F,)`` float array of per-frame scores in ``[0, 1]`` (may be empty).
    """

    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    if pos.ndim != 3 or vel.shape != pos.shape:
        raise ValueError("positions and velocities must have matching shape (T, K, 2)")
    if movement_axis not in (0, 1) or lateral_axis not in (0, 1):
        raise ValueError("movement_axis and lateral_axis must be 0 or 1")
    if movement_axis == lateral_axis:
        raise ValueError("movement_axis and lateral_axis must differ")
    if pos.shape[0] == 0 or pos.shape[1] < 2:
        return np.empty((0,), dtype=float)
    movement = vel[:, :, movement_axis]
    lateral = pos[:, :, lateral_axis]
    positive = movement > movement_threshold_mps
    negative = movement < -movement_threshold_mps
    scores: list[float] = []
    for frame_index, (lateral_t, pos_mask, neg_mask) in enumerate(
        zip(lateral, positive, negative, strict=True)
    ):
        # A gridded track is NaN outside its observed lifespan.  Exclude it from
        # both direction groups and the lateral spread without discarding the
        # other observed pedestrians in the frame.
        observed = np.isfinite(lateral_t) & np.all(np.isfinite(vel[frame_index]), axis=1)
        pos_mask = pos_mask & observed
        neg_mask = neg_mask & observed
        if np.count_nonzero(pos_mask) == 0 or np.count_nonzero(neg_mask) == 0:
            continue
        pos_mean = float(np.mean(lateral_t[pos_mask]))
        neg_mean = float(np.mean(lateral_t[neg_mask]))
        spread = float(np.max(lateral_t[observed]) - np.min(lateral_t[observed]))
        if spread > 0.0:
            scores.append(abs(pos_mean - neg_mean) / spread)
    return np.asarray(scores, dtype=float)


def lane_formation_comparison(
    sim_positions: np.ndarray,
    sim_velocities: np.ndarray,
    real_positions: np.ndarray,
    real_velocities: np.ndarray,
    *,
    config: RealismMetricConfig,
    movement_axis: int = 0,
    lateral_axis: int = 1,
) -> dict[str, Any]:
    """Compare emergent lane-formation structure between simulation and real.

    Both crowds are scored with :func:`lane_formation_score_curve`; the
    comparison reports each curve's mean and the absolute delta of the means.

    Returns:
        A JSON-safe mapping with per-source mean lane scores and the delta. The
        ``status`` is ``"empty"`` when either crowd has no scorable frames.
    """

    sim_curve = lane_formation_score_curve(
        sim_positions,
        sim_velocities,
        movement_axis=movement_axis,
        lateral_axis=lateral_axis,
        movement_threshold_mps=config.movement_threshold_mps,
    )
    real_curve = lane_formation_score_curve(
        real_positions,
        real_velocities,
        movement_axis=movement_axis,
        lateral_axis=lateral_axis,
        movement_threshold_mps=config.movement_threshold_mps,
    )
    sim_mean = float(np.mean(sim_curve)) if sim_curve.size else 0.0
    real_mean = float(np.mean(real_curve)) if real_curve.size else 0.0
    if sim_curve.size == 0 or real_curve.size == 0:
        return {
            "metric_id": "lane_formation_comparison",
            "sim": {"mean_score": sim_mean, "frame_count": int(sim_curve.size)},
            "real": {"mean_score": real_mean, "frame_count": int(real_curve.size)},
            "mean_score_delta": float(abs(sim_mean - real_mean)),
            "status": STATUS_EMPTY,
        }
    return {
        "metric_id": "lane_formation_comparison",
        "sim": {"mean_score": sim_mean, "frame_count": int(sim_curve.size)},
        "real": {"mean_score": real_mean, "frame_count": int(real_curve.size)},
        "mean_score_delta": float(abs(sim_mean - real_mean)),
        "status": STATUS_OK,
    }


# --------------------------------------------------------------------------- #
# 4. Speed-distribution distance (diagnostic-only)
# --------------------------------------------------------------------------- #

SPEED_DISTRIBUTION_UNITS = "m/s"
SPEED_DISTRIBUTION_CLAIM_BOUNDARY = (
    "diagnostic-only empirical 1-Wasserstein distance between simulation and "
    "reference speed distributions; no calibrated threshold, ranking, or "
    "paper-facing claim"
)


def speed_distribution_distance(
    sim_velocities: np.ndarray,
    real_velocities: np.ndarray,
) -> dict[str, Any]:
    """Compute the empirical 1-Wasserstein distance between speed distributions.

    Speed samples are finite Euclidean norms from ``(T, K, 2)`` velocity
    arrays.  Non-finite observations are excluded and counted.

    Returns:
        A JSON-safe mapping with ``status``, ``distance`` (when ok), per-arm
        sample counts and statistics, ``units``, ``evidence_status``, and a
        controlled ``empty_reason`` when unavailable.
    """

    sim_samples = _extract_speed_samples(sim_velocities)
    real_samples = _extract_speed_samples(real_velocities)
    excluded = _count_nonfinite_speed_observations(
        sim_velocities
    ) + _count_nonfinite_speed_observations(real_velocities)
    if sim_samples.size == 0 or real_samples.size == 0:
        empty_reason = (
            "no finite speed samples in simulation arm"
            if sim_samples.size == 0
            else "no finite speed samples in real arm"
        )
        if sim_samples.size == 0 and real_samples.size == 0:
            empty_reason = "no finite speed samples in either arm"
        return {
            "metric_id": "speed_distribution_distance",
            "status": STATUS_EMPTY,
            "distance": None,
            "sim_sample_count": int(sim_samples.size),
            "real_sample_count": int(real_samples.size),
            "sim": _distribution_arm_stats(sim_samples),
            "real": _distribution_arm_stats(real_samples),
            "units": SPEED_DISTRIBUTION_UNITS,
            "evidence_status": EVIDENCE_STATUS_DIAGNOSTIC_ONLY,
            "claim_boundary": SPEED_DISTRIBUTION_CLAIM_BOUNDARY,
            "empty_reason": empty_reason,
            "excluded_nonfinite_count": excluded,
        }
    distance = _empirical_wasserstein_1d(sim_samples, real_samples)
    return {
        "metric_id": "speed_distribution_distance",
        "status": STATUS_OK,
        "distance": distance,
        "sim_sample_count": int(sim_samples.size),
        "real_sample_count": int(real_samples.size),
        "sim": _distribution_arm_stats(sim_samples),
        "real": _distribution_arm_stats(real_samples),
        "units": SPEED_DISTRIBUTION_UNITS,
        "evidence_status": EVIDENCE_STATUS_DIAGNOSTIC_ONLY,
        "claim_boundary": SPEED_DISTRIBUTION_CLAIM_BOUNDARY,
        "empty_reason": None,
        "excluded_nonfinite_count": excluded,
    }


def _extract_speed_samples(velocities: np.ndarray) -> np.ndarray:
    """Extract finite Euclidean speed norms from a ``(T, K, 2)`` velocity array.

    Returns:
        A 1-D array of finite speed samples (may be empty).
    """

    vel = np.asarray(velocities, dtype=float)
    if vel.ndim != 3 or vel.shape[2] != 2:
        raise ValueError("velocities must have shape (T, K, 2)")
    speed = np.linalg.norm(vel, axis=2).reshape(-1)
    return speed[np.isfinite(speed)]


# --------------------------------------------------------------------------- #
# 5. Proxemic-distribution distance (diagnostic-only)
# --------------------------------------------------------------------------- #

PROXEMIC_DISTRIBUTION_UNITS = "m"
PROXEMIC_DISTRIBUTION_CLAIM_BOUNDARY = (
    "diagnostic-only empirical 1-Wasserstein distance between simulation and "
    "reference within-frame proxemic (pedestrian-pair) distributions; no "
    "calibrated threshold, ranking, or paper-facing claim"
)


def proxemic_distribution_distance(
    sim_positions: np.ndarray,
    real_positions: np.ndarray,
) -> dict[str, Any]:
    """Compute the empirical 1-Wasserstein distance between proxemic distributions.

    Proxemic samples are each unordered distinct pedestrian pair once per
    frame, excluding the diagonal but preserving coincident distinct
    pedestrians.  Non-finite observations are excluded and counted.

    Returns:
        A JSON-safe mapping with ``status``, ``distance`` (when ok), per-arm
        sample counts and statistics, ``units``, ``evidence_status``, and a
        controlled ``empty_reason`` when unavailable.
    """

    sim_samples = _extract_proxemic_samples(sim_positions)
    real_samples = _extract_proxemic_samples(real_positions)
    excluded_sim = _count_nonfinite_position_observations(sim_positions)
    excluded_real = _count_nonfinite_position_observations(real_positions)
    excluded = excluded_sim + excluded_real
    if sim_samples.size == 0 or real_samples.size == 0:
        empty_reasons: list[str] = []
        if sim_samples.size == 0:
            empty_reasons.append(_proxemic_empty_reason(sim_positions, "simulation"))
        if real_samples.size == 0:
            empty_reasons.append(_proxemic_empty_reason(real_positions, "real"))
        empty_reason = "; ".join(empty_reasons)
        return {
            "metric_id": "proxemic_distribution_distance",
            "status": STATUS_EMPTY,
            "distance": None,
            "sim_sample_count": int(sim_samples.size),
            "real_sample_count": int(real_samples.size),
            "sim": _distribution_arm_stats(sim_samples),
            "real": _distribution_arm_stats(real_samples),
            "units": PROXEMIC_DISTRIBUTION_UNITS,
            "evidence_status": EVIDENCE_STATUS_DIAGNOSTIC_ONLY,
            "claim_boundary": PROXEMIC_DISTRIBUTION_CLAIM_BOUNDARY,
            "empty_reason": empty_reason,
            "excluded_nonfinite_count": excluded,
        }
    distance = _empirical_wasserstein_1d(sim_samples, real_samples)
    return {
        "metric_id": "proxemic_distribution_distance",
        "status": STATUS_OK,
        "distance": distance,
        "sim_sample_count": int(sim_samples.size),
        "real_sample_count": int(real_samples.size),
        "sim": _distribution_arm_stats(sim_samples),
        "real": _distribution_arm_stats(real_samples),
        "units": PROXEMIC_DISTRIBUTION_UNITS,
        "evidence_status": EVIDENCE_STATUS_DIAGNOSTIC_ONLY,
        "claim_boundary": PROXEMIC_DISTRIBUTION_CLAIM_BOUNDARY,
        "empty_reason": None,
        "excluded_nonfinite_count": excluded,
    }


def _extract_proxemic_samples(positions: np.ndarray) -> np.ndarray:
    """Extract unordered distinct-pair distances from a ``(T, K, 2)`` position array.

    For each frame, the Euclidean distance between every unordered distinct
    pair of pedestrians is recorded once.  Frames with fewer than two
    observed (finite) pedestrians are skipped.  The diagonal is excluded but
    coincident distinct pedestrians (distance zero) are preserved.

    Returns:
        A 1-D array of pairwise distances (may be empty).
    """

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("positions must have shape (T, K, 2)")
    if pos.shape[0] == 0 or pos.shape[1] < 2:
        return np.empty(0, dtype=float)
    all_distances: list[np.ndarray] = []
    for t in range(pos.shape[0]):
        frame = pos[t]
        observed = np.all(np.isfinite(frame), axis=1)
        obs_indices = np.nonzero(observed)[0]
        if obs_indices.size < 2:
            continue
        obs_positions = frame[obs_indices]
        diff = obs_positions[:, None, :] - obs_positions[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        # Extract upper triangle (excluding diagonal) for unordered pairs.
        tri_indices = np.triu_indices(obs_positions.shape[0], k=1)
        all_distances.append(dist[tri_indices])
    if not all_distances:
        return np.empty(0, dtype=float)
    return np.concatenate(all_distances)


def _count_nonfinite_speed_observations(velocities: np.ndarray) -> int:
    """Count non-finite vector observations in a ``(T, K, 2)`` array.

    Returns:
        Number of pedestrian-frame velocity vectors with at least one non-finite
        component.
    """

    vel = np.asarray(velocities, dtype=float)
    if vel.ndim != 3 or vel.shape[2] != 2:
        raise ValueError("velocities must have shape (T, K, 2)")
    return int(np.count_nonzero(~np.all(np.isfinite(vel), axis=2)))


def _count_nonfinite_position_observations(positions: np.ndarray) -> int:
    """Count non-finite pedestrian-frame observations in a ``(T, K, 2)`` array.

    Returns:
        Number of pedestrian-frame position vectors with at least one non-finite
        component.
    """

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("positions must have shape (T, K, 2)")
    return int(np.count_nonzero(~np.all(np.isfinite(pos), axis=2)))


def _proxemic_empty_reason(positions: np.ndarray, arm_name: str) -> str:
    """Explain whether an arm lacks finite pair opportunities or finite samples.

    Returns:
        A controlled empty-result reason for the named arm.
    """

    pos = np.asarray(positions, dtype=float)
    has_pair = any(np.count_nonzero(np.all(np.isfinite(frame), axis=1)) >= 2 for frame in pos)
    if not has_pair:
        return f"fewer than two finite pedestrians in every frame of {arm_name} arm"
    return f"no finite proxemic samples in {arm_name} arm"


# --------------------------------------------------------------------------- #
# Common distribution helpers
# --------------------------------------------------------------------------- #


def _empirical_wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Unweighted empirical 1-Wasserstein distance over finite 1-D samples.

    Both arms are filtered to finite samples and deterministically sorted. The empirical
    quantile functions
    are integrated over the union of their step boundaries, which handles
    unequal sample counts without histogram bins. This is a descriptive
    distribution distance, not a pass/fail threshold.

    Raises:
        ValueError: If either arm has no finite sample.

    Returns:
        The scalar Wasserstein-1 distance in the units of the input samples.
    """

    flat_a = np.asarray(a, dtype=float).ravel()
    flat_b = np.asarray(b, dtype=float).ravel()
    sa = np.sort(flat_a[np.isfinite(flat_a)])
    sb = np.sort(flat_b[np.isfinite(flat_b)])
    if sa.size == 0 or sb.size == 0:
        raise ValueError("both arms must contain at least one finite sample")
    index_a = 0
    index_b = 0
    position = 0.0
    distance = 0.0
    while position < 1.0:
        next_a = (index_a + 1) / sa.size
        next_b = (index_b + 1) / sb.size
        next_position = min(next_a, next_b)
        distance += (next_position - position) * abs(sa[index_a] - sb[index_b])
        position = next_position
        if index_a + 1 < sa.size and next_a <= position:
            index_a += 1
        if index_b + 1 < sb.size and next_b <= position:
            index_b += 1
    return float(distance)


def _distribution_arm_stats(samples: np.ndarray) -> dict[str, Any]:
    """Return summary statistics for a 1-D sample arm.

    Returns:
        A mapping with ``mean``, ``std``, ``quantile_010``, ``quantile_050``,
        ``quantile_090``, and ``count``.  Values are ``None`` when the arm is
        empty.
    """

    if samples.size == 0:
        return {
            "mean": None,
            "std": None,
            "quantile_010": None,
            "quantile_050": None,
            "quantile_090": None,
            "count": 0,
        }
    flat = np.asarray(samples, dtype=float).ravel()
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "quantile_010": float(np.quantile(flat, 0.10)),
        "quantile_050": float(np.quantile(flat, 0.50)),
        "quantile_090": float(np.quantile(flat, 0.90)),
        "count": int(flat.size),
    }


# --------------------------------------------------------------------------- #
# Orchestrator + scorecard
# --------------------------------------------------------------------------- #


def build_dataset_scorecard(  # noqa: PLR0913 - explicit metric families are contract inputs
    *,
    dataset_id: str,
    config: RealismMetricConfig,
    rmse_metrics: Sequence[dict[str, Any]] | None,
    fundamental_diagram: dict[str, Any] | None,
    lane_formation: dict[str, Any] | None,
    reference_source: str,
    notes: Sequence[str] | None = None,
    reconstruction: dict[str, Any] | None = None,
    speed_distribution: dict[str, Any] | None = None,
    proxemic_distribution: dict[str, Any] | None = None,
    interaction_segmentation: InteractionSegmentationResult | Mapping[str, Any] | None = None,
    interaction_minimum_event_counts: Mapping[str, int] | None = None,
) -> RealismScorecard:
    """Aggregate per-metric results into a per-dataset scorecard.

    Args:
        dataset_id: Reference dataset id (e.g. ``"eth-ucy/eth"``).
        config: Metric configuration used.
        rmse_metrics: Per-pair RMSE metric mappings (may be empty/``None``).
        fundamental_diagram: Fundamental-diagram comparison mapping (or ``None``).
        lane_formation: Lane-formation comparison mapping (or ``None``).
        reference_source: Provenance note for the real reference data.
        notes: Caveat notes.
        reconstruction: Content-light reconstruction readiness summary.
        speed_distribution: Speed-distribution distance mapping (or ``None``).
        proxemic_distribution: Proxemic-distribution distance mapping (or ``None``).
        interaction_segmentation: Optional interaction-window result or serialized mapping.
        interaction_minimum_event_counts: Optional contract floors used to mark sparse classes.

    Returns:
        A :class:`RealismScorecard` with aggregated statistics.
    """

    rmse_list = list(rmse_metrics or [])
    ok_rmse = [item for item in rmse_list if item.get("status") == STATUS_OK]
    rmse_values = np.asarray(
        [float(item["rmse_m"]) for item in ok_rmse if "rmse_m" in item],
        dtype=float,
    )
    rmse_summary: dict[str, Any]
    if rmse_values.size:
        rmse_summary = {
            "pair_count": int(rmse_values.size),
            "skipped_pair_count": int(len(rmse_list) - rmse_values.size),
            "rmse_m": {
                "mean": float(np.mean(rmse_values)),
                "std": float(np.std(rmse_values)),
                "min": float(np.min(rmse_values)),
                "max": float(np.max(rmse_values)),
                "median": float(np.median(rmse_values)),
            },
        }
    else:
        rmse_summary = {
            "pair_count": 0,
            "skipped_pair_count": len(rmse_list),
            "status": next(
                (item.get("status") for item in rmse_list if item.get("status")),
                STATUS_EMPTY,
            ),
        }

    status = _derive_scorecard_status(
        rmse=rmse_summary, fundamental=fundamental_diagram, lane=lane_formation
    )
    metrics: dict[str, Any] = {
        "trajectory_rmse": rmse_summary,
        "fundamental_diagram_comparison": fundamental_diagram
        or _empty_metric("fundamental_diagram_comparison"),
        "lane_formation_comparison": lane_formation or _empty_metric("lane_formation_comparison"),
    }
    # Add distribution diagnostics only when both simulation and real arms
    # were supplied. Preserve an explicit ``empty`` result and its reason rather
    # than silently dropping an unavailable diagnostic.
    if speed_distribution is not None:
        metrics["speed_distribution_distance"] = speed_distribution
    if proxemic_distribution is not None:
        metrics["proxemic_distribution_distance"] = proxemic_distribution
    if interaction_segmentation is not None:
        if isinstance(interaction_segmentation, InteractionSegmentationResult):
            interaction_metric = interaction_segmentation.to_dict()
        elif isinstance(interaction_segmentation, Mapping):
            interaction_metric = dict(interaction_segmentation)
        else:
            raise TypeError("interaction_segmentation must be a result or mapping")
        counts = interaction_metric.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("interaction_segmentation must contain a counts mapping")
        if interaction_minimum_event_counts is not None:
            interaction_metric["event_count_status"] = _interaction_event_count_status(
                counts,
                interaction_minimum_event_counts,
            )
        metrics["interaction_conditioned_segmentation"] = interaction_metric
    elif interaction_minimum_event_counts is not None:
        raise ValueError("interaction_minimum_event_counts requires interaction_segmentation")
    return RealismScorecard(
        dataset_id=dataset_id,
        status=status,
        metrics=metrics,
        config=_config_to_dict(config),
        reference_source=reference_source,
        notes=list(notes or []),
        reconstruction=reconstruction,
    )


def run_realism_validation(  # noqa: PLR0913 - metric inputs are explicit for callers
    *,
    dataset_id: str,
    crowds: RealismCrowdInputs | None = None,
    config: RealismMetricConfig | None = None,
    rmse_pairs: Sequence[RealismTrackPair] | None = None,
    reference_source: str = "",
    notes: Sequence[str] | None = None,
    reconstruction: dict[str, Any] | None = None,
    movement_axis: int = 0,
    lateral_axis: int = 1,
    interaction_segmentation: InteractionSegmentationResult | Mapping[str, Any] | None = None,
    interaction_minimum_event_counts: Mapping[str, int] | None = None,
) -> RealismScorecard:
    """Run the realism metrics and build a per-dataset scorecard.

    This is the pure-metric orchestrator. It takes already-collected simulation
    and real arrays/track pairs and computes the three #4975 metrics. The
    caller is responsible for collecting the simulation trace (e.g. from the
    Simulator) and the real reference (e.g. from
    :mod:`robot_sf.data.external.eth_ucy_trajectories`).

    Pass the simulation and real crowd arrays together via ``crowds`` (a
    :class:`RealismCrowdInputs`). If arrays on either side are ``None``, the
    orchestrator omits the corresponding distribution metrics (fail-closed), and
    a partial run still produces a labeled scorecard. A missing real reference is
    never reported as a passing realism result.

    The optional ``reconstruction`` mapping is serialized as a content-light readiness
    summary; it does not change any metric computation or promote benchmark evidence.

    Returns:
        A :class:`RealismScorecard` aggregating all computed metrics.
    """

    cfg = config or RealismMetricConfig()
    pairs = match_tracks(rmse_pairs)
    rmse_metrics = [trajectory_rmse(pair, config=cfg) for pair in pairs]

    fundamental: dict[str, Any] | None = None
    lane: dict[str, Any] | None = None
    speed_dist: dict[str, Any] | None = None
    proxemic_dist: dict[str, Any] | None = None
    if crowds is not None and _crowds_complete(crowds):
        sim_points = speed_density_points(
            crowds.sim_positions, crowds.sim_velocities, neighbor_radius_m=cfg.neighbor_radius_m
        )
        real_points = speed_density_points(
            crowds.real_positions, crowds.real_velocities, neighbor_radius_m=cfg.neighbor_radius_m
        )
        fundamental = fundamental_diagram_comparison(sim_points, real_points)
        lane = lane_formation_comparison(
            crowds.sim_positions,
            crowds.sim_velocities,
            crowds.real_positions,
            crowds.real_velocities,
            config=cfg,
            movement_axis=movement_axis,
            lateral_axis=lateral_axis,
        )
        speed_dist = speed_distribution_distance(crowds.sim_velocities, crowds.real_velocities)
        proxemic_dist = proxemic_distribution_distance(crowds.sim_positions, crowds.real_positions)

    return build_dataset_scorecard(
        dataset_id=dataset_id,
        config=cfg,
        rmse_metrics=rmse_metrics,
        fundamental_diagram=fundamental,
        lane_formation=lane,
        reference_source=reference_source,
        notes=notes,
        reconstruction=reconstruction,
        speed_distribution=speed_dist,
        proxemic_distribution=proxemic_dist,
        interaction_segmentation=interaction_segmentation,
        interaction_minimum_event_counts=interaction_minimum_event_counts,
    )


def build_track_reconstruction_plan(
    track_set: TrackSet | None,
    *,
    dataset_id: str | None = None,
    padding_m: float = 1.0,
    direction_epsilon_m: float = 1e-6,
    scene_geometry: RealismSceneGeometry | None = None,
) -> RealismReconstructionPlan:
    """Build trajectory-derived replay seeds and a conservative flow summary.

    The returned ``pedestrians`` can be passed to the existing
    ``MapDefinition.single_pedestrians`` field by a scenario runner. Without a
    ``scene_geometry`` adapter input, bounds are derived from observed positions and padded
    only to keep a replay container non-degenerate. When supplied, the scene contract's bounds
    and static blocking polygons are used instead, and every observed track sample is checked
    against that geometry. The plan remains partial because the track parser does not provide
    per-waypoint timestamps or a simulator trace.

    Args:
        track_set: Parsed real track set, or ``None`` when the asset is not staged.
        dataset_id: Optional scorecard-facing id. Defaults to ``asset/split``.
        padding_m: Non-negative padding around observed position bounds.
        direction_epsilon_m: Displacement below this value is treated as stationary.
        scene_geometry: Optional trusted scene bounds and static blocking obstacles. This is
            caller-supplied adapter input; trajectory files do not provide it.

    Returns:
        A ``partial`` plan for non-empty tracks or a ``not_available`` plan when
        no parsed tracks are supplied.

    Raises:
        ValueError: If padding, direction tolerance, or track arrays are malformed.
    """

    _require_positive_finite(padding_m, "padding_m")
    _require_non_negative_finite(direction_epsilon_m, "direction_epsilon_m")
    resolved_dataset_id = dataset_id or (
        f"{track_set.asset_id}/{track_set.split}" if track_set is not None else "unknown"
    )
    split = track_set.split if track_set is not None else "unknown"
    geometry_blocker = (
        f"Static scene geometry is not encoded in {_track_set_source_name(track_set)} "
        "trajectory tracks; "
        "trajectory bounds are diagnostic-only and cannot seed a scene-faithful benchmark."
    )
    geometry_status = (
        "scene_geometry_validated" if scene_geometry is not None else "trajectory_bounds_only"
    )
    timing_blocker = (
        "Replay seed inputs do not encode per-waypoint timestamps in the simulator definition; "
        "a simulator trace adapter is required for time-faithful trajectory comparison."
    )
    if track_set is None or not track_set.tracks:
        return _empty_reconstruction_plan(
            dataset_id=resolved_dataset_id,
            split=split,
            geometry_blocker=geometry_blocker,
            timing_blocker=timing_blocker,
            scene_geometry=scene_geometry,
        )

    track_arrays = _validated_track_arrays(track_set)
    replay_start_time_s = min(float(times[0]) for times, _positions in track_arrays)

    all_positions = np.concatenate([positions for _times, positions in track_arrays], axis=0)
    if scene_geometry is None:
        min_xy = np.min(all_positions, axis=0) - float(padding_m)
        max_xy = np.max(all_positions, axis=0) + float(padding_m)
        scene_bounds = (
            (float(min_xy[0]), float(min_xy[1])),
            (float(max_xy[0]), float(max_xy[1])),
        )
        blockers = (geometry_blocker, timing_blocker)
    else:
        _validate_tracks_against_scene_geometry(track_arrays, scene_geometry)
        scene_bounds = scene_geometry.bounds_m
        blockers = (timing_blocker,)
    displacements = np.asarray(
        [positions[-1] - positions[0] for _times, positions in track_arrays], dtype=float
    )
    axis_magnitudes = np.sum(np.abs(displacements), axis=0)
    flow_axis_index: int | None = None
    if float(np.max(axis_magnitudes)) > direction_epsilon_m:
        flow_axis_index = int(np.argmax(axis_magnitudes))
    flow_axis = None if flow_axis_index is None else ("x" if flow_axis_index == 0 else "y")
    pedestrians, direction_counts, entry_exit_flows = _build_reconstruction_pedestrians(
        track_set,
        track_arrays,
        flow_axis_index=flow_axis_index,
        direction_epsilon_m=direction_epsilon_m,
        replay_start_time_s=replay_start_time_s,
    )

    return RealismReconstructionPlan(
        dataset_id=resolved_dataset_id,
        split=track_set.split,
        status="partial",
        geometry_status=geometry_status,
        scene_bounds_m=scene_bounds,
        pedestrians=tuple(pedestrians),
        flow_axis=flow_axis,
        flow_direction_counts=direction_counts,
        total_sample_count=sum(times.shape[0] for times, _positions in track_arrays),
        blockers=blockers,
        entry_exit_flows=tuple(entry_exit_flows),
        timing_status="entry_delay_only",
        scene_geometry=scene_geometry,
    )


def _empty_reconstruction_plan(
    *,
    dataset_id: str,
    split: str,
    geometry_blocker: str,
    timing_blocker: str,
    scene_geometry: RealismSceneGeometry | None,
) -> RealismReconstructionPlan:
    """Build the fail-closed plan used when no parsed tracks are available.

    Returns:
        A ``not_available`` reconstruction plan with actionable blockers.
    """

    blockers = ["No parsed real tracks are available; stage the dataset and rerun the parser."]
    if scene_geometry is None:
        blockers.append(geometry_blocker)
    blockers.append(timing_blocker)
    return RealismReconstructionPlan(
        dataset_id=dataset_id,
        split=split,
        status=STATUS_NOT_AVAILABLE,
        geometry_status=(
            "scene_geometry_validated" if scene_geometry is not None else "unavailable"
        ),
        scene_bounds_m=scene_geometry.bounds_m if scene_geometry is not None else None,
        pedestrians=(),
        flow_axis=None,
        flow_direction_counts={
            "positive": 0,
            "negative": 0,
            "stationary": 0,
            "off_axis": 0,
        },
        total_sample_count=0,
        blockers=tuple(blockers),
        entry_exit_flows=(),
        timing_status="unavailable",
        scene_geometry=scene_geometry,
    )


def _normalize_scene_bounds(
    bounds_m: Any,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Normalize finite ``((min_x, min_y), (max_x, max_y))`` scene bounds.

    Returns:
        Normalized floating-point scene bounds.
    """

    try:
        bounds = np.asarray(bounds_m, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds_m must have shape ((min_x, min_y), (max_x, max_y))") from exc
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("bounds_m must have shape ((min_x, min_y), (max_x, max_y))")
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("bounds_m minimum must be strictly below its maximum")
    return (
        (float(bounds[0, 0]), float(bounds[0, 1])),
        (float(bounds[1, 0]), float(bounds[1, 1])),
    )


def _polygon_signed_area(polygon: np.ndarray) -> float:
    """Return the signed shoelace area of a polygon represented as ``(N, 2)``."""

    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return float(
        0.5
        * np.sum(
            x_coords * np.roll(y_coords, -1) - y_coords * np.roll(x_coords, -1),
            dtype=float,
        )
    )


def _point_on_segment(
    point: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    """Return whether a point lies on a polygon edge within numeric tolerance."""

    point_xy = np.asarray(point, dtype=float)
    start_xy = np.asarray(start, dtype=float)
    end_xy = np.asarray(end, dtype=float)
    edge = end_xy - start_xy
    relative = point_xy - start_xy
    cross = float(edge[0] * relative[1] - edge[1] * relative[0])
    if abs(cross) > 1e-9:
        return False
    return bool(
        np.all(point_xy >= np.minimum(start_xy, end_xy) - 1e-9)
        and np.all(point_xy <= np.maximum(start_xy, end_xy) + 1e-9)
    )


def _point_in_polygon(point: np.ndarray, polygon: tuple[tuple[float, float], ...]) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""

    point_xy = np.asarray(point, dtype=float)
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point_xy, start, end):
            return True
        start_x, start_y = start
        end_x, end_y = end
        if (start_y > point_xy[1]) != (end_y > point_xy[1]):
            intersection_x = start_x + (point_xy[1] - start_y) * (end_x - start_x) / (
                end_y - start_y
            )
            if point_xy[0] < intersection_x:
                inside = not inside
    return inside


def _validate_tracks_against_scene_geometry(
    track_arrays: list[tuple[np.ndarray, np.ndarray]],
    scene_geometry: RealismSceneGeometry,
) -> None:
    """Fail closed when observed track samples violate the supplied scene contract."""

    lower = np.asarray(scene_geometry.bounds_m[0], dtype=float)
    upper = np.asarray(scene_geometry.bounds_m[1], dtype=float)
    for track_index, (_times, positions) in enumerate(track_arrays):
        if np.any(positions < lower) or np.any(positions > upper):
            raise ValueError(
                f"track {track_index} contains samples outside the supplied scene bounds"
            )
        for obstacle in scene_geometry.obstacles:
            if any(_point_in_polygon(point, obstacle.polygon_m) for point in positions):
                raise ValueError(
                    f"track {track_index} intersects static obstacle {obstacle.obstacle_id!r}"
                )


def _validated_track_arrays(
    track_set: TrackSet,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate parsed track arrays before deriving replay inputs.

    Returns:
        Validated ``(time_s, positions)`` arrays in track order.
    """

    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for track in track_set.tracks:
        times = np.asarray(track.time_s, dtype=float).reshape(-1)
        positions = np.asarray(track.positions, dtype=float)
        if times.shape[0] < 2 or positions.shape != (times.shape[0], 2):
            raise ValueError(
                f"track {track.pedestrian_id} must have matching time/position arrays with "
                "at least two samples"
            )
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError(f"track {track.pedestrian_id} contains non-finite values")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError(f"track {track.pedestrian_id} times must be strictly increasing")
        arrays.append((times, positions))
    return arrays


def _build_reconstruction_pedestrians(
    track_set: TrackSet,
    track_arrays: list[tuple[np.ndarray, np.ndarray]],
    *,
    flow_axis_index: int | None,
    direction_epsilon_m: float,
    replay_start_time_s: float,
) -> tuple[
    list[SinglePedestrianDefinition],
    dict[str, int],
    list[RealismEntryExitFlow],
]:
    """Convert tracks to simulator definitions and count inferred directions.

    Returns:
        A triple containing replay definitions, conservative flow-direction counts, and
        observed entry/exit flow records.
    """

    direction_counts = {"positive": 0, "negative": 0, "stationary": 0, "off_axis": 0}
    pedestrians: list[SinglePedestrianDefinition] = []
    entry_exit_flows: list[RealismEntryExitFlow] = []
    for track, (times, positions) in zip(track_set.tracks, track_arrays, strict=True):
        waypoints = _trajectory_waypoints(positions)
        entry_delay_s = float(times[0] - replay_start_time_s)
        flow_direction = _increment_direction_count(
            direction_counts,
            positions[-1] - positions[0],
            flow_axis_index=flow_axis_index,
            direction_epsilon_m=direction_epsilon_m,
        )
        entry_exit_flows.append(
            RealismEntryExitFlow(
                pedestrian_id=int(track.pedestrian_id),
                entry_time_s=float(times[0]),
                exit_time_s=float(times[-1]),
                entry_position=(float(positions[0, 0]), float(positions[0, 1])),
                exit_position=(float(positions[-1, 0]), float(positions[-1, 1])),
                flow_direction=flow_direction,
            )
        )
        pedestrians.append(
            SinglePedestrianDefinition(
                id=f"{track_set.split}_p{track.pedestrian_id}",
                start=(float(positions[0, 0]), float(positions[0, 1])),
                trajectory=waypoints,
                start_delay_s=entry_delay_s,
                metadata={
                    "reconstruction_mode": "trajectory_waypoint_replay",
                    "source_asset_id": track_set.asset_id,
                    "source_split": track_set.split,
                    "source_pedestrian_id": int(track.pedestrian_id),
                    "observed_sample_count": int(times.shape[0]),
                    "observed_duration_s": float(times[-1] - times[0]),
                    "entry_time_s": float(times[0]),
                    "exit_time_s": float(times[-1]),
                    "entry_delay_s": entry_delay_s,
                    "entry_position": [float(value) for value in positions[0]],
                    "exit_position": [float(value) for value in positions[-1]],
                    "flow_direction": flow_direction,
                    "timing_status": "entry_delay_only",
                    "claim_boundary": RECONSTRUCTION_CLAIM_BOUNDARY,
                },
            )
        )
    return pedestrians, direction_counts, entry_exit_flows


def _trajectory_waypoints(positions: np.ndarray) -> list[tuple[float, float]]:
    """Convert positions after the initial sample to de-duplicated waypoints.

    Returns:
        Waypoints after the initial position, with consecutive duplicates removed.
    """

    waypoints: list[tuple[float, float]] = []
    for point in positions[1:]:
        waypoint = (float(point[0]), float(point[1]))
        if not waypoints or waypoint != waypoints[-1]:
            waypoints.append(waypoint)
    return waypoints


def _increment_direction_count(
    counts: dict[str, int],
    displacement: np.ndarray,
    *,
    flow_axis_index: int | None,
    direction_epsilon_m: float,
) -> str:
    """Increment and return one conservative direction category for a displacement.

    Returns:
        The direction bucket used in ``counts``.
    """

    displacement_norm = float(np.linalg.norm(displacement))
    if displacement_norm <= direction_epsilon_m:
        direction = "stationary"
    elif flow_axis_index is None or abs(displacement[flow_axis_index]) <= direction_epsilon_m:
        direction = "off_axis"
    elif displacement[flow_axis_index] > 0.0:
        direction = "positive"
    else:
        direction = "negative"
    counts[direction] += 1
    return direction


def _require_non_negative_finite(value: float, name: str) -> None:
    """Validate a non-negative finite reconstruction parameter."""

    try:
        valid = not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0.0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_positive_finite(value: float, name: str) -> None:
    """Validate a positive finite reconstruction parameter."""

    try:
        valid = not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")


def _crowds_complete(crowds: RealismCrowdInputs) -> bool:
    """Return whether both sim and real crowd arrays are present."""

    return all(
        getattr(crowds, name) is not None
        for name in ("sim_positions", "sim_velocities", "real_positions", "real_velocities")
    )


def _track_set_source_name(track_set: TrackSet | None) -> str:
    """Return a human-readable source family for a parsed track set."""

    if track_set is None:
        return "real reference"
    return "SDD" if track_set.asset_id == "sdd" else "ETH/UCY"


def _track_set_reference_id(track_set: TrackSet) -> str:
    """Return the scorecard reference id without changing metric semantics."""

    if track_set.asset_id == "sdd":
        return f"sdd/{track_set.scene}/{track_set.split}"
    return f"{track_set.asset_id}/{track_set.split}"


def run_realism_validation_from_track_set(  # noqa: PLR0913 - explicit metric and scene inputs
    *,
    dataset_id: str,
    track_set: TrackSet | None,
    sim_positions: np.ndarray | None = None,
    sim_velocities: np.ndarray | None = None,
    rmse_pairs: Sequence[RealismTrackPair] | None = None,
    config: RealismMetricConfig | None = None,
    notes: Sequence[str] | None = None,
    movement_axis: int = 0,
    scene_geometry: RealismSceneGeometry | None = None,
    interaction_config: InteractionSegmentationConfig | None = None,
    interaction_context: RealismInteractionContext | None = None,
    interaction_minimum_event_counts: Mapping[str, int] | None = None,
) -> RealismScorecard:
    """Run realism validation against a parsed real trajectory track set.

    Convenience wrapper that derives the real reference distributions from a
    parsed ETH/UCY or SDD track set and fails closed when the track set is absent
    (``not_available``). When the track set is present, the real positions are
    gridded onto a common time axis and velocities are finite-differenced to
    build the real crowd arrays for the distribution metrics.

    Returns:
        A :class:`RealismScorecard` labeled ``not_available`` when ``track_set``
        is ``None`` or empty, otherwise the full metric scorecard.
    """

    cfg = config or RealismMetricConfig()
    base_notes = list(notes or [])
    resolved_interaction_context = interaction_context
    if resolved_interaction_context is None and scene_geometry is not None:
        resolved_interaction_context = RealismInteractionContext(scene_geometry=scene_geometry)
    interaction_segmentation = segment_interactions(
        track_set,
        config=interaction_config,
        context=resolved_interaction_context,
    )
    reconstruction = build_track_reconstruction_plan(
        track_set,
        dataset_id=dataset_id,
        scene_geometry=scene_geometry,
    )
    if track_set is None or not track_set.tracks:
        return build_dataset_scorecard(
            dataset_id=dataset_id,
            config=cfg,
            rmse_metrics=None,
            fundamental_diagram=None,
            lane_formation=None,
            reference_source=(
                f"{_track_set_source_name(track_set)} split not available; see "
                f"{getattr(track_set, 'docs_path', 'docs/datasets/eth-ucy.md')}"
                if track_set is not None
                else "real reference track set not provided"
            ),
            notes=base_notes
            + [
                "Real reference data not staged; metric values are not available. "
                "This is not success evidence (fail-closed). Stage the dataset per "
                "docs/datasets/eth-ucy.md and re-run."
            ],
            reconstruction=reconstruction.summary_dict(),
            interaction_segmentation=interaction_segmentation,
            interaction_minimum_event_counts=interaction_minimum_event_counts,
        )

    real_positions, real_velocities = _gridded_crowd_from_tracks(track_set, cfg)
    reference_source = (
        f"{_track_set_reference_id(track_set)} ({track_set.format}), "
        f"{len(track_set.tracks)} pedestrians, gridded at {cfg.resample_hz} Hz"
    )
    crowds = RealismCrowdInputs(
        sim_positions=sim_positions,
        sim_velocities=sim_velocities,
        real_positions=real_positions,
        real_velocities=real_velocities,
    )
    return run_realism_validation(
        dataset_id=dataset_id,
        crowds=crowds,
        config=cfg,
        rmse_pairs=rmse_pairs,
        reference_source=reference_source,
        notes=base_notes
        + [
            f"Real reference gridded from {len(track_set.tracks)} parsed pedestrians; "
            "velocities finite-differenced on the common grid."
        ],
        reconstruction=reconstruction.summary_dict(),
        movement_axis=movement_axis,
        lateral_axis=1,
        interaction_segmentation=interaction_segmentation,
        interaction_minimum_event_counts=interaction_minimum_event_counts,
    )


def run_realism_validation_from_staged_dataset(  # noqa: PLR0913 - explicit metric and scene inputs
    *,
    dataset_id: str,
    dataset: RealismStagedDatasetReference,
    sim_positions: np.ndarray | None = None,
    sim_velocities: np.ndarray | None = None,
    rmse_pairs: Sequence[RealismTrackPair] | None = None,
    config: RealismMetricConfig | None = None,
    notes: Sequence[str] | None = None,
    movement_axis: int = 0,
    interaction_config: InteractionSegmentationConfig | None = None,
    interaction_context: RealismInteractionContext | None = None,
    interaction_minimum_event_counts: Mapping[str, int] | None = None,
) -> RealismScorecard:
    """Run the scorecard path only after provenance-gated ETH/UCY loading.

    This is the canonical integration entrypoint for a first staged-data scorecard. A missing
    or incomplete manifest becomes a ``not_available`` scorecard instead of being silently
    interpreted as real-data evidence. The successful path still requires caller-supplied
    simulation arrays and/or matched pairs; the loader does not invent a simulation trace.

    The provenance check validates the compact registry manifest but does not rehash the local
    tree. Re-run the canonical staging/provenance command when the staged bytes change.

    Returns:
        A scorecard with ``not_available`` status when staging/provenance is incomplete, or the
        metric scorecard when the manifest and parsed real reference are available.
    """

    cfg = config or RealismMetricConfig()
    base_notes = list(notes or [])
    try:
        track_set = load_provenance_validated_track_set(
            dataset.split,
            root=dataset.root,
            provenance_manifest=dataset.provenance_manifest,
        )
    except EthUcyDataError as exc:
        reconstruction = build_track_reconstruction_plan(
            None,
            dataset_id=dataset_id,
            scene_geometry=dataset.scene_geometry,
        )
        return build_dataset_scorecard(
            dataset_id=dataset_id,
            config=cfg,
            rmse_metrics=None,
            fundamental_diagram=None,
            lane_formation=None,
            reference_source=f"eth-ucy/{dataset.split} provenance-gated load unavailable",
            notes=base_notes
            + [
                f"Staged ETH/UCY input unavailable: {exc}",
                "This is not success evidence (fail-closed); complete acquisition and provenance "
                "staging before rerunning.",
            ],
            reconstruction=reconstruction.summary_dict(),
            interaction_segmentation=segment_interactions(
                None,
                config=interaction_config,
                context=interaction_context,
            ),
            interaction_minimum_event_counts=interaction_minimum_event_counts,
        )

    reconstruction = build_track_reconstruction_plan(
        track_set,
        dataset_id=dataset_id,
        scene_geometry=dataset.scene_geometry,
    )
    return run_realism_validation_from_track_set(
        dataset_id=dataset_id,
        track_set=track_set,
        sim_positions=sim_positions,
        sim_velocities=sim_velocities,
        rmse_pairs=rmse_pairs,
        config=cfg,
        notes=base_notes
        + [
            "ETH/UCY provenance manifest passed the registry readiness check; the loader did not "
            "rehash the local tree.",
            f"Trajectory replay seed plan status: {reconstruction.status}; "
            f"static geometry status: {reconstruction.geometry_status}.",
            "The replay seed is diagnostic-only until a simulator trace is supplied; the "
            "geometry status above records whether caller-supplied scene semantics passed.",
        ],
        movement_axis=movement_axis,
        scene_geometry=dataset.scene_geometry,
        interaction_config=interaction_config,
        interaction_context=interaction_context,
        interaction_minimum_event_counts=interaction_minimum_event_counts,
    )


def write_realism_scorecard(
    scorecard: RealismScorecard,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the CI-friendly scorecard as JSON + Markdown.

    Returns:
        A mapping of artifact name to written path (``summary_json``,
        ``scorecard_md``).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "scorecard.json"
    scorecard_md = output_dir / "scorecard.md"
    summary_json.write_text(
        json.dumps(scorecard.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scorecard_md.write_text(render_scorecard_markdown(scorecard), encoding="utf-8")
    return {"summary_json": summary_json, "scorecard_md": scorecard_md}


def render_scorecard_markdown(scorecard: RealismScorecard) -> str:
    """Render a compact reviewer-readable scorecard.

    Leads with the claim boundary and the dataset status so a CI reader can
    immediately tell whether the metric values are real-reference results or a
    fail-closed ``not_available`` placeholder.

    Returns:
        A Markdown string summarizing the scorecard.
    """

    sc = scorecard.to_dict()
    lines = [
        f"# Pedestrian Realism Scorecard — {sc['dataset_id']}",
        "",
        f"Claim boundary: {sc['claim_boundary']}.",
        "",
        f"**Status: `{sc['status']}`**  |  evidence status: `{sc['evidence_status']}`  "
        f"|  schema `{sc['schema_version']}`",
        "",
    ]
    if sc["reference_source"]:
        lines += [f"Reference: {sc['reference_source']}", ""]
    reconstruction = sc.get("reconstruction")
    if isinstance(reconstruction, dict):
        lines += [
            "## Reconstruction Readiness",
            "",
            f"- status: `{reconstruction.get('status', 'unavailable')}`",
            f"- geometry: `{reconstruction.get('geometry_status', 'unavailable')}`",
            f"- timing: `{reconstruction.get('timing_status', 'unavailable')}`",
            f"- entry/exit flows: {reconstruction.get('entry_exit_flow_count', 0)}",
            "",
        ]
    interaction = sc["metrics"].get("interaction_conditioned_segmentation")
    if isinstance(interaction, dict):
        lines += [
            "## Interaction-Conditioned Segmentation",
            "",
            f"- status: `{interaction.get('status', 'empty')}`",
            f"- windows: {interaction.get('window_count', 0)}",
            "- labels are primary window classes; sparse classes remain explicit:",
            "",
            "| class | observed windows | minimum | status |",
            "| --- | ---: | ---: | --- |",
        ]
        counts = interaction.get("counts", {})
        floor_status = interaction.get("event_count_status", {})
        floor_rows = floor_status.get("rows", {}) if isinstance(floor_status, dict) else {}
        for label in INTERACTION_CLASSES:
            row = floor_rows.get(label, {})
            observed = row.get("observed", counts.get(label, 0))
            minimum = row.get("minimum", "—")
            row_status = row.get("status", "not_evaluated")
            lines.append(f"| `{label}` | {observed} | {minimum} | `{row_status}` |")
        lines += [
            "",
            f"- claim boundary: {interaction.get('claim_boundary', 'descriptive stratification only')}",
            "",
        ]
    rmse = sc["metrics"].get("trajectory_rmse", {})
    if "rmse_m" in rmse:
        lines += [
            "## Trajectory RMSE",
            "",
            f"- pairs scored: {rmse.get('pair_count', 0)}",
            f"- mean RMSE: {rmse['rmse_m']['mean']:.4f} m",
            f"- median RMSE: {rmse['rmse_m']['median']:.4f} m",
            f"- min/max: {rmse['rmse_m']['min']:.4f} / {rmse['rmse_m']['max']:.4f} m",
            "",
        ]
    else:
        lines += [
            "## Trajectory RMSE",
            "",
            f"- no scored pairs ({rmse.get('status', 'empty')})",
            "",
        ]
    fd = sc["metrics"].get("fundamental_diagram_comparison", {})
    lane = sc["metrics"].get("lane_formation_comparison", {})
    lines += [
        "## Fundamental Diagram Comparison",
        "",
        f"- status: `{fd.get('status', 'empty')}`",
        f"- sim samples: {fd.get('sim_sample_count', 0)} | real samples: {fd.get('real_sample_count', 0)}",
        f"- mean speed delta: {fd.get('mean_speed_delta_mps', 0.0):.4f} m/s",
        f"- speed marginal distance: {fd.get('speed_marginal_distance_mps', 0.0):.4f} m/s",
        "",
        "## Lane-Formation Comparison",
        "",
        f"- status: `{lane.get('status', 'empty')}`",
        f"- sim mean score: {lane.get('sim', {}).get('mean_score', 0.0):.4f} "
        f"({lane.get('sim', {}).get('frame_count', 0)} frames)",
        f"- real mean score: {lane.get('real', {}).get('mean_score', 0.0):.4f} "
        f"({lane.get('real', {}).get('frame_count', 0)} frames)",
        f"- mean score delta: {lane.get('mean_score_delta', 0.0):.4f}",
        "",
    ]
    for metric_id, title in (
        ("speed_distribution_distance", "Speed-Distribution Distance"),
        ("proxemic_distribution_distance", "Proxemic-Distribution Distance"),
    ):
        diagnostic = sc["metrics"].get(metric_id)
        if diagnostic is None:
            continue
        distance = diagnostic.get("distance")
        distance_text = "unavailable" if distance is None else f"{distance:.4f}"
        lines += [
            f"## {title}",
            "",
            f"- status: `{diagnostic.get('status', 'empty')}`",
            f"- distance: {distance_text} {diagnostic.get('units', '')}".rstrip(),
            f"- sim/real samples: {diagnostic.get('sim_sample_count', 0)} / "
            f"{diagnostic.get('real_sample_count', 0)}",
            f"- excluded non-finite observations: {diagnostic.get('excluded_nonfinite_count', 0)}",
            f"- claim boundary: {diagnostic.get('claim_boundary', 'diagnostic-only')}",
            "",
        ]
        empty_reason = diagnostic.get("empty_reason")
        if empty_reason:
            lines.insert(len(lines) - 1, f"- empty reason: {empty_reason}")
    if sc["notes"]:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in sc["notes"]]
        lines += [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _require_finite_float(value: Any, name: str) -> float:
    """Return a finite scalar as ``float`` and reject booleans."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _require_positive_finite_float(value: Any, name: str) -> float:
    """Validate a finite positive scalar.

    Returns:
        The normalized scalar.
    """

    normalized = _require_finite_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_non_negative_finite_float(value: Any, name: str) -> float:
    """Validate a finite non-negative scalar.

    Returns:
        The normalized scalar.
    """

    normalized = _require_finite_float(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _interaction_event_count_status(
    counts: Mapping[str, Any],
    minimum_event_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Mark each interaction class as sufficient or ``insufficient_events``.

    Returns:
        Overall floor status and one observed/minimum row per interaction class.
    """

    if set(minimum_event_counts) != set(INTERACTION_CLASSES):
        raise ValueError("interaction minimum counts must name exactly the interaction classes")
    rows: dict[str, dict[str, int | str]] = {}
    for label in INTERACTION_CLASSES:
        observed = counts.get(label, 0)
        minimum = minimum_event_counts[label]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(f"interaction count for {label!r} must be a non-negative integer")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ValueError(f"interaction minimum for {label!r} must be a non-negative integer")
        rows[label] = {
            "observed": int(observed),
            "minimum": int(minimum),
            "status": "sufficient" if observed >= minimum else "insufficient_events",
        }
    status = (
        "sufficient"
        if all(row["status"] == "sufficient" for row in rows.values())
        else "insufficient_events"
    )
    return {"status": status, "rows": rows}


def _gridded_crowd_from_tracks(
    track_set: TrackSet,
    config: RealismMetricConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid a parsed real track set onto a common time axis for crowd metrics.

    Each pedestrian track is resampled to ``config.resample_hz`` over the global
    time span; missing pedestrians at a frame are filled with NaN and the
    distribution metrics ignore non-finite values.

    Returns:
        A ``(positions, velocities)`` tuple, each shaped ``(T, K, 2)``.
    """

    tracks = list(track_set.tracks)
    if not tracks:
        empty = np.empty((0, 0, 2), dtype=float)
        return empty, empty.copy()
    global_start = min(float(track.time_s[0]) for track in tracks)
    global_end = max(float(track.time_s[-1]) for track in tracks)
    step = 1.0 / config.resample_hz
    n_frames = max(math.floor((global_end - global_start) / step) + 1, 2)
    grid_time = global_start + np.arange(n_frames) * step
    k = len(tracks)
    positions = np.full((n_frames, k, 2), np.nan, dtype=float)
    for col, track in enumerate(tracks):
        t = np.asarray(track.time_s, dtype=float)
        p = np.asarray(track.positions, dtype=float)
        gx = np.interp(grid_time, t, p[:, 0], left=np.nan, right=np.nan)
        gy = np.interp(grid_time, t, p[:, 1], left=np.nan, right=np.nan)
        positions[:, col, 0] = gx
        positions[:, col, 1] = gy
    velocities = _finite_difference(positions, step)
    return positions, velocities


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """First differences along the time axis divided by ``dt``.

    The returned array has the same time length as the input (the last sample is
    repeated) so crowd-metric shapes stay aligned. NaNs propagate.

    Returns:
        The finite-differenced velocity array, time-aligned to the input.
    """

    if values.shape[0] < 2:
        return np.zeros_like(values)
    diff = np.diff(values, axis=0) / dt
    return np.concatenate((diff, diff[-1:]), axis=0)


def _finite_points(points: np.ndarray) -> np.ndarray:
    """Drop non-finite rows from an ``(N, 2)`` point array.

    Returns:
        The subset of rows where both coordinates are finite.
    """

    arr = np.asarray(points, dtype=float).reshape(-1, arr_first_dim(points))
    if arr.shape[0] == 0:
        return arr
    mask = np.all(np.isfinite(arr), axis=1)
    return arr[mask]


def arr_first_dim(points: np.ndarray) -> int:
    """Return the column count of a possibly-ragged point array (defensive)."""

    arr = np.asarray(points, dtype=float)
    return int(arr.shape[1]) if arr.ndim == 2 else 1


def _density_speed_summary(points: np.ndarray) -> dict[str, float]:
    """Return mean density and speed for a finite ``(N, 2)`` point cloud."""

    return {
        "mean_density_ped_per_m2": float(np.mean(points[:, 0])),
        "mean_speed_mps": float(np.mean(points[:, 1])),
        "median_speed_mps": float(np.median(points[:, 1])),
    }


def _sorted_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar distance between two 1-D distributions via sorted-sample matching.

    Uses the mean absolute difference of the two resampled-to-equal-length sorted
    samples (a discrete 1-Wasserstein estimate). This is descriptive only.

    Returns:
        The scalar sorted-sample distance between the two distributions.
    """

    sa = np.sort(a[np.isfinite(a)])
    sb = np.sort(b[np.isfinite(b)])
    if sa.size == 0 or sb.size == 0:
        return 0.0
    n = min(sa.size, sb.size)
    if sa.size != sb.size:
        # Resample both to ``n`` quantiles for a fair scalar comparison.
        qa = np.quantile(sa, np.linspace(0.0, 1.0, n))
        qb = np.quantile(sb, np.linspace(0.0, 1.0, n))
        return float(np.mean(np.abs(qa - qb)))
    return float(np.mean(np.abs(sa - sb)))


def _empty_metric(metric_id: str) -> dict[str, Any]:
    """Return a standardized empty-metric placeholder mapping."""

    return {"metric_id": metric_id, "status": STATUS_EMPTY, "count": 0}


def _entry_exit_time_span(
    flows: Sequence[RealismEntryExitFlow],
) -> dict[str, float] | None:
    """Summarize observed entry/exit times without exporting trajectory coordinates.

    Returns:
        First-entry and last-exit times, or ``None`` when no flow records exist.
    """

    if not flows:
        return None
    return {
        "first_entry_time_s": float(min(flow.entry_time_s for flow in flows)),
        "last_exit_time_s": float(max(flow.exit_time_s for flow in flows)),
    }


def _derive_scorecard_status(
    *,
    rmse: dict[str, Any],
    fundamental: dict[str, Any] | None,
    lane: dict[str, Any] | None,
) -> str:
    """Derive the overall scorecard status from component metric statuses.

    A scorecard is ``not_available`` only when no real-reference metric computed
    any value; otherwise it is ``ok`` even if some components are ``empty``.

    Returns:
        ``"ok"`` when at least one metric computed a value, else
        ``"not_available"``.
    """

    has_value = (
        bool(rmse.get("pair_count"))
        or (fundamental is not None and fundamental.get("status") == STATUS_OK)
        or (lane is not None and lane.get("status") == STATUS_OK)
    )
    if not has_value:
        return STATUS_NOT_AVAILABLE
    return STATUS_OK


def _config_to_dict(config: RealismMetricConfig) -> dict[str, float]:
    """Return a JSON-safe configuration mapping."""

    return {
        "resample_hz": float(config.resample_hz),
        "neighbor_radius_m": float(config.neighbor_radius_m),
        "movement_threshold_mps": float(config.movement_threshold_mps),
        "max_rmse_cap_m": float(config.max_rmse_cap_m),
    }
