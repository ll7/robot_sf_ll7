"""Empirical pedestrian-model validation harness vs public trajectory datasets.

Plain-language summary: this is the trajectory-level *realism* harness requested
by issue #4975. It compares simulated pedestrian trajectories against real
reference tracks (parsed from staged public datasets such as ETH/UCY) using three
metric families, and emits a CI-friendly per-dataset *scorecard* artifact so a
force-model or parameter PR can show its realism delta. It is deliberately
distinct from the issue #3971 ``pedestrian_flow_validation`` harness, which only
runs *synthetic* no-robot fixtures and never compares against real tracks.

The three metric families requested by #4975:

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
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from robot_sf.data.external.eth_ucy_trajectories import EthUcyTrackSet

__all__ = [
    "REALISM_CLAIM_BOUNDARY",
    "REALISM_SCORECARD_SCHEMA_VERSION",
    "RealismCrowdInputs",
    "RealismMetricConfig",
    "RealismScorecard",
    "RealismTrackPair",
    "build_dataset_scorecard",
    "fundamental_diagram_comparison",
    "lane_formation_comparison",
    "lane_formation_score_curve",
    "match_tracks",
    "render_scorecard_markdown",
    "resample_track",
    "run_realism_validation",
    "run_realism_validation_from_track_set",
    "speed_density_points",
    "trajectory_rmse",
    "write_realism_scorecard",
]

REALISM_SCORECARD_SCHEMA_VERSION = "pedestrian_realism_validation.scorecard.v1"
REALISM_CLAIM_BOUNDARY = (
    "trajectory-level empirical realism metrics vs public trajectory datasets; "
    "no calibrated realism threshold, benchmark ranking, or paper-facing claim"
)

#: Status reported when the real reference data is not staged. Per the repository
#: fail-closed contract this is never treated as success evidence.
STATUS_NOT_AVAILABLE = "not_available"
STATUS_OK = "ok"
STATUS_EMPTY = "empty"


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
class RealismCrowdInputs:
    """Matched simulation/real crowd arrays for distribution-metric comparison.

    Bundles the ``(T, K, 2)`` position/velocity arrays the fundamental-diagram
    and lane-formation comparisons need, so the orchestrator signature stays
    small. Either both simulation arrays or both real arrays may be ``None``; the
    corresponding distribution metric then degrades to ``empty`` (fail-closed).

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
    """

    dataset_id: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    reference_source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping representation of the scorecard."""

        return {
            "schema_version": REALISM_SCORECARD_SCHEMA_VERSION,
            "claim_boundary": REALISM_CLAIM_BOUNDARY,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "metrics": self.metrics,
            "config": self.config,
            "reference_source": self.reference_source,
            "notes": list(self.notes),
        }


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
# Orchestrator + scorecard
# --------------------------------------------------------------------------- #


def build_dataset_scorecard(
    *,
    dataset_id: str,
    config: RealismMetricConfig,
    rmse_metrics: Sequence[dict[str, Any]] | None,
    fundamental_diagram: dict[str, Any] | None,
    lane_formation: dict[str, Any] | None,
    reference_source: str,
    notes: Sequence[str] | None = None,
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
    return RealismScorecard(
        dataset_id=dataset_id,
        status=status,
        metrics=metrics,
        config=_config_to_dict(config),
        reference_source=reference_source,
        notes=list(notes or []),
    )


def run_realism_validation(
    *,
    dataset_id: str,
    crowds: RealismCrowdInputs | None = None,
    config: RealismMetricConfig | None = None,
    rmse_pairs: Sequence[RealismTrackPair] | None = None,
    reference_source: str = "",
    notes: Sequence[str] | None = None,
    movement_axis: int = 0,
    lateral_axis: int = 1,
) -> RealismScorecard:
    """Run all three realism metric families and build a per-dataset scorecard.

    This is the pure-metric orchestrator. It takes already-collected simulation
    and real arrays/track pairs and computes the three #4975 metrics. The
    caller is responsible for collecting the simulation trace (e.g. from the
    Simulator) and the real reference (e.g. from
    :mod:`robot_sf.data.external.eth_ucy_trajectories`).

    Pass the simulation and real crowd arrays together via ``crowds`` (a
    :class:`RealismCrowdInputs`); ``None`` arrays on either side make the
    corresponding distribution metric degrade to ``empty`` (fail-closed), so a
    partial run still produces a labeled scorecard. A missing real reference is
    never reported as a passing realism result.

    Returns:
        A :class:`RealismScorecard` aggregating all computed metrics.
    """

    cfg = config or RealismMetricConfig()
    pairs = match_tracks(rmse_pairs)
    rmse_metrics = [trajectory_rmse(pair, config=cfg) for pair in pairs]

    fundamental: dict[str, Any] | None = None
    lane: dict[str, Any] | None = None
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

    return build_dataset_scorecard(
        dataset_id=dataset_id,
        config=cfg,
        rmse_metrics=rmse_metrics,
        fundamental_diagram=fundamental,
        lane_formation=lane,
        reference_source=reference_source,
        notes=notes,
    )


def _crowds_complete(crowds: RealismCrowdInputs) -> bool:
    """Return whether both sim and real crowd arrays are present."""

    return all(
        getattr(crowds, name) is not None
        for name in ("sim_positions", "sim_velocities", "real_positions", "real_velocities")
    )


def run_realism_validation_from_track_set(
    *,
    dataset_id: str,
    track_set: EthUcyTrackSet | None,
    sim_positions: np.ndarray | None = None,
    sim_velocities: np.ndarray | None = None,
    rmse_pairs: Sequence[RealismTrackPair] | None = None,
    config: RealismMetricConfig | None = None,
    notes: Sequence[str] | None = None,
    movement_axis: int = 0,
) -> RealismScorecard:
    """Run realism validation against a parsed real ETH/UCY track set.

    Convenience wrapper that derives the real reference distributions from a
    parsed :class:`EthUcyTrackSet` and fails closed when the track set is absent
    (``not_available``). When the track set is present, the real positions are
    gridded onto a common time axis and velocities are finite-differenced to
    build the real crowd arrays for the distribution metrics.

    Returns:
        A :class:`RealismScorecard` labeled ``not_available`` when ``track_set``
        is ``None`` or empty, otherwise the full metric scorecard.
    """

    cfg = config or RealismMetricConfig()
    base_notes = list(notes or [])
    if track_set is None or not track_set.tracks:
        return build_dataset_scorecard(
            dataset_id=dataset_id,
            config=cfg,
            rmse_metrics=None,
            fundamental_diagram=None,
            lane_formation=None,
            reference_source=(
                f"eth-ucy split not staged; see {getattr(track_set, 'docs_path', 'docs/datasets/eth-ucy.md')}"
                if track_set is not None
                else "real reference track set not provided"
            ),
            notes=base_notes
            + [
                "Real reference data not staged; metric values are not available. "
                "This is not success evidence (fail-closed). Stage the dataset per "
                "docs/datasets/eth-ucy.md and re-run."
            ],
        )

    real_positions, real_velocities = _gridded_crowd_from_tracks(track_set, cfg)
    reference_source = (
        f"eth-ucy/{track_set.split} ({track_set.format}), "
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
        movement_axis=movement_axis,
        lateral_axis=1,
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
        f"**Status: `{sc['status']}`**  |  schema `{sc['schema_version']}`",
        "",
    ]
    if sc["reference_source"]:
        lines += [f"Reference: {sc['reference_source']}", ""]
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
    if sc["notes"]:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in sc["notes"]]
        lines += [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _gridded_crowd_from_tracks(
    track_set: EthUcyTrackSet,
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
