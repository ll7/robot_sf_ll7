"""Metric stubs for the Social Navigation Benchmark.

Each metric is exposed as a pure function accepting a structured
`EpisodeData` container. For now these return ``np.nan`` placeholders
with TODO notes referencing the formal definitions in
`docs/dev/issues/social-navigation-benchmark/metrics_spec.md`.

Implementations will be filled in incrementally; keeping the signatures
stable allows other components (runner, aggregation, CI) to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

# --- Constants (draft; duplicated from spec for local use) ---
D_COLL = 0.25  # collision distance threshold (m)
D_NEAR = 0.50  # near-miss lower bound (m)
FORCE_EXCEED_PLACEHOLDER = 5.0  # temporary placeholder; will be calibrated


@dataclass
class EpisodeData:
    """Container for a single episode trajectory.

    Attributes
    ----------
    robot_pos : (T,2) array
    robot_vel : (T,2) array
    robot_acc : (T,2) array
    peds_pos : (T,K,2) array
    ped_forces : (T,K,2) array (social-force magnitudes per ped)
    goal : (2,) array robot target position
    dt : float timestep
    reached_goal_step : int | None (first step index reaching goal)  # optional helper
    """

    robot_pos: np.ndarray
    robot_vel: np.ndarray
    robot_acc: np.ndarray
    peds_pos: np.ndarray
    ped_forces: np.ndarray
    goal: np.ndarray
    dt: float
    reached_goal_step: int | None = None


# --- Metric stub functions ---
def success(_data: EpisodeData) -> float:  # noqa: D401
    """Stub: success (TODO implement)."""
    return float("nan")


def time_to_goal_norm(_data: EpisodeData, _horizon: int) -> float:
    """Stub: steps_to_goal / H if success else 1.0 (TODO)."""
    return float("nan")


def collisions(data: EpisodeData) -> float:
    """Count timesteps where min pedestrian distance < D_COLL.

    If no pedestrians are present (K=0) returns 0.0.
    """
    if data.peds_pos.shape[1] == 0:
        return 0.0
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)  # (T,K)
    min_d = dists.min(axis=1)
    return float(np.count_nonzero(min_d < D_COLL))


def near_misses(data: EpisodeData) -> float:
    """Count timesteps with d_coll <= min distance < d_near.

    If no pedestrians present returns 0.0.
    """
    if data.peds_pos.shape[1] == 0:
        return 0.0
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)
    min_d = dists.min(axis=1)
    mask = (min_d >= D_COLL) & (min_d < D_NEAR)
    return float(np.count_nonzero(mask))


def min_distance(data: EpisodeData) -> float:
    """Return global minimum distance to any pedestrian.

    Returns NaN when there are no pedestrians.
    """
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)
    return float(dists.min())


def path_efficiency(_data: EpisodeData, _shortest_path_len: float) -> float:
    """Stub: shortest / actual path length (clipped) (TODO)."""
    return float("nan")


def force_quantiles(_data: EpisodeData, qs: Iterable[float] = (0.5, 0.9, 0.95)) -> Dict[str, float]:
    """Stub: force magnitude quantiles (TODO)."""
    return {f"force_q{int(q * 100)}": float("nan") for q in qs}


def force_exceed_events(_data: EpisodeData, _threshold: float = FORCE_EXCEED_PLACEHOLDER) -> float:
    """Stub: count of force events above threshold (TODO)."""
    return float("nan")


def comfort_exposure(_data: EpisodeData, _threshold: float = FORCE_EXCEED_PLACEHOLDER) -> float:
    """Stub: normalized force exceed exposure (TODO)."""
    return float("nan")


def jerk_mean(_data: EpisodeData) -> float:
    """Stub: mean jerk magnitude (TODO)."""
    return float("nan")


def energy(_data: EpisodeData) -> float:
    """Stub: sum acceleration magnitudes (TODO)."""
    return float("nan")


def force_gradient_norm_mean(_data: EpisodeData) -> float:
    """Stub: mean gradient norm along path (optional) (TODO)."""
    return float("nan")


def snqi(
    _metric_values: Dict[str, float],
    _weights: Dict[str, float],
    _normalized_inputs: Dict[str, float] | None = None,
) -> float:
    """Stub: composite Social Navigation Quality Index (TODO)."""
    return float("nan")


# --- Orchestrator ---
METRIC_NAMES: List[str] = [
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "path_efficiency",
    "force_q50",
    "force_q90",
    "force_q95",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "energy",
    "force_gradient_norm_mean",
]


def compute_all_metrics(
    data: EpisodeData, *, horizon: int, shortest_path_len: float | None = None
) -> Dict[str, float]:
    """Call each metric stub and return a dict.

    Parameters
    ----------
    data : EpisodeData
        Episode container.
    horizon : int
        Episode horizon (H) used by time-normalized metrics.
    shortest_path_len : float | None
        Pre-computed straight-line / planner shortest path length; if None uses
        Euclidean start->goal distance.
    """
    if shortest_path_len is None:
        shortest_path_len = float(np.linalg.norm(data.robot_pos[0] - data.goal))  # simple fallback

    values: Dict[str, float] = {}
    values["success"] = success(data)
    values["time_to_goal_norm"] = time_to_goal_norm(data, horizon)
    values["collisions"] = collisions(data)
    values["near_misses"] = near_misses(data)
    values["min_distance"] = min_distance(data)
    values["path_efficiency"] = path_efficiency(data, shortest_path_len)
    values.update(force_quantiles(data))
    values["force_exceed_events"] = force_exceed_events(data)
    values["comfort_exposure"] = comfort_exposure(data)
    values["jerk_mean"] = jerk_mean(data)
    values["energy"] = energy(data)
    values["force_gradient_norm_mean"] = force_gradient_norm_mean(data)
    return values


__all__ = [
    "EpisodeData",
    "compute_all_metrics",
    "METRIC_NAMES",
    # individual metrics (exported for future direct testing)
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "path_efficiency",
    "force_quantiles",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "energy",
    "force_gradient_norm_mean",
    "snqi",
]
