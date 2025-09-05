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
    # Optional pre-sampled force field grid: dict with keys X,Y,Fx,Fy (2D arrays)
    force_field_grid: Dict[str, np.ndarray] | None = None


# --- Metric stub functions ---
def success(data: EpisodeData, *, horizon: int) -> float:  # noqa: D401
    """Return 1 if goal reached before horizon with zero collisions else 0.

    Uses `reached_goal_step` if provided; if absent returns 0 (unknown / not reached).
    """
    if data.reached_goal_step is None:
        return 0.0
    if data.reached_goal_step >= horizon:
        return 0.0
    # treat any collision as failure
    if collisions(data) > 0:
        return 0.0
    return 1.0


def time_to_goal_norm(data: EpisodeData, horizon: int) -> float:
    """Normalized time to goal; 1.0 if not successful.

    success definition mirrors `success` metric (no collision + reached early).
    """
    if success(data, horizon=horizon) == 1.0:
        assert data.reached_goal_step is not None
        return float(data.reached_goal_step) / float(horizon)
    return 1.0


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


def path_efficiency(data: EpisodeData, shortest_path_len: float) -> float:
    """Compute shortest_path_len / actual_path_len (clipped to 1).

    Actual path taken: positions up to goal step (inclusive) if reached, else full horizon.
    If actual length is ~0 (stationary) returns 1.0.
    """
    if data.robot_pos.shape[0] < 2:
        return 1.0
    end_idx = (
        data.reached_goal_step
        if data.reached_goal_step is not None
        else data.robot_pos.shape[0] - 1
    )
    end_idx = min(end_idx, data.robot_pos.shape[0] - 1)
    # slice positions including end index
    pos_slice = data.robot_pos[: end_idx + 1]
    diffs = pos_slice[1:] - pos_slice[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    actual = float(seg_lengths.sum())
    if actual <= 1e-9:
        return 1.0
    ratio = shortest_path_len / actual if actual > 0 else 1.0
    if ratio > 1.0:
        ratio = 1.0
    return float(ratio)


def force_quantiles(data: EpisodeData, qs: Iterable[float] = (0.5, 0.9, 0.95)) -> Dict[str, float]:
    """Compute quantiles of pedestrian force magnitudes.

    Returns NaN for each quantile if there are no pedestrians.
    """
    K = data.peds_pos.shape[1]
    if K == 0:
        return {f"force_q{int(q * 100)}": float("nan") for q in qs}
    mags = np.linalg.norm(data.ped_forces, axis=2)  # (T,K)
    flat = mags.ravel()
    return {f"force_q{int(q * 100)}": float(np.quantile(flat, q)) for q in qs}


def force_exceed_events(data: EpisodeData, threshold: float = FORCE_EXCEED_PLACEHOLDER) -> float:
    """Count (t,k) events where |F| > threshold.

    Returns 0 if no pedestrians.
    """
    if data.peds_pos.shape[1] == 0:
        return 0.0
    mags = np.linalg.norm(data.ped_forces, axis=2)
    return float(np.count_nonzero(mags > threshold))


def comfort_exposure(data: EpisodeData, threshold: float = FORCE_EXCEED_PLACEHOLDER) -> float:
    """Normalized exposure to high force events.

    force_exceed_events / (K * T) where K=#peds, T=#timesteps. 0 if K==0 or T==0.
    """
    K = data.peds_pos.shape[1]
    T = data.peds_pos.shape[0]
    if K == 0 or T == 0:
        return 0.0
    events = force_exceed_events(data, threshold=threshold)
    return float(events / (K * T))


def jerk_mean(data: EpisodeData) -> float:
    """Mean jerk magnitude.

    Jerk is difference of consecutive acceleration vectors.
    For T acceleration samples there are T-1 deltas; formula in spec uses (T-2) with a_t defined per step
    difference. We interpret as average norm of first (T-1)-1 jerk vectors: (a_{t+1}-a_t) for t=0..T-2.
    If fewer than 3 timesteps, returns 0.0.
    """
    acc = data.robot_acc
    T = acc.shape[0]
    if T < 3:
        return 0.0
    diffs = acc[1:] - acc[:-1]  # length T-1
    # Use first T-2 differences as per definition (exclude last to align with spec denominator) if T>2
    jerk_vecs = diffs[:-1]
    norms = np.linalg.norm(jerk_vecs, axis=1)
    denom = T - 2
    if denom <= 0:
        return 0.0
    return float(norms.sum() / denom)


def energy(data: EpisodeData) -> float:
    """Sum of acceleration magnitudes over time.

    If no timesteps returns 0.0.
    """
    acc = data.robot_acc
    if acc.size == 0:
        return 0.0
    norms = np.linalg.norm(acc, axis=1)
    return float(norms.sum())


def _bilinear(x: float, y: float, X: np.ndarray, Y: np.ndarray, V: np.ndarray) -> float:
    """Bilinear interpolate V on grid defined by X,Y (both shape (ny,nx))."""
    # Assume rectilinear grid aligned so X,Y vary independently
    xs = X[0]
    ys = Y[:, 0]
    if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
        return float("nan")
    # find indices
    ix = np.searchsorted(xs, x) - 1
    iy = np.searchsorted(ys, y) - 1
    ix = max(0, min(ix, xs.size - 2))
    iy = max(0, min(iy, ys.size - 2))
    x1, x2 = xs[ix], xs[ix + 1]
    y1, y2 = ys[iy], ys[iy + 1]
    if (x2 - x1) == 0 or (y2 - y1) == 0:
        return float("nan")
    q11 = V[iy, ix]
    q21 = V[iy, ix + 1]
    q12 = V[iy + 1, ix]
    q22 = V[iy + 1, ix + 1]
    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)
    return float(
        (1 - tx) * (1 - ty) * q11 + tx * (1 - ty) * q21 + (1 - tx) * ty * q12 + tx * ty * q22
    )


def force_gradient_norm_mean(data: EpisodeData) -> float:
    """Mean gradient norm of force magnitude along robot path.

    Requires `force_field_grid` with keys X,Y,Fx,Fy containing uniform rectilinear grid.
    Returns NaN if grid not present or insufficient points.
    """
    grid = data.force_field_grid
    if grid is None:
        return float("nan")
    try:
        X = grid["X"]
        Y = grid["Y"]
        Fx = grid["Fx"]
        Fy = grid["Fy"]
    except KeyError:
        return float("nan")
    if X.ndim != 2 or Fx.shape != X.shape or Fy.shape != X.shape:
        return float("nan")
    # Compute magnitude and its spatial gradient
    M = np.sqrt(Fx * Fx + Fy * Fy)
    # assume uniform spacing
    if X.shape[1] < 2 or X.shape[0] < 2:
        return float("nan")
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    if dx == 0 or dy == 0:
        return float("nan")
    dMy, dMx = np.gradient(M, dy, dx)
    grad_norm = np.sqrt(dMx * dMx + dMy * dMy)
    samples = []
    for x, y in data.robot_pos:
        val = _bilinear(float(x), float(y), X, Y, grad_norm)
        if np.isfinite(val):
            samples.append(val)
    if not samples:
        return float("nan")
    return float(np.mean(samples))


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
    values["success"] = success(data, horizon=horizon)
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
