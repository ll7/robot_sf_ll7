"""Social navigation benchmark metrics (implemented).

The functions in this module follow the definitions in
`docs/dev/issues/social-navigation-benchmark/metrics_spec.md`. Core,
comfort, smoothness, and paper metrics are implemented; a few optional
metrics intentionally return ``NaN`` when their required inputs are not
provided (for example, ``force_gradient_norm_mean`` needs a sampled force
field grid, and ``time_to_goal`` is undefined when the goal is never
reached).

Implemented categories:
- Core navigation: success, time_to_goal_norm, collisions, near_misses, min_distance,
  mean_distance, path_efficiency, avg_speed.
- Comfort/force: force_quantiles, per_ped_force_quantiles, force_exceed_events,
  comfort_exposure, force_gradient_norm_mean.
- Smoothness/energy: jerk_mean, curvature_mean, energy.
- Paper Table 1 metrics: success_rate, collision_count, wall_collisions,
  agent_collisions, human_collisions, timeout, failure_to_progress,
  stalled_time, time_to_goal, path_length, success_path_length,
  velocity_* / acceleration_* / jerk_* extrema, clearing_distance_*,
  space_compliance, distance_to_human_min, time_to_collision_min,
  aggregated_time.

Missing/optional data handling:
- Empty pedestrian sets (K=0) return 0.0 for collision counts and ``NaN`` for distances where
  undefined.
- Force-based metrics return ``NaN`` when force arrays are missing, all zeros, or contain no
  finite values; comfort metrics fall back to ``NaN`` in the same cases.
- Metrics that require goal attainment (e.g., ``time_to_goal``) return ``NaN`` if the goal was
  not reached; timeout is encoded separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

# Centralized constants imported from benchmark.constants to avoid drift.
from robot_sf.benchmark.constants import (
    COLLISION_DIST as D_COLL,
)
from robot_sf.benchmark.constants import (
    COMFORT_FORCE_THRESHOLD,
)
from robot_sf.benchmark.constants import (
    NEAR_MISS_DIST as D_NEAR,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    obstacles : (M,2) array | None
        Obstacle/wall positions for collision detection (default: None).
        Used by wall_collisions (WC) and clearing_distance (CD) metrics.
    other_agents_pos : (T,J,2) array | None
        Other robot positions for multi-agent scenarios (default: None).
        Used by agent_collisions (AC) metric.
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
    force_field_grid: dict[str, np.ndarray] | None = None
    # Optional fields for paper metrics (2306.16740v4)
    obstacles: np.ndarray | None = None
    other_agents_pos: np.ndarray | None = None


def has_force_data(data: EpisodeData) -> bool:
    """Return True when pedestrian force data is present and non-trivial.

    Force-based metrics are meaningless when the episode contains pedestrians
    but ``ped_forces`` is either missing, all zeros, or entirely non-finite.
    The helper treats the K=0 case as valid (no force data expected).
    """

    if data.peds_pos.shape[1] == 0:
        return True
    if data.ped_forces.shape != data.peds_pos.shape:
        return False
    if not np.isfinite(data.ped_forces).any():
        return False
    return not np.allclose(data.ped_forces, 0.0)


# --- Internal helper functions for paper metrics ---


def _compute_ped_velocities(peds_pos: np.ndarray, dt: float) -> np.ndarray:
    """Compute pedestrian velocities from positions via finite difference.

    Parameters
    ----------
    peds_pos : (T,K,2) array
        Pedestrian positions over time
    dt : float
        Timestep duration

    Returns
    -------
    (T-1,K,2) array
        Pedestrian velocities (first-order finite difference)
        Returns empty array if T < 2
    """
    if peds_pos.shape[0] < 2:
        return np.empty((0, peds_pos.shape[1], 2))
    return np.diff(peds_pos, axis=0) / dt


def _compute_jerk(robot_acc: np.ndarray, dt: float) -> np.ndarray:
    """Compute jerk (acceleration derivative) via finite difference.

    Parameters
    ----------
    robot_acc : (T,2) array
        Robot acceleration over time
    dt : float
        Timestep duration

    Returns
    -------
    (T-1,2) array
        Jerk vectors (first-order finite difference of acceleration)
        Returns empty array if T < 2
    """
    if robot_acc.shape[0] < 2:
        return np.empty((0, 2))
    return np.diff(robot_acc, axis=0) / dt


def _compute_distance_matrix(data: EpisodeData) -> np.ndarray:
    """Compute robot-pedestrian distance matrix.

    Parameters
    ----------
    data : EpisodeData
        Episode data container

    Returns
    -------
    (T,K) array
        Distance from robot to each pedestrian at each timestep
        Returns empty array if K=0
    """
    if data.peds_pos.shape[1] == 0:
        return np.empty((data.robot_pos.shape[0], 0))
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    return np.linalg.norm(diffs, axis=2)


# --- Metric stub functions ---
def success(data: EpisodeData, *, horizon: int) -> float:
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


def mean_distance(data: EpisodeData) -> float:
    """Return mean over time of the minimum robot–pedestrian distance.

    At each timestep t, compute d_t = min_k ||peds_pos[t, k] - robot_pos[t]||.
    Return mean_t d_t. Returns NaN if there are no pedestrians.
    """
    # If no pedestrians (K==0), undefined -> NaN to mirror min_distance behavior
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)  # (T,K)
    min_per_t = dists.min(axis=1)  # (T,)
    return float(np.mean(min_per_t))


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
    ratio = min(ratio, 1.0)
    return float(ratio)


def force_quantiles(data: EpisodeData, qs: Iterable[float] = (0.5, 0.9, 0.95)) -> dict[str, float]:
    """Compute quantiles of pedestrian force magnitudes.

    Returns NaN for each quantile if there are no pedestrians or timesteps.
    """
    K = data.peds_pos.shape[1]
    T = data.peds_pos.shape[0]
    if K == 0 or T == 0:
        return {f"force_q{int(q * 100)}": float("nan") for q in qs}
    if not has_force_data(data):
        return {f"force_q{int(q * 100)}": float("nan") for q in qs}
    mags = np.linalg.norm(data.ped_forces, axis=2)  # (T,K)
    flat = mags.ravel()
    return {f"force_q{int(q * 100)}": float(np.nanquantile(flat, q)) for q in qs}


def per_ped_force_quantiles(
    data: EpisodeData, qs: Iterable[float] = (0.5, 0.9, 0.95)
) -> dict[str, float]:
    """Compute per-pedestrian force quantiles then average across pedestrians.

    For each pedestrian k:
    1. Extract force magnitude time series: M_k = ||F_{k,t}||_2 for all t
    2. Compute quantiles Q_k(q) for each requested quantile q using nanquantile
    3. Average Q_k(q) across all pedestrians k using nanmean

    This differs from force_quantiles() which flattens all (t,k) samples before
    computing quantiles. The per-ped approach preserves individual pedestrian
    experiences before aggregating, revealing whether high forces are concentrated
    on specific individuals or distributed evenly.

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory with ped_forces (T,K,2) array
    qs : Iterable[float], optional
        Quantile levels to compute, by default (0.5, 0.9, 0.95)

    Returns
    -------
    dict[str, float]
        Dictionary with keys ped_force_q{50,90,95} mapping to float values.
        Returns NaN for all keys if K=0 (no pedestrians).
        Returns NaN if all pedestrians have no finite force samples.

    Examples
    --------
    >>> # Episode with 3 peds: ped0=[10,10,10], ped1=[1,1,1], ped2=[1,1,1]
    >>> # Per-ped medians: 10, 1, 1 → mean = 4.0
    >>> # Aggregated median (force_q50): 1.0 (6 samples of 1, 3 samples of 10)
    """
    K = data.peds_pos.shape[1]
    T = data.peds_pos.shape[0]

    # Handle edge cases: no pedestrians or no timesteps
    if K == 0 or T == 0:
        return {f"ped_force_q{int(q * 100)}": float("nan") for q in qs}

    if not has_force_data(data):
        return {f"ped_force_q{int(q * 100)}": float("nan") for q in qs}

    # Compute force magnitudes: (T,K)
    mags = np.linalg.norm(data.ped_forces, axis=2)

    # Compute quantiles per pedestrian: (len(qs), K)
    # Use nanquantile to handle NaN values (missing timesteps)
    per_ped_quantiles = np.nanquantile(mags, q=list(qs), axis=0)

    # Average across pedestrians: (len(qs),)
    # Use nanmean to exclude pedestrians with all-NaN samples
    mean_quantiles = np.nanmean(per_ped_quantiles, axis=1)

    return {f"ped_force_q{int(q * 100)}": float(mean_quantiles[i]) for i, q in enumerate(qs)}


def force_exceed_events(data: EpisodeData, threshold: float = COMFORT_FORCE_THRESHOLD) -> float:
    """Count (t,k) events where |F| > threshold.

    Returns 0 if no pedestrians.
    """
    if data.peds_pos.shape[1] == 0:
        return 0.0
    if not has_force_data(data):
        return float("nan")
    mags = np.linalg.norm(data.ped_forces, axis=2)
    return float(np.count_nonzero(mags > threshold))


def comfort_exposure(data: EpisodeData, threshold: float = COMFORT_FORCE_THRESHOLD) -> float:
    """Normalized exposure to high force events.

    force_exceed_events / (K * T) where K=#peds, T=#timesteps. 0 if K==0 or T==0.
    """
    K = data.peds_pos.shape[1]
    T = data.peds_pos.shape[0]
    if K == 0 or T == 0:
        return 0.0
    if not has_force_data(data):
        return float("nan")
    events = force_exceed_events(data, threshold=threshold)
    return float(events / (K * T))


def jerk_mean(data: EpisodeData) -> float:
    """
    Mean magnitude of jerk (time derivative of acceleration).

    Computes jerk vectors as consecutive differences of acceleration: j_t = a_{t+1} - a_t.
    The function averages the norms of the first T-2 jerk vectors (i.e., uses diffs[:-1]) so the denominator is T-2.
    Returns 0.0 if there are fewer than three acceleration samples.

    Parameters:
        data (EpisodeData): episode container; this function reads data.robot_acc.

    Returns:
        float: mean jerk magnitude (0.0 when insufficient timesteps).
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


def curvature_mean(data: EpisodeData) -> float:
    """Mean path curvature.

    Curvature is computed using the cross product formula: κ = |v × a| / |v|³
    where v is velocity and a is acceleration, both computed from position differences.
    For T position samples there are T-1 velocity samples and T-2 acceleration samples.
    If fewer than 4 timesteps, returns 0.0.
    """
    pos = data.robot_pos
    T = pos.shape[0]
    if T < 4:
        return 0.0

    # Validate dt
    try:
        dt = float(data.dt)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(dt) or dt <= 0.0:
        return 0.0

    # Vectorized finite differences
    vel = np.diff(pos, axis=0) / dt  # (T-1, 2)
    acc = np.diff(vel, axis=0) / dt  # (T-2, 2)

    # Align velocity and acceleration arrays (both length T-2)
    v = vel[1:]
    a = acc

    # Compute cross products and velocity magnitudes
    cross = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0])
    v_mag = np.linalg.norm(v, axis=1)

    # Mask near-zero speeds to avoid division by zero
    eps = 1e-9
    mask = v_mag > eps
    if not np.any(mask):
        return 0.0

    denom = v_mag**3
    kappa = np.zeros_like(v_mag)
    kappa[mask] = cross[mask] / denom[mask]

    # Filter non-finite values before averaging
    finite = np.isfinite(kappa)
    kappa = kappa[finite]
    if kappa.size == 0:
        return 0.0
    return float(np.mean(kappa))


def energy(data: EpisodeData) -> float:
    """Sum of acceleration magnitudes over time.

    If no timesteps returns 0.0.
    """
    acc = data.robot_acc
    if acc.size == 0:
        return 0.0
    norms = np.linalg.norm(acc, axis=1)
    return float(norms.sum())


def avg_speed(data: EpisodeData) -> float:
    """Average robot speed magnitude over the episode timeline.

    Uses all available timesteps up to the recorded trajectory length.
    Returns 0.0 if there are no velocity samples.
    """
    vel = data.robot_vel
    if vel.size == 0:
        return 0.0
    speeds = np.linalg.norm(vel, axis=1)
    if speeds.size == 0:
        return 0.0
    return float(np.mean(speeds))


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
        (1 - tx) * (1 - ty) * q11 + tx * (1 - ty) * q21 + (1 - tx) * ty * q12 + tx * ty * q22,
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
    metric_values: dict[str, float],
    weights: dict[str, float],
    baseline_stats: dict[str, dict[str, float]] | None = None,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Social Navigation Quality Index (SNQI) from raw metric values.

    One-line summary:
        Aggregates multiple navigation metrics into a single scalar quality score using
        weighted positive and penalized components.

    Detailed behavior:
        - Expects metric_values produced by compute_all_metrics; uses the following keys (with defaults):
          "success" (default 0.0), "time_to_goal_norm" (default 1.0), "collisions",
          "near_misses", "comfort_exposure" (default 0.0), "force_exceed_events",
          "jerk_mean", "curvature_mean".
        - Success contributes positively; time-to-goal and other listed metrics are treated as penalties.
        - Penalized metrics (collisions, near_misses, force_exceed_events, jerk_mean, curvature_mean)
          are optionally normalized to [0, 1] using baseline_stats. If baseline_stats is None or a
          metric is missing from baseline_stats, its normalized value defaults to 0.0 (optimistic).
          Normalization for a metric uses median ('med') and 95th percentile ('p95') from baseline_stats;
          normalized = clamp((value - med) / max(p95 - med, eps), 0, 1).
        - Each component is multiplied by its weight (weights dict). Missing weight entries default to 1.0.
        - The final SNQI is: w_success*success - w_time*time_norm - sum(weights * normalized_penalties)
          and is returned as a float.

    Parameters:
        metric_values: dict
            Raw metric values from compute_all_metrics.
        weights: dict
            Per-component weights. Expected keys include (but are not limited to):
            "w_success", "w_time", "w_collisions", "w_near", "w_comfort",
            "w_force_exceed", "w_jerk", "w_curvature". Missing weights default to 1.0.
        baseline_stats: dict | None
            Optional mapping from metric name to {'med': float, 'p95': float} used to normalize
            penalized metrics into [0,1]. If None or a metric entry is missing, that metric's
            normalized contribution is treated as 0.0.
        eps: float
            Small number to avoid division-by-zero when computing normalization denominators.

    Returns:
        float
            The aggregated SNQI score (higher is better).
    """

    def _norm(name: str, value: float) -> float:
        """TODO docstring. Document this function.

        Args:
            name: TODO docstring.
            value: TODO docstring.

        Returns:
            TODO docstring.
        """
        if baseline_stats is None or name not in baseline_stats:
            return 0.0
        med = baseline_stats[name].get("med", 0.0)
        p95 = baseline_stats[name].get("p95", med)
        denom = (p95 - med) if (p95 - med) > eps else 1.0
        norm = (value - med) / denom
        if norm < 0:
            norm = 0.0
        if norm > 1:
            norm = 1.0
        return float(norm)

    def _safe(val: float | None, default: float = 0.0) -> float:
        """TODO docstring. Document this function.

        Args:
            val: TODO docstring.
            default: TODO docstring.

        Returns:
            TODO docstring.
        """
        v = float(val) if val is not None else default
        return v if math.isfinite(v) else default

    succ = _safe(metric_values.get("success", 0.0))
    time_norm = _safe(metric_values.get("time_to_goal_norm", 1.0), default=1.0)
    coll = _norm("collisions", _safe(metric_values.get("collisions", 0.0)))
    near = _norm("near_misses", _safe(metric_values.get("near_misses", 0.0)))
    comfort = _safe(metric_values.get("comfort_exposure", 0.0))
    force_ex = _norm("force_exceed_events", _safe(metric_values.get("force_exceed_events", 0.0)))
    jerk_n = _norm("jerk_mean", _safe(metric_values.get("jerk_mean", 0.0)))
    curvature_n = _norm("curvature_mean", _safe(metric_values.get("curvature_mean", 0.0)))

    w_success = weights.get("w_success", 1.0)
    w_time = weights.get("w_time", 1.0)
    w_collisions = weights.get("w_collisions", 1.0)
    w_near = weights.get("w_near", 1.0)
    w_comfort = weights.get("w_comfort", 1.0)
    w_force_ex = weights.get("w_force_exceed", 1.0)
    w_jerk = weights.get("w_jerk", 1.0)
    w_curvature = weights.get("w_curvature", 1.0)

    score = (
        w_success * succ
        - w_time * time_norm
        - w_collisions * coll
        - w_near * near
        - w_comfort * comfort
        - w_force_ex * force_ex
        - w_jerk * jerk_n
        - w_curvature * curvature_n
    )
    return float(score)


# =============================================================================
# Paper Metrics from 2306.16740v4 (Table 1)
# =============================================================================


# --- NHT (Navigation/Hard Task) Metrics ---


def success_rate(data: EpisodeData, *, horizon: int) -> float:
    """Binary success indicator (1.0 if goal reached before horizon without collisions, else 0.0).

    From paper 2306.16740v4 Table 1: Success (S) metric.
    Note: This returns per-episode success (S). Aggregate as success rate (SR) by averaging.

    Formula: S = 1 if (reached_goal_step < horizon) AND (collisions == 0) else 0

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    horizon : int
        Maximum allowed timesteps

    Returns
    -------
    float
        1.0 if successful, 0.0 otherwise
        Range: [0, 1]
        Units: boolean

    Edge Cases
    -----------
    - If reached_goal_step is None → 0.0 (goal not reached)
    - If reached_goal_step >= horizon → 0.0 (timeout)
    - If any collisions occurred → 0.0 (collision failure)

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Success (S)" metric
    """
    # Success requires reaching the goal before horizon and having no
    # collisions of any type (wall, agent, or human). The older helper
    # `success()` only checked human collisions; use `collision_count`
    # which aggregates wall_collisions + agent_collisions + human_collisions.
    if data.reached_goal_step is None:
        return 0.0
    if data.reached_goal_step >= horizon:
        return 0.0
    # treat any collision (wall, agent, human) as failure
    if collision_count(data) > 0:
        return 0.0
    return 1.0


def collision_count(data: EpisodeData) -> float:
    """Total collision count (sum of wall, agent, human collisions).

    From paper 2306.16740v4 Table 1: Collision (C) metric.

    Formula: C = WC + AC + HC

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Total number of collisions
        Range: [0, ∞)
        Units: collision count

    Edge Cases
    -----------
    - If all collision data unavailable → 0.0
    - Gracefully handles None obstacles/other_agents

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Collision (C)" metric
    """
    wc = wall_collisions(data)
    ac = agent_collisions(data)
    hc = human_collisions(data)
    return wc + ac + hc


def wall_collisions(data: EpisodeData, *, threshold: float = D_COLL) -> float:
    """Count collisions with walls/obstacles.

    From paper 2306.16740v4 Table 1: Wall Collisions (WC) metric.

    Formula: WC = sum_t I(min_m ||robot_pos[t] - obstacles[m]|| < threshold)

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    threshold : float, optional
        Collision distance threshold (default: D_COLL constant)

    Returns
    -------
    float
        Number of timesteps with wall collision
        Range: [0, ∞)
        Units: collision count

    Edge Cases
    -----------
    - If obstacles is None → 0.0
    - If M=0 (no obstacles) → 0.0

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Wall Collisions (WC)" metric
    """
    if data.obstacles is None or data.obstacles.shape[0] == 0:
        return 0.0

    # Vectorized implementation for performance
    # robot_pos: (T, 2), obstacles: (M, 2)
    # Expand dims for broadcasting: (T, 1, 2) and (1, M, 2)
    diffs = data.robot_pos[:, None, :] - data.obstacles[None, :, :]
    # dists will have shape (T, M)
    dists = np.linalg.norm(diffs, axis=2)
    # min_dists_per_t will have shape (T,)
    min_dists_per_t = np.min(dists, axis=1)
    # Count timesteps where the minimum distance is below the threshold
    collision_count_val = np.count_nonzero(min_dists_per_t < threshold)

    return float(collision_count_val)


def agent_collisions(data: EpisodeData, *, threshold: float = D_COLL) -> float:
    """Count collisions with other agents/robots.

    From paper 2306.16740v4 Table 1: Agent Collisions (AC) metric.

    Formula: AC = sum_t I(min_j ||robot_pos[t] - agents[t,j]|| < threshold)

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    threshold : float, optional
        Collision distance threshold (default: D_COLL constant)

    Returns
    -------
    float
        Number of timesteps with agent collision
        Range: [0, ∞)
        Units: collision count

    Edge Cases
    -----------
    - If other_agents_pos is None → 0.0
    - If J=0 (no other agents) → 0.0

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Agent Collisions (AC)" metric
    """
    if data.other_agents_pos is None or data.other_agents_pos.shape[1] == 0:
        return 0.0

    # Vectorized implementation for performance
    # robot_pos: (T, 2), other_agents_pos: (T, J, 2)
    # Expand dims for broadcasting: (T, 1, 2) and (T, J, 2)
    diffs = data.other_agents_pos - data.robot_pos[:, None, :]
    # dists will have shape (T, J)
    dists = np.linalg.norm(diffs, axis=2)
    # min_dists_per_t will have shape (T,)
    min_dists_per_t = np.min(dists, axis=1)
    # Count timesteps where the minimum distance is below the threshold
    collision_count_val = np.count_nonzero(min_dists_per_t < threshold)

    return float(collision_count_val)


def human_collisions(data: EpisodeData, *, threshold: float = D_COLL) -> float:
    """Count collisions with pedestrians/humans.

    From paper 2306.16740v4 Table 1: Human Collisions (HC) metric.

    Formula: HC = sum_t I(min_k ||robot_pos[t] - peds_pos[t,k]|| < threshold)

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    threshold : float, optional
        Collision distance threshold (default: D_COLL constant)

    Returns
    -------
    float
        Number of timesteps with human collision
        Range: [0, ∞)
        Units: collision count

    Edge Cases
    -----------
    - If K=0 (no pedestrians) → 0.0

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Human Collisions (HC)" metric
    """
    if data.peds_pos.shape[1] == 0:
        return 0.0
    dists = _compute_distance_matrix(data)
    min_d = dists.min(axis=1)
    return float(np.count_nonzero(min_d < threshold))


def timeout(data: EpisodeData, *, horizon: int) -> float:
    """Binary indicator for timeout failure.

    From paper 2306.16740v4 Table 1: Timeout Before reaching goal (TO) metric.

    Formula: TO = 1 if reached_goal_step is None else 0

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    horizon : int
        Maximum allowed timesteps (for consistency, not used in formula)

    Returns
    -------
    float
        1.0 if timeout, 0.0 otherwise
        Range: [0, 1]
        Units: boolean (timeout indicator)

    Edge Cases
    -----------
    - If reached_goal_step is None → 1.0 (timeout occurred)
    - If reached_goal_step >= horizon → 1.0 (timeout occurred)

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Timeout Before reaching goal (TO)" metric
    """
    if data.reached_goal_step is None:
        return 1.0
    if data.reached_goal_step >= horizon:
        return 1.0
    return 0.0


def failure_to_progress(
    data: EpisodeData,
    *,
    distance_threshold: float = 0.1,
    time_threshold: float = 5.0,
) -> float:
    """Count failure-to-progress events.

    From paper 2306.16740v4 Table 1: Failure to progress (FP) metric.

    Formula: Count intervals where robot doesn't reduce distance to goal
             by distance_threshold within time_threshold window.

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    distance_threshold : float, optional
        Minimum progress required (meters, default: 0.1)
    time_threshold : float, optional
        Time window to check progress (seconds, default: 5.0)

    Returns
    -------
    float
        Number of failure-to-progress events
        Range: [0, ∞)
        Units: failure count

    Edge Cases
    -----------
    - If T*dt < time_threshold → 0.0 (too short to evaluate)
    - Slides window over trajectory and counts failures

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Failure to progress (FP)" metric
    """
    T = data.robot_pos.shape[0]
    window_steps = int(np.ceil(time_threshold / data.dt))

    if T < window_steps:
        return 0.0

    # Compute distance to goal at each timestep
    dists_to_goal = np.linalg.norm(data.robot_pos - data.goal, axis=1)

    # Vectorized sliding window check
    start_dists = dists_to_goal[: T - window_steps + 1]
    end_dists = dists_to_goal[window_steps - 1 :]
    progress = start_dists - end_dists
    failure_count_val = np.count_nonzero(progress < distance_threshold)

    return float(failure_count_val)


def stalled_time(data: EpisodeData, *, velocity_threshold: float = 0.05) -> float:
    """Total time robot speed is below threshold.

    From paper 2306.16740v4 Table 1: Stalled time (ST) metric.

    Formula: ST = sum_t I(||robot_vel[t]|| < threshold) * dt

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    velocity_threshold : float, optional
        Minimum speed to not be stalled (m/s, default: 0.05)

    Returns
    -------
    float
        Total stalled time duration
        Range: [0, ∞)
        Units: seconds

    Edge Cases
    -----------
    - If never stalled → 0.0

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Stalled time (ST)" metric
    """
    speeds = np.linalg.norm(data.robot_vel, axis=1)
    stalled_count = np.count_nonzero(speeds < velocity_threshold)
    return float(stalled_count * data.dt)


def time_to_goal(data: EpisodeData) -> float:
    """Time from start to goal.

    From paper 2306.16740v4 Table 1: Time to reach goal (T) metric.

    Formula: T = reached_goal_step * dt (if reached, else NaN)

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Time to reach goal in seconds, or NaN if not reached
        Range: [0, ∞) or NaN
        Units: seconds

    Edge Cases
    -----------
    - If reached_goal_step is None → NaN (goal not reached)

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Time to reach goal (T)" metric
    """
    if data.reached_goal_step is None:
        return float("nan")
    return float(data.reached_goal_step * data.dt)


def path_length(data: EpisodeData) -> float:
    """Total path length traveled by robot.

    From paper 2306.16740v4 Table 1: Path length (PL) metric.

    Formula: PL = sum_{t=0}^{T-1} ||robot_pos[t+1] - robot_pos[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Total path length
        Range: [0, ∞)
        Units: meters

    Edge Cases
    -----------
    - If T=1 (single timestep) → 0.0

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Path length (PL)" metric
    """
    if data.robot_pos.shape[0] < 2:
        return 0.0

    diffs = data.robot_pos[1:] - data.robot_pos[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return float(seg_lengths.sum())


def success_path_length(
    data: EpisodeData,
    *,
    horizon: int,
    optimal_length: float,
) -> float:
    """Success weighted by path efficiency.

    From paper 2306.16740v4 Table 1: Success weighted by path length (SPL) metric.

    Formula: SPL = S * (optimal_length / max(actual_length, optimal_length))

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    horizon : int
        Maximum allowed timesteps
    optimal_length : float
        Shortest possible path length (meters)

    Returns
    -------
    float
        Success weighted by path efficiency
        Range: [0, 1]
        Units: unitless ratio

    Edge Cases
    -----------
    - If not successful → 0.0
    - If actual_length < optimal_length → efficiency capped at 1.0

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Success weighted by path length (SPL)" metric
    """
    s = success_rate(data, horizon=horizon)
    if s == 0.0:
        return 0.0

    actual_length = path_length(data)
    efficiency = optimal_length / max(actual_length, optimal_length)
    return float(s * efficiency)


# --- SHT (Social/Human-aware Task) Metrics ---


def velocity_min(data: EpisodeData) -> float:
    """Minimum linear velocity magnitude.

    From paper 2306.16740v4 Table 1: Velocity-based features (V_min).

    Formula: V_min = min_t ||robot_vel[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Minimum velocity magnitude
        Range: [-∞, ∞) (typically [0, ∞) for magnitudes)
        Units: m/s

    Edge Cases
    -----------
    - If T=0 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Velocity-based features (V_min)" metric
    """
    if data.robot_vel.shape[0] == 0:
        return float("nan")
    speeds = np.linalg.norm(data.robot_vel, axis=1)
    return float(np.min(speeds))


def velocity_avg(data: EpisodeData) -> float:
    """Average linear velocity magnitude.

    From paper 2306.16740v4 Table 1: Velocity-based features (V_avg).

    Formula: V_avg = mean_t ||robot_vel[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Average velocity magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s

    Edge Cases
    -----------
    - If T=0 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Velocity-based features (V_avg)" metric
    """
    # Reuse existing avg_speed implementation
    return avg_speed(data)


def velocity_max(data: EpisodeData) -> float:
    """Maximum linear velocity magnitude.

    From paper 2306.16740v4 Table 1: Velocity-based features (V_max).

    Formula: V_max = max_t ||robot_vel[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Maximum velocity magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s

    Edge Cases
    -----------
    - If T=0 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Velocity-based features (V_max)" metric
    """
    if data.robot_vel.shape[0] == 0:
        return float("nan")
    speeds = np.linalg.norm(data.robot_vel, axis=1)
    return float(np.max(speeds))


def acceleration_min(data: EpisodeData) -> float:
    """Minimum linear acceleration magnitude.

    From paper 2306.16740v4 Table 1: Linear acceleration based features (A_min).

    Formula: A_min = min_t ||robot_acc[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Minimum acceleration magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s²

    Edge Cases
    -----------
    - If T=0 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Linear acceleration based features (A_min)" metric
    """
    if data.robot_acc.shape[0] == 0:
        return float("nan")
    acc_mags = np.linalg.norm(data.robot_acc, axis=1)
    return float(np.min(acc_mags))


def acceleration_avg(data: EpisodeData) -> float:
    """Average linear acceleration magnitude.

    From paper 2306.16740v4 Table 1: Linear acceleration based features (A_avg).

    Formula: A_avg = mean_t ||robot_acc[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Average acceleration magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s²

    Edge Cases
    -----------
    - If T=0 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Linear acceleration based features (A_avg)" metric
    """
    if data.robot_acc.shape[0] == 0:
        return float("nan")
    acc_mags = np.linalg.norm(data.robot_acc, axis=1)
    return float(np.mean(acc_mags))


def acceleration_max(data: EpisodeData) -> float:
    """Maximum linear acceleration magnitude.

    From paper 2306.16740v4 Table 1: Linear acceleration based features (A_max).

    Formula: A_max = max_t ||robot_acc[t]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Maximum acceleration magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s²

    Edge Cases
    -----------
    - If T=0 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Linear acceleration based features (A_max)" metric
    """
    if data.robot_acc.shape[0] == 0:
        return float("nan")
    acc_mags = np.linalg.norm(data.robot_acc, axis=1)
    return float(np.max(acc_mags))


def jerk_min(data: EpisodeData) -> float:
    """Minimum jerk magnitude.

    From paper 2306.16740v4 Table 1: Movement jerk (J_min).

    Formula: J_min = min_t ||jerk[t]|| where jerk = d(acc)/dt

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Minimum jerk magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s³

    Edge Cases
    -----------
    - If T < 2 → NaN (insufficient data for jerk computation)

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Movement jerk (J_min)" metric
    """
    jerk_vecs = _compute_jerk(data.robot_acc, data.dt)
    if jerk_vecs.shape[0] == 0:
        return float("nan")
    jerk_mags = np.linalg.norm(jerk_vecs, axis=1)
    return float(np.min(jerk_mags))


def jerk_avg(data: EpisodeData) -> float:
    """Average jerk magnitude.

    From paper 2306.16740v4 Table 1: Movement jerk (J_avg).

    Formula: J_avg = mean_t ||jerk[t]|| where jerk = d(acc)/dt

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Average jerk magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s³

    Edge Cases
    -----------
    - If T < 2 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Movement jerk (J_avg)" metric
    """
    jerk_vecs = _compute_jerk(data.robot_acc, data.dt)
    if jerk_vecs.shape[0] == 0:
        return float("nan")
    jerk_mags = np.linalg.norm(jerk_vecs, axis=1)
    return float(np.mean(jerk_mags))


def jerk_max(data: EpisodeData) -> float:
    """Maximum jerk magnitude.

    From paper 2306.16740v4 Table 1: Movement jerk (J_max).

    Formula: J_max = max_t ||jerk[t]|| where jerk = d(acc)/dt

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Maximum jerk magnitude
        Range: [-∞, ∞) (typically [0, ∞))
        Units: m/s³

    Edge Cases
    -----------
    - If T < 2 → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Movement jerk (J_max)" metric
    """
    jerk_vecs = _compute_jerk(data.robot_acc, data.dt)
    if jerk_vecs.shape[0] == 0:
        return float("nan")
    jerk_mags = np.linalg.norm(jerk_vecs, axis=1)
    return float(np.max(jerk_mags))


def clearing_distance_min(data: EpisodeData) -> float:
    """Minimum distance to obstacles.

    From paper 2306.16740v4 Table 1: Clearing distance (CD_min).

    Formula: CD_min = min_t min_m ||robot_pos[t] - obstacles[m]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Minimum clearing distance
        Range: [0, ∞)
        Units: meters

    Edge Cases
    -----------
    - If obstacles is None → NaN
    - If M=0 (no obstacles) → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Clearing distance (CD_min)" metric
    """
    if data.obstacles is None or data.obstacles.shape[0] == 0:
        return float("nan")

    # Vectorized implementation for performance
    # robot_pos: (T, 2), obstacles: (M, 2)
    # Expand dims for broadcasting: (T, 1, 2) and (1, M, 2)
    diffs = data.robot_pos[:, None, :] - data.obstacles[None, :, :]
    # dists will have shape (T, M)
    dists = np.linalg.norm(diffs, axis=2)
    # min_dists_per_t will have shape (T,)
    min_dists_per_t = np.min(dists, axis=1)
    # Return the minimum across all timesteps
    return float(np.min(min_dists_per_t))


def clearing_distance_avg(data: EpisodeData) -> float:
    """Average minimum distance to obstacles.

    From paper 2306.16740v4 Table 1: Clearing distance (CD_avg).

    Formula: CD_avg = mean_t min_m ||robot_pos[t] - obstacles[m]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Average clearing distance
        Range: [0, ∞)
        Units: meters

    Edge Cases
    -----------
    - If obstacles is None → NaN
    - If M=0 (no obstacles) → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Clearing distance (CD_avg)" metric
    """
    if data.obstacles is None or data.obstacles.shape[0] == 0:
        return float("nan")

    # Vectorized implementation for performance
    # robot_pos: (T, 2), obstacles: (M, 2)
    # Expand dims for broadcasting: (T, 1, 2) and (1, M, 2)
    diffs = data.robot_pos[:, None, :] - data.obstacles[None, :, :]
    # dists will have shape (T, M)
    dists = np.linalg.norm(diffs, axis=2)
    # min_dists_per_t will have shape (T,)
    min_dists_per_t = np.min(dists, axis=1)
    # Return the mean across all timesteps
    return float(np.mean(min_dists_per_t))


def space_compliance(data: EpisodeData, *, threshold: float = 0.5) -> float:
    """Ratio of trajectory within personal space threshold.

    From paper 2306.16740v4 Table 1: Space compliance (SC).

    Formula: SC = (# timesteps where min_k distance < threshold) / T

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    threshold : float, optional
        Personal space threshold (meters, default: 0.5)

    Returns
    -------
    float
        Fraction of trajectory violating personal space
        Range: [0, 1]
        Units: unitless ratio

    Edge Cases
    -----------
    - If K=0 (no pedestrians) → NaN
    - Default threshold: 0.5m (Personal Space Compliance)

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Space compliance (SC)" metric
    """
    if data.peds_pos.shape[1] == 0:
        return float("nan")

    dists = _compute_distance_matrix(data)  # (T, K)
    min_dists = dists.min(axis=1)  # (T,)
    violations = np.count_nonzero(min_dists < threshold)
    T = data.robot_pos.shape[0]

    return float(violations / T)


def distance_to_human_min(data: EpisodeData) -> float:
    """Minimum distance to any human/pedestrian.

    From paper 2306.16740v4 Table 1: Minimum distance to human (DH_min).

    Formula: DH_min = min_t min_k ||robot_pos[t] - peds_pos[t,k]||

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Minimum distance to any pedestrian
        Range: [0, ∞)
        Units: meters

    Edge Cases
    -----------
    - If K=0 (no pedestrians) → NaN

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Minimum distance to human (DH_min)" metric
    """
    # Reuse existing min_distance implementation
    return min_distance(data)


def time_to_collision_min(data: EpisodeData) -> float:
    """Minimum time to collision with pedestrian.

    From paper 2306.16740v4 Table 1: Minimum time to collision (TTC).

    Formula: TTC = min_{t,k} (d / ||v_rel||) for approaching pairs
             where d = distance, v_rel = v_robot - v_ped

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container

    Returns
    -------
    float
        Minimum time to collision
        Range: [0, ∞)
        Units: seconds

    Edge Cases
    -----------
    - If K=0 (no pedestrians) → NaN
    - If no approaching pairs (v_rel points away) → NaN
    - Approaching = relative velocity pointing toward pedestrian

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Minimum time to collision (TTC)" metric
    """
    if data.peds_pos.shape[1] == 0:
        return float("nan")

    ped_vels = _compute_ped_velocities(data.peds_pos, data.dt)
    if ped_vels.shape[0] == 0:
        return float("nan")

    # Align arrays to T-1 length
    robot_vel_aligned = data.robot_vel[1:]
    robot_pos_aligned = data.robot_pos[1:]
    peds_pos_aligned = data.peds_pos[1:]

    # Vectorized calculation
    v_rel = robot_vel_aligned[:, None, :] - ped_vels
    d_vec = peds_pos_aligned - robot_pos_aligned[:, None, :]

    # Correct approaching condition: dot(v_rel, d_vec) > 0 means distance is decreasing.
    dot_product = np.einsum("ijk,ijk->ij", v_rel, d_vec)
    approaching_mask = dot_product > 0

    # Calculate magnitudes for all pairs
    v_rel_mag = np.linalg.norm(v_rel, axis=2)
    d_mag = np.linalg.norm(d_vec, axis=2)

    # Calculate TTC for all pairs, use np.inf for non-approaching or zero-velocity pairs
    ttc_matrix = np.full_like(d_mag, np.inf)

    # Create a mask for valid calculations (approaching and non-zero relative speed)
    valid_calc_mask = approaching_mask & (v_rel_mag > 1e-9)

    # Calculate TTC only for valid pairs
    ttc_matrix[valid_calc_mask] = d_mag[valid_calc_mask] / v_rel_mag[valid_calc_mask]

    min_ttc = np.min(ttc_matrix)

    return float(min_ttc) if np.isfinite(min_ttc) else float("nan")


def aggregated_time(data: EpisodeData, *, cooperative_agents: list[int] | None = None) -> float:
    """Time taken for subset of cooperative agents to meet their goals.

    From paper 2306.16740v4 Table 1: Aggregated Time (AT).

    Formula: AT = time for cooperative_agents to all reach goals

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    cooperative_agents : list[int] | None, optional
        Indices of cooperative agents to track (default: None = single robot)

    Returns
    -------
    float
        Aggregated time for cooperative agents
        Range: [0, ∞)
        Units: seconds

    Edge Cases
    -----------
    - If cooperative_agents is None → use single robot time_to_goal
    - Multi-agent coordination not implemented yet → returns time_to_goal

    Paper Reference
    ---------------
    Section 3.2, Table 1: "Aggregated Time (AT)" metric

    Notes
    -----
    Full multi-agent support requires additional coordination data.
    Current implementation returns single-robot time for backward compatibility.
    """
    # For single-robot scenarios, return time to goal
    # Multi-agent extension would require reached_goal_step per agent
    _ = cooperative_agents  # Acknowledge parameter for linter
    return time_to_goal(data)


# --- Orchestrator ---
METRIC_NAMES: list[str] = [
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "mean_distance",
    "path_efficiency",
    "avg_speed",
    "force_q50",
    "force_q90",
    "force_q95",
    "ped_force_q50",
    "ped_force_q90",
    "ped_force_q95",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "curvature_mean",
    "energy",
    "force_gradient_norm_mean",
]


def compute_all_metrics(
    data: EpisodeData,
    *,
    horizon: int,
    shortest_path_len: float | None = None,
) -> dict[str, float]:
    """
    Compute all defined social-navigation metrics for an episode and return them as a mapping.

    Calls each metric implementation on the provided EpisodeData and collects their scalar results into a dict keyed by metric name.

    Parameters:
        data: EpisodeData
            Episode trajectory and auxiliary data used by the metrics.
        horizon: int
            Episode horizon (number of timesteps) used by time-normalized metrics.
        shortest_path_len: float | None
            Precomputed shortest-path length from start to goal. If None, the Euclidean
            distance from the episode start to the goal is used as a fallback.

    Returns:
        Dict[str, float]: A mapping from metric name to its computed scalar value. Keys include
        "success", "time_to_goal_norm", "collisions", "near_misses", "min_distance",
        "path_efficiency", force quantile keys (e.g. "force_q50"), "force_exceed_events",
        "comfort_exposure", "jerk_mean", "curvature_mean", "energy", "avg_speed",
        and "force_gradient_norm_mean".
    """
    if shortest_path_len is None:
        shortest_path_len = float(np.linalg.norm(data.robot_pos[0] - data.goal))  # simple fallback

    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0
    if ped_count > 0 and not has_force_data(data):
        logger.bind(pedestrians=ped_count, steps=int(data.peds_pos.shape[0])).warning(
            "Missing pedestrian force data; force-based metrics will be NaN.",
        )

    values: dict[str, float] = {}
    values["success"] = success(data, horizon=horizon)
    values["time_to_goal_norm"] = time_to_goal_norm(data, horizon)
    values["collisions"] = collisions(data)
    values["near_misses"] = near_misses(data)
    values["min_distance"] = min_distance(data)
    values["mean_distance"] = mean_distance(data)
    values["path_efficiency"] = path_efficiency(data, shortest_path_len)
    values.update(force_quantiles(data))
    values.update(per_ped_force_quantiles(data))
    values["force_exceed_events"] = force_exceed_events(data)
    values["comfort_exposure"] = comfort_exposure(data)
    values["jerk_mean"] = jerk_mean(data)
    values["curvature_mean"] = curvature_mean(data)
    values["energy"] = energy(data)
    values["avg_speed"] = avg_speed(data)
    values["force_gradient_norm_mean"] = force_gradient_norm_mean(data)
    return values


__all__ = [
    "METRIC_NAMES",
    "EpisodeData",
    "acceleration_avg",
    "acceleration_max",
    "acceleration_min",
    "agent_collisions",
    "aggregated_time",
    "avg_speed",
    "clearing_distance_avg",
    "clearing_distance_min",
    "collision_count",
    "collisions",
    "comfort_exposure",
    "compute_all_metrics",
    "curvature_mean",
    "distance_to_human_min",
    "energy",
    "failure_to_progress",
    "force_exceed_events",
    "force_gradient_norm_mean",
    "force_quantiles",
    "has_force_data",
    "human_collisions",
    "jerk_avg",
    "jerk_max",
    "jerk_mean",
    "jerk_min",
    "mean_distance",
    "min_distance",
    "near_misses",
    "path_efficiency",
    "path_length",
    "snqi",
    "space_compliance",
    "stalled_time",
    "success",
    "success_path_length",
    "success_rate",
    "time_to_collision_min",
    "time_to_goal",
    "time_to_goal_norm",
    "timeout",
    "velocity_avg",
    "velocity_max",
    "velocity_min",
    "wall_collisions",
]
