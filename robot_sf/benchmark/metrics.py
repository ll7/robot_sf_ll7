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
    except Exception:
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
    metric_values: Dict[str, float],
    weights: Dict[str, float],
    baseline_stats: Dict[str, Dict[str, float]] | None = None,
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

    succ = metric_values.get("success", 0.0)
    time_norm = metric_values.get("time_to_goal_norm", 1.0)
    coll = _norm("collisions", metric_values.get("collisions", 0.0))
    near = _norm("near_misses", metric_values.get("near_misses", 0.0))
    comfort = metric_values.get("comfort_exposure", 0.0)
    force_ex = _norm("force_exceed_events", metric_values.get("force_exceed_events", 0.0))
    jerk_n = _norm("jerk_mean", metric_values.get("jerk_mean", 0.0))
    curvature_n = _norm("curvature_mean", metric_values.get("curvature_mean", 0.0))

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


# --- Orchestrator ---
METRIC_NAMES: List[str] = [
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "path_efficiency",
    "avg_speed",
    "force_q50",
    "force_q90",
    "force_q95",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "curvature_mean",
    "energy",
    "force_gradient_norm_mean",
]


def compute_all_metrics(
    data: EpisodeData, *, horizon: int, shortest_path_len: float | None = None
) -> Dict[str, float]:
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
    values["curvature_mean"] = curvature_mean(data)
    values["energy"] = energy(data)
    values["avg_speed"] = avg_speed(data)
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
    "curvature_mean",
    "energy",
    "force_gradient_norm_mean",
    "avg_speed",
    "snqi",
]
