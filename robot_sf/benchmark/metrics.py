"""Social navigation benchmark metrics (implemented).

The functions in this module follow the definitions in
`docs/dev/issues/social-navigation-benchmark/metrics_spec.md`. Core,
comfort, smoothness, and paper metrics are implemented; a few optional
metrics intentionally return ``NaN`` when their required inputs are not
provided (for example, ``force_gradient_norm_mean`` needs a sampled force
field grid, and ``time_to_goal`` is undefined when the goal is never
reached).

Implemented categories:
- Core navigation: success, time_to_goal_norm, time_to_goal_norm_success_only,
  time_to_goal_ideal_ratio, collisions, near_misses, min_distance,
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
- SocNavBench subset: socnavbench_path_length, socnavbench_path_length_ratio,
  socnavbench_path_irregularity.
- Experimental (optional): pedestrian-impact deltas for acceleration and
  heading turn-rate near-vs-far from the robot (enabled via
  ``compute_all_metrics(..., experimental_ped_impact=True)``), diagnostic
  human-interaction exposure proxies (enabled via
  ``compute_all_metrics(..., experimental_human_interaction_proxy=True)``), and
  opt-in time-to-collision near-miss counts (enabled via
  ``compute_all_metrics(..., experimental_near_miss_ttc=True)``).

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
from robot_sf.benchmark.group_space_metrics import compute_group_space_metrics
from robot_sf.benchmark.signal_metrics import calculate_signal_metrics

if TYPE_CHECKING:
    from collections.abc import Iterable


ROLLOVER_STABILITY_METADATA_KEY = "rollover_stability"
ROLLOVER_CRITICAL_EVENT = "ROLLOVER_CRITICAL"
CLEAR_TRACKING_METADATA_KEY = "clear_tracking_uncertainty"
SOCIAL_GROUPS_METADATA_KEY = "social_groups"


@dataclass
class EpisodeData:
    """Container for a single episode trajectory.

    Attributes
    ----------
    robot_pos : (T,2) array
    robot_vel : (T,2) array
    robot_acc : (T,2) array
    peds_pos : (T,K,2) array
    ped_forces : (T,K,2) array
        Total force vectors applied to each pedestrian. This is the sum of all
        active force components (desired, social, obstacle, group, and optional
        robot interaction when enabled). The data is not decomposed by source.
    goal : (2,) array robot target position
    dt : float timestep
    reached_goal_step : int | None (first step index reaching goal)  # optional helper
    obstacles : (M,2) array | None
        Obstacle/wall positions for collision detection (default: None).
        Used by wall_collisions (WC) and clearing_distance (CD) metrics.
    other_agents_pos : (T,J,2) array | None
        Other robot positions for multi-agent scenarios (default: None).
        Used by agent_collisions (AC) metric.
    robot_radius : float
        Robot footprint radius in meters for clearance-based robot-pedestrian metrics. Runners
        should populate this from the episode configuration; the default is a compatibility
        fallback for synthetic tests and legacy callers.
    ped_radius : float
        Pedestrian footprint radius in meters for clearance-based robot-pedestrian metrics. Runners
        should populate this from the episode/simulation configuration; the default is a
        compatibility fallback for synthetic tests and legacy callers.
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
    robot_radius: float = 1.0
    ped_radius: float = 0.4
    episode_metadata: dict[str, Any] | None = None


def has_force_data(data: EpisodeData) -> bool:
    """Return True when pedestrian force data is present and non-trivial.

    Force-based metrics are meaningless when the episode contains pedestrians
    but ``ped_forces`` is either missing, all zeros, or entirely non-finite.
    The helper treats the K=0 case as valid (no force data expected).

    Returns:
        True if valid force data is available, False otherwise.
    """

    if data.peds_pos.shape[1] == 0:
        return True
    if data.ped_forces.shape != data.peds_pos.shape:
        return False
    if not np.isfinite(data.ped_forces).any():
        return False
    return not np.allclose(data.ped_forces, 0.0)


def force_sample_stats(data: EpisodeData) -> dict[str, int | float | str]:
    """Summarize force sample validity for transparent degenerate-case reporting.

    Returns:
        Dict with status and counts of raw/finite/invalid/zero/nonzero samples.
    """
    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0
    if ped_count == 0:
        return {
            "status": "no-pedestrians",
            "raw_samples": 0,
            "finite_samples": 0,
            "invalid_samples": 0,
            "zero_force_samples": 0,
            "nonzero_force_samples": 0,
            "valid_fraction": 0.0,
        }

    if data.ped_forces.shape != data.peds_pos.shape:
        return {
            "status": "shape-mismatch",
            "raw_samples": 0,
            "finite_samples": 0,
            "invalid_samples": 0,
            "zero_force_samples": 0,
            "nonzero_force_samples": 0,
            "valid_fraction": 0.0,
        }

    magnitudes = np.linalg.norm(data.ped_forces, axis=2)
    raw_samples = int(magnitudes.size)
    if raw_samples == 0:
        return {
            "status": "empty",
            "raw_samples": 0,
            "finite_samples": 0,
            "invalid_samples": 0,
            "zero_force_samples": 0,
            "nonzero_force_samples": 0,
            "valid_fraction": 0.0,
        }

    finite_mask = np.isfinite(magnitudes)
    finite_samples = int(np.count_nonzero(finite_mask))
    invalid_samples = int(raw_samples - finite_samples)
    zero_force_samples = int(np.count_nonzero(np.abs(magnitudes[finite_mask]) <= 1e-12))
    nonzero_force_samples = int(finite_samples - zero_force_samples)

    if finite_samples == 0:
        status = "all-invalid"
    elif nonzero_force_samples == 0:
        status = "all-zero"
    else:
        status = "ok"

    return {
        "status": status,
        "raw_samples": raw_samples,
        "finite_samples": finite_samples,
        "invalid_samples": invalid_samples,
        "zero_force_samples": zero_force_samples,
        "nonzero_force_samples": nonzero_force_samples,
        "valid_fraction": float(finite_samples / raw_samples) if raw_samples > 0 else 0.0,
    }


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


def _compute_clearance_matrix(data: EpisodeData) -> np.ndarray:
    """Compute robot-pedestrian surface clearance matrix.

    Clearance is center-to-center distance minus ``robot_radius + ped_radius``.
    Values below zero indicate geometric overlap/contact.

    Returns:
        (T,K) array with per-timestep robot-pedestrian clearances in meters.
    """
    center_dists = _compute_distance_matrix(data)
    if center_dists.size == 0:
        return center_dists
    return center_dists - float(data.robot_radius + data.ped_radius)


def _compute_robot_ped_distance_summary(data: EpisodeData) -> dict[str, float]:
    """Compute aggregate robot-pedestrian metrics from one distance matrix.

    Returns:
        Mapping of distance-derived aggregate metrics used by `compute_all_metrics`.
    """
    step_count = int(data.robot_pos.shape[0]) if data.robot_pos.ndim >= 2 else 0
    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0
    if step_count == 0 or ped_count == 0:
        return {
            "human_collisions": 0.0,
            "near_misses": 0.0,
            "min_distance": float("nan"),
            "mean_distance": float("nan"),
            "min_clearance": float("nan"),
            "mean_clearance": float("nan"),
            "robot_ped_within_5m_frac": float("nan"),
        }

    dists = _compute_distance_matrix(data)
    min_dists = dists.min(axis=1)
    clearances = dists - float(data.robot_radius + data.ped_radius)
    min_clearances = clearances.min(axis=1)
    return {
        "human_collisions": float(np.count_nonzero(min_clearances < 0.0)),
        "near_misses": float(np.count_nonzero((min_clearances >= 0.0) & (min_clearances < D_NEAR))),
        "min_distance": float(dists.min()),
        "mean_distance": float(np.mean(min_dists)),
        "min_clearance": float(clearances.min()),
        "mean_clearance": float(np.mean(min_clearances)),
        "robot_ped_within_5m_frac": float(np.count_nonzero(min_dists < 5.0) / step_count),
    }


def _compute_near_miss_ttc_metrics(
    data: EpisodeData,
    *,
    t_thr: float | None,
) -> dict[str, Any]:
    """Return opt-in TTC near-miss metrics without changing legacy ``near_misses``.

    Unsupported timing or trajectory inputs fail closed: the count key is omitted and
    status/reason metadata explains why the TTC diagnostic could not be computed.
    """
    from robot_sf.benchmark.near_miss_ttc import (  # noqa: PLC0415
        DIAGNOSTIC_TTC_THRESHOLD_S,
        NearMissTtcInputError,
        compute_ttc_near_miss_diagnostic,
    )

    threshold_s = DIAGNOSTIC_TTC_THRESHOLD_S if t_thr is None else float(t_thr)
    try:
        diagnostic = compute_ttc_near_miss_diagnostic(data, t_thr=threshold_s)
    except NearMissTtcInputError as exc:
        reasons = tuple(exc.readiness.reasons) if exc.readiness is not None else (str(exc),)
        return {
            "near_misses_ttc_status": "unsupported-inputs",
            "near_misses_ttc_threshold_s": threshold_s,
            "near_misses_ttc_unsupported_reasons": list(reasons),
        }

    count = float(diagnostic["near_miss_ttc__count"])
    return {
        **diagnostic,
        "near_misses_ttc": count,
        "near_misses_ttc_status": str(diagnostic["near_miss_ttc__status"]),
        "near_misses_ttc_threshold_s": float(diagnostic["near_miss_ttc__threshold_s"]),
    }


def _rolling_nanmean(samples: np.ndarray, *, window: int) -> np.ndarray:
    """Compute trailing rolling mean with NaN-aware averaging per pedestrian.

    Returns:
        Array with the same shape as ``samples`` containing smoothed values.
    """
    if samples.ndim != 2:
        return np.asarray(samples, dtype=float)
    if samples.size == 0:
        return np.asarray(samples, dtype=float)
    if int(window) <= 1:
        return np.asarray(samples, dtype=float)

    window = int(window)
    n_steps, n_peds = samples.shape
    kernel = np.ones(window, dtype=float)
    smoothed = np.full((n_steps, n_peds), np.nan, dtype=float)
    finite = np.isfinite(samples)
    values = np.where(finite, samples, 0.0)
    for ped_idx in range(n_peds):
        sums = np.convolve(values[:, ped_idx], kernel, mode="full")[:n_steps]
        counts = np.convolve(finite[:, ped_idx].astype(float), kernel, mode="full")[:n_steps]
        valid = counts > 0.0
        smoothed[valid, ped_idx] = sums[valid] / counts[valid]
    return smoothed


def _ped_accel_magnitude(ped_vel: np.ndarray, *, dt: float) -> np.ndarray:
    """Return per-pedestrian acceleration magnitude from velocity samples."""
    if ped_vel.shape[0] < 2:
        return np.empty((0, ped_vel.shape[1]), dtype=float)
    if dt <= 0.0:
        return np.full((ped_vel.shape[0] - 1, ped_vel.shape[1]), np.nan, dtype=float)
    ped_acc = np.diff(ped_vel, axis=0) / dt
    return np.linalg.norm(ped_acc, axis=2)


def _ped_turn_rate_magnitude(
    ped_vel: np.ndarray,
    *,
    dt: float,
    speed_eps: float = 1e-3,
) -> np.ndarray:
    """Return absolute pedestrian heading-rate from velocity samples.

    Low-speed headings are treated as invalid and marked NaN to avoid noisy
    angle flips when velocity is near zero.
    """
    if ped_vel.shape[0] < 2:
        return np.empty((0, ped_vel.shape[1]), dtype=float)
    if dt <= 0.0:
        return np.full((ped_vel.shape[0] - 1, ped_vel.shape[1]), np.nan, dtype=float)

    headings = np.arctan2(ped_vel[..., 1], ped_vel[..., 0])  # (T-1, K)
    heading_delta = np.diff(headings, axis=0)
    # Wrap to [-pi, pi] before dividing by dt.
    heading_delta = (heading_delta + np.pi) % (2.0 * np.pi) - np.pi
    turn_rate = np.abs(heading_delta / dt)

    speed_prev = np.linalg.norm(ped_vel[:-1], axis=2)
    speed_next = np.linalg.norm(ped_vel[1:], axis=2)
    valid = (speed_prev > speed_eps) & (speed_next > speed_eps)
    turn_rate[~valid] = np.nan
    return turn_rate


def _masked_nanmean(samples: np.ndarray, mask: np.ndarray) -> float:
    """Return NaN-aware mean over samples constrained by mask."""
    masked = np.where(mask, samples, np.nan)
    finite = np.isfinite(masked)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(masked))


def _masked_nanmean_axis0(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return per-column NaN-aware mean over samples constrained by mask."""
    masked = np.where(mask, samples, np.nan)
    sums = np.nansum(masked, axis=0)
    counts = np.sum(np.isfinite(masked), axis=0)
    out = np.full(masked.shape[1], np.nan, dtype=float)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def experimental_ped_impact_metrics(
    data: EpisodeData,
    *,
    radius_m: float = 2.0,
    window_steps: int = 5,
) -> dict[str, float]:
    """Compute optional pedestrian-impact metrics (near-vs-far deltas).

    Semantics:
    - Near/far split uses robot-pedestrian distance with threshold ``radius_m``.
    - Signals are smoothed by trailing rolling mean over ``window_steps``.
    - Aggregation is per-pedestrian delta (near_mean - far_mean) first, then
      averaged/median across pedestrians to reduce scenario-density bias.

    Returns:
        Mapping of ``ped_impact_*`` scalar metrics.
    """
    radius = float(radius_m) if math.isfinite(radius_m) and radius_m > 0.0 else 2.0
    try:
        window_candidate = int(window_steps)
    except (TypeError, ValueError, OverflowError):
        window_candidate = 5
    window = window_candidate if window_candidate > 0 else 1
    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0

    metrics: dict[str, float] = {
        "ped_impact_radius_m": radius,
        "ped_impact_window_steps": float(window),
        "ped_impact_ped_count": float(ped_count),
        "ped_impact_near_samples": 0.0,
        "ped_impact_far_samples": 0.0,
        "ped_impact_near_sample_frac": 0.0,
        "ped_impact_accel_near_mean": float("nan"),
        "ped_impact_accel_far_mean": float("nan"),
        "ped_impact_accel_delta_mean": float("nan"),
        "ped_impact_accel_delta_median": float("nan"),
        "ped_impact_accel_delta_valid": 0.0,
        "ped_impact_turn_rate_near_mean": float("nan"),
        "ped_impact_turn_rate_far_mean": float("nan"),
        "ped_impact_turn_rate_delta_mean": float("nan"),
        "ped_impact_turn_rate_delta_median": float("nan"),
        "ped_impact_turn_rate_delta_valid": 0.0,
    }
    if ped_count == 0:
        return metrics

    distances = _compute_distance_matrix(data)
    ped_vel = _compute_ped_velocities(data.peds_pos, data.dt)
    if ped_vel.shape[0] < 2:
        return metrics

    accel_mag = _ped_accel_magnitude(ped_vel, dt=data.dt)
    turn_rate_mag = _ped_turn_rate_magnitude(ped_vel, dt=data.dt)
    if accel_mag.shape[0] == 0 or turn_rate_mag.shape[0] == 0:
        return metrics

    # Align distance at time i+1 with second-difference signals (T-2, K).
    aligned_dist = distances[1:]
    sample_count = min(aligned_dist.shape[0], accel_mag.shape[0], turn_rate_mag.shape[0])
    if sample_count <= 0:
        return metrics
    aligned_dist = aligned_dist[:sample_count]
    accel_mag = accel_mag[:sample_count]
    turn_rate_mag = turn_rate_mag[:sample_count]

    accel_smoothed = _rolling_nanmean(accel_mag, window=window)
    turn_smoothed = _rolling_nanmean(turn_rate_mag, window=window)

    near_mask = np.isfinite(aligned_dist) & (aligned_dist <= radius)
    far_mask = np.isfinite(aligned_dist) & (aligned_dist > radius)
    near_samples = int(np.count_nonzero(near_mask))
    far_samples = int(np.count_nonzero(far_mask))
    total_samples = near_samples + far_samples
    metrics["ped_impact_near_samples"] = float(near_samples)
    metrics["ped_impact_far_samples"] = float(far_samples)
    metrics["ped_impact_near_sample_frac"] = (
        float(near_samples / total_samples) if total_samples > 0 else 0.0
    )

    metrics["ped_impact_accel_near_mean"] = _masked_nanmean(accel_smoothed, near_mask)
    metrics["ped_impact_accel_far_mean"] = _masked_nanmean(accel_smoothed, far_mask)
    accel_near_per_ped = _masked_nanmean_axis0(accel_smoothed, near_mask)
    accel_far_per_ped = _masked_nanmean_axis0(accel_smoothed, far_mask)
    accel_valid = np.isfinite(accel_near_per_ped) & np.isfinite(accel_far_per_ped)
    if accel_valid.any():
        accel_delta = accel_near_per_ped[accel_valid] - accel_far_per_ped[accel_valid]
        metrics["ped_impact_accel_delta_mean"] = float(np.mean(accel_delta))
        metrics["ped_impact_accel_delta_median"] = float(np.median(accel_delta))
        metrics["ped_impact_accel_delta_valid"] = float(np.count_nonzero(accel_valid))

    metrics["ped_impact_turn_rate_near_mean"] = _masked_nanmean(turn_smoothed, near_mask)
    metrics["ped_impact_turn_rate_far_mean"] = _masked_nanmean(turn_smoothed, far_mask)
    turn_near_per_ped = _masked_nanmean_axis0(turn_smoothed, near_mask)
    turn_far_per_ped = _masked_nanmean_axis0(turn_smoothed, far_mask)
    turn_valid = np.isfinite(turn_near_per_ped) & np.isfinite(turn_far_per_ped)
    if turn_valid.any():
        turn_delta = turn_near_per_ped[turn_valid] - turn_far_per_ped[turn_valid]
        metrics["ped_impact_turn_rate_delta_mean"] = float(np.mean(turn_delta))
        metrics["ped_impact_turn_rate_delta_median"] = float(np.median(turn_delta))
        metrics["ped_impact_turn_rate_delta_valid"] = float(np.count_nonzero(turn_valid))

    return metrics


def experimental_social_acceptability_metrics(
    data: EpisodeData,
    *,
    proxemic_radius_m: float = 1.2,
) -> dict[str, float]:
    """Compute optional trajectory-only social-acceptability pilot metrics.

    The pilot summarizes robot-pedestrian personal-space intrusion from surface
    clearance samples. It is a deterministic proxy for review and falsification
    workflows, not a validated human-subject social-compliance score.

    Returns:
        Mapping of ``social_proxemic_*`` scalar metrics.
    """
    radius = (
        float(proxemic_radius_m)
        if math.isfinite(proxemic_radius_m) and proxemic_radius_m > 0.0
        else 1.2
    )
    step_count = int(data.robot_pos.shape[0]) if data.robot_pos.ndim >= 2 else 0
    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0
    metrics: dict[str, float] = {
        "social_proxemic_available": 1.0 if step_count > 0 and ped_count > 0 else 0.0,
        "social_proxemic_radius_m": radius,
        "social_proxemic_ped_count": float(ped_count),
        "social_proxemic_intrusion_steps": 0.0,
        "social_proxemic_intrusion_frac": 0.0,
        "social_proxemic_intrusion_area_m_s": 0.0,
        "social_proxemic_min_clearance_m": float("nan"),
    }
    if step_count <= 0 or ped_count == 0:
        return metrics

    clearances = _compute_clearance_matrix(data)
    if clearances.size == 0:
        return metrics
    finite = np.isfinite(clearances)
    if not finite.any():
        return metrics

    intrusion_depth = np.where(finite, np.maximum(radius - clearances, 0.0), 0.0)
    intrusion_mask = intrusion_depth > 0.0
    intrusion_steps = int(np.count_nonzero(np.any(intrusion_mask, axis=1)))
    metrics["social_proxemic_intrusion_steps"] = float(intrusion_steps)
    metrics["social_proxemic_intrusion_frac"] = (
        float(intrusion_steps / step_count) if step_count > 0 else 0.0
    )
    metrics["social_proxemic_intrusion_area_m_s"] = float(np.sum(intrusion_depth) * data.dt)
    metrics["social_proxemic_min_clearance_m"] = float(np.min(clearances[finite]))
    return metrics


def experimental_human_interaction_proxy_metrics(
    data: EpisodeData,
    *,
    proxemic_radius_m: float = 1.2,
    yield_speed_mps: float = 0.15,
) -> dict[str, float]:
    """Compute simulation-proxy human-interaction exposure metrics.

    These reductions use only robot/pedestrian trajectories and footprint radii.
    They are diagnostic simulation proxies for mechanism reports, not validated
    human comfort, social compliance, or safety metrics.

    Formulas:
    - discomfort exposure: ``sum(max(radius - clearance, 0) * dt)`` in ``m*s``.
    - intrusion duration: count of timesteps with any personal-space intrusion times ``dt``.
    - time to yield: seconds from first intrusion to the first later robot speed below
      ``yield_speed_mps``.
    - robot yield distance: nearest center-distance to a pedestrian at the yield timestep.
    - pedestrian path deviation proxy: mean per-pedestrian extra path length over straight-line
      displacement, in meters.

    Returns:
        Mapping of ``human_proxy_*`` scalar metrics.
    """
    radius = (
        float(proxemic_radius_m)
        if math.isfinite(proxemic_radius_m) and proxemic_radius_m > 0.0
        else 1.2
    )
    yield_speed = (
        float(yield_speed_mps)
        if math.isfinite(yield_speed_mps) and yield_speed_mps >= 0.0
        else 0.15
    )
    dt = float(data.dt) if math.isfinite(float(data.dt)) and float(data.dt) > 0.0 else 1.0
    step_count = int(data.robot_pos.shape[0]) if data.robot_pos.ndim >= 2 else 0
    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0
    metrics: dict[str, float] = {
        "human_proxy_available": 1.0 if step_count > 0 and ped_count > 0 else 0.0,
        "human_proxy_proxemic_radius_m": radius,
        "human_proxy_yield_speed_mps": yield_speed,
        "human_proxy_ped_count": float(ped_count),
        "human_proxy_timestep_count": float(step_count),
        "human_discomfort_exposure_m_s": 0.0,
        "intrusion_duration_s": 0.0,
        "time_to_yield_s": float("nan"),
        "robot_yield_distance_m": float("nan"),
        "pedestrian_path_deviation_proxy_m": float("nan"),
        "group_split_intrusion_available": 0.0,
    }
    if step_count <= 0 or ped_count == 0:
        return metrics

    clearances = _compute_clearance_matrix(data)
    center_dists = _compute_distance_matrix(data)
    finite_clearance = np.isfinite(clearances)
    if finite_clearance.any():
        intrusion_depth = np.where(
            finite_clearance,
            np.maximum(radius - clearances, 0.0),
            0.0,
        )
        intrusion_mask = intrusion_depth > 0.0
        intrusion_by_step = np.any(intrusion_mask, axis=1)
        metrics["human_discomfort_exposure_m_s"] = float(np.sum(intrusion_depth) * dt)
        metrics["intrusion_duration_s"] = float(np.count_nonzero(intrusion_by_step) * dt)

        intrusion_indices = np.flatnonzero(intrusion_by_step)
        if intrusion_indices.size > 0:
            first_intrusion = int(intrusion_indices[0])
            robot_speeds = _robot_speed_samples(data, dt=dt)
            search_speeds = robot_speeds[first_intrusion:]
            yield_offsets = np.flatnonzero(search_speeds <= yield_speed)
            if yield_offsets.size > 0:
                yield_step = first_intrusion + int(yield_offsets[0])
                metrics["time_to_yield_s"] = float((yield_step - first_intrusion) * dt)
                finite_dists = center_dists[yield_step, np.isfinite(center_dists[yield_step])]
                if finite_dists.size > 0:
                    metrics["robot_yield_distance_m"] = float(np.min(finite_dists))

    metrics["pedestrian_path_deviation_proxy_m"] = _pedestrian_path_deviation_proxy(data.peds_pos)
    return metrics


def _robot_speed_samples(data: EpisodeData, *, dt: float) -> np.ndarray:
    """Return one robot speed sample per trajectory timestep."""
    if data.robot_vel.shape == data.robot_pos.shape and np.isfinite(data.robot_vel).any():
        return np.linalg.norm(np.where(np.isfinite(data.robot_vel), data.robot_vel, 0.0), axis=1)
    if data.robot_pos.shape[0] < 2:
        return np.zeros((data.robot_pos.shape[0],), dtype=float)
    step_vel = np.diff(data.robot_pos, axis=0) / dt
    speeds = np.linalg.norm(np.where(np.isfinite(step_vel), step_vel, 0.0), axis=1)
    return np.concatenate(([speeds[0]], speeds))


def _pedestrian_path_deviation_proxy(peds_pos: np.ndarray) -> float:
    """Return mean per-pedestrian extra path length over straight-line displacement."""
    if peds_pos.ndim != 3 or peds_pos.shape[0] < 2 or peds_pos.shape[1] == 0:
        return float("nan")

    deviations: list[float] = []
    for ped_idx in range(peds_pos.shape[1]):
        traj = peds_pos[:, ped_idx, :]
        finite_mask = np.isfinite(traj).all(axis=1)
        finite_traj = traj[finite_mask]
        if finite_traj.shape[0] < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(finite_traj, axis=0), axis=1)
        path_length = float(np.sum(segment_lengths))
        displacement = float(np.linalg.norm(finite_traj[-1] - finite_traj[0]))
        deviations.append(max(path_length - displacement, 0.0))
    if not deviations:
        return float("nan")
    return float(np.mean(deviations))


# --- Metric stub functions ---
def success(data: EpisodeData, *, horizon: int) -> float:
    """Return benchmark success using the same semantics as `success_rate`.

    Success requires route completion before ``horizon`` and zero total collisions
    (`wall_collisions + agent_collisions + human_collisions`). This function is kept as the
    historical core-metric entry point; new benchmark-facing code may call `success_rate` directly.

    Returns:
        1.0 if successful, 0.0 otherwise.
    """
    return success_rate(data, horizon=horizon)


def time_to_goal_norm(data: EpisodeData, horizon: int) -> float:
    """Normalized time to goal; 1.0 if not successful.

    Success definition mirrors benchmark-facing `success_rate` semantics.

    Returns:
        Normalized time to goal (0.0-1.0), or 1.0 if not successful.
    """
    if success_rate(data, horizon=horizon) == 1.0:
        assert data.reached_goal_step is not None
        return float(data.reached_goal_step) / float(horizon)
    return 1.0


def time_to_goal_norm_success_only(data: EpisodeData, horizon: int) -> float:
    """Success-only normalized time-to-goal value.

    Returns:
        Normalized time-to-goal for successful episodes, NaN otherwise.
    """
    if success_rate(data, horizon=horizon) != 1.0:
        return float("nan")
    assert data.reached_goal_step is not None
    return float(data.reached_goal_step) / float(horizon)


def ideal_time_to_goal(
    data: EpisodeData,
    *,
    shortest_path_len: float,
    robot_max_speed: float,
) -> float:
    """Return ideal travel time as shortest-path length divided by max speed.

    Returns:
        Ideal lower-bound travel time in seconds, or NaN if inputs are invalid.
    """
    if not math.isfinite(shortest_path_len) or shortest_path_len < 0.0:
        return float("nan")
    if not math.isfinite(robot_max_speed) or robot_max_speed <= 0.0:
        return float("nan")
    return float(shortest_path_len / robot_max_speed)


def _resolved_robot_max_speed(data: EpisodeData, *, robot_max_speed: float | None) -> float:
    """Resolve the effective robot max speed from override, observations, or fallback.

    Args:
        data: Episode trajectory data, including robot velocities.
        robot_max_speed: Optional caller-provided max speed override.

    Returns:
        float: Positive finite speed. Uses valid ``robot_max_speed`` first, then
        max observed ``||robot_vel||``, and finally falls back to ``1.0``.
    """
    if robot_max_speed is not None and math.isfinite(robot_max_speed) and robot_max_speed > 0.0:
        return float(robot_max_speed)
    if data.robot_vel.size > 0:
        observed = float(np.linalg.norm(data.robot_vel, axis=1).max(initial=0.0))
        if math.isfinite(observed) and observed > 0.0:
            return observed
    return 1.0


def time_to_goal_ideal_ratio(
    data: EpisodeData,
    *,
    horizon: int,
    shortest_path_len: float,
    robot_max_speed: float,
) -> float:
    """Success-only ratio between achieved time and ideal shortest-path time.

    Returns:
        ``time_to_goal / ideal_time`` for successful episodes, NaN otherwise.
    """
    if success_rate(data, horizon=horizon) != 1.0:
        return float("nan")
    actual = time_to_goal(data)
    if not math.isfinite(actual):
        return float("nan")
    ideal = ideal_time_to_goal(
        data,
        shortest_path_len=shortest_path_len,
        robot_max_speed=robot_max_speed,
    )
    if not math.isfinite(ideal) or ideal <= 0.0:
        return float("nan")
    return float(actual / ideal)


def collisions(data: EpisodeData) -> float:
    """Count timesteps with pedestrian footprint overlap.

    This legacy helper is pedestrian-only for compatibility with earlier core metrics. Benchmark
    totals use `collision_count`, which sums wall, agent, and human collisions. The pedestrian
    predicate matches `human_collisions()` with its default radius-aware threshold.

    If no pedestrians are present (K=0) returns 0.0.

    Returns:
        Number of timesteps with robot-pedestrian footprint overlap.
    """
    return human_collisions(data)


def near_misses(data: EpisodeData) -> float:
    """Count timesteps with small positive robot-pedestrian clearance.

    If no pedestrians present returns 0.0.

    Returns:
        Number of near-miss timesteps.
    """
    if data.peds_pos.shape[1] == 0:
        return 0.0
    clearances = _compute_clearance_matrix(data)  # (T,K)
    min_clearance = clearances.min(axis=1)
    mask = (min_clearance >= 0.0) & (min_clearance < D_NEAR)
    return float(np.count_nonzero(mask))


def min_distance(data: EpisodeData) -> float:
    """Return global minimum robot-pedestrian center distance.

    Returns NaN when there are no pedestrians.

    Returns:
        Minimum center-to-center distance in meters, or NaN if no pedestrians.
    """
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    dists = _compute_distance_matrix(data)
    return float(dists.min())


def mean_distance(data: EpisodeData) -> float:
    """Return mean over time of minimum robot-pedestrian center distance.

    At each timestep t, compute distance_t = min_k ||robot - ped_k||.
    Return mean_t d_t. Returns NaN if there are no pedestrians.

    Returns:
        Mean minimum center distance in meters, or NaN if no pedestrians.
    """
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    dists = _compute_distance_matrix(data)  # (T,K)
    min_per_t = dists.min(axis=1)  # (T,)
    return float(np.mean(min_per_t))


def min_clearance(data: EpisodeData) -> float:
    """Return global minimum robot-pedestrian surface clearance."""
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    clearances = _compute_clearance_matrix(data)
    return float(clearances.min())


def mean_clearance(data: EpisodeData) -> float:
    """Return mean over time of minimum robot-pedestrian surface clearance."""
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    clearances = _compute_clearance_matrix(data)
    min_per_t = clearances.min(axis=1)
    return float(np.mean(min_per_t))


def robot_ped_within_distance_frac(data: EpisodeData, *, threshold: float) -> float:
    """Fraction of timesteps where min robot-ped distance is below a threshold.

    Returns:
        float: Fraction of timesteps within the threshold, or NaN if no pedestrians.
    """
    if data.peds_pos.shape[1] == 0:
        return float("nan")
    dists = _compute_distance_matrix(data)
    min_dists = dists.min(axis=1)
    violations = np.count_nonzero(min_dists < threshold)
    return float(violations / data.robot_pos.shape[0])


def robot_ped_within_5m_frac(data: EpisodeData) -> float:
    """Fraction of timesteps where min robot-ped distance is below 5 meters.

    Returns:
        float: Fraction of timesteps within 5 meters, or NaN if no pedestrians.
    """
    return robot_ped_within_distance_frac(data, threshold=5.0)


def ped_force_mean(data: EpisodeData) -> float:
    """Mean magnitude of pedestrian force vectors.

    Returns:
        float: Mean force magnitude, or NaN when force data is unavailable.
    """
    if not has_force_data(data):
        return float("nan")
    mags = np.linalg.norm(data.ped_forces, axis=2)
    if not np.isfinite(mags).any():
        return float("nan")
    return float(np.nanmean(mags))


def path_efficiency(data: EpisodeData, shortest_path_len: float) -> float:
    """Compute shortest_path_len / actual_path_len (clipped to 1).

    Actual path taken: positions up to goal step (inclusive) if reached, else full horizon.
    If actual length is ~0 (stationary) returns 1.0.

    Returns:
        Path efficiency ratio (0.0 to 1.0).
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

    Returns:
        Dictionary mapping quantile labels to force magnitudes.
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
    qs_array = np.asarray(tuple(qs), dtype=float)
    K = data.peds_pos.shape[1]
    T = data.peds_pos.shape[0]

    # Handle edge cases: no pedestrians or no timesteps
    if K == 0 or T == 0:
        return {f"ped_force_q{int(q * 100)}": float("nan") for q in qs_array}

    if not has_force_data(data):
        return {f"ped_force_q{int(q * 100)}": float("nan") for q in qs_array}

    # Compute force magnitudes: (T,K)
    mags = np.hypot(data.ped_forces[..., 0], data.ped_forces[..., 1])

    # Compute quantiles per pedestrian: (len(qs), K)
    # Use nanquantile to handle NaN values (missing timesteps)
    per_ped_quantiles = np.nanquantile(mags, q=qs_array, axis=0)

    # Average across pedestrians: (len(qs),)
    # Use nanmean to exclude pedestrians with all-NaN samples
    mean_quantiles = np.nanmean(per_ped_quantiles, axis=1)

    return {f"ped_force_q{int(q * 100)}": float(mean_quantiles[i]) for i, q in enumerate(qs_array)}


def force_exceed_events(data: EpisodeData, threshold: float = COMFORT_FORCE_THRESHOLD) -> float:
    """Count (t,k) events where |F| > threshold.

    Returns 0 if no pedestrians.

    Returns:
        Number of force threshold exceedance events.
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

    Returns:
        Normalized comfort exposure ratio (0.0 to 1.0).
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
    """Mean magnitude of jerk (time derivative of acceleration).

    Computes jerk vectors as consecutive differences of acceleration: j_t = a_{t+1} - a_t.
    The function averages the norms of the first T-2 jerk vectors (i.e., uses diffs[:-1]) so the denominator is T-2.
    Returns 0.0 if there are fewer than three acceleration samples.

    Args:
        data: Episode container providing ``robot_acc`` samples.

    Returns:
        float: Mean jerk magnitude (0.0 when insufficient timesteps).
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

    Returns:
        Mean path curvature, or 0.0 if insufficient data.
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

    Returns:
        Total energy (sum of acceleration magnitudes).
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

    Returns:
        Mean speed in meters per second.
    """
    vel = data.robot_vel
    if vel.size == 0:
        return 0.0
    speeds = np.linalg.norm(vel, axis=1)
    if speeds.size == 0:
        return 0.0
    return float(np.mean(speeds))


def evaluate_stability_margin(
    v: float,
    yaw_rate: float,
    *,
    t_w: float = 0.8,
    L: float = 1.2,
    h_c: float = 0.6,
    a: float = 0.5,
) -> float:
    """Compute a three-wheeled-vehicle rollover stability margin.

    A value of ``1.0`` indicates no lateral-acceleration load, while ``0.0`` means the
    estimated lateral acceleration is at or beyond the critical rollover threshold. Geometry
    parameters follow the reviewer-supplied TWV proxy: rear track width ``t_w``, wheelbase
    ``L``, center-of-gravity height ``h_c``, and CG distance from the front axle ``a``.

    Returns:
        Stability margin in ``[0.0, 1.0]`` or ``NaN`` when speed/yaw-rate samples are invalid.
    """
    speed = float(v)
    turn_rate = float(yaw_rate)
    for name, value in {"t_w": t_w, "L": L, "h_c": h_c, "a": a}.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(speed) or not math.isfinite(turn_rate):
        return float("nan")
    critical_lateral_accel = 9.81 * (float(t_w) / (2.0 * float(h_c))) * (float(a) / float(L))
    if critical_lateral_accel <= 0.0:
        return float("nan")
    lateral_accel = abs(speed * turn_rate)
    return float(max(0.0, min(1.0, 1.0 - lateral_accel / critical_lateral_accel)))


def _rollover_stability_config(data: EpisodeData) -> dict[str, Any] | None:
    """Return opt-in rollover instrumentation config from episode metadata."""
    metadata = data.episode_metadata
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(ROLLOVER_STABILITY_METADATA_KEY)
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return None
    return dict(raw)


def _metadata_series(value: object, *, length: int) -> np.ndarray | None:
    """Return a finite numeric metadata series broadcast or trimmed to episode length."""
    if value is None:
        return None
    if isinstance(value, str):
        return None
    if isinstance(value, Sequence):
        arr = np.asarray(value, dtype=float).reshape(-1)
    else:
        try:
            arr = np.asarray([float(value)], dtype=float)
        except (TypeError, ValueError):
            return None
    if arr.size == 0:
        return None
    if arr.size == 1:
        arr = np.full(length, float(arr[0]), dtype=float)
    elif arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), mode="edge")
    elif arr.size > length:
        arr = arr[:length]
    return arr.astype(float, copy=False)


def _robot_speed_series(data: EpisodeData) -> np.ndarray:
    """Return per-timestep robot forward-speed magnitudes."""
    if data.robot_vel.size:
        speeds = np.linalg.norm(np.asarray(data.robot_vel, dtype=float), axis=1)
    elif data.robot_pos.shape[0] >= 2 and data.dt > 0:
        step_speeds = np.linalg.norm(np.diff(data.robot_pos, axis=0), axis=1) / float(data.dt)
        speeds = np.pad(step_speeds, (1, 0), mode="edge")
    else:
        speeds = np.zeros(int(data.robot_pos.shape[0]), dtype=float)
    return speeds.astype(float, copy=False)


def _robot_yaw_rate_series(data: EpisodeData, config: Mapping[str, Any]) -> np.ndarray:
    """Return explicit or velocity-heading-derived yaw-rate samples."""
    length = int(data.robot_pos.shape[0])
    for key in ("yaw_rates", "yaw_rate", "robot_yaw_rates", "robot_yaw_rate"):
        series = _metadata_series(config.get(key), length=length)
        if series is not None:
            return series

    velocities = np.asarray(data.robot_vel, dtype=float)
    if velocities.ndim != 2 or velocities.shape[0] < 2 or data.dt <= 0.0:
        return np.zeros(length, dtype=float)

    speeds = np.linalg.norm(velocities, axis=1)
    headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
    deltas = np.diff(headings) / float(data.dt)
    yaw_rates = np.pad(deltas, (1, 0), mode="edge")
    yaw_rates[speeds <= 1e-9] = 0.0
    if yaw_rates.size < length:
        yaw_rates = np.pad(yaw_rates, (0, length - yaw_rates.size), mode="edge")
    return yaw_rates[:length].astype(float, copy=False)


def rollover_stability_metrics(data: EpisodeData) -> dict[str, Any]:
    """Compute optional TWV rollover-stability metrics for one episode.

    The block is disabled unless ``episode_metadata["rollover_stability"]["enabled"]`` is true.
    This keeps existing benchmark evidence unchanged while letting successor campaigns expose
    dynamically unsafe, collision-free maneuvers alongside collision counters.

    Returns:
        Optional flat metrics for campaign rows, or an empty mapping when instrumentation is off.
    """
    config = _rollover_stability_config(data)
    if config is None:
        return {}

    speeds = _robot_speed_series(data)
    yaw_rates = _robot_yaw_rate_series(data, config)
    sample_count = int(min(speeds.size, yaw_rates.size))
    if sample_count == 0:
        return {
            "rollover_stability_enabled": 1.0,
            "rollover_critical": 0.0,
            "rollover_critical_count": 0.0,
            "rollover_critical_fraction": 0.0,
            "rollover_min_stability_margin": float("nan"),
            "rollover_lateral_accel_abs_max": float("nan"),
            "rollover_event": "",
        }

    params = {
        "t_w": float(config.get("t_w", 0.8)),
        "L": float(config.get("L", 1.2)),
        "h_c": float(config.get("h_c", 0.6)),
        "a": float(config.get("a", 0.5)),
    }
    margins = np.asarray(
        [
            evaluate_stability_margin(speed, yaw_rate, **params)
            for speed, yaw_rate in zip(speeds[:sample_count], yaw_rates[:sample_count], strict=True)
        ],
        dtype=float,
    )
    lateral_accel = np.abs(speeds[:sample_count] * yaw_rates[:sample_count])
    finite_margins = margins[np.isfinite(margins)]
    critical_count = int(np.count_nonzero(finite_margins <= 0.0))
    denominator = int(finite_margins.size)
    return {
        "rollover_stability_enabled": 1.0,
        "rollover_critical": 1.0 if critical_count else 0.0,
        "rollover_critical_count": float(critical_count),
        "rollover_critical_fraction": (
            float(critical_count / denominator) if denominator else float("nan")
        ),
        "rollover_min_stability_margin": (
            float(np.min(finite_margins)) if denominator else float("nan")
        ),
        "rollover_lateral_accel_abs_max": float(np.nanmax(lateral_accel)),
        "rollover_event": ROLLOVER_CRITICAL_EVENT if critical_count else "",
    }


def clear_tracking_metrics(data: EpisodeData) -> dict[str, Any]:
    """Return optional CLEAR tracking uncertainty metrics from episode metadata.

    The payload is produced by ScenarioBelief diagnostics and copied into flat
    campaign columns only when ``episode_metadata["clear_tracking_uncertainty"]``
    carries ``enabled: true``. Default benchmark rows are unchanged.

    Returns:
        Flat CLEAR metric columns, or an empty mapping when the diagnostic is off.
    """
    metadata = data.episode_metadata
    if not isinstance(metadata, Mapping):
        return {}
    raw = metadata.get(CLEAR_TRACKING_METADATA_KEY)
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return {}

    def _float_value(key: str, default: float = float("nan")) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    def _count_value(key: str) -> float:
        value = _float_value(key, 0.0)
        return value if math.isfinite(value) and value >= 0.0 else 0.0

    return {
        "clear_tracking_enabled": 1.0,
        "clear_ground_truth_count": _count_value("ground_truth_count"),
        "clear_detection_count": _count_value("detection_count"),
        "clear_missed_detection_count": _count_value("missed_detection_count"),
        "clear_false_positive_count": _count_value("false_positive_count"),
        "clear_id_switch_count": _count_value("id_switch_count"),
        "clear_mota": _float_value("mota"),
        "clear_motp_m": _float_value("motp_m"),
        "clear_motp_match_count": _count_value("motp_match_count"),
    }


def group_space_metrics(data: EpisodeData) -> dict[str, Any]:
    """Return group-space intrusion metrics from declared social groups.

    Group geometry is read from
    ``episode_metadata["social_groups"]["groups"]`` (a list of JSON-safe group
    specs). When absent, this returns an empty mapping so default benchmark rows
    are unchanged.

    Returns:
        Flat group-space metric columns, or an empty mapping when no social
        groups are declared for the episode.
    """
    metadata = data.episode_metadata
    if not isinstance(metadata, Mapping):
        return {}
    payload = metadata.get(SOCIAL_GROUPS_METADATA_KEY)
    if not isinstance(payload, Mapping):
        return {}
    groups = payload.get("groups")
    if not groups:
        return {}
    return compute_group_space_metrics(data.robot_pos, groups)


def _bilinear(x: float, y: float, X: np.ndarray, Y: np.ndarray, V: np.ndarray) -> float:
    """Bilinear interpolate V on grid defined by X,Y (both shape (ny,nx)).

    Returns:
        Interpolated value, or NaN if point is outside grid bounds.
    """
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


def _bilinear_many(
    x: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear interpolation matching `_bilinear` boundary semantics.

    Returns:
        Interpolated values with NaN for samples outside the grid or on degenerate cells.
    """
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    out = np.full(x_arr.shape, np.nan, dtype=float)

    xs = X[0]
    ys = Y[:, 0]
    if xs.size < 2 or ys.size < 2:
        return out

    flat_x = x_arr.ravel()
    flat_y = y_arr.ravel()
    valid = (
        np.isfinite(flat_x)
        & np.isfinite(flat_y)
        & (xs[0] <= flat_x)
        & (flat_x <= xs[-1])
        & (ys[0] <= flat_y)
        & (flat_y <= ys[-1])
    )
    if not np.any(valid):
        return out

    valid_idx = np.flatnonzero(valid)
    x_valid = flat_x[valid_idx]
    y_valid = flat_y[valid_idx]

    ix = np.searchsorted(xs, x_valid) - 1
    iy = np.searchsorted(ys, y_valid) - 1
    ix = np.clip(ix, 0, xs.size - 2)
    iy = np.clip(iy, 0, ys.size - 2)

    x1 = xs[ix]
    x2 = xs[ix + 1]
    y1 = ys[iy]
    y2 = ys[iy + 1]
    dx = x2 - x1
    dy = y2 - y1
    nondegenerate = (dx != 0) & (dy != 0)
    if not np.any(nondegenerate):
        return out

    target_idx = valid_idx[nondegenerate]
    ix = ix[nondegenerate]
    iy = iy[nondegenerate]
    x_valid = x_valid[nondegenerate]
    y_valid = y_valid[nondegenerate]
    x1 = x1[nondegenerate]
    y1 = y1[nondegenerate]
    dx = dx[nondegenerate]
    dy = dy[nondegenerate]

    q11 = V[iy, ix]
    q21 = V[iy, ix + 1]
    q12 = V[iy + 1, ix]
    q22 = V[iy + 1, ix + 1]
    tx = (x_valid - x1) / dx
    ty = (y_valid - y1) / dy
    out.ravel()[target_idx] = (
        (1 - tx) * (1 - ty) * q11 + tx * (1 - ty) * q21 + (1 - tx) * ty * q12 + tx * ty * q22
    )
    return out


def force_gradient_norm_mean(data: EpisodeData) -> float:
    """Mean gradient norm of force magnitude along robot path.

    Requires `force_field_grid` with keys X,Y,Fx,Fy containing uniform rectilinear grid.
    Returns NaN if grid not present or insufficient points.

    Returns:
        Mean force gradient norm, or NaN if grid data unavailable.
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
    samples = _bilinear_many(data.robot_pos[:, 0], data.robot_pos[:, 1], X, Y, grad_norm)
    finite_samples = samples[np.isfinite(samples)]
    if finite_samples.size == 0:
        return float("nan")
    return float(np.mean(finite_samples))


def snqi(
    metric_values: dict[str, float],
    weights: dict[str, float],
    baseline_stats: dict[str, dict[str, float]] | None = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Social Navigation Quality Index (SNQI).

    Aggregates multiple navigation metrics into a single scalar by rewarding success, penalizing
    slow progress, and subtracting normalized comfort/safety penalties.

    Args:
        metric_values: Raw metric values produced by ``compute_all_metrics``. Relevant keys include
            ``success`` (defaults to 0.0), ``time_to_goal_norm`` (defaults to 1.0), ``collisions``,
            ``near_misses``, ``comfort_exposure`` (defaults to 0.0), ``force_exceed_events``,
            ``jerk_mean``, and ``curvature_mean``.
        weights: Per-component weights, e.g., ``w_success``, ``w_time``, ``w_collisions``,
            ``w_near``, ``w_comfort``, ``w_force_exceed``, ``w_jerk``, ``w_curvature``. Missing
            weights default to ``1.0``.
        baseline_stats: Optional normalization statistics mapping metric name to
            ``{"med": float, "p95": float}``. When provided, each penalized metric is normalized to
            ``[0, 1]`` via ``(value - med) / max(p95 - med, eps)`` before weighting. Missing entries
            default to ``0.0`` contribution, effectively optimistic.
        eps: Small epsilon to avoid divide-by-zero when ``p95 == med`` during normalization.

    Returns:
        float: Aggregated SNQI score (higher is better).
    """

    def _norm(name: str, value: float) -> float:
        """Normalize a penalized metric to ``[0, 1]`` based on baseline stats.

        Returns:
            Normalized metric value clipped to [0, 1].
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
        """Return a finite float, falling back to ``default`` when input is None/NaN.

        Returns:
            Validated finite float value or default.
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


def human_collisions(data: EpisodeData, *, threshold: float | None = None) -> float:
    """Count collisions with pedestrians/humans.

    From paper 2306.16740v4 Table 1: Human Collisions (HC) metric.

    Default formula:
        HC = sum_t I(min_k clearance(robot, pedestrian_k) < 0)

    where clearance is center distance minus ``robot_radius + ped_radius``. Passing
    ``threshold`` opts into the legacy center-distance predicate:
        HC = sum_t I(min_k ||robot_pos[t] - peds_pos[t,k]|| < threshold)

    Parameters
    ----------
    data : EpisodeData
        Episode trajectory container
    threshold : float | None, optional
        Optional center-distance threshold. When omitted, the episode robot/pedestrian radii define
        geometric contact.

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
    clearances = _compute_clearance_matrix(data)
    min_clearance = clearances.min(axis=1)
    if threshold is None:
        clearance_threshold = 0.0
    else:
        clearance_threshold = float(threshold) - float(data.robot_radius + data.ped_radius)
    return float(np.count_nonzero(min_clearance < clearance_threshold))


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
    if not math.isfinite(data.dt) or data.dt <= 0.0:
        return float("nan")
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
    if not math.isfinite(data.dt) or data.dt < 0.0:
        return float("nan")
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


def socnavbench_path_length(data: EpisodeData) -> float:
    """SocNavBench path length metric (distance traveled).

    This mirrors ``third_party/socnavbench/metrics/cost_functions.py::path_length``.

    Returns:
        float: Total path length in meters.
    """
    return path_length(data)


def socnavbench_path_length_ratio(data: EpisodeData) -> float:
    """SocNavBench path length ratio (distance / displacement).

    Mirrors ``cost_functions.path_length_ratio`` using the episode trajectory
    and the goal as the reference end configuration.

    Returns:
        float: Path length ratio (>= 1.0 when goal displacement > 0).
    """
    if data.robot_pos.shape[0] == 0:
        return float("nan")
    epsilon = 1e-5
    distance = socnavbench_path_length(data) + epsilon
    displacement = float(np.linalg.norm(data.goal - data.robot_pos[0]))
    if displacement <= 0.0:
        return float("inf")
    return float(distance / displacement)


def _socnavbench_trajectory_with_heading(data: EpisodeData) -> np.ndarray:
    """Construct a SocNavBench-style trajectory array including heading.

    Returns:
        np.ndarray: (T, 3) array of [x, y, heading] entries.
    """
    positions = np.asarray(data.robot_pos, dtype=float)
    if positions.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if positions.shape[0] < 2:
        headings = np.zeros((positions.shape[0],), dtype=float)
    else:
        diffs = positions[1:] - positions[:-1]
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        headings = np.concatenate([headings, headings[-1:]], axis=0)
    return np.column_stack([positions, headings])


def socnavbench_path_irregularity(data: EpisodeData) -> float:
    """SocNavBench path irregularity metric (unnecessary turning per unit length).

    Mirrors ``cost_functions.path_irregularity`` with goal taken from the episode.

    Returns:
        float: Path irregularity value (rad/m).
    """
    trajectory = _socnavbench_trajectory_with_heading(data)
    if trajectory.shape[0] == 0:
        return float("nan")
    goal_heading = float(trajectory[-1, 2]) if trajectory.shape[1] >= 3 else 0.0
    goal_config = np.array([data.goal[0], data.goal[1], goal_heading], dtype=float)

    traj_xy = trajectory[:, :2]
    headings = trajectory[:, 2] if trajectory.shape[1] >= 3 else np.zeros((trajectory.shape[0],))
    heading_vectors = np.column_stack([np.cos(headings), np.sin(headings)])
    point_to_goal_traj = np.squeeze(goal_config)[:2] - traj_xy
    denom = np.linalg.norm(point_to_goal_traj, axis=1) + 1e-10
    cos_theta = np.sum(point_to_goal_traj * heading_vectors, axis=1) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_to_goal_traj = np.arccos(cos_theta)
    return float(np.mean(np.abs(theta_to_goal_traj)))


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
    "distributional_disruption",
    "success",
    "time_to_goal_norm",
    "time_to_goal_norm_success_only",
    "time_to_goal_success_only_valid",
    "time_to_goal_ideal_ratio",
    "time_to_goal_ideal_ratio_valid",
    "collisions",
    "ped_collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
    "total_collision_count",
    "near_misses",
    "min_distance",
    "mean_distance",
    "min_clearance",
    "mean_clearance",
    "robot_ped_within_5m_frac",
    "path_efficiency",
    "socnavbench_path_length",
    "socnavbench_path_length_ratio",
    "socnavbench_path_irregularity",
    "avg_speed",
    "force_q50",
    "force_q90",
    "force_q95",
    "ped_force_q50",
    "ped_force_q90",
    "ped_force_q95",
    "ped_force_mean",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "curvature_mean",
    "energy",
    "force_gradient_norm_mean",
    "wall_collisions",
    "clearing_distance_min",
    "clearing_distance_avg",
    "failure_to_progress",
    "stalled_time",
    "signal_red_phase_violations",
    "signal_stop_line_crossings_under_red",
    "signal_min_distance_to_stop_line_before_crossing_m",
    "signal_delay_after_green_onset_s",
    "signal_pedestrian_conflict_during_legal_crossing_count",
    "signal_unavailable_exclusion_count",
    "signal_metrics_denominator",
    "signal_metrics_evidence",
]


def compute_all_metrics(  # noqa: PLR0913, PLR0915
    data: EpisodeData,
    *,
    horizon: int,
    shortest_path_len: float | None = None,
    robot_max_speed: float | None = None,
    experimental_ped_impact: bool = False,
    experimental_human_interaction_proxy: bool = False,
    experimental_near_miss_ttc: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    near_miss_ttc_threshold_s: float | None = None,
    social_proxemic_radius_m: float = 1.2,
    human_proxy_yield_speed_mps: float = 0.15,
    control_data: EpisodeData | None = None,
) -> dict[str, Any]:
    """Compute all defined metrics for an episode and return them as a mapping.

    Calls each metric implementation on the provided ``EpisodeData`` instance and collects scalar
    outputs keyed by metric name.

    Args:
        data: Episode trajectory and auxiliary data consumed by the metric functions.
        horizon: Episode horizon (number of timesteps) used by normalized metrics.
        shortest_path_len: Optional precomputed shortest path length from start to goal. When
            ``None``, the Euclidean distance between initial pose and goal acts as fallback.
        robot_max_speed: Optional robot speed cap (m/s) for ideal-time normalization. When
            ``None``, observed max speed (fallback ``1.0``) is used.
        experimental_ped_impact: Enable optional experimental ``ped_impact_*`` metrics that
            estimate near-vs-far pedestrian acceleration and turn-rate deltas.
        experimental_human_interaction_proxy: Enable optional diagnostic ``human_proxy_*`` metrics
            for mechanism reports. These are simulation proxies only.
        experimental_near_miss_ttc: Enable the opt-in, diagnostic-only time-to-collision near-miss
            surface (``near_misses_ttc`` plus ``near_miss_ttc__*`` status/threshold keys). The
            legacy distance-based ``near_misses`` metric is unchanged and emitted regardless. When
            inputs are unsupported the count key is omitted and a ``near_misses_ttc_status`` of
            ``"unsupported-inputs"`` is reported (fail-closed, never a false zero).
        ped_impact_radius_m: Near/far distance threshold in meters used by experimental pedestrian
            impact metrics.
        ped_impact_window_steps: Trailing smoothing window length (timesteps) used by
            experimental pedestrian impact metrics.
        near_miss_ttc_threshold_s: TTC threshold (seconds) for the opt-in near-miss count when
            ``experimental_near_miss_ttc`` is enabled. ``None`` uses the uncalibrated diagnostic
            placeholder ``DIAGNOSTIC_TTC_THRESHOLD_S``.
        social_proxemic_radius_m: Personal-space radius used by exploratory social acceptability
            metrics when ``experimental_ped_impact`` is enabled.
        human_proxy_yield_speed_mps: Robot speed threshold used to detect proxy yielding.
        control_data: Optional robot-absent/control trajectory used to compute the schema-backed
            ``distributional_disruption`` block.

    Returns:
        dict[str, Any]: Mapping from metric name (e.g., ``success``, ``force_q50``,
        ``force_gradient_norm_mean``) to the computed scalar value. When
        ``experimental_ped_impact`` is enabled, additional ``ped_impact_*`` and
        exploratory ``social_proxemic_*`` keys are included. When
        ``experimental_near_miss_ttc`` is enabled, opt-in ``near_misses_ttc`` and
        ``near_miss_ttc__*`` diagnostic keys are included (diagnostic-only, not benchmark
        evidence).
    """
    if shortest_path_len is None:
        shortest_path_len = float(np.linalg.norm(data.robot_pos[0] - data.goal))  # simple fallback

    ped_count = int(data.peds_pos.shape[1]) if data.peds_pos.ndim >= 2 else 0
    if ped_count > 0 and not has_force_data(data):
        logger.bind(pedestrians=ped_count, steps=int(data.peds_pos.shape[0])).warning(
            "Missing pedestrian force data; force-based metrics will be NaN.",
        )

    robot_ped_summary = _compute_robot_ped_distance_summary(data)
    obstacle_collision_count = wall_collisions(data)
    agent_collision_count = agent_collisions(data)
    ped_collision_count = robot_ped_summary["human_collisions"]
    total_collision_count = ped_collision_count + obstacle_collision_count + agent_collision_count
    episode_success = (
        data.reached_goal_step is not None
        and data.reached_goal_step < horizon
        and total_collision_count == 0
    )

    values: dict[str, Any] = {}
    if isinstance(data.episode_metadata, dict):
        values["_episode_metadata"] = dict(data.episode_metadata)
    try:
        values.update(calculate_signal_metrics(data))
    except Exception:
        logger.exception("Failed to compute signal metrics; falling back to unavailable defaults.")
        values.update(
            {
                "signal_red_phase_violations": 0,
                "signal_stop_line_crossings_under_red": 0,
                "signal_min_distance_to_stop_line_before_crossing_m": float("nan"),
                "signal_delay_after_green_onset_s": float("nan"),
                "signal_pedestrian_conflict_during_legal_crossing_count": 0,
                "signal_unavailable_exclusion_count": 1,
                "signal_metrics_denominator": 0,
                "signal_metrics_evidence": {
                    "state": "unavailable",
                    "exclusion_reason": "signal_metrics_computation_failed",
                },
            }
        )
    # Use collision-count-based success semantics for benchmark-facing outputs.
    values["success"] = 1.0 if episode_success else 0.0
    values["time_to_goal_norm"] = (
        float(data.reached_goal_step) / float(horizon) if episode_success else 1.0
    )
    values["time_to_goal_norm_success_only"] = (
        float(data.reached_goal_step) / float(horizon) if episode_success else float("nan")
    )
    values["time_to_goal_success_only_valid"] = (
        1.0 if math.isfinite(values["time_to_goal_norm_success_only"]) else 0.0
    )
    speed_cap = _resolved_robot_max_speed(data, robot_max_speed=robot_max_speed)
    if episode_success:
        actual_time_to_goal = time_to_goal(data)
        ideal = ideal_time_to_goal(
            data,
            shortest_path_len=shortest_path_len,
            robot_max_speed=speed_cap,
        )
        values["time_to_goal_ideal_ratio"] = (
            float(actual_time_to_goal / ideal)
            if math.isfinite(actual_time_to_goal) and math.isfinite(ideal) and ideal > 0.0
            else float("nan")
        )
    else:
        values["time_to_goal_ideal_ratio"] = float("nan")
    values["time_to_goal_ideal_ratio_valid"] = (
        1.0 if math.isfinite(values["time_to_goal_ideal_ratio"]) else 0.0
    )
    values["collisions"] = total_collision_count
    values["ped_collision_count"] = ped_collision_count
    values["obstacle_collision_count"] = obstacle_collision_count
    values["agent_collision_count"] = agent_collision_count
    values["total_collision_count"] = total_collision_count
    values["near_misses"] = robot_ped_summary["near_misses"]
    values["min_distance"] = robot_ped_summary["min_distance"]
    values["mean_distance"] = robot_ped_summary["mean_distance"]
    values["min_clearance"] = robot_ped_summary["min_clearance"]
    values["mean_clearance"] = robot_ped_summary["mean_clearance"]
    values["robot_ped_within_5m_frac"] = robot_ped_summary["robot_ped_within_5m_frac"]
    values["path_efficiency"] = path_efficiency(data, shortest_path_len)
    values["socnavbench_path_length"] = socnavbench_path_length(data)
    values["socnavbench_path_length_ratio"] = socnavbench_path_length_ratio(data)
    values["socnavbench_path_irregularity"] = socnavbench_path_irregularity(data)
    values.update(force_quantiles(data))
    values.update(per_ped_force_quantiles(data))
    values["ped_force_mean"] = ped_force_mean(data)
    values["force_exceed_events"] = force_exceed_events(data)
    values["force_sample_stats"] = force_sample_stats(data)
    values["comfort_exposure"] = comfort_exposure(data)
    values["jerk_mean"] = jerk_mean(data)
    values["curvature_mean"] = curvature_mean(data)
    values["energy"] = energy(data)
    values["avg_speed"] = avg_speed(data)
    values["force_gradient_norm_mean"] = force_gradient_norm_mean(data)
    values.update(rollover_stability_metrics(data))
    values.update(clear_tracking_metrics(data))
    values.update(group_space_metrics(data))
    if experimental_near_miss_ttc:
        values.update(_compute_near_miss_ttc_metrics(data, t_thr=near_miss_ttc_threshold_s))
    values["wall_collisions"] = values["obstacle_collision_count"]
    values["clearing_distance_min"] = clearing_distance_min(data)
    values["clearing_distance_avg"] = clearing_distance_avg(data)
    values["failure_to_progress"] = failure_to_progress(data)
    values["stalled_time"] = stalled_time(data)
    if experimental_ped_impact:
        values.update(
            experimental_ped_impact_metrics(
                data,
                radius_m=ped_impact_radius_m,
                window_steps=ped_impact_window_steps,
            )
        )
        values.update(
            experimental_social_acceptability_metrics(
                data,
                proxemic_radius_m=social_proxemic_radius_m,
            )
        )
    if experimental_human_interaction_proxy:
        values.update(
            experimental_human_interaction_proxy_metrics(
                data,
                proxemic_radius_m=social_proxemic_radius_m,
                yield_speed_mps=human_proxy_yield_speed_mps,
            )
        )
    values["distributional_disruption"] = build_distributional_disruption_block(data, control_data)
    return values


def post_process_metrics(
    metrics_raw: dict[str, float],
    *,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Normalize metric output and compute SNQI if configured.

    Returns:
        Normalized metrics dictionary.
    """
    metrics: dict[str, Any] = dict(metrics_raw.items())
    metrics["success"] = bool(metrics.get("success", 0.0) == 1.0)
    fq = {k: v for k, v in metrics.items() if str(k).startswith("force_q")}
    if fq:
        metrics["force_quantiles"] = {
            "q50": float(fq.get("force_q50", float("nan"))),
            "q90": float(fq.get("force_q90", float("nan"))),
            "q95": float(fq.get("force_q95", float("nan"))),
        }
        for k in list(fq.keys()):
            metrics.pop(k, None)
    if snqi_weights is not None:
        snqi_val = snqi(metrics, snqi_weights, baseline_stats=snqi_baseline)
        metrics["snqi"] = float(snqi_val) if math.isfinite(snqi_val) else 0.0
    for count_key in (
        "collisions",
        "ped_collision_count",
        "obstacle_collision_count",
        "agent_collision_count",
        "total_collision_count",
        "near_misses",
        "force_exceed_events",
        "ped_impact_window_steps",
        "ped_impact_ped_count",
        "ped_impact_near_samples",
        "ped_impact_far_samples",
        "ped_impact_accel_delta_valid",
        "ped_impact_turn_rate_delta_valid",
        "social_proxemic_ped_count",
        "social_proxemic_intrusion_steps",
        "human_proxy_ped_count",
        "human_proxy_timestep_count",
        "shield_decision_count",
        "shield_intervention_count",
        "shield_override_count",
        "shield_hard_constraint_violation_count",
        "signal_red_phase_violations",
        "signal_stop_line_crossings_under_red",
        "signal_pedestrian_conflict_during_legal_crossing_count",
        "signal_unavailable_exclusion_count",
        "signal_metrics_denominator",
        "rollover_critical_count",
        "group_count",
        "group_intrusion_step_count",
        "group_metric_timestep_count",
    ):
        if count_key in metrics and metrics[count_key] is not None:
            try:
                metrics[count_key] = int(metrics[count_key])
            except (OverflowError, TypeError, ValueError):
                invalid_value = metrics.pop(count_key, None)
                logger.bind(
                    event="metrics_count_coercion_failed",
                    metric_key=count_key,
                    metric_value=repr(invalid_value),
                ).warning("Dropping metric count value that could not be coerced to int.")
    for valid_key in (
        "time_to_goal_success_only_valid",
        "time_to_goal_ideal_ratio_valid",
        "social_proxemic_available",
        "human_proxy_available",
        "group_split_intrusion_available",
        "group_space_available",
        "rollover_stability_enabled",
        "rollover_critical",
    ):
        if valid_key in metrics and metrics[valid_key] is not None:
            try:
                metrics[valid_key] = bool(int(metrics[valid_key]))
            except (OverflowError, TypeError, ValueError):
                invalid_value = metrics.pop(valid_key, None)
                logger.bind(
                    event="metrics_flag_coercion_failed",
                    metric_key=valid_key,
                    metric_value=repr(invalid_value),
                ).warning("Dropping metric flag value that could not be coerced to bool.")
    _attach_pedestrian_impact_block(metrics)
    _attach_social_acceptability_block(metrics)
    _attach_human_interaction_proxy_block(metrics)
    _attach_clear_tracking_block(metrics)
    _attach_group_space_block(metrics)
    _attach_social_mini_game_block(metrics)
    metrics.pop("_episode_metadata", None)
    return _sanitize_metrics(metrics)


def _attach_pedestrian_impact_block(metrics: dict[str, Any]) -> None:
    """Attach a schema-backed pedestrian-impact block when flat opt-in metrics are present."""
    if "ped_impact_radius_m" not in metrics:
        return

    metrics["pedestrian_impact"] = {
        "schema_version": "pedestrian-impact.v1",
        "parameters": {
            "near_radius_m": metrics.get("ped_impact_radius_m"),
            "window_steps": metrics.get("ped_impact_window_steps"),
        },
        "units": {
            "accel": "m/s^2",
            "turn_rate": "rad/s",
            "near_radius": "m",
            "sample_counts": "samples",
            "sample_fraction": "fraction",
        },
        "sample_counts": {
            "pedestrians": metrics.get("ped_impact_ped_count"),
            "near_samples": metrics.get("ped_impact_near_samples"),
            "far_samples": metrics.get("ped_impact_far_samples"),
            "near_sample_frac": metrics.get("ped_impact_near_sample_frac"),
        },
        "canonical_reductions": {
            "accel_delta_mean": metrics.get("ped_impact_accel_delta_mean"),
            "accel_delta_median": metrics.get("ped_impact_accel_delta_median"),
            "accel_delta_valid_pedestrians": metrics.get("ped_impact_accel_delta_valid"),
            "turn_rate_delta_mean": metrics.get("ped_impact_turn_rate_delta_mean"),
            "turn_rate_delta_median": metrics.get("ped_impact_turn_rate_delta_median"),
            "turn_rate_delta_valid_pedestrians": metrics.get("ped_impact_turn_rate_delta_valid"),
        },
        "component_means": {
            "accel_near_mean": metrics.get("ped_impact_accel_near_mean"),
            "accel_far_mean": metrics.get("ped_impact_accel_far_mean"),
            "turn_rate_near_mean": metrics.get("ped_impact_turn_rate_near_mean"),
            "turn_rate_far_mean": metrics.get("ped_impact_turn_rate_far_mean"),
        },
    }


def _attach_social_acceptability_block(metrics: dict[str, Any]) -> None:
    """Attach an exploratory social-acceptability block when pilot metrics are present."""
    if "social_proxemic_radius_m" not in metrics:
        return

    metrics["social_acceptability"] = {
        "schema_version": "social-acceptability-pilot.v1",
        "status": "exploratory",
        "parameters": {
            "proxemic_radius_m": metrics.get("social_proxemic_radius_m"),
        },
        "units": {
            "clearance": "m",
            "intrusion_area": "m*s",
            "intrusion_fraction": "fraction",
            "sample_counts": "count",
        },
        "available": metrics.get("social_proxemic_available"),
        "sample_counts": {
            "pedestrians": metrics.get("social_proxemic_ped_count"),
            "timesteps": metrics.get("social_proxemic_intrusion_steps"),
        },
        "proxemic": {
            "intrusion_steps": metrics.get("social_proxemic_intrusion_steps"),
            "intrusion_frac": metrics.get("social_proxemic_intrusion_frac"),
            "intrusion_area_m_s": metrics.get("social_proxemic_intrusion_area_m_s"),
            "min_clearance_m": metrics.get("social_proxemic_min_clearance_m"),
        },
        "interpretation": (
            "Exploratory trajectory-only proxemic proxy; not a replacement for SNQI, "
            "headline safety metrics, or human-subject validation."
        ),
    }


def _attach_human_interaction_proxy_block(metrics: dict[str, Any]) -> None:
    """Attach a schema-backed human-interaction proxy block when flat metrics are present."""
    if "human_proxy_proxemic_radius_m" not in metrics:
        return

    metrics["human_interaction_proxy"] = {
        "schema_version": "human-interaction-proxy.v1",
        "status": "simulation_proxy",
        "parameters": {
            "proxemic_radius_m": metrics.get("human_proxy_proxemic_radius_m"),
            "yield_speed_mps": metrics.get("human_proxy_yield_speed_mps"),
        },
        "units": {
            "discomfort_exposure": "m*s",
            "duration": "s",
            "time_to_yield": "s",
            "distance": "m",
            "path_deviation": "m",
            "sample_counts": "count",
        },
        "available": metrics.get("human_proxy_available"),
        "sample_counts": {
            "pedestrians": metrics.get("human_proxy_ped_count"),
            "timesteps": metrics.get("human_proxy_timestep_count"),
        },
        "canonical_reductions": {
            "human_discomfort_exposure_m_s": metrics.get("human_discomfort_exposure_m_s"),
            "intrusion_duration_s": metrics.get("intrusion_duration_s"),
            "time_to_yield_s": metrics.get("time_to_yield_s"),
            "robot_yield_distance_m": metrics.get("robot_yield_distance_m"),
            "pedestrian_path_deviation_proxy_m": metrics.get("pedestrian_path_deviation_proxy_m"),
            "group_split_intrusion_available": metrics.get("group_split_intrusion_available"),
        },
        "exclusions": {
            "group_split_intrusion": (
                "Not computed without group-membership or social-group labels in EpisodeData."
            ),
        },
        "interpretation": (
            "Diagnostic simulation-proxy metrics for mechanism reports only; not validated "
            "human comfort, human-subject, safety, or paper-grade social-compliance evidence."
        ),
    }


def _attach_clear_tracking_block(metrics: dict[str, Any]) -> None:
    """Attach a schema-backed CLEAR tracking diagnostic block when present."""
    if "clear_tracking_enabled" not in metrics:
        return
    metrics["clear_tracking_uncertainty"] = {
        "schema_version": "clear-tracking-metrics.v1",
        "enabled": bool(metrics.get("clear_tracking_enabled")),
        "mota": metrics.get("clear_mota"),
        "motp_m": metrics.get("clear_motp_m"),
        "counts": {
            "ground_truth": metrics.get("clear_ground_truth_count"),
            "detections": metrics.get("clear_detection_count"),
            "missed_detections": metrics.get("clear_missed_detection_count"),
            "false_positives": metrics.get("clear_false_positive_count"),
            "id_switches": metrics.get("clear_id_switch_count"),
            "motp_matches": metrics.get("clear_motp_match_count"),
        },
        "claim_boundary": (
            "CLEAR-style diagnostic metrics for synthetic ScenarioBelief observation "
            "contracts; not calibrated real-sensor evidence."
        ),
    }


def _attach_group_space_block(metrics: dict[str, Any]) -> None:
    """Attach a schema-backed group-space intrusion block when present.

    The block is emitted only when ``group_space_available`` is present (i.e. the
    episode declared social groups); default rows are unaffected.
    """
    if "group_space_available" not in metrics:
        return
    metrics["group_space"] = {
        "schema_version": "group-space-metrics.v1",
        "available": bool(metrics.get("group_space_available")),
        "group_count": metrics.get("group_count"),
        "definitions": {
            "group_intrusion_episode_rate": (
                "Per-episode binary indicator; aggregate mean is the fraction of "
                "episodes with >=1 timestep inside any group o-space."
            ),
            "group_intrusion_time_ratio": (
                "Fraction of timesteps where the robot center lies inside any group o-space."
            ),
            "min_distance_to_group_boundary": (
                "Signed o-space boundary clearance (m); negative inside, minimized over "
                "timesteps to reflect worst-case intrusion severity."
            ),
        },
        "metrics": {
            "group_intrusion_episode_rate": metrics.get("group_intrusion_episode_rate"),
            "group_intrusion_time_ratio": metrics.get("group_intrusion_time_ratio"),
            "min_distance_to_group_centroid": metrics.get("min_distance_to_group_centroid"),
            "min_distance_to_group_boundary": metrics.get("min_distance_to_group_boundary"),
        },
        "support": {
            "group_intrusion_step_count": metrics.get("group_intrusion_step_count"),
            "group_metric_timestep_count": metrics.get("group_metric_timestep_count"),
            "nearest_group_id": metrics.get("nearest_group_id"),
        },
        "claim_boundary": (
            "Diagnostic social-space intrusion metrics for declared groups; not "
            "validated human-comfort or safety evidence."
        ),
    }


def _social_mini_game_metric_row(
    *,
    metric: str,
    status: str,
    unit: str,
    denominator: str,
    value: float | int | bool | None = None,
    support_count: int = 0,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Build one Social Mini-Game metric row with explicit availability semantics.

    Returns:
        Row dictionary ready for the ``social_mini_game`` metric block.
    """
    row: dict[str, Any] = {
        "metric": metric,
        "status": status,
        "unit": unit,
        "denominator": denominator,
        "support_count": int(support_count),
    }
    if value is not None:
        row["value"] = value
    if unavailable_reason:
        row["unavailable_reason"] = unavailable_reason
    return row


def _social_mini_game_mechanism_family(metrics: dict[str, Any]) -> str:
    """Resolve a mechanism-family label from optional episode metadata.

    Returns:
        Mechanism-family label, or ``unknown`` when no supported metadata key is present.
    """
    metadata = metrics.get("_episode_metadata")
    if not isinstance(metadata, dict):
        return "unknown"
    for key in (
        "social_mini_game_mechanism_family",
        "mechanism_family",
        "mechanism_aware_suite_id",
        "target_mechanism",
        "mechanism_id",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _attach_social_mini_game_block(metrics: dict[str, Any]) -> None:
    """Attach Social Mini-Game diagnostic metric rows when source metrics are present."""
    rows: list[dict[str, Any]] = []

    makespan_ratio = metrics.get("time_to_goal_ideal_ratio")
    makespan_valid = bool(metrics.get("time_to_goal_ideal_ratio_valid", False))
    rows.append(
        _social_mini_game_metric_row(
            metric="makespan_ratio",
            status="available" if makespan_valid and makespan_ratio is not None else "undefined",
            unit="ratio",
            denominator="successful episodes with finite ideal-time baseline",
            value=makespan_ratio if makespan_valid and makespan_ratio is not None else None,
            support_count=1 if makespan_valid and makespan_ratio is not None else 0,
            unavailable_reason=(
                None
                if makespan_valid and makespan_ratio is not None
                else "episode did not reach the goal with a finite ideal-time baseline"
            ),
        )
    )

    path_ratio = metrics.get("socnavbench_path_length_ratio")
    path_ratio_available = isinstance(path_ratio, int | float) and math.isfinite(float(path_ratio))
    rows.append(
        _social_mini_game_metric_row(
            metric="path_deviation_ratio",
            status="available" if path_ratio_available else "undefined",
            unit="ratio_over_straight_line",
            denominator="robot trajectory length divided by start-goal displacement",
            value=float(path_ratio) - 1.0 if path_ratio_available else None,
            support_count=1 if path_ratio_available else 0,
            unavailable_reason=(
                None
                if path_ratio_available
                else "start-goal displacement or trajectory length was unavailable"
            ),
        )
    )

    deadlock_value = metrics.get("failure_to_progress")
    deadlock_available = isinstance(deadlock_value, int | float) and math.isfinite(
        float(deadlock_value)
    )
    rows.append(
        _social_mini_game_metric_row(
            metric="deadlock_frequency",
            status="available" if deadlock_available else "undefined",
            unit="events_per_episode",
            denominator="one episode",
            value=float(deadlock_value) if deadlock_available else None,
            support_count=1 if deadlock_available else 0,
            unavailable_reason=(
                None
                if deadlock_available
                else "failure-to-progress metric was unavailable for this episode"
            ),
        )
    )

    rows.append(
        _social_mini_game_metric_row(
            metric="flow_throughput",
            status="unavailable",
            unit="pedestrians_per_second",
            denominator="pedestrian arrivals or exits over elapsed scenario time",
            unavailable_reason="episode data does not carry pedestrian arrival or exit counts",
        )
    )

    distributional_block = metrics.get("distributional_disruption")
    distributional_available = (
        isinstance(distributional_block, dict)
        and distributional_block.get("schema_version") == "distributional-disruption.v1"
        and bool(distributional_block.get("cohort_metrics"))
    )
    distributional_support_counts = (
        distributional_block.get("support_counts") if isinstance(distributional_block, dict) else {}
    )
    if not isinstance(distributional_support_counts, dict):
        distributional_support_counts = {}
    rows.append(
        _social_mini_game_metric_row(
            metric="distributional_inconvenience",
            status="available" if distributional_available else "unavailable",
            unit="diagnostic_cohort_delta",
            denominator="matched robot-present and control pedestrian trajectories by cohort",
            support_count=(
                sum(
                    int(value)
                    for value in distributional_support_counts.values()
                    if isinstance(value, int | float)
                )
                if distributional_available
                else 0
            ),
            unavailable_reason=(
                None
                if distributional_available
                else "control trajectory cohort comparison was unavailable"
            ),
        )
    )

    human_block = metrics.get("human_interaction_proxy")
    human_reductions = (
        human_block.get("canonical_reductions") if isinstance(human_block, dict) else {}
    )
    invasiveness_value = (
        human_reductions.get("human_discomfort_exposure_m_s")
        if isinstance(human_reductions, dict)
        else None
    )
    invasiveness_available = isinstance(invasiveness_value, int | float) and math.isfinite(
        float(invasiveness_value)
    )
    rows.append(
        _social_mini_game_metric_row(
            metric="invasiveness",
            status="available" if invasiveness_available else "unavailable",
            unit="m*s",
            denominator="proxemic exposure over robot-pedestrian trajectory samples",
            value=float(invasiveness_value) if invasiveness_available else None,
            support_count=(
                int((human_block.get("sample_counts") or {}).get("timesteps") or 0)
                if isinstance(human_block, dict)
                else 0
            ),
            unavailable_reason=(
                None
                if invasiveness_available
                else "human-interaction proxy metrics were not enabled or lacked support"
            ),
        )
    )

    metrics["social_mini_game"] = {
        "schema_version": "social-mini-game-metrics.v1",
        "status": "diagnostic",
        "mechanism_family": _social_mini_game_mechanism_family(metrics),
        "rows": rows,
        "interpretation": (
            "Diagnostic Social Mini-Game mechanism metrics. Rows distinguish valid zero values "
            "from unavailable or undefined metrics and are not paper-grade evidence by themselves."
        ),
    }


def _distributional_disruption_static_fields() -> dict[str, Any]:
    """Return fixed schema fields for the distributional disruption block."""
    return {
        "schema_version": "distributional-disruption.v1",
        "claim_boundary": (
            "These metrics are diagnostic simulation measures for analyzing per-subgroup "
            "displacement and inconvenience distribution in controlled settings. "
            "They do not represent real-world ethical outcomes."
        ),
        "baseline_condition": "control_run_without_robot",
        "cohort_definitions": {
            "slow_speed_tier": "Pedestrians with average control speed <= 1.0 m/s",
            "fast_speed_tier": "Pedestrians with average control speed > 1.0 m/s and <= 1.8 m/s",
            "extreme_speed_tier": "Pedestrians with average control speed > 1.8 m/s",
        },
        "units": {
            "displacement_mean_m": "meters",
            "delay_mean_s": "seconds",
        },
        "metric_definitions": {
            "displacement_mean_m": {
                "formula": "mean_t ||robot_present_position_t - control_position_t||",
                "denominator": (
                    "matched timesteps per pedestrian, then supported pedestrians per cohort"
                ),
            },
            "delay_mean_s": {
                "formula": (
                    "max(0, robot_present_path_length - control_path_length) / "
                    "max(control_mean_speed, 0.1)"
                ),
                "denominator": "supported pedestrians per cohort",
            },
        },
        "non_claims": (
            "We make no claims regarding real-world fairness, equity, bias, "
            "protected attributes, demographic groups, or disparate impact. "
            "These measures are diagnostic simulation proxies only."
        ),
    }


def _distributional_disruption_empty_counts() -> dict[str, int]:
    """Return zeroed support counts for the supported observable speed-tier cohorts."""
    return {
        "slow_speed_tier": 0,
        "fast_speed_tier": 0,
        "extreme_speed_tier": 0,
    }


def _distributional_disruption_missing_block(status: str, reason: str) -> dict[str, Any]:
    """Return a block with all cohorts explicitly marked as missing or unavailable."""
    static_fields = _distributional_disruption_static_fields()
    return {
        **static_fields,
        "support_counts": _distributional_disruption_empty_counts(),
        "cohort_metrics": {},
        "missing_data": {
            cohort: {"status": status, "reason": reason}
            for cohort in static_fields["cohort_definitions"]
        },
    }


def _pedestrian_control_speed(control_pos: np.ndarray, dt: float) -> float:
    """Return mean pedestrian speed for a control trajectory."""
    if len(control_pos) < 2 or dt <= 0:
        return 0.0
    diffs = np.diff(control_pos, axis=0)
    speeds = np.linalg.norm(diffs, axis=-1) / dt
    return float(np.mean(speeds))


def _speed_tier_cohort(avg_control_speed: float) -> str:
    """Map an observable control speed to the distributional cohort label.

    Returns:
        Distributional speed-tier cohort key.
    """
    if avg_control_speed <= 1.0:
        return "slow_speed_tier"
    if avg_control_speed <= 1.8:
        return "fast_speed_tier"
    return "extreme_speed_tier"


def _trajectory_path_length(positions: np.ndarray) -> float:
    """Return the cumulative path length for one pedestrian trajectory."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=-1)))


def _distributional_disruption_sample(
    data: EpisodeData,
    control_data: EpisodeData,
    ped_index: int,
    matched_len: int,
) -> tuple[str, float, float]:
    """Return cohort, displacement, and delay proxy for one matched pedestrian."""
    control_pos = control_data.peds_pos[:, ped_index, :]
    avg_control_speed = _pedestrian_control_speed(control_pos, control_data.dt)
    cohort = _speed_tier_cohort(avg_control_speed)

    displacement = 0.0
    if matched_len > 0:
        dists = np.linalg.norm(
            data.peds_pos[:matched_len, ped_index, :]
            - control_data.peds_pos[:matched_len, ped_index, :],
            axis=-1,
        )
        displacement = float(np.mean(dists))

    present_path_len = _trajectory_path_length(data.peds_pos[:, ped_index, :])
    control_path_len = _trajectory_path_length(control_pos)
    delay = max(0.0, (present_path_len - control_path_len) / max(0.1, avg_control_speed))
    return cohort, displacement, delay


def build_distributional_disruption_block(
    data: EpisodeData,
    control_data: EpisodeData | None = None,
    min_support: int = 2,
) -> dict[str, Any]:
    """Build a schema-backed distributional disruption metric block.

    Args:
        data: Robot-present episode data.
        control_data: Optional control (robot-absent) episode data.
        min_support: Minimum support count to compute cohort metrics.

    Returns:
        Structured block matching the distributional-disruption.v1 schema.
    """
    if control_data is None:
        return _distributional_disruption_missing_block(
            "unavailable",
            "No control trace provided",
        )

    static_fields = _distributional_disruption_static_fields()
    cohort_definitions = static_fields["cohort_definitions"]
    support_counts = _distributional_disruption_empty_counts()
    cohort_metrics: dict[str, dict[str, float]] = {}
    missing_data: dict[str, dict[str, str | int]] = {}

    if data.peds_pos.ndim < 3 or control_data.peds_pos.ndim < 3:
        return _distributional_disruption_missing_block(
            "missing",
            "Pedestrian traces are unavailable or malformed",
        )

    num_peds = min(data.peds_pos.shape[1], control_data.peds_pos.shape[1])
    matched_len = min(data.peds_pos.shape[0], control_data.peds_pos.shape[0])
    cohort_displacements = {cohort: [] for cohort in cohort_definitions}
    cohort_delays = {cohort: [] for cohort in cohort_definitions}

    for ped_index in range(num_peds):
        cohort, displacement, delay = _distributional_disruption_sample(
            data,
            control_data,
            ped_index,
            matched_len,
        )
        support_counts[cohort] += 1
        cohort_displacements[cohort].append(displacement)
        cohort_delays[cohort].append(delay)

    for cohort in cohort_definitions:
        count = support_counts[cohort]
        if count >= min_support:
            cohort_metrics[cohort] = {
                "displacement_mean_m": float(np.mean(cohort_displacements[cohort])),
                "delay_mean_s": float(np.mean(cohort_delays[cohort])),
            }
        elif count > 0:
            missing_data[cohort] = {
                "status": "under_supported",
                "reason": f"Fewer than {min_support} samples (support count: {count})",
                "support_count": count,
                "minimum_support": min_support,
            }
        else:
            missing_data[cohort] = {
                "status": "missing",
                "reason": "No samples available",
            }

    return {
        **static_fields,
        "support_counts": support_counts,
        "cohort_metrics": cohort_metrics,
        "missing_data": missing_data,
    }


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove NaN/inf metric entries to keep JSON serialization clean.

    Returns:
        Cleaned metrics dictionary with NaN/inf values removed.
    """
    preserve_empty_dict_keys = {"cohort_metrics", "missing_data"}
    clean: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            nested = _sanitize_metrics(val)
            if nested or key in preserve_empty_dict_keys:
                clean[key] = nested
            continue
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            continue
        clean[key] = val
    return clean


__all__ = [
    "METRIC_NAMES",
    "ROLLOVER_CRITICAL_EVENT",
    "EpisodeData",
    "acceleration_avg",
    "acceleration_max",
    "acceleration_min",
    "agent_collisions",
    "aggregated_time",
    "avg_speed",
    "build_distributional_disruption_block",
    "clearing_distance_avg",
    "clearing_distance_min",
    "collision_count",
    "collisions",
    "comfort_exposure",
    "compute_all_metrics",
    "curvature_mean",
    "distance_to_human_min",
    "energy",
    "evaluate_stability_margin",
    "experimental_human_interaction_proxy_metrics",
    "experimental_social_acceptability_metrics",
    "failure_to_progress",
    "force_exceed_events",
    "force_gradient_norm_mean",
    "force_quantiles",
    "force_sample_stats",
    "has_force_data",
    "human_collisions",
    "ideal_time_to_goal",
    "jerk_avg",
    "jerk_max",
    "jerk_mean",
    "jerk_min",
    "mean_clearance",
    "mean_distance",
    "min_clearance",
    "min_distance",
    "near_misses",
    "path_efficiency",
    "path_length",
    "ped_force_mean",
    "post_process_metrics",
    "robot_ped_within_5m_frac",
    "robot_ped_within_distance_frac",
    "snqi",
    "socnavbench_path_irregularity",
    "socnavbench_path_length",
    "socnavbench_path_length_ratio",
    "space_compliance",
    "stalled_time",
    "success",
    "success_path_length",
    "success_rate",
    "time_to_collision_min",
    "time_to_goal",
    "time_to_goal_ideal_ratio",
    "time_to_goal_norm",
    "time_to_goal_norm_success_only",
    "timeout",
    "velocity_avg",
    "velocity_max",
    "velocity_min",
    "wall_collisions",
]
