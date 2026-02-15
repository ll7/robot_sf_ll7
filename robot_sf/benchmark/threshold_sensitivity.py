"""Threshold sensitivity helpers for near-miss and comfort metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.constants import COLLISION_DIST, COMFORT_FORCE_THRESHOLD, NEAR_MISS_DIST
from robot_sf.benchmark.metrics import EpisodeData


@dataclass(slots=True)
class SensitivityEpisode:
    """Episode payload for threshold sensitivity analysis."""

    family: str
    data: EpisodeData
    episode_id: str | None = None


def near_miss_count(
    data: EpisodeData,
    *,
    collision_distance: float,
    near_miss_distance: float,
) -> float:
    """Count timesteps in the near-miss distance band for an episode.

    Returns:
        Number of near-miss timesteps for the requested distance band.
    """
    if near_miss_distance <= collision_distance:
        raise ValueError("near_miss_distance must be greater than collision_distance")
    min_dist, _nearest_idx = _nearest_pedestrian_distances(data)
    mask = (min_dist >= collision_distance) & (min_dist < near_miss_distance)
    return float(np.count_nonzero(mask))


def comfort_exposure_ratio(data: EpisodeData, *, force_threshold: float) -> float:
    """Compute comfort-exposure ratio for a force threshold.

    Returns:
        Ratio in [0, 1] when force samples are available; NaN when force samples
        are unavailable for non-empty pedestrian episodes.
    """
    ped_count = _ped_count(data)
    if ped_count == 0:
        return 0.0
    if data.ped_forces.shape != data.peds_pos.shape:
        return float("nan")
    mags = np.linalg.norm(data.ped_forces, axis=2)
    finite_mask = np.isfinite(mags)
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count == 0:
        return float("nan")
    events = int(np.count_nonzero((mags > force_threshold) & finite_mask))
    return float(events / finite_count)


def speed_weighted_near_miss(
    data: EpisodeData,
    *,
    collision_distance: float,
    near_miss_distance: float,
    relative_speed_reference: float = 1.0,
    max_weight: float = 3.0,
) -> float:
    """Compute a speed-weighted near-miss score based on relative speed.

    Returns:
        Weighted near-miss score where each event is scaled by relative speed.
    """
    if relative_speed_reference <= 0.0:
        raise ValueError("relative_speed_reference must be positive")
    min_dist, nearest_idx = _nearest_pedestrian_distances(data)
    near_mask = (min_dist >= collision_distance) & (min_dist < near_miss_distance)
    if not np.any(near_mask):
        return 0.0
    ped_vel = _ped_velocities(data.peds_pos, dt=data.dt)
    if data.robot_vel.shape[0] != data.peds_pos.shape[0]:
        logger.warning(
            "speed_weighted_near_miss received shape-mismatched trajectories; returning 0.0",
            robot_steps=int(data.robot_vel.shape[0]),
            ped_steps=int(data.peds_pos.shape[0]),
        )
        return 0.0
    near_steps = np.where(near_mask)[0]
    if near_steps.size == 0:
        return 0.0
    ped_indices = nearest_idx[near_steps].astype(int)
    valid_mask = ped_indices >= 0
    if not np.any(valid_mask):
        return 0.0
    valid_steps = near_steps[valid_mask]
    valid_ped_indices = ped_indices[valid_mask]
    rel_vel = data.robot_vel[valid_steps] - ped_vel[valid_steps, valid_ped_indices]
    rel_speed = np.linalg.norm(rel_vel, axis=1)
    weights = np.minimum(max_weight, np.maximum(0.0, rel_speed / relative_speed_reference))
    return float(np.sum(weights))


def ttc_gated_near_miss_count(
    data: EpisodeData,
    *,
    collision_distance: float,
    near_miss_distance: float,
    ttc_horizon_sec: float,
) -> float:
    """Count near-miss timesteps that also satisfy a TTC safety horizon.

    Returns:
        Number of near-miss timesteps with TTC <= configured horizon.
    """
    if ttc_horizon_sec <= 0.0:
        raise ValueError("ttc_horizon_sec must be positive")
    if _ped_count(data) == 0:
        return 0.0
    dists, rel_vecs, rel_speed = _distance_and_relative_speed(data)
    in_band = (dists >= collision_distance) & (dists < near_miss_distance)
    if not np.any(in_band):
        return 0.0
    approaching = (
        np.einsum("ijk,ijk->ij", rel_vecs, data.peds_pos - data.robot_pos[:, None, :]) > 0.0
    )
    valid = in_band & approaching & (rel_speed > 1e-9)
    ttc = np.full(dists.shape, np.inf, dtype=float)
    ttc[valid] = dists[valid] / rel_speed[valid]
    event_steps = np.any(ttc <= ttc_horizon_sec, axis=1)
    return float(np.count_nonzero(event_steps))


def analyze_threshold_sensitivity(
    episodes: list[SensitivityEpisode],
    *,
    collision_grid: list[float],
    near_miss_grid: list[float],
    comfort_grid: list[float],
    ttc_horizons_sec: list[float],
    relative_speed_reference: float = 1.0,
) -> dict[str, Any]:
    """Aggregate threshold sensitivity outputs across scenario families.

    Returns:
        Report payload with per-family baseline values and sweep summaries.
    """
    grouped: dict[str, list[SensitivityEpisode]] = defaultdict(list)
    for ep in episodes:
        grouped[str(ep.family)].append(ep)

    out: dict[str, Any] = {
        "threshold_defaults": {
            "collision_distance_m": float(COLLISION_DIST),
            "near_miss_distance_m": float(NEAR_MISS_DIST),
            "comfort_force_threshold": float(COMFORT_FORCE_THRESHOLD),
        },
        "families": {},
    }
    for family, family_eps in sorted(grouped.items()):
        baseline_near = [
            near_miss_count(
                ep.data,
                collision_distance=COLLISION_DIST,
                near_miss_distance=NEAR_MISS_DIST,
            )
            for ep in family_eps
        ]
        baseline_comfort = [
            comfort_exposure_ratio(ep.data, force_threshold=COMFORT_FORCE_THRESHOLD)
            for ep in family_eps
        ]
        weighted_near = [
            speed_weighted_near_miss(
                ep.data,
                collision_distance=COLLISION_DIST,
                near_miss_distance=NEAR_MISS_DIST,
                relative_speed_reference=relative_speed_reference,
            )
            for ep in family_eps
        ]

        ttc_sweep: list[dict[str, float]] = []
        for horizon in ttc_horizons_sec:
            values = [
                ttc_gated_near_miss_count(
                    ep.data,
                    collision_distance=COLLISION_DIST,
                    near_miss_distance=NEAR_MISS_DIST,
                    ttc_horizon_sec=horizon,
                )
                for ep in family_eps
            ]
            ttc_sweep.append(
                {
                    "ttc_horizon_sec": float(horizon),
                    **_summary_stats(values),
                    "corr_vs_distance_near_miss": _pearson_corr(values, baseline_near),
                },
            )

        near_sweep: list[dict[str, float]] = []
        for collision in collision_grid:
            for near in near_miss_grid:
                if near <= collision:
                    continue
                values = [
                    near_miss_count(
                        ep.data,
                        collision_distance=collision,
                        near_miss_distance=near,
                    )
                    for ep in family_eps
                ]
                near_sweep.append(
                    {
                        "collision_distance_m": float(collision),
                        "near_miss_distance_m": float(near),
                        **_summary_stats(values),
                    },
                )

        comfort_sweep = []
        for threshold in comfort_grid:
            values = [
                comfort_exposure_ratio(ep.data, force_threshold=threshold) for ep in family_eps
            ]
            comfort_sweep.append(
                {
                    "comfort_force_threshold": float(threshold),
                    **_summary_stats(values),
                },
            )

        out["families"][family] = {
            "episode_count": len(family_eps),
            "distance_near_miss_baseline": _summary_stats(baseline_near),
            "comfort_exposure_baseline": _summary_stats(baseline_comfort),
            "speed_weighted_near_miss": {
                **_summary_stats(weighted_near),
                "corr_vs_distance_near_miss": _pearson_corr(weighted_near, baseline_near),
            },
            "near_miss_threshold_sweep": near_sweep,
            "comfort_threshold_sweep": comfort_sweep,
            "ttc_gated_sweep": ttc_sweep,
        }
    return out


def sensitivity_episodes_from_replay_records(  # noqa: C901, PLR0912
    records: list[dict[str, Any]],
) -> list[SensitivityEpisode]:
    """Create sensitivity episodes from records containing replay payloads.

    Returns:
        Parsed sensitivity episode list for records with replay traces.
    """
    out: list[SensitivityEpisode] = []
    for rec in records:
        replay_steps = rec.get("replay_steps")
        replay_peds = rec.get("replay_peds")
        if not isinstance(replay_steps, list) or not isinstance(replay_peds, list):
            continue
        if not replay_steps or not replay_peds:
            continue
        if len(replay_steps) != len(replay_peds):
            continue

        robot_xy: list[list[float]] = []
        valid_steps = True
        for step in replay_steps:
            if not isinstance(step, (list, tuple)) or len(step) < 3:
                valid_steps = False
                break
            robot_xy.append([float(step[1]), float(step[2])])
        if not valid_steps:
            continue

        robot_pos = np.asarray(robot_xy, dtype=float)
        t_steps = int(robot_pos.shape[0])
        ped_count = max((len(step) for step in replay_peds if isinstance(step, list)), default=0)
        peds_pos = np.full((t_steps, ped_count, 2), np.nan, dtype=float)

        for t, peds in enumerate(replay_peds):
            if not isinstance(peds, list):
                continue
            for k, ped in enumerate(peds[:ped_count]):
                if isinstance(ped, (list, tuple)) and len(ped) >= 2:
                    peds_pos[t, k, 0] = float(ped[0])
                    peds_pos[t, k, 1] = float(ped[1])

        ped_forces = np.full((t_steps, ped_count, 2), np.nan, dtype=float)
        replay_forces = rec.get("replay_ped_forces")
        if isinstance(replay_forces, list) and len(replay_forces) == t_steps:
            for t, forces in enumerate(replay_forces):
                if not isinstance(forces, list):
                    continue
                for k, force in enumerate(forces[:ped_count]):
                    if isinstance(force, (list, tuple)) and len(force) >= 2:
                        ped_forces[t, k, 0] = float(force[0])
                        ped_forces[t, k, 1] = float(force[1])

        raw_dt = rec.get("replay_dt")
        parsed_dt = float(raw_dt) if raw_dt is not None else 0.1
        dt = parsed_dt if parsed_dt > 0 else 0.1
        robot_vel = _robot_velocity(robot_pos, dt)
        robot_acc = _robot_velocity(robot_vel, dt)
        goal = robot_pos[-1] if t_steps else np.zeros(2, dtype=float)
        scenario_params = (
            rec.get("scenario_params") if isinstance(rec.get("scenario_params"), dict) else {}
        )
        family = _scenario_family(rec, scenario_params)
        out.append(
            SensitivityEpisode(
                family=family,
                episode_id=str(rec.get("episode_id"))
                if rec.get("episode_id") is not None
                else None,
                data=EpisodeData(
                    robot_pos=robot_pos,
                    robot_vel=robot_vel,
                    robot_acc=robot_acc,
                    peds_pos=peds_pos,
                    ped_forces=ped_forces,
                    goal=np.asarray(goal, dtype=float),
                    dt=dt,
                    reached_goal_step=None,
                ),
            ),
        )
    return out


def _scenario_family(record: dict[str, Any], scenario_params: dict[str, Any]) -> str:
    """Resolve scenario family key as ``archetype:density`` with fallbacks.

    Returns:
        Scenario family identifier.
    """
    metadata = (
        scenario_params.get("metadata") if isinstance(scenario_params.get("metadata"), dict) else {}
    )
    archetype = (
        metadata.get("archetype")
        or scenario_params.get("archetype")
        or record.get("archetype")
        or "unknown"
    )
    density = (
        metadata.get("density")
        or scenario_params.get("density")
        or record.get("density")
        or "unknown"
    )
    return f"{archetype}:{density}"


def _nearest_pedestrian_distances(data: EpisodeData) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step nearest pedestrian distance and index."""
    if _ped_count(data) == 0:
        return np.full(data.robot_pos.shape[0], np.inf, dtype=float), np.full(
            data.robot_pos.shape[0], -1, dtype=int
        )
    diffs = data.peds_pos - data.robot_pos[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)
    invalid = ~np.isfinite(dists)
    dists = np.where(invalid, np.inf, dists)
    nearest_idx = np.argmin(dists, axis=1)
    min_dist = np.min(dists, axis=1)
    nearest_idx = np.where(np.isfinite(min_dist), nearest_idx, -1)
    return min_dist.astype(float), nearest_idx.astype(int)


def _ped_velocities(peds_pos: np.ndarray, *, dt: float) -> np.ndarray:
    """Compute per-step pedestrian velocities from replay positions.

    Returns:
        Per-step velocity array with the same shape as ``peds_pos``.
    """
    vel = np.zeros_like(peds_pos, dtype=float)
    if peds_pos.shape[0] < 2 or dt <= 0:
        return vel
    delta = peds_pos[1:] - peds_pos[:-1]
    finite = np.isfinite(peds_pos[1:]) & np.isfinite(peds_pos[:-1])
    valid = finite[..., 0] & finite[..., 1]
    vel_step = np.zeros_like(delta, dtype=float)
    vel_step[valid] = delta[valid] / dt
    vel[1:] = vel_step
    return vel


def _distance_and_relative_speed(data: EpisodeData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance, relative vectors, and relative speed matrices.

    Returns:
        Tuple of (distance matrix, relative velocity vectors, relative speed matrix).
    """
    if _ped_count(data) == 0:
        empty = np.empty((data.robot_pos.shape[0], 0), dtype=float)
        return empty, np.empty((data.robot_pos.shape[0], 0, 2), dtype=float), empty
    ped_vel = _ped_velocities(data.peds_pos, dt=data.dt)
    rel_vel = data.robot_vel[:, None, :] - ped_vel
    rel_pos = data.peds_pos - data.robot_pos[:, None, :]
    dists = np.linalg.norm(rel_pos, axis=2)
    rel_speed = np.linalg.norm(rel_vel, axis=2)
    dists = np.where(np.isfinite(dists), dists, np.inf)
    rel_speed = np.where(np.isfinite(rel_speed), rel_speed, 0.0)
    rel_vel = np.where(np.isfinite(rel_vel), rel_vel, 0.0)
    return dists, rel_vel, rel_speed


def _robot_velocity(pos: np.ndarray, dt: float) -> np.ndarray:
    """Compute first-order finite-difference velocities.

    Returns:
        Velocity array with same shape as ``pos``.
    """
    vel = np.zeros_like(pos, dtype=float)
    if pos.shape[0] < 2 or dt <= 0:
        return vel
    vel[1:] = (pos[1:] - pos[:-1]) / dt
    return vel


def _ped_count(data: EpisodeData) -> int:
    """Return number of pedestrians in an episode with defensive shape handling.

    Returns:
        Number of pedestrians when `peds_pos` is three-dimensional; otherwise 0.
    """
    if data.peds_pos.ndim < 3:
        return 0
    return int(data.peds_pos.shape[1])


def _summary_stats(values: list[float]) -> dict[str, float]:
    """Compute summary stats while ignoring NaN values.

    Returns:
        Dictionary with mean, median, and p95 statistics.
    """
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation on finite paired values.

    Returns:
        Correlation coefficient, or NaN for degenerate inputs.
    """
    if len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x_f = x[mask]
    y_f = y[mask]
    if np.std(x_f) <= 1e-12 or np.std(y_f) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_f, y_f)[0, 1])


__all__ = [
    "SensitivityEpisode",
    "analyze_threshold_sensitivity",
    "comfort_exposure_ratio",
    "near_miss_count",
    "sensitivity_episodes_from_replay_records",
    "speed_weighted_near_miss",
    "ttc_gated_near_miss_count",
]
