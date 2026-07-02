"""Diagnostic summaries for synthetic pedestrian trajectory quality.

These helpers intentionally report distributions only. They do not apply realism
thresholds or decide whether a pedestrian model is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DistributionSummary = dict[str, float | int | str]
TrajectoryQualityReport = dict[str, DistributionSummary | bool | str]


@dataclass(frozen=True)
class TrajectoryQualityConfig:
    """Configuration for pedestrian trajectory-quality diagnostics."""

    stop_speed_threshold_mps: float = 0.05
    pairwise_sample_stride: int = 1
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    min_heading_speed_mps: float = 1e-6


def compute_trajectory_quality_distributions(
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
    *,
    dt_s: float,
    config: TrajectoryQualityConfig | None = None,
) -> TrajectoryQualityReport:
    """Return JSON-safe quality distributions for pedestrian trajectories.

    Parameters
    ----------
    positions:
        Pedestrian positions shaped ``(T, K, 2)`` in meters.
    velocities:
        Optional pedestrian velocities shaped ``(T, K, 2)`` or ``(T - 1, K, 2)`` in meters per
        second. When omitted, velocities are finite-differenced from consecutive positions.
    dt_s:
        Positive timestep duration in seconds.
    config:
        Diagnostic configuration. Defaults are intentionally descriptive, not pass/fail thresholds.
    """

    cfg = config or TrajectoryQualityConfig()
    positions_arr = _as_trajectory_array(positions, name="positions")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if cfg.pairwise_sample_stride <= 0:
        raise ValueError("pairwise_sample_stride must be positive")

    step_count, ped_count, _ = positions_arr.shape
    if step_count == 0:
        return _empty_report("empty")
    if ped_count == 0:
        return _empty_report("no_pedestrians")

    velocity_arr = _resolve_velocities(positions_arr, velocities, dt_s)
    speed = np.linalg.norm(velocity_arr, axis=2)
    acceleration = _finite_difference(velocity_arr, dt_s)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=2)

    report: TrajectoryQualityReport = {
        "status": "ok",
        "diagnostic_only": True,
        "thresholds_applied": False,
        "speed_mps": _summarize_distribution(speed, cfg.quantiles),
        "acceleration_mps2": _summarize_distribution(acceleration_magnitude, cfg.quantiles),
        "curvature_1pm": _summarize_distribution(
            _curvature_samples(velocity_arr, acceleration, cfg.min_heading_speed_mps),
            cfg.quantiles,
        ),
        "turning_angle_rad": _summarize_distribution(
            _turning_angle_samples(velocity_arr, cfg.min_heading_speed_mps),
            cfg.quantiles,
        ),
        "pairwise_distance_m": _summarize_distribution(
            _pairwise_distance_samples(positions_arr, cfg.pairwise_sample_stride),
            cfg.quantiles,
        ),
        "stop_frequency_hz": _summarize_distribution(
            _stop_frequency_samples(speed, dt_s, cfg.stop_speed_threshold_mps),
            cfg.quantiles,
        ),
        "stop_fraction": _summarize_distribution(
            _stop_fraction_samples(speed, cfg.stop_speed_threshold_mps),
            cfg.quantiles,
        ),
    }
    return report


def _empty_report(status: str) -> TrajectoryQualityReport:
    """Return an empty diagnostic report with stable metric keys."""

    empty = _summarize_distribution(np.array([], dtype=float), TrajectoryQualityConfig().quantiles)
    return {
        "status": status,
        "diagnostic_only": True,
        "thresholds_applied": False,
        "speed_mps": dict(empty),
        "acceleration_mps2": dict(empty),
        "curvature_1pm": dict(empty),
        "turning_angle_rad": dict(empty),
        "pairwise_distance_m": dict(empty),
        "stop_frequency_hz": dict(empty),
        "stop_fraction": dict(empty),
    }


def _as_trajectory_array(values: np.ndarray, *, name: str) -> np.ndarray:
    """Convert and validate a ``(T, K, 2)`` trajectory array.

    Returns
    -------
    np.ndarray
        Floating-point trajectory array.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"{name} must have shape (T, K, 2)")
    return arr


def _resolve_velocities(
    positions: np.ndarray,
    velocities: np.ndarray | None,
    dt_s: float,
) -> np.ndarray:
    """Return velocities aligned to finite-difference samples."""

    if velocities is None:
        return _finite_difference(positions, dt_s)

    arr = _as_trajectory_array(velocities, name="velocities")
    if arr.shape[1:] != positions.shape[1:]:
        raise ValueError("velocities must have the same pedestrian and coordinate dimensions")
    if arr.shape[0] not in {positions.shape[0], max(positions.shape[0] - 1, 0)}:
        raise ValueError("velocities must have T or T - 1 samples")
    return arr


def _finite_difference(values: np.ndarray, dt_s: float) -> np.ndarray:
    """Return first differences along time divided by ``dt_s``."""

    if values.shape[0] < 2:
        return np.empty((0, values.shape[1], values.shape[2]), dtype=float)
    return np.diff(values, axis=0) / dt_s


def _summarize_distribution(
    values: np.ndarray,
    quantiles: tuple[float, ...],
) -> DistributionSummary:
    """Summarize finite values using stable JSON-safe scalar fields.

    Returns
    -------
    DistributionSummary
        Count, invalid-count, extrema, moments, and configured quantiles.
    """

    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    invalid_count = int(flat.size - finite.size)
    summary: DistributionSummary = {
        "status": "empty" if finite.size == 0 else "ok",
        "count": int(finite.size),
        "invalid_count": invalid_count,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
    for quantile in quantiles:
        summary[f"q{_quantile_label(quantile)}"] = 0.0
    if finite.size == 0:
        return summary

    summary.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    for quantile in quantiles:
        summary[f"q{_quantile_label(quantile)}"] = float(np.quantile(finite, quantile))
    return summary


def _quantile_label(quantile: float) -> str:
    """Return compact percentile-like labels, e.g. ``0.05`` -> ``05``."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantiles must be in [0, 1]")
    return f"{quantile * 100:g}".replace(".", "p")


def _curvature_samples(
    velocities: np.ndarray,
    accelerations: np.ndarray,
    min_heading_speed_mps: float,
) -> np.ndarray:
    """Compute planar curvature samples from velocity and acceleration vectors.

    Returns
    -------
    np.ndarray
        Finite curvature samples in inverse meters.
    """

    sample_count = min(velocities.shape[0] - 1, accelerations.shape[0])
    if sample_count <= 0:
        return np.array([], dtype=float)
    v = velocities[:sample_count]
    a = accelerations[:sample_count]
    speed = np.linalg.norm(v, axis=2)
    cross = np.abs(v[:, :, 0] * a[:, :, 1] - v[:, :, 1] * a[:, :, 0])
    finite = np.isfinite(cross) & np.isfinite(speed) & (speed > min_heading_speed_mps)
    return cross[finite] / np.power(speed[finite], 3)


def _turning_angle_samples(velocities: np.ndarray, min_heading_speed_mps: float) -> np.ndarray:
    """Return wrapped heading deltas between consecutive velocity vectors."""

    if velocities.shape[0] < 2:
        return np.array([], dtype=float)
    prev = velocities[:-1]
    curr = velocities[1:]
    prev_speed = np.linalg.norm(prev, axis=2)
    curr_speed = np.linalg.norm(curr, axis=2)
    valid = (
        np.isfinite(prev).all(axis=2)
        & np.isfinite(curr).all(axis=2)
        & (prev_speed > min_heading_speed_mps)
        & (curr_speed > min_heading_speed_mps)
    )
    prev_valid = prev[valid]
    curr_valid = curr[valid]
    prev_heading = np.arctan2(prev_valid[:, 1], prev_valid[:, 0])
    curr_heading = np.arctan2(curr_valid[:, 1], curr_valid[:, 0])
    turn = np.abs(
        np.arctan2(np.sin(curr_heading - prev_heading), np.cos(curr_heading - prev_heading))
    )
    return turn


def _pairwise_distance_samples(positions: np.ndarray, sample_stride: int) -> np.ndarray:
    """Return finite pairwise pedestrian distances for sampled timesteps."""

    if positions.shape[1] < 2:
        return np.array([], dtype=float)
    samples: list[np.ndarray] = []
    for step_positions in positions[::sample_stride]:
        finite = step_positions[np.isfinite(step_positions).all(axis=1)]
        if finite.shape[0] < 2:
            continue
        deltas = finite[:, None, :] - finite[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        upper = np.triu_indices(finite.shape[0], k=1)
        samples.append(distances[upper])
    if not samples:
        return np.array([], dtype=float)
    return np.concatenate(samples)


def _stop_frequency_samples(
    speed: np.ndarray,
    dt_s: float,
    stop_speed_threshold_mps: float,
) -> np.ndarray:
    """Return per-pedestrian stopped-step counts divided by observed duration."""

    if speed.size == 0:
        return np.array([], dtype=float)
    finite = np.isfinite(speed)
    observed_steps = finite.sum(axis=0)
    duration_s = observed_steps.astype(float) * dt_s
    stopped_steps = (finite & (speed <= stop_speed_threshold_mps)).sum(axis=0)
    valid = duration_s > 0.0
    return stopped_steps[valid].astype(float) / duration_s[valid]


def _stop_fraction_samples(speed: np.ndarray, stop_speed_threshold_mps: float) -> np.ndarray:
    """Return per-pedestrian fraction of finite samples at or below stop speed."""

    if speed.size == 0:
        return np.array([], dtype=float)
    finite = np.isfinite(speed)
    observed_steps = finite.sum(axis=0)
    stopped_steps = (finite & (speed <= stop_speed_threshold_mps)).sum(axis=0)
    valid = observed_steps > 0
    return stopped_steps[valid].astype(float) / observed_steps[valid].astype(float)


def ensure_json_safe(report: dict[str, Any]) -> dict[str, Any]:
    """Return ``report`` after checking all leaf values are JSON-safe scalars."""

    for value in report.values():
        if isinstance(value, dict):
            ensure_json_safe(value)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"non JSON-safe value: {value!r}")
    return report
