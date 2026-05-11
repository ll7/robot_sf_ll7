"""Minimal multi-AMV benchmark helpers.

This module provides the first narrow multi-robot benchmark slice: scenario
settings parsing and inter-robot metric computation. It intentionally avoids a
fleet-optimization abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiAmvSettings:
    """Scenario-level settings for the minimal multi-AMV benchmark slice."""

    num_robots: int = 1
    near_miss_distance_m: float = 1.0
    collision_distance_m: float = 0.4
    deadlock_speed_mps: float = 0.05
    deadlock_window_steps: int = 10


def multi_amv_settings_from_scenario(scenario: dict[str, Any]) -> MultiAmvSettings:
    """Parse the optional ``multi_amv`` scenario block.

    Returns:
        MultiAmvSettings: Validated settings for the minimal multi-AMV slice.
    """
    raw = scenario.get("multi_amv")
    if raw is None:
        return MultiAmvSettings()
    if not isinstance(raw, dict):
        raise ValueError("multi_amv must be a mapping.")
    allowed = {
        "num_robots",
        "near_miss_distance_m",
        "collision_distance_m",
        "deadlock_speed_mps",
        "deadlock_window_steps",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"multi_amv contains unknown keys: {', '.join(unknown)}.")
    settings = MultiAmvSettings(
        num_robots=int(raw.get("num_robots", 1)),
        near_miss_distance_m=float(raw.get("near_miss_distance_m", 1.0)),
        collision_distance_m=float(raw.get("collision_distance_m", 0.4)),
        deadlock_speed_mps=float(raw.get("deadlock_speed_mps", 0.05)),
        deadlock_window_steps=int(raw.get("deadlock_window_steps", 10)),
    )
    if settings.num_robots < 1:
        raise ValueError("multi_amv.num_robots must be >= 1.")
    if settings.collision_distance_m <= 0.0:
        raise ValueError("multi_amv.collision_distance_m must be > 0.")
    if settings.near_miss_distance_m <= settings.collision_distance_m:
        raise ValueError("multi_amv.near_miss_distance_m must be > collision_distance_m.")
    if settings.deadlock_speed_mps < 0.0:
        raise ValueError("multi_amv.deadlock_speed_mps must be >= 0.")
    if settings.deadlock_window_steps < 1:
        raise ValueError("multi_amv.deadlock_window_steps must be >= 1.")
    return settings


def inter_robot_metrics(
    robot_positions: np.ndarray,
    *,
    dt: float,
    settings: MultiAmvSettings,
) -> dict[str, float | bool]:
    """Compute minimal inter-robot safety/deadlock metrics from trajectories.

    Args:
        robot_positions: Array shaped ``(steps, robots, 2)``.
        dt: Simulation step duration in seconds.
        settings: Multi-AMV metric thresholds.

    Returns:
        dict[str, float | bool]: JSON-safe inter-robot metrics where collision/near-miss
        events count contiguous encounter runs, and deadlock detection is fleet-wide for this
        first slice.
    """
    positions = np.asarray(robot_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("robot_positions must have shape (steps, robots, 2).")
    steps, robots, _ = positions.shape
    pair_count = (robots * (robots - 1)) // 2
    if robots < 2:
        return {
            "robot_count": float(robots),
            "pair_count": 0.0,
            "min_inter_robot_distance_m": float("nan"),
            "inter_robot_collision_events": 0.0,
            "inter_robot_near_miss_events": 0.0,
            "deadlock_steps": 0.0,
            "deadlock_detected": False,
        }
    if steps == 0:
        return {
            "robot_count": float(robots),
            "pair_count": float(pair_count),
            "min_inter_robot_distance_m": float("nan"),
            "inter_robot_collision_events": 0.0,
            "inter_robot_near_miss_events": 0.0,
            "deadlock_steps": 0.0,
            "deadlock_detected": False,
        }

    pair_distances = []
    for i in range(robots):
        for j in range(i + 1, robots):
            pair_distances.append(np.linalg.norm(positions[:, i, :] - positions[:, j, :], axis=1))
    distances = np.stack(pair_distances, axis=1)
    min_per_step = np.min(distances, axis=1)
    collision_events = _count_true_runs(min_per_step < settings.collision_distance_m)
    near_miss_mask = (min_per_step >= settings.collision_distance_m) & (
        min_per_step < settings.near_miss_distance_m
    )
    near_miss_events = _count_true_runs(near_miss_mask)

    deadlock_steps = 0
    deadlock_detected = False
    if steps >= 2:
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=2) / max(float(dt), 1e-9)
        # This minimal first slice only marks deadlock when the whole fleet stays slow.
        all_slow = np.all(speeds <= settings.deadlock_speed_mps, axis=1)
        deadlock_steps = int(np.count_nonzero(all_slow))
        deadlock_detected = _has_consecutive_true(all_slow, settings.deadlock_window_steps)

    return {
        "robot_count": float(robots),
        "pair_count": float(pair_count),
        "min_inter_robot_distance_m": float(np.min(min_per_step)),
        "inter_robot_collision_events": float(collision_events),
        "inter_robot_near_miss_events": float(near_miss_events),
        "deadlock_steps": float(deadlock_steps),
        "deadlock_detected": bool(deadlock_detected),
    }


def _count_true_runs(values: np.ndarray) -> int:
    """Count contiguous true runs in a boolean sequence.

    Returns:
        Number of distinct true runs in the sequence.
    """

    mask = np.asarray(values, dtype=bool)
    if mask.size == 0:
        return 0
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    starts = np.logical_not(padded[:-1]) & padded[1:]
    return int(np.count_nonzero(starts))


def _has_consecutive_true(values: np.ndarray, window: int) -> bool:
    """Return whether a boolean sequence contains ``window`` consecutive true values."""
    run = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        if run >= window:
            return True
    return False
