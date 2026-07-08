"""Step-level SNQI proxy metrics used by ``RobotEnv`` reward metadata."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS

_DEFAULT_COLLISION_DIST = 0.25
_DEFAULT_NEAR_MISS_DIST = 0.50
_DEFAULT_COMFORT_FORCE_THRESHOLD = 2.0
_DEFAULT_PED_RADIUS = 0.4
_SNQI_THRESHOLD_CACHE: tuple[float, float, float] | None = None


@dataclass(slots=True)
class StepSNQIProxyState:
    """Running state for SNQI step-level proxy computation.

    The benchmark ``jerk_mean`` is an episodic quantity. For per-step reward metadata this
    state maintains a running proxy from finite differences of robot position.
    """

    prev_robot_pos: np.ndarray | None = None
    prev_robot_vel: np.ndarray | None = None
    prev_robot_acc: np.ndarray | None = None
    jerk_sum: float = 0.0
    jerk_count: int = 0


class StepSNQIProxy:
    """Own per-episode SNQI proxy state for ``RobotEnv`` step metadata."""

    def __init__(self) -> None:
        """Initialize a fresh proxy state."""
        self.state = StepSNQIProxyState()

    def prime(self, simulator: Any) -> None:
        """Reset state and seed previous robot position from ``simulator`` when available."""
        self.state = StepSNQIProxyState()
        robot_pose = first_robot_pose(getattr(simulator, "robot_poses", []))
        if robot_pose is not None:
            self.state.prev_robot_pos = extract_robot_xy(robot_pose)

    def compute_step_metrics(
        self,
        simulator: Any,
        *,
        dt: float,
        ped_positions_override: Any | None = None,
    ) -> dict[str, float]:
        """Compute SNQI-aligned proxy metrics and update running state.

        Returns:
            dict[str, float]: Step-level SNQI proxy metadata.
        """
        return compute_snqi_step_proxies(
            simulator=simulator,
            dt=dt,
            proxy_state=self.state,
            ped_positions_override=ped_positions_override,
        )


def extract_robot_xy(robot_pose: Any) -> np.ndarray:
    """Extract a 2D robot position from heterogeneous simulator pose payloads.

    Args:
        robot_pose: Backend-dependent pose object; typically ``((x, y), theta)``.

    Returns:
        np.ndarray: ``[x, y]`` position vector, or zeros for invalid payloads.
    """
    if robot_pose is None:
        return np.zeros(2, dtype=float)
    source = robot_pose[0] if isinstance(robot_pose, (tuple, list)) else robot_pose
    try:
        flattened = np.asarray(source, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(2, dtype=float)
    if flattened.size >= 2:
        return flattened[:2]
    return np.zeros(2, dtype=float)


def first_robot_pose(robot_poses: Any) -> Any | None:
    """Return the first robot pose from common sequence or array payloads."""
    if isinstance(robot_poses, (list, tuple)):
        return robot_poses[0] if robot_poses else None
    if isinstance(robot_poses, np.ndarray) and robot_poses.size > 0:
        return robot_poses if robot_poses.ndim == 1 else robot_poses[0]
    return None


def coerce_xy_rows(data: Any) -> np.ndarray:
    """Coerce simulator arrays into ``(N, 2)`` float rows.

    Returns:
        np.ndarray: Array with shape ``(N, 2)``; invalid layouts produce an empty array.
    """
    if data is None:
        return np.zeros((0, 2), dtype=float)
    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError, KeyError, IndexError):
        return np.zeros((0, 2), dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 2) if values.size % 2 == 0 else np.zeros((0, 2), dtype=float)
    if values.ndim == 2 and values.shape[-1] >= 2:
        return values[:, :2]
    return np.zeros((0, 2), dtype=float)


def resolve_snqi_thresholds() -> tuple[float, float, float]:
    """Resolve SNQI threshold constants lazily to avoid import cycles.

    Returns:
        tuple[float, float, float]: ``(collision_dist, near_miss_dist, comfort_force_threshold)``.
    """
    global _SNQI_THRESHOLD_CACHE
    if _SNQI_THRESHOLD_CACHE is not None:
        return _SNQI_THRESHOLD_CACHE
    try:
        constants_module = importlib.import_module("robot_sf.benchmark.constants")
        collision_dist = float(constants_module.COLLISION_DIST)
        near_miss_dist = float(constants_module.NEAR_MISS_DIST)
        comfort_force_threshold = float(constants_module.COMFORT_FORCE_THRESHOLD)

        _SNQI_THRESHOLD_CACHE = (
            collision_dist,
            near_miss_dist,
            comfort_force_threshold,
        )
    except (ImportError, ModuleNotFoundError, AttributeError):
        _SNQI_THRESHOLD_CACHE = (
            _DEFAULT_COLLISION_DIST,
            _DEFAULT_NEAR_MISS_DIST,
            _DEFAULT_COMFORT_FORCE_THRESHOLD,
        )
    return _SNQI_THRESHOLD_CACHE


def _coerce_positive_float(value: Any, default: float) -> float:
    """Return ``value`` as a positive finite float, otherwise ``default``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) and numeric > 0.0 else default


def _resolve_robot_radius(simulator: Any) -> float:
    """Resolve robot footprint radius from simulator metadata when available.

    Returns:
        Positive robot radius in meters.
    """
    direct = getattr(simulator, "robot_radius", None)
    if direct is not None:
        return _coerce_positive_float(direct, DEFAULT_ROBOT_RADIUS)
    robots = getattr(simulator, "robots", None)
    if robots:
        config = getattr(robots[0], "config", None)
        return _coerce_positive_float(getattr(config, "radius", None), DEFAULT_ROBOT_RADIUS)
    return DEFAULT_ROBOT_RADIUS


def _resolve_ped_radius(simulator: Any) -> float:
    """Resolve pedestrian footprint radius from simulator metadata when available.

    Returns:
        Positive pedestrian radius in meters.
    """
    direct = getattr(simulator, "ped_radius", None)
    if direct is not None:
        return _coerce_positive_float(direct, _DEFAULT_PED_RADIUS)
    config = getattr(simulator, "config", None)
    return _coerce_positive_float(getattr(config, "ped_radius", None), _DEFAULT_PED_RADIUS)


def compute_snqi_step_proxies(
    *,
    simulator: Any,
    dt: float,
    proxy_state: StepSNQIProxyState,
    ped_positions_override: Any | None = None,
) -> dict[str, float]:
    """Compute per-step SNQI proxy terms from simulator state.

    The generated values are intentionally lightweight proxies:
    - ``near_misses``: exact per-step threshold event for robot-ped min distance.
    - ``force_exceed_events``: exact per-step count above comfort threshold.
    - ``comfort_exposure``: per-step normalized force exceed ratio.
    - ``jerk_mean``: running mean of finite-difference jerk magnitude proxy.

    Args:
        simulator: Active simulator backend for the environment.
        dt: Simulation timestep in seconds.
        proxy_state: Running state used to estimate jerk proxy over steps.
        ped_positions_override: Optional step-local pedestrian position snapshot. When provided,
            this avoids re-reading ``simulator.ped_pos``.

    Returns:
        dict[str, float]: Step-level SNQI-aligned metadata terms.
    """
    d_coll, d_near, comfort_force_threshold = resolve_snqi_thresholds()
    robot_pos = extract_robot_xy(first_robot_pose(getattr(simulator, "robot_poses", [])))

    ped_pos_source = (
        ped_positions_override
        if ped_positions_override is not None
        else getattr(simulator, "ped_pos", np.zeros((0, 2), dtype=float))
    )
    ped_pos = coerce_xy_rows(ped_pos_source)

    near_misses = 0.0
    min_distance = float("nan")
    min_clearance = float("nan")
    center_distance_near_miss_diagnostic = 0.0
    if ped_pos.size > 0:
        deltas = ped_pos[:, :2] - robot_pos
        min_distance = float(np.linalg.norm(deltas, axis=1).min())
        radius_sum = _resolve_robot_radius(simulator) + _resolve_ped_radius(simulator)
        min_clearance = min_distance - radius_sum
        near_misses = 1.0 if 0.0 <= min_clearance < d_near else 0.0
        center_distance_near_miss_diagnostic = 1.0 if d_coll <= min_distance < d_near else 0.0

    ped_forces = coerce_xy_rows(
        getattr(simulator, "last_ped_forces", np.zeros((0, 2), dtype=float))
    )
    force_magnitudes = np.linalg.norm(ped_forces, axis=1) if ped_forces.size > 0 else np.zeros((0,))
    force_exceed_events = float(np.count_nonzero(force_magnitudes > comfort_force_threshold))
    # Backends can transiently report different lengths for positions/forces between ticks.
    ped_count = max(int(ped_pos.shape[0]), int(force_magnitudes.shape[0]))
    comfort_exposure = force_exceed_events / float(max(1, ped_count))

    if dt > 1e-9:
        if proxy_state.prev_robot_pos is not None:
            vel = (robot_pos - proxy_state.prev_robot_pos) / dt
            if proxy_state.prev_robot_vel is not None:
                acc = (vel - proxy_state.prev_robot_vel) / dt
                if proxy_state.prev_robot_acc is not None:
                    jerk = (acc - proxy_state.prev_robot_acc) / dt
                    jerk_mag = math.hypot(float(jerk[0]), float(jerk[1]))
                    if np.isfinite(jerk_mag):
                        proxy_state.jerk_sum += jerk_mag
                        proxy_state.jerk_count += 1
                proxy_state.prev_robot_acc = acc
            proxy_state.prev_robot_vel = vel
        proxy_state.prev_robot_pos = robot_pos
    jerk_mean = (
        proxy_state.jerk_sum / float(proxy_state.jerk_count) if proxy_state.jerk_count > 0 else 0.0
    )

    return {
        "near_misses": float(near_misses),
        "min_distance": float(min_distance),
        "min_clearance": float(min_clearance),
        "center_distance_near_miss_diagnostic": float(center_distance_near_miss_diagnostic),
        "force_exceed_events": float(force_exceed_events),
        "comfort_exposure": float(comfort_exposure),
        "jerk_mean": float(jerk_mean),
    }
