"""Experimental reachable-velocity sampler derived from SocNav's bounded sampler.

The planner scores a short open-loop forecast; its soft costs are not a safety
certificate. Pedestrian velocities are held constant only in that forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, pi
from typing import Any, Generic, TypeVar

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi_closed
from robot_sf.planner.clearance_geometry import swept_pedestrian_clearance
from robot_sf.planner.socnav_base import SamplingPlannerAdapter
from robot_sf.planner.socnav_sampling_v2 import (
    _goal_direction,
    _ObstacleClearance,
    _pedestrian_world_velocities,
    _positive,
)
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.reverse_drive import bound_drive_settings


@dataclass(frozen=True)
class VelocitySamplerWeights:
    """Scoring parameters for the opt-in velocity sampler."""

    horizon_s: float = 2.4
    forecast_dt: float = 0.2
    pedestrian_weight: float = 1.0
    obstacle_weight: float = 1.6
    clearance_scale: float = 0.5
    progress_weight: float = 1.0
    heading_weight: float = 0.5
    speed_weight: float = 0.3
    switch_weight: float = 0.04
    speed_samples: int = 3
    turn_samples: int = 9

    def __post_init__(self) -> None:
        """Reject invalid costs or discretizations before a campaign starts."""
        for name in ("horizon_s", "forecast_dt", "clearance_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "pedestrian_weight",
            "obstacle_weight",
            "progress_weight",
            "heading_weight",
            "speed_weight",
            "switch_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.speed_samples < 2 or self.turn_samples < 2:
            raise ValueError("at least two speed and turn samples are required")


def reachable_commands(
    speed: float,
    turn: float,
    dt: float,
    drive: DifferentialDriveSettings,
    *,
    speed_cap: float,
    turn_cap: float,
    speed_samples: int,
    turn_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Cartesian grid of one-step reachable velocity endpoints.

    Returns:
        tuple[np.ndarray, np.ndarray]: Linear and angular samples in matching order.
    """
    assert drive.max_linear_decel is not None
    lo_v = max(0.0, speed - float(drive.max_linear_decel) * dt)
    hi_v = min(speed_cap, float(drive.max_linear_speed), speed + drive.max_linear_accel * dt)
    lo_w = max(-turn_cap, -drive.max_angular_speed, turn - drive.max_angular_accel * dt)
    hi_w = min(turn_cap, drive.max_angular_speed, turn + drive.max_angular_accel * dt)
    if lo_v > hi_v + 1e-6 or lo_w > hi_w + 1e-6:
        raise ValueError("observed drive state lies outside reachable command limits")
    v, w = np.meshgrid(
        np.linspace(lo_v, max(lo_v, hi_v), speed_samples),
        np.linspace(lo_w, max(lo_w, hi_w), turn_samples),
        indexing="ij",
    )
    return v.ravel(), w.ravel()


def _forecast(  # noqa: PLR0913
    position: np.ndarray,
    heading: float,
    speed0: float,
    turn0: float,
    speeds: np.ndarray,
    turns: np.ndarray,
    dt: float,
    horizon: float,
    forecast_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast one actual drive step, then hold each reachable command.

    Later samples use midpoint odometry like the plant. The final time may be
    shorter than the forecast interval so the requested horizon is respected.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Positions, headings, and sample times.
    """
    count = max(1, ceil(max(0.0, horizon - dt) / forecast_dt))
    steps = [dt] + [
        min(forecast_dt, max(0.0, horizon - dt - i * forecast_dt)) for i in range(count)
    ]
    steps = [step for step in steps if step > 1e-12]
    n = speeds.size
    positions = np.empty((n, len(steps) + 1, 2), dtype=float)
    headings = np.empty((n, len(steps) + 1), dtype=float)
    times = np.zeros(len(steps) + 1, dtype=float)
    positions[:, 0, :] = position
    headings[:, 0] = heading
    for i, step in enumerate(steps):
        average_v = (speed0 + speeds) * 0.5 if i == 0 else speeds
        average_w = (turn0 + turns) * 0.5 if i == 0 else turns
        next_heading = headings[:, i] + average_w * step
        midpoint = (headings[:, i] + next_heading) * 0.5
        positions[:, i + 1, 0] = positions[:, i, 0] + average_v * step * np.cos(midpoint)
        positions[:, i + 1, 1] = positions[:, i, 1] + average_v * step * np.sin(midpoint)
        headings[:, i + 1] = next_heading
        times[i + 1] = times[i] + step
    return positions, headings, times


def _risk(surface_clearance: np.ndarray, scale: float) -> np.ndarray:
    """Smooth, finite penalty for small or negative surface clearance.

    Returns:
        np.ndarray: Per-sample soft cost.
    """
    return np.logaddexp(0.0, np.clip(-surface_clearance / scale, -30.0, 30.0))


SamplerWeightsT = TypeVar("SamplerWeightsT", bound=VelocitySamplerWeights)


# Python 3.11 cannot parse PEP 695 type parameters.
class VelocitySamplerAdapter(SamplingPlannerAdapter, Generic[SamplerWeightsT]):  # noqa: UP046
    """SocNav-derived sampler whose outputs fit the live drive's one-step window."""

    def __init__(self, *, config: Any, weights: SamplerWeightsT) -> None:
        """Initialize the sampler and its per-episode counters."""
        if not config.sampling_pedestrian_prediction:
            raise ValueError("velocity sampler requires pedestrian velocity prediction")
        super().__init__(config=config, use_upstream=False)
        self.weights = weights
        self.reset()

    def reset(self, *, seed: int | None = None) -> None:
        """Clear passing-side preference between episodes."""
        del seed
        self._last_side = 0
        self._steps = 0
        self._side_switches = 0
        self._pedestrian_steps = 0
        self._static_grid_steps = 0
        self._path_field_steps = 0

    def bind_env(self, env: Any) -> None:
        """Bind the actual forward-only drive and the baseline's static map."""
        super().bind_env(env)
        drive = bound_drive_settings(env)
        if (
            not isinstance(drive, DifferentialDriveSettings)
            or drive.allow_backwards
            or drive.limited_reverse
        ):
            raise ValueError("velocity sampler requires a bound forward-only differential drive")
        self._velocity_drive = drive

    def plan(self, observation: dict) -> tuple[float, float]:  # noqa: PLR0915
        """Choose the lowest soft-cost reachable linear and angular velocities.

        Returns:
            tuple[float, float]: Reachable linear and angular velocity targets.
        """
        drive = getattr(self, "_velocity_drive", None)
        if drive is None:
            raise RuntimeError("velocity sampler requires bound drive limits")
        robot, goal, pedestrians = self._socnav_fields(observation)
        position = self._as_1d_float(robot["position"], pad=2)[:2]
        heading = float(self._as_1d_float(robot["heading"], pad=1)[0])
        target = self._as_1d_float(goal["current"], pad=2)[:2]
        current = self._as_1d_float(robot.get("speed", [0.0]), pad=1)
        speed0 = float(current[0])
        turn0 = (
            float(current[1])
            if current.size > 1
            else float(self._as_1d_float(robot.get("angular_velocity", [0.0]), pad=1)[0])
        )
        dt = self._simulation_timestep(observation)
        if not np.isfinite([*position, heading, speed0, turn0, dt]).all() or dt <= 0:
            raise ValueError("velocity sampler needs finite drive state and timestep")
        weights = self.weights
        speeds, turns = reachable_commands(
            speed0,
            turn0,
            dt,
            drive,
            speed_cap=min(float(self.config.max_linear_speed), drive.max_linear_speed),
            turn_cap=min(float(self.config.max_angular_speed), drive.max_angular_speed),
            speed_samples=weights.speed_samples,
            turn_samples=weights.turn_samples,
        )
        robot_radius = _positive(robot.get("radius")) or drive.radius
        ped_radius = _positive(pedestrians.get("radius") if pedestrians else None) or 0.0
        raw = pedestrians.get("positions", []) if pedestrians else []
        peds = np.asarray(raw if raw is not None else [], dtype=float).reshape(-1, 2)
        ped_count = (
            int(self._as_1d_float(pedestrians.get("count", [0]), pad=1)[0]) if pedestrians else 0
        )
        if ped_count < 0 or ped_count > len(peds) or not np.isfinite(peds[:ped_count]).all():
            raise ValueError("pedestrian position observations are incomplete")
        peds = peds[:ped_count]
        raw_velocities = pedestrians.get("velocities") if pedestrians else None
        if len(peds):
            if raw_velocities is None:
                raise ValueError("pedestrian velocity observations are required")
            observed = np.asarray(raw_velocities, dtype=float).reshape(-1, 2)
            if len(observed) < len(peds) or not np.isfinite(observed[: len(peds)]).all():
                raise ValueError("pedestrian velocity observations are incomplete")
        ped_vel = _pedestrian_world_velocities(self.config, pedestrians, peds, heading)
        direction, path_used = _goal_direction(self, position, target, robot_radius)
        target_angle = atan2(direction[1], direction[0])
        points, headings, times = _forecast(
            position,
            heading,
            speed0,
            turn0,
            speeds,
            turns,
            dt,
            weights.horizon_s,
            weights.forecast_dt,
        )
        ped_clear = swept_pedestrian_clearance(
            points, times, peds, ped_vel, robot_radius + ped_radius
        )
        clearance = _ObstacleClearance(self, observation)
        # Sample both segment ends and midpoints for static-grid swept clearance.
        midpoints = (points[:, :-1] + points[:, 1:]) * 0.5
        samples = np.concatenate((points[:, 1:], midpoints), axis=1)
        wall_clear = clearance(samples.reshape(-1, 2)).reshape(len(speeds), -1) - robot_radius
        ped_cost = _risk(ped_clear, weights.clearance_scale)
        wall_cost = _risk(wall_clear, weights.clearance_scale)
        ped_cost = 0.5 * (ped_cost.max(axis=1) + ped_cost.mean(axis=1))
        wall_cost = 0.5 * (wall_cost.max(axis=1) + wall_cost.mean(axis=1))
        progress = (points[:, -1] - position) @ direction
        heading_error = np.arctan2(
            np.sin(headings[:, -1] - target_angle), np.cos(headings[:, -1] - target_angle)
        )
        desired_turn = np.clip(
            float(self.config.angular_gain) * wrap_angle_pi_closed(target_angle - heading),
            -drive.max_angular_speed,
            drive.max_angular_speed,
        )
        sides = np.sign(turns - desired_turn).astype(int)
        nearby = bool(len(peds) and np.min(np.linalg.norm(peds - position, axis=1)) < 5.0)
        switch = (
            (sides != 0) & (self._last_side != 0) & (sides != self._last_side)
            if nearby
            else np.zeros(len(speeds), dtype=bool)
        )
        costs = (
            weights.pedestrian_weight * ped_cost
            + weights.obstacle_weight * wall_cost
            - weights.progress_weight * progress / weights.horizon_s
            + weights.heading_weight * np.abs(heading_error) / pi
            - weights.speed_weight * speeds
            + weights.switch_weight * switch
        )
        chosen = int(np.argmin(costs))
        side = int(sides[chosen])
        if nearby and side:
            self._side_switches += int(self._last_side not in (0, side))
            self._last_side = side
        elif not nearby:
            self._last_side = 0
        self._steps += 1
        self._pedestrian_steps += int(bool(len(peds)))
        self._static_grid_steps += int(clearance.available)
        self._path_field_steps += int(path_used)
        self._last_velocity_decision = {
            "cost": float(costs[chosen]),
            "path_distance": bool(path_used),
        }
        return float(speeds[chosen]), float(turns[chosen])

    def diagnostics(self) -> dict[str, Any]:
        """Expose the implementation identity and input-use counters.

        Returns:
            dict[str, Any]: Adapter identity and per-episode counters.
        """
        return {
            "planner_type": "VelocitySamplerAdapter",
            "implementation_mode": "reachable_velocity_sampling",
            "fallback_triggered": False,
            "steps": self._steps,
            "side_switches": self._side_switches,
            "pedestrian_steps": self._pedestrian_steps,
            "static_grid_steps": self._static_grid_steps,
            "path_field_steps": self._path_field_steps,
        }
