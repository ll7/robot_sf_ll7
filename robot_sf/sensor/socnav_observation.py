"""
SocNavBench-inspired structured observation builder.

Provides a lightweight, in-process equivalent of SocNavBench's sense payload so
planners can be reused without socket I/O.
"""

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
from gymnasium import spaces
from loguru import logger

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.planner.predictive_foresight import (
    PredictiveForesightEncoder,
    predictive_foresight_config_from_source,
    predictive_foresight_spaces,
)
from robot_sf.sim.simulator import Simulator

DEFAULT_MAX_PEDS = 64
"""Default upper bound for structured SocNav observations when config doesn't set max_total_pedestrians."""

SOCNAV_POSITION_CAP_M = 50.0
"""Global cap for SocNav position-like observations to keep bounds consistent."""


def socnav_observation_space(
    map_def: MapDefinition,
    env_config: RobotSimulationConfig,
    max_pedestrians: int,
) -> spaces.Dict:
    """
    Create a Gym ``spaces.Dict`` mirroring a SocNavBench-style sim state.

    Returns:
        spaces.Dict: Structured observation specification for the SocNav mode.
    """
    pos_cap = np.array([SOCNAV_POSITION_CAP_M, SOCNAV_POSITION_CAP_M], dtype=np.float32)
    pos_low = np.array([0.0, 0.0], dtype=np.float32)
    pos_high = pos_cap
    heading_low = np.array([-pi], dtype=np.float32)
    heading_high = np.array([pi], dtype=np.float32)
    # Extract realistic speed limits from robot configuration
    if hasattr(env_config.robot_config, "max_linear_speed"):
        max_speed = env_config.robot_config.max_linear_speed
    else:
        max_speed = 2.0  # Conservative fallback for unknown robot types
    speed_bounds = np.array([max_speed, max_speed], dtype=np.float32)
    radius_bounds = np.array([0.0], dtype=np.float32)

    ped_positions_high = np.broadcast_to(pos_high, (max_pedestrians, 2)).astype(np.float32)
    ped_positions_low = np.broadcast_to(pos_low, (max_pedestrians, 2)).astype(np.float32)
    ped_vel_low = np.broadcast_to(-speed_bounds, (max_pedestrians, 2)).astype(np.float32)
    ped_vel_high = np.broadcast_to(speed_bounds, (max_pedestrians, 2)).astype(np.float32)

    return spaces.Dict(
        {
            "robot": spaces.Dict(
                {
                    "position": spaces.Box(low=pos_low, high=pos_high, dtype=np.float32),
                    "heading": spaces.Box(low=heading_low, high=heading_high, dtype=np.float32),
                    "speed": spaces.Box(
                        low=-speed_bounds,
                        high=speed_bounds,
                        dtype=np.float32,
                    ),
                    "velocity_xy": spaces.Box(
                        low=-speed_bounds,
                        high=speed_bounds,
                        dtype=np.float32,
                    ),
                    "angular_velocity": spaces.Box(
                        low=np.array([-np.finfo(np.float32).max], dtype=np.float32),
                        high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "radius": spaces.Box(
                        low=radius_bounds,
                        high=np.array([SOCNAV_POSITION_CAP_M], dtype=np.float32),
                        dtype=np.float32,
                    ),
                },
            ),
            "goal": spaces.Dict(
                {
                    "current": spaces.Box(low=pos_low, high=pos_high, dtype=np.float32),
                    "next": spaces.Box(low=pos_low, high=pos_high, dtype=np.float32),
                },
            ),
            "pedestrians": spaces.Dict(
                {
                    "positions": spaces.Box(
                        low=ped_positions_low,
                        high=ped_positions_high,
                        dtype=np.float32,
                    ),
                    "velocities": spaces.Box(
                        low=ped_vel_low,
                        high=ped_vel_high,
                        dtype=np.float32,
                    ),
                    "radius": spaces.Box(
                        low=radius_bounds,
                        high=np.array([SOCNAV_POSITION_CAP_M], dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "count": spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([float(max_pedestrians)], dtype=np.float32),
                        dtype=np.float32,
                    ),
                },
            ),
            "map": spaces.Dict(
                {
                    "size": spaces.Box(
                        low=pos_low,
                        high=pos_cap,
                        dtype=np.float32,
                    ),
                },
            ),
            "sim": spaces.Dict(
                {
                    "timestep": spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                        dtype=np.float32,
                    ),
                },
            ),
            **(
                {
                    "predictive": predictive_foresight_spaces(
                        predictive_foresight_config_from_source(
                            env_config,
                            default_max_agents=max_pedestrians,
                        )
                    )
                }
                if getattr(env_config, "predictive_foresight_enabled", False)
                else {}
            ),
        }
    )


@dataclass
class SocNavObservationFusion:
    """Structured observation builder used when ``ObservationMode.SOCNAV_STRUCT`` is enabled."""

    simulator: Simulator
    env_config: RobotSimulationConfig
    max_pedestrians: int
    robot_index: int = 0
    truncation_warned: bool = False
    _predictive_foresight: PredictiveForesightEncoder | None = None

    def __post_init__(self) -> None:
        """Initialize optional predictive foresight encoder."""
        if bool(getattr(self.env_config, "predictive_foresight_enabled", False)):
            self._predictive_foresight = PredictiveForesightEncoder(
                predictive_foresight_config_from_source(
                    self.env_config,
                    default_max_agents=self.max_pedestrians,
                )
            )

    def reset_cache(self) -> None:
        """No-op to match the SensorFusion interface."""
        return None

    def _robot_velocity_xy(self, wrapped_heading: float) -> np.ndarray:
        """Return the robot world-frame planar velocity for the structured observation."""
        robot = self.simulator.robots[self.robot_index]
        velocity_xy = getattr(getattr(robot, "state", None), "velocity_xy", None)
        if velocity_xy is not None:
            arr = np.asarray(velocity_xy, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return arr[:2]

        current_speed = np.asarray(robot.current_speed, dtype=np.float32).reshape(-1)
        linear_speed = float(current_speed[0]) if current_speed.size > 0 else 0.0
        return np.array(
            [
                linear_speed * float(np.cos(wrapped_heading)),
                linear_speed * float(np.sin(wrapped_heading)),
            ],
            dtype=np.float32,
        )

    def next_obs(self) -> dict[str, Any]:
        """Return the latest structured observation aligned to the declared space."""
        ped_positions = np.asarray(self.simulator.ped_pos, dtype=np.float32)
        try:
            ped_velocities = np.asarray(self.simulator.ped_vel, dtype=np.float32)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive fallback
            ped_velocities = np.zeros_like(ped_positions, dtype=np.float32)
        total_peds = ped_positions.shape[0]
        if total_peds > self.max_pedestrians and not self.truncation_warned:
            logger.warning(
                "SocNav structured obs truncating pedestrians: seen={}, max_pedestrians={}. "
                "Increase the configured max_pedestrians (or SimulationSettings.max_total_pedestrians) to avoid data loss.",
                total_peds,
                self.max_pedestrians,
            )
            self.truncation_warned = True

        robot_pose = self.simulator.robots[self.robot_index].pose
        robot_pos = np.asarray(robot_pose[0], dtype=np.float32)
        # Order pedestrians by distance to robot (closest-first)
        if ped_positions.size > 0:
            rel = ped_positions - robot_pos
            dists = np.linalg.norm(rel, axis=1)
            order = np.argsort(dists)
            ped_positions = ped_positions[order]
            ped_velocities = ped_velocities[order]

        ped_positions = ped_positions[: self.max_pedestrians]
        ped_velocities = ped_velocities[: self.max_pedestrians]
        padded = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        padded_vel = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        if ped_positions.size > 0:
            padded[: ped_positions.shape[0]] = ped_positions
        if ped_velocities.size > 0:
            # Convert pedestrian velocities to ego frame (rotate by -heading)
            heading = float(robot_pose[1])
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            vx = ped_velocities[:, 0]
            vy = ped_velocities[:, 1]
            ego_vx = cos_h * vx + sin_h * vy
            ego_vy = -sin_h * vx + cos_h * vy
            padded_vel[: ped_velocities.shape[0]] = np.stack([ego_vx, ego_vy], axis=1)

        goal = np.asarray(self.simulator.goal_pos[self.robot_index], dtype=np.float32)
        next_goal = self.simulator.next_goal_pos[self.robot_index]
        next_goal_arr = (
            np.asarray(next_goal, dtype=np.float32)
            if next_goal is not None
            else np.zeros(2, dtype=np.float32)
        )

        def _clip_positions(values: np.ndarray) -> np.ndarray:
            return np.clip(values, 0.0, SOCNAV_POSITION_CAP_M)

        robot_pos_clipped = _clip_positions(robot_pos)
        goal_clipped = _clip_positions(goal)
        next_goal_clipped = _clip_positions(next_goal_arr)
        padded = _clip_positions(padded)
        map_size = np.array(
            [self.simulator.map_def.width, self.simulator.map_def.height],
            dtype=np.float32,
        )
        map_size = np.minimum(map_size, SOCNAV_POSITION_CAP_M)

        # Wrap heading to [-pi, pi] to stay within declared observation bounds
        wrapped_heading = ((robot_pose[1] + np.pi) % (2 * np.pi)) - np.pi
        robot_speed = np.asarray(
            self.simulator.robots[self.robot_index].current_speed, dtype=np.float32
        )
        robot_velocity_xy = self._robot_velocity_xy(wrapped_heading)
        obs = {
            "robot": {
                "position": robot_pos_clipped,
                "heading": np.array([wrapped_heading], dtype=np.float32),
                "speed": robot_speed,
                "velocity_xy": robot_velocity_xy,
                "angular_velocity": np.array(
                    [float(robot_speed[1]) if robot_speed.size > 1 else 0.0],
                    dtype=np.float32,
                ),
                "radius": np.array(
                    [self.simulator.robots[self.robot_index].config.radius], dtype=np.float32
                ),
            },
            "goal": {
                "current": goal_clipped,
                "next": next_goal_clipped,
            },
            "pedestrians": {
                "positions": padded,
                "velocities": padded_vel,
                "radius": np.array(
                    [min(self.env_config.sim_config.ped_radius, SOCNAV_POSITION_CAP_M)],
                    dtype=np.float32,
                ),
                "count": np.array(
                    [float(min(len(ped_positions), self.max_pedestrians))], dtype=np.float32
                ),
            },
            "map": {
                "size": map_size,
            },
            "sim": {
                "timestep": np.array(
                    [self.simulator.config.time_per_step_in_secs], dtype=np.float32
                ),
            },
        }
        if self._predictive_foresight is not None:
            obs["predictive"] = self._predictive_foresight.encode(obs)
        return obs
