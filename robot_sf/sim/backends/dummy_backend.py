"""Dummy simulator backend for testing and smoke tests.

Provides a small simulator surface that matches the robot-environment contract
well enough for backend-selection demos and CI smoke execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.nav.navigation import RouteNavigator, sample_route

if TYPE_CHECKING:
    from robot_sf.common.types import RobotPose
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.nav.map_config import MapDefinition


class DummySimulator:
    """Minimal simulator with enough metadata for RobotEnv smoke runs."""

    def __init__(
        self,
        *,
        map_def: MapDefinition,
        seed: int = 0,
        step_dt: float = 0.1,
        goal_proximity_threshold: float = 1.0,
    ):
        """Initialize the deterministic dummy simulator.

        Args:
            map_def: Map metadata used by RobotEnv collision and sensor helpers.
            seed: Seed used to reset the dummy random-number generator.
            step_dt: Fixed dummy timestep in seconds.
            goal_proximity_threshold: Goal completion tolerance.
        """
        self.map_def = map_def
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_dt = float(step_dt)
        self.goal_proximity_threshold = float(goal_proximity_threshold)
        self.timestep = 0
        self.robots = [_MockRobot()]
        self.robot_navs = [RouteNavigator(proximity_threshold=self.goal_proximity_threshold)]
        self._ped_pos = np.empty((0, 2), dtype=float)
        self.ped_radii = np.empty((0,), dtype=float)
        self.reset_state()

    def reset_state(self):
        """Reset simulator to initial state."""
        self.timestep = 0
        self.rng = np.random.default_rng(self.seed)
        route = sample_route(self.map_def, None)
        navigator = self.robot_navs[0]
        navigator.new_route(route[1:], start_pos=route[0])
        self.robots[0].reset_state((route[0], navigator.initial_orientation))

    def step_once(self, actions):
        """Advance one timestep using a simple unicycle-style pose update."""
        self.timestep += 1
        action = actions[0] if actions else (0.0, 0.0)
        self.robots[0].apply_action(action, dt=self.step_dt)
        self.robot_navs[0].update_position(self.robots[0].pose[0])

    @property
    def robot_poses(self):
        """Return current robot poses."""
        return [robot.pose for robot in self.robots]

    @property
    def robot_pos(self):
        """Return current robot positions."""
        return [robot.pose[0] for robot in self.robots]

    @property
    def ped_pos(self):
        """Return pedestrian positions."""
        return self._ped_pos

    @property
    def goal_pos(self):
        """Return current goal waypoints."""
        return [navigator.current_waypoint for navigator in self.robot_navs]

    @property
    def next_goal_pos(self):
        """Return next goal waypoints when present."""
        return [navigator.next_waypoint for navigator in self.robot_navs]

    def get_obstacle_lines(self) -> np.ndarray:
        """Return flat obstacle segments for occupancy checks."""
        obstacle_lines = np.asarray(self.map_def.obstacles_pysf, dtype=float)
        return obstacle_lines.reshape((-1, 4))


@dataclass
class _MockRobot:
    """Mock robot for dummy simulator."""

    pose: RobotPose = ((0.0, 0.0), 0.0)
    current_speed: tuple[float, float] = (0.0, 0.0)

    def parse_action(self, action):
        """Pass-through action parsing.

        Returns:
            Two-element ``(linear, angular)`` tuple.
        """
        values = np.asarray(action, dtype=float).reshape(-1)
        linear = float(values[0]) if values.size > 0 else 0.0
        angular = float(values[1]) if values.size > 1 else 0.0
        return linear, angular

    def reset_state(self, pose: RobotPose) -> None:
        """Reset pose and clear the last speed command."""
        (x, y), theta = pose
        self.pose = ((float(x), float(y)), float(theta))
        self.current_speed = (0.0, 0.0)

    def apply_action(self, action, *, dt: float) -> None:
        """Integrate a simple planar motion model for smoke coverage."""
        linear, angular = self.parse_action(action)
        (x, y), theta = self.pose
        next_theta = theta + angular * dt
        next_x = x + linear * cos(next_theta) * dt
        next_y = y + linear * sin(next_theta) * dt
        self.pose = ((next_x, next_y), next_theta)
        self.current_speed = (linear, angular)


def dummy_factory(env_config: EnvSettings, map_def: MapDefinition, _peds: bool) -> DummySimulator:
    """Create a dummy simulator instance for testing.

    Parameters
    ----------
    env_config : EnvSettings
        Environment configuration (seed extracted if available)
    map_def : MapDefinition
        Map definition used to expose obstacle and route metadata
    _peds : bool
        Pedestrian interaction flag (unused in dummy)

    Returns
    -------
    DummySimulator
        A minimal simulator instance that exposes the RobotEnv-facing simulator contract.
    """
    seed = getattr(env_config, "seed", 0)
    sim_settings = getattr(env_config, "sim_config", env_config)
    return DummySimulator(
        map_def=map_def,
        seed=seed,
        step_dt=getattr(sim_settings, "time_per_step_in_secs", 0.1),
        goal_proximity_threshold=getattr(sim_settings, "goal_radius", 1.0),
    )
