"""
Lightweight adapters for running SocNavBench-style planners against the structured
observation mode.

These adapters are intentionally minimal to provide an in-process bridge while
full SocNavBench planners are integrated. They operate on the SocNav structured
observation emitted when `ObservationMode.SOCNAV_STRUCT` is enabled.
"""

from dataclasses import dataclass
from math import atan2, pi

import numpy as np


@dataclass
class SocNavPlannerConfig:
    """Simple config for SocNav-like planner adapters."""

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0
    angular_gain: float = 1.0
    goal_tolerance: float = 0.25


class SamplingPlannerAdapter:
    """
    Minimal waypoint-to-velocity adapter inspired by SocNavBench sampling planner.

    This is a placeholder that consumes structured SocNav observations and returns
    differential-drive (v, w) commands. It is designed so that the internals can be
    swapped for the real SocNavBench sampling planner without changing callers.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None):
        """Initialize the adapter with optional planner configuration."""

        self.config = config or SocNavPlannerConfig()

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute a (v, w) command from the structured observation.

        Args:
            observation: SocNav structured observation Dict (robot, goal, pedestrians, map, sim).

        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        robot_state = observation["robot"]
        goal_state = observation["goal"]
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(robot_state["heading"][0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        distance = float(np.linalg.norm(to_goal))
        if distance < self.config.goal_tolerance:
            return 0.0, 0.0

        desired_heading = atan2(to_goal[1], to_goal[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)

        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )

        # Slow down when sharply turning
        linear_scale = max(0.0, 1.0 - abs(heading_error) / pi)
        linear = float(
            np.clip(distance * linear_scale, 0.0, self.config.max_linear_speed),
        )
        return linear, angular

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap angle to [-pi, pi].

        Returns:
            float: Wrapped angle in radians.
        """
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle


class SocNavPlannerPolicy:
    """Thin policy wrapper to plug planner adapters into Gym loops."""

    def __init__(self, adapter: SamplingPlannerAdapter | None = None):
        """Initialize the policy with a planner adapter."""

        self.adapter = adapter or SamplingPlannerAdapter()

    def act(self, observation: dict) -> tuple[float, float]:
        """Return (v, w) action for a SocNav structured observation."""
        return self.adapter.plan(observation)
