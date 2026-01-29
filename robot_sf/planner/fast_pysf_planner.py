"""Local fast-pysf planner that operates on simulator ground truth.

This planner uses the live PySocialForce simulator state to compute
interaction forces, then converts a holonomic velocity command into a
differential-drive action via an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from robot_sf.robot.action_adapters import DiffDriveAdapterConfig, holonomic_to_diff_drive_action
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

if TYPE_CHECKING:
    from robot_sf.sim.simulator import Simulator


@dataclass
class FastPysfPlannerConfig:
    """Configuration for the fast-pysf local planner."""

    desired_speed: float = 1.5
    tau: float = 0.5
    interaction_weight: float = 1.0
    max_force: float = 40.0
    goal_tolerance: float = 0.25


class FastPysfPlannerPolicy:
    """Planner policy that produces differential-drive actions from ground truth."""

    def __init__(
        self,
        simulator: Simulator,
        *,
        robot_index: int = 0,
        config: FastPysfPlannerConfig | None = None,
        adapter_config: DiffDriveAdapterConfig | None = None,
    ) -> None:
        """Initialize the planner policy.

        Args:
            simulator: Live robot simulator instance (ground truth).
            robot_index: Which robot to control.
            config: Planner configuration.
            adapter_config: Adapter configuration for holonomic-to-diff-drive conversion.
        """
        self.simulator = simulator
        self.robot_index = robot_index
        self.config = config or FastPysfPlannerConfig()
        self.adapter_config = adapter_config or DiffDriveAdapterConfig()
        self._wrapper = FastPysfWrapper(simulator.pysf_sim)
        self._warned_missing_goal = False

    def reset(self) -> None:
        """Reset planner state (noop for now)."""
        return None

    def predict(self, _obs=None, **_kwargs):
        """Return a diff-drive action compatible with Gym predict signatures."""
        return self.action(), None

    def action(self) -> np.ndarray:
        """Compute a differential-drive action from current simulator state.

        Returns:
            np.ndarray: Differential-drive action ``[v, omega]``.
        """
        robot = self.simulator.robots[self.robot_index]
        (x, y), heading = robot.pose
        robot_pos = np.array([x, y], dtype=float)

        goal_list = self.simulator.goal_pos
        if not goal_list or self.robot_index >= len(goal_list):
            if not self._warned_missing_goal:
                logger.warning("FastPysfPlannerPolicy missing goal; returning zero action.")
                self._warned_missing_goal = True
            return np.zeros(2, dtype=float)

        goal = np.asarray(goal_list[self.robot_index], dtype=float)
        to_goal = goal - robot_pos
        goal_dist = float(np.linalg.norm(to_goal))
        if goal_dist < self.config.goal_tolerance:
            return np.zeros(2, dtype=float)

        linear_speed = float(robot.current_speed[0])
        robot_vel = np.array(
            [linear_speed * cos(heading), linear_speed * sin(heading)], dtype=float
        )

        desired_speed = min(self.config.desired_speed, goal_dist / max(self._dt, 1e-6))
        desired_vel = (to_goal / goal_dist) * desired_speed
        desired_force = (desired_vel - robot_vel) / max(self.config.tau, 1e-6)

        interaction_force = self._wrapper.get_forces_at(
            robot_pos, include_desired=False, include_robot=False
        )
        total_force = desired_force + self.config.interaction_weight * interaction_force
        total_force = self._clip_force(total_force)

        holonomic_vel = robot_vel + total_force * self._dt
        return holonomic_to_diff_drive_action(
            holonomic_vel,
            robot.pose,
            max_linear_speed=robot.config.max_linear_speed,
            max_angular_speed=robot.config.max_angular_speed,
            config=self.adapter_config,
        )

    @property
    def _dt(self) -> float:
        """Simulation timestep in seconds."""
        return float(self.simulator.config.time_per_step_in_secs)

    def _clip_force(self, force: np.ndarray) -> np.ndarray:
        """Clip total force magnitude to avoid numerical spikes.

        Returns:
            np.ndarray: Clipped force vector.
        """
        norm = float(np.linalg.norm(force))
        if norm <= self.config.max_force or norm < 1e-6:
            return force
        return force / norm * self.config.max_force


__all__ = ["FastPysfPlannerConfig", "FastPysfPlannerPolicy"]
