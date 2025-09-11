"""Social Force Planner for the Social Navigation Benchmark.

This module implements a configurable Social Force planner that uses the pysocialforce
library to compute navigation forces and generate robot actions. The planner supports
both velocity and unicycle action spaces and provides deterministic behavior when seeded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import atan2
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pysocialforce as pysf
from pysocialforce.config import (
    DesiredForceConfig,
    ObstacleForceConfig,
    SceneConfig,
    SimulatorConfig,
)
from pysocialforce.config import (
    SocialForceConfig as PySFSocialForceConfig,
)

from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


@dataclass
class SFPlannerConfig:
    """Configuration for the Social Force planner."""

    # Kinematics
    mode: str = "velocity"  # "velocity" or "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0
    accel_max: float = 2.0
    dt: float = 0.1

    # Goal dynamics
    desired_speed: float = 1.0
    tau: float = 0.5  # relaxation time

    # Interaction forces
    A: float = 5.1  # social force amplitude
    B: float = 0.35  # social force range (gamma)
    lambda_anisotropy: float = 2.0  # anisotropy factor
    sigma_phi: float = 90.0  # field of view
    cutoff_radius: float = 10.0  # interaction cutoff distance

    # Obstacle forces
    A_obs: float = 10.0  # obstacle force amplitude
    B_obs: float = 0.0  # obstacle force sigma

    # Body contact (optional)
    k_body: float = 0.0  # body force stiffness
    kappa_slide: float = 0.0  # sliding friction

    # Numerics
    integration: str = "euler"  # "euler" or "semi_implicit"
    clip_force: bool = True
    max_force: float = 100.0

    # Stochasticity
    noise_std: float = 0.0

    # I/O
    action_space: str = "velocity"  # "velocity" or "unicycle"
    safety_clamp: bool = True


@dataclass
class Observation:
    """Observation data structure for the Social Force planner."""

    dt: float
    robot: Dict[str, Any]  # position, velocity, goal, radius
    agents: List[Dict[str, Any]]  # list of agent states
    obstacles: List[Any] = field(default_factory=list)  # optional obstacles


class BasePolicy:
    """Abstract base class for baseline policies."""

    def __init__(self, config: Any, *, seed: Optional[int] = None):
        pass

    def reset(self, *, seed: Optional[int] = None) -> None:
        pass

    def configure(self, config: Any) -> None:
        pass

    def step(self, obs: Any) -> Dict[str, float]:
        pass

    def close(self) -> None:
        pass


class SocialForcePlanner(BasePolicy):
    """Social Force-based navigation planner.

    This planner uses the Social Force model to compute navigation commands
    for a robot in a pedestrian environment. It supports both velocity and
    unicycle action spaces and provides deterministic behavior when seeded.

    The planner computes forces from:
    - Goal attraction (desired force)
    - Pedestrian repulsion (social force)
    - Obstacle avoidance (obstacle force)

    And converts these forces to appropriate robot actions.
    """

    def __init__(self, config: Union[Dict, SFPlannerConfig], *, seed: Optional[int] = None):
        """Initialize the Social Force planner.

        Args:
            config: Configuration dict or SFPlannerConfig instance
            seed: Random seed for deterministic behavior
        """
        self.config = self._parse_config(config)
        self._rng = np.random.default_rng(seed)
        self._robot_state = None
        self._sim = None
        self._wrapper = None

        # State tracking
        self._last_position = None
        self._last_velocity = None

    def _parse_config(self, config: Union[Dict, SFPlannerConfig]) -> SFPlannerConfig:
        """Parse configuration from dict or dataclass."""
        if isinstance(config, dict):
            return SFPlannerConfig(**config)
        elif isinstance(config, SFPlannerConfig):
            return config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset the planner state.

        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._robot_state = None
        self._sim = None
        self._wrapper = None
        self._last_position = None
        self._last_velocity = None

    def configure(self, config: Union[Dict, SFPlannerConfig]) -> None:
        """Update planner configuration.

        Args:
            config: New configuration dict or SFPlannerConfig instance
        """
        self.config = self._parse_config(config)

    def step(self, obs: Union[Observation, Dict]) -> Dict[str, float]:
        """Compute action for the given observation.

        Args:
            obs: Observation containing robot state, agents, and environment info

        Returns:
            Action dict with velocity or unicycle commands
        """
        # Parse observation
        if isinstance(obs, dict):
            obs = Observation(**obs)

        # Extract robot state
        robot_pos = np.array(obs.robot["position"], dtype=float)
        robot_vel = np.array(obs.robot["velocity"], dtype=float)
        robot_goal = np.array(obs.robot["goal"], dtype=float)
        robot_radius = float(obs.robot["radius"])

        # Update state tracking
        if self._last_position is None:
            self._last_position = robot_pos.copy()
            self._last_velocity = robot_vel.copy()

        # Use dt from observation or config
        dt = getattr(obs, "dt", self.config.dt)

        # Create minimal simulation context for force calculations
        if self._sim is None or self._wrapper is None:
            self._setup_simulation(obs)

        # Compute Social Force-based action
        total_force = self._compute_total_force(robot_pos, robot_vel, robot_goal, robot_radius, obs)

        # Convert force to action
        action = self._force_to_action(total_force, robot_pos, robot_vel, robot_goal, dt)

        # Update state tracking
        self._last_position = robot_pos.copy()
        self._last_velocity = robot_vel.copy()

        return action

    def _setup_simulation(self, obs: Union[Observation, Dict]) -> None:
        """Setup minimal pysocialforce simulation for force computation."""
        # Create pedestrian states from observation
        agent_states = getattr(obs, "agents", [])
        n_agents = len(agent_states)

        if n_agents > 0:
            # Extract agent positions and create PedState
            positions = np.array([agent["position"] for agent in agent_states])
            velocities = np.array([agent.get("velocity", [0.0, 0.0]) for agent in agent_states])
            goals = np.array([agent.get("goal", [0.0, 0.0]) for agent in agent_states])

            # Create state array in the format expected by PedState
            # State format: [x, y, vx, vy, gx, gy, tau]
            state_array = np.column_stack([
                positions,     # x, y
                velocities,    # vx, vy
                goals,         # gx, gy
            ])
        else:
            # Empty state array
            state_array = np.zeros((0, 6))  # [x, y, vx, vy, gx, gy]

        # Create pysf configuration
        pysf_config = self._create_pysf_config()

        # Create minimal simulator
        obstacles = getattr(obs, "obstacles", [])
        self._sim = pysf.Simulator(
            state=state_array,
            groups=[[]] * n_agents,  # no grouping
            obstacles=obstacles,
            config=pysf_config,
        )

        # Create wrapper for force queries
        self._wrapper = FastPysfWrapper(self._sim)

    def _create_pysf_config(self) -> SimulatorConfig:
        """Create pysocialforce configuration from planner config."""
        return SimulatorConfig(
            scene_config=SceneConfig(
                agent_radius=0.35,
                dt_secs=self.config.dt,
                max_speed_multiplier=1.0,
                tau=self.config.tau,
            ),
            desired_force_config=DesiredForceConfig(
                factor=1.0,
                relaxation_time=self.config.tau,
            ),
            social_force_config=PySFSocialForceConfig(
                factor=self.config.A,
                lambda_importance=self.config.lambda_anisotropy,
                gamma=self.config.B,
                n=2,
                n_prime=3,
            ),
            obstacle_force_config=ObstacleForceConfig(
                factor=self.config.A_obs,
                sigma=self.config.B_obs,
            ),
        )

    def _compute_total_force(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        robot_radius: float,
        obs: Union[Observation, Dict],
    ) -> np.ndarray:
        """Compute total Social Force at robot position.

        Returns:
            2D force vector (fx, fy)
        """
        # Get forces at robot position using wrapper
        total_force = self._wrapper.get_forces_at(
            robot_pos,
            include_desired=True,
            desired_goal=robot_goal,
            include_robot=False,  # Robot doesn't interact with itself
        )

        # Apply force clipping if enabled
        if self.config.clip_force:
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > self.config.max_force:
                total_force = total_force / force_magnitude * self.config.max_force

        # Add noise if configured
        if self.config.noise_std > 0:
            noise = self._rng.normal(0, self.config.noise_std, size=2)
            total_force += noise

        return total_force

    def _force_to_action(
        self,
        force: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        """Convert force to action based on configured action space."""

        if self.config.action_space == "velocity":
            return self._force_to_velocity_action(force, robot_vel, dt)
        elif self.config.action_space == "unicycle":
            return self._force_to_unicycle_action(force, robot_pos, robot_vel, robot_goal, dt)
        else:
            raise ValueError(f"Unknown action space: {self.config.action_space}")

    def _force_to_velocity_action(
        self, force: np.ndarray, robot_vel: np.ndarray, dt: float
    ) -> Dict[str, float]:
        """Convert force to velocity action (vx, vy)."""
        # Compute desired velocity using Social Force dynamics
        # F = m * (v_desired - v_current) / tau
        # v_desired = v_current + F * tau / m (assuming unit mass)

        desired_vel = robot_vel + force * self.config.tau

        # Apply velocity limits if safety clamping is enabled
        if self.config.safety_clamp:
            speed = np.linalg.norm(desired_vel)
            if speed > self.config.v_max:
                desired_vel = desired_vel / speed * self.config.v_max

        return {"vx": float(desired_vel[0]), "vy": float(desired_vel[1])}

    def _force_to_unicycle_action(
        self,
        force: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        """Convert force to unicycle action (v, omega)."""
        # Compute desired velocity direction from force
        force_magnitude = np.linalg.norm(force)
        if force_magnitude < 1e-6:
            return {"v": 0.0, "omega": 0.0}

        desired_direction = force / force_magnitude

        # Current robot orientation (assume facing in velocity direction)
        current_speed = np.linalg.norm(robot_vel)
        if current_speed > 1e-6:
            current_direction = robot_vel / current_speed
            current_heading = atan2(current_direction[1], current_direction[0])
        else:
            # If stationary, face towards goal
            goal_direction = robot_goal - robot_pos
            if np.linalg.norm(goal_direction) > 1e-6:
                goal_direction = goal_direction / np.linalg.norm(goal_direction)
                current_heading = atan2(goal_direction[1], goal_direction[0])
            else:
                current_heading = 0.0

        # Desired heading from force direction
        desired_heading = atan2(desired_direction[1], desired_direction[0])

        # Compute angular error and velocity
        heading_error = desired_heading - current_heading

        # Normalize angle to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # Compute angular velocity (simple proportional control)
        omega = heading_error / dt

        # Compute linear velocity based on force magnitude and goal distance
        goal_distance = np.linalg.norm(robot_goal - robot_pos)
        desired_speed = min(self.config.desired_speed, goal_distance / dt)

        # Scale speed by force magnitude (weak force = slower movement)
        speed_scale = min(1.0, force_magnitude / 1.0)  # normalize by expected force
        v = desired_speed * speed_scale

        # Apply limits if safety clamping is enabled
        if self.config.safety_clamp:
            v = max(0.0, min(v, self.config.v_max))
            omega = max(-self.config.omega_max, min(omega, self.config.omega_max))

        return {"v": float(v), "omega": float(omega)}

    def close(self) -> None:
        """Clean up resources."""
        self._sim = None
        self._wrapper = None

    def get_metadata(self) -> Dict[str, Any]:
        """Get planner metadata for episode output."""
        config_dict = asdict(self.config)
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[
            :16
        ]

        return {
            "algorithm": "social_force",
            "config": config_dict,
            "config_hash": config_hash,
        }


__all__ = ["SocialForcePlanner", "SFPlannerConfig", "Observation"]
