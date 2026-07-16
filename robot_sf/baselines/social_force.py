"""Social Force Planner for the Social Navigation Benchmark.

A minimal Social Force baseline that uses the vendored fast-pysf (pysocialforce)
components via a small wrapper to compute forces and turn them into actions.

Goals:
- Deterministic when seeded.
- Safe with empty agents/obstacles (no NaNs).
- Simple API: step(Observation|dict) -> action dict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import atan2
from typing import TYPE_CHECKING, Any

import numpy as np
import pysocialforce as pysf
from pysocialforce.config import (
    DesiredForceConfig,
    ObstacleForceConfig,
    SceneConfig,
    SimulatorConfig,
)
from pysocialforce.config import SocialForceConfig as PySFSocialForceConfig

from robot_sf.baselines.interface import (
    Observation,
    is_observation_mapping,
    observation_from_mapping,
)
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class SFPlannerConfig:
    """Configuration for the social-force planner baseline."""

    # Kinematics
    mode: str = "velocity"  # "velocity" or "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0
    accel_max: float = 2.0
    dt: float = 0.1

    # Goal dynamics
    desired_speed: float = 1.0
    tau: float = 0.5

    # Interaction forces
    A: float = 5.1
    B: float = 0.35
    lambda_anisotropy: float = 2.0
    sigma_phi: float = 90.0
    cutoff_radius: float = 10.0

    # Obstacle forces
    A_obs: float = 10.0
    B_obs: float = 0.0

    # Body contact (optional)
    k_body: float = 0.0
    kappa_slide: float = 0.0

    # Numerics
    integration: str = "euler"
    clip_force: bool = True
    max_force: float = 100.0

    # Stochasticity
    noise_std: float = 0.0

    # I/O
    action_space: str = "velocity"  # or "unicycle"
    safety_clamp: bool = True
    # Adaptive behavior (tuning aids)
    speed_scale_to_vmax: bool = True  # Scale desired speed up to v_max (aggressive)
    interaction_weight: float = 3.0  # Stronger repulsion to maintain clearance

    # AMMV-aware interaction (vehicle-awareness for pedestrian response)
    ammv_aware_enabled: bool = False  # Toggle AMMV-aware interaction term
    ammv_repulsion_amplitude: float = 3.0  # Strength of speed-scaled repulsion [N]
    ammv_repulsion_range: float = 0.5  # Spatial range of AMMV-effect [m]
    ammv_speed_factor: float = 0.3  # Robot speed amplification factor
    ammv_diagnostics_enabled: bool = False  # Track AMMV force components


class BasePolicy:
    """Interface for benchmark baseline policies."""

    def __init__(self, config: Any, *, seed: int | None = None):
        """Initialize the policy.

        Args:
            config: Planner configuration payload.
            seed: Optional random seed.
        """
        raise NotImplementedError

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal state.

        Args:
            seed: Optional new seed.
        """
        raise NotImplementedError

    def configure(self, config: Any) -> None:
        """Update planner configuration.

        Args:
            config: New configuration payload.
        """
        raise NotImplementedError

    def step(self, obs: Any) -> dict[str, float]:
        """Compute an action for the current observation.

        Args:
            obs: Observation payload.

        Returns:
            Action dict in velocity or unicycle format.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources."""
        raise NotImplementedError


class SocialForcePlanner(BasePolicy):
    """Social-force baseline planner using PySocialForce interactions."""

    # Numerical stability epsilon used for safe division and small-magnitude checks.
    EPSILON: float = 1e-9

    class _RNGCompat:
        """Expose randint() and normal() using numpy's Generator."""

        def __init__(self, seed: int | None):
            """Initialize RNG compatibility wrapper.

            Args:
                seed: Optional RNG seed.
            """
            self._gen = np.random.default_rng(seed)

        def randint(
            self,
            low: int,
            high: int | None = None,
            size: int | tuple[int, ...] | None = None,
        ):
            """Generate random integers from low (inclusive) to high (exclusive).

            Provides compatibility with legacy random.randint() where a single
            argument means [0, low).

            Args:
                low: Lower bound (inclusive) or upper bound (exclusive) if high is None.
                high: Upper bound (exclusive). If None, low is treated as upper bound
                    with lower bound of 0.
                size: Output shape.

            Returns:
                Random integer(s) in the specified range.
            """
            if high is None:
                # Compatibility mode: randint(n) -> integers in [0, n)
                return self._gen.integers(0, low, size=size)
            return self._gen.integers(low, high, size=size)

        def normal(
            self,
            loc: float = 0.0,
            scale: float = 1.0,
            size: int | tuple[int, ...] | None = None,
        ):
            """Return samples from a normal distribution.

            Args:
                loc: Mean of the distribution.
                scale: Standard deviation of the distribution.
                size: Output shape.

            Returns:
                Random values from normal distribution.
            """
            return self._gen.normal(loc, scale, size)

    def __init__(self, config: dict[str, Any] | SFPlannerConfig, *, seed: int | None = None):
        """Initialize the social-force planner.

        Args:
            config: Planner configuration or dict payload.
            seed: Optional RNG seed.
        """
        self.config = self._parse_config(config)
        self._rng = self._RNGCompat(seed)
        self._sim: Any | None = None
        self._wrapper: FastPysfWrapper | None = None
        self._last_position: np.ndarray | None = None
        self._last_velocity: np.ndarray | None = None
        self._robot_state: dict[str, Any] | None = None
        self._last_ammv_force: np.ndarray | None = None
        self._last_ammv_diagnostics: dict[str, Any] = self._empty_ammv_diagnostics()

    def _parse_config(self, config: dict[str, Any] | SFPlannerConfig) -> SFPlannerConfig:
        """Normalize config input into an SFPlannerConfig.

        Args:
            config: Configuration object or dict.

        Returns:
            Parsed SFPlannerConfig.
        """
        if isinstance(config, dict):
            return SFPlannerConfig(**config)  # type: ignore[arg-type]
        if isinstance(config, SFPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal state and RNG.

        Args:
            seed: Optional new seed.
        """
        if seed is not None:
            self._rng = self._RNGCompat(seed)
        self._sim = None
        self._wrapper = None
        self._last_position = None
        self._last_velocity = None
        self._robot_state = None
        self._last_ammv_force = None
        self._last_ammv_diagnostics = self._empty_ammv_diagnostics()

    def configure(self, config: dict[str, Any] | SFPlannerConfig) -> None:
        """Update the planner configuration.

        Args:
            config: Configuration object or dict.
        """
        self.config = self._parse_config(config)

    def step(self, obs: Observation | dict) -> dict[str, float]:
        """Compute an action for the given observation.

        Args:
            obs: Observation payload.

        Returns:
            Action dict in configured action space.
        """
        if is_observation_mapping(obs):
            obs = observation_from_mapping(obs)
        assert isinstance(obs, Observation)

        robot_pos = np.asarray(obs.robot["position"], dtype=float)
        robot_vel = np.asarray(obs.robot["velocity"], dtype=float)
        robot_goal = np.asarray(obs.robot["goal"], dtype=float)
        robot_radius = float(obs.robot["radius"])

        if self._last_position is None:
            self._last_position = robot_pos.copy()
            self._last_velocity = robot_vel.copy()

        dt = float(getattr(obs, "dt", self.config.dt))

        if self._sim is None or self._wrapper is None:
            self._setup_simulation(obs)

        self._robot_state = {
            "pos": robot_pos.tolist(),
            "vel": robot_vel.tolist(),
            "radius": robot_radius,
        }

        total_force = self._compute_total_force(robot_pos, robot_goal, robot_vel, obs.agents)
        action = self._force_to_action(total_force, robot_pos, robot_vel, robot_goal, dt)

        self._last_position = robot_pos.copy()
        self._last_velocity = robot_vel.copy()
        return action

    def _setup_simulation(self, obs: Observation) -> None:
        """Initialize an internal PySocialForce simulator for pedestrians.

        Args:
            obs: Observation containing agent state and obstacles.
        """
        agent_states = getattr(obs, "agents", [])
        n_agents = len(agent_states)

        if n_agents > 0:
            positions = np.asarray([a["position"] for a in agent_states], dtype=float)
            velocities = np.asarray(
                [a.get("velocity", [0.0, 0.0]) for a in agent_states],
                dtype=float,
            )
            goals = np.asarray([a.get("goal", [0.0, 0.0]) for a in agent_states], dtype=float)
            state_array = np.column_stack([positions, velocities, goals])
        else:
            state_array = np.zeros((0, 6), dtype=float)

        cfg = self._create_pysf_config()

        obstacles: Sequence | None = getattr(obs, "obstacles", None)
        if obstacles is not None and len(obstacles) == 0:
            obstacles = None

        self._sim = pysf.Simulator(
            state=state_array,
            groups=[[]] * n_agents,
            obstacles=obstacles,  # type: ignore[arg-type]
            config=cfg,
        )
        self._wrapper = FastPysfWrapper(self._sim)

    def _create_pysf_config(self) -> SimulatorConfig:
        """Build a PySocialForce simulator configuration from planner settings.


        Returns:
            SimulatorConfig for pysocialforce.
        """
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
        robot_goal: np.ndarray,
        robot_vel: np.ndarray,
        agent_states: list[dict[str, Any]],
    ) -> np.ndarray:
        """Compute total acceleration-like force (desired + interactions).

        We explicitly form desired acceleration to avoid double-counting and
        optionally scale desired speed up to ``v_max`` for long straight runs so
        the robot can reach distant goals within benchmark episode limits.
        Interaction (social + obstacle) forces are scaled by a configurable
        weight to tune clearance behavior.

        Returns:
            Total force vector combining goal attraction and obstacle/pedestrian repulsion.
        """
        goal_vec = robot_goal - robot_pos
        dist = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / dist if dist > self.EPSILON else np.zeros_like(goal_vec)

        # Adaptive desired speed: allow v_max when sufficiently far from goal
        base_speed = self.config.desired_speed
        if self.config.speed_scale_to_vmax and base_speed < self.config.v_max:
            base_speed = self.config.v_max
        desired_speed = min(base_speed, dist / max(self.config.dt, 1e-6))
        v_des = goal_dir * desired_speed
        desired = (v_des - robot_vel) / max(self.config.tau, 1e-6)

        # Interaction forces (exclude desired to prevent double counting)
        assert self._wrapper is not None, "Wrapper must be initialized"
        interactions = self._wrapper.get_forces_at(
            robot_pos,  # type: ignore[arg-type]  # ndarray is a Sequence
            include_desired=False,
            include_robot=False,
        )
        interactions *= self.config.interaction_weight

        ammv_force = self._compute_ammv_aware_force(robot_pos, robot_vel, agent_states)
        total = desired + interactions + ammv_force
        # Replace any NaNs/Infs defensively
        total = np.nan_to_num(
            total,
            nan=0.0,
            posinf=self.config.max_force,
            neginf=-self.config.max_force,
        )
        if self.config.clip_force:
            m = float(np.linalg.norm(total))
            if m > self.config.max_force:
                total = total / (m + self.EPSILON) * self.config.max_force
        if self.config.noise_std > 0:
            total = total + np.asarray(
                self._rng.normal(0.0, self.config.noise_std, size=2),
                dtype=float,
            )
        return total

    @staticmethod
    def _empty_ammv_diagnostics() -> dict[str, Any]:
        """Return an empty diagnostic payload for the optional AMMV term."""
        return {
            "enabled": False,
            "agent_count": 0,
            "intrusion_count": 0,
            "max_force_magnitude": 0.0,
            "min_lateral_clearance": None,
            "min_time_to_collision": None,
            "agents": [],
        }

    def _store_ammv_diagnostics(
        self,
        ammv_force: np.ndarray,
        diagnostics: dict[str, Any],
    ) -> None:
        """Store AMMV force and diagnostics when tracking is enabled."""
        if self.config.ammv_diagnostics_enabled:
            self._last_ammv_force = ammv_force.copy()
        self._last_ammv_diagnostics = diagnostics

    def _compute_ammv_aware_force(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        agent_states: list[dict[str, Any]],
    ) -> np.ndarray:
        """AMMV-aware speed-dependent repulsion from nearby pedestrians.

        Models stronger pedestrian repulsion when the robot moves faster,
        reflecting the intuition that pedestrians give more clearance to
        fast-moving vehicles. Force magnitude scales with robot speed and
        decays exponentially with distance.

        Returns:
            AMMV-aware force vector, or zeros when disabled or no pedestrians.
        """
        diagnostics = self._empty_ammv_diagnostics()
        diagnostics["enabled"] = self.config.ammv_aware_enabled

        if not self.config.ammv_aware_enabled:
            self._last_ammv_diagnostics = diagnostics
            ammv_force = np.zeros(2, dtype=float)
            self._store_ammv_diagnostics(ammv_force, diagnostics)
            return ammv_force

        ped_pos = np.asarray([a["position"] for a in agent_states], dtype=float)
        n_peds = ped_pos.shape[0]
        if n_peds == 0:
            self._last_ammv_diagnostics = diagnostics
            ammv_force = np.zeros(2, dtype=float)
            self._store_ammv_diagnostics(ammv_force, diagnostics)
            return ammv_force
        ped_vels = np.asarray(
            [a.get("velocity", [0.0, 0.0]) for a in agent_states],
            dtype=float,
        )

        robot_speed = float(np.linalg.norm(robot_vel))
        speed_mult = 1.0 + self.config.ammv_speed_factor * robot_speed
        ref_dir = (
            robot_vel / robot_speed
            if robot_speed > self.EPSILON
            else np.array([1.0, 0.0], dtype=float)
        )

        total = np.zeros(2, dtype=float)
        agent_rows: list[dict[str, float | bool | None]] = []
        intrusion_count = 0
        max_force_magnitude = 0.0
        min_lateral_clearance: float | None = None
        min_time_to_collision: float | None = None

        for i in range(n_peds):
            diff = robot_pos - ped_pos[i]
            dist = float(np.linalg.norm(diff))
            if dist < self.EPSILON:
                continue

            effective_range = self.config.ammv_repulsion_range * speed_mult
            if dist > effective_range:
                continue

            direction = diff / dist
            magnitude = (
                self.config.ammv_repulsion_amplitude * speed_mult * np.exp(-dist / effective_range)
            )
            total += direction * magnitude

            ped_vel = ped_vels[i]
            relative_pos = ped_pos[i] - robot_pos
            relative_vel = ped_vel - robot_vel
            lateral_clearance = abs(
                float(ref_dir[0] * relative_pos[1] - ref_dir[1] * relative_pos[0])
            )
            closing_speed = -float(np.dot(relative_pos, relative_vel)) / max(dist, self.EPSILON)
            time_to_collision = dist / closing_speed if closing_speed > self.EPSILON else None
            intrusion = lateral_clearance <= self.config.ammv_repulsion_range or (
                time_to_collision is not None
                and time_to_collision <= max(self.config.dt, self.EPSILON)
            )
            intrusion_count += int(intrusion)
            max_force_magnitude = max(
                max_force_magnitude, float(np.linalg.norm(direction * magnitude))
            )
            min_lateral_clearance = (
                lateral_clearance
                if min_lateral_clearance is None
                else min(min_lateral_clearance, lateral_clearance)
            )
            if time_to_collision is not None:
                min_time_to_collision = (
                    time_to_collision
                    if min_time_to_collision is None
                    else min(min_time_to_collision, time_to_collision)
                )
            agent_rows.append(
                {
                    "distance": dist,
                    "relative_bearing": float(np.arctan2(relative_pos[1], relative_pos[0])),
                    "speed": float(np.linalg.norm(ped_vel)),
                    "time_to_collision": time_to_collision,
                    "lateral_clearance": lateral_clearance,
                    "force_magnitude": float(np.linalg.norm(direction * magnitude)),
                    "intrusion": intrusion,
                }
            )

        diagnostics.update(
            {
                "agent_count": len(agent_rows),
                "intrusion_count": intrusion_count,
                "max_force_magnitude": max_force_magnitude,
                "min_lateral_clearance": min_lateral_clearance,
                "min_time_to_collision": min_time_to_collision,
                "agents": agent_rows,
            }
        )

        self._store_ammv_diagnostics(total, diagnostics)

        return total

    def _force_to_action(
        self,
        force: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        dt: float,
    ) -> dict[str, float]:
        """Convert a force vector into an action dict.

        Args:
            force: Force vector in world coordinates.
            robot_pos: Robot position.
            robot_vel: Robot velocity.
            robot_goal: Robot goal.
            dt: Timestep duration.

        Returns:
            Action dict in configured action space.
        """
        if self.config.action_space == "velocity":
            return self._force_to_velocity_action(force, robot_vel)
        if self.config.action_space == "unicycle":
            return self._force_to_unicycle_action(force, robot_pos, robot_vel, robot_goal, dt)
        raise ValueError(f"Unknown action space: {self.config.action_space}")

    def _force_to_velocity_action(
        self,
        force: np.ndarray,
        robot_vel: np.ndarray,
    ) -> dict[str, float]:
        """Map force to a velocity-space action.

        Args:
            force: Force vector.
            robot_vel: Current robot velocity.

        Returns:
            Action dict with (vx, vy).
        """
        desired_vel = robot_vel + force * self.config.tau
        if self.config.safety_clamp:
            speed = float(np.linalg.norm(desired_vel))
            if speed > self.config.v_max:
                desired_vel = desired_vel / (speed + self.EPSILON) * self.config.v_max
        return {"vx": float(desired_vel[0]), "vy": float(desired_vel[1])}

    def _force_to_unicycle_action(
        self,
        force: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        dt: float,
    ) -> dict[str, float]:
        """Map force to a unicycle action (v, omega).

        Args:
            force: Force vector.
            robot_pos: Robot position.
            robot_vel: Robot velocity.
            robot_goal: Robot goal.
            dt: Timestep duration.

        Returns:
            Action dict with (v, omega).
        """
        mag = float(np.linalg.norm(force))
        if mag < 1e-6:
            return {"v": 0.0, "omega": 0.0}
        desired_dir = force / (mag + self.EPSILON)
        cur_speed = float(np.linalg.norm(robot_vel))
        if cur_speed > 1e-6:
            cur_dir = robot_vel / (cur_speed + self.EPSILON)
            cur_heading = atan2(cur_dir[1], cur_dir[0])
        else:
            goal_dir = robot_goal - robot_pos
            if float(np.linalg.norm(goal_dir)) > 1e-6:
                goal_dir = goal_dir / (float(np.linalg.norm(goal_dir)) + self.EPSILON)
                cur_heading = atan2(goal_dir[1], goal_dir[0])
            else:
                cur_heading = 0.0
        desired_heading = atan2(desired_dir[1], desired_dir[0])
        err = desired_heading - cur_heading
        while err > np.pi:
            err -= 2 * np.pi
        while err < -np.pi:
            err += 2 * np.pi
        omega = err / max(dt, 1e-3)
        goal_dist = float(np.linalg.norm(robot_goal - robot_pos))
        desired_speed = min(self.config.desired_speed, goal_dist / max(dt, 1e-3))
        v = desired_speed * min(1.0, mag)
        if self.config.safety_clamp:
            v = max(0.0, min(v, self.config.v_max))
            omega = max(-self.config.omega_max, min(omega, self.config.omega_max))
        return {"v": float(v), "omega": float(omega)}

    def close(self) -> None:
        """Release internal simulator state."""
        self._sim = None
        self._wrapper = None

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the planner.


        Returns:
            Metadata dict including config hash.
        """
        config_dict = asdict(self.config)
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        metadata: dict[str, Any] = {
            "algorithm": "social_force",
            "config": config_dict,
            "config_hash": config_hash,
            "status": "ok",
        }
        if self.config.ammv_diagnostics_enabled:
            ammv_force = (
                self._last_ammv_force
                if self._last_ammv_force is not None
                else np.zeros(2, dtype=float)
            )
            ammv_mag = float(np.linalg.norm(ammv_force))
            metadata["ammv_force_magnitude"] = ammv_mag
            metadata["ammv_force_x"] = float(ammv_force[0])
            metadata["ammv_force_y"] = float(ammv_force[1])
            metadata["ammv_diagnostics"] = self._last_ammv_diagnostics
        return metadata


__all__ = ["Observation", "SFPlannerConfig", "SocialForcePlanner"]
