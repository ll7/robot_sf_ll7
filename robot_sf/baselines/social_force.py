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
from dataclasses import asdict, dataclass, field
from math import atan2
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pysocialforce as pysf
from pysocialforce.config import (
    DesiredForceConfig,
    ObstacleForceConfig,
    SceneConfig,
    SimulatorConfig,
)
from pysocialforce.config import SocialForceConfig as PySFSocialForceConfig

from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


@dataclass
class SFPlannerConfig:
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


@dataclass
class Observation:
    dt: float
    robot: Dict[str, Any]
    agents: List[Dict[str, Any]]
    obstacles: List[Any] = field(default_factory=list)


class BasePolicy:
    def __init__(self, config: Any, *, seed: Optional[int] = None):
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None) -> None:
        raise NotImplementedError

    def configure(self, config: Any) -> None:
        raise NotImplementedError

    def step(self, obs: Any) -> Dict[str, float]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SocialForcePlanner(BasePolicy):
    # Numerical stability epsilon used for safe division and small-magnitude checks.
    EPSILON: float = 1e-9

    class _RNGCompat:
        """Expose randint() and normal() using numpy's Generator."""

        def __init__(self, seed: Optional[int]):
            self._gen = np.random.default_rng(seed)

        def randint(
            self, low: int, high: Optional[int] = None, size: Optional[int | tuple[int, ...]] = None
        ):
            return self._gen.integers(low, high=high, size=size)

        def normal(
            self, loc: float = 0.0, scale: float = 1.0, size: Optional[int | tuple[int, ...]] = None
        ):
            return self._gen.normal(loc, scale, size)

    def __init__(self, config: Union[Dict, SFPlannerConfig], *, seed: Optional[int] = None):
        self.config = self._parse_config(config)
        self._rng = self._RNGCompat(seed)
        self._sim: Optional[Any] = None
        self._wrapper: Optional[FastPysfWrapper] = None
        self._last_position: Optional[np.ndarray] = None
        self._last_velocity: Optional[np.ndarray] = None
        self._robot_state: Optional[Dict[str, Any]] = None

    def _parse_config(self, config: Union[Dict, SFPlannerConfig]) -> SFPlannerConfig:
        if isinstance(config, dict):
            return SFPlannerConfig(**config)
        if isinstance(config, SFPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = self._RNGCompat(seed)
        self._sim = None
        self._wrapper = None
        self._last_position = None
        self._last_velocity = None
        self._robot_state = None

    def configure(self, config: Union[Dict, SFPlannerConfig]) -> None:
        self.config = self._parse_config(config)

    def step(self, obs: Union[Observation, Dict]) -> Dict[str, float]:
        if isinstance(obs, dict):
            obs = Observation(**obs)  # type: ignore[arg-type]
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

        total_force = self._compute_total_force(robot_pos, robot_goal, robot_vel)
        action = self._force_to_action(total_force, robot_pos, robot_vel, robot_goal, dt)

        self._last_position = robot_pos.copy()
        self._last_velocity = robot_vel.copy()
        return action

    def _setup_simulation(self, obs: Observation) -> None:
        agent_states = getattr(obs, "agents", [])
        n_agents = len(agent_states)

        if n_agents > 0:
            positions = np.asarray([a["position"] for a in agent_states], dtype=float)
            velocities = np.asarray(
                [a.get("velocity", [0.0, 0.0]) for a in agent_states], dtype=float
            )
            goals = np.asarray([a.get("goal", [0.0, 0.0]) for a in agent_states], dtype=float)
            state_array = np.column_stack([positions, velocities, goals])
        else:
            state_array = np.zeros((0, 6), dtype=float)

        cfg = self._create_pysf_config()

        obstacles: Optional[Sequence] = getattr(obs, "obstacles", None)
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
        self, robot_pos: np.ndarray, robot_goal: np.ndarray, robot_vel: np.ndarray
    ) -> np.ndarray:
        """Compute total acceleration-like force (desired + interactions).

        We explicitly form desired acceleration to avoid double-counting and
        optionally scale desired speed up to ``v_max`` for long straight runs so
        the robot can reach distant goals within benchmark episode limits.
        Interaction (social + obstacle) forces are scaled by a configurable
        weight to tune clearance behavior.
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
        interactions = self._wrapper.get_forces_at(
            robot_pos,
            include_desired=False,
            include_robot=False,
        )
        interactions *= self.config.interaction_weight

        total = desired + interactions
        # Replace any NaNs/Infs defensively
        total = np.nan_to_num(
            total, nan=0.0, posinf=self.config.max_force, neginf=-self.config.max_force
        )
        if self.config.clip_force:
            m = float(np.linalg.norm(total))
            if m > self.config.max_force:
                total = total / (m + self.EPSILON) * self.config.max_force
        if self.config.noise_std > 0:
            total = total + np.asarray(
                self._rng.normal(0.0, self.config.noise_std, size=2), dtype=float
            )
        return total

    def _force_to_action(
        self,
        force: np.ndarray,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        if self.config.action_space == "velocity":
            return self._force_to_velocity_action(force, robot_vel)
        if self.config.action_space == "unicycle":
            return self._force_to_unicycle_action(force, robot_pos, robot_vel, robot_goal, dt)
        raise ValueError(f"Unknown action space: {self.config.action_space}")

    def _force_to_velocity_action(
        self, force: np.ndarray, robot_vel: np.ndarray
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
        self._sim = None
        self._wrapper = None

    def get_metadata(self) -> Dict[str, Any]:
        config_dict = asdict(self.config)
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        return {"algorithm": "social_force", "config": config_dict, "config_hash": config_hash}


__all__ = ["SocialForcePlanner", "SFPlannerConfig", "Observation"]
