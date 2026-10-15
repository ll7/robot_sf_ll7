"""Default-off ORCA-seeded native-drive MPC, reusing the hybrid's CEM sampler."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from shapely import distance, linestrings

from robot_sf.planner.clearance_geometry import swept_relative_clearance
from robot_sf.planner.drive_rollout import native_drive_rollout_batch
from robot_sf.planner.hybrid_orca_sampler import (
    HybridORCASamplerAdapter,
    HybridORCASamplerConfig,
    build_hybrid_orca_sampler_build_config,
)
from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
from robot_sf.planner.prediction_mpc import ConstantVelocityPedestrianPredictor
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.reverse_drive import bound_drive_settings


@dataclass
class ORCAMPCConfig(MPPISocialConfig):
    """Fixed-budget v1 research configuration; dt must equal the control period."""

    max_linear_speed: float = 2.0
    max_angular_speed: float = 1.0
    horizon_steps: int = 30
    rollout_dt: float = 0.1
    sample_count: int = 48
    iterations: int = 2
    goal_progress_weight: float = 5.0
    clearance_weight: float = 1.0
    smoothness_weight: float = 0.15
    hard_clearance: float = 0.02
    progress_escape_enabled: bool = False

    def __post_init__(self) -> None:
        """Reject invalid budgets, limits and cost weights before acquisition."""
        for name in ("horizon_steps", "sample_count", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_count < 8:
            raise ValueError("sample_count must be at least eight")
        for name in ("rollout_dt", "max_linear_speed", "max_angular_speed"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "hard_clearance",
            "goal_progress_weight",
            "clearance_weight",
            "smoothness_weight",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self._validate_search_contract()

    def _validate_search_contract(self) -> None:
        """Validate the inherited stochastic search options used by B."""
        if not 0 < self.elite_fraction <= 1 or not np.isfinite(self.elite_fraction):
            raise ValueError("elite_fraction must lie in (0,1]")
        for name in ("init_linear_std", "init_angular_std", "min_linear_std", "min_angular_std"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.prediction_backend != "constant_velocity":
            raise ValueError("orca_mpc v1 requires constant_velocity prediction")
        if self.progress_escape_enabled:
            raise ValueError("orca_mpc forbids post-search command overrides")


class _DriveMPCSampler(MPPISocialPlannerAdapter):
    """Use the unchanged MPPI/CEM loop with ORCA seeds and physical scoring."""

    def __init__(self, config: ORCAMPCConfig) -> None:
        super().__init__(config)
        self.predictor = ConstantVelocityPedestrianPredictor()
        self.drive = None
        self.previous = None
        self.seed_command = (0.0, 0.0)
        self.last: dict[str, Any] = {}

    def _crowd_speed_cap(self, robot_pos, ped_pos) -> float:
        return min(self.config.max_linear_speed, self.drive.max_linear_speed)

    def _extract_state(self, observation) -> tuple:
        state = list(super()._extract_state(observation))
        robot, goal, _peds = self._socnav_fields(observation)
        current = np.asarray(goal["current"], float)
        state[3] = (
            current if np.linalg.norm(current - state[0]) > self.config.goal_tolerance else state[3]
        )
        self.current_vw = np.asarray(robot["speed"], float).reshape(2)
        self._validate_observed_pedestrians(_peds)
        self.observation = observation
        self.dt = self._simulation_timestep(observation)
        if not np.isclose(self.dt, self.config.rollout_dt, rtol=0, atol=1e-7):
            raise ValueError("orca_mpc rollout_dt must equal the observed control period")
        if not all(np.isfinite(value).all() for value in state):
            raise ValueError("orca_mpc requires finite observed state")
        self.active_goal = state[3]
        return tuple(state)

    @staticmethod
    def _validate_observed_pedestrians(peds) -> None:
        """Reject absent or invalid active pedestrian velocities instead of holding."""
        positions = np.asarray(peds["positions"], float).reshape(-1, 2)
        velocities = np.asarray(peds["velocities"], float).reshape(-1, 2)
        count = float(np.asarray(peds["count"]).reshape(-1)[0])
        radius = float(np.asarray(peds["radius"]).reshape(-1)[0])
        if (
            not np.isfinite(count)
            or count != int(count)
            or not 0 <= count <= len(positions)
            or velocities.shape != positions.shape
            or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("orca_mpc requires complete finite CV pedestrian observations")

    def _initial_sequence(self, mean) -> np.ndarray:
        self.goal_seed = mean.copy()
        self.goal_seed[:, 0] = min(self.config.max_linear_speed, self.drive.max_linear_speed)
        mean[:] = self.seed_command
        self.seeds = [mean.copy(), self.goal_seed, np.zeros_like(mean)]
        if self.previous is not None:
            self.seeds.append(np.vstack((self.previous[1:], self.previous[-1:])))
        for turn in (-1.0, 0.0, 1.0):
            sequence = self.goal_seed.copy()
            sequence[: len(mean) // 2, 1] = turn * min(
                self.config.max_angular_speed, self.drive.max_angular_speed
            )
            self.seeds.append(sequence)
        self.best_cost = float("inf")
        self.best_sequence = None
        self.best_clearance = None
        return mean

    def _forecast(self, batch, *, braking_steps=0) -> tuple:
        return native_drive_rollout_batch(
            batch, self.drive, *self.current_vw, self.dt, braking_steps=braking_steps
        )

    def _prepare_candidates(self, batch) -> np.ndarray:
        batch[: len(self.seeds)] = self.seeds
        # Feasible endpoints themselves become the optimizer's command sequences.
        return self._forecast(batch)[2]

    def _batch_sequence_cost(self, *, batch, robot_pos, heading, goal, **_kwargs) -> np.ndarray:
        drive, cfg = self.drive, self.config
        tail = int(np.ceil(drive.max_linear_speed / drive.max_linear_decel / self.dt))
        local, _headings, _velocities = self._forecast(batch, braking_steps=tail)
        rotation = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )
        paths = np.concatenate(
            (np.broadcast_to(robot_pos, (len(batch), 1, 2)), local @ rotation + robot_pos), axis=1
        )
        # Bound chord interpolation error for a drive-limited continuous motion.
        padding = (
            (
                max(drive.max_linear_accel, drive.max_linear_decel)
                + drive.max_linear_speed * drive.max_angular_speed
            )
            * self.dt**2
            / 8
        )
        static = (
            np.full(len(batch), np.inf)
            if self._static_clearance.geometry.is_empty
            else distance(linestrings(paths), self._static_clearance.geometry)
            - drive.radius
            - padding
        )
        count = cfg.horizon_steps
        futures = self.predictor.predict(
            self.observation, horizon_steps=count, dt=self.dt
        ).positions_world
        _robot, _goal, peds = self._socnav_fields(self.observation)
        positions = np.asarray(peds["positions"], float).reshape(-1, 2)[: len(futures)]
        futures = np.concatenate((positions[:, None], futures), axis=1)
        radius = float(np.asarray(peds["radius"]).reshape(-1)[0])
        dynamic = (
            swept_relative_clearance(paths[:, : count + 1], futures, drive.radius + radius)
            - padding
        )
        minimum = np.minimum(static, dynamic)
        progress = np.linalg.norm(goal - robot_pos) - np.linalg.norm(paths[:, count] - goal, axis=1)
        smoothness = np.mean(
            np.sum(
                np.diff(
                    np.concatenate(
                        (np.broadcast_to(self.current_vw, (len(batch), 1, 2)), batch), axis=1
                    ),
                    axis=1,
                )
                ** 2,
                axis=-1,
            ),
            axis=1,
        )
        costs = (
            -cfg.goal_progress_weight * progress
            + cfg.clearance_weight / np.maximum(minimum + 0.05, 0.05)
            + cfg.smoothness_weight * smoothness
        )
        costs[minimum < cfg.hard_clearance] = np.inf
        index = int(np.argmin(costs))
        if costs[index] < self.best_cost:
            self.best_cost = float(costs[index])
            self.best_sequence = batch[index].copy()
            self.best_clearance = float(minimum[index])
        return costs

    def plan(self, observation) -> tuple[float, float]:
        if self.drive is None or self._static_clearance is None:
            raise RuntimeError("orca_mpc requires live drive and exact static geometry binding")
        self.best_sequence = None
        command = super().plan(observation)
        robot, _goal, _ = self._socnav_fields(observation)
        if np.linalg.norm(self.active_goal - robot["position"]) <= self.config.goal_tolerance:
            self.previous = None
            self.last = {"status": "goal_reached"}
            return tuple(float(v) for v in self._forecast(np.zeros((1, 1, 2)))[2][0, 0])
        if self.best_sequence is None:
            command = tuple(self._forecast(np.zeros((1, 1, 2)))[2][0, 0])
            self.previous = None
            self.last = {"status": "no_admissible_braking"}
        else:
            self.previous = self.best_sequence.copy()
            self.last = {
                "status": "admissible_sampled_sequence",
                "cost": self.best_cost,
                "minimum_predicted_clearance_m": self.best_clearance,
                "sequence_vw": self.best_sequence.tolist(),
            }
        return float(command[0]), float(command[1])


class ORCAMPCAdapter(HybridORCASamplerAdapter):
    """Reuse the hybrid ORCA/sampler composition and replan every control step."""

    def __init__(self, config=None, *, allow_fallback=False) -> None:
        """Build the native ORCA seed head and existing CEM search head."""
        if allow_fallback:
            raise ValueError("orca_mpc requires native ORCA; no fallback supported")
        payload = config or {}
        keys = {f.name for f in fields(ORCAMPCConfig)}
        cfg = ORCAMPCConfig(**{k: v for k, v in payload.items() if k in keys})
        orca = build_hybrid_orca_sampler_build_config(payload).socnav
        super().__init__(
            HybridORCASamplerConfig(goal_tolerance=cfg.goal_tolerance),
            orca_adapter=ORCAPlannerAdapter(orca, allow_fallback=False),
            sampler_adapter=_DriveMPCSampler(cfg),
        )
        self.mpc_config = cfg
        self._mpc_counts: dict[str, int] = {}

    def bind_env(self, env) -> None:
        """Bind live forward drive and map primitives without pedestrian futures."""
        drive = bound_drive_settings(env)
        if not isinstance(drive, DifferentialDriveSettings) or drive.min_linear_speed < 0:
            raise ValueError("orca_mpc v1 requires a forward differential drive")
        values = [
            drive.max_linear_accel,
            drive.max_linear_decel,
            drive.max_angular_accel,
            drive.max_linear_speed,
            drive.max_angular_speed,
            drive.radius,
            drive.wheel_radius,
            drive.interaxis_length,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("orca_mpc requires finite positive live drive limits")
        super().bind_env(env)
        self.primary_adapter.bind_env(env)
        self.sampler_adapter._bind_static_obstacles(env)
        if self.sampler_adapter._static_clearance is None:
            raise ValueError("orca_mpc requires exact map geometry")
        self.sampler_adapter.drive = drive

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the prior sequence, decisions and sampler RNG for each episode."""
        super().reset(seed=seed)
        self.sampler_adapter._rng = np.random.default_rng(
            self.mpc_config.random_seed if seed is None else seed
        )
        self.sampler_adapter.previous = None
        self.sampler_adapter.last = {}
        self._mpc_counts = {}

    def plan(self, observation) -> tuple[float, float]:
        """Seed the existing CEM optimizer from ORCA and execute its first command.

        Returns:
            Feasible first linear/angular endpoint, or flagged reachable braking.
        """
        seed = self.primary_adapter.plan(observation)
        self.sampler_adapter.seed_command = seed
        command = self.sampler_adapter.plan(observation)
        status = self.sampler_adapter.last["status"]
        self._mpc_counts[status] = self._mpc_counts.get(status, 0) + 1
        return command

    def diagnostics(self) -> dict[str, Any]:
        """Expose unresolved steps separately from sampled admissible decisions.

        Returns:
            Versioned empirical diagnostics including the native ORCA dependency.
        """
        return {
            "schema_version": "orca_mpc.v1",
            "decision_counts": dict(self._mpc_counts),
            "last_decision": dict(self.sampler_adapter.last),
            "orca": self.primary_adapter.diagnostics(),
            "prediction": "constant_velocity",
            "empirical_only": True,
        }


def build_orca_mpc_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the composite payload for the opt-in adapter builder.

    Returns:
        A shallow configuration copy; adapter construction validates it.
    """
    return dict(payload)
