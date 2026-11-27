"""Dynamic risk-aware MPPI based on Trevisan et al. (arXiv:2506.21205v2)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import atan2, cos, isclose, sin
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtri
from scipy.stats import qmc

from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.planner.dra_mppi_corridor import braking_path_is_clear, passing_controls, passing_lane
from robot_sf.planner.dra_mppi_crossing import crossing_spans, select_stoppable_plan
from robot_sf.planner.drive_rollout import native_drive_rollout_batch
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin
from robot_sf.robot.differential_drive import DifferentialDriveSettings

from .dra_mppi_risk import joint_collision_probabilities

# Section V. The simulator's 20 Hz increment determines the diffusion per second.
_PREDICTION_INCREMENT_S = 0.05
_PEDESTRIAN_VELOCITY_STD = 0.30
_MC_COLUMNS = 160
_MC_ROWS = 125
_SPLINE_KNOTS = 4
_ETA_LOW = 5.0
_ETA_HIGH = 15.0


@dataclass(frozen=True)
class DRAMPPIConfig:
    """Controller parameters and the published unimodal experiment settings."""

    horizon_steps: int = 20
    rollout_dt: float = 0.2
    sample_count: int = 400
    risk_threshold: float = 0.05
    robot_radius_m: float = DEFAULT_ROBOT_RADIUS
    max_linear_speed: float = 1.5
    max_angular_speed: float = 1.0
    reference_speed: float = 1.5
    linear_noise_std: float = 0.8
    angular_noise_std: float = 0.75
    tracking_weight: float = 2.0
    lateral_weight: float = 0.5
    speed_weight: float = 1.0
    rotation_weight: float = 0.15
    risk_soft_weight: float = 10.0
    risk_hard_weight: float = 40.0
    random_seed: int = 1031
    corridor_passing: bool = False
    crossing_yield: bool = False

    def __post_init__(self) -> None:
        """Reject invalid dimensions, nonfinite coefficients and negative weights."""
        if self.horizon_steps < _SPLINE_KNOTS or self.sample_count < 2:
            raise ValueError("DRA-MPPI requires at least four stages and two control samples")
        positive = (
            self.rollout_dt,
            self.robot_radius_m,
            self.max_linear_speed,
            self.max_angular_speed,
            self.reference_speed,
            self.linear_noise_std,
            self.angular_noise_std,
            self.tracking_weight,
            self.risk_hard_weight,
        )
        nonnegative = (
            self.lateral_weight,
            self.speed_weight,
            self.rotation_weight,
            self.risk_soft_weight,
        )
        if not np.isfinite((*positive, *nonnegative)).all() or min(positive) <= 0:
            raise ValueError("DRA-MPPI requires finite positive timestep, limits and scales")
        if min(nonnegative) < 0 or not 0 < self.risk_threshold < 1:
            raise ValueError("DRA-MPPI requires nonnegative weights and a threshold in (0, 1)")
        if not isinstance(self.corridor_passing, bool):
            raise ValueError("DRA-MPPI corridor_passing must be a boolean")
        if not isinstance(self.crossing_yield, bool):
            raise ValueError("DRA-MPPI crossing_yield must be a boolean")


def build_dra_mppi_config(config: dict[str, Any]) -> DRAMPPIConfig:
    """Parse controller parameters from a benchmark algorithm configuration.

    Returns:
        The configuration for the dynamic risk-aware MPPI controller.
    """
    return DRAMPPIConfig(
        **{
            field.name: config[field.name]
            for field in fields(DRAMPPIConfig)
            if field.name in config
        }
    )


class DRAMPPIAdapter(OccupancyAwarePlannerMixin):
    """Receding-horizon control with a stage-wise probability of pedestrian contact."""

    def __init__(self, config: DRAMPPIConfig) -> None:
        """Build the Halton Gaussian splines used at each decision."""
        self.config = config
        config_times = np.arange(config.horizon_steps, dtype=float)
        knot_times = np.linspace(0, config.horizon_steps - 1, _SPLINE_KNOTS)
        halton = qmc.Halton(2 * _SPLINE_KNOTS, scramble=True, rng=config.random_seed)
        knots = ndtri(halton.random(config.sample_count - 1)).reshape(-1, _SPLINE_KNOTS, 2)
        self._noise = CubicSpline(knot_times, knots, axis=1, bc_type="natural")(config_times)
        self._noise *= [config.linear_noise_std, config.angular_noise_std]
        self._times = (config_times + 1) * config.rollout_dt
        self._forecast_std = _PEDESTRIAN_VELOCITY_STD * np.sqrt(
            _PREDICTION_INCREMENT_S * self._times
        )
        self._drive_settings: DifferentialDriveSettings | None = None
        self.reset()

    def reset(self) -> None:
        """Clear episode-local warm start, reference segment, and diagnostics."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._nominal = np.zeros((self.config.horizon_steps, 2))
        self._beta = 1.0
        self._reference_start: np.ndarray | None = None
        self._last_target_xy: tuple[float, float] | None = None
        self._planning_steps = 0
        self._no_static_rollout_steps = 0
        self._static_blend_rejections = 0
        self._all_rollouts_over_risk = 0
        self._passing_side = 0
        self._corridor_steps = 0
        self._risk_blend_rejections = 0
        self._corridor_yield_steps = 0
        self._yield_guard_rejections = 0
        self._crossings = np.empty((0, 2))
        self._crossing_direction = np.zeros(2)
        self._crossing_checks = 0
        self._crossing_replans = 0
        self._crossing_yields = 0

    def bind_env(self, env: Any) -> None:
        """Read the same static geometry and drive limits used for simulation."""
        self._drive_settings = None
        self._bind_static_obstacles(env)
        robots = getattr(getattr(env, "simulator", None), "robots", None)
        settings = getattr(robots[0], "config", None) if robots else None
        if not isinstance(settings, DifferentialDriveSettings) or self._static_clearance is None:
            raise ValueError("DRA-MPPI requires a differential drive and continuous map geometry")
        if not isclose(settings.radius, self.config.robot_radius_m, abs_tol=1e-6):
            raise ValueError("DRA-MPPI footprint must match the simulated differential drive")
        self._drive_settings = settings

    def _rollouts(
        self,
        controls: np.ndarray,
        position: np.ndarray,
        heading: float,
        velocity: np.ndarray,
        dt: float,
        substeps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world-frame positions, polar velocities and static feasibility."""
        drive = self._drive_settings
        obstacles = self._static_clearance
        assert drive is not None and obstacles is not None
        local, _, velocities = native_drive_rollout_batch(
            np.repeat(controls, substeps, axis=1), drive, velocity[0], velocity[1], dt
        )
        c, s = cos(heading), sin(heading)
        x, y = local[..., 0], local[..., 1]
        world = position + np.stack((c * x - s * y, s * x + c * y), axis=-1)
        points = np.concatenate((np.broadcast_to(position, (len(world), 1, 2)), world), axis=1)
        clearance = obstacles.rollout_clearance(points, self.config.robot_radius_m)
        return (
            world[:, substeps - 1 :: substeps],
            velocities[:, substeps - 1 :: substeps],
            clearance > 0.0,
        )

    def _stage_costs(
        self, world: np.ndarray, velocities: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Track the active waypoint and reference segment while minimizing rotation.

        Returns:
            The tracking, speed and rotation costs at every rollout stage.
        """
        config = self.config
        delta = world - goal
        distance = np.linalg.norm(delta, axis=2)
        assert self._reference_start is not None
        segment = goal - self._reference_start
        norm = np.linalg.norm(segment)
        lateral = np.zeros_like(distance)
        if norm > 0.0:
            unit_x, unit_y = segment / norm
            lateral = delta[..., 0] * unit_y - delta[..., 1] * unit_x

        # Near the active waypoint, use the reference speed the drive can brake from.
        drive = self._drive_settings
        assert drive is not None and drive.max_linear_decel is not None
        reference = np.minimum(
            config.reference_speed, np.sqrt(2.0 * drive.max_linear_decel * distance)
        )
        return (
            config.tracking_weight * distance
            + config.lateral_weight * lateral**2
            + config.speed_weight * (velocities[..., 0] - reference) ** 2
            + config.rotation_weight * velocities[..., 1] ** 2
        )

    def _normalize_pedestrians(self, ped_state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Normalize pedestrian positions and ego-frame velocities.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(positions_world, velocities_ego)`` arrays.
        """
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        if ped_positions.ndim == 1:
            ped_positions = (
                ped_positions.reshape(-1, 2) if ped_positions.size % 2 == 0 else np.zeros((0, 2))
            )
        elif ped_positions.ndim != 2:
            ped_positions = np.zeros((0, 2), dtype=float)
        if ped_positions.ndim == 2 and ped_positions.shape[1] != 2:
            ped_positions = (
                ped_positions[:, :2]
                if ped_positions.shape[1] > 2
                else np.pad(
                    ped_positions,
                    ((0, 0), (0, 2 - ped_positions.shape[1])),
                    constant_values=0.0,
                )
            )

        ped_count = int(
            self._as_1d_float(ped_state.get("count", [ped_positions.shape[0]]), pad=1)[0]
        )
        ped_count = max(0, min(ped_count, int(ped_positions.shape[0])))
        ped_positions = ped_positions[:ped_count]

        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_velocities.ndim == 1:
            ped_velocities = (
                ped_velocities.reshape(-1, 2)
                if ped_velocities.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )
        elif ped_velocities.ndim != 2:
            ped_velocities = np.zeros((0, 2), dtype=float)
        if ped_velocities.ndim == 2 and ped_velocities.shape[1] != 2:
            ped_velocities = (
                ped_velocities[:, :2]
                if ped_velocities.shape[1] > 2
                else np.pad(
                    ped_velocities,
                    ((0, 0), (0, 2 - ped_velocities.shape[1])),
                    constant_values=0.0,
                )
            )
        if ped_velocities.shape[0] < ped_count:
            ped_velocities = np.pad(
                ped_velocities,
                ((0, ped_count - ped_velocities.shape[0]), (0, 0)),
                constant_values=0.0,
            )
        ped_velocities = ped_velocities[:ped_count]
        return ped_positions, ped_velocities

    def _planning_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Extract world-frame state and update the active tracking segment.

        Returns:
            Robot pose and velocity, target, pedestrian positions, world velocities and radius.
        """
        robot, goal_state, pedestrians = self._socnav_fields(observation)
        position = np.asarray(robot["position"], dtype=float).reshape(2)
        heading = float(self._as_1d_float(robot["heading"])[0])
        velocity = self._as_1d_float(robot["speed"], pad=2)[:2]
        if "angular_velocity" in robot:
            velocity = np.array([velocity[0], self._as_1d_float(robot["angular_velocity"])[0]])
        goal = np.asarray(goal_state["current"], dtype=float).reshape(2)
        if self._last_target_xy is None or tuple(goal) != self._last_target_xy:
            self._reference_start = (
                position.copy()
                if self._last_target_xy is None
                else np.asarray(self._last_target_xy)
            )
            self._last_target_xy = (float(goal[0]), float(goal[1]))
            self._passing_side = 0
            if self.config.crossing_yield:
                assert self._static_clearance is not None
                self._crossings, self._crossing_direction = crossing_spans(
                    self._static_clearance, self._reference_start, goal, self.config.robot_radius_m
                )

        peds, ego_velocity = self._normalize_pedestrians(pedestrians)
        c, s = cos(heading), sin(heading)
        ped_velocity = ego_velocity @ np.array([[c, s], [-s, c]])
        ped_radius = float(self._as_1d_float(pedestrians.get("radius", [0.0]))[0])

        return position, heading, velocity, goal, peds, ped_velocity, ped_radius

    def _risks(
        self,
        points: np.ndarray,
        means: np.ndarray,
        radius: float,
        shifts: np.ndarray,
    ) -> np.ndarray:
        """Integrate pedestrian forecasts over the contact disk at each stage.

        Returns:
            Joint contact probabilities at the rollout stage endpoints.
        """
        return joint_collision_probabilities(
            points, means, self._forecast_std, radius, shifts, _MC_COLUMNS, _MC_ROWS
        )

    def _weighted_plan(self, controls: np.ndarray, total: np.ndarray, best: int) -> np.ndarray:
        """Average sampled controls using the MPPI weights and update the temperature.

        Returns:
            The proposed control sequence.
        """
        weights = np.exp(-(total - total[best]) / self._beta)
        eta = float(weights.sum())
        plan = ((weights / eta)[:, None, None] * controls).sum(axis=0)
        if eta < _ETA_LOW:
            self._beta = min(3.0, self._beta * 1.1)
        elif eta > _ETA_HIGH:
            self._beta = max(0.05, self._beta * 0.9)
        return plan

    def _near_crossing(self, position: np.ndarray) -> bool:
        """Cover the approach and the stretch where merging traffic may stay ahead.

        Returns:
            Whether the robot is within the crossing or its approach and exit distances.
        """
        progress = (position - self._reference_start) @ self._crossing_direction
        radius = self.config.robot_radius_m
        approaching = progress >= self._crossings[:, 0] - 6.0 * radius
        leaving = progress <= self._crossings[:, 1] + 10.0 * radius
        return bool(np.any(approaching & leaving))

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:  # noqa: C901, PLR0915
        """Apply Algorithm 1 with optional corridor passing and crossing yield.

        Returns:
            Linear speed in metres per second and angular speed in radians per second.
        """
        config = self.config
        drive = self._drive_settings
        obstacles = getattr(self, "_static_clearance", None)
        if drive is None or obstacles is None:
            raise RuntimeError("DRA-MPPI requires a bound differential drive and map geometry")
        dt = self._simulation_timestep(observation)
        substeps = round(config.rollout_dt / dt)
        if substeps < 1 or not isclose(substeps * dt, config.rollout_dt, abs_tol=1e-8):
            raise ValueError("DRA-MPPI rollout stages must be a multiple of the control timestep")

        position, heading, velocity, goal, peds, ped_velocity, ped_radius = self._planning_state(
            observation
        )
        assert self._reference_start is not None
        means = peds[:, None, :] + self._times[None, :, None] * ped_velocity[:, None, :]
        if not np.isfinite(means).all() or (len(peds) and not 0 < ped_radius < np.inf):
            raise ValueError(
                "DRA-MPPI requires finite pedestrian forecasts and a positive footprint"
            )

        linear_limit = min(config.max_linear_speed, drive.max_linear_speed)
        angular_limit = min(config.max_angular_speed, drive.max_angular_speed)
        contact_radius = config.robot_radius_m + ped_radius
        lane = None
        if config.corridor_passing and len(peds):
            lane = passing_lane(
                obstacles,
                position,
                np.stack((self._reference_start, goal)),
                peds,
                ped_velocity,
                config.robot_radius_m,
                contact_radius,
                self._passing_side,
            )
        self._passing_side = 0 if lane is None else lane.side
        self._corridor_steps += int(lane is not None)
        controls = np.zeros((config.sample_count, config.horizon_steps, 2))
        controls[1:] = np.clip(
            self._nominal + self._noise, [0.0, -angular_limit], [linear_limit, angular_limit]
        )
        if lane is not None:
            lane_bank, yield_turn = passing_controls(
                (lane.point - position) @ lane.normal,
                heading - atan2(lane.direction[1], lane.direction[0]),
                config.horizon_steps,
                config.rollout_dt,
                linear_limit,
                angular_limit,
                1.3 * config.robot_radius_m,
            )
            count = min(len(lane_bank), len(controls) - 1)
            controls[-count:] = lane_bank[np.linspace(0, len(lane_bank) - 1, count, dtype=int)]
        world, velocities, admissible = self._rollouts(
            controls, position, heading, velocity, dt, substeps
        )
        self._planning_steps += 1
        if not admissible.any():
            self._no_static_rollout_steps += 1
            self._nominal.fill(0.0)
            return 0.0, 0.0

        costs = self._stage_costs(world, velocities, goal)
        if lane is not None:
            costs += 3.0 * ((world - lane.point) @ lane.normal) ** 2
        if len(means):
            shifts = self._rng.random((config.horizon_steps, 2))
            risks = self._risks(world, means, contact_radius, shifts)
            above = risks > config.risk_threshold
            costs += config.risk_soft_weight * risks + config.risk_hard_weight * above
            risk_feasible = admissible & ~above.any(axis=1)
            has_risk_feasible = bool(risk_feasible.any())
            self._all_rollouts_over_risk += int(not has_risk_feasible)
            if lane is not None:
                if not has_risk_feasible:
                    command = (0.0, float(yield_turn))
                    clear = braking_path_is_clear(
                        command,
                        position,
                        heading,
                        velocity,
                        dt,
                        drive,
                        obstacles,
                        peds,
                        ped_velocity,
                        contact_radius,
                    )
                    self._corridor_yield_steps += 1
                    self._yield_guard_rejections += int(not clear)
                    self._nominal.fill(0.0)
                    return command if clear else (0.0, 0.0)
                admissible = risk_feasible
        total = np.where(admissible, costs.sum(axis=1), np.inf)
        best = int(total.argmin())
        plan = self._weighted_plan(controls, total, best)

        # The averaged controls require their own constraint check.
        blend_world, _, blend_valid = self._rollouts(
            plan[None, ...], position, heading, velocity, dt, substeps
        )
        if not blend_valid[0]:
            self._static_blend_rejections += 1
            plan = controls[best]
        elif lane is not None:
            risk = self._risks(blend_world, means, contact_radius, shifts)
            if (risk > config.risk_threshold).any():
                self._risk_blend_rejections += 1
                plan = controls[best]
        if config.crossing_yield and lane is None and len(peds) and self._near_crossing(position):
            self._crossing_checks += 1
            stoppable = select_stoppable_plan(
                plan,
                controls,
                total,
                position,
                heading,
                velocity,
                dt,
                drive,
                obstacles,
                peds,
                ped_velocity,
                contact_radius,
            )
            if stoppable is None:
                self._crossing_yields += 1
                self._nominal.fill(0.0)
                return 0.0, 0.0
            self._crossing_replans += int(stoppable is not plan)
            plan = stoppable
        shifted = np.concatenate((plan[1:], plan[-1:]), axis=0)
        self._nominal = plan + (shifted - plan) / substeps
        return float(plan[0, 0]), float(plan[0, 1])

    def diagnostics(self) -> dict[str, Any]:
        """Expose observations about the online controller for the benchmark runner.

        Returns:
            Prediction identity and counts of static and probabilistic constraint events.
        """
        return {
            "planner_type": type(self).__name__,
            "pedestrian_prediction": "constant_velocity_gaussian_diffusion",
            "risk_estimator": "shared_shifted_grid_marginal_then_joint",
            "plan_steps": self._planning_steps,
            "no_static_rollout_steps": self._no_static_rollout_steps,
            "static_blend_rejections": self._static_blend_rejections,
            "all_rollouts_over_risk_steps": self._all_rollouts_over_risk,
            "corridor_steps": self._corridor_steps,
            "risk_blend_rejections": self._risk_blend_rejections,
            "corridor_yield_steps": self._corridor_yield_steps,
            "yield_guard_rejections": self._yield_guard_rejections,
            **(
                {
                    "crossing_checks": self._crossing_checks,
                    "crossing_replans": self._crossing_replans,
                    "crossing_yields": self._crossing_yields,
                }
                if self.config.crossing_yield
                else {}
            ),
        }
