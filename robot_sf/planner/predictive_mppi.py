"""Predictive MPPI/CEM local planner.

This planner reuses the learned pedestrian predictor from the predictive
planner, but optimizes a short action sequence instead of a single lattice
command. The executed control is the first action of the best sampled sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi_array
from robot_sf.planner.clearance_geometry import (
    CENTER_CLEARANCE_V1,
    obstacle_rollout_admissible,
    occupied_cell_clearance,
    pedestrian_clearance,
    surface_search_radius_cells,
    validate_clearance_model,
    validate_surface_clearance_radii,
)
from robot_sf.planner.goal_target import (
    LEGACY_NEXT_GOAL_V1,
    select_goal_target,
    validate_goal_target_version,
)
from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.socnav import (
    OccupancyAwarePlannerMixin,
    PredictionPlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.planner.socnav_base import _SOCNAV_CONFIG_INIT_KEYS
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings

_DEFAULT_ITERATIONS = 4
_DEFAULT_GOAL_PROGRESS_WEIGHT = 6.0
_DEFAULT_CLEARANCE_WEIGHT = 3.0
_DEFAULT_PROGRESS_ESCAPE_DISTANCE_M = 1.2


def _wrap_angle_batch(angle: np.ndarray) -> np.ndarray:
    """Batched angle wrapping to ``[-pi, pi)`` (numpy equivalent of ``_wrap_angle``).

    Returns:
        np.ndarray: Wrapped angles in ``[-pi, pi)``.
    """
    return wrap_angle_pi_array(angle)


@dataclass
class PredictiveMPPIConfig:
    """Configuration for :class:`PredictiveMPPIAdapter`."""

    socnav: SocNavPlannerConfig
    random_seed: int = 42
    horizon_steps: int = 12
    rollout_dt: float = 0.2
    sample_count: int = 128
    iterations: int = _DEFAULT_ITERATIONS
    elite_fraction: float = 0.2
    init_linear_std: float = 0.35
    init_angular_std: float = 0.65
    min_linear_std: float = 0.05
    min_angular_std: float = 0.08
    goal_tolerance: float = 0.25
    goal_target_version: str = LEGACY_NEXT_GOAL_V1
    max_linear_speed: float = 1.4
    max_angular_speed: float = 1.3
    near_distance: float = 0.7
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12
    hard_ped_clearance: float = 0.62
    hard_obstacle_clearance: float = 0.30
    first_step_ped_clearance: float = 0.75
    first_step_obstacle_clearance: float = 0.35
    invalid_sequence_cost: float = 1e6
    goal_progress_weight: float = _DEFAULT_GOAL_PROGRESS_WEIGHT
    heading_weight: float = 0.8
    clearance_weight: float = _DEFAULT_CLEARANCE_WEIGHT
    obstacle_weight: float = 1.6
    smoothness_weight: float = 0.2
    ttc_weight: float = 0.45
    occupancy_weight: float = 0.35
    anchor_bias_weight: float = 0.08
    progress_escape_enabled: bool = True
    progress_escape_distance: float = _DEFAULT_PROGRESS_ESCAPE_DISTANCE_M
    progress_escape_speed: float = 0.55
    progress_escape_heading_gain: float = 1.5
    clearance_model: str = CENTER_CLEARANCE_V1

    def __post_init__(self) -> None:
        """Validate route selection and physical geometry before execution."""
        validate_goal_target_version(self.goal_target_version)
        validate_clearance_model(self.clearance_model)
        radii = {
            "robot_radius": self.socnav.predictive_robot_radius,
            "pedestrian_radius": self.socnav.predictive_pedestrian_radius,
        }
        validate_surface_clearance_radii(self.clearance_model, **radii)
        validate_surface_clearance_radii(self.socnav.predictive_clearance_model, **radii)


class PredictiveMPPIAdapter(OccupancyAwarePlannerMixin):
    """Short-horizon sequence optimizer over learned pedestrian forecasts."""

    def __init__(self, config: PredictiveMPPIConfig, *, allow_fallback: bool = False) -> None:
        """Initialize predictive optimizer and deterministic RNG state."""
        self.config = config
        self._drive_settings = DifferentialDriveSettings()
        self._rng = np.random.default_rng(int(config.random_seed))
        self._no_admissible_command = False
        self._no_admissible_command_count = 0
        self._recovery_command = False
        self._recovery_command_count = 0
        self._predictor = PredictionPlannerAdapter(
            config=config.socnav,
            allow_fallback=allow_fallback,
        )

    def bind_env(self, env: Any) -> None:
        """Bind the episode's original static grid geometry."""
        self._bind_static_obstacles(env)
        self._drive_settings = DifferentialDriveSettings()
        robots = getattr(getattr(env, "simulator", None), "robots", None)
        if robots:
            settings = getattr(robots[0], "config", None)
            # Unsupported drives must retain the full margin. Standalone callers
            # without a drive use the production differential-drive defaults.
            self._drive_settings = (
                settings if isinstance(settings, DifferentialDriveSettings) else None
            )

    def _static_recovery_available(self) -> bool:
        """Require both exact static geometry and a supported drive model.

        Returns:
            bool: Whether recovery can be evaluated with native drive motion.
        """
        return super()._static_recovery_available() and self._drive_settings is not None

    def _in_static_recovery(self, current_obs: float) -> bool:
        """Identify the below-margin exception, never enabled for unbound grids.

        Returns:
            bool: Whether the initial clearance is in the recovery interval.
        """
        return self._static_recovery_available() and (
            0.0 < current_obs < float(self.config.hard_obstacle_clearance)
        )

    def _recovery_drive_rollout(
        self,
        sequence: np.ndarray,
        observation: dict[str, object],
        robot_pos: np.ndarray,
        heading: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Integrate recovery from observed velocity through the native drive.

        The velocity command is converted to acceleration just as in the map
        runner. Native motion clips acceleration/deceleration and uses wheel
        odometry. Check a terminal braking coast too: the prediction horizon
        must not turn a still-moving body into an instantaneous stop.

        Returns:
            tuple: Local positions, headings, and swept terminal-coast clearance.
        """
        robot_state, _, _ = self._predictor._socnav_fields(observation)
        _, _, speed, _ = self._extract_state(observation)
        angular = float(self._as_1d_float(robot_state.get("angular_velocity", [0.0]), pad=1)[0])
        drive = DifferentialDriveRobot(self._drive_settings)
        drive.state.velocity = (speed, angular)
        drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds((speed, angular))
        dt = float(self.config.rollout_dt)
        positions, headings = [], []
        for action in sequence:
            velocity = np.asarray(drive.current_speed)
            drive.apply_action(tuple((np.asarray(action) - velocity) / dt), dt)
            positions.append(drive.pos)
            headings.append(drive.pose[1])

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        previous = robot_pos + rotation @ np.asarray(drive.pos)
        coast_clearance = float("inf")
        while abs(drive.current_speed[0]) > 1e-9:
            velocity = np.asarray(drive.current_speed)
            drive.apply_action(tuple(-velocity / dt), dt)
            point = robot_pos + rotation @ np.asarray(drive.pos)
            coast_clearance = min(
                coast_clearance, self._exact_obstacle_clearance(point, previous=previous)
            )
            previous = point
        return np.asarray(positions), np.asarray(headings), coast_clearance

    def _extract_state(
        self, observation: dict[str, object]
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Extract robot pose/speed and active goal from structured observation.

        Returns:
            tuple[np.ndarray, float, float, np.ndarray]: Robot position, heading,
            linear speed, and resolved goal position.
        """
        robot_state, goal_state, _ped_state = self._predictor._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        heading = float(self._predictor._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        speed = float(self._predictor._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
        goal_next = self._predictor._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._predictor._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[
            :2
        ]
        goal = select_goal_target(
            robot_pos, goal_current, goal_next, version=self.config.goal_target_version
        )
        return robot_pos, heading, speed, goal

    def _predict_future(self, observation: dict[str, object]) -> tuple[np.ndarray, np.ndarray, int]:
        """Predict pedestrian futures and resolve an effective evaluation horizon.

        Returns:
            tuple[np.ndarray, np.ndarray, int]: Predicted pedestrian futures,
            validity mask, and rollout step count.
        """
        state, mask, _robot_pos, _robot_heading = self._predictor._build_model_input(observation)
        future = self._predictor._predict_trajectories(state, mask)
        learned_steps = self._predictor._effective_rollout_steps(future_peds=future, mask=mask)
        steps = min(
            max(1, int(self.config.horizon_steps)),
            max(1, int(future.shape[1])),
        )
        steps = min(max(steps, learned_steps), int(future.shape[1]))
        return future, mask, steps

    def _speed_cap(self, future: np.ndarray, mask: np.ndarray) -> float:
        """Apply the predictor's near-field risk cap to MPPI candidate speeds.

        Returns:
            float: Maximum allowed linear speed for this decision step.
        """
        ratio = self._predictor._risk_speed_cap_ratio(future_peds=future, mask=mask)
        return float(np.clip(ratio, 0.1, 1.0)) * float(self.config.max_linear_speed)

    def _min_obstacle_clearance(
        self,
        point: np.ndarray,
        observation: dict[str, object] | None = None,
        *,
        grid_payload: tuple[np.ndarray, dict[str, object]] | None = None,
    ) -> float:
        """Estimate obstacle clearance at a world-space point from occupancy grids.

        Returns:
            float: Minimum obstacle distance in meters, ``0.0`` when occupied.
        """
        exact = self._exact_obstacle_clearance(point)
        if exact is not None:
            return exact
        if grid_payload is None:
            if observation is None:
                raise ValueError(
                    "Predictive MPPI obstacle clearance requires observation "
                    "when grid_payload is absent"
                )
            grid_payload = self._extract_grid_payload(observation)
        if grid_payload is None:
            return float("inf")
        grid, meta = grid_payload
        channel = self._grid_channel_index(meta, "obstacles")
        if channel < 0:
            channel = self._preferred_channel(meta)
        if channel < 0 or channel >= grid.shape[0]:
            return float("inf")

        rc = self._world_to_grid(point, meta, grid_shape=(grid.shape[1], grid.shape[2]))
        if rc is None:
            return (
                -float(self.config.socnav.predictive_robot_radius)
                if self.config.clearance_model == "surface_v2"
                else 0.0
            )
        row, col = rc
        channel_grid = np.asarray(grid[channel], dtype=float)
        threshold = float(self.config.obstacle_threshold)
        if channel_grid[row, col] >= threshold and self.config.clearance_model != "surface_v2":
            return 0.0

        resolution = max(float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0]), 1e-6)
        radius = (
            surface_search_radius_cells(
                self.config.socnav.predictive_robot_radius,
                max(
                    self.config.hard_obstacle_clearance,
                    self.config.first_step_obstacle_clearance,
                    self.config.near_distance,
                ),
                resolution,
            )
            if self.config.clearance_model == "surface_v2"
            else max(int(self.config.obstacle_search_cells), 1)
        )
        r0 = max(0, row - radius)
        r1 = min(channel_grid.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(channel_grid.shape[1], col + radius + 1)
        window = channel_grid[r0:r1, c0:c1]
        obs_idx = np.argwhere(window >= threshold)
        if obs_idx.size == 0:
            return float("inf")

        dr = obs_idx[:, 0] + r0 - row
        dc = obs_idx[:, 1] + c0 - col
        return occupied_cell_clearance(
            dr,
            dc,
            resolution=resolution,
            model=self.config.clearance_model,
            robot_radius=self.config.socnav.predictive_robot_radius,
            point_offset_xy_m=self._point_offset_in_grid_cell(point, meta, row, col),
        )

    def _pedestrian_clearance(self, center_distance: float | np.ndarray) -> float | np.ndarray:
        """Convert pedestrian centre distance through the configured geometry model.

        Returns:
            float | np.ndarray: Centre or surface clearance in metres.
        """
        return pedestrian_clearance(
            center_distance,
            model=self.config.clearance_model,
            robot_radius=self.config.socnav.predictive_robot_radius,
            pedestrian_radius=self.config.socnav.predictive_pedestrian_radius,
        )

    def _sequence_rollout(  # noqa: PLR0913, PLR0915
        self,
        sequence: np.ndarray,
        *,
        robot_pos: np.ndarray,
        heading: float,
        goal: np.ndarray,
        future: np.ndarray,
        mask: np.ndarray,
        observation: dict[str, object],
        anchor_action: tuple[float, float],
        grid_payload: tuple[np.ndarray, dict[str, object]] | None = None,
    ) -> float:
        """Evaluate one control sequence; lower is better.

        Returns:
            float: Scalar sequence cost.
        """
        dt = float(self.config.rollout_dt)
        local_pos = np.zeros(2, dtype=float)
        local_heading = 0.0
        start_dist = float(np.linalg.norm(goal - robot_pos))
        min_clear = float("inf")
        min_obs = float("inf")
        first_clear = float("inf")
        first_obs = float("inf")
        ttc_penalty = smooth_penalty = anchor_penalty = 0.0
        prev_action = np.array([0.0, 0.0], dtype=float)
        anchor = np.asarray(anchor_action, dtype=float)
        future_steps = int(future.shape[1])
        valid_idx = np.where(mask > 0.5)[0]

        current_obs = self._min_obstacle_clearance(
            robot_pos, observation=observation, grid_payload=grid_payload
        )
        drive_rollout = None
        if self._in_static_recovery(current_obs):
            drive_rollout = self._recovery_drive_rollout(sequence, observation, robot_pos, heading)
            min_obs = min(min_obs, drive_rollout[2])

        previous_world = robot_pos
        for step, action in enumerate(sequence):
            v = float(action[0])
            w = float(action[1])
            if drive_rollout is None:
                local_pos = local_pos + np.array(
                    [v * np.cos(local_heading) * dt, v * np.sin(local_heading) * dt],
                    dtype=float,
                )
                local_heading = _wrap_angle(local_heading + w * dt)
            else:
                local_pos = drive_rollout[0][step]
                local_heading = drive_rollout[1][step]

            ped_idx = min(step, future_steps - 1)
            ped_t = future[:, ped_idx, :]
            if ped_t.size > 0 and valid_idx.size > 0:
                dists = np.linalg.norm(ped_t - local_pos[None, :], axis=1)
                valid_dist = dists[valid_idx]
                if valid_dist.size > 0:
                    clearance = self._pedestrian_clearance(valid_dist)
                    assert isinstance(clearance, np.ndarray)
                    min_clear = min(min_clear, float(np.min(clearance)))
                    if step == 0:
                        first_clear = min(first_clear, float(np.min(clearance)))
                    threshold = float(self.config.near_distance)
                    shortfall = np.maximum(0.0, threshold - clearance)
                    time_weight = 1.0 / (float(step + 1) * dt + 1e-6)
                    ttc_penalty += float(np.sum(shortfall * time_weight))

            cos_h = float(np.cos(heading))
            sin_h = float(np.sin(heading))
            world_point = robot_pos + np.array(
                [
                    cos_h * local_pos[0] - sin_h * local_pos[1],
                    sin_h * local_pos[0] + cos_h * local_pos[1],
                ],
                dtype=float,
            )
            obs_clear = self._obstacle_motion_clearance(
                world_point, previous_world, observation, grid_payload
            )
            previous_world = world_point
            min_obs = min(min_obs, obs_clear)
            if step == 0:
                first_obs = min(first_obs, obs_clear)
            smooth_penalty += float(np.linalg.norm(action - prev_action))
            anchor_penalty += float(np.linalg.norm(action - anchor))
            prev_action = np.asarray(action, dtype=float)

        hard_constraint_cost = self._hard_constraint_cost(
            min_clear=min_clear,
            min_obs=min_obs,
            first_clear=first_clear,
            first_obs=first_obs,
            current_obs=current_obs,
        )
        if hard_constraint_cost is not None:
            return hard_constraint_cost

        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        final_world = robot_pos + np.array(
            [
                cos_h * local_pos[0] - sin_h * local_pos[1],
                sin_h * local_pos[0] + cos_h * local_pos[1],
            ],
            dtype=float,
        )
        end_dist = float(np.linalg.norm(goal - final_world))
        progress = start_dist - end_dist
        goal_heading = float(np.arctan2(goal[1] - final_world[1], goal[0] - final_world[0]))
        heading_score = float(np.cos(_wrap_angle(goal_heading - (heading + local_heading))))

        mean_occ_penalty = 0.0
        if np.linalg.norm(local_pos) > 1e-6:
            direction = final_world - robot_pos
            obstacle_penalty, ped_penalty = self._predictor._path_penalty(
                robot_pos=robot_pos,
                direction=direction,
                observation=observation,
                base_distance=float(np.linalg.norm(final_world - robot_pos)),
                num_samples=max(2, int(sequence.shape[0])),
            )
            mean_occ_penalty = float(obstacle_penalty + 0.5 * ped_penalty)

        reward = (
            float(self.config.goal_progress_weight) * progress
            + float(self.config.heading_weight) * heading_score
            + float(self.config.clearance_weight) * min(min_clear, 2.0)
            + float(self.config.obstacle_weight) * min(min_obs, 2.0)
            - float(self.config.ttc_weight) * ttc_penalty
            - float(self.config.smoothness_weight) * smooth_penalty
            - float(self.config.occupancy_weight) * mean_occ_penalty
            - float(self.config.anchor_bias_weight) * anchor_penalty
        )
        return -reward

    def _batch_sequence_rollout(  # noqa: PLR0913, C901, PLR0915, PLR0912
        self,
        batch: np.ndarray,
        *,
        robot_pos: np.ndarray,
        heading: float,
        goal: np.ndarray,
        future: np.ndarray,
        mask: np.ndarray,
        observation: dict[str, object],
        anchor_action: tuple[float, float],
        grid_payload: tuple[np.ndarray, dict[str, object]] | None = None,
    ) -> np.ndarray:
        """Vectorized rollout evaluation for a batch of control sequences.

        Integrates all samples through the unicycle model in local coordinates,
        evaluates pedestrian clearance, TTC, obstacle clearance, and hard constraints.
        Per-step pose integration is bit-stable with the scalar loop.

        Returns:
            np.ndarray: Shape ``(samples,)`` of sequence costs (lower is better).
        """
        dt = float(self.config.rollout_dt)
        horizon = int(batch.shape[1])
        samples = int(batch.shape[0])
        start_dist = float(np.linalg.norm(goal - robot_pos))
        future_steps = int(future.shape[1])
        valid_idx = np.where(mask > 0.5)[0]

        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        anchor = np.asarray(anchor_action, dtype=float)

        # Per-sample accumulators (local frame)
        local_pos = np.zeros((samples, 2), dtype=float)
        local_heading = np.zeros(samples, dtype=float)
        min_clear = np.full(samples, float("inf"), dtype=float)
        min_obs = np.full(samples, float("inf"), dtype=float)
        first_clear = np.full(samples, float("inf"), dtype=float)
        first_obs = np.full(samples, float("inf"), dtype=float)
        ttc_pen = np.zeros(samples, dtype=float)
        smooth_pen = np.zeros(samples, dtype=float)
        anchor_pen = np.zeros(samples, dtype=float)
        prev_action = np.zeros((samples, 2), dtype=float)

        current_obs = self._min_obstacle_clearance(
            robot_pos, observation=observation, grid_payload=grid_payload
        )
        drive_rollouts = None
        if self._in_static_recovery(current_obs):
            drive_rollouts = [
                self._recovery_drive_rollout(sequence, observation, robot_pos, heading)
                for sequence in batch
            ]
            min_obs = np.minimum(min_obs, [rollout[2] for rollout in drive_rollouts])

        previous_world = np.tile(robot_pos, (samples, 1))
        for step in range(horizon):
            v = batch[:, step, 0]  # (samples,)
            w = batch[:, step, 1]  # (samples,)

            if drive_rollouts is None:
                local_pos = local_pos + np.column_stack(
                    [v * np.cos(local_heading) * dt, v * np.sin(local_heading) * dt]
                )
                local_heading = _wrap_angle_batch(local_heading + w * dt)
            else:
                local_pos = np.asarray([rollout[0][step] for rollout in drive_rollouts])
                local_heading = np.asarray([rollout[1][step] for rollout in drive_rollouts])

            ped_idx = min(step, future_steps - 1)
            ped_t = future[:, ped_idx, :]  # (peds, 2)
            if ped_t.size > 0 and valid_idx.size > 0:
                dists = np.linalg.norm(ped_t - local_pos[:, None, :], axis=2)  # (samples, peds)
                valid_dist = dists[:, valid_idx]  # (samples, valid_peds)
                if valid_dist.size > 0:
                    clearance = self._pedestrian_clearance(valid_dist)
                    assert isinstance(clearance, np.ndarray)
                    sample_min = np.min(clearance, axis=1)  # (samples,)
                    min_clear = np.minimum(min_clear, sample_min)
                    if step == 0:
                        first_clear = np.minimum(first_clear, sample_min)
                    threshold = float(self.config.near_distance)
                    shortfall = np.maximum(0.0, threshold - clearance)
                    time_weight = 1.0 / ((step + 1) * dt + 1e-6)
                    ttc_pen += np.sum(shortfall * time_weight, axis=1)

            # World coordinates for obstacle check
            world_pos_x = robot_pos[0] + cos_h * local_pos[:, 0] - sin_h * local_pos[:, 1]
            world_pos_y = robot_pos[1] + sin_h * local_pos[:, 0] + cos_h * local_pos[:, 1]

            # Obstacle clearance per sample
            for s in range(samples):
                obs_clear = self._min_obstacle_clearance(
                    np.array([world_pos_x[s], world_pos_y[s]], dtype=float),
                    observation=observation,
                    grid_payload=grid_payload,
                )
                point = np.array([world_pos_x[s], world_pos_y[s]])
                swept = self._exact_obstacle_clearance(point, previous=previous_world[s])
                if swept is not None:
                    obs_clear = min(obs_clear, swept)
                previous_world[s] = point
                min_obs[s] = min(min_obs[s], obs_clear)
                if step == 0:
                    first_obs[s] = min(first_obs[s], obs_clear)

            action_col = batch[:, step, :]  # (samples, 2)
            smooth_pen += np.linalg.norm(action_col - prev_action, axis=1)
            anchor_pen += np.linalg.norm(action_col - anchor, axis=1)
            prev_action = action_col

        # Hard constraint rejection per sample
        costs = np.full(samples, float(self.config.invalid_sequence_cost), dtype=float)
        alive = np.ones(samples, dtype=bool)
        for s in range(samples):
            hc = self._hard_constraint_cost(
                min_clear=float(min_clear[s]),
                min_obs=float(min_obs[s]),
                first_clear=float(first_clear[s]),
                first_obs=float(first_obs[s]),
                current_obs=current_obs,
            )
            if hc is not None:
                alive[s] = False
                costs[s] = hc

        if not np.any(alive):
            return costs

        # Final scoring for surviving samples
        final_local = local_pos[alive]
        final_world_x = robot_pos[0] + cos_h * final_local[:, 0] - sin_h * final_local[:, 1]
        final_world_y = robot_pos[1] + sin_h * final_local[:, 0] + cos_h * final_local[:, 1]
        final_world = np.column_stack([final_world_x, final_world_y])

        end_dist = np.linalg.norm(goal - final_world, axis=1)
        progress = start_dist - end_dist
        goal_headings = np.arctan2(goal[1] - final_world_y, goal[0] - final_world_x)
        heading_score = np.cos(_wrap_angle_batch(goal_headings - (heading + local_heading[alive])))

        mean_occ = np.zeros(np.sum(alive), dtype=float)
        alive_local_norms = np.linalg.norm(final_local, axis=1)
        occ_needed = alive_local_norms > 1e-6
        if np.any(occ_needed):
            # Process each sample needing occupancy separately (predictor call)
            for i in range(np.sum(alive)):
                if occ_needed[i]:
                    direction = final_world[i] - robot_pos
                    obstacle_pen, ped_pen = self._predictor._path_penalty(
                        robot_pos=robot_pos,
                        direction=direction,
                        observation=observation,
                        base_distance=float(np.linalg.norm(final_world[i] - robot_pos)),
                        num_samples=max(2, int(horizon)),
                    )
                    mean_occ[i] = float(obstacle_pen + 0.5 * ped_pen)

        reward = (
            float(self.config.goal_progress_weight) * progress
            + float(self.config.heading_weight) * heading_score
            + float(self.config.clearance_weight) * np.minimum(min_clear[alive], 2.0)
            + float(self.config.obstacle_weight) * np.minimum(min_obs[alive], 2.0)
            - float(self.config.ttc_weight) * ttc_pen[alive]
            - float(self.config.smoothness_weight) * smooth_pen[alive]
            - float(self.config.occupancy_weight) * mean_occ
            - float(self.config.anchor_bias_weight) * anchor_pen[alive]
        )
        costs[alive] = -reward
        return costs

    def _constant_sequence(self, action: tuple[float, float], horizon: int) -> np.ndarray:
        """Build a fixed-action control sequence for arbitration candidates.

        Returns:
            np.ndarray: Array with shape ``(horizon, 2)`` containing repeated actions.
        """
        seq = np.zeros((max(1, int(horizon)), 2), dtype=float)
        seq[:, 0] = float(action[0])
        seq[:, 1] = float(action[1])
        return seq

    def _hard_constraint_cost(
        self,
        *,
        min_clear: float,
        min_obs: float,
        first_clear: float,
        first_obs: float,
        current_obs: float = float("inf"),
    ) -> float | None:
        """Return a large penalty for unsafe sequences, otherwise ``None``.

        Returns:
            float | None: Hard rejection cost for unsafe sequences, otherwise ``None``.
        """
        if min_clear < float(self.config.hard_ped_clearance):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.hard_ped_clearance) - min_clear) * 1e3
            )
        recovery = self._in_static_recovery(current_obs)
        if (
            not obstacle_rollout_admissible(
                current_obs if self._static_recovery_available() else float("inf"),
                min_obs,
                float(self.config.hard_obstacle_clearance),
            )
            if self.config.clearance_model == "surface_v2"
            else min_obs < float(self.config.hard_obstacle_clearance)
        ):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.hard_obstacle_clearance) - min_obs) * 1e3
            )
        if first_clear < float(self.config.first_step_ped_clearance):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.first_step_ped_clearance) - first_clear) * 5e2
            )
        if not recovery and first_obs < float(self.config.first_step_obstacle_clearance):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.first_step_obstacle_clearance) - first_obs) * 5e2
            )
        return None

    def _recovery_rotation(  # noqa: PLR0913
        self,
        current_obs,
        robot_pos,
        heading,
        goal,
        horizon,
        future,
        mask,
        observation,
        anchor_action,
        grid_payload,
    ):
        """Score a recovery turn including actual drive-limited braking coast.

        Returns:
            tuple | None: Command and cost, or None outside static recovery.
        """
        if not self._in_static_recovery(current_obs):
            return None
        target_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
        turn = float(
            np.clip(
                _wrap_angle(target_heading - heading),
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )
        rotation = (0.0, turn)
        cost = self._sequence_rollout(
            self._constant_sequence(rotation, horizon),
            robot_pos=robot_pos,
            heading=heading,
            goal=goal,
            future=future,
            mask=mask,
            observation=observation,
            anchor_action=anchor_action,
            grid_payload=grid_payload,
        )
        return np.asarray(rotation), cost

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        """Return the first action from the best sampled control sequence."""
        self._no_admissible_command = False
        self._recovery_command = False
        # Recovery rollouts consume observed speed and yaw rate, including
        # sampled/anchor/stop sequences, rather than assuming the body is at rest.
        robot_pos, heading, _, goal = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        future, mask, steps = self._predict_future(observation)
        speed_cap = self._speed_cap(future, mask)
        anchor_action = self._predictor.plan(observation)

        horizon = max(1, int(steps))
        samples = max(int(self.config.sample_count), 8)
        iterations = max(int(self.config.iterations), 1)
        elite_n = max(2, round(samples * float(self.config.elite_fraction)))

        grid_payload = self._cache_grid_payload(observation)

        mean = np.zeros((horizon, 2), dtype=float)
        mean[:, 0] = min(float(anchor_action[0]), speed_cap)
        mean[:, 1] = np.clip(
            float(anchor_action[1]),
            -float(self.config.max_angular_speed),
            float(self.config.max_angular_speed),
        )
        std = np.zeros_like(mean)
        std[:, 0] = float(self.config.init_linear_std)
        std[:, 1] = float(self.config.init_angular_std)

        best_sequence = mean.copy()
        best_cost = self._sequence_rollout(
            mean,
            robot_pos=robot_pos,
            heading=heading,
            goal=goal,
            future=future,
            mask=mask,
            observation=observation,
            anchor_action=anchor_action,
            grid_payload=grid_payload,
        )

        for _ in range(iterations):
            noise = self._rng.normal(0.0, 1.0, size=(samples, horizon, 2))
            batch = mean[None, :, :] + noise * std[None, :, :]
            batch[:, :, 0] = np.clip(batch[:, :, 0], 0.0, speed_cap)
            batch[:, :, 1] = np.clip(
                batch[:, :, 1],
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
            batch[0] = mean

            costs = self._batch_sequence_rollout(
                batch,
                robot_pos=robot_pos,
                heading=heading,
                goal=goal,
                future=future,
                mask=mask,
                observation=observation,
                anchor_action=anchor_action,
                grid_payload=grid_payload,
            )
            elite_idx = np.argsort(costs)[:elite_n]
            elites = batch[elite_idx]
            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0)
            std[:, 0] = np.maximum(std[:, 0], float(self.config.min_linear_std))
            std[:, 1] = np.maximum(std[:, 1], float(self.config.min_angular_std))
            if float(costs[elite_idx[0]]) < best_cost:
                best_cost = float(costs[elite_idx[0]])
                best_sequence = batch[elite_idx[0]].copy()

        arbitration: list[tuple[np.ndarray, float]] = [
            (
                best_sequence[0].copy(),
                self._sequence_rollout(
                    self._constant_sequence(
                        (float(best_sequence[0, 0]), float(best_sequence[0, 1])), horizon
                    ),
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    future=future,
                    mask=mask,
                    observation=observation,
                    anchor_action=anchor_action,
                    grid_payload=grid_payload,
                ),
            ),
            (
                np.asarray(anchor_action, dtype=float),
                self._sequence_rollout(
                    self._constant_sequence(anchor_action, horizon),
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    future=future,
                    mask=mask,
                    observation=observation,
                    anchor_action=anchor_action,
                    grid_payload=grid_payload,
                ),
            ),
            (
                np.zeros(2, dtype=float),
                self._sequence_rollout(
                    self._constant_sequence((0.0, 0.0), horizon),
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    future=future,
                    mask=mask,
                    observation=observation,
                    anchor_action=anchor_action,
                    grid_payload=grid_payload,
                ),
            ),
        ]
        current_obs = self._min_obstacle_clearance(
            robot_pos, observation=observation, grid_payload=grid_payload
        )
        recovery_rotation = self._recovery_rotation(
            current_obs,
            robot_pos,
            heading,
            goal,
            horizon,
            future,
            mask,
            observation,
            anchor_action,
            grid_payload,
        )
        if recovery_rotation is not None:
            arbitration.append(recovery_rotation)
        selected_action, selected_cost = min(arbitration, key=lambda item: float(item[1]))
        action = selected_action
        if (
            recovery_rotation is not None
            and action[0] == 0.0
            and action[1] == 0.0
            and recovery_rotation[1] < float(self.config.invalid_sequence_cost)
        ):
            # Prefer a feasible turn over stasis. Its score already includes
            # constrained braking from the actual observed drive velocity.
            action, selected_cost = recovery_rotation
        if bool(self.config.progress_escape_enabled):
            goal_dist = float(np.linalg.norm(goal - robot_pos))
            if (
                goal_dist > float(self.config.progress_escape_distance)
                and float(action[0]) < float(self.config.progress_escape_speed) * 0.6
            ):
                goal_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
                heading_err = _wrap_angle(goal_heading - heading)
                forced_action = np.zeros(2, dtype=float)
                forced_action[0] = float(np.clip(self.config.progress_escape_speed, 0.0, speed_cap))
                forced_action[1] = float(
                    np.clip(
                        heading_err * float(self.config.progress_escape_heading_gain),
                        -float(self.config.max_angular_speed),
                        float(self.config.max_angular_speed),
                    )
                )
                forced_cost = self._sequence_rollout(
                    self._constant_sequence(
                        (float(forced_action[0]), float(forced_action[1])),
                        horizon,
                    ),
                    robot_pos=robot_pos,
                    heading=heading,
                    goal=goal,
                    future=future,
                    mask=mask,
                    observation=observation,
                    anchor_action=anchor_action,
                    grid_payload=grid_payload,
                )
                if forced_cost < float(self.config.invalid_sequence_cost):
                    action = forced_action
                    selected_cost = forced_cost
        self._record_admissibility(current_obs, selected_cost)
        if self._no_admissible_command:
            action = np.zeros(2)
        return float(action[0]), float(action[1])

    def _record_admissibility(self, current_obs: float, selected_cost: float) -> None:
        """Record final selected-command feasibility and recovery diagnostics."""
        self._no_admissible_command = selected_cost >= float(self.config.invalid_sequence_cost)
        self._no_admissible_command_count += int(self._no_admissible_command)
        self._recovery_command = bool(
            self._in_static_recovery(current_obs) and not self._no_admissible_command
        )
        self._recovery_command_count += int(self._recovery_command)

    def diagnostics(self) -> dict[str, Any]:
        """Return execution diagnostics."""
        decision = {
            "no_admissible_command": self._no_admissible_command,
            "no_admissible_command_count": self._no_admissible_command_count,
            "recovery_command": self._recovery_command,
            "recovery_command_count": self._recovery_command_count,
        }
        return {"planner_type": "PredictiveMPPIAdapter", **decision, "last_decision": decision}

    def foresight_diagnostics(self) -> dict[str, Any]:
        """Expose the nested predictor's checkpoint-load and fallback provenance.

        Returns:
            Structured predictor runtime provenance for benchmark admission.
        """
        return self._predictor.foresight_diagnostics()

    def foresight_degraded(self) -> bool:
        """Return whether the nested predictor used its constant-velocity fallback."""
        return self._predictor.foresight_degraded()


def build_predictive_mppi_config(cfg: dict[str, object] | None) -> PredictiveMPPIConfig:
    """Build :class:`PredictiveMPPIConfig` from a root mapping payload.

    Returns:
        PredictiveMPPIConfig: Parsed planner configuration.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    socnav_allowed = {
        field.name for field in fields(SocNavPlannerConfig)
    } | _SOCNAV_CONFIG_INIT_KEYS
    socnav_kwargs = {key: value for key, value in cfg.items() if key in socnav_allowed}
    socnav = SocNavPlannerConfig(**socnav_kwargs)
    return PredictiveMPPIConfig(
        socnav=socnav,
        random_seed=int(cfg.get("random_seed", 42)),
        horizon_steps=int(cfg.get("horizon_steps", 12)),
        rollout_dt=float(cfg.get("rollout_dt", socnav.predictive_rollout_dt)),
        sample_count=int(cfg.get("sample_count", 128)),
        iterations=int(cfg.get("iterations", _DEFAULT_ITERATIONS)),
        elite_fraction=float(cfg.get("elite_fraction", 0.2)),
        init_linear_std=float(cfg.get("init_linear_std", 0.35)),
        init_angular_std=float(cfg.get("init_angular_std", 0.65)),
        min_linear_std=float(cfg.get("min_linear_std", 0.05)),
        min_angular_std=float(cfg.get("min_angular_std", 0.08)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        goal_target_version=str(cfg.get("goal_target_version", LEGACY_NEXT_GOAL_V1)),
        max_linear_speed=float(cfg.get("max_linear_speed", 1.4)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.3)),
        near_distance=float(cfg.get("near_distance", 0.7)),
        obstacle_threshold=float(cfg.get("obstacle_threshold", 0.5)),
        obstacle_search_cells=int(cfg.get("obstacle_search_cells", 12)),
        hard_ped_clearance=float(cfg.get("hard_ped_clearance", 0.62)),
        hard_obstacle_clearance=float(cfg.get("hard_obstacle_clearance", 0.30)),
        first_step_ped_clearance=float(cfg.get("first_step_ped_clearance", 0.75)),
        first_step_obstacle_clearance=float(cfg.get("first_step_obstacle_clearance", 0.35)),
        invalid_sequence_cost=float(cfg.get("invalid_sequence_cost", 1e6)),
        goal_progress_weight=float(cfg.get("goal_progress_weight", _DEFAULT_GOAL_PROGRESS_WEIGHT)),
        heading_weight=float(cfg.get("heading_weight", 0.8)),
        clearance_weight=float(cfg.get("clearance_weight", _DEFAULT_CLEARANCE_WEIGHT)),
        obstacle_weight=float(cfg.get("obstacle_weight", 1.6)),
        smoothness_weight=float(cfg.get("smoothness_weight", 0.2)),
        ttc_weight=float(cfg.get("ttc_weight", 0.45)),
        occupancy_weight=float(cfg.get("occupancy_weight", 0.35)),
        anchor_bias_weight=float(cfg.get("anchor_bias_weight", 0.08)),
        progress_escape_enabled=bool(cfg.get("progress_escape_enabled", True)),
        progress_escape_distance=float(
            cfg.get("progress_escape_distance", _DEFAULT_PROGRESS_ESCAPE_DISTANCE_M)
        ),
        progress_escape_speed=float(cfg.get("progress_escape_speed", 0.55)),
        progress_escape_heading_gain=float(cfg.get("progress_escape_heading_gain", 1.5)),
        clearance_model=str(cfg.get("clearance_model", CENTER_CLEARANCE_V1)),
    )


__all__ = [
    "PredictiveMPPIAdapter",
    "PredictiveMPPIConfig",
    "build_predictive_mppi_config",
]
