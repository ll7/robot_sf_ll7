"""Predictive MPPI/CEM local planner.

This planner reuses the learned pedestrian predictor from the predictive
planner, but optimizes a short action sequence instead of a single lattice
command. The executed control is the first action of the best sampled sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.socnav import (
    OccupancyAwarePlannerMixin,
    PredictionPlannerAdapter,
    SocNavPlannerConfig,
)


@dataclass
class PredictiveMPPIConfig:
    """Configuration for :class:`PredictiveMPPIAdapter`."""

    socnav: SocNavPlannerConfig
    random_seed: int = 42
    horizon_steps: int = 12
    rollout_dt: float = 0.2
    sample_count: int = 128
    iterations: int = 4
    elite_fraction: float = 0.2
    init_linear_std: float = 0.35
    init_angular_std: float = 0.65
    min_linear_std: float = 0.05
    min_angular_std: float = 0.08
    goal_tolerance: float = 0.25
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
    goal_progress_weight: float = 6.0
    heading_weight: float = 0.8
    clearance_weight: float = 3.0
    obstacle_weight: float = 1.6
    smoothness_weight: float = 0.2
    ttc_weight: float = 0.45
    occupancy_weight: float = 0.35
    anchor_bias_weight: float = 0.08
    progress_escape_enabled: bool = True
    progress_escape_distance: float = 1.2
    progress_escape_speed: float = 0.55
    progress_escape_heading_gain: float = 1.5


class PredictiveMPPIAdapter(OccupancyAwarePlannerMixin):
    """Short-horizon sequence optimizer over learned pedestrian forecasts."""

    def __init__(self, config: PredictiveMPPIConfig, *, allow_fallback: bool = False) -> None:
        """Initialize predictive optimizer and deterministic RNG state."""
        self.config = config
        self._rng = np.random.default_rng(int(config.random_seed))
        self._predictor = PredictionPlannerAdapter(
            config=config.socnav,
            allow_fallback=allow_fallback,
        )

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
        goal = goal_next if np.linalg.norm(goal_next - robot_pos) > 1e-6 else goal_current
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

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, object]) -> float:
        """Estimate obstacle clearance at a world-space point from occupancy grids.

        Returns:
            float: Minimum obstacle distance in meters, ``0.0`` when occupied.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return float("inf")
        grid, meta = payload
        channel = self._grid_channel_index(meta, "obstacles")
        if channel < 0:
            channel = self._preferred_channel(meta)
        if channel < 0 or channel >= grid.shape[0]:
            return float("inf")

        rc = self._world_to_grid(point, meta, grid_shape=(grid.shape[1], grid.shape[2]))
        if rc is None:
            return 0.0
        row, col = rc
        channel_grid = np.asarray(grid[channel], dtype=float)
        threshold = float(self.config.obstacle_threshold)
        if channel_grid[row, col] >= threshold:
            return 0.0

        radius = max(int(self.config.obstacle_search_cells), 1)
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
        cell_dist = np.sqrt(dr.astype(float) ** 2 + dc.astype(float) ** 2)
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        return float(np.min(cell_dist) * max(resolution, 1e-6))

    def _sequence_rollout(
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
        ttc_penalty = 0.0
        smooth_penalty = 0.0
        anchor_penalty = 0.0
        prev_action = np.array([0.0, 0.0], dtype=float)
        anchor = np.asarray(anchor_action, dtype=float)
        future_steps = int(future.shape[1])

        for step, action in enumerate(sequence):
            v = float(action[0])
            w = float(action[1])
            local_pos = local_pos + np.array(
                [v * np.cos(local_heading) * dt, v * np.sin(local_heading) * dt],
                dtype=float,
            )
            local_heading = _wrap_angle(local_heading + w * dt)

            ped_idx = min(step, future_steps - 1)
            ped_t = future[:, ped_idx, :]
            if ped_t.size > 0:
                dists = np.linalg.norm(ped_t - local_pos[None, :], axis=1)
                valid_dist = dists[mask > 0.5]
                if valid_dist.size > 0:
                    min_clear = min(min_clear, float(np.min(valid_dist)))
                    if step == 0:
                        first_clear = min(first_clear, float(np.min(valid_dist)))
                    threshold = float(self.config.near_distance)
                    shortfall = np.maximum(0.0, threshold - valid_dist)
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
            obs_clear = self._min_obstacle_clearance(world_point, observation)
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
        if min_obs < float(self.config.hard_obstacle_clearance):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.hard_obstacle_clearance) - min_obs) * 1e3
            )
        if first_clear < float(self.config.first_step_ped_clearance):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.first_step_ped_clearance) - first_clear) * 5e2
            )
        if first_obs < float(self.config.first_step_obstacle_clearance):
            return (
                float(self.config.invalid_sequence_cost)
                + (float(self.config.first_step_obstacle_clearance) - first_obs) * 5e2
            )
        return None

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        """Return the first action from the best sampled control sequence."""
        robot_pos, heading, _speed, goal = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        future, mask, steps = self._predict_future(observation)
        speed_cap = self._speed_cap(future, mask)
        anchor_action = self._predictor.plan(observation)

        horizon = max(1, int(steps))
        samples = max(int(self.config.sample_count), 8)
        iterations = max(int(self.config.iterations), 1)
        elite_n = max(2, round(samples * float(self.config.elite_fraction)))

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

            costs = np.asarray(
                [
                    self._sequence_rollout(
                        batch[i],
                        robot_pos=robot_pos,
                        heading=heading,
                        goal=goal,
                        future=future,
                        mask=mask,
                        observation=observation,
                        anchor_action=anchor_action,
                    )
                    for i in range(samples)
                ],
                dtype=float,
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
                ),
            ),
        ]
        action = min(arbitration, key=lambda item: float(item[1]))[0]
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
                )
                if forced_cost < float(self.config.invalid_sequence_cost):
                    action = forced_action
        return float(action[0]), float(action[1])


def build_predictive_mppi_config(cfg: dict[str, object] | None) -> PredictiveMPPIConfig:
    """Build :class:`PredictiveMPPIConfig` from a root mapping payload.

    Returns:
        PredictiveMPPIConfig: Parsed planner configuration.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    socnav_allowed = {field.name for field in fields(SocNavPlannerConfig)}
    socnav_kwargs = {key: value for key, value in cfg.items() if key in socnav_allowed}
    socnav = SocNavPlannerConfig(**socnav_kwargs)
    return PredictiveMPPIConfig(
        socnav=socnav,
        random_seed=int(cfg.get("random_seed", 42)),
        horizon_steps=int(cfg.get("horizon_steps", 12)),
        rollout_dt=float(cfg.get("rollout_dt", socnav.predictive_rollout_dt)),
        sample_count=int(cfg.get("sample_count", 128)),
        iterations=int(cfg.get("iterations", 4)),
        elite_fraction=float(cfg.get("elite_fraction", 0.2)),
        init_linear_std=float(cfg.get("init_linear_std", 0.35)),
        init_angular_std=float(cfg.get("init_angular_std", 0.65)),
        min_linear_std=float(cfg.get("min_linear_std", 0.05)),
        min_angular_std=float(cfg.get("min_angular_std", 0.08)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
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
        goal_progress_weight=float(cfg.get("goal_progress_weight", 6.0)),
        heading_weight=float(cfg.get("heading_weight", 0.8)),
        clearance_weight=float(cfg.get("clearance_weight", 3.0)),
        obstacle_weight=float(cfg.get("obstacle_weight", 1.6)),
        smoothness_weight=float(cfg.get("smoothness_weight", 0.2)),
        ttc_weight=float(cfg.get("ttc_weight", 0.45)),
        occupancy_weight=float(cfg.get("occupancy_weight", 0.35)),
        anchor_bias_weight=float(cfg.get("anchor_bias_weight", 0.08)),
        progress_escape_enabled=bool(cfg.get("progress_escape_enabled", True)),
        progress_escape_distance=float(cfg.get("progress_escape_distance", 1.2)),
        progress_escape_speed=float(cfg.get("progress_escape_speed", 0.55)),
        progress_escape_heading_gain=float(cfg.get("progress_escape_heading_gain", 1.5)),
    )


__all__ = [
    "PredictiveMPPIAdapter",
    "PredictiveMPPIConfig",
    "build_predictive_mppi_config",
]
