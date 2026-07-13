"""Sampling-based social local planner (MPPI/CEM style).

This planner optimizes short control sequences against a risk-aware objective and
returns the first action of the best sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.risk_dwa import _wrap_angle
from robot_sf.planner.socnav import OccupancyAwarePlannerMixin


def _wrap_angle_batch(angle: np.ndarray) -> np.ndarray:
    """Batched angle wrapping to ``[-pi, pi)`` (numpy equivalent of ``_wrap_angle``).

    Returns:
        np.ndarray: Wrapped angles in ``[-pi, pi)``.
    """
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


@dataclass
class MPPISocialConfig:
    """Configuration for :class:`MPPISocialPlannerAdapter`."""

    random_seed: int = 42
    max_linear_speed: float = 1.2
    max_angular_speed: float = 1.2

    horizon_steps: int = 10
    rollout_dt: float = 0.2
    sample_count: int = 96
    iterations: int = 3
    elite_fraction: float = 0.2

    init_linear_std: float = 0.35
    init_angular_std: float = 0.7
    min_linear_std: float = 0.06
    min_angular_std: float = 0.10

    goal_tolerance: float = 0.25
    near_field_distance: float = 2.5
    near_field_speed_cap: float = 0.55
    density_norm_count: float = 8.0

    goal_progress_weight: float = 4.0
    heading_weight: float = 1.0
    clearance_weight: float = 2.0
    obstacle_weight: float = 1.4
    smoothness_weight: float = 0.2
    ttc_weight: float = 0.35

    near_distance: float = 0.7
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12

    prediction_backend: str = "constant_velocity"
    progress_escape_enabled: bool = True
    progress_escape_distance: float = 1.0
    progress_escape_speed: float = 0.45
    progress_escape_heading_gain: float = 1.4


class MPPISocialPlannerAdapter(OccupancyAwarePlannerMixin):
    """Stochastic optimizer over short unicycle control sequences."""

    def __init__(self, config: MPPISocialConfig | None = None) -> None:
        """Initialize adapter and deterministic RNG state."""
        self.config = config or MPPISocialConfig()
        self._rng = np.random.default_rng(int(self.config.random_seed))

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract robot, goal, and pedestrian state from a SocNav observation.

        Returns:
            tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
            Robot position, heading, speed, goal, pedestrian positions, and
            pedestrian velocities.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        speed = float(self._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
        goal_next = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        goal = goal_next if np.linalg.norm(goal_next - robot_pos) > 1e-6 else goal_current

        ped_positions_raw = ped_state.get("positions")
        ped_velocities_raw = ped_state.get("velocities")
        ped_pos = np.asarray([] if ped_positions_raw is None else ped_positions_raw, dtype=float)
        ped_vel = np.asarray([] if ped_velocities_raw is None else ped_velocities_raw, dtype=float)
        if ped_pos.ndim != 2 or ped_pos.shape[-1] != 2:
            ped_pos = np.zeros((0, 2), dtype=float)
        if ped_vel.ndim != 2 or ped_vel.shape[-1] != 2 or ped_vel.shape[0] != ped_pos.shape[0]:
            ped_vel = np.zeros_like(ped_pos)

        return robot_pos, heading, speed, goal, ped_pos, ped_vel

    def _crowd_speed_cap(self, robot_pos: np.ndarray, ped_pos: np.ndarray) -> float:
        """Reduce maximum rollout speed as near-field pedestrian density increases.

        Returns:
            float: Density-adjusted speed cap.
        """
        if ped_pos.size == 0:
            return float(self.config.max_linear_speed)
        dists = np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)
        near_count = float(np.sum(dists <= float(self.config.near_field_distance)))
        density = near_count / max(float(self.config.density_norm_count), 1.0)
        cap = float(self.config.max_linear_speed) * (1.0 - density)
        cap = max(float(self.config.near_field_speed_cap), cap)
        return float(np.clip(cap, 0.05, float(self.config.max_linear_speed)))

    def _predict_ped_positions(
        self, ped_pos: np.ndarray, ped_vel: np.ndarray, t: float
    ) -> np.ndarray:
        """Predict pedestrian positions at one rollout timestamp.

        Returns:
            np.ndarray: Predicted pedestrian positions in world coordinates.
        """
        backend = str(self.config.prediction_backend).strip().lower()
        if backend in {"constant_velocity", "learned"}:
            # `learned` currently falls back to CV in this adapter implementation.
            return ped_pos + ped_vel * float(t)
        if backend == "stationary":
            return ped_pos
        return ped_pos + ped_vel * float(t)

    def _min_obstacle_clearance(
        self,
        point: np.ndarray,
        observation: dict[str, Any] | None = None,
        *,
        grid_payload: tuple[np.ndarray, dict[str, Any]] | None = None,
    ) -> float:
        """Estimate grid-obstacle clearance around one rollout point.

        Returns:
            float: Approximate obstacle clearance in meters, ``inf`` when unknown.
        """
        if grid_payload is None:
            assert observation is not None
            grid_payload = self._extract_grid_payload(observation)
        if grid_payload is None:
            return float("inf")
        grid, meta = grid_payload
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
        cell_dist_sq = dr.astype(float) ** 2 + dc.astype(float) ** 2
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        return float(np.sqrt(np.min(cell_dist_sq)) * max(resolution, 1e-6))

    def _sequence_cost(  # noqa: PLR0913
        self,
        *,
        sequence: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
        current_speed: float,
        goal: np.ndarray,
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
        observation: dict[str, Any],
        grid_payload: tuple[np.ndarray, dict[str, Any]] | None = None,
    ) -> float:
        """Evaluate one sampled MPPI control sequence.

        Returns:
            float: Lower-is-better rollout cost.
        """
        dt = float(self.config.rollout_dt)
        x = np.array(robot_pos, dtype=float)
        theta = float(heading)
        start_dist = float(np.linalg.norm(goal - x))
        min_clear = float("inf")
        min_obs = float("inf")
        min_ttc = float("inf")

        prev_v = float(current_speed)
        smooth_penalty = 0.0
        for step, action in enumerate(sequence):
            v = float(action[0])
            w = float(action[1])
            x = x + np.array([v * np.cos(theta) * dt, v * np.sin(theta) * dt])
            theta = _wrap_angle(theta + w * dt)
            t = (step + 1) * dt

            if ped_pos.size > 0:
                ped_t = self._predict_ped_positions(ped_pos, ped_vel, t)
                dists = np.linalg.norm(ped_t - x[None, :], axis=1)
                if dists.size > 0:
                    min_clear = min(min_clear, float(np.min(dists)))
                rel_pos = ped_t - x[None, :]
                robot_vel = np.array([v * np.cos(theta), v * np.sin(theta)], dtype=float)
                rel_vel = ped_vel - robot_vel[None, :]
                rel_speed_sq = np.sum(rel_vel * rel_vel, axis=1)
                valid = rel_speed_sq > 1e-6
                if np.any(valid):
                    ttc = -np.sum(rel_pos[valid] * rel_vel[valid], axis=1) / rel_speed_sq[valid]
                    ttc = ttc[ttc > 0.0]
                    if ttc.size > 0:
                        min_ttc = min(min_ttc, float(np.min(ttc)))

            min_obs = min(
                min_obs,
                self._min_obstacle_clearance(x, observation=observation, grid_payload=grid_payload),
            )
            smooth_penalty += abs(v - prev_v) + 0.2 * abs(w)
            prev_v = v

        end_dist = float(np.linalg.norm(goal - x))
        progress = start_dist - end_dist
        goal_heading = float(np.arctan2(goal[1] - x[1], goal[0] - x[0]))
        heading_score = float(np.cos(_wrap_angle(goal_heading - theta)))
        near_penalty = max(0.0, float(self.config.near_distance) - min_clear)
        obs_penalty = max(0.0, float(self.config.near_distance) - min_obs)
        ttc_term = 0.0 if not np.isfinite(min_ttc) else 1.0 / max(min_ttc, 1e-3)

        reward = (
            float(self.config.goal_progress_weight) * progress
            + float(self.config.heading_weight) * heading_score
            + float(self.config.clearance_weight) * min(min_clear, 2.0)
            + float(self.config.obstacle_weight) * min(min_obs, 2.0)
            - 2.0 * near_penalty
            - 1.5 * obs_penalty
            - float(self.config.ttc_weight) * ttc_term
            - float(self.config.smoothness_weight) * smooth_penalty
        )
        return -reward

    def _batch_sequence_cost(  # noqa: PLR0913
        self,
        *,
        batch: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
        current_speed: float,
        goal: np.ndarray,
        ped_pos: np.ndarray,
        ped_vel: np.ndarray,
        observation: dict[str, Any],
        grid_payload: tuple[np.ndarray, dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Vectorized cost evaluation for a batch of control sequences.

        Integrates all samples through the unicycle model and evaluates pedestrian
        clearance, TTC, smoothness, and grid obstacles. Per-step pose integration
        is bit-stable with the scalar loop. Pedestrian clearance is broadcast as a
        ``(samples, steps, peds)`` tensor with a single ``min`` reduction. Obstacle
        clearance still iterates per-step (grid lookup) but covers all samples at once.

        Returns:
            np.ndarray: Shape ``(samples,)`` of sequence costs (lower is better).
        """
        dt = float(self.config.rollout_dt)
        horizon = int(batch.shape[1])
        samples = int(batch.shape[0])
        start_dist = float(np.linalg.norm(goal - robot_pos))

        # Per-sample accumulators
        pos = np.full((samples, 2), float(robot_pos[0]), dtype=float)
        pos[:, 1] = float(robot_pos[1])
        theta = np.full(samples, float(heading), dtype=float)
        min_clear = np.full(samples, float("inf"), dtype=float)
        min_obs = np.full(samples, float("inf"), dtype=float)
        min_ttc = np.full(samples, float("inf"), dtype=float)
        smooth_pen = np.zeros(samples, dtype=float)
        prev_v = np.full(samples, float(current_speed), dtype=float)

        for step in range(horizon):
            v = batch[:, step, 0]  # (samples,)
            w = batch[:, step, 1]  # (samples,)

            pos = pos + np.column_stack([v * np.cos(theta) * dt, v * np.sin(theta) * dt])
            theta = _wrap_angle_batch(theta + w * dt)
            t = (step + 1) * dt

            if ped_pos.size > 0:
                ped_t = self._predict_ped_positions(ped_pos, ped_vel, t)  # (peds, 2)
                # (samples, 1, 2) - (1, peds, 2) -> (samples, peds)
                dists = np.linalg.norm(pos[:, None, :] - ped_t[None, :, :], axis=2)
                sample_min = np.min(dists, axis=1)
                min_clear = np.minimum(min_clear, sample_min)

                # TTC computation
                robot_vel = np.column_stack([v * np.cos(theta), v * np.sin(theta)])  # (samples, 2)
                rel_pos = ped_t[None, :, :] - pos[:, None, :]  # (samples, peds, 2)
                rel_vel = ped_vel[None, :, :] - robot_vel[:, None, :]  # (samples, peds, 2)
                rel_speed_sq = np.sum(rel_vel * rel_vel, axis=2)  # (samples, peds)
                rel_dot = -np.sum(rel_pos * rel_vel, axis=2)  # (samples, peds)
                valid = rel_speed_sq > 1e-6
                # Where valid, compute ttc; else 0
                ttc_vals = np.where(
                    valid, np.where(rel_dot > 0, rel_dot / rel_speed_sq, -1.0), -1.0
                )
                # Take min positive TTC per sample
                pos_ttc = np.where((ttc_vals > 0), ttc_vals, float("inf"))
                sample_min_ttc = np.min(pos_ttc, axis=1)
                min_ttc = np.minimum(min_ttc, sample_min_ttc)

            # Obstacle clearance per sample at this step
            for s in range(samples):
                obs_clear = self._min_obstacle_clearance(
                    pos[s], observation=observation, grid_payload=grid_payload
                )
                min_obs[s] = min(min_obs[s], obs_clear)

            smooth_pen += np.abs(v - prev_v) + 0.2 * np.abs(w)
            prev_v = v

        end_dist = np.linalg.norm(goal - pos, axis=1)
        progress = start_dist - end_dist

        goal_headings = np.arctan2(goal[1] - pos[:, 1], goal[0] - pos[:, 0])
        heading_score = np.cos(_wrap_angle_batch(goal_headings - theta))

        near_pen = np.maximum(0.0, float(self.config.near_distance) - min_clear)
        obs_pen = np.maximum(0.0, float(self.config.near_distance) - min_obs)

        ttc_term = np.where(
            np.isfinite(min_ttc),
            1.0 / np.maximum(min_ttc, 1e-3),
            0.0,
        )

        reward = (
            float(self.config.goal_progress_weight) * progress
            + float(self.config.heading_weight) * heading_score
            + float(self.config.clearance_weight) * np.minimum(min_clear, 2.0)
            + float(self.config.obstacle_weight) * np.minimum(min_obs, 2.0)
            - 2.0 * near_pen
            - 1.5 * obs_pen
            - float(self.config.ttc_weight) * ttc_term
            - float(self.config.smoothness_weight) * smooth_pen
        )
        return -reward

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return MPPI/CEM optimized `(v, omega)` command."""
        robot_pos, heading, speed, goal, ped_pos, ped_vel = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        speed_cap = self._crowd_speed_cap(robot_pos, ped_pos)
        goal_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
        heading_err = _wrap_angle(goal_heading - heading)

        horizon = max(int(self.config.horizon_steps), 1)
        samples = max(int(self.config.sample_count), 8)
        iterations = max(int(self.config.iterations), 1)
        elite_n = max(2, round(samples * float(self.config.elite_fraction)))

        grid_payload = self._cache_grid_payload(observation)

        mean = np.zeros((horizon, 2), dtype=float)
        mean[:, 0] = min(speed_cap, max(0.0, speed + 0.1))
        mean[:, 1] = np.clip(
            heading_err / max(horizon * float(self.config.rollout_dt), 1e-3),
            -self.config.max_angular_speed,
            self.config.max_angular_speed,
        )

        std = np.zeros_like(mean)
        std[:, 0] = float(self.config.init_linear_std)
        std[:, 1] = float(self.config.init_angular_std)

        best_sequence = mean.copy()
        best_cost = float("inf")

        for _ in range(iterations):
            noise = self._rng.normal(0.0, 1.0, size=(samples, horizon, 2))
            batch = mean[None, :, :] + noise * std[None, :, :]
            batch[:, :, 0] = np.clip(batch[:, :, 0], 0.0, speed_cap)
            batch[:, :, 1] = np.clip(
                batch[:, :, 1],
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )

            costs = self._batch_sequence_cost(
                batch=batch,
                robot_pos=robot_pos,
                heading=heading,
                current_speed=speed,
                goal=goal,
                ped_pos=ped_pos,
                ped_vel=ped_vel,
                observation=observation,
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

        action = best_sequence[0]
        if bool(self.config.progress_escape_enabled):
            to_goal = float(np.linalg.norm(goal - robot_pos))
            if (
                to_goal > float(self.config.progress_escape_distance)
                and float(action[0]) < float(self.config.progress_escape_speed) * 0.6
            ):
                goal_heading = float(np.arctan2(goal[1] - robot_pos[1], goal[0] - robot_pos[0]))
                heading_err = _wrap_angle(goal_heading - heading)
                action[0] = float(np.clip(self.config.progress_escape_speed, 0.0, speed_cap))
                action[1] = float(
                    np.clip(
                        heading_err * float(self.config.progress_escape_heading_gain),
                        -float(self.config.max_angular_speed),
                        float(self.config.max_angular_speed),
                    )
                )
        return float(action[0]), float(action[1])


def build_mppi_social_config(cfg: dict[str, Any] | None) -> MPPISocialConfig:
    """Build :class:`MPPISocialConfig` from mapping payload.

    Returns:
        MPPISocialConfig: Parsed planner config.
    """
    if not isinstance(cfg, dict):
        return MPPISocialConfig()
    return MPPISocialConfig(
        random_seed=int(cfg.get("random_seed", 42)),
        max_linear_speed=float(cfg.get("max_linear_speed", 1.2)),
        max_angular_speed=float(cfg.get("max_angular_speed", 1.2)),
        horizon_steps=int(cfg.get("horizon_steps", 10)),
        rollout_dt=float(cfg.get("rollout_dt", 0.2)),
        sample_count=int(cfg.get("sample_count", 96)),
        iterations=int(cfg.get("iterations", 3)),
        elite_fraction=float(cfg.get("elite_fraction", 0.2)),
        init_linear_std=float(cfg.get("init_linear_std", 0.35)),
        init_angular_std=float(cfg.get("init_angular_std", 0.7)),
        min_linear_std=float(cfg.get("min_linear_std", 0.06)),
        min_angular_std=float(cfg.get("min_angular_std", 0.1)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        near_field_distance=float(cfg.get("near_field_distance", 2.5)),
        near_field_speed_cap=float(cfg.get("near_field_speed_cap", 0.55)),
        density_norm_count=float(cfg.get("density_norm_count", 8.0)),
        goal_progress_weight=float(cfg.get("goal_progress_weight", 4.0)),
        heading_weight=float(cfg.get("heading_weight", 1.0)),
        clearance_weight=float(cfg.get("clearance_weight", 2.0)),
        obstacle_weight=float(cfg.get("obstacle_weight", 1.4)),
        smoothness_weight=float(cfg.get("smoothness_weight", 0.2)),
        ttc_weight=float(cfg.get("ttc_weight", 0.35)),
        near_distance=float(cfg.get("near_distance", 0.7)),
        obstacle_threshold=float(cfg.get("obstacle_threshold", 0.5)),
        obstacle_search_cells=int(cfg.get("obstacle_search_cells", 12)),
        prediction_backend=str(cfg.get("prediction_backend", "constant_velocity")),
        progress_escape_enabled=bool(cfg.get("progress_escape_enabled", True)),
        progress_escape_distance=float(cfg.get("progress_escape_distance", 1.0)),
        progress_escape_speed=float(cfg.get("progress_escape_speed", 0.45)),
        progress_escape_heading_gain=float(cfg.get("progress_escape_heading_gain", 1.4)),
    )


__all__ = [
    "MPPISocialConfig",
    "MPPISocialPlannerAdapter",
    "build_mppi_social_config",
]
