"""Compact predictor-derived foresight features for PPO and benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from gymnasium import spaces

from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig


@dataclass(slots=True)
class PredictiveForesightConfig:
    """Configuration for compact predictor-derived observation features."""

    enabled: bool = False
    model_id: str = "predictive_proxy_selected_v2_full"
    checkpoint_path: str | None = None
    device: str = "cpu"
    max_agents: int = 16
    horizon_steps: int = 8
    rollout_dt: float = 0.2
    ego_conditioning: bool = False
    near_distance: float = 0.7
    front_corridor_length: float = 3.0
    front_corridor_half_width: float = 1.0


def predictive_foresight_config_from_source(
    source: Any,
    *,
    default_max_agents: int = 16,
) -> PredictiveForesightConfig:
    """Build a foresight config from an object exposing predictive_foresight_* attributes.

    Returns:
        PredictiveForesightConfig: Normalized foresight configuration.
    """
    return PredictiveForesightConfig(
        enabled=bool(getattr(source, "predictive_foresight_enabled", False)),
        model_id=str(getattr(source, "predictive_foresight_model_id", "")),
        checkpoint_path=getattr(source, "predictive_foresight_checkpoint_path", None),
        device=str(getattr(source, "predictive_foresight_device", "cpu")),
        max_agents=int(getattr(source, "predictive_foresight_max_agents", default_max_agents)),
        horizon_steps=int(getattr(source, "predictive_foresight_horizon_steps", 8)),
        rollout_dt=float(getattr(source, "predictive_foresight_rollout_dt", 0.2)),
        ego_conditioning=bool(getattr(source, "predictive_foresight_ego_conditioning", False)),
        near_distance=float(getattr(source, "predictive_foresight_near_distance", 0.7)),
        front_corridor_length=float(
            getattr(source, "predictive_foresight_front_corridor_length", 3.0)
        ),
        front_corridor_half_width=float(
            getattr(source, "predictive_foresight_front_corridor_half_width", 1.0)
        ),
    )


def predictive_foresight_spaces() -> spaces.Dict:
    """Return observation spaces for compact predictor-derived features."""
    return spaces.Dict(
        {
            "min_clearance": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([50.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "ttc_risk": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1_000.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "crossing_count": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([64.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "gap_scores": spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([10.0, 10.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "flow_alignment": spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "uncertainty": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        },
    )


class PredictiveForesightEncoder:
    """Lazy encoder that derives compact social-foresight features from a predictive model."""

    def __init__(self, config: PredictiveForesightConfig) -> None:
        """Store config and construct a minimal predictive adapter lazily."""
        self.config = config
        self._adapter = PredictionPlannerAdapter(
            SocNavPlannerConfig(
                predictive_model_id=config.model_id,
                predictive_checkpoint_path=config.checkpoint_path,
                predictive_device=config.device,
                predictive_max_agents=config.max_agents,
                predictive_horizon_steps=config.horizon_steps,
                predictive_rollout_dt=config.rollout_dt,
                predictive_ego_conditioning=config.ego_conditioning,
                predictive_near_distance=config.near_distance,
            ),
            allow_fallback=True,
        )

    def encode(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return compact predictor-derived features for a structured observation."""
        state, mask, robot_pos, robot_heading = self._adapter._build_model_input(observation)
        future = self._adapter._predict_trajectories(state, mask)
        steps = self._adapter._effective_rollout_steps(future_peds=future, mask=mask)
        min_clearance = self._adapter._min_predicted_distance(
            future_peds=future,
            mask=mask,
            steps=steps,
        )
        ttc_risk = self._ttc_risk(future=future, mask=mask, steps=steps)
        crossing_count = self._crossing_count(future=future, mask=mask, steps=steps)
        gap_scores = self._gap_scores(future=future, mask=mask, steps=steps)
        flow_alignment = self._flow_alignment(
            observation=observation,
            state=state,
            mask=mask,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
        )
        if not isfinite(min_clearance):
            min_clearance = 50.0
        return {
            "min_clearance": np.array([float(np.clip(min_clearance, 0.0, 50.0))], dtype=np.float32),
            "ttc_risk": np.array([float(np.clip(ttc_risk, 0.0, 1_000.0))], dtype=np.float32),
            "crossing_count": np.array(
                [float(np.clip(crossing_count, 0.0, float(self.config.max_agents)))],
                dtype=np.float32,
            ),
            "gap_scores": np.asarray(gap_scores, dtype=np.float32),
            "flow_alignment": np.array(
                [float(np.clip(flow_alignment, -1.0, 1.0))], dtype=np.float32
            ),
            "uncertainty": np.array([0.0], dtype=np.float32),
        }

    def _ttc_risk(self, *, future: np.ndarray, mask: np.ndarray, steps: int) -> float:
        """Summarize short-horizon close-approach risk against a stationary robot origin.

        Returns:
            float: Non-negative TTC-style risk summary.
        """
        threshold = float(self.config.near_distance)
        if threshold <= 0.0 or future.size == 0:
            return 0.0
        penalty = 0.0
        dt = max(float(self.config.rollout_dt), 1e-3)
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return 0.0
        for t in range(min(int(steps), future.shape[1])):
            valid = future[valid_idx, t, :]
            if valid.size == 0:
                continue
            dist = np.linalg.norm(valid, axis=1)
            shortfall = np.maximum(0.0, threshold - dist)
            penalty += float(np.sum(shortfall / (float(t + 1) * dt + 1e-6)))
        return penalty

    def _crossing_count(self, *, future: np.ndarray, mask: np.ndarray, steps: int) -> float:
        """Count predicted agents entering the front corridor and crossing the centerline.

        Returns:
            float: Number of unique predicted corridor-crossing agents.
        """
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return 0.0
        length = float(self.config.front_corridor_length)
        width = float(self.config.front_corridor_half_width)
        count = 0
        for idx in valid_idx:
            traj = future[idx, :steps, :]
            ahead = traj[:, 0] > 0.0
            in_corridor = np.abs(traj[:, 1]) <= width
            inside_corridor = ahead & in_corridor & (traj[:, 0] <= length)
            enters = bool(np.any(inside_corridor))
            crosses = bool(
                np.any(
                    (np.signbit(traj[:-1, 1]) != np.signbit(traj[1:, 1]))
                    & inside_corridor[:-1]
                    & inside_corridor[1:]
                )
            )
            if enters and crosses:
                count += 1
        return float(count)

    def _gap_scores(self, *, future: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
        """Estimate left/right gap openness in the front corridor.

        Returns:
            np.ndarray: Two-element `[left_gap, right_gap]` openness estimate.
        """
        valid_idx = np.where(mask > 0.5)[0]
        width = float(self.config.front_corridor_half_width)
        length = float(self.config.front_corridor_length)
        left = width
        right = width
        if valid_idx.size == 0:
            return np.array([left, right], dtype=np.float32)
        for idx in valid_idx:
            traj = future[idx, :steps, :]
            in_front = (traj[:, 0] > 0.0) & (traj[:, 0] <= length)
            if not np.any(in_front):
                continue
            sample = traj[in_front]
            pos_y = sample[:, 1]
            left_samples = pos_y[pos_y >= 0.0]
            right_samples = np.abs(pos_y[pos_y < 0.0])
            if left_samples.size > 0:
                left = min(left, float(np.min(left_samples)))
            if right_samples.size > 0:
                right = min(right, float(np.min(right_samples)))
        return np.array([max(left, 0.0), max(right, 0.0)], dtype=np.float32)

    def _flow_alignment(
        self,
        *,
        observation: dict[str, Any],
        state: np.ndarray,
        mask: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> float:
        """Compute mean pedestrian flow alignment with the robot's goal direction.

        Returns:
            float: Alignment score in `[-1, 1]`.
        """
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return 0.0
        _robot_state, goal_state, _ped_state = self._adapter._socnav_fields(observation)
        goal = self._adapter._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        rel_goal_world = goal - robot_pos
        goal_norm = float(np.linalg.norm(rel_goal_world))
        if goal_norm <= 1e-6:
            return 0.0
        goal_dir_world = rel_goal_world / goal_norm
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        vel_ego = state[valid_idx, 2:4]
        vel_world = np.stack(
            [
                cos_h * vel_ego[:, 0] - sin_h * vel_ego[:, 1],
                sin_h * vel_ego[:, 0] + cos_h * vel_ego[:, 1],
            ],
            axis=1,
        )
        speeds = np.linalg.norm(vel_world, axis=1)
        active = speeds > 1e-6
        if not np.any(active):
            return 0.0
        proj = np.sum(vel_world[active] * goal_dir_world.reshape(1, 2), axis=1) / speeds[active]
        return float(np.mean(np.clip(proj, -1.0, 1.0)))


__all__ = [
    "PredictiveForesightConfig",
    "PredictiveForesightEncoder",
    "predictive_foresight_spaces",
]
