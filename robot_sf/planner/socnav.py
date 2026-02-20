"""
Lightweight adapters for running SocNavBench-style planners against the structured
observation mode.

These adapters are intentionally minimal to provide an in-process bridge while
full SocNavBench planners are integrated. They operate on the SocNav structured
observation emitted when `ObservationMode.SOCNAV_STRUCT` is enabled.
"""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, pi
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:  # pragma: no cover - optional dependency
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import tensorflow.compat.v1 as tf  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    tf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import rvo2  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    rvo2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pysocialforce import forces as sf_forces  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    sf_forces = None  # type: ignore[assignment]

from robot_sf.models import resolve_model_path
from robot_sf.nav.occupancy_grid import OBSERVATION_CHANNEL_ORDER
from robot_sf.nav.occupancy_grid_utils import world_to_ego
from robot_sf.planner.predictive_model import (
    PredictiveTrajectoryModel,
    load_predictive_checkpoint,
)

_SOCNAV_ROOT_ENV = "ROBOT_SF_SOCNAV_ROOT"
_SOCNAV_ALLOW_UNTRUSTED_ENV = "ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT"
_SOCNAV_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "socnavbench"
_SOCNAV_REQUIRED_MODULES = (
    "control_pipelines.control_pipeline_v0",
    "objectives.goal_distance",
    "params.central_params",
    "planners.sampling_planner",
)
_SOCNAV_ASSET_SETUP_DOC = "docs/socnav_assets_setup.md"
_SOCNAV_ASSET_SETUP_CMD = "uv run python scripts/tools/prepare_socnav_assets.py"
_SACADRL_MODEL_ID = "ga3c_cadrl_iros18"
_PREDICTIVE_MODEL_ID = "predictive_proxy_selected_v1"
_SACADRL_STATE_ORDER = (
    "num_other_agents",
    "dist_to_goal",
    "heading_ego_frame",
    "pref_speed",
    "radius",
    "other_agents_states",
)


class OccupancyAwarePlannerMixin:
    """Shared helpers for planners that can leverage occupancy grid observations."""

    _CHANNEL_KEYS = tuple(channel.value for channel in OBSERVATION_CHANNEL_ORDER)

    @staticmethod
    def _as_1d_float(values: Any, *, pad: int | None = None, default: float = 0.0) -> np.ndarray:
        """Normalize metadata values to at least 1D float array with optional padding.

        Returns:
            np.ndarray: At-least-1D float array, padded to length ``pad`` when provided.
        """
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if pad is not None and arr.size < pad:
            arr = np.pad(arr, (0, pad - arr.size), constant_values=default)
        return arr

    def _extract_grid_payload(self, observation: dict) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Extract occupancy grid tensor and metadata from observation.

        Returns:
            tuple[np.ndarray, dict[str, Any]] | None: Grid array and metadata dict
            when present, otherwise ``None``.
        """
        grid = observation.get("occupancy_grid")
        if grid is None:
            return None

        # Reconstruct metadata from flattened fields (SB3 compatibility format)
        meta = {}
        for key in (
            "origin",
            "resolution",
            "size",
            "use_ego_frame",
            "center_on_robot",
            "channel_indices",
            "robot_pose",
        ):
            flat_key = f"occupancy_grid_meta_{key}"
            if flat_key in observation:
                meta[key] = observation[flat_key]

        # If no metadata fields found, try old format (backward compatibility)
        if not meta:
            meta = observation.get("occupancy_grid_meta")

        if meta is None or not meta:
            return None

        try:
            grid_arr = np.asarray(grid)
        except (TypeError, ValueError):
            return None
        if grid_arr.ndim < 3:
            return None
        return grid_arr, meta

    def _socnav_fields(self, observation: dict) -> tuple[dict, dict, dict]:
        """Normalize SocNav observation (nested or flattened) into standard dicts.

        Returns:
            tuple[dict, dict, dict]: (robot_state, goal_state, ped_state) dictionaries.
        """
        if "robot" in observation:
            robot_state = observation["robot"]
            goal_state = observation.get("goal", {})
            ped_state = observation.get("pedestrians", {})
        else:
            pos_arr = self._as_1d_float(observation.get("robot_position", [0.0, 0.0]), pad=2)
            robot_state = {
                "position": pos_arr,
                "heading": self._as_1d_float(observation.get("robot_heading", [0.0]), pad=1),
                "speed": self._as_1d_float(observation.get("robot_speed", [0.0]), pad=1),
                "radius": self._as_1d_float(observation.get("robot_radius", [0.0]), pad=1),
            }
            goal_state = {
                "current": self._as_1d_float(observation.get("goal_current", [0.0, 0.0]), pad=2),
                "next": self._as_1d_float(observation.get("goal_next", [0.0, 0.0]), pad=2),
            }
            ped_state = {
                "positions": observation.get("pedestrians_positions"),
                "velocities": observation.get("pedestrians_velocities"),
                "count": self._as_1d_float(observation.get("pedestrians_count", [0]), pad=1),
                "radius": self._as_1d_float(observation.get("pedestrians_radius", [0.0]), pad=1)[0],
            }
        return robot_state, goal_state, ped_state

    def _grid_channel_index(self, meta: dict[str, Any], key: str) -> int:
        """Return channel index for a semantic key, or -1 when unavailable.

        Returns:
            int: Channel index or -1 when the channel is missing.
        """
        indices = meta.get("channel_indices")
        if indices is None:
            return -1
        try:
            pos = self._CHANNEL_KEYS.index(key)
            idx_arr = self._as_1d_float(indices)
            if pos >= idx_arr.size:
                return -1
            return int(idx_arr[pos])
        except (ValueError, TypeError, IndexError):
            return -1

    def _preferred_channel(self, meta: dict[str, Any]) -> int:
        """Prefer combined channel, else obstacles, else pedestrians.

        Returns:
            int: Channel index or -1 when no occupancy channel is present.
        """
        combined_idx = self._grid_channel_index(meta, "combined")
        if combined_idx >= 0:
            return combined_idx
        obstacle_idx = self._grid_channel_index(meta, "obstacles")
        if obstacle_idx >= 0:
            return obstacle_idx
        return self._grid_channel_index(meta, "pedestrians")

    def _world_to_grid(
        self,
        point: np.ndarray,
        meta: dict[str, Any],
        grid_shape: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Convert world coordinates to grid row/col using metadata.

        Returns:
            tuple[int, int] | None: Grid indices or None when point is out of bounds/invalid.
        """
        origin = np.asarray(meta.get("origin", [0.0, 0.0]), dtype=float)
        size = np.asarray(meta.get("size", [0.0, 0.0]), dtype=float)
        resolution_arr = self._as_1d_float(meta.get("resolution", [0.0]))
        if resolution_arr.size == 0 or resolution_arr[0] <= 0:
            return None
        resolution = float(resolution_arr[0])

        origin = self._as_1d_float(origin, pad=2)
        size = self._as_1d_float(size, pad=2)

        use_ego_arr = self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)
        use_ego = bool(use_ego_arr[0] > 0.5)
        pose_arr = self._as_1d_float(meta.get("robot_pose", [0.0, 0.0, 0.0]), pad=3)
        if use_ego:
            pose_tuple = ((float(pose_arr[0]), float(pose_arr[1])), float(pose_arr[2]))
            point = np.asarray(
                world_to_ego(float(point[0]), float(point[1]), pose_tuple), dtype=float
            )

        local = point - origin
        if np.any(local < 0.0) or local[0] > size[0] or local[1] > size[1]:
            return None

        col = int(local[0] / resolution)
        row = int(local[1] / resolution)
        row = min(max(row, 0), grid_shape[0] - 1)
        col = min(max(col, 0), grid_shape[1] - 1)
        return row, col

    def _grid_value(
        self,
        point: np.ndarray,
        grid: np.ndarray,
        meta: dict[str, Any],
        channel_idx: int,
    ) -> float:
        """Return occupancy value at a world point (treat OOB as occupied).

        Returns:
            float: Occupancy value in [0, 1], 1.0 when out of bounds.
        """
        if channel_idx < 0:
            return 0.0
        if grid.ndim < 3:
            return 1.0
        channels, height, width = grid.shape[0], grid.shape[1], grid.shape[2]
        grid_shape = (height, width)
        indices = self._world_to_grid(point, meta, grid_shape)
        if indices is None:
            return 1.0
        row, col = indices
        if channel_idx >= channels:
            return 1.0
        return float(grid[channel_idx, row, col])

    def _path_penalty(
        self,
        robot_pos: np.ndarray,
        direction: np.ndarray,
        observation: dict,
        base_distance: float,
        num_samples: int,
    ) -> tuple[float, float]:
        """Compute occupancy penalty along a candidate heading.

        Returns:
            tuple[float, float]: Mean obstacle and pedestrian occupancy along the sample line.
        """
        grid_payload = self._extract_grid_payload(observation)
        if grid_payload is None or np.linalg.norm(direction) < 1e-6:
            return 0.0, 0.0

        grid, meta = grid_payload
        if grid.ndim < 3:
            return 0.0, 0.0
        channel_idx = self._preferred_channel(meta)
        ped_idx = self._grid_channel_index(meta, "pedestrians")
        direction = direction / (np.linalg.norm(direction) + 1e-9)

        samples = np.linspace(base_distance / num_samples, base_distance, num_samples)
        obstacle_vals: list[float] = []
        ped_vals: list[float] = []
        for dist in samples:
            point = robot_pos + direction * dist
            obstacle_vals.append(self._grid_value(point, grid, meta, channel_idx))
            if ped_idx >= 0:
                ped_vals.append(self._grid_value(point, grid, meta, ped_idx))

        obstacle_penalty = float(np.mean(obstacle_vals)) if obstacle_vals else 0.0
        ped_penalty = float(np.mean(ped_vals)) if ped_vals else 0.0
        return obstacle_penalty, ped_penalty

    def _select_safe_heading(
        self,
        robot_pos: np.ndarray,
        base_direction: np.ndarray,
        observation: dict,
        sweep: float,
        num_candidates: int,
        lookahead: float,
        weight: float,
        angle_weight: float,
    ) -> tuple[np.ndarray, float]:
        """Pick heading that balances goal alignment and occupancy clearance.

        Returns:
            tuple[np.ndarray, float]: Chosen direction vector and the associated occupancy penalty.
        """
        if np.linalg.norm(base_direction) < 1e-6 or num_candidates <= 1:
            return base_direction, 0.0

        base_dir = base_direction / (np.linalg.norm(base_direction) + 1e-9)
        angles = np.linspace(-sweep / 2, sweep / 2, num_candidates)
        best_dir = base_dir
        best_cost = float("inf")
        best_penalty = 0.0

        for angle in angles:
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=float,
            )
            candidate = rot @ base_dir
            obstacle_penalty, ped_penalty = self._path_penalty(
                robot_pos, candidate, observation, lookahead, max(2, num_candidates)
            )
            penalty = obstacle_penalty + 0.5 * ped_penalty
            angle_cost = abs(angle) / (sweep / 2 if sweep > 0 else 1.0)
            cost = weight * penalty + angle_weight * angle_cost
            if cost < best_cost:
                best_cost = cost
                best_dir = candidate
                best_penalty = penalty

        return best_dir, best_penalty

    def _get_safe_heading(
        self, robot_pos: np.ndarray, base_direction: np.ndarray, observation: dict
    ) -> tuple[np.ndarray, float]:
        """Helper to call _select_safe_heading with config parameters.

        Returns:
            tuple[np.ndarray, float]: Direction vector and occupancy penalty.
        """
        # Type checker can't infer config attribute from mixin class
        # This is resolved at runtime by concrete implementations
        return self._select_safe_heading(  # type: ignore[attr-defined]
            robot_pos,
            base_direction,
            observation,
            sweep=self.config.occupancy_heading_sweep,  # type: ignore[attr-defined]
            num_candidates=self.config.occupancy_candidates,  # type: ignore[attr-defined]
            lookahead=self.config.occupancy_lookahead,  # type: ignore[attr-defined]
            weight=self.config.occupancy_weight,  # type: ignore[attr-defined]
            angle_weight=self.config.occupancy_angle_weight,  # type: ignore[attr-defined]
        )


@dataclass
class SocNavPlannerConfig:
    """Simple config for SocNav-like planner adapters."""

    max_linear_speed: float = 3.0
    max_angular_speed: float = 1.0
    angular_gain: float = 1.0
    goal_tolerance: float = 0.25
    sacadrl_neighbors: int = 3
    sacadrl_bias_weight: float = 0.6
    orca_avoidance_weight: float = 1.2
    orca_neighbor_dist: float = 10.0
    orca_max_neighbors: int = 10
    orca_time_horizon: float = 6.0
    orca_time_horizon_obst: float = 3.0
    orca_obstacle_threshold: float = 0.5
    orca_obstacle_range: float = 6.0
    orca_obstacle_max_points: int = 80
    orca_obstacle_radius_scale: float = 1.0
    orca_heading_slowdown: float = 0.2
    social_force_repulsion_weight: float = 0.8
    social_force_desired_speed: float = 1.0
    social_force_tau: float = 0.5
    social_force_factor: float = 5.1
    social_force_lambda_importance: float = 2.0
    social_force_gamma: float = 0.35
    social_force_n: int = 2
    social_force_n_prime: int = 3
    social_force_obstacle_factor: float = 10.0
    social_force_obstacle_threshold: float = 0.5
    social_force_obstacle_range: float = 6.0
    social_force_obstacle_max_points: int = 80
    social_force_obstacle_radius_scale: float = 1.0
    social_force_clip_force: bool = True
    social_force_max_force: float = 100.0
    occupancy_lookahead: float = 2.5
    occupancy_heading_sweep: float = pi * 2 / 3
    occupancy_candidates: int = 7
    occupancy_weight: float = 2.0
    occupancy_angle_weight: float = 0.3
    sacadrl_model_id: str = _SACADRL_MODEL_ID
    sacadrl_checkpoint_path: str | None = None
    sacadrl_pref_speed: float = 1.0
    sacadrl_max_other_agents: int = 3
    sacadrl_sorting_method: str = "closest_first"
    predictive_model_id: str = _PREDICTIVE_MODEL_ID
    predictive_checkpoint_path: str | None = None
    predictive_device: str = "cpu"
    predictive_max_agents: int = 16
    predictive_horizon_steps: int = 8
    predictive_rollout_dt: float = 0.2
    predictive_goal_weight: float = 1.0
    predictive_collision_weight: float = 6.0
    predictive_near_miss_weight: float = 1.5
    predictive_velocity_weight: float = 0.05
    predictive_turn_weight: float = 0.15
    predictive_ttc_weight: float = 0.0
    predictive_ttc_distance: float = 0.8
    predictive_safe_distance: float = 0.6
    predictive_near_distance: float = 1.0
    predictive_robot_radius: float = 0.3
    predictive_pedestrian_radius: float = 0.3
    predictive_speed_clearance_gain: float = 0.0
    predictive_progress_risk_weight: float = 1.0
    predictive_progress_risk_distance: float = 1.2
    predictive_hard_clearance_distance: float = 0.75
    predictive_hard_clearance_weight: float = 2.5
    predictive_adaptive_horizon_enabled: bool = True
    predictive_horizon_boost_steps: int = 4
    predictive_near_field_distance: float = 2.4
    predictive_near_field_speed_cap: float = 0.75
    predictive_near_field_speed_samples: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5)
    predictive_near_field_heading_deltas: tuple[float, ...] = (
        -pi / 2,
        -pi / 3,
        -pi / 4,
        -pi / 6,
        0.0,
        pi / 6,
        pi / 4,
        pi / 3,
        pi / 2,
    )
    predictive_candidate_speeds: tuple[float, ...] = (0.0, 0.5, 1.0)
    predictive_candidate_heading_deltas: tuple[float, ...] = (
        -pi / 4,
        -pi / 8,
        0.0,
        pi / 8,
        pi / 4,
    )


class SamplingPlannerAdapter(OccupancyAwarePlannerMixin):
    """
    Minimal waypoint-to-velocity adapter inspired by the SocNavBench sampling planner.

    Warning:
        By default this adapter uses a lightweight heuristic placeholder. Set
        ``use_upstream=True`` to delegate to the upstream SocNavBench sampling planner
        (benchmark-ready), and optionally allow fallback when dependencies are missing.
    """

    class _GoalDistanceObjective:
        """Minimal goal-distance objective for the upstream sampling planner."""

        def __init__(self, goal_pos: np.ndarray | None = None) -> None:
            self._goal_pos = (
                np.zeros(2, dtype=float) if goal_pos is None else np.asarray(goal_pos, dtype=float)
            )

        def set_goal(self, goal_pos: np.ndarray) -> None:
            """Update the target goal position used for distance costs."""
            self._goal_pos = np.asarray(goal_pos, dtype=float)

        def evaluate_function(
            self, trajectory: Any, sim_state_hist: Any | None = None
        ) -> np.ndarray:
            """Return per-trajectory goal distance costs (lower is better)."""
            positions = trajectory.position_nk2()
            if positions.size == 0:
                return np.array([])
            valid_horizons = getattr(trajectory, "valid_horizons_n1", None)
            if valid_horizons is None:
                final_pos = positions[:, -1, :]
            else:
                idx = np.asarray(valid_horizons, dtype=int).reshape(-1) - 1
                idx = np.clip(idx, 0, positions.shape[1] - 1)
                final_pos = positions[np.arange(positions.shape[0]), idx, :]
            goal = self._goal_pos.reshape(1, 2)
            return np.linalg.norm(final_pos - goal, axis=1)

    def __init__(
        self,
        config: SocNavPlannerConfig | None = None,
        socnav_root: Path | None = None,
        planner_factory: Callable[[], Any] | None = None,
        *,
        use_upstream: bool = False,
        allow_fallback: bool = True,
    ):
        """Initialize the adapter with optional planner configuration."""

        self.config = config or SocNavPlannerConfig()
        self._planner = None
        self._goal_objective: SamplingPlannerAdapter._GoalDistanceObjective | None = None
        self._use_upstream = bool(use_upstream)
        self._allow_fallback = bool(allow_fallback)

        if self._use_upstream:
            if planner_factory is not None:
                self._planner = self._safe_call_factory(planner_factory)
            else:
                self._planner = self._load_upstream_planner(socnav_root)
            if self._planner is None and self._allow_fallback:
                logger.warning(
                    "SamplingPlannerAdapter is running in fallback heuristic mode and "
                    "is not benchmark-ready."
                )
            if self._planner is None and not self._allow_fallback:
                raise RuntimeError(
                    "SamplingPlannerAdapter could not load the upstream planner. "
                    "Set allow_fallback=True to use the heuristic fallback."
                )
        else:
            logger.warning(
                "SamplingPlannerAdapter is a heuristic fallback and is not benchmark-ready."
            )

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute a (v, w) command from the structured observation.

        Args:
            observation: SocNav structured observation Dict (robot, goal, pedestrians, map, sim).

        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        if self._planner is not None:
            return self._plan_upstream(observation)
        return self._heuristic_plan(observation)

    def _heuristic_plan(self, observation: dict) -> tuple[float, float]:
        """Compute a heuristic (v, w) command from the structured observation.

        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state["position"], pad=2)[:2]
        robot_heading = float(self._as_1d_float(robot_state["heading"], pad=1)[0])
        goal = self._as_1d_float(goal_state["current"], pad=2)[:2]

        to_goal = goal - robot_pos
        distance = float(np.linalg.norm(to_goal))
        if distance < self.config.goal_tolerance:
            return 0.0, 0.0

        # Light pedestrian repulsion to keep base planner pedestrian-aware
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        ped_count = (
            int(self._as_1d_float(ped_state.get("count", [0]), pad=1)[0]) if ped_state else 0
        )
        ped_positions = ped_positions[:ped_count]
        repulse = np.zeros(2, dtype=float)
        for ped in ped_positions:
            delta = robot_pos - ped
            dist = np.linalg.norm(delta) + 1e-6
            repulse += delta / dist**2

        base_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)
        if np.linalg.norm(repulse) > 1e-6:
            base_vec = base_vec + self.config.social_force_repulsion_weight * repulse
            if np.linalg.norm(base_vec) > 1e-6:
                base_vec = base_vec / np.linalg.norm(base_vec)

        # Adjust heading to favor obstacle-free paths when grid is available
        adjusted_vec, occ_penalty = self._get_safe_heading(robot_pos, base_vec, observation)

        desired_heading = atan2(adjusted_vec[1], adjusted_vec[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)

        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )

        # Slow down when sharply turning or when path shows occupancy
        linear_scale = max(0.0, 1.0 - abs(heading_error) / pi)
        linear_scale *= max(0.0, 1.0 - occ_penalty)
        linear = float(
            np.clip(distance * linear_scale, 0.0, self.config.max_linear_speed),
        )
        return linear, angular

    def _plan_upstream(self, observation: dict) -> tuple[float, float]:
        """Compute a (v, w) command using the upstream SocNavBench planner.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        if self._planner is None:
            return self._heuristic_plan(observation)

        try:
            robot_state, goal_state, _ = self._socnav_fields(observation)
            pos = robot_state["position"]
            robot_pos = np.asarray(pos, dtype=float)
            heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
            if self._goal_objective is not None:
                self._goal_objective.set_goal(goal_state["current"])
            start_config = self._planner.opt_waypt.__class__.from_pos3([pos[0], pos[1], heading])
            goal = goal_state["current"]
            goal_config = self._planner.opt_waypt.__class__.from_pos3([goal[0], goal[1], 0.0])
            data = self._planner.optimize(start_config=start_config, goal_config=goal_config)
            traj = data.get("trajectory")
            if traj is None:
                return self._heuristic_plan(observation)
            # NOTE: upstream returns a trajectory and controller matrices; for now we
            # consume only the immediate waypoint to preserve the (v, w) interface and
            # avoid binding to controller specifics. This keeps the adapter lightweight
            # while still aligning heading toward the planned path.
            next_pos = traj.position_nk2()[0, 0]
            to_next = next_pos - pos
            direction = to_next / (np.linalg.norm(to_next) + 1e-9)
            direction, occ_penalty = self._get_safe_heading(robot_pos, direction, observation)
            desired_heading = atan2(direction[1], direction[0])
            heading_error = self._wrap_angle(desired_heading - heading)
            angular = float(
                np.clip(
                    self.config.angular_gain * heading_error,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed,
                ),
            )
            linear = float(
                np.clip(
                    np.linalg.norm(to_next),
                    0.0,
                    self.config.max_linear_speed * max(0.0, 1.0 - occ_penalty),
                ),
            )
            return linear, angular
        except Exception as exc:  # pragma: no cover - safety net
            if self._allow_fallback:
                return self._heuristic_plan(observation)
            raise RuntimeError("SocNavBench planner failed during _plan_upstream.") from exc

    def _safe_call_factory(self, factory: Callable[[], Any]) -> Any | None:
        """Invoke a user-provided factory defensively.

        Returns:
            Planner instance from the factory or ``None`` on failure.
        """
        try:
            return factory()
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover
            return self._handle_socnav_failure(
                f"SocNavBench planner factory failed: {exc}", exc=exc
            )

    def _handle_socnav_failure(
        self, message: str, *, exc: Exception | None = None, not_found: bool = False
    ) -> Any | None:
        """Handle SocNavBench initialization failures with optional fallback.

        Returns:
            None when fallback is allowed; otherwise raises a descriptive error.
        """
        if self._allow_fallback:
            logger.warning("{}", message)
            return None
        if not_found:
            raise FileNotFoundError(message) from exc
        raise RuntimeError(message) from exc

    @staticmethod
    def _resolve_socnav_root(socnav_root: Path | None) -> Path:
        """Resolve the SocNavBench root directory.

        Returns:
            Path: Resolved SocNavBench root path.
        """
        if socnav_root is not None:
            return Path(socnav_root).expanduser()
        env_root = os.getenv(_SOCNAV_ROOT_ENV)
        if env_root:
            return Path(env_root).expanduser()
        return _SOCNAV_DEFAULT_ROOT

    @staticmethod
    def _allow_untrusted_socnav_root() -> bool:
        """Determine whether the environment explicitly allows untrusted roots.

        Returns:
            bool: True when the environment variable enables untrusted roots.
        """
        value = os.getenv(_SOCNAV_ALLOW_UNTRUSTED_ENV, "")
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _is_trusted_socnav_root(root: Path) -> bool:
        """Check whether the SocNavBench root lives inside the repository.

        Returns:
            bool: True when the root resolves under the repository directory.
        """
        repo_root = Path(__file__).resolve().parents[2]
        try:
            root.resolve().relative_to(repo_root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _validate_socnav_root(root: Path) -> list[Path]:
        """Validate that the SocNavBench root contains expected modules.

        Returns:
            list[Path]: Missing module paths.
        """
        missing: list[Path] = []
        for module in _SOCNAV_REQUIRED_MODULES:
            rel_path = Path(*module.split(".")).with_suffix(".py")
            if not (root / rel_path).exists():
                missing.append(root / rel_path)
        return missing

    @staticmethod
    def _import_socnav_modules(root: Path) -> tuple[tuple[Any, Any, Any] | None, str | None]:
        """Import upstream SocNavBench modules from the provided root.

        Returns:
            tuple[tuple[Any, Any, Any] | None, str | None]:
            ``((central_params, sampling_planner, DotMap), None)`` on success;
            otherwise ``(None, error_message)`` on failure.
        """
        root_str = str(root)
        sys_path_inserted = False
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            sys_path_inserted = True
        try:
            import params.central_params as central  # type: ignore  # noqa: PLC0415
            import planners.sampling_planner as sp  # type: ignore  # noqa: PLC0415
            from dotmap import DotMap  # type: ignore  # noqa: PLC0415

            return (central, sp, DotMap), None
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover
            hint = ""
            if isinstance(exc, ModuleNotFoundError) and "skfmm" in str(exc):
                hint = (
                    " Missing dependency `skfmm` detected. "
                    "Install SocNav prerequisites (for example `uv sync --extra socnav`)."
                )
            return None, f"{type(exc).__name__}: {exc}.{hint}"
        finally:
            if sys_path_inserted:
                try:
                    sys.path.remove(root_str)
                except ValueError:
                    pass

    def _resolve_robot_dt(self, socnav_params: Any) -> float:
        """Resolve the robot dynamics timestep from SocNavBench params.

        Returns:
            float: Robot timestep.
        """
        dyn_params = getattr(socnav_params, "robot_dynamics_params", None)
        if dyn_params is None:
            return 0.1
        return float(getattr(dyn_params, "dt", 0.1))

    def _resolve_camera_dt(self, socnav_params: Any) -> float:
        """Resolve the camera timestep from SocNavBench params.

        Returns:
            float: Camera timestep.
        """
        camera_params = getattr(socnav_params, "camera_params", None)
        if camera_params is None:
            return 0.1
        return float(getattr(camera_params, "dt", 0.1))

    def _build_sampling_params(self, central: Any, sp: Any, DotMap: Any) -> Any | None:
        """Build sampling planner parameters for the upstream planner.

        Returns:
            Params object or ``None`` on failure.
        """
        params = DotMap()
        params.planner = sp.SamplingPlanner
        try:
            params.control_pipeline_params = central.create_control_pipeline_params()
        except SystemExit as exc:
            return self._handle_socnav_failure(
                "SocNavBench control pipeline parameters failed to load. "
                "Ensure the SocNavBench data directories exist. "
                f"See `{_SOCNAV_ASSET_SETUP_DOC}` and run `{_SOCNAV_ASSET_SETUP_CMD}`.",
                exc=exc,
            )
        return params

    def _load_upstream_planner(self, socnav_root: Path | None) -> Any | None:
        """Best-effort import of SocNavBench SamplingPlanner with defaults.

        Returns:
            Planner instance or ``None`` on failure.
        """
        env_root = os.getenv(_SOCNAV_ROOT_ENV)
        root_source = "argument" if socnav_root is not None else ("env" if env_root else "default")
        root = self._resolve_socnav_root(socnav_root).resolve()
        if not root.exists():
            message = (
                "SocNavBench root not found at "
                f"'{root}'. Set {_SOCNAV_ROOT_ENV} or pass socnav_root."
            )
            return self._handle_socnav_failure(message, not_found=True)

        if root_source != "default" and not self._is_trusted_socnav_root(root):
            if not self._allow_untrusted_socnav_root():
                message = (
                    "SocNavBench root is outside the repository root. "
                    f"Refusing to load from '{root}'. Set {_SOCNAV_ALLOW_UNTRUSTED_ENV}=1 "
                    "to explicitly allow untrusted SocNavBench roots."
                )
                return self._handle_socnav_failure(message)
            logger.warning(
                "Using SocNavBench root outside the repository: '{}'. Ensure this path is trusted.",
                root,
            )

        missing = self._validate_socnav_root(root)
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            message = (
                "SocNavBench root is missing required modules: "
                f"{missing_str}. Ensure the SocNavBench repo is complete."
            )
            return self._handle_socnav_failure(message, not_found=True)

        prev_cwd = Path.cwd()
        try:
            # Upstream SocNavBench params resolve INI paths relative to cwd at import time.
            os.chdir(root)
            modules, import_error = self._import_socnav_modules(root)
            if modules is None:
                message = "Failed to import SocNavBench modules."
                if import_error:
                    message = f"{message} {import_error}"
                return self._handle_socnav_failure(message)
            central, sp, DotMap = modules
            params = self._build_sampling_params(central, sp, DotMap)
            if params is None:
                return None
            try:
                # Upstream SocNavBench uses a class-level singleton control pipeline.
                # In repeated benchmark episodes, params can diverge/mutate across runs,
                # which triggers an internal equality assertion on re-init.
                # Resetting the cache keeps per-run planner construction deterministic.
                try:
                    import control_pipelines.control_pipeline_v0 as cp_v0  # type: ignore  # noqa: PLC0415

                    cp_v0.ControlPipelineV0.pipeline = None
                except (ImportError, AttributeError):
                    pass
                obj_fn = self._GoalDistanceObjective()
                self._goal_objective = obj_fn
                return sp.SamplingPlanner(obj_fn=obj_fn, params=params)
            except (
                AssertionError,
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:  # pragma: no cover
                return self._handle_socnav_failure(
                    "Failed to initialize SocNavBench SamplingPlanner: "
                    f"{exc}. If this is an asset/data issue, see `{_SOCNAV_ASSET_SETUP_DOC}` "
                    f"and run `{_SOCNAV_ASSET_SETUP_CMD}`.",
                    exc=exc,
                )
        finally:
            os.chdir(prev_cwd)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap angle to [-pi, pi].

        Returns:
            float: Wrapped angle in radians.
        """
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle


class SocNavPlannerPolicy:
    """Thin policy wrapper to plug planner adapters into Gym loops."""

    def __init__(self, adapter: SamplingPlannerAdapter | None = None):
        """Initialize the policy with a planner adapter."""

        self.adapter = adapter or SamplingPlannerAdapter()

    def act(self, observation: dict) -> tuple[float, float]:
        """Return (v, w) action for a SocNav structured observation."""
        return self.adapter.plan(observation)


class SocNavBenchComplexPolicy(SocNavPlannerPolicy):
    """
    Policy that prefers the upstream SocNavBench SamplingPlanner when available.

    By default this policy requires the upstream SocNavBench planner. Set
    ``allow_fallback=True`` to use the lightweight adapter when dependencies are missing.
    """

    def __init__(
        self,
        socnav_root: Path | None = None,
        adapter_config: SocNavPlannerConfig | None = None,
        *,
        allow_fallback: bool = False,
    ):
        """Initialize the policy, preferring the upstream SocNavBench planner when present."""

        adapter = SocNavBenchSamplingAdapter(
            config=adapter_config,
            socnav_root=socnav_root,
            allow_fallback=allow_fallback,
        )
        super().__init__(adapter=adapter)


class SocialForcePlannerAdapter(SamplingPlannerAdapter):
    """Social-force planner adapter using fast-pysf interaction forces."""

    _EPS = 1e-6

    def __init__(self, config: SocNavPlannerConfig | None = None) -> None:
        """Initialize the social-force adapter with optional configuration."""
        self.config = config or SocNavPlannerConfig()
        if sf_forces is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pysocialforce is required for SocialForcePlannerAdapter. "
                "Install the fast-pysf dependency."
            )

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) using social-force goal + interaction forces.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        robot_speed = self._as_1d_float(robot_state.get("speed", [0.0, 0.0]), pad=2)
        linear_speed = float(robot_speed[0])
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        robot_vel = np.array([linear_speed * cos_h, linear_speed * sin_h], dtype=float)

        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        to_goal = goal - robot_pos
        goal_dist = float(np.linalg.norm(to_goal))
        if goal_dist < self.config.goal_tolerance:
            return 0.0, 0.0

        dt = self._resolve_dt(observation)
        desired_speed = min(self.config.social_force_desired_speed, self.config.max_linear_speed)
        desired_speed = min(desired_speed, goal_dist / max(dt, self._EPS))
        goal_dir = to_goal / (goal_dist + self._EPS)
        desired_vel = goal_dir * desired_speed
        desired_force = (desired_vel - robot_vel) / max(self.config.social_force_tau, self._EPS)

        social_force = self._compute_social_force(robot_pos, robot_vel, ped_state, robot_heading)
        obstacle_force = self._compute_obstacle_force(
            observation, robot_pos, robot_heading, robot_vel, robot_state
        )
        interaction_force = self.config.social_force_repulsion_weight * (
            social_force + obstacle_force
        )

        total_force = desired_force + interaction_force
        total_force = self._clip_force(total_force)

        desired_vel = robot_vel + total_force * dt
        speed = float(np.linalg.norm(desired_vel))
        if speed < self._EPS:
            return 0.0, 0.0
        if speed > self.config.max_linear_speed:
            desired_vel = desired_vel / (speed + self._EPS) * self.config.max_linear_speed
            speed = self.config.max_linear_speed

        desired_heading = atan2(desired_vel[1], desired_vel[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                speed * max(0.0, 1.0 - abs(heading_error) / pi),
                0.0,
                self.config.max_linear_speed,
            ),
        )
        return linear, angular

    def _resolve_dt(self, observation: dict) -> float:
        """Return the simulation timestep (fallback to config defaults)."""
        sim = observation.get("sim", {})
        timestep = self._as_1d_float(sim.get("timestep", [0.0]), pad=1)[0]
        if timestep <= 0.0:
            return float(self.config.social_force_tau)
        return float(timestep)

    @staticmethod
    def _rotate_velocities_to_world(velocities: np.ndarray, heading: float) -> np.ndarray:
        """Rotate ego-frame velocities into world coordinates.

        Returns:
            np.ndarray: Rotated velocity vectors in world coordinates.
        """
        if velocities.size == 0:
            return velocities
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        vx = cos_h * velocities[:, 0] - sin_h * velocities[:, 1]
        vy = sin_h * velocities[:, 0] + cos_h * velocities[:, 1]
        return np.stack([vx, vy], axis=1)

    def _compute_social_force(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        ped_state: dict,
        robot_heading: float,
    ) -> np.ndarray:
        """Compute social-force repulsion from pedestrians.

        Returns:
            np.ndarray: Combined social-force vector.
        """
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        if ped_positions.ndim == 1:
            ped_positions = ped_positions.reshape(-1, 2)
        ped_count = int(self._as_1d_float(ped_state.get("count", [0]), pad=1)[0])
        ped_positions = ped_positions[:ped_count]
        if ped_positions.size == 0:
            return np.zeros(2, dtype=float)

        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_velocities.size == 0:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        elif ped_velocities.ndim == 1:
            ped_velocities = ped_velocities.reshape(-1, 2)
        ped_velocities = ped_velocities[:ped_count]
        ped_vel_world = self._rotate_velocities_to_world(ped_velocities, robot_heading)

        total = np.zeros(2, dtype=float)
        for ped_pos, ped_vel in zip(ped_positions, ped_vel_world, strict=False):
            pos_diff = (robot_pos - ped_pos).astype(float)
            vel_diff = (robot_vel - ped_vel).astype(float)
            try:
                fx, fy = sf_forces.social_force_ped_ped(
                    pos_diff,
                    vel_diff,
                    int(self.config.social_force_n),
                    int(self.config.social_force_n_prime),
                    float(self.config.social_force_lambda_importance),
                    float(self.config.social_force_gamma),
                )
                total += np.array([fx, fy], dtype=float)
            except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
                continue
        return total * float(self.config.social_force_factor)

    def _compute_obstacle_force(
        self,
        observation: dict,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_vel: np.ndarray,
        robot_state: dict,
    ) -> np.ndarray:
        """Compute obstacle repulsion using occupancy-grid obstacle points.

        Returns:
            np.ndarray: Combined obstacle repulsion vector.
        """
        centers, radii = self._extract_obstacles_from_grid(observation, robot_pos, robot_heading)
        if centers.size == 0:
            return np.zeros(2, dtype=float)

        if np.linalg.norm(robot_vel) > self._EPS:
            ortho = np.array([-robot_vel[1], robot_vel[0]], dtype=float)
        else:
            ortho = np.array([-np.sin(robot_heading), np.cos(robot_heading)], dtype=float)

        robot_radius = float(self._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0])
        total = np.zeros(2, dtype=float)
        for center, radius in zip(centers, radii, strict=False):
            line = (float(center[0]), float(center[1]), float(center[0]), float(center[1]))
            try:
                fx, fy = sf_forces.obstacle_force(line, ortho, robot_pos, robot_radius + radius)
                total += np.array([fx, fy], dtype=float)
            except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
                continue
        return total * float(self.config.social_force_obstacle_factor)

    @staticmethod
    def _grid_cell_centers(
        indices: np.ndarray, origin: np.ndarray, resolution: float
    ) -> np.ndarray:
        """Convert grid indices to grid-frame centers.

        Returns:
            np.ndarray: Grid-frame centers for the provided indices.
        """
        rows = indices[:, 0].astype(float)
        cols = indices[:, 1].astype(float)
        x = origin[0] + (cols + 0.5) * resolution
        y = origin[1] + (rows + 0.5) * resolution
        return np.stack([x, y], axis=1)

    @staticmethod
    def _ego_centers_to_world(
        centers: np.ndarray, robot_pos: np.ndarray, robot_heading: float
    ) -> np.ndarray:
        """Rotate/translate ego-frame centers into world coordinates.

        Returns:
            np.ndarray: World-space centers.
        """
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        x_world = cos_h * centers[:, 0] - sin_h * centers[:, 1]
        y_world = sin_h * centers[:, 0] + cos_h * centers[:, 1]
        return np.stack([x_world, y_world], axis=1) + np.asarray(robot_pos, dtype=float)

    @staticmethod
    def _select_nearby_points(
        centers: np.ndarray,
        robot_pos: np.ndarray,
        max_range: float,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter centers by range and cap to the closest points.

        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered centers and squared distances.
        """
        offsets = centers - np.asarray(robot_pos, dtype=float)
        dist_sq = np.einsum("ij,ij->i", offsets, offsets)
        keep = dist_sq <= max_range**2
        if not np.any(keep):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
        centers = centers[keep]
        dist_sq = dist_sq[keep]
        if max_points > 0 and centers.shape[0] > max_points:
            order = np.argsort(dist_sq)[:max_points]
            centers = centers[order]
            dist_sq = dist_sq[order]
        return centers, dist_sq

    def _extract_obstacles_from_grid(
        self, observation: dict, robot_pos: np.ndarray, robot_heading: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nearby obstacle centers from the occupancy grid.

        Returns:
            tuple[np.ndarray, np.ndarray]: World-space obstacle centers and per-point radii.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        grid, meta = payload
        if grid.ndim < 3:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        channel_idx = self._grid_channel_index(meta, "obstacles")
        if channel_idx < 0:
            channel_idx = self._grid_channel_index(meta, "combined")
        if channel_idx < 0 or channel_idx >= grid.shape[0]:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        resolution_arr = self._as_1d_float(meta.get("resolution", [0.0]), pad=1)
        resolution = float(resolution_arr[0])
        if resolution <= 0.0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        obstacle_mask = grid[channel_idx] >= float(self.config.social_force_obstacle_threshold)
        if not np.any(obstacle_mask):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        centers = self._grid_cell_centers(indices, origin, resolution)
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        if use_ego:
            centers = self._ego_centers_to_world(centers, robot_pos, robot_heading)

        centers, _dist_sq = self._select_nearby_points(
            centers,
            robot_pos,
            float(self.config.social_force_obstacle_range),
            max(int(self.config.social_force_obstacle_max_points), 0),
        )
        if centers.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        base_radius = (
            0.5 * np.sqrt(2.0) * resolution * float(self.config.social_force_obstacle_radius_scale)
        )
        radii = np.full((centers.shape[0],), base_radius, dtype=float)
        return centers, radii

    def _clip_force(self, force: np.ndarray) -> np.ndarray:
        """Clip total force magnitude to avoid numerical spikes.

        Returns:
            np.ndarray: Clipped force vector.
        """
        if not self.config.social_force_clip_force:
            return force
        norm = float(np.linalg.norm(force))
        if norm < self._EPS or norm <= self.config.social_force_max_force:
            return force
        return force / (norm + self._EPS) * float(self.config.social_force_max_force)


class ORCAPlannerAdapter(SamplingPlannerAdapter):
    """ORCA planner adapter using rvo2 when available.

    Set ``allow_fallback=True`` to use the heuristic implementation when rvo2 is unavailable.
    """

    @dataclass
    class _OrcaLine:
        """ORCA half-plane constraint line."""

        point: np.ndarray
        direction: np.ndarray

    _EPS = 1e-6

    def __init__(self, config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False):
        """Initialize the ORCA adapter with optional rvo2 fallback."""
        self.config = config or SocNavPlannerConfig()
        self._allow_fallback = allow_fallback
        self._fallback_warned = False

    def _ensure_rvo2(self) -> bool:
        """Return True when rvo2 is available, else handle fallback/error behavior.

        Returns:
            bool: True when rvo2 is available, False when falling back.
        """
        if rvo2 is not None:
            return True
        if self._allow_fallback:
            if not self._fallback_warned:
                logger.warning(
                    "rvo2 not available; falling back to heuristic ORCA behavior. "
                    "Install the 'orca' extra for the benchmark-ready implementation.",
                )
                self._fallback_warned = True
            return False
        raise RuntimeError(
            "rvo2 is required for the benchmark-ready ORCA planner. "
            "Install via `uv sync --extra orca` or set allow_fallback=True."
        )

    @staticmethod
    def _det(a: np.ndarray, b: np.ndarray) -> float:
        """2D determinant (cross product z-component).

        Returns:
            float: Determinant value for the 2D vectors.
        """
        return float(a[0] * b[1] - a[1] * b[0])

    @classmethod
    def _normalize(cls, vec: np.ndarray) -> np.ndarray:
        """Return a unit vector (or zeros when norm is too small).

        Returns:
            np.ndarray: Normalized 2D vector or zeros when norm is near zero.
        """
        norm = np.linalg.norm(vec)
        if norm < cls._EPS:
            return np.zeros(2, dtype=float)
        return vec / norm

    @classmethod
    def _linear_program_interval(
        cls,
        lines: list[_OrcaLine],
        line_no: int,
        radius: float,
    ) -> tuple[bool, float, float]:
        """Compute the feasible interval on a constraint line.

        Returns:
            tuple[bool, float, float]: (success, t_left, t_right).
        """
        line = lines[line_no]
        dot = float(np.dot(line.point, line.direction))
        discriminant = dot * dot + radius * radius - float(np.dot(line.point, line.point))
        if discriminant < 0.0:
            return False, 0.0, 0.0
        sqrt_discriminant = float(np.sqrt(discriminant))
        t_left = -dot - sqrt_discriminant
        t_right = -dot + sqrt_discriminant

        for i in range(line_no):
            denom = cls._det(line.direction, lines[i].direction)
            numer = cls._det(lines[i].direction, line.point - lines[i].point)
            if abs(denom) <= cls._EPS:
                if numer < 0.0:
                    return False, 0.0, 0.0
                continue
            t = numer / denom
            if denom >= 0.0:
                t_right = min(t_right, t)
            else:
                t_left = max(t_left, t)
            if t_left > t_right:
                return False, 0.0, 0.0
        return True, t_left, t_right

    @classmethod
    def _linear_program1(
        cls,
        lines: list[_OrcaLine],
        line_no: int,
        radius: float,
        opt_velocity: np.ndarray,
        direction_opt: bool,
    ) -> tuple[bool, np.ndarray]:
        """Solve a 1D linear program on a single constraint line.

        Returns:
            tuple[bool, np.ndarray]: (success, resulting velocity) tuple.
        """
        success, t_left, t_right = cls._linear_program_interval(lines, line_no, radius)
        if not success:
            return False, opt_velocity
        line = lines[line_no]

        if direction_opt:
            if np.dot(opt_velocity, line.direction) > 0.0:
                result = line.point + t_right * line.direction
            else:
                result = line.point + t_left * line.direction
        else:
            t = float(np.dot(line.direction, opt_velocity - line.point))
            if t < t_left:
                result = line.point + t_left * line.direction
            elif t > t_right:
                result = line.point + t_right * line.direction
            else:
                result = line.point + t * line.direction
        return True, result

    @classmethod
    def _linear_program2(
        cls,
        lines: list[_OrcaLine],
        radius: float,
        opt_velocity: np.ndarray,
        direction_opt: bool,
    ) -> tuple[int, np.ndarray]:
        """Solve a 2D linear program with circular bound and half-plane constraints.

        Returns:
            tuple[int, np.ndarray]: (violating line index, resulting velocity).
        """
        if direction_opt:
            result = cls._normalize(opt_velocity) * radius
        elif np.linalg.norm(opt_velocity) > radius:
            result = cls._normalize(opt_velocity) * radius
        else:
            result = opt_velocity.copy()

        for i, line in enumerate(lines):
            if cls._det(line.direction, line.point - result) > 0.0:
                temp = result.copy()
                success, result = cls._linear_program1(
                    lines, i, radius, opt_velocity, direction_opt
                )
                if not success:
                    return i, temp
        return len(lines), result

    @classmethod
    def _linear_program3(
        cls,
        lines: list[_OrcaLine],
        num_obst_lines: int,
        begin_line: int,
        radius: float,
        result: np.ndarray,
    ) -> np.ndarray:
        """Resolve infeasible constraints via projection (ORCA fallback).

        Returns:
            np.ndarray: Adjusted velocity satisfying constraints when possible.
        """
        distance = 0.0
        for i in range(begin_line, len(lines)):
            if cls._det(lines[i].direction, lines[i].point - result) > distance:
                proj_lines = list(lines[:num_obst_lines])
                for j in range(num_obst_lines, i):
                    determinant = cls._det(lines[i].direction, lines[j].direction)
                    if abs(determinant) <= cls._EPS:
                        if np.dot(lines[i].direction, lines[j].direction) > 0.0:
                            continue
                        point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        point = (
                            lines[i].point
                            + (
                                cls._det(lines[j].direction, lines[i].point - lines[j].point)
                                / determinant
                            )
                            * lines[i].direction
                        )
                    direction = cls._normalize(lines[j].direction - lines[i].direction)
                    proj_lines.append(cls._OrcaLine(point=point, direction=direction))
                temp_result = result.copy()
                perp_direction = np.array([-lines[i].direction[1], lines[i].direction[0]])
                _idx, result = cls._linear_program2(proj_lines, radius, perp_direction, True)
                if cls._det(lines[i].direction, lines[i].point - result) > distance:
                    result = temp_result
                distance = cls._det(lines[i].direction, lines[i].point - result)
        return result

    @staticmethod
    def _ego_to_world(vec: np.ndarray, heading: float) -> np.ndarray:
        """Rotate an ego-frame vector into world coordinates.

        Returns:
            np.ndarray: Vector expressed in world coordinates.
        """
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        return np.array([cos_h * vec[0] - sin_h * vec[1], sin_h * vec[0] + cos_h * vec[1]])

    @staticmethod
    def _world_to_ego_vec(vec: np.ndarray, heading: float) -> np.ndarray:
        """Rotate a world-frame vector into ego coordinates.

        Returns:
            np.ndarray: Vector expressed in ego coordinates.
        """
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        return np.array([cos_h * vec[0] + sin_h * vec[1], -sin_h * vec[0] + cos_h * vec[1]])

    @classmethod
    def _preferred_velocity(
        cls, goal: np.ndarray, robot_pos: np.ndarray, robot_heading: float, max_speed: float
    ) -> np.ndarray:
        """Compute the preferred velocity toward the goal in ego coordinates.

        Returns:
            np.ndarray: Preferred velocity in ego coordinates.
        """
        goal_ego = np.asarray(
            world_to_ego(float(goal[0]), float(goal[1]), (robot_pos, robot_heading)),
            dtype=float,
        )
        return cls._normalize(goal_ego) * max_speed

    @classmethod
    def _extract_pedestrians(cls, ped_state: dict) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Extract pedestrian positions/velocities and metadata.

        Returns:
            tuple[np.ndarray, np.ndarray, int, float]: Positions, velocities, count, radius.
        """
        raw_positions = ped_state.get("positions")
        if raw_positions is None:
            ped_positions = np.zeros((0, 2), dtype=float)
        else:
            ped_positions = np.asarray(raw_positions, dtype=float)
        raw_velocities = ped_state.get("velocities")
        if raw_velocities is None:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        else:
            ped_velocities = np.asarray(raw_velocities, dtype=float)
        ped_count = int(np.asarray(ped_state.get("count", [0]), dtype=float)[0])
        ped_positions = ped_positions[:ped_count]
        if ped_velocities.size == 0:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        ped_velocities = ped_velocities[:ped_count]
        ped_radius_arr = np.asarray(ped_state.get("radius", [0.3]), dtype=float)
        ped_radius = float(ped_radius_arr[0] if ped_radius_arr.ndim > 0 else ped_radius_arr)
        return ped_positions, ped_velocities, ped_count, ped_radius

    def _build_orca_lines(  # noqa: PLR0913
        self,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_velocity: np.ndarray,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        robot_radius: float,
        ped_radius: float | np.ndarray,
        time_step: float,
        time_horizon: float | None = None,
        neighbor_dist: float | None = None,
    ) -> list[_OrcaLine]:
        """Build ORCA half-plane constraints for nearby pedestrians/obstacles.

        Returns:
            list[_OrcaLine]: ORCA half-plane constraints.
        """
        lines: list[self._OrcaLine] = []
        effective_time_horizon = max(
            time_horizon if time_horizon is not None else self.config.orca_time_horizon,
            self._EPS,
        )
        effective_neighbor_dist = max(
            neighbor_dist if neighbor_dist is not None else self.config.orca_neighbor_dist,
            0.0,
        )
        neighbor_dist_sq = effective_neighbor_dist**2
        inv_time_horizon = 1.0 / effective_time_horizon
        inv_time_step = 1.0 / max(time_step, self._EPS)

        use_radius_array = isinstance(ped_radius, np.ndarray)
        for index, (ped_pos_world, ped_vel) in enumerate(
            zip(ped_positions, ped_velocities, strict=False)
        ):
            ped_pos_ego = np.asarray(
                world_to_ego(
                    float(ped_pos_world[0]),
                    float(ped_pos_world[1]),
                    (robot_pos, robot_heading),
                ),
                dtype=float,
            )
            if np.dot(ped_pos_ego, ped_pos_ego) > neighbor_dist_sq:
                continue

            rel_pos = ped_pos_ego
            rel_vel = robot_velocity - ped_vel
            dist_sq = float(np.dot(rel_pos, rel_pos))
            ped_radius_value = float(ped_radius[index]) if use_radius_array else float(ped_radius)
            combined_radius = robot_radius + ped_radius_value
            combined_radius_sq = combined_radius**2

            if dist_sq > combined_radius_sq:
                w = rel_vel - inv_time_horizon * rel_pos
                w_length_sq = float(np.dot(w, w))
                dot = float(np.dot(w, rel_pos))
                if dot < 0.0 and dot * dot > combined_radius_sq * w_length_sq:
                    w_length = float(np.sqrt(w_length_sq))
                    unit_w = w / max(w_length, self._EPS)
                    direction = np.array([unit_w[1], -unit_w[0]])
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    leg = float(np.sqrt(max(dist_sq - combined_radius_sq, 0.0)))
                    if self._det(rel_pos, w) > 0.0:
                        direction = np.array(
                            [
                                rel_pos[0] * leg - rel_pos[1] * combined_radius,
                                rel_pos[1] * leg + rel_pos[0] * combined_radius,
                            ]
                        ) / max(dist_sq, self._EPS)
                    else:
                        direction = -np.array(
                            [
                                rel_pos[0] * leg + rel_pos[1] * combined_radius,
                                rel_pos[1] * leg - rel_pos[0] * combined_radius,
                            ]
                        ) / max(dist_sq, self._EPS)
                    u = float(np.dot(rel_vel, direction)) * direction - rel_vel
            else:
                w = rel_vel - inv_time_step * rel_pos
                w_length = float(np.linalg.norm(w))
                unit_w = w / max(w_length, self._EPS)
                direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * inv_time_step - w_length) * unit_w

            lines.append(self._OrcaLine(point=robot_velocity + 0.5 * u, direction=direction))
        return lines

    @staticmethod
    def _grid_cell_centers(
        indices: np.ndarray, origin: np.ndarray, resolution: float
    ) -> np.ndarray:
        """Convert grid indices to grid-frame centers.

        Returns:
            np.ndarray: Grid-frame centers for the provided indices.
        """
        rows = indices[:, 0].astype(float)
        cols = indices[:, 1].astype(float)
        x = origin[0] + (cols + 0.5) * resolution
        y = origin[1] + (rows + 0.5) * resolution
        return np.stack([x, y], axis=1)

    @staticmethod
    def _ego_centers_to_world(
        centers: np.ndarray, robot_pos: np.ndarray, robot_heading: float
    ) -> np.ndarray:
        """Rotate/translate ego-frame centers into world coordinates.

        Returns:
            np.ndarray: World-space centers.
        """
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        x_world = cos_h * centers[:, 0] - sin_h * centers[:, 1]
        y_world = sin_h * centers[:, 0] + cos_h * centers[:, 1]
        return np.stack([x_world, y_world], axis=1) + np.asarray(robot_pos, dtype=float)

    @staticmethod
    def _select_nearby_points(
        centers: np.ndarray,
        robot_pos: np.ndarray,
        max_range: float,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter centers by range and cap to the closest points.

        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered centers and squared distances.
        """
        offsets = centers - np.asarray(robot_pos, dtype=float)
        dist_sq = np.einsum("ij,ij->i", offsets, offsets)
        keep = dist_sq <= max_range**2
        if not np.any(keep):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
        centers = centers[keep]
        dist_sq = dist_sq[keep]
        if max_points > 0 and centers.shape[0] > max_points:
            order = np.argsort(dist_sq)[:max_points]
            centers = centers[order]
            dist_sq = dist_sq[order]
        return centers, dist_sq

    def _extract_obstacles_from_grid(
        self, observation: dict, robot_pos: np.ndarray, robot_heading: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nearby obstacle points from the occupancy grid.

        Returns:
            tuple[np.ndarray, np.ndarray]: World-space obstacle centers and per-point radii.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        grid, meta = payload
        if grid.ndim < 3:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        channel_idx = self._grid_channel_index(meta, "obstacles")
        if channel_idx < 0:
            channel_idx = self._grid_channel_index(meta, "combined")
        if channel_idx < 0 or channel_idx >= grid.shape[0]:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        resolution_arr = self._as_1d_float(meta.get("resolution", [0.0]), pad=1)
        resolution = float(resolution_arr[0])
        if resolution <= 0.0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        obstacle_mask = grid[channel_idx] >= float(self.config.orca_obstacle_threshold)
        if not np.any(obstacle_mask):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        centers = self._grid_cell_centers(indices, origin, resolution)
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        if use_ego:
            centers = self._ego_centers_to_world(centers, robot_pos, robot_heading)

        centers, _dist_sq = self._select_nearby_points(
            centers,
            robot_pos,
            float(self.config.orca_obstacle_range),
            max(int(self.config.orca_obstacle_max_points), 0),
        )
        if centers.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        base_radius = (
            0.5 * np.sqrt(2.0) * resolution * float(self.config.orca_obstacle_radius_scale)
        )
        radii = np.full((centers.shape[0],), base_radius, dtype=float)
        return centers, radii

    def _solve_orca_velocity(
        self, lines: list[_OrcaLine], preferred_velocity: np.ndarray
    ) -> np.ndarray:
        """Solve ORCA constraints for the new velocity.

        Returns:
            np.ndarray: Resulting velocity that satisfies the constraints.
        """
        if not lines:
            return preferred_velocity
        line_fail, new_velocity = self._linear_program2(
            lines,
            self.config.max_linear_speed,
            preferred_velocity,
            False,
        )
        if line_fail < len(lines):
            new_velocity = self._linear_program3(
                lines,
                num_obst_lines=0,
                begin_line=line_fail,
                radius=self.config.max_linear_speed,
                result=new_velocity,
            )
        return new_velocity

    def _velocity_to_command(
        self,
        *,
        velocity: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
        observation: dict,
    ) -> tuple[float, float]:
        """Convert a velocity vector into (v, w) command with occupancy penalty.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        speed = float(np.linalg.norm(velocity))
        if speed < self._EPS:
            return 0.0, 0.0
        world_dir = self._ego_to_world(velocity, robot_heading)
        world_dir = self._normalize(world_dir)
        world_dir, occ_penalty = self._get_safe_heading(robot_pos, world_dir, observation)
        desired_heading = atan2(world_dir[1], world_dir[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                1.5 * self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )
        heading_scale = 1.0 - min(1.0, abs(heading_error) / (pi / 2)) * float(
            self.config.orca_heading_slowdown
        )
        linear = float(
            np.clip(
                speed,
                0.0,
                self.config.max_linear_speed
                * max(0.0, 1.0 - occ_penalty)
                * max(0.0, heading_scale),
            )
        )
        return linear, angular

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) using rvo2 ORCA or a heuristic fallback.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        if not self._ensure_rvo2():
            return self._heuristic_plan(observation)
        return self._rvo2_plan(observation)

    def _rvo2_plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) using the rvo2 ORCA solver.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        preferred_velocity_ego = self._preferred_velocity(
            goal, robot_pos, robot_heading, self.config.max_linear_speed
        )
        if np.linalg.norm(preferred_velocity_ego) < self._EPS:
            return 0.0, 0.0

        preferred_velocity_world = self._ego_to_world(preferred_velocity_ego, robot_heading)

        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        if time_step <= self._EPS:
            logger.warning(
                "Invalid timestep ({}) for ORCA planner; defaulting to 0.1s.",
                time_step,
            )
            time_step = 0.1

        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])
        robot_speed = float(np.asarray(robot_state.get("speed", [0.0]), dtype=float)[0])
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        robot_velocity_world = np.array(
            [robot_speed * cos_h, robot_speed * sin_h],
            dtype=float,
        )

        ped_positions, ped_velocities, ped_count, ped_radius = self._extract_pedestrians(ped_state)
        if ped_count > 0:
            ped_vel_world = np.zeros_like(ped_velocities, dtype=float)
            ped_vel_world[:, 0] = cos_h * ped_velocities[:, 0] - sin_h * ped_velocities[:, 1]
            ped_vel_world[:, 1] = sin_h * ped_velocities[:, 0] + cos_h * ped_velocities[:, 1]
        else:
            ped_vel_world = np.zeros_like(ped_velocities, dtype=float)

        max_neighbors = int(self.config.orca_max_neighbors)
        if max_neighbors <= 0:
            max_neighbors = max(1, ped_count)

        neighbor_dist = float(self.config.orca_neighbor_dist)
        time_horizon = float(self.config.orca_time_horizon)
        time_horizon_obst = float(self.config.orca_time_horizon_obst)
        max_speed = float(self.config.max_linear_speed)

        sim = rvo2.PyRVOSimulator(
            time_step,
            neighbor_dist,
            max_neighbors,
            time_horizon,
            time_horizon_obst,
            robot_radius,
            max_speed,
        )
        robot_id = sim.addAgent(
            tuple(robot_pos),
            neighbor_dist,
            max_neighbors,
            time_horizon,
            time_horizon_obst,
            robot_radius,
            max_speed,
            tuple(robot_velocity_world),
        )

        ped_ids: list[int] = []
        for idx in range(ped_count):
            ped_speed = float(np.linalg.norm(ped_vel_world[idx]))
            ped_max_speed = max(ped_speed, max_speed)
            ped_id = sim.addAgent(
                tuple(ped_positions[idx]),
                neighbor_dist,
                max_neighbors,
                time_horizon,
                time_horizon_obst,
                ped_radius,
                ped_max_speed,
                tuple(ped_vel_world[idx]),
            )
            ped_ids.append(ped_id)

        obstacle_positions, obstacle_radii = self._extract_obstacles_from_grid(
            observation, robot_pos, robot_heading
        )
        if obstacle_positions.size:
            for center, radius in zip(obstacle_positions, obstacle_radii, strict=False):
                half = float(radius)
                vertices = [
                    (float(center[0] - half), float(center[1] - half)),
                    (float(center[0] + half), float(center[1] - half)),
                    (float(center[0] + half), float(center[1] + half)),
                    (float(center[0] - half), float(center[1] + half)),
                ]
                sim.addObstacle(vertices)
            sim.processObstacles()

        sim.setAgentPrefVelocity(robot_id, tuple(preferred_velocity_world))
        for ped_id, vel in zip(ped_ids, ped_vel_world, strict=False):
            sim.setAgentPrefVelocity(ped_id, tuple(vel))

        sim.doStep()
        new_velocity_world = np.asarray(sim.getAgentVelocity(robot_id), dtype=float)
        new_velocity_ego = self._world_to_ego_vec(new_velocity_world, robot_heading)
        return self._velocity_to_command(
            velocity=new_velocity_ego,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            observation=observation,
        )

    def _heuristic_plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) using the legacy ORCA-inspired heuristic.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        preferred_velocity = self._preferred_velocity(
            goal, robot_pos, robot_heading, self.config.max_linear_speed
        )
        if np.linalg.norm(preferred_velocity) < self._EPS:
            return 0.0, 0.0

        ped_positions, ped_velocities, _ped_count, ped_radius = self._extract_pedestrians(ped_state)
        robot_speed = float(np.asarray(robot_state.get("speed", [0.0]), dtype=float)[0])
        robot_velocity = np.array([robot_speed, 0.0], dtype=float)

        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])

        lines = self._build_orca_lines(
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            robot_velocity=robot_velocity,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
            time_step=time_step,
        )
        obstacle_positions, obstacle_radii = self._extract_obstacles_from_grid(
            observation, robot_pos, robot_heading
        )
        if obstacle_positions.size:
            obstacle_velocities = np.zeros_like(obstacle_positions, dtype=float)
            lines.extend(
                self._build_orca_lines(
                    robot_pos=robot_pos,
                    robot_heading=robot_heading,
                    robot_velocity=robot_velocity,
                    ped_positions=obstacle_positions,
                    ped_velocities=obstacle_velocities,
                    robot_radius=robot_radius,
                    ped_radius=obstacle_radii,
                    time_step=time_step,
                    time_horizon=self.config.orca_time_horizon_obst,
                    neighbor_dist=self.config.orca_obstacle_range,
                )
            )
        new_velocity = self._solve_orca_velocity(lines, preferred_velocity)
        return self._velocity_to_command(
            velocity=new_velocity,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            observation=observation,
        )


def _sacadrl_actions() -> np.ndarray:
    """Return the discrete GA3C-CADRL action set (speed scale, delta heading)."""
    actions = np.mgrid[1.0:1.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 12].reshape(2, -1).T
    actions = np.vstack(
        [
            actions,
            np.mgrid[0.5:0.6:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T,
        ]
    )
    actions = np.vstack(
        [
            actions,
            np.mgrid[0.0:0.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T,
        ]
    )
    return actions


class _SACADRLModel:
    """Tensorflow checkpoint wrapper for GA3C-CADRL policy inference."""

    def __init__(self, checkpoint_prefix: Path, *, device: str = "/cpu:0"):
        """Load the GA3C-CADRL model from the provided checkpoint prefix."""
        if tf is None:  # pragma: no cover - optional dependency
            raise RuntimeError("TensorFlow is required to run the GA3C-CADRL (SA-CADRL) baseline.")

        self._tf = tf
        self._actions = _sacadrl_actions()
        self._graph = self._tf.Graph()
        with self._graph.as_default():
            with self._tf.device(device):
                self._sess = self._tf.Session(
                    graph=self._graph,
                    config=self._tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=self._tf.GPUOptions(allow_growth=True),
                    ),
                )
                saver = self._tf.train.import_meta_graph(
                    f"{checkpoint_prefix}.meta", clear_devices=True
                )
                self._sess.run(self._tf.global_variables_initializer())
                saver.restore(self._sess, str(checkpoint_prefix))
                self._softmax = self._graph.get_tensor_by_name("Softmax:0")
                self._x = self._graph.get_tensor_by_name("X:0")
        self._input_dim = int(self._x.shape[-1])

    @property
    def actions(self) -> np.ndarray:
        """Discrete action table of [speed_scale, delta_heading]."""
        return self._actions

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return softmax action probabilities for the provided observations."""
        obs = self._crop(obs)
        return self._sess.run(self._softmax, feed_dict={self._x: obs})

    def _crop(self, obs: np.ndarray) -> np.ndarray:
        """Pad or crop observations to match the expected input dimension.

        Returns:
            np.ndarray: Observation array sized to the network input dimension.
        """
        if obs.shape[-1] > self._input_dim:
            return obs[:, : self._input_dim]
        if obs.shape[-1] < self._input_dim:
            padded = np.zeros((obs.shape[0], self._input_dim), dtype=obs.dtype)
            padded[:, : obs.shape[1]] = obs
            return padded
        return obs


class SACADRLPlannerAdapter(SamplingPlannerAdapter):
    """GA3C-CADRL (SA-CADRL) planner adapter backed by a TensorFlow checkpoint.

    Set ``allow_fallback=True`` to permit heuristic behavior when the checkpoint
    or TensorFlow dependency is unavailable.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False):
        """Initialize the adapter and configure optional heuristic fallback."""
        self.config = config or SocNavPlannerConfig()
        self._allow_fallback = allow_fallback
        self._model: _SACADRLModel | None = None
        self._load_error: Exception | None = None
        self._fallback_warned = False

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using the GA3C-CADRL model (or fallback heuristics).

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        model = self._ensure_model()
        if model is None:
            return self._heuristic_plan(observation)

        obs_vec, pref_speed, dist_to_goal = self._build_network_input(observation)
        if dist_to_goal <= self.config.goal_tolerance:
            return 0.0, 0.0

        predictions = model.predict(obs_vec)[0]
        action_idx = int(np.argmax(predictions))
        raw_action = model.actions[action_idx]
        linear = float(pref_speed * raw_action[0])
        delta_heading = float(raw_action[1])

        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        if time_step <= 1e-6:
            logger.warning(
                "Invalid timestep ({}) for SACADRLPlannerAdapter; defaulting to 0.1s.",
                time_step,
            )
            time_step = 0.1

        angular = float(delta_heading / time_step)
        linear = float(np.clip(linear, 0.0, self.config.max_linear_speed))
        angular = float(
            np.clip(angular, -self.config.max_angular_speed, self.config.max_angular_speed)
        )
        return linear, angular

    def _ensure_model(self) -> _SACADRLModel | None:
        """Load the model checkpoint on demand and honor fallback settings.

        Returns:
            _SACADRLModel | None: Loaded model instance or ``None`` when falling back.
        """
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None if self._allow_fallback else self._raise_cached_error()
        try:
            self._model = self._build_model()
        except Exception as exc:
            if self._allow_fallback:
                self._load_error = exc
                if not self._fallback_warned:
                    logger.warning(
                        "Falling back to heuristic SACADRL behavior: {}. "
                        "Set allow_fallback=False to fail fast.",
                        exc,
                    )
                    self._fallback_warned = True
                return None
            raise
        return self._model

    def _raise_cached_error(self) -> None:
        """Re-raise cached initialization error when fallback is disabled."""
        assert self._load_error is not None
        raise self._load_error

    def _build_model(self) -> _SACADRLModel:
        """Resolve the GA3C-CADRL checkpoint and construct the TF model wrapper.

        Returns:
            _SACADRLModel: Loaded GA3C-CADRL model wrapper.
        """
        checkpoint_prefix = self._resolve_checkpoint_prefix()
        return _SACADRLModel(checkpoint_prefix, device="/cpu:0")

    def _resolve_checkpoint_prefix(self) -> Path:
        """Resolve the model checkpoint prefix for the GA3C-CADRL checkpoint.

        Returns:
            Path: Checkpoint prefix path without file extensions.
        """
        if self.config.sacadrl_checkpoint_path:
            checkpoint_path = Path(self.config.sacadrl_checkpoint_path).expanduser()
        else:
            checkpoint_path = resolve_model_path(self.config.sacadrl_model_id)

        if checkpoint_path.suffix == ".meta":
            prefix = checkpoint_path.with_suffix("")
        else:
            prefix = checkpoint_path

        meta_path = prefix.with_suffix(".meta")
        if not meta_path.exists():
            raise FileNotFoundError(f"GA3C-CADRL checkpoint meta file not found: {meta_path}")
        index_path = prefix.with_suffix(".index")
        if not index_path.exists():
            raise FileNotFoundError(f"GA3C-CADRL checkpoint index file not found: {index_path}")
        data_files = list(prefix.parent.glob(f"{prefix.name}.data*"))
        if not data_files:
            raise FileNotFoundError(
                f"GA3C-CADRL checkpoint data file not found for prefix: {prefix}"
            )
        return prefix

    def _compute_goal_frame(
        self, robot_pos: np.ndarray, robot_heading: float, goal: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
        """Compute the goal-aligned frame and heading in ego coordinates.

        Returns:
            tuple[float, np.ndarray, np.ndarray, float]: Distance to goal, parallel unit
            vector, orthogonal unit vector, and heading in ego frame.
        """
        to_goal = goal - robot_pos
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal > 1e-8:
            ref_prll = to_goal / dist_to_goal
        else:
            ref_prll = np.array([1.0, 0.0], dtype=float)
        ref_orth = np.array([-ref_prll[1], ref_prll[0]], dtype=float)
        ref_angle = atan2(ref_prll[1], ref_prll[0])
        heading_ego_frame = self._wrap_angle(robot_heading - ref_angle)
        return dist_to_goal, ref_prll, ref_orth, heading_ego_frame

    def _normalize_pedestrians(  # noqa: C901
        self, ped_state: dict
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Normalize pedestrian positions/velocities and radius from observation.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Positions array, velocities array,
            and shared pedestrian radius.
        """
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        if ped_positions.size == 0:
            ped_positions = np.zeros((0, 2), dtype=float)
        elif ped_positions.ndim == 1:
            ped_positions = (
                ped_positions.reshape(-1, 2)
                if ped_positions.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )
        elif ped_positions.ndim == 2 and ped_positions.shape[1] != 2:
            if ped_positions.shape[1] > 2:
                ped_positions = ped_positions[:, :2]
            else:
                ped_positions = np.pad(
                    ped_positions,
                    ((0, 0), (0, 2 - ped_positions.shape[1])),
                    constant_values=0.0,
                )
        elif ped_positions.ndim != 2:
            ped_positions = np.zeros((0, 2), dtype=float)

        count_arr = np.asarray(
            ped_state.get("count", [ped_positions.shape[0]]), dtype=float
        ).reshape(-1)
        ped_count = int(count_arr[0]) if count_arr.size else int(ped_positions.shape[0])
        ped_count = max(0, min(ped_count, int(ped_positions.shape[0])))
        ped_positions = ped_positions[:ped_count]

        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_velocities.size == 0:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        elif ped_velocities.ndim == 1:
            ped_velocities = (
                ped_velocities.reshape(-1, 2)
                if ped_velocities.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )
        elif ped_velocities.ndim == 2 and ped_velocities.shape[1] != 2:
            if ped_velocities.shape[1] > 2:
                ped_velocities = ped_velocities[:, :2]
            else:
                ped_velocities = np.pad(
                    ped_velocities,
                    ((0, 0), (0, 2 - ped_velocities.shape[1])),
                    constant_values=0.0,
                )
        elif ped_velocities.ndim != 2:
            ped_velocities = np.zeros((0, 2), dtype=float)

        if ped_velocities.shape[0] < ped_count:
            pad_rows = ped_count - ped_velocities.shape[0]
            ped_velocities = np.pad(
                ped_velocities,
                ((0, pad_rows), (0, 0)),
                constant_values=0.0,
            )
        ped_velocities = ped_velocities[:ped_count]

        radius_arr = np.asarray(ped_state.get("radius", [0.3]), dtype=float).reshape(-1)
        ped_radius = float(radius_arr[0]) if radius_arr.size else 0.3
        return ped_positions, ped_velocities, ped_radius

    def _ego_to_global_velocities(
        self, robot_heading: float, ped_velocities: np.ndarray
    ) -> np.ndarray:
        """Convert ego-frame pedestrian velocities to global-frame velocities.

        Returns:
            np.ndarray: Global-frame velocities with the same shape as input.
        """
        cos_h = np.cos(robot_heading)
        sin_h = np.sin(robot_heading)
        v_global = np.zeros_like(ped_velocities, dtype=float)
        if ped_velocities.size:
            v_global[:, 0] = cos_h * ped_velocities[:, 0] - sin_h * ped_velocities[:, 1]
            v_global[:, 1] = sin_h * ped_velocities[:, 0] + cos_h * ped_velocities[:, 1]
        return v_global

    def _build_other_agents_states(
        self,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        robot_pos: np.ndarray,
        robot_radius: float,
        ped_radius: float,
        ref_prll: np.ndarray,
        ref_orth: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Build the ordered other-agent state tensor for the GA3C-CADRL input.

        Returns:
            tuple[np.ndarray, float]: Other-agent state matrix and count.
        """
        max_other = max(0, int(self.config.sacadrl_max_other_agents))
        other_states = np.zeros((max_other, 7), dtype=float)
        sorting = []
        for idx in range(ped_positions.shape[0]):
            rel = ped_positions[idx] - robot_pos
            dist_center = float(np.linalg.norm(rel))
            dist_2_other = dist_center - robot_radius - ped_radius
            p_orth = float(np.dot(rel, ref_orth))
            sorting.append((idx, dist_2_other, p_orth))

        if sorting:
            if self.config.sacadrl_sorting_method == "closest_last":
                sorted_ids = sorted(sorting, key=lambda x: (-x[1], x[2]))
            else:
                sorted_ids = sorted(sorting, key=lambda x: (x[1], x[2]))
            selected = [idx for idx, _dist, _orth in sorted_ids[:max_other]]
        else:
            selected = []

        for slot, idx in enumerate(selected):
            rel = ped_positions[idx] - robot_pos
            p_parallel = float(np.dot(rel, ref_prll))
            p_orth = float(np.dot(rel, ref_orth))
            v_parallel = float(np.dot(ped_velocities[idx], ref_prll))
            v_orth = float(np.dot(ped_velocities[idx], ref_orth))
            dist_2_other = float(np.linalg.norm(rel) - robot_radius - ped_radius)
            combined_radius = robot_radius + ped_radius
            other_states[slot] = np.array(
                [
                    p_parallel,
                    p_orth,
                    v_parallel,
                    v_orth,
                    ped_radius,
                    combined_radius,
                    dist_2_other,
                ],
                dtype=float,
            )
        return other_states, float(len(selected))

    def _build_network_input(self, observation: dict) -> tuple[np.ndarray, float, float]:
        """Convert a SocNav observation into the GA3C-CADRL network input vector.

        Returns:
            tuple[np.ndarray, float, float]: Batched observation vector, preferred speed,
            and distance to goal.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])

        goal = np.asarray(goal_state["current"], dtype=float)
        dist_to_goal, ref_prll, ref_orth, heading_ego_frame = self._compute_goal_frame(
            robot_pos, robot_heading, goal
        )

        ped_positions, ped_velocities, ped_radius = self._normalize_pedestrians(ped_state)
        v_global = self._ego_to_global_velocities(robot_heading, ped_velocities)
        other_states, num_other_agents = self._build_other_agents_states(
            ped_positions,
            v_global,
            robot_pos,
            robot_radius,
            ped_radius,
            ref_prll,
            ref_orth,
        )
        pref_speed = float(self.config.sacadrl_pref_speed)

        obs_dict = {
            "num_other_agents": np.array([num_other_agents], dtype=np.float32),
            "dist_to_goal": np.array([dist_to_goal], dtype=np.float32),
            "heading_ego_frame": np.array([heading_ego_frame], dtype=np.float32),
            "pref_speed": np.array([pref_speed], dtype=np.float32),
            "radius": np.array([robot_radius], dtype=np.float32),
            "other_agents_states": other_states.astype(np.float32),
        }
        vec_obs = np.array([], dtype=np.float32)
        for state in _SACADRL_STATE_ORDER:
            vec_obs = np.hstack([vec_obs, obs_dict[state].flatten()])
        vec_obs = np.expand_dims(vec_obs, axis=0)
        return vec_obs, pref_speed, dist_to_goal

    def _heuristic_plan(self, observation: dict) -> tuple[float, float]:
        """Fallback heuristic that biases toward the goal while repulsing pedestrians.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions, _ped_velocities, _ped_radius = self._normalize_pedestrians(ped_state)
        if ped_positions.shape[0] > 0:
            dists = np.linalg.norm(ped_positions - robot_pos, axis=1)
            neighbor_count = max(0, int(self.config.sacadrl_neighbors))
            nearest_idx = np.argsort(dists)[:neighbor_count]
            bias = np.zeros(2, dtype=float)
            for idx in nearest_idx:
                delta = robot_pos - ped_positions[idx]
                dist = dists[idx] + 1e-6
                bias += delta / dist**1.5
            combined = goal_vec + self.config.sacadrl_bias_weight * bias
        else:
            combined = goal_vec

        if np.linalg.norm(combined) > 1e-6:
            combined = combined / np.linalg.norm(combined)

        combined, occ_penalty = self._get_safe_heading(robot_pos, combined, observation)

        desired_heading = atan2(combined[1], combined[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                np.linalg.norm(to_goal),
                0.0,
                self.config.max_linear_speed
                * max(0.0, 1.0 - abs(heading_error) / pi)
                * max(0.0, 1.0 - occ_penalty),
            ),
        )
        return linear, angular


class PredictionPlannerAdapter(SamplingPlannerAdapter):
    """RGL-inspired predictive planner using a learned trajectory model + sampled rollout."""

    _EPS = 1e-6

    def __init__(self, config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False):
        """Initialize predictive planner adapter and deferred model loading."""
        self.config = config or SocNavPlannerConfig()
        self._allow_fallback = bool(allow_fallback)
        self._model: PredictiveTrajectoryModel | None = None
        self._load_error: Exception | None = None
        self._fallback_warned = False
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        """Resolve runtime device string for predictive model inference.

        Returns:
            str: Torch device identifier.
        """
        requested = str(self.config.predictive_device).strip().lower()
        if requested.startswith("cuda"):
            if torch is not None and torch.cuda.is_available():
                return requested
            logger.warning(
                "Predictive planner requested device '{}' but CUDA is unavailable; using CPU.",
                requested,
            )
        return "cpu"

    def _ensure_model(self) -> PredictiveTrajectoryModel | None:
        """Load predictive model checkpoint on-demand.

        Returns:
            PredictiveTrajectoryModel | None: Model instance or None when fallback is enabled.
        """
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None if self._allow_fallback else self._raise_cached_error()
        try:
            self._model = self._build_model()
        except Exception as exc:
            if self._allow_fallback:
                self._load_error = exc
                if not self._fallback_warned:
                    logger.warning(
                        "Falling back to constant-velocity predictive planner behavior: {}. "
                        "Set allow_fallback=False to fail fast.",
                        exc,
                    )
                    self._fallback_warned = True
                return None
            raise
        return self._model

    def _raise_cached_error(self) -> None:
        """Re-raise cached predictive-model initialization error."""
        assert self._load_error is not None
        raise self._load_error

    def _resolve_checkpoint_path(self) -> Path:
        """Resolve predictive model checkpoint path.

        Returns:
            Path: Checkpoint path.
        """
        if self.config.predictive_checkpoint_path:
            checkpoint = Path(self.config.predictive_checkpoint_path).expanduser()
        else:
            checkpoint = resolve_model_path(self.config.predictive_model_id)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Predictive planner checkpoint not found: {checkpoint}")
        return checkpoint

    def _build_model(self) -> PredictiveTrajectoryModel:
        """Construct predictive model from a checkpoint.

        Returns:
            PredictiveTrajectoryModel: Loaded model instance.
        """
        if torch is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "PyTorch is required for PredictionPlannerAdapter. Install torch dependency."
            )
        checkpoint_path = self._resolve_checkpoint_path()
        model, _payload = load_predictive_checkpoint(checkpoint_path, map_location=self._device)
        model.eval()
        return model

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

    def _build_model_input(
        self, observation: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Build predictive-model input from SocNav structured observation.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            ``(state, mask, robot_pos, robot_heading)``.
        """
        robot_state, _goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        ped_positions, ped_velocities_ego = self._normalize_pedestrians(ped_state)

        max_agents = max(1, int(self.config.predictive_max_agents))
        state = np.zeros((max_agents, 4), dtype=np.float32)
        mask = np.zeros((max_agents,), dtype=np.float32)
        count = min(max_agents, ped_positions.shape[0])
        if count > 0:
            rel = ped_positions[:count] - robot_pos.reshape(1, 2)
            cos_h = float(np.cos(robot_heading))
            sin_h = float(np.sin(robot_heading))
            rel_x = cos_h * rel[:, 0] + sin_h * rel[:, 1]
            rel_y = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
            state[:count, 0] = rel_x
            state[:count, 1] = rel_y
            state[:count, 2:4] = ped_velocities_ego[:count]
            mask[:count] = 1.0
        return state, mask, robot_pos, robot_heading

    def _constant_velocity_prediction(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate constant-velocity fallback trajectories.

        Returns:
            np.ndarray: Predicted trajectories ``(N, T, 2)`` in robot frame.
        """
        steps = max(1, int(self.config.predictive_horizon_steps))
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        future = np.zeros((state.shape[0], steps, 2), dtype=np.float32)
        for t in range(steps):
            tau = float(t + 1) * dt
            future[:, t, 0] = state[:, 0] + tau * state[:, 2]
            future[:, t, 1] = state[:, 1] + tau * state[:, 3]
        future *= mask[:, None, None]
        return future

    def _predict_trajectories(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict future pedestrian trajectories in robot frame.

        Returns:
            np.ndarray: Predicted trajectories ``(N, T, 2)``.
        """
        model = self._ensure_model()
        if model is None:
            return self._constant_velocity_prediction(state, mask)
        assert torch is not None
        with torch.no_grad():
            state_t = torch.from_numpy(state[None]).to(self._device)
            mask_t = torch.from_numpy(mask[None]).to(self._device)
            out = model(state_t, mask_t)
            future = out["future_positions"][0].detach().cpu().numpy().astype(np.float32)
        return future

    def _min_predicted_distance(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        steps: int | None = None,
    ) -> float:
        """Return minimum predicted ped distance to robot origin in local frame."""
        if future_peds.size == 0:
            return float("inf")
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return float("inf")
        t_max = future_peds.shape[1] if steps is None else min(int(steps), future_peds.shape[1])
        if t_max <= 0:
            return float("inf")
        valid = future_peds[valid_idx, :t_max, :]
        dist = np.linalg.norm(valid, axis=2)
        return float(np.min(dist)) if dist.size > 0 else float("inf")

    def _effective_rollout_steps(self, *, future_peds: np.ndarray, mask: np.ndarray) -> int:
        """Select evaluation horizon steps, with optional near-field boosting.

        Returns:
            int: Number of rollout steps for candidate evaluation.
        """
        base_steps = max(1, int(self.config.predictive_horizon_steps))
        max_steps = max(1, int(future_peds.shape[1]))
        if not bool(self.config.predictive_adaptive_horizon_enabled):
            return min(base_steps, max_steps)
        min_pred_dist = self._min_predicted_distance(
            future_peds=future_peds,
            mask=mask,
            steps=min(base_steps, max_steps),
        )
        near_field = float(self.config.predictive_near_field_distance)
        if min_pred_dist <= near_field:
            boosted = base_steps + max(0, int(self.config.predictive_horizon_boost_steps))
            return min(boosted, max_steps)
        return min(base_steps, max_steps)

    def _risk_speed_cap_ratio(self, *, future_peds: np.ndarray, mask: np.ndarray) -> float:
        """Compute a risk-aware cap on speed ratio based on near predicted pedestrians.

        Returns:
            float: Speed cap ratio in ``[0.1, 1.0]``.
        """
        near_field = float(self.config.predictive_near_field_distance)
        if near_field <= 0.0:
            return 1.0
        min_pred_dist = self._min_predicted_distance(
            future_peds=future_peds,
            mask=mask,
            steps=max(2, min(int(self.config.predictive_horizon_steps), future_peds.shape[1])),
        )
        if not np.isfinite(min_pred_dist):
            return 1.0
        cap = float(self.config.predictive_near_field_speed_cap)
        cap = float(np.clip(cap, 0.1, 1.0))
        if min_pred_dist <= near_field:
            return cap
        if min_pred_dist <= near_field * 1.5:
            return min(1.0, cap + 0.15)
        return 1.0

    def _candidate_set(
        self, *, future_peds: np.ndarray, mask: np.ndarray
    ) -> list[tuple[float, float]]:
        """Build a risk-adaptive candidate command lattice.

        Returns:
            list[tuple[float, float]]: Candidate ``(v, omega)`` commands.
        """
        cap_ratio = self._risk_speed_cap_ratio(future_peds=future_peds, mask=mask)
        base_speed_ratios = [float(v) for v in self.config.predictive_candidate_speeds]
        heading_deltas = [float(v) for v in self.config.predictive_candidate_heading_deltas]
        near_field = float(self.config.predictive_near_field_distance)
        min_pred_dist = self._min_predicted_distance(
            future_peds=future_peds,
            mask=mask,
            steps=max(2, min(int(self.config.predictive_horizon_steps), future_peds.shape[1])),
        )
        if np.isfinite(min_pred_dist) and min_pred_dist <= near_field:
            base_speed_ratios.extend(
                float(v) for v in self.config.predictive_near_field_speed_samples
            )
            heading_deltas.extend(
                float(v) for v in self.config.predictive_near_field_heading_deltas
            )

        speed_ratios = sorted({float(np.clip(v, 0.0, 1.0)) for v in base_speed_ratios})
        heading_deltas = sorted(set(heading_deltas))
        dt = max(float(self.config.predictive_rollout_dt), self._EPS)
        max_v = float(self.config.max_linear_speed) * float(np.clip(cap_ratio, 0.1, 1.0))
        candidates: list[tuple[float, float]] = []
        for ratio in speed_ratios:
            v = float(np.clip(ratio * self.config.max_linear_speed, 0.0, max_v))
            for delta in heading_deltas:
                omega = float(
                    np.clip(
                        delta / dt,
                        -self.config.max_angular_speed,
                        self.config.max_angular_speed,
                    )
                )
                candidates.append((v, omega))
        candidates.append((0.0, 0.0))
        # Keep deterministic ordering for stable benchmark outputs.
        return sorted(set(candidates), key=lambda x: (round(x[0], 6), round(x[1], 6)))

    @staticmethod
    def _rollout_robot(
        *,
        v: float,
        w: float,
        dt: float,
        steps: int,
    ) -> np.ndarray:
        """Roll out robot trajectory in its local frame under unicycle dynamics.

        Returns:
            np.ndarray: Trajectory ``(steps, 2)`` in local robot frame.
        """
        pos = np.zeros(2, dtype=float)
        heading = 0.0
        traj = np.zeros((steps, 2), dtype=float)
        for i in range(steps):
            heading += float(w) * dt
            pos[0] += float(v) * np.cos(heading) * dt
            pos[1] += float(v) * np.sin(heading) * dt
            traj[i] = pos
        return traj

    def _goal_progress(
        self,
        robot_state: dict,
        goal_state: dict,
        v: float,
        w: float,
        *,
        steps: int | None = None,
    ) -> float:
        """Compute progress toward the goal over the rollout horizon.

        Returns:
            float: Positive value when a candidate reduces distance to goal.
        """
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        initial_dist = float(np.linalg.norm(goal - robot_pos))
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        steps = max(1, int(steps if steps is not None else self.config.predictive_horizon_steps))
        local_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps)

        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        x_world = cos_h * local_traj[-1, 0] - sin_h * local_traj[-1, 1]
        y_world = sin_h * local_traj[-1, 0] + cos_h * local_traj[-1, 1]
        final_world = robot_pos + np.array([x_world, y_world], dtype=float)
        final_dist = float(np.linalg.norm(goal - final_world))
        return initial_dist - final_dist

    def _collision_cost(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int | None = None,
    ) -> tuple[float, float]:
        """Compute collision and near-miss penalties for a candidate action.

        Returns:
            tuple[float, float]: ``(collision_penalty, near_miss_penalty)``.
        """
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        steps = max(1, int(steps if steps is not None else self.config.predictive_horizon_steps))
        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps)
        radius_margin = float(self.config.predictive_robot_radius) + float(
            self.config.predictive_pedestrian_radius
        )
        speed_margin = float(self.config.predictive_speed_clearance_gain) * abs(float(v))
        safe_dist = float(self.config.predictive_safe_distance) + radius_margin + speed_margin
        near_dist = max(
            float(self.config.predictive_near_distance) + radius_margin + speed_margin, safe_dist
        )
        collisions = 0.0
        near_misses = 0.0
        for t in range(min(steps, future_peds.shape[1])):
            delta = future_peds[:, t, :] - robot_traj[t].reshape(1, 2)
            dist = np.linalg.norm(delta, axis=1)
            valid_dist = dist[mask > 0.5]
            if valid_dist.size == 0:
                continue
            collisions += float(np.sum(np.maximum(0.0, safe_dist - valid_dist)))
            near_misses += float(np.sum(np.maximum(0.0, near_dist - valid_dist)))
        return collisions, near_misses

    def _min_clearance(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int,
    ) -> float:
        """Compute minimum predicted robot-pedestrian clearance for a candidate.

        Returns:
            float: Minimum center-to-center clearance in meters.
        """
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=max(1, int(steps)))
        valid_idx = np.where(mask > 0.5)[0]
        if valid_idx.size == 0:
            return float("inf")
        ped = future_peds[valid_idx, : robot_traj.shape[0], :]
        if ped.size == 0:
            return float("inf")
        delta = ped - robot_traj.reshape(1, robot_traj.shape[0], 2)
        dist = np.linalg.norm(delta, axis=2)
        return float(np.min(dist)) if dist.size > 0 else float("inf")

    def _ttc_penalty(
        self,
        *,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int | None = None,
    ) -> float:
        """Compute a TTC-style penalty for near-term close approaches.

        Returns:
            float: Penalty that increases for earlier/closer predicted encounters.
        """
        radius_margin = float(self.config.predictive_robot_radius) + float(
            self.config.predictive_pedestrian_radius
        )
        speed_margin = float(self.config.predictive_speed_clearance_gain) * abs(float(v))
        threshold = float(self.config.predictive_ttc_distance) + radius_margin + speed_margin
        if threshold <= 0.0:
            return 0.0
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        steps = max(1, int(steps if steps is not None else self.config.predictive_horizon_steps))
        robot_traj = self._rollout_robot(v=v, w=w, dt=dt, steps=steps)
        penalty = 0.0
        for t in range(min(steps, future_peds.shape[1])):
            delta = future_peds[:, t, :] - robot_traj[t].reshape(1, 2)
            dist = np.linalg.norm(delta, axis=1)
            valid_dist = dist[mask > 0.5]
            if valid_dist.size == 0:
                continue
            shortfall = np.maximum(0.0, threshold - valid_dist)
            time_weight = 1.0 / (float(t + 1) * dt + self._EPS)
            penalty += float(np.sum(shortfall * time_weight))
        return penalty

    def _score_action(
        self,
        *,
        observation: dict,
        future_peds: np.ndarray,
        mask: np.ndarray,
        v: float,
        w: float,
        steps: int,
    ) -> float:
        """Score candidate action by combining goal, safety, smoothness, and occupancy costs.

        Returns:
            float: Scalar cost (lower is better).
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        goal_progress = self._goal_progress(robot_state, goal_state, v, w, steps=steps)
        collision_pen, near_pen = self._collision_cost(
            future_peds=future_peds,
            mask=mask,
            v=v,
            w=w,
            steps=steps,
        )
        ttc_pen = self._ttc_penalty(
            future_peds=future_peds,
            mask=mask,
            v=v,
            w=w,
            steps=steps,
        )
        min_clearance = self._min_clearance(
            future_peds=future_peds,
            mask=mask,
            v=v,
            w=w,
            steps=steps,
        )
        progress_risk_shortfall = max(
            0.0, float(self.config.predictive_progress_risk_distance) - float(min_clearance)
        )
        progress_risk_pen = max(0.0, goal_progress) * progress_risk_shortfall
        hard_clearance_shortfall = max(
            0.0, float(self.config.predictive_hard_clearance_distance) - float(min_clearance)
        )

        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        dt = max(float(self.config.predictive_rollout_dt), 1e-3)
        candidate_heading = robot_heading + w * dt
        direction = np.array([np.cos(candidate_heading), np.sin(candidate_heading)], dtype=float)
        _, occ_penalty = self._path_penalty(
            robot_pos=robot_pos,
            direction=direction,
            observation=observation,
            base_distance=max(float(v) * float(self.config.predictive_horizon_steps) * dt, 1e-3),
            num_samples=max(2, int(self.config.predictive_horizon_steps)),
        )

        return (
            -float(self.config.predictive_goal_weight) * goal_progress
            + float(self.config.predictive_collision_weight) * collision_pen
            + float(self.config.predictive_near_miss_weight) * near_pen
            + float(self.config.predictive_progress_risk_weight) * progress_risk_pen
            + float(self.config.predictive_hard_clearance_weight) * hard_clearance_shortfall
            + float(self.config.predictive_velocity_weight) * abs(v)
            + float(self.config.predictive_turn_weight) * abs(w)
            + float(self.config.predictive_ttc_weight) * ttc_pen
            + float(self.config.occupancy_weight) * occ_penalty
        )

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) via predictive rollout search over learned trajectories.

        Returns:
            tuple[float, float]: Linear and angular command.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        state, mask, _robot_pos, _robot_heading = self._build_model_input(observation)
        future = self._predict_trajectories(state, mask)
        steps = self._effective_rollout_steps(future_peds=future, mask=mask)

        best = (0.0, 0.0)
        best_cost = float("inf")
        for v, w in self._candidate_set(future_peds=future, mask=mask):
            cost = self._score_action(
                observation=observation,
                future_peds=future,
                mask=mask,
                v=v,
                w=w,
                steps=steps,
            )
            if cost < best_cost:
                best_cost = cost
                best = (v, w)
        return best


def make_social_force_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for social-force-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SocialForcePlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SocialForcePlannerAdapter(config=config))


def make_orca_policy(
    config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False
) -> SocNavPlannerPolicy:
    """
    Convenience constructor for ORCA-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping ORCAPlannerAdapter.
    """

    return SocNavPlannerPolicy(
        adapter=ORCAPlannerAdapter(config=config, allow_fallback=allow_fallback)
    )


def make_sacadrl_policy(
    config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False
) -> SocNavPlannerPolicy:
    """
    Convenience constructor for GA3C-CADRL (SA-CADRL) planner policy.

    Set ``allow_fallback=True`` to use heuristic behavior when the checkpoint
    cannot be loaded (e.g., missing TensorFlow dependency).

    Returns:
        SocNavPlannerPolicy: Policy wrapping SACADRLPlannerAdapter.
    """

    return SocNavPlannerPolicy(
        adapter=SACADRLPlannerAdapter(config=config, allow_fallback=allow_fallback)
    )


def make_prediction_policy(
    config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False
) -> SocNavPlannerPolicy:
    """
    Convenience constructor for predictive planner policy.

    Set ``allow_fallback=True`` to permit constant-velocity fallback behavior
    when the predictive model checkpoint cannot be loaded.

    Returns:
        SocNavPlannerPolicy: Policy wrapping PredictionPlannerAdapter.
    """
    return SocNavPlannerPolicy(
        adapter=PredictionPlannerAdapter(config=config, allow_fallback=allow_fallback)
    )


class SocNavBenchSamplingAdapter(SamplingPlannerAdapter):
    """
    Adapter that attempts to delegate to the upstream SocNavBench SamplingPlanner.

    Warning:
        This adapter requires the upstream SocNavBench planner by default. Set
        ``allow_fallback=True`` to fall back to the heuristic SamplingPlannerAdapter;
        in fallback mode it is **not benchmark-ready**.
    """

    def __init__(
        self,
        config: SocNavPlannerConfig | None = None,
        socnav_root: Path | None = None,
        planner_factory: Callable[[], Any] | None = None,
        *,
        allow_fallback: bool = False,
    ) -> None:
        """Initialize the adapter with upstream delegation enabled."""

        super().__init__(
            config=config,
            socnav_root=socnav_root,
            planner_factory=planner_factory,
            use_upstream=True,
            allow_fallback=allow_fallback,
        )
