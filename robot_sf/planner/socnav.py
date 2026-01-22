"""
Lightweight adapters for running SocNavBench-style planners against the structured
observation mode.

These adapters are intentionally minimal to provide an in-process bridge while
full SocNavBench planners are integrated. They operate on the SocNav structured
observation emitted when `ObservationMode.SOCNAV_STRUCT` is enabled.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, pi
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.nav.occupancy_grid import OBSERVATION_CHANNEL_ORDER
from robot_sf.nav.occupancy_grid_utils import world_to_ego


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
    social_force_repulsion_weight: float = 0.8
    occupancy_lookahead: float = 2.5
    occupancy_heading_sweep: float = pi * 2 / 3
    occupancy_candidates: int = 7
    occupancy_weight: float = 2.0
    occupancy_angle_weight: float = 0.3


class SamplingPlannerAdapter(OccupancyAwarePlannerMixin):
    """
    Minimal waypoint-to-velocity adapter inspired by SocNavBench sampling planner.

    This is a placeholder that consumes structured SocNav observations and returns
    differential-drive (v, w) commands. It is designed so that the internals can be
    swapped for the real SocNavBench sampling planner without changing callers.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None):
        """Initialize the adapter with optional planner configuration."""

        self.config = config or SocNavPlannerConfig()

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute a (v, w) command from the structured observation.

        Args:
            observation: SocNav structured observation Dict (robot, goal, pedestrians, map, sim).

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

    Falls back to the lightweight adapter when upstream dependencies are missing.
    """

    def __init__(
        self,
        socnav_root: Path | None = None,
        adapter_config: SocNavPlannerConfig | None = None,
    ):
        """Initialize the policy, preferring the upstream SocNavBench planner when present."""

        adapter = SocNavBenchSamplingAdapter(config=adapter_config, socnav_root=socnav_root)
        super().__init__(adapter=adapter)


class SocialForcePlannerAdapter(SamplingPlannerAdapter):
    """Heuristic social-force style planner: goal attraction plus pedestrian repulsion."""

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using goal attraction and inverse-square pedestrian repulsion.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions = np.asarray(ped_state["positions"], dtype=float)
        ped_count = int(np.asarray(ped_state.get("count", [0]), dtype=float)[0])
        ped_positions = ped_positions[:ped_count]
        repulse = np.zeros(2, dtype=float)
        for ped in ped_positions:
            delta = robot_pos - ped
            dist = np.linalg.norm(delta) + 1e-6
            repulse += delta / dist**2

        combined = goal_vec + self.config.social_force_repulsion_weight * repulse
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


class ORCAPlannerAdapter(SamplingPlannerAdapter):
    """Simplified ORCA-inspired adapter using reciprocal-style avoidance."""

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using goal direction plus reciprocal-style avoidance.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions = np.asarray(ped_state["positions"], dtype=float)
        ped_count: int = int(np.asarray(ped_state.get("count", [0]), dtype=float)[0])
        ped_positions = ped_positions[:ped_count]

        avoidance = np.zeros(2, dtype=float)
        for ped in ped_positions:
            delta = ped - robot_pos
            dist = np.linalg.norm(delta) + 1e-6
            if dist < 5.0:
                avoidance -= delta / dist

        combined = goal_vec + self.config.orca_avoidance_weight * avoidance
        if np.linalg.norm(combined) > 1e-6:
            combined = combined / np.linalg.norm(combined)

        combined, occ_penalty = self._get_safe_heading(robot_pos, combined, observation)

        desired_heading = atan2(combined[1], combined[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                1.5 * self.config.angular_gain * heading_error,
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


class SACADRLPlannerAdapter(SamplingPlannerAdapter):
    """Heuristic SA-CADRL-style adapter using nearest pedestrians to bias heading."""

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute (v, w) using nearest pedestrian bias similar to SA-CADRL heuristics.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        to_goal = goal - robot_pos
        goal_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        ped_positions = np.asarray(ped_state["positions"], dtype=float)
        ped_count = int(np.asarray(ped_state.get("count", [0]), dtype=float)[0])
        ped_positions = ped_positions[:ped_count]
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


def make_social_force_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for social-force-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SocialForcePlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SocialForcePlannerAdapter(config=config))


def make_orca_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for ORCA-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping ORCAPlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=ORCAPlannerAdapter(config=config))


def make_sacadrl_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for SA-CADRL-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SACADRLPlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SACADRLPlannerAdapter(config=config))


class SocNavBenchSamplingAdapter(SamplingPlannerAdapter):
    """
    Adapter that attempts to delegate to the upstream SocNavBench SamplingPlanner.

    If upstream dependencies are unavailable, it falls back to the lightweight
    SamplingPlannerAdapter behavior.
    """

    def __init__(
        self,
        config: SocNavPlannerConfig | None = None,
        socnav_root: Path | None = None,
        planner_factory: Callable[[], Any] | None = None,
    ):
        """Initialize the adapter and optionally load the upstream planner."""

        super().__init__(config=config)
        self._planner = None
        # Allow passing an already-initialized planner for advanced use.
        if planner_factory is not None:
            self._planner = self._safe_call_factory(planner_factory)
        else:
            self._planner = self._load_upstream_planner(socnav_root)

    @staticmethod
    def _safe_call_factory(factory: Callable[[], Any]) -> Any | None:
        """
        Invoke a user-provided factory defensively.

        Returns:
            Planner instance from the factory or ``None`` on failure.
        """
        try:
            return factory()
        except Exception:  # pragma: no cover - defensive fallback  # noqa: BLE001
            return None

    def _load_upstream_planner(self, socnav_root: Path | None) -> Any | None:
        """
        Best-effort import of SocNavBench SamplingPlanner with defaults.

        Returns:
            SamplingPlanner | None: Upstream planner when dependencies resolve; otherwise ``None``.
        """
        root = socnav_root or Path(__file__).resolve().parents[2] / "output" / "SocNavBench"
        root_str = str(Path(root).resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            import control_pipelines.control_pipeline_v0 as cp  # type: ignore  # noqa: PLC0415
            import objectives.goal_distance as gd  # type: ignore  # noqa: PLC0415
            import params.central_params as central  # type: ignore  # noqa: PLC0415
            import planners.sampling_planner as sp  # type: ignore  # noqa: PLC0415
            from dotmap import DotMap  # type: ignore  # noqa: PLC0415
        except Exception:  # pragma: no cover - optional dependency path  # noqa: BLE001
            return None

        try:
            socnav_params = central.create_socnav_params()
            robot_params = central.create_robot_params()

            # Minimal parameter DotMap inspired by upstream defaults
            p = DotMap()
            p.planner = DotMap()
            p.control_pipeline_params = DotMap()
            p.control_pipeline_params.pipeline = cp.ControlPipelineV0
            p.control_pipeline_params.system_dynamics_params = DotMap(
                system="dubins", dt=robot_params.delta_theta
            )
            p.control_pipeline_params.planning_horizon = 1.0
            p.dt = (
                socnav_params.camera_params.dt if hasattr(socnav_params, "camera_params") else 0.1
            )
            p.planning_horizon = 1
            obj_fn = gd.GoalDistance(p=None)
            return sp.SamplingPlanner(obj_fn=obj_fn, params=p)
        except Exception:  # pragma: no cover - optional dependency path  # noqa: BLE001
            return None

    def plan(self, observation: dict) -> tuple[float, float]:
        """
        Compute a (v, w) command, preferring the upstream SocNavBench planner when available.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        if self._planner is None:
            return super().plan(observation)

        try:
            robot_state, goal_state, _ = self._socnav_fields(observation)
            pos = robot_state["position"]
            robot_pos = np.asarray(pos, dtype=float)
            heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
            start_config = self._planner.opt_waypt.__class__.from_pos3([pos[0], pos[1], heading])
            goal = goal_state["current"]
            goal_config = self._planner.opt_waypt.__class__.from_pos3([goal[0], goal[1], 0.0])
            data = self._planner.optimize(start_config=start_config, goal_config=goal_config)
            traj = data.get("trajectory")
            if traj is None:
                return super().plan(observation)
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
        except Exception:  # pragma: no cover - safety net  # noqa: BLE001
            return super().plan(observation)
