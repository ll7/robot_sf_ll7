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

from robot_sf.nav.occupancy_grid import OBSERVATION_CHANNEL_ORDER
from robot_sf.nav.occupancy_grid_utils import world_to_ego

_SOCNAV_ROOT_ENV = "ROBOT_SF_SOCNAV_ROOT"
_SOCNAV_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "thrid_party" / "socnavbench"
_SOCNAV_REQUIRED_MODULES = (
    "control_pipelines.control_pipeline_v0",
    "objectives.goal_distance",
    "params.central_params",
    "planners.sampling_planner",
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
    orca_time_horizon: float = 6.0
    orca_time_horizon_obst: float = 3.0
    orca_obstacle_threshold: float = 0.5
    orca_obstacle_range: float = 6.0
    orca_obstacle_max_points: int = 80
    orca_obstacle_radius_scale: float = 1.0
    orca_heading_slowdown: float = 0.2
    social_force_repulsion_weight: float = 0.8
    occupancy_lookahead: float = 2.5
    occupancy_heading_sweep: float = pi * 2 / 3
    occupancy_candidates: int = 7
    occupancy_weight: float = 2.0
    occupancy_angle_weight: float = 0.3


class SamplingPlannerAdapter(OccupancyAwarePlannerMixin):
    """
    Minimal waypoint-to-velocity adapter inspired by the SocNavBench sampling planner.

    Warning:
        This adapter is a lightweight heuristic placeholder. It is **not** a
        benchmark-ready implementation of the SocNavBench sampling planner.
        Use it only as a simple baseline or fallback when upstream dependencies
        are unavailable.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None):
        """Initialize the adapter with optional planner configuration."""

        self.config = config or SocNavPlannerConfig()
        logger.warning("SamplingPlannerAdapter is a heuristic fallback and is not benchmark-ready.")

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
    """Heuristic social-force style planner: goal attraction plus pedestrian repulsion.

    Warning:
        This is a heuristic baseline and is not benchmark-ready.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None):
        """Initialize the social-force adapter with optional configuration."""
        super().__init__(config=config)
        logger.warning(
            "SocialForcePlannerAdapter is a heuristic baseline and is not benchmark-ready."
        )

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

    @dataclass
    class _OrcaLine:
        """ORCA half-plane constraint line."""

        point: np.ndarray
        direction: np.ndarray

    _EPS = 1e-6

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

    def _build_orca_lines(
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
        """
        Compute (v, w) using goal direction plus reciprocal-style avoidance.

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


class SACADRLPlannerAdapter(SamplingPlannerAdapter):
    """Heuristic SA-CADRL-style adapter using nearest pedestrians to bias heading.

    Warning:
        This is a heuristic baseline and is not a learned SA-CADRL model.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None):
        """Initialize the SA-CADRL adapter with optional configuration."""
        super().__init__(config=config)
        logger.warning(
            "SACADRLPlannerAdapter is a heuristic baseline and is not a learned SA-CADRL model."
        )

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

    Warning:
        This is a heuristic baseline and is not benchmark-ready.

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

    Warning:
        This is a heuristic baseline and is not a learned SA-CADRL model.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SACADRLPlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SACADRLPlannerAdapter(config=config))


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
    ):
        """Initialize the adapter and optionally load the upstream planner."""

        super().__init__(config=config)
        self._planner = None
        self._allow_fallback = bool(allow_fallback)
        # Allow passing an already-initialized planner for advanced use.
        if planner_factory is not None:
            self._planner = self._safe_call_factory(planner_factory)
        else:
            self._planner = self._load_upstream_planner(socnav_root)
        if self._planner is None and self._allow_fallback:
            logger.warning(
                "SocNavBenchSamplingAdapter is running in fallback heuristic mode and "
                "is not benchmark-ready."
            )
        if self._planner is None and not self._allow_fallback:
            raise RuntimeError(
                "SocNavBenchSamplingAdapter could not load the upstream planner. "
                "Set allow_fallback=True to use the heuristic fallback."
            )

    def _safe_call_factory(self, factory: Callable[[], Any]) -> Any | None:
        """
        Invoke a user-provided factory defensively.

        Returns:
            Planner instance from the factory or ``None`` on failure.
        """
        try:
            return factory()
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self._allow_fallback:
                logger.warning("SocNavBench planner factory failed: {}", exc)
                return None
            raise RuntimeError("SocNavBench planner factory failed.") from exc

    @staticmethod
    def _resolve_socnav_root(socnav_root: Path | None) -> Path:
        """
        Resolve the SocNavBench root directory.

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
    def _validate_socnav_root(root: Path) -> list[Path]:
        """
        Validate that the SocNavBench root contains expected modules.

        Returns:
            list[Path]: Missing module paths (empty when valid).
        """
        missing: list[Path] = []
        for module in _SOCNAV_REQUIRED_MODULES:
            module_path = Path(*module.split(".")).with_suffix(".py")
            candidate = root / module_path
            if not candidate.exists():
                missing.append(candidate)
        return missing

    def _load_upstream_planner(self, socnav_root: Path | None) -> Any | None:
        """
        Best-effort import of SocNavBench SamplingPlanner with defaults.

        Returns:
            SamplingPlanner | None: Upstream planner when dependencies resolve; otherwise ``None``.
        """
        root = self._resolve_socnav_root(socnav_root).resolve()
        if not root.exists():
            message = (
                "SocNavBench root not found at "
                f"'{root}'. Set {_SOCNAV_ROOT_ENV} or pass socnav_root."
            )
            if self._allow_fallback:
                logger.warning("{}", message)
                return None
            raise FileNotFoundError(message)

        missing = self._validate_socnav_root(root)
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            message = (
                "SocNavBench root is missing required modules: "
                f"{missing_str}. Ensure the SocNavBench repo is complete."
            )
            if self._allow_fallback:
                logger.warning("{}", message)
                return None
            raise FileNotFoundError(message)

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            import control_pipelines.control_pipeline_v0 as cp  # type: ignore  # noqa: PLC0415
            import objectives.goal_distance as gd  # type: ignore  # noqa: PLC0415
            import params.central_params as central  # type: ignore  # noqa: PLC0415
            import planners.sampling_planner as sp  # type: ignore  # noqa: PLC0415
            from dotmap import DotMap  # type: ignore  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency path
            message = f"Failed to import SocNavBench modules from '{root}': {exc}"
            if self._allow_fallback:
                logger.warning("{}", message)
                return None
            raise RuntimeError(message) from exc

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
        except Exception as exc:  # pragma: no cover - optional dependency path
            message = f"Failed to initialize SocNavBench SamplingPlanner: {exc}"
            if self._allow_fallback:
                logger.warning("{}", message)
                return None
            raise RuntimeError(message) from exc

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
