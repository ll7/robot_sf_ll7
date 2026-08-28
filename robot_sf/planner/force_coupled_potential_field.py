"""Opt-in force-coupled dynamic potential-field local planner (issue #7889).

Implements one experimental local-planner comparator: a documented
force-coupled dynamic-potential-field core with a bounded target-following
step, exposed through the canonical :class:`LocalPlannerProtocol`.  It is
opt-in only and does not change any default planner or release roster.

The implementation follows the published method description of Jing et al.,
"Local path planning for autonomous vehicles: a dynamic potential field-guided
and force-coupled adaptive pure pursuit approach" (Scientific Reports 2026)
for the force-coupling and potential-field elements; it is not a faithful
reproduction and makes no benchmark claim.  See the method card in
``docs/context/issue_7889_force_coupled_potential_field.md`` for the exact
source-to-implementation map.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.nav.occupancy_grid_utils import ego_to_world
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin

PLANNER_TYPE = "force_coupled_potential_field"


@dataclass(frozen=True)
class ForceCoupledPotentialFieldConfig:
    """Immutable experimental configuration for the force-coupled planner.

    Attractive and repulsive weights, influence cutoff, force saturation,
    look-ahead bounds, speed/rate limits, timestep, and numerical guards are
    all declared here; invalid values are rejected in ``__post_init__``.
    """

    attractive_weight: float = 1.0
    repulsive_weight: float = 2.0
    influence_radius_m: float = 3.0
    force_saturation: float = 5.0
    look_ahead_min_m: float = 0.5
    look_ahead_max_m: float = 2.0
    look_ahead_gain: float = 0.8
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    max_linear_rate: float = 0.8
    max_angular_rate: float = 1.5
    control_dt: float = 0.2
    numerical_epsilon: float = 1e-6
    obstacle_grid_threshold: float = 0.5
    obstacle_grid_max_points: int = 256
    obstacle_input_mode: str = "observation_contract"
    pedestrian_input_mode: str = "observation_contract"

    def __post_init__(self) -> None:
        """Reject invalid configuration values before planning begins."""
        positives = {
            "attractive_weight": self.attractive_weight,
            "repulsive_weight": self.repulsive_weight,
            "influence_radius_m": self.influence_radius_m,
            "force_saturation": self.force_saturation,
            "look_ahead_min_m": self.look_ahead_min_m,
            "look_ahead_max_m": self.look_ahead_max_m,
            "look_ahead_gain": self.look_ahead_gain,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "max_linear_rate": self.max_linear_rate,
            "max_angular_rate": self.max_angular_rate,
            "control_dt": self.control_dt,
            "numerical_epsilon": self.numerical_epsilon,
        }
        for name, value in positives.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.look_ahead_min_m > self.look_ahead_max_m:
            raise ValueError("look_ahead_min_m must not exceed look_ahead_max_m")
        if not math.isfinite(float(self.obstacle_grid_threshold)) or not (
            0.0 <= float(self.obstacle_grid_threshold) <= 1.0
        ):
            raise ValueError("obstacle_grid_threshold must be a finite number in [0, 1]")
        if (
            isinstance(self.obstacle_grid_max_points, bool)
            or not isinstance(self.obstacle_grid_max_points, int)
            or self.obstacle_grid_max_points <= 0
        ):
            raise ValueError("obstacle_grid_max_points must be a positive integer")
        if self.obstacle_input_mode not in {"observation_contract", "oracle"}:
            raise ValueError("obstacle_input_mode must be observation_contract or oracle")
        if self.pedestrian_input_mode not in {"observation_contract", "oracle"}:
            raise ValueError("pedestrian_input_mode must be observation_contract or oracle")

    def digest(self) -> str:
        """Return a stable configuration digest for diagnostics.

        Returns:
            A 64-character lowercase SHA-256 hex digest of the config.
        """
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_force_coupled_potential_field_config(
    cfg: dict[str, Any] | None,
) -> ForceCoupledPotentialFieldConfig:
    """Build the immutable planner config from an algorithm mapping.

    Runner-only keys such as ``allow_testing_algorithms`` and ``planner_variant``
    are intentionally ignored so the durable YAML can be passed directly.

    Returns:
        The validated force-coupled planner configuration.
    """
    payload = cfg if isinstance(cfg, dict) else {}
    allowed = {field.name for field in fields(ForceCoupledPotentialFieldConfig)}
    return ForceCoupledPotentialFieldConfig(
        **{key: value for key, value in payload.items() if key in allowed}
    )


def _as_2d_points(value: Any) -> np.ndarray:
    """Coerce a raw position payload into an ``(N, 2)`` finite float array.

    Returns:
        The ``(N, 2)`` float array.

    Raises:
        ValueError: When the payload cannot reshape or contains non-finite values.
    """
    arr = np.asarray(value, dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite position payload")
    return arr


class ForceCoupledPotentialFieldPlanner(OccupancyAwarePlannerMixin):
    """Opt-in force-coupled dynamic potential-field local planner.

    Implements the canonical :class:`LocalPlannerProtocol`:
    ``plan`` / ``reset(*, seed=...)`` / ``diagnostics`` / ``close``.
    """

    def __init__(
        self,
        config: ForceCoupledPotentialFieldConfig | None = None,
        *,
        planner_type: str = PLANNER_TYPE,
    ) -> None:
        """Initialize the planner with an immutable configuration."""
        self.config = config or ForceCoupledPotentialFieldConfig()
        self.planner_type = planner_type
        self._last_linear: float = 0.0
        self._last_angular: float = 0.0
        self._last_diagnostics: dict[str, Any] = {}
        self._degradation_reasons: list[str] = []
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> None:
        """Reset deterministic planner state while accepting the canonical seed."""
        if seed is not None:
            int(seed)
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._last_diagnostics = {}
        self._degradation_reasons = []

    def close(self) -> None:
        """Release held resources. Idempotent."""
        self._closed = True

    # -- planning ----------------------------------------------------------

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the ``(linear_speed, angular_rate)`` command tuple.

        Args:
            observation: Structured planner observation payload carrying the
                robot pose, goal, and (under the observation contract) obstacle
                and pedestrian positions.

        Returns:
            The constrained ``(linear, angular)`` command.

        Raises:
            ValueError: When required inputs are missing, invalid, or
                non-finite (fail closed; never a silent nominal success).
        """
        if self._closed:
            self._record_failure(status="unavailable", reason="planner is closed")
            raise ValueError("planner is closed")
        try:
            robot, goal, obstacles, pedestrians, missing_inputs = self._observation_fields(
                observation
            )
            control_dt, control_dt_source = self._control_timestep(observation)
        except OverflowError as exc:
            reason = f"numeric conversion overflow: {exc}"
            self._record_failure(status="invalid_input", reason=reason)
            raise ValueError(reason) from exc
        except (IndexError, TypeError, ValueError) as exc:
            self._record_failure(status="invalid_input", reason=str(exc))
            raise ValueError(str(exc)) from exc

        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                target = self._select_target(robot, goal)
                attractive = self._attractive_force(robot, target)
                obstacle_repulsive, obstacle_zero_distance = self._repulsive_force(robot, obstacles)
                pedestrian_repulsive, pedestrian_zero_distance = self._repulsive_force(
                    robot, pedestrians
                )
                repulsive = obstacle_repulsive + pedestrian_repulsive
                total_force = attractive + obstacle_repulsive + pedestrian_repulsive
                saturated = self._saturate(total_force)
                if not all(
                    np.all(np.isfinite(force))
                    for force in (attractive, repulsive, total_force, saturated)
                ):
                    raise ValueError("non-finite force computation")

                distance = float(np.hypot(goal[0] - robot[0], goal[1] - robot[1]))
                force_norm = float(np.hypot(saturated[0], saturated[1]))
                goal_reached = distance <= self.config.numerical_epsilon
                force_cancellation_guard = (
                    force_norm <= self.config.numerical_epsilon and not goal_reached
                )
                if goal_reached or force_cancellation_guard:
                    # atan2(0, 0) would silently invent the world-frame +x direction.
                    # Stop at the goal, and fail closed at a potential-field local minimum.
                    raw_linear = 0.0
                    raw_angular = 0.0
                else:
                    desired_heading = math.atan2(saturated[1], saturated[0])
                    heading_error = wrap_angle_pi(desired_heading - robot[2])
                    raw_linear = float(self.config.look_ahead_gain * distance)
                    raw_angular = float(heading_error)

                zero_distance_stop = bool(obstacle_zero_distance or pedestrian_zero_distance)
                if zero_distance_stop:
                    # Request a stop without violating the issue contract's hard
                    # command-rate predicates. Diagnostics retain the overlap trigger
                    # while the constrained command decelerates deterministically.
                    raw_linear = 0.0
                    raw_angular = 0.0
                linear, angular = self._constrain_command(
                    raw_linear, raw_angular, control_dt=control_dt
                )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            reason = f"non-finite force or command computation: {exc}"
            self._record_failure(status="invalid_input", reason=reason)
            raise ValueError(reason) from exc
        self._last_diagnostics = self._build_diagnostics(
            state=(robot, goal, target),
            forces={
                "attractive": attractive,
                "obstacle_repulsive": obstacle_repulsive,
                "pedestrian_repulsive": pedestrian_repulsive,
                "repulsive": repulsive,
                "total": total_force,
                "saturated": saturated,
            },
            raw_command=(raw_linear, raw_angular),
            command=(linear, angular),
            previous_command=(self._last_linear, self._last_angular),
            visibility=(
                missing_inputs,
                {
                    "obstacles": obstacle_zero_distance,
                    "pedestrians": pedestrian_zero_distance,
                },
            ),
            stop_state=(goal_reached, force_cancellation_guard),
            control_timestep=(control_dt, control_dt_source),
        )
        self._last_linear = linear
        self._last_angular = angular
        return (linear, angular)

    def diagnostics(self) -> dict[str, Any]:
        """Return the last planning-step diagnostics.

        Returns:
            The diagnostics payload, carrying ``planner_type`` at minimum.
        """
        if not self._last_diagnostics:
            return {"planner_type": self.planner_type}
        return dict(self._last_diagnostics)

    # -- internals ---------------------------------------------------------

    def _observation_fields(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Extract and validate robot, goal, obstacle, and pedestrian fields.

        Returns:
            The ``(robot, goal, obstacles, pedestrians, missing_inputs)`` tuple.

        Raises:
            ValueError: When required fields are missing or non-finite.
        """
        if not isinstance(observation, dict):
            raise ValueError("observation must be a mapping")
        robot, goal = self._robot_and_goal(observation)
        obstacles_raw, pedestrians_raw, missing_inputs = self._visibility_inputs(observation)
        if self.config.obstacle_input_mode == "observation_contract":
            grid_obstacles = (
                self._obstacles_from_occupancy_grid(observation, robot)
                if "obstacles" in missing_inputs
                else None
            )
            if grid_obstacles is None:
                obstacles = self._as_obstacles(obstacles_raw)
            else:
                obstacles = grid_obstacles
                missing_inputs.remove("obstacles")
        else:
            obstacles = _as_2d_points(obstacles_raw)
        if self.config.pedestrian_input_mode == "observation_contract":
            pedestrians = self._as_pedestrians(pedestrians_raw)
        else:
            pedestrians = _as_2d_points(pedestrians_raw)
        return robot, goal, obstacles, pedestrians, missing_inputs

    def _control_timestep(self, observation: dict[str, Any]) -> tuple[float, str]:
        """Resolve one positive control timestep from runtime observation metadata.

        Returns:
            The effective timestep and its provenance label.
        """
        sim = observation.get("sim")
        if isinstance(sim, dict) and "timestep" in sim:
            raw_dt = sim["timestep"]
            source = "observation.sim.timestep"
        elif "sim_timestep" in observation:
            raw_dt = observation["sim_timestep"]
            source = "observation.sim_timestep"
        elif "dt" in observation:
            raw_dt = observation["dt"]
            source = "observation.dt"
        else:
            return float(self.config.control_dt), "config.control_dt"
        values = np.asarray(raw_dt, dtype=float).reshape(-1)
        if values.size != 1 or not math.isfinite(float(values[0])) or float(values[0]) <= 0.0:
            raise ValueError("control timestep must contain exactly one positive finite value")
        return float(values[0]), source

    @staticmethod
    def _robot_and_goal(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Extract finite robot pose and goal vectors from nested or flat state.

        Returns:
            The validated ``(robot_pose, goal_position)`` pair.
        """
        robot_raw = observation.get("robot")
        goal_raw = observation.get("goal")
        if isinstance(robot_raw, dict):
            position = robot_raw.get("position")
            heading = robot_raw.get("heading")
            if position is not None and heading is not None:
                robot_raw = [*np.asarray(position, dtype=float).reshape(-1)[:2], _scalar(heading)]
        if isinstance(goal_raw, dict):
            goal_raw = goal_raw.get("current")
        if robot_raw is None and "robot_position" in observation:
            robot_raw = [
                *np.asarray(observation.get("robot_position"), dtype=float).reshape(-1)[:2],
                _scalar(observation.get("robot_heading"), field="robot_heading"),
            ]
        if goal_raw is None:
            goal_raw = observation.get("goal_current")
        if robot_raw is None or goal_raw is None:
            raise ValueError("observation requires robot and goal fields")
        robot = np.asarray(robot_raw, dtype=float).reshape(-1)
        goal = np.asarray(goal_raw, dtype=float).reshape(-1)
        if robot.shape[0] < 3 or goal.shape[0] < 2:
            raise ValueError("robot requires [x, y, theta]; goal requires [x, y]")
        if not np.all(np.isfinite(robot[:3])) or not np.all(np.isfinite(goal[:2])):
            raise ValueError("non-finite robot or goal payload")
        return robot[:3], goal[:2]

    @staticmethod
    def _visibility_inputs(observation: dict[str, Any]) -> tuple[Any, Any, list[str]]:
        """Resolve optional obstacle/pedestrian payloads and missing-input labels.

        Returns:
            The obstacle payload, pedestrian payload, and missing-input names.
        """
        missing_inputs: list[str] = []
        if "obstacles" in observation:
            obstacles_raw = observation["obstacles"]
        elif "obstacles_positions" in observation:
            obstacles_raw = {"positions": observation["obstacles_positions"]}
        else:
            obstacles_raw = []
            missing_inputs.append("obstacles")
        if "pedestrians" in observation:
            pedestrians_raw = observation["pedestrians"]
        elif "pedestrians_positions" in observation:
            pedestrians_raw = {
                "positions": observation["pedestrians_positions"],
                "count": observation.get("pedestrians_count"),
            }
        else:
            pedestrians_raw = []
            missing_inputs.append("pedestrians")
        return obstacles_raw, pedestrians_raw, missing_inputs

    @staticmethod
    def _as_obstacles(value: Any) -> np.ndarray:
        """Coerce an observation-contract obstacle payload to ``(N, 2)``.

        Returns:
            The ``(N, 2)`` float array.
        """
        if isinstance(value, dict):
            if "positions" not in value:
                raise ValueError("obstacles mapping requires positions")
            value = value["positions"]
        return _as_2d_points(value)

    @staticmethod
    def _as_pedestrians(value: Any) -> np.ndarray:
        """Coerce an observation-contract pedestrian payload to ``(N, 2)``.

        Returns:
            The active ``(N, 2)`` float array.
        """
        if isinstance(value, dict):
            if "positions" not in value:
                raise ValueError("pedestrians mapping requires positions")
            positions = _as_2d_points(value["positions"])
            count_raw = value.get("count")
            if count_raw is None:
                count = len(positions)
            else:
                count_values = np.asarray(count_raw, dtype=float).reshape(-1)
                if count_values.size != 1:
                    raise ValueError("pedestrian count must contain exactly one value")
                count_value = float(count_values[0])
                if not math.isfinite(count_value) or not count_value.is_integer():
                    raise ValueError("pedestrian count must be a finite integer")
                count = int(count_value)
            if not 0 <= count <= len(positions):
                raise ValueError("pedestrian count must be between zero and positions length")
            return positions[:count]
        return _as_2d_points(value)

    def _obstacles_from_occupancy_grid(
        self,
        observation: dict[str, Any],
        robot: np.ndarray,
    ) -> np.ndarray | None:
        """Return nearby world-frame static-obstacle cell centers when a grid is present.

        Returns:
            Nearby obstacle centers, an empty array for a valid obstacle-free grid, or
            ``None`` when the observation has no valid static-obstacle grid contract.
        """
        payload = self._validated_obstacle_grid_payload(observation)
        if payload is None:
            return None
        grid, meta, channel_idx, resolution = payload
        threshold = float(self.config.obstacle_grid_threshold)
        obstacle_mask = grid[channel_idx] >= threshold
        robot_grid_indices = self._world_to_grid(robot[:2], meta, (grid.shape[1], grid.shape[2]))
        if robot_grid_indices is None:
            raise ValueError("robot pose lies outside supplied occupancy grid")
        robot_row, robot_col = robot_grid_indices
        if bool(grid[channel_idx, robot_row, robot_col] >= threshold):
            raise ValueError(
                "robot pose maps to an occupied static-obstacle cell; "
                "exact overlap geometry is unavailable"
            )
        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return np.zeros((0, 2), dtype=float)

        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        if not np.all(np.isfinite(origin[:2])):
            raise ValueError("occupancy grid origin must be finite")
        centers = np.column_stack(
            (
                origin[0] + (indices[:, 1] + 0.5) * resolution,
                origin[1] + (indices[:, 0] + 0.5) * resolution,
            )
        )
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        if use_ego:
            pose = self._as_1d_float(
                meta.get("robot_pose", [robot[0], robot[1], robot[2]]),
                pad=3,
            )
            if not np.all(np.isfinite(pose[:3])):
                raise ValueError("occupancy grid robot_pose must be finite")
            robot_pose = ((float(pose[0]), float(pose[1])), float(pose[2]))
            centers = np.asarray(
                [ego_to_world(float(x), float(y), robot_pose) for x, y in centers],
                dtype=float,
            )

        offsets = centers - robot[:2]
        distance_sq = np.einsum("ij,ij->i", offsets, offsets)
        within_influence = distance_sq <= self.config.influence_radius_m**2
        centers = centers[within_influence]
        distance_sq = distance_sq[within_influence]
        if centers.shape[0] > self.config.obstacle_grid_max_points:
            order = np.argsort(distance_sq, kind="stable")[: self.config.obstacle_grid_max_points]
            centers = centers[order]
        return centers

    def _validated_obstacle_grid_payload(  # noqa: C901
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any], int, float] | None:
        """Validate the canonical grid contract without treating combined occupancy as static.

        Returns:
            Validated grid, metadata, explicit obstacle-channel index, and resolution, or
            ``None`` when no explicit obstacle channel is available.

        Raises:
            ValueError: When a supplied obstacle-grid payload is malformed.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            if observation.get("occupancy_grid") is not None:
                raise ValueError(
                    "supplied occupancy grid must be a three-dimensional tensor with metadata"
                )
            return None
        grid, raw_meta = payload
        if not isinstance(raw_meta, dict):
            raise ValueError("occupancy grid metadata must be a mapping")
        if grid.ndim != 3 or any(dimension <= 0 for dimension in grid.shape):
            raise ValueError("occupancy grid must have a non-empty [channels, height, width] shape")
        if not np.all(np.isfinite(grid)) or np.any(grid < 0.0) or np.any(grid > 1.0):
            raise ValueError("occupancy grid values must be finite and within [0, 1]")

        self._required_grid_metadata(raw_meta, "origin", 2)
        resolution = float(self._required_grid_metadata(raw_meta, "resolution", 1)[0])
        size = self._required_grid_metadata(raw_meta, "size", 2)
        use_ego_frame = self._required_grid_metadata(raw_meta, "use_ego_frame", 1)[0]
        center_on_robot = self._required_grid_metadata(raw_meta, "center_on_robot", 1)[0]
        channel_indices = self._required_grid_metadata(
            raw_meta, "channel_indices", len(self._CHANNEL_KEYS)
        )
        self._required_grid_metadata(raw_meta, "robot_pose", 3)
        if resolution <= 0.0:
            raise ValueError("occupancy grid resolution must be positive")
        if np.any(size <= 0.0):
            raise ValueError("occupancy grid size must be positive")
        frame_flags = np.asarray([use_ego_frame, center_on_robot])
        if not np.all(np.isin(frame_flags, [0.0, 1.0])):
            raise ValueError("occupancy grid frame flags must be boolean values")
        grid_extent = np.asarray([grid.shape[2] * resolution, grid.shape[1] * resolution])
        extent_tolerance = max(1e-6, resolution * 1e-5)
        if np.any(size > grid_extent + extent_tolerance) or np.any(
            grid_extent - size >= resolution - extent_tolerance
        ):
            raise ValueError("occupancy grid size does not match shape and resolution")

        if not np.all(np.equal(channel_indices, np.floor(channel_indices))):
            raise ValueError("occupancy grid channel indices must be integers")
        channel_indices_int = channel_indices.astype(int)
        present_indices = channel_indices_int[channel_indices_int >= 0]
        if np.any(channel_indices_int < -1) or np.any(channel_indices_int >= grid.shape[0]):
            raise ValueError("occupancy grid channel indices are out of range")
        if len(np.unique(present_indices)) != len(present_indices):
            raise ValueError("occupancy grid channel indices must be unique")
        channel_idx = int(channel_indices_int[0])
        if channel_idx < 0:
            return None
        return grid, raw_meta, channel_idx, resolution

    @staticmethod
    def _required_grid_metadata(meta: dict[str, Any], key: str, expected_size: int) -> np.ndarray:
        """Read one exact-size, finite numeric occupancy-grid metadata field.

        Returns:
            The validated metadata values as a one-dimensional float array.
        """
        if key not in meta:
            raise ValueError(f"occupancy grid metadata missing {key}")
        try:
            values = np.asarray(meta[key], dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"occupancy grid metadata {key} must be numeric") from exc
        if values.size != expected_size or not np.all(np.isfinite(values)):
            raise ValueError(
                f"occupancy grid metadata {key} must contain exactly {expected_size} finite values"
            )
        return values

    def _select_target(self, robot: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Select the look-ahead target along the robot-to-goal direction.

        Returns:
            The ``(x, y)`` target point.
        """
        delta = goal[:2] - robot[:2]
        distance = float(np.hypot(delta[0], delta[1]))
        if distance <= self.config.numerical_epsilon:
            return goal[:2]
        look_ahead = min(
            self.config.look_ahead_max_m,
            max(self.config.look_ahead_min_m, self.config.look_ahead_gain * distance),
        )
        direction = delta / max(distance, self.config.numerical_epsilon)
        return robot[:2] + direction * look_ahead

    def _attractive_force(self, robot: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the attractive force toward the declared local target.

        Returns:
            The ``(fx, fy)`` attractive force.
        """
        delta = target - robot[:2]
        distance = float(np.hypot(delta[0], delta[1]))
        if distance <= self.config.numerical_epsilon:
            return np.zeros(2, dtype=float)
        return self.config.attractive_weight * delta / distance

    def _repulsive_force(self, robot: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, int]:
        """Return one source family's repulsive force and zero-distance guard count.

        Returns:
            The ``(force, zero_distance_guard_count)`` tuple.
        """
        total = np.zeros(2, dtype=float)
        zero_distance_guards = 0
        for point in points:
            delta = robot[:2] - point
            distance = float(np.hypot(delta[0], delta[1]))
            if distance <= self.config.numerical_epsilon:
                zero_distance_guards += 1
                continue
            if distance > self.config.influence_radius_m:
                continue
            magnitude = self.config.repulsive_weight * (
                1.0 / distance - 1.0 / self.config.influence_radius_m
            )
            if not math.isfinite(magnitude):
                raise ValueError("non-finite repulsive force magnitude")
            total = total + magnitude * delta / distance
        return total, zero_distance_guards

    def _saturate(self, force: np.ndarray) -> np.ndarray:
        """Saturate the combined force magnitude.

        Returns:
            The saturated ``(fx, fy)`` force.
        """
        norm = float(np.hypot(force[0], force[1]))
        if norm <= self.config.force_saturation:
            return force
        return force * (self.config.force_saturation / norm)

    def _constrain_command(
        self, raw_linear: float, raw_angular: float, *, control_dt: float
    ) -> tuple[float, float]:
        """Apply configured speed and command-rate limits as hard predicates.

        Returns:
            The constrained ``(linear, angular)`` command tuple.

        Raises:
            ValueError: When the constrained command is non-finite.
        """
        linear = float(
            np.clip(raw_linear, -self.config.max_linear_speed, self.config.max_linear_speed)
        )
        angular = float(
            np.clip(raw_angular, -self.config.max_angular_speed, self.config.max_angular_speed)
        )

        linear = float(
            np.clip(
                linear,
                self._last_linear - self.config.max_linear_rate * control_dt,
                self._last_linear + self.config.max_linear_rate * control_dt,
            )
        )
        angular = float(
            np.clip(
                angular,
                self._last_angular - self.config.max_angular_rate * control_dt,
                self._last_angular + self.config.max_angular_rate * control_dt,
            )
        )
        if not (math.isfinite(linear) and math.isfinite(angular)):
            raise ValueError("non-finite constrained command")
        return (linear, angular)

    def _build_diagnostics(  # noqa: C901
        self,
        *,
        state: tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: dict[str, np.ndarray],
        raw_command: tuple[float, float],
        command: tuple[float, float],
        previous_command: tuple[float, float],
        visibility: tuple[list[str], dict[str, int]],
        stop_state: tuple[bool, bool],
        control_timestep: tuple[float, str],
    ) -> dict[str, Any]:
        """Build the versioned diagnostics payload for one planning step.

        Returns:
            The diagnostics dict with force components, commands, and status.
        """
        robot, goal, target = state
        missing_inputs, zero_distance_guards = visibility
        raw_linear, raw_angular = raw_command
        linear, angular = command
        goal_reached, force_cancellation_guard = stop_state
        control_dt, control_dt_source = control_timestep
        current_degradation_reasons: list[str] = []
        if missing_inputs:
            current_degradation_reasons.append(
                "optional visibility inputs unavailable: " + ", ".join(missing_inputs)
            )
        overlapping_sources = [
            source for source, count in zero_distance_guards.items() if count > 0
        ]
        overlap_stop_requested = bool(overlapping_sources)
        overlap_stop_pending, overlap_stop_applied = self._overlap_stop_state(
            requested=overlap_stop_requested,
            command=(linear, angular),
        )
        emergency_stop = overlap_stop_applied
        if overlapping_sources:
            current_degradation_reasons.append(
                "zero-distance overlap stop: " + ", ".join(overlapping_sources)
            )
        if force_cancellation_guard:
            current_degradation_reasons.append("near-zero total force local-minimum stop")
        for reason in current_degradation_reasons:
            if reason not in self._degradation_reasons:
                self._degradation_reasons.append(reason)
        ever_degraded = bool(self._degradation_reasons)
        active_constraints = self._active_constraints(
            linear,
            angular,
            raw_linear,
            raw_angular,
            previous_command,
            control_dt=control_dt,
        )
        if overlapping_sources:
            active_constraints.append("zero_distance_stop_requested")
        if overlap_stop_pending:
            active_constraints.append("zero_distance_stop_pending")
        if overlap_stop_applied:
            active_constraints.append("zero_distance_stop")
            active_constraints.append("emergency_stop")
        if goal_reached:
            active_constraints.append("goal_reached_stop")
        if force_cancellation_guard:
            active_constraints.append("force_cancellation_stop")
        return {
            "diagnostics_schema": "force_coupled_potential_field.v1",
            "planner_type": self.planner_type,
            "config_digest": self.config.digest(),
            "robot": [float(robot[0]), float(robot[1]), float(robot[2])],
            "goal": [float(goal[0]), float(goal[1])],
            "selected_target": [float(target[0]), float(target[1])],
            "attractive_force": _force_list(forces["attractive"]),
            "obstacle_repulsive_force": _force_list(forces["obstacle_repulsive"]),
            "pedestrian_repulsive_force": _force_list(forces["pedestrian_repulsive"]),
            "repulsive_force": _force_list(forces["repulsive"]),
            "total_force": _force_list(forces["total"]),
            "saturated_force": _force_list(forces["saturated"]),
            "raw_command": [float(raw_linear), float(raw_angular)],
            "constrained_command": [float(linear), float(angular)],
            "active_constraints": active_constraints,
            "control_dt": control_dt,
            "control_dt_source": control_dt_source,
            "zero_distance_guards": zero_distance_guards,
            "goal_reached": goal_reached,
            "force_cancellation_guard": force_cancellation_guard,
            "emergency_stop": emergency_stop,
            "overlap_stop_requested": overlap_stop_requested,
            "overlap_stop_pending": overlap_stop_pending,
            "overlap_stop_applied": overlap_stop_applied,
            "missing_inputs": list(missing_inputs),
            "invalid_input": False,
            "non_finite_input": False,
            "fallback": False,
            "step_degraded": bool(current_degradation_reasons),
            "ever_degraded": ever_degraded,
            "degradation_reasons": list(self._degradation_reasons),
            "degraded": ever_degraded,
            "status": "degraded" if ever_degraded else "ok",
            "status_reason": (
                "; ".join(current_degradation_reasons)
                if current_degradation_reasons
                else (
                    "episode previously degraded: " + "; ".join(self._degradation_reasons)
                    if ever_degraded
                    else "nominal"
                )
            ),
        }

    def _overlap_stop_state(
        self, *, requested: bool, command: tuple[float, float]
    ) -> tuple[bool, bool]:
        """Return the pending/applied state for one overlap stop request.

        Returns:
            The ``(pending, applied)`` booleans.
        """
        if not requested:
            return False, False
        applied = all(abs(value) <= self.config.numerical_epsilon for value in command)
        return not applied, applied

    def _active_constraints(
        self,
        linear: float,
        angular: float,
        raw_linear: float,
        raw_angular: float,
        previous_command: tuple[float, float],
        *,
        control_dt: float,
    ) -> list[str]:
        """Report which hard constraints were active on the last step.

        Returns:
            A list of active constraint names; empty when none were active.
        """
        active: list[str] = []
        if abs(linear) >= self.config.max_linear_speed - self.config.numerical_epsilon:
            active.append("linear_speed_limit")
        if abs(angular) >= self.config.max_angular_speed - self.config.numerical_epsilon:
            active.append("angular_speed_limit")
        linear_rate = abs(linear - previous_command[0]) / max(control_dt, 1e-9)
        angular_rate = abs(angular - previous_command[1]) / max(control_dt, 1e-9)
        if linear_rate >= self.config.max_linear_rate - self.config.numerical_epsilon:
            active.append("linear_rate_limit")
        if angular_rate >= self.config.max_angular_rate - self.config.numerical_epsilon:
            active.append("angular_rate_limit")
        if not (math.isfinite(raw_linear) and math.isfinite(raw_angular)):
            active.append("non_finite_raw_command")
        return active

    def _record_failure(self, *, status: str, reason: str) -> None:
        """Record a stable fail-closed diagnostic before raising to the caller."""
        normalized_reason = reason.lower()
        self._last_diagnostics = {
            "diagnostics_schema": "force_coupled_potential_field.v1",
            "planner_type": self.planner_type,
            "config_digest": self.config.digest(),
            "status": status,
            "status_reason": reason,
            "missing_inputs": [],
            "invalid_input": status == "invalid_input",
            "non_finite_input": any(
                marker in normalized_reason for marker in ("non-finite", "overflow", "floating")
            ),
            "fallback": False,
            "degraded": status == "degraded",
            "step_degraded": status == "degraded",
            "ever_degraded": bool(self._degradation_reasons) or status == "degraded",
            "degradation_reasons": list(self._degradation_reasons),
        }


def _force_list(force: np.ndarray) -> list[float]:
    """Convert a 2-D force array to a JSON-ready float list.

    Returns:
        The ``[fx, fy]`` list.
    """
    return [float(force[0]), float(force[1])]


def _scalar(value: Any, *, field: str = "heading") -> float:
    """Return exactly one finite scalar from an observation payload."""
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1 or not math.isfinite(float(values[0])):
        raise ValueError(f"{field} must contain exactly one finite value")
    return float(values[0])
