"""Shared social-RL graph observation adapter.

This module converts Robot SF's structured SocNav observation into a deterministic,
deployment-safe graph-style tensor bundle for CrowdNav-family and graph/social-RL
candidate assessments. It intentionally uses only current robot, pedestrian, goal, and
optional static-obstacle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PEDESTRIAN_FEATURE_NAMES: tuple[str, ...] = (
    "rel_x",
    "rel_y",
    "rel_vx",
    "rel_vy",
    "distance",
    "bearing",
    "radius",
)
"""Column names for `pedestrian_features` and `pedestrian_history`."""

ROBOT_FEATURE_NAMES: tuple[str, ...] = (
    "goal_rel_x",
    "goal_rel_y",
    "velocity_x",
    "velocity_y",
    "heading",
    "radius",
    "goal_distance",
)
"""Column names for `robot_features`."""

STATIC_OBSTACLE_FEATURE_NAMES: tuple[str, ...] = (
    "rel_mid_x",
    "rel_mid_y",
    "dir_x",
    "dir_y",
    "length",
    "distance",
    "radius",
)
"""Column names for `static_obstacle_features`."""

FORBIDDEN_DEPLOYMENT_FIELD_FRAGMENTS: tuple[str, ...] = (
    "future",
    "trajectory_label",
    "future_trajectory",
)
"""Input key fragments rejected by the deployment graph adapter."""


@dataclass(frozen=True, slots=True)
class SocialGraphObservationConfig:
    """Configuration for graph/social-RL observation construction."""

    max_pedestrians: int = 8
    history_steps: int = 1
    include_static_obstacles: bool = False
    max_static_obstacles: int = 0
    robot_frame: bool = True

    def __post_init__(self) -> None:
        """Validate adapter dimensions."""
        if self.max_pedestrians < 0:
            raise ValueError("max_pedestrians must be >= 0")
        if self.history_steps <= 0:
            raise ValueError("history_steps must be > 0")
        if self.max_static_obstacles < 0:
            raise ValueError("max_static_obstacles must be >= 0")
        if self.include_static_obstacles and self.max_static_obstacles == 0:
            raise ValueError(
                "max_static_obstacles must be > 0 when include_static_obstacles is enabled"
            )


@dataclass(frozen=True, slots=True)
class _SocNavFields:
    """Normalized fields extracted from nested or flat SocNav observations."""

    robot_position: np.ndarray
    robot_heading: float
    robot_velocity_xy: np.ndarray
    robot_radius: float
    goal_position: np.ndarray
    pedestrian_positions: np.ndarray
    pedestrian_velocities: np.ndarray
    pedestrian_radius: float


class SocialGraphObservationAdapter:
    """Stateful adapter that maintains a bounded pedestrian feature history."""

    def __init__(self, config: SocialGraphObservationConfig | None = None) -> None:
        """Create an adapter with an empty preallocated history buffer."""
        self.config = config or SocialGraphObservationConfig()
        H = self.config.history_steps
        M = self.config.max_pedestrians
        F = len(PEDESTRIAN_FEATURE_NAMES)
        self._buffer = np.zeros((H, M, F), dtype=np.float32)
        self._filled = 0

    def reset(self) -> None:
        """Clear cached pedestrian history between episodes."""
        self._filled = 0

    def build(
        self,
        observation: dict[str, Any],
        *,
        obstacle_segments: Any | None = None,
    ) -> dict[str, np.ndarray]:
        """Build the graph observation and update the history buffer.

        Returns:
            Dictionary of NumPy arrays representing current robot, pedestrian, static-obstacle,
            and edge features.
        """
        graph = build_social_graph_observation(
            observation,
            config=self.config,
            obstacle_segments=obstacle_segments,
        )
        current = graph["pedestrian_features"]
        if self._filled < self.config.history_steps:
            self._buffer[:] = current[np.newaxis, :, :]
            self._filled = self.config.history_steps
        else:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = current
        graph["pedestrian_history"] = self._buffer.copy()
        return graph


def build_social_graph_observation(
    observation: dict[str, Any],
    *,
    config: SocialGraphObservationConfig | None = None,
    obstacle_segments: Any | None = None,
) -> dict[str, np.ndarray]:
    """Build a deterministic graph-style observation from Robot SF SocNav fields.

    Returns:
        Dictionary of float and boolean arrays. Pedestrians are sorted by distance with stable
        geometric tie-breakers, truncated to `max_pedestrians`, and accompanied by an explicit mask.
    """
    cfg = config or SocialGraphObservationConfig()
    _reject_forbidden_deployment_fields(observation)
    fields = _extract_socnav_fields(observation)
    rotation = _robot_frame_rotation(fields.robot_heading) if cfg.robot_frame else np.eye(2)

    robot_features = _build_robot_features(fields, rotation=rotation, robot_frame=cfg.robot_frame)
    pedestrian_features, pedestrian_mask = _build_pedestrian_features(
        fields,
        config=cfg,
        rotation=rotation,
    )
    static_features, static_mask = _build_static_obstacle_features(
        obstacle_segments,
        fields=fields,
        config=cfg,
        rotation=rotation,
    )
    edge_index, edge_type = _build_star_edges(
        pedestrian_mask=pedestrian_mask,
        static_obstacle_mask=static_mask,
    )

    return {
        "robot_features": robot_features,
        "pedestrian_features": pedestrian_features,
        "pedestrian_mask": pedestrian_mask,
        "pedestrian_count": np.array([int(np.count_nonzero(pedestrian_mask))], dtype=np.int32),
        "pedestrian_history": np.repeat(
            pedestrian_features[np.newaxis, :, :],
            cfg.history_steps,
            axis=0,
        ).astype(np.float32),
        "static_obstacle_features": static_features,
        "static_obstacle_mask": static_mask,
        "static_obstacle_count": np.array([int(np.count_nonzero(static_mask))], dtype=np.int32),
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def _reject_forbidden_deployment_fields(value: Any, *, path: str = "observation") -> None:
    """Raise when an input field advertises future or label-only data."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).lower()
            if any(fragment in key_lower for fragment in FORBIDDEN_DEPLOYMENT_FIELD_FRAGMENTS):
                raise ValueError(f"Forbidden deployment observation field: {path}.{key}")
            _reject_forbidden_deployment_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _reject_forbidden_deployment_fields(child, path=f"{path}[{idx}]")


def _extract_socnav_fields(observation: dict[str, Any]) -> _SocNavFields:
    """Extract nested or flattened Robot SF SocNav fields.

    Returns:
        Normalized SocNav field bundle.
    """
    robot, goal, pedestrians, names = _resolve_layout(observation)
    robot_position = _require_array(
        robot.get(names["robot_position"][0]),
        size=2,
        field=names["robot_position"][1],
    )
    robot_heading = float(
        _require_array(
            robot.get(names["robot_heading"][0]),
            size=1,
            field=names["robot_heading"][1],
        )[0]
    )
    robot_velocity_xy = _require_array(
        robot.get(names["robot_velocity_xy"][0]),
        size=2,
        field=names["robot_velocity_xy"][1],
    )
    robot_radius = float(
        _require_array(
            robot.get(names["robot_radius"][0], [0.0]),
            size=1,
            field=names["robot_radius"][1],
        )[0]
    )
    goal_position = _require_array(
        goal.get(names["goal_current"][0]),
        size=2,
        field=names["goal_current"][1],
    )
    pedestrian_positions = _xy_rows(pedestrians.get(names["pedestrian_positions"][0]))
    pedestrian_velocities = _xy_rows(pedestrians.get(names["pedestrian_velocities"][0]))
    pedestrian_count = _pedestrian_count(
        pedestrians.get(names["pedestrian_count"][0]),
        fallback=pedestrian_positions.shape[0],
    )
    pedestrian_count = max(0, min(pedestrian_count, pedestrian_positions.shape[0]))
    pedestrian_positions = pedestrian_positions[:pedestrian_count]
    pedestrian_velocities = _pad_rows(pedestrian_velocities, rows=pedestrian_count)[
        :pedestrian_count
    ]
    pedestrian_radius = float(
        _require_array(
            pedestrians.get(names["pedestrian_radius"][0], [0.0]),
            size=1,
            field=names["pedestrian_radius"][1],
        )[0]
    )
    return _SocNavFields(
        robot_position=robot_position,
        robot_heading=robot_heading,
        robot_velocity_xy=robot_velocity_xy,
        robot_radius=robot_radius,
        goal_position=goal_position,
        pedestrian_positions=pedestrian_positions,
        pedestrian_velocities=pedestrian_velocities,
        pedestrian_radius=pedestrian_radius,
    )


def _resolve_layout(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, tuple[str, str]]]:
    """Return source dicts and field-name metadata for nested or flat observations."""
    robot = observation.get("robot", {})
    goal = observation.get("goal", {})
    pedestrians = observation.get("pedestrians", {})
    if robot or goal or pedestrians:
        return (
            robot,
            goal,
            pedestrians,
            {
                "robot_position": ("position", "robot.position"),
                "robot_heading": ("heading", "robot.heading"),
                "robot_speed": ("speed", "robot.speed"),
                "robot_velocity_xy": ("velocity_xy", "robot.velocity_xy"),
                "robot_radius": ("radius", "robot.radius"),
                "goal_current": ("current", "goal.current"),
                "pedestrian_positions": ("positions", "pedestrians.positions"),
                "pedestrian_velocities": ("velocities", "pedestrians.velocities"),
                "pedestrian_count": ("count", "pedestrians.count"),
                "pedestrian_radius": ("radius", "pedestrians.radius"),
            },
        )
    return (
        observation,
        observation,
        observation,
        {
            "robot_position": ("robot_position", "robot_position"),
            "robot_heading": ("robot_heading", "robot_heading"),
            "robot_speed": ("robot_speed", "robot_speed"),
            "robot_velocity_xy": ("robot_velocity_xy", "robot_velocity_xy"),
            "robot_radius": ("robot_radius", "robot_radius"),
            "goal_current": ("goal_current", "goal_current"),
            "pedestrian_positions": ("pedestrians_positions", "pedestrians_positions"),
            "pedestrian_velocities": ("pedestrians_velocities", "pedestrians_velocities"),
            "pedestrian_count": ("pedestrians_count", "pedestrians_count"),
            "pedestrian_radius": ("pedestrians_radius", "pedestrians_radius"),
        },
    )


def _require_array(value: Any, *, size: int, field: str) -> np.ndarray:
    """Return a required float array slice or raise a contract error."""
    arr = np.asarray([] if value is None else value, dtype=np.float32).reshape(-1)
    if arr.size < size:
        raise ValueError(f"Missing or malformed required field: {field}")
    return arr[:size]


def _xy_rows(value: Any) -> np.ndarray:
    """Normalize arbitrary XY values to an `(N, 2)` float array.

    Returns:
        Two-column float array.
    """
    arr = np.asarray([] if value is None else value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size < 2:
            return np.zeros((0, 2), dtype=np.float32)
        if arr.size % 2 != 0:
            return np.zeros((0, 2), dtype=np.float32)
        return arr.reshape(-1, 2)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32)


def _pedestrian_count(value: Any, *, fallback: int) -> int:
    """Return the declared pedestrian count, or a safe fallback."""
    arr = np.asarray([] if value is None else value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return int(fallback)
    return int(arr[0])


def _pad_rows(values: np.ndarray, *, rows: int) -> np.ndarray:
    """Pad a two-column array to at least `rows` rows.

    Returns:
        Original or padded two-column array.
    """
    if values.shape[0] >= rows:
        return values
    padded = np.zeros((rows, 2), dtype=np.float32)
    if values.size > 0:
        padded[: values.shape[0]] = values
    return padded


def _robot_frame_rotation(heading: float) -> np.ndarray:
    """Return the world-to-robot-frame rotation matrix."""
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    return np.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float32)


def _build_robot_features(
    fields: _SocNavFields,
    *,
    rotation: np.ndarray,
    robot_frame: bool,
) -> np.ndarray:
    """Build robot-level graph context features.

    Returns:
        One-dimensional robot context feature vector.
    """
    goal_rel = fields.goal_position - fields.robot_position
    velocity = fields.robot_velocity_xy
    if robot_frame:
        goal_rel = rotation @ goal_rel
        velocity = rotation @ velocity
    goal_distance = float(np.linalg.norm(goal_rel))
    return np.array(
        [
            float(goal_rel[0]),
            float(goal_rel[1]),
            float(velocity[0]),
            float(velocity[1]),
            float(fields.robot_heading),
            float(fields.robot_radius),
            goal_distance,
        ],
        dtype=np.float32,
    )


def _build_pedestrian_features(
    fields: _SocNavFields,
    *,
    config: SocialGraphObservationConfig,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded pedestrian feature rows and active mask."""
    features = np.zeros(
        (config.max_pedestrians, len(PEDESTRIAN_FEATURE_NAMES)),
        dtype=np.float32,
    )
    mask = np.zeros((config.max_pedestrians,), dtype=bool)
    if config.max_pedestrians == 0 or fields.pedestrian_positions.size == 0:
        return features, mask

    rel = fields.pedestrian_positions - fields.robot_position
    robot_velocity = fields.robot_velocity_xy
    if config.robot_frame:
        rel = rel @ rotation.T
        robot_velocity = robot_velocity @ rotation.T
        # SocNavObservationFusion already emits pedestrian velocities in the robot frame.
        rel_vel = fields.pedestrian_velocities - robot_velocity
    else:
        # Convert SocNav's robot-frame pedestrian velocities back to world frame.
        pedestrian_velocity_world = fields.pedestrian_velocities @ rotation
        rel_vel = pedestrian_velocity_world - robot_velocity
    distances = np.linalg.norm(rel, axis=1)
    bearings = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.lexsort(
        (
            np.arange(rel.shape[0], dtype=np.int64),
            rel_vel[:, 1],
            rel_vel[:, 0],
            rel[:, 1],
            rel[:, 0],
            distances,
        )
    )
    active_count = min(config.max_pedestrians, order.shape[0])
    selected = order[:active_count]
    features[:active_count, 0] = rel[selected, 0]
    features[:active_count, 1] = rel[selected, 1]
    features[:active_count, 2] = rel_vel[selected, 0]
    features[:active_count, 3] = rel_vel[selected, 1]
    features[:active_count, 4] = distances[selected]
    features[:active_count, 5] = bearings[selected]
    features[:active_count, 6] = fields.pedestrian_radius
    mask[:active_count] = True
    return features, mask


def _build_static_obstacle_features(
    obstacle_segments: Any | None,
    *,
    fields: _SocNavFields,
    config: SocialGraphObservationConfig,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded static-obstacle segment features and active mask."""
    features = np.zeros(
        (config.max_static_obstacles, len(STATIC_OBSTACLE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    mask = np.zeros((config.max_static_obstacles,), dtype=bool)
    if (
        not config.include_static_obstacles
        or config.max_static_obstacles == 0
        or obstacle_segments is None
    ):
        return features, mask

    segments = np.asarray(obstacle_segments, dtype=np.float32)
    if segments.size == 0:
        return features, mask
    segments = segments.reshape(-1, segments.shape[-1])
    if segments.shape[1] < 4:
        return features, mask
    starts = segments[:, :2]
    ends = segments[:, 2:4]
    mids = (starts + ends) / 2.0
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    safe_lengths = np.where(lengths > 0.0, lengths, 1.0)
    unit_dirs = directions / safe_lengths[:, np.newaxis]
    rel_mid = mids - fields.robot_position
    if config.robot_frame:
        rel_mid = rel_mid @ rotation.T
        unit_dirs = unit_dirs @ rotation.T
    distances = np.linalg.norm(rel_mid, axis=1)
    order = np.lexsort(
        (
            np.arange(rel_mid.shape[0], dtype=np.int64),
            rel_mid[:, 1],
            rel_mid[:, 0],
            distances,
        )
    )
    active_count = min(config.max_static_obstacles, order.shape[0])
    selected = order[:active_count]
    features[:active_count, 0] = rel_mid[selected, 0]
    features[:active_count, 1] = rel_mid[selected, 1]
    features[:active_count, 2] = unit_dirs[selected, 0]
    features[:active_count, 3] = unit_dirs[selected, 1]
    features[:active_count, 4] = lengths[selected]
    features[:active_count, 5] = distances[selected]
    features[:active_count, 6] = 0.0
    mask[:active_count] = True
    return features, mask


def _build_star_edges(
    *,
    pedestrian_mask: np.ndarray,
    static_obstacle_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build typed edges from active non-robot nodes to the robot node.

    Returns:
        Edge index array and integer edge-type labels.
    """
    active_ped_indices = np.flatnonzero(pedestrian_mask).astype(np.int64) + 1
    num_ped = active_ped_indices.shape[0]
    static_offset = 1 + pedestrian_mask.shape[0]
    active_static_indices = np.flatnonzero(static_obstacle_mask).astype(np.int64) + static_offset
    num_static = active_static_indices.shape[0]
    total_edges = num_ped + num_static
    if total_edges == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    edge_index = np.zeros((2, total_edges), dtype=np.int64)
    edge_index[0, :num_ped] = active_ped_indices
    edge_index[0, num_ped:] = active_static_indices
    edge_type = np.zeros((total_edges,), dtype=np.int64)
    edge_type[num_ped:] = 1
    return edge_index, edge_type
