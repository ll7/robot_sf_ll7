"""
SocNavBench-inspired structured observation builder.

Provides a lightweight, in-process equivalent of SocNavBench's sense payload so
planners can be reused without socket I/O.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import pi
from typing import Any

import numpy as np
from gymnasium import spaces
from loguru import logger
from shapely.geometry import LineString
from shapely.prepared import prep

from robot_sf.common.math_utils import wrap_angle_pi, wrap_angle_pi_array
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.planner.predictive_foresight import (
    PredictiveForesightEncoder,
    predictive_foresight_config_from_source,
    predictive_foresight_spaces,
)
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianObservationSnapshot,
    PedestrianTracker,
    PedestrianTrackingResult,
    TrackStatus,
)
from robot_sf.sim.simulator import Simulator

DEFAULT_MAX_PEDS = 64
"""Default upper bound for structured SocNav observations when config doesn't set max_total_pedestrians."""

SOCNAV_POSITION_CAP_M = 50.0
"""Global cap for SocNav position-like observations to keep bounds consistent."""

MAX_ROUTE_WAYPOINTS = 32
"""Maximum number of route waypoints exposed in the structured observation.

The total route_waypoints array may contain up to this many entries (including
the robot position prefix when remaining waypoints are non-empty). Capping
prevents unbounded memory growth for maps with extremely dense route waypoint
lists while keeping the DWA global-route probe within a manageable search
radius for nearest-waypoint resolution.
"""


def dynamic_pedestrian_occlusion_mask(
    ped_positions: np.ndarray,
    *,
    robot_pos: np.ndarray,
    pedestrian_radius: float,
    base_visible: np.ndarray | None = None,
) -> np.ndarray:
    """Return planner-facing pedestrian visibility after dynamic body occlusion.

    Contract for this first helper:
    - pedestrians are circular blockers with radius ``pedestrian_radius``;
    - only a target pedestrian's center-line visibility is tested;
    - a nearer visible pedestrian blocks a farther target when the line segment from
      robot center to target center passes through the nearer pedestrian disk;
    - ground-truth simulator state is not modified.
    """
    ped_positions = np.asarray(ped_positions, dtype=float)
    if ped_positions.size == 0:
        return np.zeros((0,), dtype=bool)
    if pedestrian_radius < 0:
        raise ValueError("pedestrian_radius must be >= 0")

    if base_visible is None:
        visible = np.ones((ped_positions.shape[0],), dtype=bool)
    else:
        mask = np.asarray(base_visible, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != ped_positions.shape[0]:
            raise ValueError("base_visible must be a 1D mask matching ped_positions length")
        visible = mask.copy()
    robot = np.asarray(robot_pos, dtype=float)
    rel = ped_positions - robot
    dists = np.linalg.norm(rel, axis=1)
    order = np.argsort(dists, kind="stable")

    for target_idx in order:
        if not visible[target_idx]:
            continue
        target_dist = float(dists[target_idx])
        if target_dist <= 0.0:
            continue
        if _blocked_by_nearer_pedestrian(
            target_idx=target_idx,
            order=order,
            visible=visible,
            dists=dists,
            rel=rel,
            pedestrian_radius=pedestrian_radius,
        ):
            visible[target_idx] = False
    return visible


def _blocked_by_nearer_pedestrian(
    *,
    target_idx: int,
    order: np.ndarray,
    visible: np.ndarray,
    dists: np.ndarray,
    rel: np.ndarray,
    pedestrian_radius: float,
) -> bool:
    """Return whether a nearer visible pedestrian occludes the target pedestrian."""
    target_dist = float(dists[target_idx])
    target_vec = rel[target_idx]
    for blocker_idx in order:
        if blocker_idx == target_idx or not visible[blocker_idx]:
            continue
        blocker_dist = float(dists[blocker_idx])
        if blocker_dist >= target_dist:
            break
        if _point_blocks_segment(
            rel_point=rel[blocker_idx],
            radius=pedestrian_radius,
            segment_vec=target_vec,
            segment_len_sq=target_dist * target_dist,
        ):
            return True
    return False


def _point_blocks_segment(
    *,
    rel_point: np.ndarray,
    radius: float,
    segment_vec: np.ndarray,
    segment_len_sq: float,
) -> bool:
    """Return whether a disk centered at ``point`` blocks a center-line segment."""
    if segment_len_sq <= 0.0:
        return False
    projection = float(np.dot(rel_point, segment_vec) / segment_len_sq)
    if projection <= 0.0 or projection >= 1.0:
        return False
    dist_sq = float(np.dot(rel_point, rel_point) - (projection**2 * segment_len_sq))
    return max(0.0, dist_sq) <= radius * radius


def _map_position_cap(map_def: Any) -> np.ndarray:
    """Return per-axis position caps that preserve coordinates on larger maps."""
    width = float(getattr(map_def, "width", SOCNAV_POSITION_CAP_M) or SOCNAV_POSITION_CAP_M)
    height = float(getattr(map_def, "height", SOCNAV_POSITION_CAP_M) or SOCNAV_POSITION_CAP_M)
    return np.array(
        [
            max(SOCNAV_POSITION_CAP_M, width),
            max(SOCNAV_POSITION_CAP_M, height),
        ],
        dtype=np.float32,
    )


def socnav_observation_space(
    map_def: MapDefinition,
    env_config: RobotSimulationConfig,
    max_pedestrians: int,
) -> spaces.Dict:
    """
    Create a Gym ``spaces.Dict`` mirroring a SocNavBench-style sim state.

    Returns:
        spaces.Dict: Structured observation specification for the SocNav mode.
    """
    pos_cap = _map_position_cap(map_def)
    pos_low = np.array([0.0, 0.0], dtype=np.float32)
    pos_high = pos_cap
    heading_low = np.array([-pi], dtype=np.float32)
    heading_high = np.array([pi], dtype=np.float32)
    # Extract realistic speed limits from robot configuration
    if hasattr(env_config.robot_config, "max_linear_speed"):
        max_speed = env_config.robot_config.max_linear_speed
    else:
        max_speed = 2.0  # Conservative fallback for unknown robot types
    speed_bounds = np.array([max_speed, max_speed], dtype=np.float32)
    radius_bounds = np.array([0.0], dtype=np.float32)

    ped_positions_high = np.broadcast_to(pos_high, (max_pedestrians, 2)).astype(np.float32)
    ped_positions_low = np.broadcast_to(pos_low, (max_pedestrians, 2)).astype(np.float32)
    ped_vel_low = np.broadcast_to(-speed_bounds, (max_pedestrians, 2)).astype(np.float32)
    ped_vel_high = np.broadcast_to(speed_bounds, (max_pedestrians, 2)).astype(np.float32)
    ped_track_id_low = np.full((max_pedestrians,), -1, dtype=np.int64)
    ped_track_id_high = np.full((max_pedestrians,), np.iinfo(np.int32).max, dtype=np.int64)
    visibility_settings = getattr(env_config, "observation_visibility", None)
    include_track_ids = bool(getattr(visibility_settings, "include_track_ids", False))

    return spaces.Dict(
        {
            "robot": spaces.Dict(
                {
                    "position": spaces.Box(low=pos_low, high=pos_high, dtype=np.float32),
                    "heading": spaces.Box(low=heading_low, high=heading_high, dtype=np.float32),
                    "speed": spaces.Box(
                        low=-speed_bounds,
                        high=speed_bounds,
                        dtype=np.float32,
                    ),
                    "velocity_xy": spaces.Box(
                        low=-speed_bounds,
                        high=speed_bounds,
                        dtype=np.float32,
                    ),
                    "angular_velocity": spaces.Box(
                        low=np.array([-np.finfo(np.float32).max], dtype=np.float32),
                        high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "radius": spaces.Box(
                        low=radius_bounds,
                        high=np.array([float(np.max(pos_cap))], dtype=np.float32),
                        dtype=np.float32,
                    ),
                },
            ),
            "goal": spaces.Dict(
                {
                    "current": spaces.Box(low=pos_low, high=pos_high, dtype=np.float32),
                    "next": spaces.Box(low=pos_low, high=pos_high, dtype=np.float32),
                },
            ),
            "pedestrians": spaces.Dict(
                {
                    "positions": spaces.Box(
                        low=ped_positions_low,
                        high=ped_positions_high,
                        dtype=np.float32,
                    ),
                    "velocities": spaces.Box(
                        low=ped_vel_low,
                        high=ped_vel_high,
                        dtype=np.float32,
                    ),
                    "radius": spaces.Box(
                        low=radius_bounds,
                        high=np.array([float(np.max(pos_cap))], dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "count": spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([float(max_pedestrians)], dtype=np.float32),
                        dtype=np.float32,
                    ),
                    **(
                        {
                            "track_id": spaces.Box(
                                low=ped_track_id_low,
                                high=ped_track_id_high,
                                dtype=np.int64,
                            )
                        }
                        if include_track_ids
                        else {}
                    ),
                },
            ),
            "map": spaces.Dict(
                {
                    "size": spaces.Box(
                        low=pos_low,
                        high=pos_cap,
                        dtype=np.float32,
                    ),
                },
            ),
            "sim": spaces.Dict(
                {
                    "timestep": spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                        dtype=np.float32,
                    ),
                },
            ),
            **(
                {
                    "predictive": predictive_foresight_spaces(
                        predictive_foresight_config_from_source(
                            env_config,
                            default_max_agents=max_pedestrians,
                        )
                    )
                }
                if getattr(env_config, "predictive_foresight_enabled", False)
                else {}
            ),
            **(
                {
                    "route_waypoints": spaces.Box(
                        low=np.zeros((MAX_ROUTE_WAYPOINTS, 2), dtype=np.float32),
                        high=np.broadcast_to(pos_high, (MAX_ROUTE_WAYPOINTS, 2)).astype(np.float32),
                        dtype=np.float32,
                    ),
                }
                if getattr(env_config, "include_route_waypoints", False)
                else {}
            ),
        }
    )


@dataclass
class SocNavObservationFusion:
    """Structured observation builder used when ``ObservationMode.SOCNAV_STRUCT`` is enabled."""

    simulator: Simulator
    env_config: RobotSimulationConfig
    max_pedestrians: int
    robot_index: int = 0
    truncation_warned: bool = False
    _predictive_foresight: PredictiveForesightEncoder | None = None
    _last_heading: float | None = None
    _pedestrian_tracker: PedestrianTracker | None = field(init=False, default=None, repr=False)
    _tracking_step_index: int = field(init=False, default=0, repr=False)
    _buf_ped_track_ids: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize optional predictive foresight encoder and reusable buffers."""
        visibility_settings = getattr(self.env_config, "observation_visibility", None)
        if bool(getattr(visibility_settings, "include_track_ids", False)):
            tracking_spec = getattr(visibility_settings, "tracking_config", None)
            if tracking_spec is not None and not isinstance(tracking_spec, Mapping):
                raise TypeError("observation_visibility.tracking_config must be a mapping or None")
            tracking_mapping = {} if tracking_spec is None else dict(tracking_spec)
            if tracking_mapping.get("enabled") is False:
                raise ValueError(
                    "observation_visibility.include_track_ids requires tracking_config.enabled"
                )
            tracking_mapping["enabled"] = True
            self._pedestrian_tracker = PedestrianTracker(tracking_mapping)
            self._buf_ped_track_ids = np.full((self.max_pedestrians,), -1, dtype=np.int64)
        if bool(getattr(self.env_config, "predictive_foresight_enabled", False)):
            self._predictive_foresight = PredictiveForesightEncoder(
                predictive_foresight_config_from_source(
                    self.env_config,
                    default_max_agents=self.max_pedestrians,
                )
            )
        self._buf_ped_positions = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        self._buf_ped_velocities = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        self._cache_static_map_def_id = None
        self._cache_static_obstacles_id = None
        self._cache_static_prepared: list[tuple[Any, Any]] = []
        self._cache_position_cap_map_def_id: int | None = None
        self._cache_position_cap_value: np.ndarray | None = None
        self._cache_position_cap_width: float | None = None
        self._cache_position_cap_height: float | None = None
        self._lost_pedestrian_memory: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def reset_cache(self) -> None:
        """Reset internal caches to match the SensorFusion interface."""
        self._last_heading = None
        self._cache_static_map_def_id = None
        self._cache_static_obstacles_id = None
        self._cache_static_prepared = []
        self._cache_position_cap_map_def_id = None
        self._cache_position_cap_value = None
        self._cache_position_cap_width = None
        self._cache_position_cap_height = None
        self._lost_pedestrian_memory.clear()
        self._tracking_step_index = 0
        if self._pedestrian_tracker is not None:
            self._pedestrian_tracker.reset()
        if self._buf_ped_track_ids is not None:
            self._buf_ped_track_ids.fill(-1)

    def _position_cap(self) -> np.ndarray:
        """Return cached map position cap, refreshing when map_def identity or dimensions change.

        Avoids allocating a new array every step via ``_map_position_cap``.
        """
        map_def = getattr(self.simulator, "map_def", None)
        if map_def is not None:
            width = float(getattr(map_def, "width", 0.0) or 0.0)
            height = float(getattr(map_def, "height", 0.0) or 0.0)
            map_def_id = id(map_def)
            cache_valid = (
                self._cache_position_cap_map_def_id == map_def_id
                and self._cache_position_cap_width == width
                and self._cache_position_cap_height == height
            )
            if not cache_valid:
                self._cache_position_cap_map_def_id = map_def_id
                self._cache_position_cap_width = width
                self._cache_position_cap_height = height
                self._cache_position_cap_value = _map_position_cap(map_def)
        elif self._cache_position_cap_map_def_id is not None:
            self._cache_position_cap_map_def_id = None
            self._cache_position_cap_value = _map_position_cap(None)
            self._cache_position_cap_width = None
            self._cache_position_cap_height = None
        if self._cache_position_cap_value is None:
            self._cache_position_cap_value = _map_position_cap(None)
        return self._cache_position_cap_value

    def _robot_velocity_xy(self, wrapped_heading: float) -> np.ndarray:
        """Return the robot world-frame planar velocity for the structured observation."""
        robot = self.simulator.robots[self.robot_index]
        velocity_xy = getattr(getattr(robot, "state", None), "velocity_xy", None)
        if velocity_xy is not None:
            arr = np.asarray(velocity_xy, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return arr[:2]

        current_speed = np.asarray(robot.current_speed, dtype=np.float32).reshape(-1)
        linear_speed = float(current_speed[0]) if current_speed.size > 0 else 0.0
        return np.array(
            [
                linear_speed * float(np.cos(wrapped_heading)),
                linear_speed * float(np.sin(wrapped_heading)),
            ],
            dtype=np.float32,
        )

    def _robot_angular_velocity(self, wrapped_heading: float) -> np.ndarray:
        """Return robot angular velocity while avoiding drivetrain-specific contract drift."""
        robot = self.simulator.robots[self.robot_index]
        state = getattr(robot, "state", None)
        velocity_vw = getattr(state, "velocity_vw", None)
        if velocity_vw is not None:
            arr = np.asarray(velocity_vw, dtype=np.float32).reshape(-1)
            if arr.size > 1:
                return np.array([float(arr[1])], dtype=np.float32)

        velocity = getattr(state, "velocity", None)
        if isinstance(velocity, tuple) and len(velocity) > 1:
            return np.array([float(velocity[1])], dtype=np.float32)

        dt = float(getattr(self.simulator.config, "time_per_step_in_secs", 0.0) or 0.0)
        if self._last_heading is None or dt <= 0.0:
            angular_velocity = 0.0
        else:
            delta = wrap_angle_pi(wrapped_heading - self._last_heading)
            angular_velocity = float(delta / dt)
        self._last_heading = float(wrapped_heading)
        return np.array([angular_velocity], dtype=np.float32)

    def _visible_pedestrian_mask(
        self,
        ped_positions: np.ndarray,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> np.ndarray:
        """Return the planner-facing visibility mask for pedestrian positions."""
        if ped_positions.size == 0:
            return np.zeros((0,), dtype=bool)
        settings = getattr(self.env_config, "observation_visibility", None)
        if settings is None or not bool(getattr(settings, "enabled", False)):
            return np.ones((ped_positions.shape[0],), dtype=bool)

        visible = np.ones((ped_positions.shape[0],), dtype=bool)
        rel = ped_positions - robot_pos
        dists = np.linalg.norm(rel, axis=1)

        max_range_m = getattr(settings, "max_range_m", None)
        if max_range_m is not None:
            visible &= dists <= float(max_range_m)

        fov_degrees = float(getattr(settings, "fov_degrees", 360.0))
        if fov_degrees < 360.0:
            bearings = np.arctan2(rel[:, 1], rel[:, 0])
            deltas = wrap_angle_pi_array(bearings - robot_heading)
            visible &= np.abs(deltas) <= np.deg2rad(fov_degrees) / 2.0

        if bool(getattr(settings, "static_occlusion", False)):
            for idx, ped_pos in enumerate(ped_positions):
                if visible[idx] and self._statically_occluded(robot_pos=robot_pos, ped_pos=ped_pos):
                    visible[idx] = False
        if bool(getattr(settings, "dynamic_occlusion", False)):
            visible = dynamic_pedestrian_occlusion_mask(
                ped_positions,
                robot_pos=robot_pos,
                pedestrian_radius=float(getattr(self.env_config.sim_config, "ped_radius", 0.0)),
                base_visible=visible,
            )
        return visible

    def _pedestrians_with_lost_memory(
        self,
        *,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        visibility_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return visible pedestrians plus memories and their source-row indices.

        This is presentation memory only.  When track IDs are enabled, the
        observation-derived tracker owns identity and this helper merely keeps
        the existing short-horizon display behavior for invisible rows.
        """
        settings = getattr(self.env_config, "observation_visibility", None)
        memory_enabled = bool(getattr(settings, "memory_for_lost_pedestrians", False))
        horizon_s = float(getattr(settings, "lost_pedestrian_memory_horizon_s", 0.0) or 0.0)
        if not memory_enabled or horizon_s <= 0.0:
            self._lost_pedestrian_memory.clear()
            source_indices = np.flatnonzero(visibility_mask).astype(np.int64, copy=False)
            return ped_positions[source_indices], ped_velocities[source_indices], source_indices

        dt = float(getattr(self.simulator.config, "time_per_step_in_secs", 0.0) or 0.0)
        step_s = max(dt, 0.0)
        keep_positions: list[np.ndarray] = []
        keep_velocities: list[np.ndarray] = []
        keep_source_indices: list[int] = []
        live_ids = set(range(ped_positions.shape[0]))

        for ped_idx, (position, velocity, visible) in enumerate(
            zip(ped_positions, ped_velocities, visibility_mask, strict=True)
        ):
            if bool(visible):
                position_copy = np.asarray(position, dtype=np.float32).copy()
                velocity_copy = np.asarray(velocity, dtype=np.float32).copy()
                self._lost_pedestrian_memory[ped_idx] = (position_copy, velocity_copy, 0.0)
                keep_positions.append(position_copy)
                keep_velocities.append(velocity_copy)
                keep_source_indices.append(ped_idx)
                continue

            remembered = self._lost_pedestrian_memory.get(ped_idx)
            if remembered is None:
                continue
            last_position, last_velocity, age_s = remembered
            next_age_s = age_s + step_s
            if next_age_s > horizon_s:
                self._lost_pedestrian_memory.pop(ped_idx, None)
                continue
            predicted_position = (last_position + last_velocity * step_s).astype(np.float32)
            velocity_copy = last_velocity.astype(np.float32, copy=True)
            self._lost_pedestrian_memory[ped_idx] = (
                predicted_position,
                velocity_copy,
                next_age_s,
            )
            keep_positions.append(predicted_position)
            keep_velocities.append(velocity_copy)
            keep_source_indices.append(ped_idx)

        for stale_idx in set(self._lost_pedestrian_memory) - live_ids:
            self._lost_pedestrian_memory.pop(stale_idx, None)

        if not keep_positions:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.stack(keep_positions).astype(np.float32, copy=False),
            np.stack(keep_velocities).astype(np.float32, copy=False),
            np.asarray(keep_source_indices, dtype=np.int64),
        )

    def _pedestrian_order(
        self,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        *,
        robot_pos: np.ndarray,
    ) -> np.ndarray:
        """Return the configured closest-first order for pedestrian rows."""
        rel = ped_positions - robot_pos
        dists = np.linalg.norm(rel, axis=1)
        settings = getattr(self.env_config, "observation_visibility", None)
        tie_break = str(getattr(settings, "ordering_tie_break", "legacy")).strip().lower()
        if tie_break == "legacy":
            # Preserve NumPy's historical default quicksort path when the option is disabled.
            return np.argsort(dists)
        if tie_break != "stable":
            raise ValueError("ordering_tie_break must be 'legacy' or 'stable'")

        # ``kind='stable'`` makes the final source index an explicit, deterministic key.
        # All preceding fields are observation content, so non-identical ties do not depend
        # on the simulator's row order.
        sort_keys = np.empty(
            (ped_positions.shape[0],),
            dtype=[
                ("distance", np.float64),
                ("relative_x", np.float64),
                ("relative_y", np.float64),
                ("velocity_x", np.float64),
                ("velocity_y", np.float64),
                ("source_index", np.int64),
            ],
        )
        sort_keys["distance"] = dists
        sort_keys["relative_x"] = rel[:, 0]
        sort_keys["relative_y"] = rel[:, 1]
        sort_keys["velocity_x"] = ped_velocities[:, 0]
        sort_keys["velocity_y"] = ped_velocities[:, 1]
        sort_keys["source_index"] = np.arange(ped_positions.shape[0], dtype=np.int64)
        return np.argsort(sort_keys, order=sort_keys.dtype.names, kind="stable")

    def _update_pedestrian_tracking(
        self,
        *,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        visibility_mask: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> tuple[np.ndarray, PedestrianTrackingResult | None]:
        """Update the opt-in observation-derived tracker and map IDs to source rows.

        Returns:
            Tuple of source-row track IDs and the immutable tracker result.  IDs
            are ``-1`` for rows without an accepted visible association.
        """
        source_track_ids = np.full((ped_positions.shape[0],), -1, dtype=np.int64)
        if self._pedestrian_tracker is None:
            return source_track_ids, None

        tracking_velocities = ped_velocities
        if tracking_velocities.shape != ped_positions.shape:
            tracking_velocities = np.zeros_like(ped_positions, dtype=np.float32)
        valid_mask = np.all(np.isfinite(ped_positions), axis=1)
        visible_mask = np.asarray(visibility_mask, dtype=bool) & valid_mask
        velocity_valid_mask = visible_mask & np.all(np.isfinite(tracking_velocities), axis=1)
        snapshot = PedestrianObservationSnapshot(
            timestamp_s=self._tracking_step_index
            * max(float(getattr(self.simulator.config, "time_per_step_in_secs", 0.0) or 0.0), 0.0),
            step_index=self._tracking_step_index,
            coordinate_frame="global_xy",
            robot_pose_global=np.array([robot_pos[0], robot_pos[1], heading], dtype=float),
            positions=ped_positions,
            velocities=tracking_velocities,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            velocity_valid_mask=velocity_valid_mask,
            radius=float(getattr(self.env_config.sim_config, "ped_radius", 0.0) or 0.0),
        )
        result = self._pedestrian_tracker.update(snapshot)
        self._tracking_step_index += 1
        for association in result.associations:
            if 0 <= association.observation_slot < source_track_ids.shape[0]:
                source_track_ids[association.observation_slot] = association.track_id
        for track in result.tracks:
            observation_slot = track.last_observation_slot
            if (
                observation_slot is not None
                and 0 <= observation_slot < source_track_ids.shape[0]
                and visible_mask[observation_slot]
                and track.step_index == result.step_index
                and TrackStatus(track.status) in {TrackStatus.TENTATIVE, TrackStatus.CONFIRMED}
            ):
                source_track_ids[observation_slot] = track.track_id
        return source_track_ids, result

    def _track_ids_for_presentation_rows(
        self,
        *,
        source_indices: np.ndarray,
        source_track_ids: np.ndarray,
        tracking_result: PedestrianTrackingResult | None,
    ) -> np.ndarray | None:
        """Map tracker IDs onto visible and presentation-memory rows.

        Visible rows use their source-slot associations.  A remembered row may
        also receive its tracker's lost ID when the source slot is unambiguous;
        otherwise ``-1`` is retained instead of inventing identity.

        Returns:
            Aligned int64 track IDs, or ``None`` when tracking is disabled.
        """
        if self._pedestrian_tracker is None:
            return None
        track_ids = np.full((source_indices.shape[0],), -1, dtype=np.int64)
        for output_index, source_index in enumerate(source_indices.tolist()):
            if 0 <= source_index < source_track_ids.shape[0]:
                track_ids[output_index] = source_track_ids[source_index]
        if tracking_result is None:
            return track_ids

        lost_tracks = {
            track.track_id: track
            for track in tracking_result.tracks
            if TrackStatus(track.status) is TrackStatus.LOST
        }
        used_track_ids = {int(track_id) for track_id in track_ids if track_id > 0}
        for output_index, source_index in enumerate(source_indices.tolist()):
            if track_ids[output_index] > 0:
                continue
            matches = [
                track
                for track in lost_tracks.values()
                if track.track_id not in used_track_ids
                and track.last_observation_slot == source_index
            ]
            if len(matches) == 1:
                track_ids[output_index] = matches[0].track_id
                used_track_ids.add(matches[0].track_id)
        return track_ids

    def _rebuild_static_occlusion_cache(self) -> None:
        """Build prepared-geometry cache from current simulator obstacle polygons."""
        map_def = getattr(self.simulator, "map_def", None)
        obstacles = getattr(map_def, "obstacles", None) if map_def is not None else None
        self._cache_static_map_def_id = id(map_def) if map_def is not None else None
        self._cache_static_obstacles_id = id(obstacles) if obstacles is not None else None
        prepared_polys: list[tuple[Any, Any]] = []
        if obstacles:
            for obstacle in obstacles:
                polygons = (
                    obstacle.iter_polygons()
                    if hasattr(obstacle, "iter_polygons")
                    else [getattr(obstacle, "geometry", None)]
                )
                for polygon in polygons:
                    if polygon is None or getattr(polygon, "is_empty", False):
                        continue
                    prepared_polys.append((prep(polygon), polygon))
        self._cache_static_prepared = prepared_polys

    def _statically_occluded(self, *, robot_pos: np.ndarray, ped_pos: np.ndarray) -> bool:
        """Return whether static map geometry blocks line of sight to a pedestrian.

        Uses a Shapely prepared-geometry cache that refreshes when
        ``simulator.map_def`` identity or ``map_def.obstacles`` identity changes.
        Semantics: occluded iff the line segment intersects the polygon interior
        (intersects without mere boundary touch).
        """
        map_def = getattr(self.simulator, "map_def", None)
        obstacles = getattr(map_def, "obstacles", None) if map_def is not None else None
        if not obstacles:
            self._cache_static_map_def_id = id(map_def) if map_def is not None else None
            self._cache_static_obstacles_id = None
            self._cache_static_prepared = []
            return False

        cache_valid = self._cache_static_map_def_id == id(
            map_def
        ) and self._cache_static_obstacles_id == id(obstacles)
        if not cache_valid:
            self._rebuild_static_occlusion_cache()

        if not self._cache_static_prepared:
            return False

        segment = LineString(
            [
                (float(robot_pos[0]), float(robot_pos[1])),
                (float(ped_pos[0]), float(ped_pos[1])),
            ]
        )
        if segment.length <= 0.0:
            return False

        for prep_geom, raw_poly in self._cache_static_prepared:
            if prep_geom.intersects(segment) and not segment.touches(raw_poly):
                return True
        return False

    def _build_route_waypoints(self, position_cap: np.ndarray) -> np.ndarray:
        """Extract remaining route waypoints for the DWA global-route probe.

        Reads from ``simulator.robot_navs`` and returns a ``(N, 2)`` float32
        array containing the robot position (when remaining waypoints exist)
        followed by remaining waypoints from ``waypoint_id`` onward. The array
        is clipped to the position cap and bounded by ``MAX_ROUTE_WAYPOINTS``.

        Contract:
        - Robot position is prepended when remaining waypoints exist.
        - When the navigator has no waypoints, returns empty ``(0, 2)``.
        - When all waypoints are visited but a route existed, returns robot
          position ``(1, 2)`` so the probe can still locate the robot on route.
        - Total rows never exceed ``MAX_ROUTE_WAYPOINTS``.
        - Malformed navigator state fails closed with an empty array.

        Returns:
            np.ndarray: Clipped, capped route waypoint array.
        """
        empty = np.zeros((0, 2), dtype=np.float32)
        navs_raw = getattr(self.simulator, "robot_navs", None)
        if navs_raw is None:
            return empty

        try:
            nav = navs_raw[self.robot_index]
            waypoint_id = int(getattr(nav, "waypoint_id", 0))
            if waypoint_id < 0:
                return empty
            remaining_arr = np.asarray(
                getattr(nav, "waypoints", None)[waypoint_id:], dtype=np.float32
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return empty

        if remaining_arr.size == 0:
            if waypoint_id == 0:
                return empty
            remaining_arr = empty
        elif remaining_arr.ndim != 2 or remaining_arr.shape[-1] != 2:
            return empty

        try:
            robot_pose = self.simulator.robots[self.robot_index].pose
            robot_pos = np.asarray(robot_pose[0], dtype=np.float32).reshape(-1)
        except (AttributeError, IndexError, TypeError, ValueError):
            return empty
        if robot_pos.shape != (2,) or not np.all(np.isfinite(robot_pos)):
            return empty
        if remaining_arr.size > 0 and not np.all(np.isfinite(remaining_arr)):
            return empty

        wp_with_robot = np.vstack([robot_pos.reshape(1, 2), remaining_arr])
        wp_with_robot = np.clip(wp_with_robot, 0.0, position_cap)
        return wp_with_robot[:MAX_ROUTE_WAYPOINTS]

    def _prepare_pedestrian_rows(
        self,
        *,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        robot_pos: np.ndarray,
        heading: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Filter, track, order, and cap pedestrian rows for the public observation.

        Returns:
            Tuple of capped positions, capped world-frame velocities, and optional
            aligned track IDs.  The ID array is ``None`` when the channel is disabled.
        """
        visibility_mask = self._visible_pedestrian_mask(
            ped_positions,
            robot_pos=robot_pos,
            robot_heading=heading,
        )
        source_track_ids, tracking_result = self._update_pedestrian_tracking(
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            visibility_mask=visibility_mask,
            robot_pos=robot_pos,
            heading=heading,
        )
        ped_positions, ped_velocities, source_indices = self._pedestrians_with_lost_memory(
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            visibility_mask=visibility_mask,
        )
        ped_track_ids = self._track_ids_for_presentation_rows(
            source_indices=source_indices,
            source_track_ids=source_track_ids,
            tracking_result=tracking_result,
        )

        if ped_positions.size > 0:
            order = self._pedestrian_order(
                ped_positions,
                ped_velocities,
                robot_pos=robot_pos,
            )
            ped_positions = ped_positions[order]
            ped_velocities = ped_velocities[order]
            if ped_track_ids is not None:
                ped_track_ids = ped_track_ids[order]

        ped_positions = ped_positions[: self.max_pedestrians]
        ped_velocities = ped_velocities[: self.max_pedestrians]
        if ped_track_ids is not None:
            ped_track_ids = ped_track_ids[: self.max_pedestrians]
        return ped_positions, ped_velocities, ped_track_ids

    def next_obs(self) -> dict[str, Any]:
        """Return the latest structured observation aligned to the declared space."""
        ped_positions = np.asarray(self.simulator.ped_pos, dtype=np.float32)
        try:
            ped_velocities = np.asarray(self.simulator.ped_vel, dtype=np.float32)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive fallback
            ped_velocities = np.zeros_like(ped_positions, dtype=np.float32)
        total_peds = ped_positions.shape[0]
        if total_peds > self.max_pedestrians and not self.truncation_warned:
            logger.warning(
                "SocNav structured obs truncating pedestrians: seen={}, max_pedestrians={}. "
                "Increase the configured max_pedestrians (or SimulationSettings.max_total_pedestrians) to avoid data loss.",
                total_peds,
                self.max_pedestrians,
            )
            self.truncation_warned = True

        robot_pose = self.simulator.robots[self.robot_index].pose
        robot_pos = np.asarray(robot_pose[0], dtype=np.float32)
        heading = float(robot_pose[1])
        ped_positions, ped_velocities, ped_track_ids = self._prepare_pedestrian_rows(
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            robot_pos=robot_pos,
            heading=heading,
        )
        self._buf_ped_positions.fill(0.0)
        self._buf_ped_velocities.fill(0.0)
        if self._buf_ped_track_ids is not None:
            self._buf_ped_track_ids.fill(-1)
        if ped_positions.size > 0:
            self._buf_ped_positions[: ped_positions.shape[0]] = ped_positions
            if self._buf_ped_track_ids is not None and ped_track_ids is not None:
                self._buf_ped_track_ids[: ped_track_ids.shape[0]] = ped_track_ids
        if ped_velocities.size > 0:
            # Convert pedestrian velocities to ego frame (rotate by -heading)
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            vx = ped_velocities[:, 0]
            vy = ped_velocities[:, 1]
            ego_vx = cos_h * vx + sin_h * vy
            ego_vy = -sin_h * vx + cos_h * vy
            valid_velocity_count = ped_velocities.shape[0]
            self._buf_ped_velocities[:valid_velocity_count, 0] = ego_vx
            self._buf_ped_velocities[:valid_velocity_count, 1] = ego_vy

        goal = np.asarray(self.simulator.goal_pos[self.robot_index], dtype=np.float32)
        next_goal = self.simulator.next_goal_pos[self.robot_index]
        next_goal_arr = (
            np.asarray(next_goal, dtype=np.float32)
            if next_goal is not None
            else np.zeros(2, dtype=np.float32)
        )

        position_cap = self._position_cap()

        def _clip_positions(values: np.ndarray) -> np.ndarray:
            """Clip world positions into the representable map extent.

            Returns:
                np.ndarray: Positions clipped to the map coordinate range.
            """
            return np.clip(values, 0.0, position_cap)

        robot_pos_clipped = _clip_positions(robot_pos)
        goal_clipped = _clip_positions(goal)
        next_goal_clipped = _clip_positions(next_goal_arr)
        np.clip(self._buf_ped_positions, 0.0, position_cap, out=self._buf_ped_positions)
        map_size = np.array(
            [self.simulator.map_def.width, self.simulator.map_def.height],
            dtype=np.float32,
        )
        map_size = np.minimum(map_size, position_cap)

        # Wrap heading to [-pi, pi] to stay within declared observation bounds
        wrapped_heading = wrap_angle_pi(float(robot_pose[1]))
        robot_speed = np.asarray(
            self.simulator.robots[self.robot_index].current_speed, dtype=np.float32
        )
        robot_velocity_xy = self._robot_velocity_xy(wrapped_heading)
        obs = {
            "robot": {
                "position": robot_pos_clipped,
                "heading": np.array([wrapped_heading], dtype=np.float32),
                "speed": robot_speed,
                "velocity_xy": robot_velocity_xy,
                "angular_velocity": self._robot_angular_velocity(wrapped_heading),
                "radius": np.array(
                    [self.simulator.robots[self.robot_index].config.radius], dtype=np.float32
                ),
            },
            "goal": {
                "current": goal_clipped,
                "next": next_goal_clipped,
            },
            "pedestrians": {
                "positions": self._buf_ped_positions.copy(),
                "velocities": self._buf_ped_velocities.copy(),
                "radius": np.array(
                    [min(self.env_config.sim_config.ped_radius, float(np.max(position_cap)))],
                    dtype=np.float32,
                ),
                "count": np.array(
                    [float(min(len(ped_positions), self.max_pedestrians))], dtype=np.float32
                ),
                **(
                    {"track_id": self._buf_ped_track_ids.copy()}
                    if self._buf_ped_track_ids is not None
                    else {}
                ),
            },
            "map": {
                "size": map_size,
            },
            "sim": {
                "timestep": np.array(
                    [self.simulator.config.time_per_step_in_secs], dtype=np.float32
                ),
            },
        }
        self._attach_optional_observations(obs, position_cap)
        return obs

    def _attach_optional_observations(self, obs: dict[str, Any], position_cap: np.ndarray) -> None:
        """Attach optional observation fields (route_waypoints, predictive) to obs dict."""
        if getattr(self.env_config, "include_route_waypoints", False):
            route_waypoints = self._build_route_waypoints(position_cap)
            # Pad to fixed size (MAX_ROUTE_WAYPOINTS, 2) with zeros
            padded = np.zeros((MAX_ROUTE_WAYPOINTS, 2), dtype=np.float32)
            if route_waypoints.size > 0:
                n = min(route_waypoints.shape[0], MAX_ROUTE_WAYPOINTS)
                padded[:n] = route_waypoints[:n]
            obs["route_waypoints"] = padded
        if self._predictive_foresight is not None:
            obs["predictive"] = self._predictive_foresight.encode(obs)
