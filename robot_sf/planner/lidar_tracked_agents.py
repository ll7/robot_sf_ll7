"""LiDAR-derived tracked-agent adapter for social-state planner tests.

This module converts sensor-fusion ``drive_state`` and ``rays`` observations
into a minimal SocNav-style observation for one wrapped social-state planner.
The tracks are endpoint clusters from the current LiDAR scan with zero
velocity and no identity persistence. That makes the adapter useful as a
fail-closed benchmark plumbing smoke, not as a perception tracker claim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from robot_sf.planner.socnav import (
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.sensor.goal_sensor import TARGET_DISTANCE_CAP_M
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


@dataclass(frozen=True)
class LidarTrackedAgentsConfig:
    """Configuration for current-frame LiDAR endpoint track extraction."""

    max_scan_dist: float = 10.0
    visual_angle_portion: float = 1.0
    normalized_observation: bool = True
    target_distance_scale: float = TARGET_DISTANCE_CAP_M
    linear_speed_scale: float = 2.0
    angular_speed_scale: float = 1.0
    robot_radius: float = 0.3
    track_radius: float = 0.3
    min_track_range: float = 0.15
    max_track_range: float | None = None
    cluster_gap_rays: int = 1
    max_tracks: int = 8
    timestep: float = 0.1

    def __post_init__(self) -> None:
        """Validate track-extraction parameters."""
        if self.max_scan_dist <= 0.0:
            raise ValueError("max_scan_dist must be > 0")
        if not 0.0 < self.visual_angle_portion <= 1.0:
            raise ValueError("visual_angle_portion must be within (0, 1]")
        if self.target_distance_scale <= 0.0:
            raise ValueError("target_distance_scale must be > 0")
        self._validate_positive_shape_params()
        if self.max_track_range is not None and self.max_track_range <= self.min_track_range:
            raise ValueError("max_track_range must be greater than min_track_range")
        if self.cluster_gap_rays < 0:
            raise ValueError("cluster_gap_rays must be >= 0")
        if self.max_tracks < 0:
            raise ValueError("max_tracks must be >= 0")
        if self.timestep <= 0.0:
            raise ValueError("timestep must be > 0")

    def _validate_positive_shape_params(self) -> None:
        """Validate positive scale, radius, and range fields."""
        if self.linear_speed_scale <= 0.0 or self.angular_speed_scale <= 0.0:
            raise ValueError("speed scales must be > 0")
        if self.robot_radius <= 0.0 or self.track_radius <= 0.0:
            raise ValueError("radii must be > 0")
        if self.min_track_range < 0.0:
            raise ValueError("min_track_range must be >= 0")

    @property
    def effective_max_track_range(self) -> float:
        """Maximum range that may produce a synthetic track."""
        return (
            float(self.max_track_range)
            if self.max_track_range is not None
            else float(self.max_scan_dist)
        )


@dataclass(frozen=True)
class LidarTrackedSocialForceBuildConfig:
    """Parsed config for the LiDAR-tracked SocialForce adapter."""

    lidar_tracking: LidarTrackedAgentsConfig
    social_force: SocNavPlannerConfig


def _latest_row(values: Any, *, name: str) -> np.ndarray:
    """Return the newest row from a scalar, vector, or stacked sensor array."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim >= 2:
        arr = arr.reshape((-1, arr.shape[-1]))[-1]
    else:
        arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr.astype(float, copy=False)


def lidar_ray_angles(num_rays: int, *, visual_angle_portion: float = 1.0) -> np.ndarray:
    """Return ray angles matching :mod:`robot_sf.sensor.range_sensor` convention.

    Returns:
        np.ndarray: Ego-frame ray angles in radians.
    """
    if num_rays <= 0:
        raise ValueError("num_rays must be > 0")
    if not 0.0 < visual_angle_portion <= 1.0:
        raise ValueError("visual_angle_portion must be within (0, 1]")
    lower = -np.pi * visual_angle_portion
    upper = np.pi * visual_angle_portion
    return np.linspace(lower, upper, int(num_rays) + 1, dtype=float)[:-1]


def _contiguous_clusters(indices: np.ndarray, *, max_gap: int) -> list[np.ndarray]:
    """Group sorted ray indices into contiguous hit clusters.

    Returns:
        list[np.ndarray]: Ray-index clusters.
    """
    if indices.size == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        current = int(idx)
        if current - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(current)
        else:
            groups.append([current])
    return [np.asarray(group, dtype=int) for group in groups]


def _merge_wraparound_cluster(
    clusters: list[np.ndarray],
    *,
    num_rays: int,
    max_gap: int,
    visual_angle_portion: float,
) -> list[np.ndarray]:
    """Merge first/last clusters when a full-circle scan wraps across -pi/pi.

    Returns:
        list[np.ndarray]: Clusters with optional wraparound merge applied.
    """
    if len(clusters) < 2 or visual_angle_portion < 1.0:
        return clusters
    first = clusters[0]
    last = clusters[-1]
    wrap_gap = int(first[0]) + int(num_rays - 1 - last[-1])
    if wrap_gap > max_gap:
        return clusters
    merged = np.concatenate([last, first])
    return [merged, *clusters[1:-1]]


def lidar_rays_to_tracked_agents(
    rays: Any,
    config: LidarTrackedAgentsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert current LiDAR ray endpoints into synthetic tracked agents.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(positions, velocities)`` arrays in
        ego/world coordinates for the zero-heading adapter frame.
    """
    cfg = config or LidarTrackedAgentsConfig()
    ray_values = _latest_row(rays, name=OBS_RAYS)
    if bool(cfg.normalized_observation):
        ray_values = ray_values * float(cfg.max_scan_dist)
    ray_values = np.clip(ray_values, 0.0, float(cfg.max_scan_dist))
    if cfg.max_tracks == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    finite_hit = (ray_values >= float(cfg.min_track_range)) & (
        ray_values < min(float(cfg.effective_max_track_range), float(cfg.max_scan_dist))
    )
    hit_indices = np.flatnonzero(finite_hit)
    clusters = _contiguous_clusters(hit_indices, max_gap=int(cfg.cluster_gap_rays))
    clusters = _merge_wraparound_cluster(
        clusters,
        num_rays=int(ray_values.size),
        max_gap=int(cfg.cluster_gap_rays),
        visual_angle_portion=float(cfg.visual_angle_portion),
    )
    angles = lidar_ray_angles(
        int(ray_values.size),
        visual_angle_portion=float(cfg.visual_angle_portion),
    )

    positions: list[np.ndarray] = []
    for cluster in clusters:
        endpoints = np.stack(
            [
                ray_values[cluster] * np.cos(angles[cluster]),
                ray_values[cluster] * np.sin(angles[cluster]),
            ],
            axis=1,
        )
        if endpoints.size == 0:
            continue
        positions.append(np.mean(endpoints, axis=0))

    if not positions:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    tracks = np.asarray(positions, dtype=np.float32)
    order = np.argsort(np.linalg.norm(tracks, axis=1), kind="stable")
    tracks = tracks[order][: int(cfg.max_tracks)]
    velocities = np.zeros_like(tracks, dtype=np.float32)
    return tracks, velocities


def sensor_fusion_to_social_force_observation(
    observation: dict[str, Any],
    config: LidarTrackedAgentsConfig | None = None,
) -> dict[str, Any]:
    """Build a SocNav observation from normalized sensor-fusion inputs.

    Returns:
        dict[str, Any]: Structured observation for :class:`SocialForcePlannerAdapter`.
    """
    cfg = config or LidarTrackedAgentsConfig()
    if not isinstance(observation, dict):
        raise ValueError("observation must be a mapping")
    if OBS_DRIVE_STATE not in observation or OBS_RAYS not in observation:
        raise ValueError("observation must contain drive_state and rays")

    drive = _latest_row(observation[OBS_DRIVE_STATE], name=OBS_DRIVE_STATE)
    if drive.size < 5:
        raise ValueError("drive_state must contain speed, turn rate, distance, and angles")
    if bool(cfg.normalized_observation):
        drive = drive.copy()
        drive[0] *= float(cfg.linear_speed_scale)
        drive[1] *= float(cfg.angular_speed_scale)
        drive[2] *= float(cfg.target_distance_scale)
        drive[3] *= np.pi
        drive[4] *= np.pi

    target_distance = float(np.clip(drive[2], 0.0, float(cfg.target_distance_scale)))
    target_angle = float(np.clip(drive[3], -np.pi, np.pi))
    goal = [
        float(target_distance * np.cos(target_angle)),
        float(target_distance * np.sin(target_angle)),
    ]
    positions, velocities = lidar_rays_to_tracked_agents(observation[OBS_RAYS], cfg)
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [float(drive[0])],
            "radius": [float(cfg.robot_radius)],
        },
        "goal": {"current": goal, "next": goal},
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "count": [int(positions.shape[0])],
            "radius": [float(cfg.track_radius)],
        },
        "sim": {"timestep": [float(cfg.timestep)]},
    }


class LidarTrackedSocialForceAdapter:
    """Testing-only LiDAR endpoint-track adapter around SocialForce."""

    def __init__(
        self,
        *,
        config: LidarTrackedAgentsConfig | None = None,
        social_force: SocialForcePlannerAdapter | None = None,
    ) -> None:
        """Initialize the adapter around a SocialForce planner."""
        self.config = config or LidarTrackedAgentsConfig()
        self.social_force = social_force or SocialForcePlannerAdapter()
        self._last_status = "not_run"
        self._last_error: str | None = None
        self._last_track_count = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset wrapped planner state when supported."""
        reset = getattr(self.social_force, "reset", None)
        if callable(reset):
            if seed is None:
                reset()
            else:
                try:
                    reset(seed=seed)
                except TypeError:
                    reset()
        self._last_status = "not_run"
        self._last_error = None
        self._last_track_count = 0

    def adapt_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Convert sensor-fusion observation into SocialForce observation.

        Returns:
            dict[str, Any]: Structured SocNav observation.
        """
        return sensor_fusion_to_social_force_observation(observation, self.config)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Plan from LiDAR-derived tracks, failing closed to stop.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        try:
            adapted = self.adapt_observation(observation)
            self._last_track_count = int(adapted["pedestrians"]["count"][0])
            linear, angular = self.social_force.plan(adapted)
        except (TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            self._last_status = "not_available"
            self._last_error = str(exc)
            self._last_track_count = 0
            return 0.0, 0.0
        self._last_status = "ok"
        self._last_error = None
        return float(linear), float(angular)

    def diagnostics(self) -> dict[str, Any]:
        """Expose adapter contract and last-run status for benchmark metadata.

        Returns:
            dict[str, Any]: JSON-compatible adapter diagnostics.
        """
        return {
            "adapter": "LidarTrackedSocialForceAdapter",
            "status": self._last_status,
            "error": self._last_error,
            "observation_level": "lidar_2d",
            "execution_mode": "adapter",
            "runtime_inputs": [OBS_DRIVE_STATE, OBS_RAYS],
            "derived_payload": "tracked_agents",
            "tracking_assumption": "single_frame_lidar_endpoint_clusters",
            "velocity_assumption": "zero_velocity_no_identity_persistence",
            "noise_policy": "raw_range_noise_only_no_tracker_noise_model",
            "occlusion_policy": "visible_ray_endpoints_only_hidden_agents_unavailable",
            "fallback_or_degraded_success": False,
            "forbidden_runtime_inputs_not_read": [
                "robot",
                "pedestrians",
                "tracked_agents",
                "map",
                "static_obstacles",
            ],
            "track_count": self._last_track_count,
            "max_tracks": int(self.config.max_tracks),
            "normalized_observation": bool(self.config.normalized_observation),
        }


def build_lidar_tracked_social_force_config(
    cfg: dict[str, Any] | None,
) -> LidarTrackedSocialForceBuildConfig:
    """Build parsed LiDAR tracking and SocialForce configs.

    Returns:
        LidarTrackedSocialForceBuildConfig: Parsed adapter and wrapped planner config.
    """
    payload = cfg if isinstance(cfg, dict) else {}
    lidar_payload = payload.get("lidar_tracking")
    if not isinstance(lidar_payload, dict):
        lidar_payload = payload
    social_force_payload = payload.get("social_force")
    if not isinstance(social_force_payload, dict):
        social_force_payload = payload
    max_track_range = lidar_payload.get("max_track_range")
    allowed_social_force = {field.name for field in fields(SocNavPlannerConfig)}
    filtered_social_force = {
        key: value for key, value in social_force_payload.items() if key in allowed_social_force
    }
    return LidarTrackedSocialForceBuildConfig(
        lidar_tracking=LidarTrackedAgentsConfig(
            max_scan_dist=float(lidar_payload.get("max_scan_dist", 10.0)),
            visual_angle_portion=float(lidar_payload.get("visual_angle_portion", 1.0)),
            normalized_observation=bool(lidar_payload.get("normalized_observation", True)),
            target_distance_scale=float(
                lidar_payload.get("target_distance_scale", TARGET_DISTANCE_CAP_M)
            ),
            linear_speed_scale=float(lidar_payload.get("linear_speed_scale", 2.0)),
            angular_speed_scale=float(lidar_payload.get("angular_speed_scale", 1.0)),
            robot_radius=float(lidar_payload.get("robot_radius", 0.3)),
            track_radius=float(lidar_payload.get("track_radius", 0.3)),
            min_track_range=float(lidar_payload.get("min_track_range", 0.15)),
            max_track_range=(None if max_track_range is None else float(max_track_range)),
            cluster_gap_rays=int(lidar_payload.get("cluster_gap_rays", 1)),
            max_tracks=int(lidar_payload.get("max_tracks", 8)),
            timestep=float(lidar_payload.get("timestep", 0.1)),
        ),
        social_force=SocNavPlannerConfig(**filtered_social_force),
    )


def build_lidar_tracked_social_force_adapter(
    cfg: dict[str, Any] | None,
) -> LidarTrackedSocialForceAdapter:
    """Build the testing-only LiDAR tracked-agent SocialForce adapter.

    Returns:
        LidarTrackedSocialForceAdapter: Configured adapter instance.
    """
    parsed = build_lidar_tracked_social_force_config(cfg)
    return LidarTrackedSocialForceAdapter(
        config=parsed.lidar_tracking,
        social_force=SocialForcePlannerAdapter(config=parsed.social_force),
    )


__all__ = [
    "LidarTrackedAgentsConfig",
    "LidarTrackedSocialForceAdapter",
    "LidarTrackedSocialForceBuildConfig",
    "build_lidar_tracked_social_force_adapter",
    "build_lidar_tracked_social_force_config",
    "lidar_ray_angles",
    "lidar_rays_to_tracked_agents",
    "sensor_fusion_to_social_force_observation",
]
