from typing import Tuple, Callable, List, Dict
from dataclasses import dataclass, field

import numpy as np
from gym import spaces

PolarVec2D = Tuple[float, float]

OBS_DRIVE_STATE = "drive_state"
OBS_RAYS = "rays"


def fused_sensor_space(
        timesteps: int, robot_obs: spaces.Box,
        target_obs: spaces.Box, lidar_obs: spaces.Box
    ) -> Tuple[spaces.Dict, spaces.Dict]:
    max_drive_state = np.array([
        robot_obs.high.tolist() + target_obs.high.tolist()
        for t in range(timesteps)], dtype=np.float32)
    min_drive_state = np.array([
        robot_obs.low.tolist() + target_obs.low.tolist()
        for t in range(timesteps)], dtype=np.float32)
    max_lidar_state = np.array([lidar_obs.high.tolist() for t in range(timesteps)], dtype=np.float32)
    min_lidar_state = np.array([lidar_obs.low.tolist() for t in range(timesteps)], dtype=np.float32)

    orig_box_drive_state = spaces.Box(low=min_drive_state, high=max_drive_state, dtype=np.float32)
    orig_box_lidar_state = spaces.Box(low=min_lidar_state, high=max_lidar_state, dtype=np.float32)
    orig_obs_space = spaces.Dict({ OBS_DRIVE_STATE: orig_box_drive_state, OBS_RAYS: orig_box_lidar_state })

    box_drive_state = spaces.Box(
        low=min_drive_state / max_drive_state,
        high=max_drive_state / max_drive_state,
        dtype=np.float32)
    box_lidar_state = spaces.Box(
        low=min_lidar_state / max_lidar_state,
        high=max_lidar_state / max_lidar_state,
        dtype=np.float32)
    norm_obs_space = spaces.Dict({ OBS_DRIVE_STATE: box_drive_state, OBS_RAYS: box_lidar_state })

    return norm_obs_space, orig_obs_space


@dataclass
class SensorFusion:
    lidar_sensor: Callable[[], np.ndarray]
    robot_speed_sensor: Callable[[], PolarVec2D]
    target_sensor: Callable[[], Tuple[float, float, float]]
    unnormed_obs_space: spaces.Dict
    drive_state_cache: List[np.ndarray] = field(init=False, default_factory=list)
    lidar_state_cache: List[np.ndarray] = field(init=False, default_factory=list)
    cache_steps: int = field(init=False)

    def __post_init__(self):
        self.cache_steps = self.unnormed_obs_space[OBS_RAYS].shape[0]

    def next_obs(self) -> Dict[str, np.ndarray]:
        lidar_state = self.lidar_sensor()
        # TODO: append beginning at the end for conv feature extractor

        speed_x, speed_rot = self.robot_speed_sensor()
        target_distance, target_angle, next_target_angle = self.target_sensor()
        drive_state = np.array([speed_x, speed_rot, target_distance, target_angle, next_target_angle])

        # info: populate cache with same states -> no movement
        if len(self.drive_state_cache) == 0:
            for _ in range(self.cache_steps):
                self.drive_state_cache.append(drive_state)
                self.lidar_state_cache.append(lidar_state)

        self.drive_state_cache.append(drive_state)
        self.lidar_state_cache.append(lidar_state)
        self.drive_state_cache.pop(0)
        self.lidar_state_cache.pop(0)

        stacked_drive_state = np.array(self.drive_state_cache, dtype=np.float32)
        stacked_lidar_state = np.array(self.lidar_state_cache, dtype=np.float32)

        max_drive = self.unnormed_obs_space[OBS_DRIVE_STATE].high
        max_lidar = self.unnormed_obs_space[OBS_RAYS].high
        return { OBS_DRIVE_STATE: stacked_drive_state / max_drive,
                 OBS_RAYS: stacked_lidar_state / max_lidar }

    def reset_cache(self):
        self.drive_state_cache.clear()
        self.lidar_state_cache.clear()
