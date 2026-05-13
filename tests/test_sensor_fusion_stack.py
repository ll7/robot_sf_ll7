"""Tests for sensor fusion stacking semantics."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from robot_sf.sensor.image_sensor_fusion import ImageSensorFusion
from robot_sf.sensor.sensor_fusion import (
    OBS_DRIVE_STATE,
    OBS_RAYS,
    SensorFusion,
    fused_sensor_space,
)


def test_fused_sensor_space_stack_shapes() -> None:
    """Stack timesteps so downstream policies see consistent history shapes."""
    robot_obs = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    target_obs = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    lidar_obs = spaces.Box(low=0.0, high=10.0, shape=(4,), dtype=np.float32)
    timesteps = 3

    norm_space, orig_space = fused_sensor_space(timesteps, robot_obs, target_obs, lidar_obs)
    assert orig_space[OBS_DRIVE_STATE].shape == (timesteps, 5)
    assert orig_space[OBS_RAYS].shape == (timesteps, 4)
    assert norm_space[OBS_DRIVE_STATE].shape == (timesteps, 5)
    assert norm_space[OBS_RAYS].shape == (timesteps, 4)


def test_sensor_fusion_stacks_history() -> None:
    """Stack history consistently so observation normalization stays deterministic."""
    robot_obs = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    target_obs = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    lidar_obs = spaces.Box(low=0.0, high=10.0, shape=(4,), dtype=np.float32)
    timesteps = 3

    _norm_space, orig_space = fused_sensor_space(timesteps, robot_obs, target_obs, lidar_obs)
    fusion = SensorFusion(
        lidar_sensor=lambda: np.ones(4, dtype=np.float32),
        robot_speed_sensor=lambda: (1.0, 0.0),
        target_sensor=lambda: (2.0, 0.1, 0.2),
        unnormed_obs_space=orig_space,
        use_next_goal=True,
    )

    obs = fusion.next_obs()
    assert obs[OBS_DRIVE_STATE].shape == (timesteps, 5)
    assert obs[OBS_RAYS].shape == (timesteps, 4)

    drive_state = np.array([1.0, 0.0, 2.0, 0.1, 0.2], dtype=np.float32)
    expected_row = drive_state / orig_space[OBS_DRIVE_STATE].high[0]
    for row in obs[OBS_DRIVE_STATE]:
        assert np.allclose(row, expected_row)


def test_sensor_fusion_history_is_oldest_to_newest() -> None:
    """Keep current observations at the last temporal index for conv extractors."""
    robot_obs = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
    target_obs = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
    lidar_obs = spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)
    timesteps = 3

    _norm_space, orig_space = fused_sensor_space(timesteps, robot_obs, target_obs, lidar_obs)
    lidar_samples = iter(
        [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([2.0, 2.0], dtype=np.float32),
        ]
    )
    speed_samples = iter([(1.0, 0.0), (2.0, 0.0)])
    target_samples = iter([(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    fusion = SensorFusion(
        lidar_sensor=lambda: next(lidar_samples),
        robot_speed_sensor=lambda: next(speed_samples),
        target_sensor=lambda: next(target_samples),
        unnormed_obs_space=orig_space,
        use_next_goal=True,
    )

    fusion.next_obs()
    obs = fusion.next_obs()

    assert np.allclose(obs[OBS_RAYS], [[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]])
    assert np.allclose(
        obs[OBS_DRIVE_STATE],
        [
            [0.1, 0.0, 0.1, 0.0, 0.0],
            [0.1, 0.0, 0.1, 0.0, 0.0],
            [0.2, 0.0, 0.2, 0.0, 0.0],
        ],
    )


def test_image_sensor_fusion_first_observation_prefills_history() -> None:
    """ImageSensorFusion should prefill temporal history on the first observation."""
    robot_obs = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
    target_obs = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
    lidar_obs = spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)
    timesteps = 3

    _norm_space, orig_space = fused_sensor_space(timesteps, robot_obs, target_obs, lidar_obs)
    fusion = ImageSensorFusion(
        lidar_sensor=lambda: np.array([1.0, 1.0], dtype=np.float32),
        robot_speed_sensor=lambda: (2.0, 0.0),
        target_sensor=lambda: (3.0, 0.0, 0.0),
        image_sensor=None,
        unnormed_obs_space=orig_space,
        use_next_goal=True,
        use_image_obs=False,
    )

    obs = fusion.next_obs()

    assert np.allclose(obs[OBS_RAYS], [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    assert np.allclose(
        obs[OBS_DRIVE_STATE],
        [
            [0.2, 0.0, 0.3, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.0, 0.0],
        ],
    )
