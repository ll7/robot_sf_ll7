"""
Test suite for the extended sensor fusion space functionality with image observations.
"""

from typing import cast

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.sensor.image_sensor import ImageSensorSettings, image_sensor_space
from robot_sf.sensor.sensor_fusion import (
    OBS_DRIVE_STATE,
    OBS_IMAGE,
    OBS_RAYS,
    fused_sensor_space,
    fused_sensor_space_with_image,
)


class TestFusedSensorSpaceWithImage:
    """Test the fused_sensor_space_with_image function."""

    @pytest.fixture
    def basic_spaces(self):
        """Create basic observation spaces for testing."""
        robot_obs = spaces.Box(low=np.array([0, -1]), high=np.array([5, 1]), dtype=np.float32)
        target_obs = spaces.Box(
            low=np.array([0, -np.pi, -np.pi]),
            high=np.array([10, np.pi, np.pi]),
            dtype=np.float32,
        )
        lidar_obs = spaces.Box(low=np.zeros(4), high=np.full(4, 10.0), dtype=np.float32)
        return robot_obs, target_obs, lidar_obs

    def test_fused_space_without_image(self, basic_spaces):
        """Test fused sensor space creation without image observations."""
        robot_obs, target_obs, lidar_obs = basic_spaces
        timesteps = 3

        norm_space, orig_space = fused_sensor_space_with_image(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=None,
        )

        # Should be identical to regular fused_sensor_space
        expected_norm, expected_orig = fused_sensor_space(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
        )

        assert norm_space.spaces.keys() == expected_norm.spaces.keys()
        assert orig_space.spaces.keys() == expected_orig.spaces.keys()
        assert OBS_IMAGE not in norm_space.spaces
        assert OBS_IMAGE not in orig_space.spaces

    def test_fused_space_with_image(self, basic_spaces):
        """Test fused sensor space creation with image observations."""
        robot_obs, target_obs, lidar_obs = basic_spaces
        timesteps = 3

        # Create image observation space
        image_settings = ImageSensorSettings(width=64, height=64, normalize=True)
        image_obs = image_sensor_space(image_settings)

        norm_space, orig_space = fused_sensor_space_with_image(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=image_obs,
        )

        # Should contain all three observation types
        assert OBS_DRIVE_STATE in norm_space.spaces
        assert OBS_RAYS in norm_space.spaces
        assert OBS_IMAGE in norm_space.spaces

        assert OBS_DRIVE_STATE in orig_space.spaces
        assert OBS_RAYS in orig_space.spaces
        assert OBS_IMAGE in orig_space.spaces

        # Image spaces should be identical in both normalized and original
        assert norm_space.spaces[OBS_IMAGE] == image_obs
        assert orig_space.spaces[OBS_IMAGE] == image_obs

    def test_image_space_properties(self, basic_spaces):
        """Test that image space properties are preserved correctly."""
        robot_obs, target_obs, lidar_obs = basic_spaces
        timesteps = 2

        # Test different image configurations
        test_configs = [
            {"width": 32, "height": 32, "normalize": True, "grayscale": False},
            {"width": 84, "height": 84, "normalize": False, "grayscale": False},
            {"width": 48, "height": 64, "normalize": True, "grayscale": True},
        ]

        for config in test_configs:
            image_settings = ImageSensorSettings(**config)
            image_obs = image_sensor_space(image_settings)

            norm_space, orig_space = fused_sensor_space_with_image(
                timesteps,
                robot_obs,
                target_obs,
                lidar_obs,
                image_obs=image_obs,
            )

            # Verify image space is preserved
            norm_image_space = norm_space.spaces[OBS_IMAGE]
            orig_image_space = orig_space.spaces[OBS_IMAGE]

            assert norm_image_space == image_obs
            assert orig_image_space == image_obs

            # Verify image space properties
            if config["grayscale"]:
                expected_shape = (config["height"], config["width"])
            else:
                expected_shape = (config["height"], config["width"], 3)

            assert norm_image_space.shape == expected_shape
            assert orig_image_space.shape == expected_shape

    def test_drive_state_and_lidar_consistency(self, basic_spaces):
        """Test that drive state and LiDAR spaces remain consistent."""
        robot_obs, target_obs, lidar_obs = basic_spaces
        timesteps = 4

        # Create image space
        image_settings = ImageSensorSettings()
        image_obs = image_sensor_space(image_settings)

        # Get spaces with image
        norm_space_img, _orig_space_img = fused_sensor_space_with_image(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=image_obs,
        )

        # Get spaces without image for comparison
        norm_space_no_img, _orig_space_no_img = fused_sensor_space(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
        )

        # Drive state and LiDAR spaces should be identical
        drive_img = cast(spaces.Box, norm_space_img.spaces[OBS_DRIVE_STATE])
        drive_no_img = cast(spaces.Box, norm_space_no_img.spaces[OBS_DRIVE_STATE])
        rays_img = cast(spaces.Box, norm_space_img.spaces[OBS_RAYS])
        rays_no_img = cast(spaces.Box, norm_space_no_img.spaces[OBS_RAYS])

        assert np.array_equal(drive_img.low, drive_no_img.low)
        assert np.array_equal(drive_img.high, drive_no_img.high)
        assert np.array_equal(rays_img.low, rays_no_img.low)
        assert np.array_equal(rays_img.high, rays_no_img.high)

    def test_return_types(self, basic_spaces):
        """Test that return types are correct."""
        robot_obs, target_obs, lidar_obs = basic_spaces
        timesteps = 2

        # Test without image
        norm_space, orig_space = fused_sensor_space_with_image(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=None,
        )

        assert isinstance(norm_space, spaces.Dict)
        assert isinstance(orig_space, spaces.Dict)

        # Test with image
        image_settings = ImageSensorSettings()
        image_obs = image_sensor_space(image_settings)

        norm_space, orig_space = fused_sensor_space_with_image(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=image_obs,
        )

        assert isinstance(norm_space, spaces.Dict)
        assert isinstance(orig_space, spaces.Dict)

    def test_timesteps_parameter(self, basic_spaces):
        """Test that timesteps parameter affects stacked observations correctly."""
        robot_obs, target_obs, lidar_obs = basic_spaces
        image_settings = ImageSensorSettings()
        image_obs = image_sensor_space(image_settings)

        # Test different timestep values
        for timesteps in [1, 2, 3, 5]:
            norm_space, _orig_space = fused_sensor_space_with_image(
                timesteps,
                robot_obs,
                target_obs,
                lidar_obs,
                image_obs=image_obs,
            )

            # Drive state should be stacked according to timesteps
            drive_shape = cast(spaces.Box, norm_space.spaces[OBS_DRIVE_STATE]).shape
            expected_drive_features = len(robot_obs.low) + len(target_obs.low)  # 2 + 3 = 5
            assert drive_shape == (timesteps, expected_drive_features)

            # LiDAR should be stacked according to timesteps
            lidar_shape = cast(spaces.Box, norm_space.spaces[OBS_RAYS]).shape
            expected_lidar_features = len(lidar_obs.low)  # 4
            assert lidar_shape == (timesteps, expected_lidar_features)

            # Image should not be affected by timesteps
            image_shape = cast(spaces.Box, norm_space.spaces[OBS_IMAGE]).shape
            assert image_shape == image_obs.shape

    def test_edge_cases(self, basic_spaces):
        """Test edge cases and error conditions."""
        robot_obs, target_obs, lidar_obs = basic_spaces

        # Test with timesteps = 1
        image_settings = ImageSensorSettings()
        image_obs = image_sensor_space(image_settings)

        norm_space, _orig_space = fused_sensor_space_with_image(
            1,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=image_obs,
        )

        assert cast(spaces.Box, norm_space.spaces[OBS_DRIVE_STATE]).shape[0] == 1
        assert cast(spaces.Box, norm_space.spaces[OBS_RAYS]).shape[0] == 1
        assert OBS_IMAGE in norm_space.spaces


class TestImageObservationSpaceIntegration:
    """Integration tests for image observation spaces."""

    def test_space_consistency_with_sensor_output(self):
        """Test that observation spaces match actual sensor outputs."""

        # Test various configurations
        configs = [
            {"width": 64, "height": 64, "normalize": True, "grayscale": False},
            {"width": 32, "height": 48, "normalize": False, "grayscale": True},
            {"width": 128, "height": 96, "normalize": True, "grayscale": True},
        ]

        for config in configs:
            settings = ImageSensorSettings(**config)
            space = image_sensor_space(settings)

            # Create a mock sensor output
            if config["grayscale"]:
                if config["normalize"]:
                    mock_output = np.random.rand(config["height"], config["width"]).astype(
                        np.float32,
                    )
                else:
                    mock_output = np.random.randint(
                        0,
                        256,
                        (config["height"], config["width"]),
                        dtype=np.uint8,
                    )
            elif config["normalize"]:
                mock_output = np.random.rand(config["height"], config["width"], 3).astype(
                    np.float32,
                )
            else:
                mock_output = np.random.randint(
                    0,
                    256,
                    (config["height"], config["width"], 3),
                    dtype=np.uint8,
                )

            # Verify the space contains the mock output
            assert space.contains(mock_output), (
                f"Space doesn't contain mock output for config {config}"
            )

    def test_complete_observation_space_structure(self):
        """Test complete observation space structure with all components."""
        # Create realistic robot and sensor spaces
        robot_obs = spaces.Box(
            low=np.array([0, -2]),  # speed_x, speed_rot
            high=np.array([3, 2]),
            dtype=np.float32,
        )
        target_obs = spaces.Box(
            low=np.array([0, -np.pi, -np.pi]),  # distance, angle, next_angle
            high=np.array([20, np.pi, np.pi]),
            dtype=np.float32,
        )
        lidar_obs = spaces.Box(
            low=np.zeros(16),  # 16 LiDAR rays
            high=np.full(16, 10.0),
            dtype=np.float32,
        )

        # Create image space
        image_settings = ImageSensorSettings(width=84, height=84, normalize=True)
        image_obs = image_sensor_space(image_settings)

        # Create fused space
        timesteps = 3
        norm_space, _orig_space = fused_sensor_space_with_image(
            timesteps,
            robot_obs,
            target_obs,
            lidar_obs,
            image_obs=image_obs,
        )

        # Create mock complete observation
        mock_obs = {
            OBS_DRIVE_STATE: np.random.rand(3, 5).astype(np.float32),  # 3 timesteps, 5 features
            OBS_RAYS: np.random.rand(3, 16).astype(np.float32),  # 3 timesteps, 16 rays
            OBS_IMAGE: np.random.rand(84, 84, 3).astype(np.float32),  # Single image
        }

        # Verify the complete observation is valid
        assert norm_space.contains(mock_obs)

        # Test individual components
        assert norm_space.spaces[OBS_DRIVE_STATE].contains(mock_obs[OBS_DRIVE_STATE])
        assert norm_space.spaces[OBS_RAYS].contains(mock_obs[OBS_RAYS])
        assert norm_space.spaces[OBS_IMAGE].contains(mock_obs[OBS_IMAGE])
