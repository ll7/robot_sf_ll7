"""
Test suite for the ImageSensorFusion class and image-enabled sensor fusion.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.sensor.image_sensor import ImageSensor, ImageSensorSettings
from robot_sf.sensor.image_sensor_fusion import ImageSensorFusion
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_IMAGE, OBS_RAYS


class TestImageSensorFusion:
    """Test the ImageSensorFusion class."""

    @pytest.fixture
    def mock_sensors(self):
        """Create mock sensors for testing."""
        # Mock LiDAR sensor
        lidar_sensor = Mock(return_value=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # Mock robot speed sensor
        speed_sensor = Mock(return_value=(1.5, 0.2))  # PolarVec2D is just a tuple

        # Mock target sensor
        target_sensor = Mock(return_value=(2.0, 0.5, 0.1))

        # Mock image sensor
        image_sensor = Mock(spec=ImageSensor)
        image_sensor.capture_frame.return_value = np.random.rand(64, 64, 3).astype(np.float32)

        return lidar_sensor, speed_sensor, target_sensor, image_sensor

    @pytest.fixture
    def mock_obs_space(self):
        """Create mock observation space."""
        obs_space = {
            OBS_DRIVE_STATE: spaces.Box(
                low=np.array([[0, -1, 0, -np.pi, -np.pi]] * 3),
                high=np.array([[5, 1, 10, np.pi, np.pi]] * 3),
                dtype=np.float32,
            ),
            OBS_RAYS: spaces.Box(
                low=np.array([[0, 0, 0, 0, 0]] * 3),
                high=np.array([[10, 10, 10, 10, 10]] * 3),
                dtype=np.float32,
            ),
            OBS_IMAGE: spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32),
        }
        return spaces.Dict(obs_space)

    def test_initialization_with_image_enabled(self, mock_sensors, mock_obs_space):
        """Test ImageSensorFusion initialization with image observations enabled."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=True,
        )

        assert fusion.lidar_sensor == lidar_sensor
        assert fusion.robot_speed_sensor == speed_sensor
        assert fusion.target_sensor == target_sensor
        assert fusion.image_sensor == image_sensor
        assert fusion.use_image_obs is True
        assert fusion.cache_steps == 3  # Based on mock obs space

    def test_initialization_with_image_disabled(self, mock_sensors, mock_obs_space):
        """Test ImageSensorFusion initialization with image observations disabled."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=False,
        )

        assert fusion.use_image_obs is False
        # Should still initialize without errors

    def test_next_obs_with_image_enabled(self, mock_sensors, mock_obs_space):
        """Test next_obs method with image observations enabled."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=True,
        )

        # Get observation
        obs = fusion.next_obs()

        # Verify observation structure
        assert isinstance(obs, dict)
        assert OBS_DRIVE_STATE in obs
        assert OBS_RAYS in obs
        assert OBS_IMAGE in obs

        # Verify shapes
        assert obs[OBS_DRIVE_STATE].shape == (3, 5)  # cache_steps x drive_state_size
        assert obs[OBS_RAYS].shape == (3, 5)  # cache_steps x lidar_size
        assert obs[OBS_IMAGE].shape == (64, 64, 3)  # image shape

        # Verify sensors were called
        lidar_sensor.assert_called()
        speed_sensor.assert_called()
        target_sensor.assert_called()
        image_sensor.capture_frame.assert_called()

    def test_next_obs_with_image_disabled(self, mock_sensors, mock_obs_space):
        """Test next_obs method with image observations disabled."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=False,
        )

        # Get observation
        obs = fusion.next_obs()

        # Verify observation structure
        assert isinstance(obs, dict)
        assert OBS_DRIVE_STATE in obs
        assert OBS_RAYS in obs
        assert OBS_IMAGE not in obs  # Should not include image

        # Verify image sensor was not called
        image_sensor.capture_frame.assert_not_called()

    def test_next_obs_without_image_sensor(self, mock_sensors, mock_obs_space):
        """Test next_obs method when image sensor is None."""
        lidar_sensor, speed_sensor, target_sensor, _ = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=None,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=True,
        )

        # Get observation
        obs = fusion.next_obs()

        # Verify observation structure
        assert isinstance(obs, dict)
        assert OBS_DRIVE_STATE in obs
        assert OBS_RAYS in obs
        assert OBS_IMAGE not in obs  # Should not include image when sensor is None

    def test_multiple_observations_consistency(self, mock_sensors, mock_obs_space):
        """Test that multiple calls to next_obs maintain consistency."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        # Make sensors return different values each call
        lidar_sensor.side_effect = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
            np.array([1.2, 2.2, 3.2, 4.2, 5.2]),
        ]

        speed_sensor.side_effect = [
            (1.0, 0.1),  # PolarVec2D is just a tuple
            (1.1, 0.2),
            (1.2, 0.3),
        ]

        target_sensor.side_effect = [(2.0, 0.5, 0.1), (2.1, 0.6, 0.2), (2.2, 0.7, 0.3)]

        image_sensor.capture_frame.side_effect = [
            np.random.rand(64, 64, 3).astype(np.float32),
            np.random.rand(64, 64, 3).astype(np.float32),
            np.random.rand(64, 64, 3).astype(np.float32),
        ]

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=True,
        )

        # Get multiple observations
        obs1 = fusion.next_obs()
        obs2 = fusion.next_obs()
        obs3 = fusion.next_obs()

        # Verify all observations have the same structure
        for obs in [obs1, obs2, obs3]:
            assert OBS_DRIVE_STATE in obs
            assert OBS_RAYS in obs
            assert OBS_IMAGE in obs
            assert obs[OBS_DRIVE_STATE].shape == (3, 5)
            assert obs[OBS_RAYS].shape == (3, 5)
            assert obs[OBS_IMAGE].shape == (64, 64, 3)

    def test_reset_cache_with_image_enabled(self, mock_sensors, mock_obs_space):
        """Test cache reset functionality with image observations."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=True,
        )

        # Get an observation to populate caches
        _ = fusion.next_obs()

        # Verify caches have data
        assert len(fusion.drive_state_cache) > 0
        assert len(fusion.lidar_state_cache) > 0
        assert len(fusion.image_state_cache) > 0

        # Reset cache
        fusion.reset_cache()

        # Verify caches are empty
        assert len(fusion.drive_state_cache) == 0
        assert len(fusion.lidar_state_cache) == 0
        assert len(fusion.image_state_cache) == 0

    def test_reset_cache_with_image_disabled(self, mock_sensors, mock_obs_space):
        """Test cache reset functionality with image observations disabled."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=False,
        )

        # Get an observation to populate caches
        _ = fusion.next_obs()

        # Reset cache should work even without image cache
        fusion.reset_cache()

        # Verify basic caches are empty
        assert len(fusion.drive_state_cache) == 0
        assert len(fusion.lidar_state_cache) == 0

    def test_normalization_consistency(self, mock_sensors, mock_obs_space):
        """Test that observations are properly normalized."""
        lidar_sensor, speed_sensor, target_sensor, image_sensor = mock_sensors

        fusion = ImageSensorFusion(
            lidar_sensor=lidar_sensor,
            robot_speed_sensor=speed_sensor,
            target_sensor=target_sensor,
            image_sensor=image_sensor,
            unnormed_obs_space=mock_obs_space,
            use_next_goal=True,
            use_image_obs=True,
        )

        obs = fusion.next_obs()

        # Verify normalization bounds for drive state and lidar
        # (These are normalized by the max values from the observation space)
        assert np.all(obs[OBS_DRIVE_STATE] >= 0.0)
        assert np.all(obs[OBS_DRIVE_STATE] <= 1.0)
        assert np.all(obs[OBS_RAYS] >= 0.0)
        assert np.all(obs[OBS_RAYS] <= 1.0)

        # Image should already be normalized in the sensor
        assert np.all(obs[OBS_IMAGE] >= 0.0)
        assert np.all(obs[OBS_IMAGE] <= 1.0)


class TestImageSensorFusionIntegration:
    """Integration tests for ImageSensorFusion with real components."""

    def test_integration_with_real_image_sensor(self):
        """Test ImageSensorFusion with real ImageSensor."""
        import pygame

        from robot_sf.nav.map_config import MapDefinitionPool
        from robot_sf.render.sim_view import SimulationView

        # Initialize pygame for the test
        pygame.init()

        try:
            # Create a proper map definition using MapDefinitionPool
            map_pool = MapDefinitionPool()
            map_def = map_pool.choose_random_map()

            # Create a real SimulationView
            sim_view = SimulationView(
                width=100,
                height=75,
                record_video=True,  # Use offscreen surface
                map_def=map_def,
            )

            # Create real ImageSensor
            settings = ImageSensorSettings(width=32, height=32, normalize=True)
            image_sensor = ImageSensor(settings, sim_view)

            # Create mock other sensors
            lidar_sensor = Mock(return_value=np.array([1.0, 2.0, 3.0]))
            speed_sensor = Mock(return_value=(1.5, 0.2))  # PolarVec2D is just a tuple
            target_sensor = Mock(return_value=(2.0, 0.5, 0.1))

            # Create observation space
            obs_space = spaces.Dict(
                {
                    OBS_DRIVE_STATE: spaces.Box(
                        low=np.array([[0, -1, 0, -np.pi, -np.pi]] * 2),
                        high=np.array([[5, 1, 10, np.pi, np.pi]] * 2),
                        dtype=np.float32,
                    ),
                    OBS_RAYS: spaces.Box(
                        low=np.array([[0, 0, 0]] * 2),
                        high=np.array([[10, 10, 10]] * 2),
                        dtype=np.float32,
                    ),
                    OBS_IMAGE: spaces.Box(low=0.0, high=1.0, shape=(32, 32, 3), dtype=np.float32),
                },
            )

            # Create fusion
            fusion = ImageSensorFusion(
                lidar_sensor=lidar_sensor,
                robot_speed_sensor=speed_sensor,
                target_sensor=target_sensor,
                image_sensor=image_sensor,
                unnormed_obs_space=obs_space,
                use_next_goal=True,
                use_image_obs=True,
            )

            # Fill screen with test pattern
            sim_view.screen.fill((100, 150, 200))

            # Get observation
            obs = fusion.next_obs()

            # Verify structure
            assert OBS_DRIVE_STATE in obs
            assert OBS_RAYS in obs
            assert OBS_IMAGE in obs

            # Verify image properties
            assert obs[OBS_IMAGE].shape == (32, 32, 3)
            assert obs[OBS_IMAGE].dtype == np.float32
            assert 0.0 <= obs[OBS_IMAGE].min() <= obs[OBS_IMAGE].max() <= 1.0

            # Clean up
            sim_view.exit_simulation()

        finally:
            pygame.quit()


class RobotEnvWithImage(RobotEnv):
    """
    Robot environment with image observations enabled.
    Extends the base RobotEnv to include image observations when requested.
    """

    def __init__(self, env_config, debug=False):
        """Init.

        Args:
            env_config: Auto-generated placeholder description.
            debug: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        super().__init__(env_config, debug)
        self.sim_ui = None if not hasattr(self, "sim_view") else self.sim_view
