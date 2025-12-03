"""
Test suite for environment configuration and integration with image observations.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.gym_env.env_util import (
    create_spaces_with_image,
    init_collision_and_sensors_with_image,
)
from robot_sf.sensor.image_sensor import ImageSensorSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_IMAGE, OBS_RAYS


class TestRobotEnvSettingsWithImage:
    """Test RobotEnvSettings with image observation configuration."""

    def test_default_image_settings(self):
        """Test default RobotEnvSettings without image observations."""
        settings = RobotEnvSettings()

        assert hasattr(settings, "use_image_obs")
        assert settings.use_image_obs is False

    def test_enable_image_observations(self):
        """Test enabling image observations in RobotEnvSettings."""
        settings = RobotEnvSettings(use_image_obs=True)

        assert settings.use_image_obs is True
        # Should automatically create image_config during post_init
        assert hasattr(settings, "image_config")
        assert isinstance(settings.image_config, ImageSensorSettings)

    def test_custom_image_config(self):
        """Test custom image configuration."""
        custom_image_config = ImageSensorSettings(
            width=128,
            height=96,
            grayscale=True,
            normalize=False,
        )

        settings = RobotEnvSettings(use_image_obs=True, image_config=custom_image_config)

        assert settings.use_image_obs is True
        assert settings.image_config == custom_image_config
        assert settings.image_config.width == 128
        assert settings.image_config.height == 96
        assert settings.image_config.grayscale is True
        assert settings.image_config.normalize is False

    def test_image_config_validation(self):
        """Test that image configuration is properly validated."""
        settings = RobotEnvSettings()

        # Should not have image_config when image observations are disabled
        assert (
            not hasattr(settings, "image_config")
            or settings.image_config is None
            or settings.use_image_obs is False
        )


class TestCreateSpacesWithImage:
    """Test the create_spaces_with_image function."""

    @patch("robot_sf.gym_env.env_util.target_sensor_space")
    @patch("robot_sf.gym_env.env_util.lidar_sensor_space")
    def test_create_spaces_without_image(self, mock_lidar_space, mock_target_space):
        """Test creating spaces without image observations."""
        # Setup mocks
        mock_target_space.return_value = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
        mock_lidar_space.return_value = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        # Create settings without image
        settings = RobotEnvSettings(use_image_obs=False)

        from robot_sf.nav.map_config import MapDefinitionPool

        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        _action_space, obs_space, orig_obs_space = create_spaces_with_image(settings, map_def)

        # Should not contain image observations
        assert OBS_IMAGE not in obs_space.spaces
        assert OBS_IMAGE not in orig_obs_space.spaces
        assert OBS_DRIVE_STATE in obs_space.spaces
        assert OBS_RAYS in obs_space.spaces

    @patch("robot_sf.gym_env.env_util.target_sensor_space")
    @patch("robot_sf.gym_env.env_util.lidar_sensor_space")
    def test_create_spaces_with_image(self, mock_lidar_space, mock_target_space):
        """Test creating spaces with image observations."""
        # Setup mocks
        mock_target_space.return_value = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
        mock_lidar_space.return_value = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        # Create settings with image
        image_config = ImageSensorSettings(width=64, height=64)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        from robot_sf.nav.map_config import MapDefinitionPool

        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        _action_space, obs_space, _orig_obs_space = create_spaces_with_image(settings, map_def)

        # Should contain all observation types
        assert OBS_DRIVE_STATE in obs_space.spaces
        assert OBS_RAYS in obs_space.spaces
        assert OBS_IMAGE in obs_space.spaces

        # Verify image space properties
        image_space = obs_space.spaces[OBS_IMAGE]
        assert image_space.shape == (64, 64, 3)  # width, height, channels

    def test_action_space_consistency(self):
        """Test that action space remains consistent regardless of image observations."""
        from robot_sf.nav.map_config import MapDefinitionPool

        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        # Create settings with and without image
        settings_no_image = RobotEnvSettings(use_image_obs=False)
        settings_with_image = RobotEnvSettings(use_image_obs=True)

        with (
            patch("robot_sf.gym_env.env_util.target_sensor_space") as mock_target,
            patch("robot_sf.gym_env.env_util.lidar_sensor_space") as mock_lidar,
        ):
            mock_target.return_value = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
            mock_lidar.return_value = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)

            action_space_no_image, _, _ = create_spaces_with_image(settings_no_image, map_def)
            action_space_with_image, _, _ = create_spaces_with_image(settings_with_image, map_def)

            # Action spaces should be identical
            assert action_space_no_image == action_space_with_image


class TestInitCollisionAndSensorsWithImage:
    """Test the init_collision_and_sensors_with_image function."""

    @pytest.fixture
    def mock_simulator(self):
        """Create a mock simulator for testing."""
        import numpy as np

        from robot_sf.nav.map_config import MapDefinitionPool
        from robot_sf.sim.simulator import Simulator

        simulator = Mock(spec=Simulator)
        # Create mock robot with proper pose attribute
        mock_robot = Mock()
        mock_robot.pose = ((0.0, 0.0), 0.0)  # ((x, y), orientation)
        mock_robot.current_speed = 0.0
        simulator.robots = [mock_robot]  # One robot
        simulator.robot_pos = [(0, 0)]
        simulator.goal_pos = [(5, 5)]
        simulator.next_goal_pos = [(10, 10)]
        map_pool = MapDefinitionPool()
        simulator.map_def = map_pool.choose_random_map()

        # Create mock pysf_sim with env.obstacles_raw
        mock_pysf_sim = Mock()
        mock_env = Mock()
        mock_env.obstacles_raw = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])  # Multiple obstacles
        mock_pysf_sim.env = mock_env
        simulator.pysf_sim = mock_pysf_sim

        simulator.ped_pos = np.array([[1.0, 1.0], [3.0, 3.0]])  # Some pedestrian positions

        return simulator

    @pytest.fixture
    def mock_obs_space(self):
        """Create mock observation space."""
        import numpy as np

        return spaces.Dict(
            {
                OBS_DRIVE_STATE: spaces.Box(low=-1, high=1, shape=(2, 5), dtype=np.float32),
                OBS_RAYS: spaces.Box(low=0, high=10, shape=(2, 4), dtype=np.float32),
                OBS_IMAGE: spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32),
            },
        )

    def test_init_without_image_observations(self, mock_simulator, mock_obs_space):
        """Test initialization without image observations."""
        settings = RobotEnvSettings(use_image_obs=False)

        occupancies, sensors = init_collision_and_sensors_with_image(
            mock_simulator,
            settings,
            mock_obs_space,
            sim_view=None,
        )

        assert len(occupancies) == 1  # One robot
        assert len(sensors) == 1  # One sensor fusion

        # Should be regular SensorFusion, not ImageSensorFusion
        sensor = sensors[0]
        assert hasattr(sensor, "lidar_sensor")
        assert hasattr(sensor, "robot_speed_sensor")
        assert hasattr(sensor, "target_sensor")

    @patch("robot_sf.sensor.image_sensor.ImageSensor")
    @patch("robot_sf.gym_env.env_util.ImageSensorFusion")
    def test_init_with_image_observations(
        self,
        mock_image_fusion,
        mock_image_sensor,
        mock_simulator,
        mock_obs_space,
    ):
        """Test initialization with image observations."""
        # Setup mocks
        mock_image_sensor_instance = Mock()
        mock_image_sensor.return_value = mock_image_sensor_instance
        mock_image_fusion_instance = Mock()
        mock_image_fusion.return_value = mock_image_fusion_instance

        # Create mock sim_view
        mock_sim_view = Mock()

        settings = RobotEnvSettings(use_image_obs=True)

        occupancies, sensors = init_collision_and_sensors_with_image(
            mock_simulator,
            settings,
            mock_obs_space,
            sim_view=mock_sim_view,
        )

        assert len(occupancies) == 1
        assert len(sensors) == 1

        # Should have created ImageSensor and ImageSensorFusion
        mock_image_sensor.assert_called_once()
        mock_image_fusion.assert_called_once()

    def test_init_with_image_but_no_sim_view(self, mock_simulator, mock_obs_space):
        """Test initialization with image observations but no sim_view."""
        settings = RobotEnvSettings(use_image_obs=True)

        occupancies, sensors = init_collision_and_sensors_with_image(
            mock_simulator,
            settings,
            mock_obs_space,
            sim_view=None,
        )

        # Should fall back to regular sensor fusion
        assert len(occupancies) == 1
        assert len(sensors) == 1


class TestImageObservationIntegration:
    """Integration tests for image observations in the environment."""

    def test_environment_config_consistency(self):
        """Test that environment configuration is consistent across different settings."""
        configs_to_test = [
            {"use_image_obs": False},
            {"use_image_obs": True},
            {"use_image_obs": True, "image_config": ImageSensorSettings(width=32, height=32)},
            {"use_image_obs": True, "image_config": ImageSensorSettings(grayscale=True)},
        ]

        for config in configs_to_test:
            settings = RobotEnvSettings(**config)

            # Should not raise any errors during initialization
            assert settings is not None
            assert hasattr(settings, "use_image_obs")

            if settings.use_image_obs:
                assert hasattr(settings, "image_config")
                assert isinstance(settings.image_config, ImageSensorSettings)

    def test_observation_space_sample_validity(self):
        """Test that sampled observations from image-enabled spaces are valid."""
        import numpy as np

        from robot_sf.nav.map_config import MapDefinitionPool

        # Create settings for testing
        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        with (
            patch("robot_sf.gym_env.env_util.target_sensor_space") as mock_target,
            patch("robot_sf.gym_env.env_util.lidar_sensor_space") as mock_lidar,
        ):
            mock_target.return_value = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
            mock_lidar.return_value = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)

            # Use RobotEnvSettings which properly supports image observations
            env_settings = RobotEnvSettings(use_image_obs=True)
            _action_space, obs_space, _orig_obs_space = create_spaces_with_image(
                env_settings,
                map_def,
            )

            # Sample from observation space
            sample_obs = obs_space.sample()

            # Verify structure
            assert isinstance(sample_obs, dict)
            assert OBS_DRIVE_STATE in sample_obs
            assert OBS_RAYS in sample_obs
            assert OBS_IMAGE in sample_obs

            # Verify the observation is valid
            assert obs_space.contains(sample_obs)

    def test_multiple_robots_support(self):
        """Test that image observations work with multiple robots."""
        import numpy as np

        from robot_sf.nav.map_config import MapDefinitionPool
        from robot_sf.sim.simulator import Simulator

        # Create mock simulator with multiple robots
        simulator = Mock(spec=Simulator)
        # Create mock robots with proper pose attributes
        mock_robot1 = Mock()
        mock_robot1.pose = ((0.0, 0.0), 0.0)  # ((x, y), orientation)
        mock_robot1.current_speed = 0.0
        mock_robot2 = Mock()
        mock_robot2.pose = ((2.0, 2.0), 0.0)  # ((x, y), orientation)
        mock_robot2.current_speed = 0.0
        simulator.robots = [mock_robot1, mock_robot2]  # Two robots
        simulator.robot_pos = [(0, 0), (2, 2)]
        simulator.goal_pos = [(5, 5), (8, 8)]
        simulator.next_goal_pos = [(10, 10), (12, 12)]
        map_pool = MapDefinitionPool()
        simulator.map_def = map_pool.choose_random_map()

        # Create mock pysf_sim with env.obstacles_raw
        mock_pysf_sim = Mock()
        mock_env = Mock()
        mock_env.obstacles_raw = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])  # Multiple obstacles
        mock_pysf_sim.env = mock_env
        simulator.pysf_sim = mock_pysf_sim

        simulator.ped_pos = np.array([[1.0, 1.0], [3.0, 3.0]])  # Some pedestrian positions

        settings = RobotEnvSettings(use_image_obs=True)
        obs_space = spaces.Dict(
            {
                OBS_DRIVE_STATE: spaces.Box(low=-1, high=1, shape=(2, 5), dtype=np.float32),
                OBS_RAYS: spaces.Box(low=0, high=10, shape=(2, 4), dtype=np.float32),
                OBS_IMAGE: spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32),
            },
        )

        occupancies, sensors = init_collision_and_sensors_with_image(
            simulator,
            settings,
            obs_space,
            sim_view=Mock(),
        )

        # Should create sensors for each robot
        assert len(occupancies) == 2
        assert len(sensors) == 2
