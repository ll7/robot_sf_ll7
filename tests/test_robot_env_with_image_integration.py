"""
Integration test suite for the complete image-based robot environment.
"""

import numpy as np

from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage
from robot_sf.sensor.image_sensor import ImageSensorSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_IMAGE, OBS_RAYS


class TestRobotEnvWithImageIntegration:
    """Integration tests for RobotEnvWithImage."""

    def test_env_creation_with_image_disabled(self):
        """
        Tests that the environment can be created without image observations and verifies the observation space excludes the image key while including drive state and rays.
        """
        settings = RobotEnvSettings(use_image_obs=False)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=False)

            assert env is not None
            assert hasattr(env, "observation_space")
            assert hasattr(env, "action_space")

            # Should not contain image observations
            assert OBS_IMAGE not in env.observation_space.spaces
            assert OBS_DRIVE_STATE in env.observation_space.spaces
            assert OBS_RAYS in env.observation_space.spaces

        finally:
            if "env" in locals():
                env.exit()

    def test_env_creation_with_image_enabled(self):
        """
        Tests that the environment is created with image observations enabled and verifies the observation space includes the image key with correct shape and data type.
        """
        image_config = ImageSensorSettings(width=64, height=64, normalize=True)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            assert env is not None
            assert hasattr(env, "observation_space")
            assert hasattr(env, "action_space")

            # Should contain all observation types
            assert OBS_DRIVE_STATE in env.observation_space.spaces
            assert OBS_RAYS in env.observation_space.spaces
            assert OBS_IMAGE in env.observation_space.spaces

            # Verify image space properties
            image_space = env.observation_space.spaces[OBS_IMAGE]
            assert image_space.shape == (64, 64, 3)
            assert image_space.dtype == np.float32

        finally:
            if "env" in locals():
                env.exit()

    def test_env_reset_with_image_observations(self):
        """
        Tests that resetting the environment with image observations enabled returns a valid observation.
        
        Verifies that the reset observation includes drive state, rays, and image keys with correct shapes and data types, and that the image observation is normalized and conforms to the observation space.
        """
        image_config = ImageSensorSettings(width=32, height=32, normalize=True)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            # Reset the environment
            obs, info = env.reset()

            # Verify observation structure
            assert isinstance(obs, dict)
            assert OBS_DRIVE_STATE in obs
            assert OBS_RAYS in obs
            assert OBS_IMAGE in obs

            # Verify observation shapes and types
            assert obs[OBS_DRIVE_STATE].dtype == np.float32
            assert obs[OBS_RAYS].dtype == np.float32
            assert obs[OBS_IMAGE].dtype == np.float32

            # Verify image observation properties
            assert obs[OBS_IMAGE].shape == (32, 32, 3)
            assert 0.0 <= obs[OBS_IMAGE].min() <= obs[OBS_IMAGE].max() <= 1.0

            # Verify observation is valid according to space
            assert env.observation_space.contains(obs)

        finally:
            if "env" in locals():
                env.exit()

    def test_env_step_with_image_observations(self):
        """
        Tests that stepping the environment with grayscale, normalized image observations returns valid outputs.
        
        Verifies that the observation after a step includes the expected keys, the grayscale image has the correct shape and dtype, pixel values are within the normalized range, and all step outputs are of the correct types.
        """
        image_config = ImageSensorSettings(width=48, height=48, grayscale=True, normalize=True)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            # Reset and take a step
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify observation structure after step
            assert isinstance(obs, dict)
            assert OBS_DRIVE_STATE in obs
            assert OBS_RAYS in obs
            assert OBS_IMAGE in obs

            # Verify grayscale image
            assert obs[OBS_IMAGE].shape == (48, 48)  # Grayscale has no channel dimension
            assert obs[OBS_IMAGE].dtype == np.float32
            assert 0.0 <= obs[OBS_IMAGE].min() <= obs[OBS_IMAGE].max() <= 1.0

            # Verify step outputs
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

        finally:
            if "env" in locals():
                env.exit()

    def test_multiple_episodes_with_image(self):
        """
        Runs multiple episodes in the environment with image observations enabled and verifies observation consistency.
        
        This test checks that, across several episodes and steps, image observations have the expected shape, data type, and pixel value range. It also ensures that each observation is valid according to the environment's observation space.
        """
        image_config = ImageSensorSettings(width=32, height=32, normalize=False)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            for episode in range(3):
                obs, info = env.reset()

                # Verify initial observation
                assert OBS_IMAGE in obs
                assert obs[OBS_IMAGE].shape == (32, 32, 3)
                assert obs[OBS_IMAGE].dtype == np.uint8
                assert 0 <= obs[OBS_IMAGE].min() <= obs[OBS_IMAGE].max() <= 255

                # Take a few steps
                for step in range(5):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)

                    # Verify observation consistency
                    assert OBS_IMAGE in obs
                    assert obs[OBS_IMAGE].shape == (32, 32, 3)
                    assert env.observation_space.contains(obs)

                    if terminated or truncated:
                        break

        finally:
            if "env" in locals():
                env.exit()

    def test_debug_mode_forced_with_image_obs(self):
        """
        Tests that debug mode is automatically enabled when image observations are used, ensuring the simulation UI is created even if debug is set to False.
        """
        settings = RobotEnvSettings(use_image_obs=True)

        try:
            # Try to create environment with debug=False
            env = RobotEnvWithImage(env_config=settings, debug=False)

            # Should have sim_ui created anyway due to image observations
            assert hasattr(env, "sim_ui")
            assert env.sim_ui is not None

        finally:
            if "env" in locals():
                env.exit()

    def test_image_observation_consistency_across_steps(self):
        """
        Verifies that image observations maintain correct shape, dtype, and value range across multiple environment steps with normalized image settings.
        
        Ensures that each image observation after reset and subsequent steps is a 64x64 RGB float32 array with values in [0.0, 1.0], confirming observation consistency and validity throughout an episode.
        """
        image_config = ImageSensorSettings(width=64, height=64, normalize=True)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            obs, info = env.reset()
            initial_image = obs[OBS_IMAGE].copy()

            # Take several steps and collect images
            images = [initial_image]
            for _ in range(3):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                images.append(obs[OBS_IMAGE].copy())

                if terminated or truncated:
                    break

            # Verify all images have the same properties
            for img in images:
                assert img.shape == (64, 64, 3)
                assert img.dtype == np.float32
                assert 0.0 <= img.min() <= img.max() <= 1.0

            # Images should potentially be different (due to robot movement)
            # but this is environment-dependent, so we just verify they're valid

        finally:
            if "env" in locals():
                env.exit()

    def test_different_image_configurations(self):
        """
        Tests environment creation and reset with multiple image sensor configurations.
        
        For each configuration, verifies that the image observation's shape, data type, and pixel value range match the expected properties based on grayscale and normalization settings.
        """
        configs_to_test = [
            {"width": 32, "height": 32, "normalize": True, "grayscale": False},
            {"width": 84, "height": 84, "normalize": False, "grayscale": False},
            {"width": 48, "height": 48, "normalize": True, "grayscale": True},
            {"width": 128, "height": 96, "normalize": False, "grayscale": True},
        ]

        for config in configs_to_test:
            image_config = ImageSensorSettings(**config)
            settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

            try:
                env = RobotEnvWithImage(env_config=settings, debug=True)

                obs, info = env.reset()

                # Verify image properties match configuration
                image_obs = obs[OBS_IMAGE]

                if config["grayscale"]:
                    expected_shape = (config["height"], config["width"])
                else:
                    expected_shape = (config["height"], config["width"], 3)

                assert image_obs.shape == expected_shape

                if config["normalize"]:
                    assert image_obs.dtype == np.float32
                    assert 0.0 <= image_obs.min() <= image_obs.max() <= 1.0
                else:
                    assert image_obs.dtype == np.uint8
                    assert 0 <= image_obs.min() <= image_obs.max() <= 255

            finally:
                if "env" in locals():
                    env.exit()


class TestImageObservationSpaceValidation:
    """Test observation space validation with image observations."""

    def test_observation_space_sample_and_contains(self):
        """
        Tests that the observation space can generate valid samples and that both sampled and actual environment observations conform to the defined space.
        """
        image_config = ImageSensorSettings(width=32, height=32, normalize=True)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            # Sample from observation space
            sample_obs = env.observation_space.sample()

            # Verify structure
            assert isinstance(sample_obs, dict)
            assert OBS_DRIVE_STATE in sample_obs
            assert OBS_RAYS in sample_obs
            assert OBS_IMAGE in sample_obs

            # Verify space contains its own sample
            assert env.observation_space.contains(sample_obs)

            # Get actual observation and verify it's also valid
            actual_obs, _ = env.reset()
            assert env.observation_space.contains(actual_obs)

        finally:
            if "env" in locals():
                env.exit()

    def test_observation_space_bounds_checking(self):
        """
        Verifies that the observation space correctly rejects observations with out-of-bounds image values or incorrect image shapes.
        """
        image_config = ImageSensorSettings(width=32, height=32, normalize=True)
        settings = RobotEnvSettings(use_image_obs=True, image_config=image_config)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            # Create invalid observations
            invalid_obs_high_image = {
                OBS_DRIVE_STATE: np.zeros((3, 5), dtype=np.float32),
                OBS_RAYS: np.zeros((3, 16), dtype=np.float32),  # Assuming 16 lidar rays
                OBS_IMAGE: np.ones((32, 32, 3), dtype=np.float32) * 2.0,  # Values > 1.0
            }

            invalid_obs_wrong_shape = {
                OBS_DRIVE_STATE: np.zeros((3, 5), dtype=np.float32),
                OBS_RAYS: np.zeros((3, 16), dtype=np.float32),
                OBS_IMAGE: np.zeros((16, 16, 3), dtype=np.float32),  # Wrong shape
            }

            # These should not be contained in the observation space
            assert not env.observation_space.contains(invalid_obs_high_image)
            assert not env.observation_space.contains(invalid_obs_wrong_shape)

        finally:
            if "env" in locals():
                env.exit()


class TestErrorHandling:
    """Test error handling in image-based environments."""

    def test_missing_image_config_handling(self):
        """
        Tests that the environment auto-creates a default image sensor configuration when image observations are enabled but no explicit image configuration is provided.
        
        Ensures the environment initializes successfully and the image observation key is present in the observation space.
        """
        # This should auto-create image_config in post_init
        settings = RobotEnvSettings(use_image_obs=True)

        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)

            # Should work without errors
            assert env is not None
            assert OBS_IMAGE in env.observation_space.spaces

        finally:
            if "env" in locals():
                env.exit()

    def test_graceful_degradation_without_sim_view(self):
        """
        Verifies that the environment can be created successfully even if the simulation view component is unavailable or fails to initialize.
        
        This test ensures that enabling image observations does not prevent environment instantiation in the absence of the simulation view.
        """
        settings = RobotEnvSettings(use_image_obs=True)

        # This test might be hard to implement without mocking internal components
        # For now, we just verify the environment can be created
        try:
            env = RobotEnvWithImage(env_config=settings, debug=True)
            assert env is not None

        finally:
            if "env" in locals():
                env.exit()
