"""
Test suite for the ImageSensor class and image observation functionality.
"""

import numpy as np
import pygame
import pytest
from gymnasium import spaces

from robot_sf.render.sim_view import SimulationView
from robot_sf.sensor.image_sensor import ImageSensor, ImageSensorSettings, image_sensor_space


class TestImageSensorSettings:
    """Test the ImageSensorSettings configuration class."""

    def test_default_settings(self):
        """Test default ImageSensorSettings initialization."""
        settings = ImageSensorSettings()
        assert settings.width == 84
        assert settings.height == 84
        assert settings.channels == 3
        assert settings.normalize is True
        assert settings.grayscale is False

    def test_custom_settings(self):
        """Test custom ImageSensorSettings initialization."""
        settings = ImageSensorSettings(
            width=128,
            height=96,
            channels=1,
            normalize=False,
            grayscale=True,
        )
        assert settings.width == 128
        assert settings.height == 96
        assert settings.channels == 1
        assert settings.normalize is False
        assert settings.grayscale is True


class TestImageSensorSpace:
    """Test the image sensor space creation function."""

    def test_rgb_normalized_space(self):
        """Test RGB normalized image space."""
        settings = ImageSensorSettings(width=64, height=64, channels=3, normalize=True)
        space = image_sensor_space(settings)

        assert isinstance(space, spaces.Box)
        assert space.shape == (64, 64, 3)
        assert space.dtype == np.float32
        assert space.low.min() == 0.0
        assert space.high.max() == 1.0

    def test_grayscale_normalized_space(self):
        """Test grayscale normalized image space."""
        settings = ImageSensorSettings(width=32, height=48, grayscale=True, normalize=True)
        space = image_sensor_space(settings)

        assert isinstance(space, spaces.Box)
        assert space.shape == (48, 32)  # height, width for grayscale
        assert space.dtype == np.float32
        assert space.low.min() == 0.0
        assert space.high.max() == 1.0

    def test_rgb_unnormalized_space(self):
        """Test RGB unnormalized image space."""
        settings = ImageSensorSettings(width=100, height=80, normalize=False)
        space = image_sensor_space(settings)

        assert isinstance(space, spaces.Box)
        assert space.shape == (80, 100, 3)
        assert space.dtype == np.uint8
        assert space.low.min() == 0
        assert space.high.max() == 255


class TestImageSensor:
    """Test the ImageSensor class functionality."""

    @pytest.fixture
    def sim_view(self):
        """Create a minimal SimulationView for testing."""
        pygame.init()
        # Create a small offscreen surface for testing
        screen = pygame.Surface((200, 150))
        screen.fill((100, 150, 200))  # Fill with a test color

        # Create a mock SimulationView with the test surface using MapDefinitionPool
        from robot_sf.nav.map_config import MapDefinitionPool

        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        sim_view = SimulationView(width=200, height=150, record_video=True, map_def=map_def)
        sim_view.screen = screen
        return sim_view

    def test_sensor_initialization(self, sim_view):
        """Test ImageSensor initialization."""
        settings = ImageSensorSettings()
        sensor = ImageSensor(settings, sim_view)

        assert sensor.settings == settings
        assert sensor.sim_view == sim_view

    def test_sensor_initialization_without_sim_view(self):
        """Test ImageSensor initialization without SimulationView."""
        settings = ImageSensorSettings()
        sensor = ImageSensor(settings)

        assert sensor.settings == settings
        assert sensor.sim_view is None

    def test_set_sim_view(self, sim_view):
        """Test setting SimulationView after initialization."""
        settings = ImageSensorSettings()
        sensor = ImageSensor(settings)
        sensor.set_sim_view(sim_view)

        assert sensor.sim_view == sim_view

    def test_capture_frame_without_sim_view(self):
        """Test capturing frame without SimulationView raises error."""
        settings = ImageSensorSettings()
        sensor = ImageSensor(settings)

        with pytest.raises(ValueError, match="SimulationView not set"):
            sensor.capture_frame()

    def test_capture_frame_rgb_normalized(self, sim_view):
        """Test capturing RGB normalized frame."""
        settings = ImageSensorSettings(width=50, height=40, normalize=True)
        sensor = ImageSensor(settings, sim_view)

        frame = sensor.capture_frame()

        assert frame.shape == (40, 50, 3)  # height, width, channels
        assert frame.dtype == np.float32
        assert 0.0 <= frame.min() <= frame.max() <= 1.0

    def test_capture_frame_rgb_unnormalized(self, sim_view):
        """Test capturing RGB unnormalized frame."""
        settings = ImageSensorSettings(width=60, height=45, normalize=False)
        sensor = ImageSensor(settings, sim_view)

        frame = sensor.capture_frame()

        assert frame.shape == (45, 60, 3)
        assert frame.dtype == np.uint8
        assert 0 <= frame.min() <= frame.max() <= 255

    def test_capture_frame_grayscale_normalized(self, sim_view):
        """Test capturing grayscale normalized frame."""
        settings = ImageSensorSettings(width=30, height=25, grayscale=True, normalize=True)
        sensor = ImageSensor(settings, sim_view)

        frame = sensor.capture_frame()

        assert frame.shape == (25, 30)  # height, width (no channels for grayscale)
        assert frame.dtype == np.float32
        assert 0.0 <= frame.min() <= frame.max() <= 1.0

    def test_capture_frame_grayscale_unnormalized(self, sim_view):
        """Test capturing grayscale unnormalized frame."""
        settings = ImageSensorSettings(width=35, height=28, grayscale=True, normalize=False)
        sensor = ImageSensor(settings, sim_view)

        frame = sensor.capture_frame()

        assert frame.shape == (28, 35)
        assert frame.dtype == np.uint8
        assert 0 <= frame.min() <= frame.max() <= 255

    def test_resize_functionality(self, sim_view):
        """Test that frames are properly resized."""
        # SimulationView has 200x150 surface, sensor wants 100x75
        settings = ImageSensorSettings(width=100, height=75)
        sensor = ImageSensor(settings, sim_view)

        frame = sensor.capture_frame()

        assert frame.shape == (75, 100, 3)  # Resized to target dimensions

    def test_grayscale_conversion(self, sim_view):
        """Test grayscale conversion produces reasonable values."""
        settings = ImageSensorSettings(grayscale=True, normalize=True)
        sensor = ImageSensor(settings, sim_view)

        frame = sensor.capture_frame()

        # Should be single channel grayscale
        assert len(frame.shape) == 2
        # Should have reasonable grayscale values (not all zeros or ones)
        assert 0.0 < frame.mean() < 1.0

    @pytest.fixture
    def cleanup_pygame(self):
        """Clean up pygame after each test."""
        yield
        pygame.quit()


class TestImageSensorIntegration:
    """Integration tests for ImageSensor with actual rendering."""

    def test_sensor_with_actual_rendering(self):
        """Test ImageSensor with actual SimulationView rendering."""
        from robot_sf.nav.map_config import MapDefinitionPool

        # Create a real SimulationView (but with video recording to avoid window)
        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        sim_view = SimulationView(
            width=200,
            height=150,
            record_video=True,  # This creates offscreen surface
            map_def=map_def,
        )

        # Create image sensor
        settings = ImageSensorSettings(width=84, height=84, normalize=True)
        sensor = ImageSensor(settings, sim_view)

        # Clear the screen with a known color
        sim_view.screen.fill((50, 100, 150))

        # Capture frame
        frame = sensor.capture_frame()

        # Verify frame properties
        assert frame.shape == (84, 84, 3)
        assert frame.dtype == np.float32
        assert 0.0 <= frame.min() <= frame.max() <= 1.0

        # Verify it captured something (not all zeros)
        assert frame.sum() > 0

        # Clean up
        sim_view.exit_simulation()

    def test_different_image_formats(self):
        """Test various image format configurations."""
        from robot_sf.nav.map_config import MapDefinitionPool

        formats_to_test = [
            {"width": 64, "height": 64, "grayscale": False, "normalize": True},
            {"width": 32, "height": 32, "grayscale": True, "normalize": True},
            {"width": 128, "height": 96, "grayscale": False, "normalize": False},
            {"width": 48, "height": 48, "grayscale": True, "normalize": False},
        ]

        map_pool = MapDefinitionPool()
        map_def = map_pool.choose_random_map()

        for fmt in formats_to_test:
            sim_view = SimulationView(width=200, height=150, record_video=True, map_def=map_def)

            settings = ImageSensorSettings(**fmt)
            sensor = ImageSensor(settings, sim_view)

            # Fill with different colors for each test
            sim_view.screen.fill(
                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
            )

            frame = sensor.capture_frame()

            # Verify shape
            if fmt["grayscale"]:
                expected_shape = (fmt["height"], fmt["width"])
            else:
                expected_shape = (fmt["height"], fmt["width"], 3)
            assert frame.shape == expected_shape

            # Verify dtype
            if fmt["normalize"]:
                assert frame.dtype == np.float32
                assert 0.0 <= frame.min() <= frame.max() <= 1.0
            else:
                assert frame.dtype == np.uint8
                assert 0 <= frame.min() <= frame.max() <= 255

            sim_view.exit_simulation()
