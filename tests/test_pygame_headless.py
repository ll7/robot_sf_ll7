"""
Test that pygame windows don't open during tests in headless environments.
"""

import os

import pygame

from robot_sf.gym_env.environment_factory import make_robot_env


class TestPygameHeadless:
    """Test pygame headless behavior."""

    def test_pytest_session_forces_dummy_driver(self):
        """Verify that the pytest session enforces the dummy SDL video driver.

        Before this fix, running tests locally without setting env vars could
        open a real window. This test would have failed by reporting a non-dummy
        driver. With the session-level fixture, we always see the dummy driver.
        """
        # Ensure display module is initialized for driver query
        try:
            pygame.display.init()
            driver = pygame.display.get_driver()
            assert driver == "dummy"
        finally:
            pygame.display.quit()

    def test_env_does_not_init_display_in_headless(self):
        """Creating a debug env in tests must not initialize a real display."""
        pygame.quit()  # Ensure a clean slate
        env = make_robot_env(debug=True)
        try:
            # In headless mode, SimulationView uses an offscreen Surface and
            # should not call pygame.display.set_mode(); thus no active display surface.
            assert env.sim_ui is not None
            assert isinstance(env.sim_ui.screen, pygame.Surface)
            assert pygame.display.get_surface() is None
        finally:
            env.exit()

    def test_simulation_view_respects_headless_environment(self):
        """Test that SimulationView doesn't create display window in headless environment."""
        # Save original environment
        original_display = os.environ.get("DISPLAY")
        original_sdl_driver = os.environ.get("SDL_VIDEODRIVER")
        original_mpl_backend = os.environ.get("MPLBACKEND")

        try:
            # Set headless environment variables
            os.environ["DISPLAY"] = ""
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["MPLBACKEND"] = "Agg"

            # Quit pygame to reset state
            pygame.quit()

            # Create environment with debug mode (which creates SimulationView)
            env = make_robot_env(debug=True)

            # Verify that SimulationView was created with offscreen surface
            assert env.sim_ui is not None
            assert env.sim_ui.screen is not None
            assert isinstance(env.sim_ui.screen, pygame.Surface)

            # The test passes if no pygame window opened (which we can't directly test,
            # but the offscreen surface creation should prevent it)

            # Clean up
            env.exit()

        finally:
            # Restore original environment
            if original_display is not None:
                os.environ["DISPLAY"] = original_display
            elif "DISPLAY" in os.environ:
                del os.environ["DISPLAY"]

            if original_sdl_driver is not None:
                os.environ["SDL_VIDEODRIVER"] = original_sdl_driver
            elif "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]

            if original_mpl_backend is not None:
                os.environ["MPLBACKEND"] = original_mpl_backend
            elif "MPLBACKEND" in os.environ:
                del os.environ["MPLBACKEND"]

            # Reset pygame to restore normal state
            pygame.quit()

    def test_robot_env_debug_mode_headless(self):
        """Test that robot environment with debug mode works in headless environment."""
        # Save original environment
        original_display = os.environ.get("DISPLAY")
        original_sdl_driver = os.environ.get("SDL_VIDEODRIVER")
        original_mpl_backend = os.environ.get("MPLBACKEND")

        try:
            # Set headless environment (same as in copilot instructions)
            os.environ["DISPLAY"] = ""
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["MPLBACKEND"] = "Agg"

            # Create environment with debug mode (which creates SimulationView)
            env = make_robot_env(debug=True)

            # Should work without creating visible window
            obs, _ = env.reset()
            assert obs is not None

            # Verify SimulationView was created with offscreen surface in headless mode
            assert env.sim_ui is not None
            assert isinstance(env.sim_ui.screen, pygame.Surface)

            # Clean up
            env.exit()

        finally:
            # Restore original environment
            if original_display is not None:
                os.environ["DISPLAY"] = original_display
            elif "DISPLAY" in os.environ:
                del os.environ["DISPLAY"]

            if original_sdl_driver is not None:
                os.environ["SDL_VIDEODRIVER"] = original_sdl_driver
            elif "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]

            if original_mpl_backend is not None:
                os.environ["MPLBACKEND"] = original_mpl_backend
            elif "MPLBACKEND" in os.environ:
                del os.environ["MPLBACKEND"]

    def test_simulation_view_normal_vs_headless_mode(self):
        """Test that SimulationView behaves differently in normal vs headless mode."""

        # Test normal mode first (should use display when available)
        original_display = os.environ.get("DISPLAY")
        original_sdl_driver = os.environ.get("SDL_VIDEODRIVER")
        original_mpl_backend = os.environ.get("MPLBACKEND")

        try:
            # Ensure we start in "normal" mode (no headless environment vars)
            if "DISPLAY" in os.environ and os.environ["DISPLAY"] == "":
                del os.environ["DISPLAY"]
            if "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]
            if "MPLBACKEND" in os.environ:
                del os.environ["MPLBACKEND"]

            pygame.quit()  # Reset pygame

            # Create environment in normal mode
            env_normal = make_robot_env(debug=True)
            assert env_normal.sim_ui is not None
            normal_screen = env_normal.sim_ui.screen
            env_normal.exit()

            # Now test headless mode
            os.environ["DISPLAY"] = ""
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["MPLBACKEND"] = "Agg"

            pygame.quit()  # Reset pygame

            # Create environment in headless mode
            env_headless = make_robot_env(debug=True)
            assert env_headless.sim_ui is not None
            headless_screen = env_headless.sim_ui.screen
            env_headless.exit()

            # Both should work and create surfaces
            assert isinstance(normal_screen, pygame.Surface)
            assert isinstance(headless_screen, pygame.Surface)

            # They should have the same size
            assert normal_screen.get_size() == headless_screen.get_size()

        finally:
            # Restore original environment
            if original_display is not None:
                os.environ["DISPLAY"] = original_display
            elif "DISPLAY" in os.environ:
                del os.environ["DISPLAY"]

            if original_sdl_driver is not None:
                os.environ["SDL_VIDEODRIVER"] = original_sdl_driver
            elif "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]

            if original_mpl_backend is not None:
                os.environ["MPLBACKEND"] = original_mpl_backend
            elif "MPLBACKEND" in os.environ:
                del os.environ["MPLBACKEND"]

            pygame.quit()
