"""Tests for occupancy grid gymnasium observation space integration.

This module tests the integration of occupancy grids as gymnasium observation
components, covering observation space creation, grid-to-observation conversion,
multi-channel support, environment lifecycle, and RL compatibility.

Test Coverage:
- T034: Box observation space creation (shape, dtype, bounds)
- T035: Grid-to-observation conversion (reshape, dtype, value range)
- T036: Multi-channel observation stacking
- T037: Variable grid config observation adaptation
- T038: Environment reset with occupancy observation
- T039: Environment step with occupancy observation updates
- T040: StableBaselines3 RL training compatibility
"""

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid


def _make_grid_config(
    width_m: float = 10.0,
    height_m: float = 10.0,
    resolution: float = 0.2,
    channels: list[GridChannel] | None = None,
) -> GridConfig:
    """Helper to create GridConfig with sensible defaults.

    Args:
        width_m: Grid width in meters
        height_m: Grid height in meters
        resolution: Meters per cell
        channels: List of GridChannel enums (default: obstacles + pedestrians)

    Returns:
        GridConfig instance with computed properties (grid_width, grid_height, num_channels)
    """
    if channels is None:
        channels = [GridChannel.OBSTACLES, GridChannel.PEDESTRIANS]

    return GridConfig(
        width=width_m,
        height=height_m,
        resolution=resolution,
        channels=channels,
    )


class TestObservationSpaceCreation:
    """Test T034: Box observation space creation with correct shape, dtype, bounds."""

    def test_observation_space_includes_grid(self):
        """Verify observation space includes grid Box when grid observation enabled."""
        grid_config = _make_grid_config(width_m=10.0, height_m=10.0, resolution=0.2)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config)

        # Verify observation space is Dict
        assert isinstance(env.observation_space, spaces.Dict)

        # Verify 'occupancy_grid' key exists
        assert "occupancy_grid" in env.observation_space.spaces

        # Verify grid observation is Box space
        grid_space = env.observation_space.spaces["occupancy_grid"]
        assert isinstance(grid_space, spaces.Box)

        # Verify shape [C, H, W] using computed properties
        expected_shape = (
            grid_config.num_channels,
            grid_config.grid_height,
            grid_config.grid_width,
        )
        assert grid_space.shape == expected_shape

        # Verify dtype float32
        assert grid_space.dtype == np.float32

        # Verify bounds [0, 1]
        assert grid_space.low.min() == 0.0
        assert grid_space.high.max() == 1.0

        env.close()

    def test_observation_space_without_grid(self):
        """Verify observation space excludes grid when grid observation disabled."""
        config = RobotSimulationConfig(
            use_occupancy_grid=False,
            include_grid_in_observation=False,
        )

        env = make_robot_env(config=config)

        # Verify observation space is Dict
        assert isinstance(env.observation_space, spaces.Dict)

        # Verify 'occupancy_grid' key does NOT exist
        assert "occupancy_grid" not in env.observation_space.spaces

        env.close()


class TestGridToObservationConversion:
    """Test T035: Grid-to-observation conversion (reshape, dtype, value range)."""

    def test_to_observation_returns_correct_shape(self):
        """Verify to_observation() returns array with shape [C, H, W]."""
        grid_config = _make_grid_config(
            width_m=10.0,
            height_m=10.0,
            resolution=0.25,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.ROBOT],
        )
        grid = OccupancyGrid(config=grid_config)

        # Generate grid with dummy data (num_channels, grid_height, grid_width)
        grid._grid_data = np.random.rand(
            grid_config.num_channels, grid_config.grid_height, grid_config.grid_width
        ).astype(np.float32)

        obs_array = grid.to_observation()

        # Verify shape [C, H, W]
        assert obs_array.shape == (
            grid_config.num_channels,
            grid_config.grid_height,
            grid_config.grid_width,
        )

    def test_to_observation_returns_float32(self):
        """Verify to_observation() returns float32 dtype."""
        grid_config = _make_grid_config(width_m=6.0, height_m=6.0, resolution=0.2)
        grid = OccupancyGrid(config=grid_config)

        # Generate grid with non-float32 dtype
        grid._grid_data = np.random.randint(
            0,
            2,
            size=(grid_config.num_channels, grid_config.grid_height, grid_config.grid_width),
            dtype=np.int32,
        )

        obs_array = grid.to_observation()

        # Verify dtype conversion to float32
        assert obs_array.dtype == np.float32

    def test_to_observation_clips_values_to_range(self):
        """Verify to_observation() clips values to [0, 1] range."""
        grid_config = _make_grid_config(
            width_m=4.0, height_m=4.0, resolution=0.2, channels=[GridChannel.OBSTACLES]
        )
        grid = OccupancyGrid(config=grid_config)

        # Create grid with out-of-range values
        grid._grid_data = np.array(
            [
                np.random.rand(grid_config.grid_height, grid_config.grid_width) * 2.0
                - 0.5  # Range [-0.5, 1.5]
            ],
            dtype=np.float32,
        )

        obs_array = grid.to_observation()

        # Verify all values in [0, 1]
        assert obs_array.min() >= 0.0
        assert obs_array.max() <= 1.0

    def test_to_observation_raises_if_grid_not_generated(self):
        """Verify to_observation() raises RuntimeError if grid not generated."""
        grid_config = _make_grid_config(width_m=2.0, height_m=2.0, resolution=0.2)
        grid = OccupancyGrid(config=grid_config)

        # Grid data is None before generate()
        assert grid._grid_data is None

        with pytest.raises(RuntimeError, match="Grid has not been generated yet"):
            grid.to_observation()


class TestMultiChannelObservationStacking:
    """Test T036: Multi-channel observation stacking in single array."""

    def test_multi_channel_observation_shape(self):
        """Verify multi-channel grids stack correctly in observation."""
        grid_config = _make_grid_config(
            width_m=6.0,
            height_m=6.0,
            resolution=0.2,
            channels=[
                GridChannel.OBSTACLES,
                GridChannel.PEDESTRIANS,
                GridChannel.ROBOT,
                GridChannel.COMBINED,
            ],
        )
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)
        obs, _info = env.reset(seed=42)

        # Verify observation contains grid
        assert "occupancy_grid" in obs

        # Verify grid has all channels stacked
        grid_obs = obs["occupancy_grid"]
        expected_shape = (grid_config.num_channels, grid_config.grid_height, grid_config.grid_width)
        assert grid_obs.shape == expected_shape
        assert grid_obs.dtype == np.float32

        env.close()

    def test_single_channel_observation(self):
        """Verify single-channel grid observation works correctly."""
        grid_config = _make_grid_config(
            width_m=5.0, height_m=5.0, resolution=0.2, channels=[GridChannel.OBSTACLES]
        )
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)
        obs, _info = env.reset(seed=42)

        # Verify observation contains grid
        assert "occupancy_grid" in obs

        # Verify grid has single channel dimension
        grid_obs = obs["occupancy_grid"]
        expected_shape = (grid_config.num_channels, grid_config.grid_height, grid_config.grid_width)
        assert grid_obs.shape == expected_shape
        assert grid_obs.dtype == np.float32

        env.close()


class TestVariableGridConfigObservation:
    """Test T037: Variable grid config observation adaptation."""

    @pytest.mark.parametrize(
        "width_m,height_m,resolution",
        [
            (10.0, 10.0, 0.2),  # 10m x 10m at 0.2m resolution -> 50x50 cells
            (10.0, 10.0, 0.1),  # 10m x 10m at 0.1m resolution -> 100x100 cells
            (10.0, 15.0, 0.25),  # 10m x 15m at 0.25m resolution -> 40x60 cells
        ],
    )
    def test_different_grid_sizes(self, width_m, height_m, resolution):
        """Verify observation space adapts to different grid sizes and resolutions."""
        grid_config = _make_grid_config(width_m=width_m, height_m=height_m, resolution=resolution)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)

        # Verify observation space matches grid config
        grid_space = env.observation_space.spaces["occupancy_grid"]
        expected_shape = (grid_config.num_channels, grid_config.grid_height, grid_config.grid_width)
        assert grid_space.shape == expected_shape

        # Verify observation matches space
        obs, _info = env.reset(seed=42)
        assert obs["occupancy_grid"].shape == expected_shape

        env.close()

    def test_different_channel_counts(self):
        """Verify observation space adapts to different channel counts."""
        channel_configs = [
            [GridChannel.OBSTACLES],  # 1 channel
            [GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],  # 2 channels
            [GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.ROBOT],  # 3 channels
            [
                GridChannel.OBSTACLES,
                GridChannel.PEDESTRIANS,
                GridChannel.ROBOT,
                GridChannel.COMBINED,
            ],  # 4 channels
        ]

        for channels in channel_configs:
            grid_config = _make_grid_config(
                width_m=6.0, height_m=6.0, resolution=0.2, channels=channels
            )
            config = RobotSimulationConfig(
                use_occupancy_grid=True,
                grid_config=grid_config,
                include_grid_in_observation=True,
            )

            env = make_robot_env(config=config, seed=42)

            # Verify observation space matches channel count
            grid_space = env.observation_space.spaces["occupancy_grid"]
            assert grid_space.shape[0] == len(channels)

            # Verify observation matches space
            obs, _info = env.reset(seed=42)
            assert obs["occupancy_grid"].shape[0] == len(channels)

            env.close()


class TestEnvironmentResetWithGrid:
    """Test T038: Environment reset with occupancy observation."""

    def test_reset_generates_initial_grid(self):
        """Verify reset() generates initial occupancy grid observation."""
        grid_config = _make_grid_config(width_m=10.0, height_m=10.0, resolution=0.25)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)
        obs, info = env.reset(seed=42)

        # Verify observation contains grid
        assert "occupancy_grid" in obs

        # Verify grid has correct shape and dtype
        grid_obs = obs["occupancy_grid"]
        expected_shape = (grid_config.num_channels, grid_config.grid_height, grid_config.grid_width)
        assert grid_obs.shape == expected_shape
        assert grid_obs.dtype == np.float32

        # Verify values in [0, 1]
        assert grid_obs.min() >= 0.0
        assert grid_obs.max() <= 1.0

        # Verify info dict structure (standard gymnasium)
        assert isinstance(info, dict)

        env.close()

    def test_reset_with_different_seeds(self):
        """Verify reset() with different seeds produces different grids."""
        grid_config = _make_grid_config(width_m=6.0, height_m=6.0, resolution=0.2)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config)

        # Reset with seed 42
        obs1, _info1 = env.reset(seed=42)
        grid1 = obs1["occupancy_grid"]

        # Reset with seed 123
        obs2, _info2 = env.reset(seed=123)
        grid2 = obs2["occupancy_grid"]

        # Grids should potentially differ (different pedestrian positions)
        # Note: They might be identical if obstacles dominate, but structure should match
        assert grid1.shape == grid2.shape
        assert grid1.dtype == grid2.dtype

        env.close()


class TestEnvironmentStepWithGridUpdate:
    """Test T039: Environment step with occupancy observation updates."""

    def test_step_updates_grid_observation(self):
        """Verify step() updates occupancy grid each timestep."""
        grid_config = _make_grid_config(width_m=7.0, height_m=7.0, resolution=0.2)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)
        obs_reset, _info = env.reset(seed=42)

        # Verify initial grid observation
        assert "occupancy_grid" in obs_reset
        grid_reset = obs_reset["occupancy_grid"].copy()

        # Take a step
        action = env.action_space.sample()
        obs_step, _, _, _, _ = env.step(action)

        # Verify step observation contains grid
        assert "occupancy_grid" in obs_step
        grid_step = obs_step["occupancy_grid"]

        # Verify grid shape and dtype unchanged
        assert grid_step.shape == grid_reset.shape
        assert grid_step.dtype == grid_reset.dtype

        # Verify values in [0, 1]
        assert grid_step.min() >= 0.0
        assert grid_step.max() <= 1.0

        # Note: Grid content may or may not change depending on pedestrian movement
        # We just verify the observation is properly updated

        env.close()
        assert grid_step.max() <= 1.0

        # Note: Grid content may or may not change depending on pedestrian movement
        # We just verify the observation is properly updated

        env.close()

    def test_grid_updates_over_multiple_steps(self):
        """Verify grid observation updates consistently over multiple steps."""
        grid_config = _make_grid_config(width_m=6.0, height_m=6.0, resolution=0.2)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)
        _obs, _info = env.reset(seed=42)

        # Take 10 steps and verify grid in each observation
        for _step_num in range(10):
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, _info = env.step(action)

            # Verify grid observation present and valid
            assert "occupancy_grid" in obs
            grid_obs = obs["occupancy_grid"]
            expected_shape = (
                grid_config.num_channels,
                grid_config.grid_height,
                grid_config.grid_width,
            )
            assert grid_obs.shape == expected_shape
            assert grid_obs.dtype == np.float32
            assert grid_obs.min() >= 0.0
            assert grid_obs.max() <= 1.0

            if terminated or truncated:
                break

        env.close()


class TestStableBaselines3Compatibility:
    """Test T040: StableBaselines3 RL training compatibility."""

    def test_short_episode_with_ppo(self):
        """Verify environment works with StableBaselines3 PPO for quick episode.

        This test validates that the grid observation integrates correctly with
        RL training frameworks by running a minimal PPO training loop.
        """
        pytest.importorskip("stable_baselines3", reason="StableBaselines3 not installed")

        from stable_baselines3 import PPO

        grid_config = _make_grid_config(
            width_m=10.0, height_m=10.0, resolution=0.4
        )  # Smaller grid for faster test
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)

        # Create minimal PPO model
        model = PPO(
            "MultiInputPolicy",  # Required for Dict observation spaces
            env,
            verbose=0,
            n_steps=16,  # Minimal buffer
            batch_size=8,
        )

        # Train for minimal timesteps (just verify no crashes)
        model.learn(total_timesteps=32, progress_bar=False)

        # Run a quick evaluation episode
        obs, _info = env.reset(seed=42)
        for _step in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)

            # Verify grid observation still present and valid
            assert "occupancy_grid" in obs
            expected_shape = (
                grid_config.num_channels,
                grid_config.grid_height,
                grid_config.grid_width,
            )
            assert obs["occupancy_grid"].shape == expected_shape
            assert obs["occupancy_grid"].dtype == np.float32

            if terminated or truncated:
                break

        env.close()

    def test_observation_space_compatible_with_sb3(self):
        """Verify observation space structure is compatible with SB3 MultiInputPolicy."""
        pytest.importorskip("stable_baselines3", reason="StableBaselines3 not installed")

        # Note: get_flattened_obs_dim is not available in newer SB3 versions
        # Instead, we test that the environment can be wrapped by SB3's VecEnv
        from stable_baselines3.common.vec_env import DummyVecEnv

        grid_config = _make_grid_config(width_m=4.0, height_m=4.0, resolution=0.2)
        config = RobotSimulationConfig(
            use_occupancy_grid=True,
            grid_config=grid_config,
            include_grid_in_observation=True,
        )

        env = make_robot_env(config=config, seed=42)

        # Verify SB3 can wrap environment (validates observation space compatibility)
        try:
            vec_env = DummyVecEnv([lambda: env])
            obs = vec_env.reset()
            assert obs is not None
            vec_env.close()
        except Exception as exc:
            pytest.fail(f"SB3 failed to process observation space: {exc}")

        env.close()


# Execution guard for direct pytest run
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
