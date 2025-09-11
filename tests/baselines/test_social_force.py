"""Unit tests for the Social Force planner baseline."""

import json
import math

import numpy as np
import pytest

from robot_sf.baselines.social_force import Observation, SFPlannerConfig, SocialForcePlanner


class TestSFPlannerConfig:
    """Tests for the SFPlannerConfig dataclass."""

    def test_default_config(self):
        """Test that default configuration creates valid instance."""
        config = SFPlannerConfig()
        assert config.mode == "velocity"
        assert config.v_max == 2.0
        assert config.desired_speed == 1.0
        assert config.action_space == "velocity"
        assert config.safety_clamp is True
        assert config.noise_std == 0.0

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {"v_max": 3.0, "desired_speed": 1.5, "A": 10.0, "action_space": "unicycle"}
        config = SFPlannerConfig(**config_dict)
        assert config.v_max == 3.0
        assert config.desired_speed == 1.5
        assert config.A == 10.0
        assert config.action_space == "unicycle"
        # Check defaults are preserved
        assert config.mode == "velocity"
        assert config.safety_clamp is True


class TestSocialForcePlanner:
    """Tests for the SocialForcePlanner class."""

    @pytest.fixture
    def default_config(self):
        """Provide default test configuration."""
        return SFPlannerConfig(
            v_max=2.0,
            desired_speed=1.0,
            noise_std=0.0,  # Deterministic
            action_space="velocity",
        )

    @pytest.fixture
    def planner(self, default_config):
        """Provide configured planner instance."""
        return SocialForcePlanner(default_config, seed=42)

    def test_planner_initialization(self, default_config):
        """Test planner can be initialized with different config types."""
        # Test with dataclass config
        planner1 = SocialForcePlanner(default_config, seed=42)
        assert planner1.config.v_max == 2.0

        # Test with dict config
        config_dict = {"v_max": 3.0, "action_space": "unicycle"}
        planner2 = SocialForcePlanner(config_dict, seed=42)
        assert planner2.config.v_max == 3.0
        assert planner2.config.action_space == "unicycle"

    def test_reset_functionality(self, planner):
        """Test that reset clears internal state."""
        # Setup some internal state
        planner._last_position = np.array([1.0, 2.0])  # noqa: SLF001
        planner._robot_state = {"dummy": "state"}  # noqa: SLF001

        # Reset and check state is cleared (same seed preserved)
        planner.reset()
        assert planner._last_position is None  # noqa: SLF001
        assert planner._robot_state is None  # noqa: SLF001

        # Capture a short RNG sequence with initial seed (42 from fixture)
        seq_before = [planner._rng.randint(1000) for _ in range(5)]  # noqa: SLF001

        # Reset with a different seed and capture a new sequence
        planner.reset(seed=100)
        seq_after_diff_seed = [planner._rng.randint(1000) for _ in range(5)]  # noqa: SLF001
        assert seq_before != seq_after_diff_seed, (
            "RNG sequence should differ after reseeding with a different seed"
        )

        # Reset back to original seed and ensure sequence reproducibility
        planner.reset(seed=42)
        seq_after_same_seed = [planner._rng.randint(1000) for _ in range(5)]  # noqa: SLF001
        assert seq_before == seq_after_same_seed, (
            "RNG sequence should match when reseeded with the original seed"
        )

    def test_configure_updates_config(self, planner):
        """Test that configure updates the configuration."""
        original_v_max = planner.config.v_max
        new_config = {"v_max": original_v_max + 1.0, "desired_speed": 2.0}

        planner.configure(new_config)
        assert planner.config.v_max == original_v_max + 1.0
        assert planner.config.desired_speed == 2.0

    def test_goal_attraction_no_obstacles(self, planner):
        """Test that robot moves toward goal when no agents/obstacles present."""
        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [1.0, 0.0],
                "radius": 0.3,
            },
            agents=[],  # No other agents
            obstacles=[],  # No obstacles
        )

        action = planner.step(obs)

        # Should move toward goal (positive x direction)
        assert action["vx"] > 0
        assert abs(action["vy"]) < 0.1  # Should be close to zero for straight line

    def test_single_pedestrian_repulsion(self, planner):
        """Test repulsive force from single pedestrian."""
        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [2.0, 0.0],
                "radius": 0.3,
            },
            agents=[
                {
                    "position": [0.5, 0.0],  # Directly in path to goal
                    "velocity": [0.0, 0.0],
                    "radius": 0.3,
                }
            ],
            obstacles=[],
        )

        action = planner.step(obs)

        # Should still move toward goal but with some deflection
        assert action["vx"] > 0  # Still moving toward goal
        # Should have some y-component to avoid pedestrian
        assert abs(action["vy"]) > 0.01

    def test_deterministic_behavior(self, default_config):
        """Test that same observation produces same action with same seed."""
        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [1.0, 1.0],
                "radius": 0.3,
            },
            agents=[{"position": [0.3, 0.3], "velocity": [0.1, -0.1], "radius": 0.3}],
        )

        # Create two planners with same seed
        planner1 = SocialForcePlanner(default_config, seed=42)
        planner2 = SocialForcePlanner(default_config, seed=42)

        action1 = planner1.step(obs)
        action2 = planner2.step(obs)

        # Actions should be identical
        assert abs(action1["vx"] - action2["vx"]) < 1e-10
        assert abs(action1["vy"] - action2["vy"]) < 1e-10

    def test_velocity_limits_respected(self, planner):
        """Test that velocity limits are enforced when safety_clamp=True."""
        planner.config.safety_clamp = True
        planner.config.v_max = 1.0

        # Create scenario that would generate high velocity
        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [100.0, 100.0],  # Very distant goal
                "radius": 0.3,
            },
            agents=[],
            obstacles=[],
        )

        action = planner.step(obs)

        # Check velocity magnitude is within limits
        speed = math.sqrt(action["vx"] ** 2 + action["vy"] ** 2)
        assert speed <= planner.config.v_max + 1e-6  # Small tolerance for floating point

    def test_unicycle_action_space(self, default_config):
        """Test unicycle action space output."""
        default_config.action_space = "unicycle"
        planner = SocialForcePlanner(default_config, seed=42)

        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [1.0, 0.0],  # Moving in +x direction
                "goal": [0.0, 1.0],  # Goal in +y direction
                "radius": 0.3,
            },
            agents=[],
            obstacles=[],
        )

        action = planner.step(obs)

        # Should have keys for unicycle control
        assert "v" in action
        assert "omega" in action
        assert "vx" not in action
        assert "vy" not in action

        # Should have positive angular velocity to turn toward goal
        assert action["omega"] > 0

    def test_force_clipping(self, planner):
        """Test that forces are clipped when enabled."""
        planner.config.clip_force = True
        planner.config.max_force = 10.0

        # This test mainly checks that the clipping code path is exercised
        # Actual force magnitude depends on the Social Force implementation
        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [1.0, 0.0],
                "radius": 0.3,
            },
            agents=[
                {"position": [0.01, 0.0], "velocity": [0.0, 0.0], "radius": 0.3}  # Very close
            ],
        )

        # Should not raise exception
        action = planner.step(obs)
        assert "vx" in action
        assert "vy" in action

    def test_noise_injection(self, default_config):
        """Test that noise affects output when enabled."""
        default_config.noise_std = 0.5
        planner = SocialForcePlanner(default_config, seed=42)

        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [1.0, 0.0],
                "radius": 0.3,
            },
            agents=[],
            obstacles=[],
        )

        # Get multiple actions - should vary due to noise
        actions = [planner.step(obs) for _ in range(5)]

        # Actions should be different due to noise
        vx_values = [a["vx"] for a in actions]
        assert len(set([round(vx, 6) for vx in vx_values])) > 1  # Should have variation

    def test_metadata_generation(self, planner):
        """Test that metadata contains expected information."""
        metadata = planner.get_metadata()

        assert metadata["algorithm"] == "social_force"
        assert "config" in metadata
        assert "config_hash" in metadata
        assert len(metadata["config_hash"]) == 16  # Should be 16 char hash

        # Config should be serializable
        config_str = json.dumps(metadata["config"])
        assert isinstance(config_str, str)

    def test_close_cleanup(self, planner):
        """Test that close cleans up resources."""
        # Setup some internal state

    planner._sim = "dummy_sim"  # noqa: SLF001
    planner._wrapper = "dummy_wrapper"  # noqa: SLF001

    planner.close()

    assert planner._sim is None  # noqa: SLF001
    assert planner._wrapper is None  # noqa: SLF001


class TestObservation:
    """Tests for the Observation dataclass."""

    def test_observation_creation(self):
        """Test observation can be created with required fields."""
        obs = Observation(
            dt=0.1,
            robot={"position": [0, 0], "velocity": [0, 0], "goal": [1, 1], "radius": 0.3},
            agents=[{"position": [0.5, 0.5], "velocity": [0, 0], "radius": 0.3}],
            obstacles=[],
        )

        assert obs.dt == 0.1
        assert len(obs.agents) == 1
        assert len(obs.obstacles) == 0

    def test_observation_default_obstacles(self):
        """Test that obstacles default to empty list."""
        obs = Observation(
            dt=0.1,
            robot={"position": [0, 0], "velocity": [0, 0], "goal": [1, 1], "radius": 0.3},
            agents=[],
        )

        assert obs.obstacles == []


if __name__ == "__main__":
    pytest.main([__file__])
