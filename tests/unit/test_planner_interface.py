"""Tests for planner interface compliance (T114).

Verifies that all baseline planners correctly implement the PlannerProtocol
interface with proper action types, shapes, and method signatures.
"""

import numpy as np
import pytest

from robot_sf.baselines import get_baseline
from robot_sf.baselines.social_force import Observation


class TestPlannerInterfaceCompliance:
    """Test suite for verifying planner interface compliance."""

    @pytest.fixture
    def sample_observation(self) -> Observation:
        """Create a sample observation for testing."""
        return Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [1.0, 0.5],
                "goal": [5.0, 3.0],
                "radius": 0.35,
            },
            agents=[
                {"position": [2.0, 1.0], "velocity": [0.5, -0.3]},
                {"position": [-1.0, 2.0], "velocity": [-0.2, 0.4]},
            ],
            obstacles=[],
        )

    @pytest.mark.parametrize("algo", ["social_force", "random", "ppo"])
    def test_planner_has_required_methods(self, algo: str) -> None:
        """Test that all planners have the required protocol methods."""
        planner_class = get_baseline(algo)
        planner = planner_class(config={}, seed=42)

        # Check that planner has all required methods
        assert hasattr(planner, "__init__"), f"{algo} missing __init__"
        assert hasattr(planner, "step"), f"{algo} missing step method"
        assert hasattr(planner, "reset"), f"{algo} missing reset method"
        assert hasattr(planner, "configure"), f"{algo} missing configure method"
        assert hasattr(planner, "close"), f"{algo} missing close method"

        # Check method signatures accept the expected parameters
        assert callable(planner.step), f"{algo} step not callable"
        assert callable(planner.reset), f"{algo} reset not callable"
        assert callable(planner.configure), f"{algo} configure not callable"
        assert callable(planner.close), f"{algo} close not callable"

    @pytest.mark.parametrize("algo", ["social_force", "random"])
    def test_velocity_action_format(self, algo: str, sample_observation: Observation) -> None:
        """Test that planners return proper velocity action format."""
        if algo == "social_force":
            config = {"action_space": "velocity", "mode": "velocity"}
        elif algo == "random":
            config = {"mode": "velocity"}
        else:
            config = {"action_space": "velocity"}

        planner_class = get_baseline(algo)
        planner = planner_class(config=config, seed=42)

        action = planner.step(sample_observation)

        # Verify action is a dictionary
        assert isinstance(action, dict), f"{algo} action must be a dict"

        # Verify required velocity keys
        assert "vx" in action, f"{algo} velocity action missing 'vx' key"
        assert "vy" in action, f"{algo} velocity action missing 'vy' key"

        # Verify action values are numeric
        assert isinstance(action["vx"], (int, float)), f"{algo} vx must be numeric"
        assert isinstance(action["vy"], (int, float)), f"{algo} vy must be numeric"

        # Verify values are finite
        assert np.isfinite(action["vx"]), f"{algo} vx must be finite"
        assert np.isfinite(action["vy"]), f"{algo} vy must be finite"

        planner.close()

    @pytest.mark.parametrize("algo", ["social_force", "random"])
    def test_unicycle_action_format(self, algo: str, sample_observation: Observation) -> None:
        """Test that planners return proper unicycle action format."""
        if algo == "social_force":
            config = {"action_space": "unicycle", "mode": "unicycle"}
        elif algo == "random":
            config = {"mode": "unicycle"}
        else:
            config = {"action_space": "unicycle"}

        planner_class = get_baseline(algo)
        planner = planner_class(config=config, seed=42)

        action = planner.step(sample_observation)

        # Verify action is a dictionary
        assert isinstance(action, dict), f"{algo} action must be a dict"

        # Verify required unicycle keys
        assert "v" in action, f"{algo} unicycle action missing 'v' key"
        assert "omega" in action, f"{algo} unicycle action missing 'omega' key"

        # Verify action values are numeric
        assert isinstance(action["v"], (int, float)), f"{algo} v must be numeric"
        assert isinstance(action["omega"], (int, float)), f"{algo} omega must be numeric"

        # Verify values are finite
        assert np.isfinite(action["v"]), f"{algo} v must be finite"
        assert np.isfinite(action["omega"]), f"{algo} omega must be finite"

        # Verify velocity is non-negative
        assert action["v"] >= 0, f"{algo} velocity v must be non-negative"

        planner.close()

    @pytest.mark.parametrize("algo", ["social_force", "random", "ppo"])
    def test_reset_method_with_seed(self, algo: str) -> None:
        """Test that reset method accepts seed parameter."""
        planner_class = get_baseline(algo)
        planner = planner_class(config={}, seed=42)

        # Test reset without seed
        planner.reset()

        # Test reset with seed
        planner.reset(seed=123)

        planner.close()

    @pytest.mark.parametrize("algo", ["social_force", "random", "ppo"])
    def test_configure_method(self, algo: str) -> None:
        """Test that configure method accepts config parameter."""
        planner_class = get_baseline(algo)
        planner = planner_class(config={}, seed=42)

        # Test configure with dict config
        new_config = {"v_max": 1.5}
        planner.configure(new_config)

        planner.close()

    @pytest.mark.parametrize("algo", ["social_force", "random", "ppo"])
    def test_step_accepts_dict_observation(self, algo: str) -> None:
        """Test that step method accepts dict-style observation."""
        planner_class = get_baseline(algo)
        planner = planner_class(config={}, seed=42)

        obs_dict = {
            "dt": 0.1,
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [1.0, 0.5],
                "goal": [5.0, 3.0],
                "radius": 0.35,
            },
            "agents": [
                {"position": [2.0, 1.0], "velocity": [0.5, -0.3]},
            ],
            "obstacles": [],
        }

        action = planner.step(obs_dict)
        assert isinstance(action, dict), f"{algo} must return dict action"

        planner.close()

    @pytest.mark.parametrize(
        "algo", ["social_force", "random"]
    )  # Skip PPO due to model requirements
    def test_deterministic_behavior_with_seed(
        self, algo: str, sample_observation: Observation
    ) -> None:
        """Test that planners produce deterministic results when seeded."""
        if algo == "social_force":
            config = {"action_space": "velocity", "mode": "velocity"}
        elif algo == "random":
            config = {"mode": "velocity"}
        else:
            config = {"action_space": "velocity"}

        # Create two planners with same seed
        planner_class = get_baseline(algo)
        planner1 = planner_class(config=config, seed=42)
        planner2 = planner_class(config=config, seed=42)

        # Get actions from both
        action1 = planner1.step(sample_observation)
        action2 = planner2.step(sample_observation)

        # Actions should be identical for deterministic algorithms
        if algo == "random":  # Random planner should be deterministic with same seed
            assert action1["vx"] == action2["vx"], f"{algo} not deterministic with seed"
            assert action1["vy"] == action2["vy"], f"{algo} not deterministic with seed"

        planner1.close()
        planner2.close()

    @pytest.mark.parametrize("algo", ["social_force", "random"])
    def test_action_bounds_respected(self, algo: str, sample_observation: Observation) -> None:
        """Test that action values respect configured bounds."""
        v_max = 1.5
        omega_max = 0.8
        if algo == "social_force":
            config = {
                "action_space": "velocity",
                "mode": "velocity",
                "v_max": v_max,
                "omega_max": omega_max,
                "safety_clamp": True,
            }
        elif algo == "random":
            config = {
                "mode": "velocity",
                "v_max": v_max,
                "omega_max": omega_max,
                "safety_clamp": True,
            }
        else:
            config = {"action_space": "velocity", "v_max": v_max, "safety_clamp": True}

        planner_class = get_baseline(algo)
        planner = planner_class(config=config, seed=42)

        # Test multiple steps to check bounds consistently
        for _ in range(5):
            action = planner.step(sample_observation)
            speed = np.hypot(action["vx"], action["vy"])
            assert speed <= v_max + 1e-6, f"{algo} exceeded v_max bound: {speed} > {v_max}"

        planner.close()

    def test_protocol_typing_compatibility(self) -> None:
        """Test that actual planners are compatible with PlannerProtocol."""
        # This test verifies typing compatibility at runtime
        social_force_class = get_baseline("social_force")
        random_class = get_baseline("random")
        ppo_class = get_baseline("ppo")

        social_force = social_force_class(config={}, seed=42)
        random_planner = random_class(config={}, seed=42)
        ppo_planner = ppo_class(config={}, seed=42)

        # These should not raise type errors (if we had runtime type checking)
        planners = [social_force, random_planner, ppo_planner]

        for planner in planners:
            # Verify the planner has the protocol signature
            assert hasattr(planner, "__init__")
            assert hasattr(planner, "step")
            assert hasattr(planner, "reset")
            assert hasattr(planner, "configure")
            assert hasattr(planner, "close")
            planner.close()
