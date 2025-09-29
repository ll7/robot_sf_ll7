"""Integration smoke tests for Social Force planner.

These tests verify that the Social Force planner integrates correctly
with the benchmark system and produces valid outputs.
"""

import json

import pytest

from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.social_force import Observation, SFPlannerConfig, SocialForcePlanner


class TestSocialForceIntegration:
    """Integration tests for Social Force planner."""

    def test_baseline_registry_integration(self):
        """Test that Social Force planner is available via baseline registry."""
        baselines = list_baselines()
        assert "baseline_sf" in baselines

        sf_class = get_baseline("baseline_sf")
        assert sf_class == SocialForcePlanner

    def test_minimal_scenario_completion(self):
        """Test planner can complete a minimal scenario without errors."""
        config = SFPlannerConfig(
            v_max=1.0,
            desired_speed=0.8,
            action_space="velocity",
            noise_std=0.0,  # Deterministic
        )

        planner = SocialForcePlanner(config, seed=42)

        # Start position and goal
        start_pos = [0.5, 3.0]
        goal_pos = [9.5, 3.0]
        goal_tolerance = 0.3
        max_steps = 100
        dt = 0.1

        robot_pos = start_pos.copy()
        robot_vel = [0.0, 0.0]

        trajectory = []

        for step in range(max_steps):
            obs = Observation(
                dt=dt,
                robot={
                    "position": robot_pos,
                    "velocity": robot_vel,
                    "goal": goal_pos,
                    "radius": 0.3,
                },
                agents=[],  # Empty scenario
                obstacles=[],
            )

            action = planner.step(obs)

            # Verify action format
            assert "vx" in action
            assert "vy" in action
            assert isinstance(action["vx"], int | float)
            assert isinstance(action["vy"], float | int)

            # Update robot state (simple integration)
            robot_vel = [action["vx"], action["vy"]]
            robot_pos[0] += robot_vel[0] * dt
            robot_pos[1] += robot_vel[1] * dt

            trajectory.append(
                {
                    "step": step,
                    "position": robot_pos.copy(),
                    "velocity": robot_vel.copy(),
                    "action": action.copy(),
                },
            )

            # Check if goal reached
            goal_distance = (
                (robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2
            ) ** 0.5

            if goal_distance < goal_tolerance:
                break

        # Should reach goal within time limit
        final_distance = (
            (robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2
        ) ** 0.5
        assert final_distance < goal_tolerance, (
            f"Failed to reach goal, final distance: {final_distance}"
        )

        # Trajectory should show progress toward goal
        assert len(trajectory) > 5  # Should take multiple steps
        assert trajectory[0]["position"][0] < trajectory[-1]["position"][0]  # Moving right

        planner.close()

    def test_pedestrian_avoidance_scenario(self):
        """Test planner avoids pedestrians while reaching goal."""
        config = SFPlannerConfig(
            v_max=1.5,
            desired_speed=1.0,
            A=5.0,  # Strong social force
            action_space="velocity",
            noise_std=0.0,
        )

        planner = SocialForcePlanner(config, seed=42)

        start_pos = [0.0, 0.0]
        goal_pos = [5.0, 0.0]
        pedestrian_pos = [2.5, 0.0]  # Directly in path

        robot_pos = start_pos.copy()
        robot_vel = [0.0, 0.0]

        max_distance_to_ped = 0.0
        min_distance_to_ped = float("inf")

        for _step in range(100):
            obs = Observation(
                dt=0.1,
                robot={
                    "position": robot_pos,
                    "velocity": robot_vel,
                    "goal": goal_pos,
                    "radius": 0.3,
                },
                agents=[{"position": pedestrian_pos, "velocity": [0.0, 0.0], "radius": 0.3}],
                obstacles=[],
            )

            action = planner.step(obs)

            # Update robot position
            robot_vel = [action["vx"], action["vy"]]
            robot_pos[0] += robot_vel[0] * 0.1
            robot_pos[1] += robot_vel[1] * 0.1

            # Track distance to pedestrian
            ped_distance = (
                (robot_pos[0] - pedestrian_pos[0]) ** 2 + (robot_pos[1] - pedestrian_pos[1]) ** 2
            ) ** 0.5
            min_distance_to_ped = min(min_distance_to_ped, ped_distance)
            max_distance_to_ped = max(max_distance_to_ped, ped_distance)

            # Stop if goal reached
            goal_distance = (
                (robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2
            ) ** 0.5
            if goal_distance < 0.3:
                break

        # Should maintain safe distance from pedestrian
        safe_distance = 0.6  # Sum of radii
        assert min_distance_to_ped > safe_distance * 0.8, (
            f"Too close to pedestrian: {min_distance_to_ped}"
        )

        # Should show avoidance behavior (deviating from straight line)
        assert max_distance_to_ped > min_distance_to_ped + 0.5, (
            "No clear avoidance behavior detected"
        )

        planner.close()

    def test_unicycle_action_space_scenario(self):
        """Test planner works with unicycle action space."""
        config = SFPlannerConfig(
            action_space="unicycle",
            v_max=1.0,
            omega_max=1.0,
            desired_speed=0.8,
            noise_std=0.0,
        )

        planner = SocialForcePlanner(config, seed=42)

        robot_pos = [0.0, 0.0]
        robot_vel = [0.0, 0.0]
        goal_pos = [3.0, 3.0]

        actions_taken = []

        for _step in range(50):
            obs = Observation(
                dt=0.1,
                robot={
                    "position": robot_pos,
                    "velocity": robot_vel,
                    "goal": goal_pos,
                    "radius": 0.3,
                },
                agents=[],
                obstacles=[],
            )

            action = planner.step(obs)
            actions_taken.append(action)

            # Verify unicycle action format
            assert "v" in action
            assert "omega" in action
            assert "vx" not in action
            assert "vy" not in action

            # Simple unicycle model integration
            # This is a simplified version - real integration would be more complex
            v = action["v"]
            _omega = action["omega"]  # Not used in this simplified integration

            # Update robot state (simplified)
            if len(actions_taken) > 1:
                # Use previous velocity for position update
                robot_pos[0] += robot_vel[0] * 0.1
                robot_pos[1] += robot_vel[1] * 0.1

            # Convert unicycle to velocity (simplified)
            robot_vel = [v, 0.0]  # Simplified - should use heading

            # Check goal
            goal_distance = (
                (robot_pos[0] - goal_pos[0]) ** 2 + (robot_pos[1] - goal_pos[1]) ** 2
            ) ** 0.5
            if goal_distance < 0.5:
                break

        # Should generate reasonable actions
        assert len(actions_taken) > 5
        velocities = [a["v"] for a in actions_taken]
        assert max(velocities) > 0.1  # Should have forward motion
        assert max(velocities) <= config.v_max + 1e-6  # Should respect limits

        planner.close()

    def test_configuration_loading_from_yaml(self):
        """Test that configuration can be loaded from YAML-like dict."""
        # Simulate loading from YAML
        yaml_config = {
            "mode": "velocity",
            "v_max": 2.5,
            "omega_max": 1.2,
            "desired_speed": 1.5,
            "tau": 0.4,
            "A": 6.0,
            "B": 0.4,
            "lambda_anisotropy": 2.5,
            "cutoff_radius": 15.0,
            "A_obs": 12.0,
            "B_obs": 0.1,
            "integration": "euler",
            "clip_force": True,
            "max_force": 150.0,
            "noise_std": 0.1,
            "action_space": "unicycle",
            "safety_clamp": True,
        }

        planner = SocialForcePlanner(yaml_config, seed=42)

        # Verify configuration was loaded correctly
        assert planner.config.v_max == 2.5
        assert planner.config.desired_speed == 1.5
        assert planner.config.A == 6.0
        assert planner.config.action_space == "unicycle"
        assert planner.config.noise_std == 0.1

        # Test that it can still generate actions
        obs = Observation(
            dt=0.1,
            robot={
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [1.0, 1.0],
                "radius": 0.3,
            },
            agents=[],
            obstacles=[],
        )

        action = planner.step(obs)
        assert "v" in action and "omega" in action

        planner.close()

    def test_metadata_for_episode_output(self):
        """Test that planner provides metadata suitable for episode output."""
        config = SFPlannerConfig(v_max=1.5, A=4.0)
        planner = SocialForcePlanner(config, seed=42)

        metadata = planner.get_metadata()

        # Should contain required fields for episode output
        assert metadata["algorithm"] == "social_force"
        assert "config" in metadata
        assert "config_hash" in metadata

        # Config should be JSON serializable
        config_json = json.dumps(metadata["config"])
        assert isinstance(config_json, str)

        # Hash should be consistent for same config
        planner2 = SocialForcePlanner(config, seed=123)  # Different seed
        metadata2 = planner2.get_metadata()
        assert metadata["config_hash"] == metadata2["config_hash"]  # Same config = same hash

        # Different config should have different hash
        config3 = SFPlannerConfig(v_max=2.0, A=4.0)  # Different v_max
        planner3 = SocialForcePlanner(config3, seed=42)
        metadata3 = planner3.get_metadata()
        assert metadata["config_hash"] != metadata3["config_hash"]

        planner.close()
        planner2.close()
        planner3.close()


@pytest.mark.slow
class TestBatchRunnerIntegration:
    """Tests that require the full benchmark runner (marked as slow)."""

    def test_episode_schema_compliance(self):
        """Test that a Social Force episode produces schema-compliant output.

        This is a conceptual test - actual implementation would require
        integrating with the batch runner system.
        """
        # This test would ideally run a small batch through the benchmark system
        # and validate the output against the episode schema.

        config = SFPlannerConfig()
        planner = SocialForcePlanner(config, seed=42)

        # Simulate episode data that would be generated
        episode_metadata = planner.get_metadata()

        # Basic validation of structure
        assert "algorithm" in episode_metadata
        assert "config" in episode_metadata
        assert "config_hash" in episode_metadata

        # The actual schema validation would happen in the batch runner
        # when integrated with the CLI system

        planner.close()

    @pytest.mark.skipif(True, reason="Requires full CLI integration")
    def test_cli_integration(self):
        """Test Social Force planner works with CLI --algo baseline_sf.

        This test would require the CLI to be updated to support baseline algorithms.
        """
        # This would test something like:
        # robot_sf_bench run --matrix test_scenarios.yaml --algo baseline_sf --out results.jsonl
        pass


if __name__ == "__main__":
    pytest.main([__file__])
