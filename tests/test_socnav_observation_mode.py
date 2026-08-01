"""Smoke tests for SocNavBench-compatible observation modes."""

import numpy as np
import pytest

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def test_socnav_struct_observation_contains_expected_keys():
    """Ensure SocNav structured observations align with the declared space."""
    env = RobotEnv(env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT))
    obs, _ = env.reset()

    assert env.observation_space.contains(obs)
    assert "robot" in obs
    assert "pedestrians" in obs
    assert obs["robot"]["position"].shape == (2,)
    assert (
        obs["pedestrians"]["positions"].shape
        == env.observation_space["pedestrians"]["positions"].shape
    )


def test_social_graph_observation_is_fixed_and_runnable_on_reset_and_step():
    """SOCIAL_GRAPH should expose bounded, space-valid reset and step observations."""
    config = RobotSimulationConfig(observation_mode=ObservationMode.SOCIAL_GRAPH)
    config.sim_config.max_total_pedestrians = 3
    env = RobotEnv(env_config=config)
    try:
        first, _ = env.reset(seed=6438)
        repeated, _ = env.reset(seed=6438)
        step_obs, _, _, _, _ = env.step(env.action_space.sample())

        expected_keys = {
            "robot_features",
            "pedestrian_features",
            "pedestrian_mask",
            "pedestrian_count",
            "pedestrian_history",
            "static_obstacle_features",
            "static_obstacle_mask",
            "static_obstacle_count",
            "edge_index",
            "edge_type",
            "edge_mask",
        }
        assert set(first) == expected_keys
        assert env.observation_space.contains(first)
        assert env.observation_space.contains(step_obs)
        np.testing.assert_array_equal(first["pedestrian_features"], repeated["pedestrian_features"])
        np.testing.assert_array_equal(first["edge_mask"], repeated["edge_mask"])
        assert first["pedestrian_features"].shape == (3, 7)
        assert first["pedestrian_history"].shape == (1, 3, 7)
        assert first["edge_index"].shape == (2, 3)
        assert first["edge_type"].shape == (3,)
        assert first["edge_mask"].shape == (3,)
        assert int(first["pedestrian_count"][0]) == int(first["pedestrian_mask"].sum())
        assert int(first["pedestrian_count"][0]) <= 3
    finally:
        env.close()


def test_social_graph_rejects_unsupported_privileged_observation_setup():
    """Unsupported critic-only fields must not silently fall back to another mode."""
    config = RobotSimulationConfig(observation_mode=ObservationMode.SOCIAL_GRAPH)

    with pytest.raises(ValueError, match="asymmetric critic"):
        RobotEnv(env_config=config, asymmetric_critic=True)
