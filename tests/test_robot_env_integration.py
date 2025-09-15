"""
Test the integration of EpisodeMetricsCollector with RobotEnv.
"""

import numpy as np

from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.robot_env import RobotEnv


def test_robot_env_without_metrics():
    """Test that RobotEnv works normally without metrics collection."""
    env = RobotEnv(collect_episode_metrics=False, reward_func=simple_reward)

    assert not hasattr(env, 'episode_metrics_collector') or env.episode_metrics_collector is None

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # Should work normally without metrics
    assert isinstance(reward, (int, float))
    assert 'meta' in info

    # Should not have episode metrics in meta
    meta = info.get('meta', {})
    assert 'mean_interpersonal_distance' not in meta


def test_robot_env_with_metrics():
    """Test that RobotEnv collects metrics when enabled."""
    env = RobotEnv(collect_episode_metrics=True, reward_func=simple_reward)

    assert hasattr(env, 'episode_metrics_collector')
    assert env.episode_metrics_collector is not None

    obs, info = env.reset()

    # Run several steps to collect some data
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    # If episode terminated, check for metrics in info
    if terminated or truncated:
        meta = info.get('meta', {})
        # Episode metrics should be present
        assert 'mean_interpersonal_distance' in meta
        assert 'ped_force_q50' in meta
        assert 'ped_force_q90' in meta
        assert 'ped_force_q95' in meta

        # Values should be either float or NaN
        for key in ['mean_interpersonal_distance', 'ped_force_q50', 'ped_force_q90', 'ped_force_q95']:
            value = meta[key]
            assert isinstance(value, (int, float, np.number)) or np.isnan(value)


def test_robot_env_metrics_reset():
    """Test that metrics collector is reset between episodes."""
    env = RobotEnv(collect_episode_metrics=True, reward_func=simple_reward)

    # First episode
    obs, info = env.reset()
    env.step(env.action_space.sample())

    # Check that collector has some data
    collector = env.episode_metrics_collector
    assert len(collector.robot_ped_distances) > 0 or len(collector.current_ped_forces) > 0

    # Reset for second episode
    obs, info = env.reset()

    # Check that collector was reset
    assert len(collector.robot_ped_distances) == 0
    assert len(collector.current_ped_forces) == 0


def test_robot_env_collect_episode_metrics_method():
    """Test the _collect_episode_metrics helper method."""
    env = RobotEnv(collect_episode_metrics=True, reward_func=simple_reward)
    obs, info = env.reset()

    # Call the data collection method directly
    initial_distances = len(env.episode_metrics_collector.robot_ped_distances)
    env._collect_episode_metrics()

    # Should have collected data (unless no pedestrians present)
    final_distances = len(env.episode_metrics_collector.robot_ped_distances)
    # Data should be collected (distances should increase or stay same if no peds)
    assert final_distances >= initial_distances
