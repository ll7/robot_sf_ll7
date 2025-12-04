"""Smoke tests for SocNavBench-compatible observation mode."""

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
