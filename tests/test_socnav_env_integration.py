"""Integration smoke: SocNav structured obs + planner policy with RobotEnv."""

import numpy as np
import pytest

from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.socnav import SocNavPlannerPolicy
from robot_sf.robot.bicycle_drive import BicycleDriveSettings


def test_socnav_policy_runs_single_step():
    """Ensure policy and structured observation mode run a step without errors."""
    env = RobotEnv(env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT))
    obs, _ = env.reset()
    policy = SocNavPlannerPolicy()
    action = policy.act(obs)
    new_obs, _, _, _, info = env.step(action)
    assert env.observation_space.contains(new_obs)
    assert "success" in info


def test_socnav_structured_observation_exposes_robot_velocity_xy():
    """Structured SocNav observations should expose explicit world-frame robot velocity."""
    env = RobotEnv(env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT))
    obs, _ = env.reset()

    assert "velocity_xy" in obs["robot"]
    assert len(obs["robot"]["velocity_xy"]) == 2


def test_socnav_structured_observation_exposes_robot_angular_velocity():
    """Structured SocNav observations should expose explicit robot angular velocity."""
    env = RobotEnv(env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT))
    obs, _ = env.reset()

    assert "angular_velocity" in obs["robot"]
    assert len(obs["robot"]["angular_velocity"]) == 1


def test_socnav_bicycle_observation_reports_turn_rate_not_heading():
    """Bicycle-drive SocNav observations should expose angular rate, not orientation."""
    env = RobotEnv(
        env_config=RobotSimulationConfig(
            observation_mode=ObservationMode.SOCNAV_STRUCT,
            robot_config=BicycleDriveSettings(max_accel=2.0, max_velocity=3.0, wheelbase=1.0),
        )
    )
    obs, _ = env.reset()
    new_obs, _, _, _, _ = env.step(np.array([1.0, 0.4], dtype=np.float32))

    dt = env.config.sim_config.time_per_step_in_secs
    prev_heading = float(obs["robot"]["heading"][0])
    new_heading = float(new_obs["robot"]["heading"][0])
    heading_delta = ((new_heading - prev_heading + np.pi) % (2.0 * np.pi)) - np.pi
    expected_turn_rate = heading_delta / dt

    assert new_obs["robot"]["angular_velocity"][0] == pytest.approx(expected_turn_rate, abs=1e-5)
    assert new_obs["robot"]["angular_velocity"][0] != pytest.approx(new_heading)
