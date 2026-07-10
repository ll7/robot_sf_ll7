"""Integration smoke: SocNav structured obs + planner policy with RobotEnv."""

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.socnav import SocNavPlannerPolicy
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings
from robot_sf.sim.sim_config import SimulationSettings


def test_robot_env_delays_actions_and_resets_the_queue_between_episodes() -> None:
    """The configured delay applies in the environment loop, not just campaign provenance."""
    env = RobotEnv(
        env_config=RobotSimulationConfig(
            sim_config=SimulationSettings(action_latency_steps=1),
        )
    )
    _obs, reset_info = env.reset(seed=123)

    assert reset_info["action_latency"]["effective_steps"] == 1
    _obs, _reward, _terminated, _truncated, first_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert first_info["meta"]["requested_action"] == pytest.approx((1.0, 0.0))
    assert first_info["meta"]["action"] == pytest.approx((0.0, 0.0))
    assert env.simulator.robots[0].current_speed == pytest.approx((0.0, 0.0))

    _obs, _reward, _terminated, _truncated, second_info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )
    assert second_info["meta"]["action"] == pytest.approx((1.0, 0.0))
    assert env.simulator.robots[0].current_speed[0] > 0.0

    env.reset(seed=123)
    _obs, _reward, _terminated, _truncated, reset_step_info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )
    assert reset_step_info["meta"]["action"] == pytest.approx((0.0, 0.0))


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


def test_socnav_structured_observation_can_expose_critic_privileged_state():
    """Asymmetric critic mode should append a critic-only privileged vector."""
    env = RobotEnv(
        env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT),
        asymmetric_critic=True,
    )
    obs, _ = env.reset()

    assert "critic_privileged_state" in obs
    assert obs["critic_privileged_state"].ndim == 1
    assert env.observation_space.contains(obs)
    assert env._critic_obs_space is not None
    assert "critic_privileged_state" not in env._critic_obs_space.spaces

    meta = env.state.meta_dict()
    expected_tail = np.array(
        [
            meta.get("step_of_episode", 0) or 0,
            env.state.sim_time_elapsed,
            meta.get("max_sim_steps", env.state.max_sim_steps),
            meta.get("distance_to_goal", 0.0),
            meta.get("prev_distance_to_goal", 0.0),
            float(bool(meta.get("is_route_complete"))),
            float(bool(meta.get("is_timesteps_exceeded"))),
            float(bool(meta.get("is_pedestrian_collision"))),
            float(bool(meta.get("is_robot_collision"))),
            float(bool(meta.get("is_obstacle_collision"))),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(obs["critic_privileged_state"][-10:], expected_tail)


def test_socnav_structured_observation_can_expose_critic_privileged_state_via_factory():
    """The public factory should pass asymmetric critic through to RobotEnv."""
    env = make_robot_env(
        config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT),
        asymmetric_critic=True,
    )
    obs, _ = env.reset()

    assert "critic_privileged_state" in obs
    assert env.observation_space.contains(obs)


def test_socnav_holonomic_observation_distinguishes_speed_from_velocity_xy():
    """Holonomic structured observations should expose world velocity separately from speed."""
    env = RobotEnv(
        env_config=RobotSimulationConfig(
            observation_mode=ObservationMode.SOCNAV_STRUCT,
            robot_config=HolonomicDriveSettings(
                radius=0.3,
                max_speed=2.0,
                max_angular_speed=2.0,
                command_mode="vx_vy",
            ),
        )
    )
    env.reset()
    obs, _, _, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))

    assert obs["robot"]["velocity_xy"] == pytest.approx((0.0, 1.0))
    assert obs["robot"]["speed"] == pytest.approx((1.0, 0.0))
    assert obs["robot"]["heading"][0] == pytest.approx(np.pi / 2)


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


def test_socnav_critic_privileged_state_metadata_ordering():
    """Verify privileged-state metadata portion ordering, shape, and expected reset/step values."""
    env = RobotEnv(
        env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT),
        asymmetric_critic=True,
    )
    obs, _ = env.reset()
    crit = obs["critic_privileged_state"]

    # Check declared shape matches emitted shape
    crit_space = env.observation_space["critic_privileged_state"]
    assert crit.shape == crit_space.shape
    assert crit.dtype == np.float32

    # Metadata lives in the last 10 elements
    meta = crit[-10:]

    assert meta[0] == 0.0, "step_of_episode should be 0 after reset"
    assert meta[1] == 0.0, "sim_time_elapsed should be 0 after reset"
    assert meta[2] > 0.0, "max_sim_steps should be positive"
    assert meta[5] == 0.0, "is_route_complete should be 0 after reset"
    assert meta[7] == 0.0, "is_pedestrian_collision should be 0 after reset"
    assert meta[8] == 0.0, "is_robot_collision should be 0 after reset"
    assert meta[9] == 0.0, "is_obstacle_collision should be 0 after reset"

    # After one step, step_of_episode must increment
    obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    new_crit = obs["critic_privileged_state"]
    assert new_crit[-10:][0] == 1.0, "step_of_episode should be 1 after one step"
    assert env.observation_space.contains(obs)


def test_socnav_critic_privileged_state_cache_reuse():
    """Prove cached leaf traversal + metadata batch survive multi-step resets without corruption."""
    env = RobotEnv(
        env_config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT),
        asymmetric_critic=True,
    )
    obs, _ = env.reset()

    assert env.observation_space.contains(obs)
    prev_step = 0.0

    for i in range(10):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        crit = obs["critic_privileged_state"]

        # step_of_episode (metadata[0]) must increment monotonically
        assert crit[-10] == prev_step + 1.0
        prev_step = crit[-10]

        # Every step must satisfy the declared observation space
        assert env.observation_space.contains(obs)
