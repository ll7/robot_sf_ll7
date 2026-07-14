"""Gymnasium API contracts for public factory-produced environments."""

from collections.abc import Mapping

import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import (
    check_action_space,
    check_observation_space,
    check_reset_return_type,
)
from gymnasium.utils.passive_env_checker import (
    check_obs,
    env_reset_passive_checker,
    env_step_passive_checker,
)

from robot_sf.gym_env.crowd_sim_env import CrowdSimulationConfig
from robot_sf.gym_env.environment_factory import (
    make_crowd_sim_env,
    make_image_robot_env,
    make_multi_robot_env,
    make_pedestrian_env,
    make_robot_env,
)
from robot_sf.gym_env.unified_config import (
    MultiRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)


def _assert_contract_metadata_keys(info: Mapping[str, object], required: set[str]) -> None:
    """Assert observed reset payload contains at least the required stable keys."""
    for key in required:
        assert key in info


def _assert_step_contract(
    step_result: tuple[object, float, bool, bool, Mapping[str, object]],
) -> None:
    """Assert 5-tuple Gymnasium step contract for terminated/truncated semantics."""
    assert len(step_result) == 5
    _, _reward, terminated, truncated, info = step_result
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, Mapping)


@pytest.mark.parametrize(
    "factory,factory_kwargs,required_reset_keys,supports_env_checker",
    [
        (
            make_robot_env,
            {
                "config": RobotSimulationConfig(map_id="uni_campus_big"),
            },
            {"map_id", "sim_time_in_secs", "time_per_step_in_secs", "max_sim_steps", "seed"},
            True,
        ),
        (
            make_pedestrian_env,
            {
                "config": PedestrianSimulationConfig(map_id="uni_campus_big"),
                "robot_model": None,
            },
            {"map_id", "sim_time_in_secs", "time_per_step_in_secs", "max_sim_steps", "seed"},
            True,
        ),
        (
            make_multi_robot_env,
            {
                "num_robots": 2,
                "config": MultiRobotConfig(map_id="uni_campus_big"),
            },
            {
                "map_id",
                "sim_time_in_secs",
                "time_per_step_in_secs",
                "max_sim_steps",
                "num_robots",
                "seed",
            },
            False,
        ),
        (
            make_crowd_sim_env,
            {
                "config": CrowdSimulationConfig(map_id="uni_campus_big"),
            },
            {
                "map_id",
                "time_per_step_in_secs",
                "sim_time_in_secs",
                "max_sim_steps",
                "num_pedestrians",
            },
            True,
        ),
    ],
)
def test_public_factories_adhere_to_gymnasium_step_reset_contract(
    factory,
    factory_kwargs,
    required_reset_keys,
    supports_env_checker,
):
    """Run Gymnasium contract checks for public factories excluding image-specific render constraints."""
    env = factory(**factory_kwargs)
    try:
        check_action_space(env.action_space)
        if supports_env_checker:
            check_observation_space(env.observation_space)
            check_reset_return_type(env)

            reset_result = env_reset_passive_checker(env)
            reset_obs, reset_info = reset_result
            check_obs(reset_obs, env.observation_space, "reset")
        else:
            reset_result = env.reset()
            reset_obs, reset_info = reset_result

        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2
        assert isinstance(reset_info, Mapping)
        _assert_contract_metadata_keys(reset_info, required_reset_keys)

        action = env.action_space.sample()
        if supports_env_checker:
            step_result = env_step_passive_checker(env, action)
        else:
            step_result = env.step(action)
        _assert_step_contract(step_result)
    finally:
        env.close()


@pytest.mark.base_sensitive
def test_image_factory_smoke_path_keeps_gymnasium_tuple_shape():
    """Smoke test for image envs, which still need a focused manual check."""
    env = make_image_robot_env(config=RobotSimulationConfig(map_id="uni_campus_big"))
    try:
        assert isinstance(env.action_space, spaces.Space)
        check_action_space(env.action_space)
        check_observation_space(env.observation_space)

        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        reset_info = result[1]
        assert isinstance(reset_info, Mapping)
        _assert_contract_metadata_keys(
            reset_info,
            {"map_id", "sim_time_in_secs", "time_per_step_in_secs", "max_sim_steps", "seed"},
        )

        action = env.action_space.sample()
        step_result = env.step(action)
        _assert_step_contract(step_result)
    finally:
        env.close()
