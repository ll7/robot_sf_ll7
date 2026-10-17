"""Distance-scaled speed shaping is opt-in and uses static footprint clearance."""

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.reward import route_completion_v3_reward
from robot_sf.gym_env.unified_config import RobotSimulationConfig


@pytest.mark.parametrize(
    ("clearance", "speed", "expected"),
    [
        (0.0, 2.0, -4.0),
        (0.25, 2.0, -3.0),
        (0.75, 2.0, -1.0),
        (1.5, 2.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.25, -2.0, -3.0),
    ],
)
def test_speed_cost_scales_with_static_clearance(clearance, speed, expected):
    meta = {"static_obstacle_clearance": clearance, "robot_speed": speed}
    route_completion_v3_reward(meta, weights={"static_speed": -2.0})
    assert meta["reward_terms"].get("static_speed", 0.0) == pytest.approx(expected)


def test_default_static_speed_weight_leaves_scalar_reward_unchanged():
    close = {"static_obstacle_clearance": 0.0, "robot_speed": 2.0}
    far = {"static_obstacle_clearance": 10.0, "robot_speed": 2.0}
    assert route_completion_v3_reward(close) == route_completion_v3_reward(far)


def test_robot_env_supplies_static_clearance_and_physical_speed():
    config = RobotSimulationConfig()
    config.sim_config.ped_density_by_area = 0.0
    env = make_robot_env(
        config=config,
        seed=1001,
        reward_name="route_completion_v3",
        reward_kwargs={"weights": {"static_speed": -2.0}},
    )
    try:
        env.reset(seed=1001)
        info = env.step((0.5, 0.0))[-1]
        meta = info["meta"]
        assert "static_obstacle_clearance" in meta
        assert meta["robot_speed"] == pytest.approx(abs(env.simulator.robots[0].current_speed[0]))
        # Independent point-to-segment oracle over collision geometry (including bounds).
        x, y = env.simulator.robot_poses[0][0]
        distances = []
        point = np.array([x, y])
        for x1, x2, y1, y2 in env.map_def.obstacles_pysf:
            start = np.array([x1, y1])
            delta = np.array([x2 - x1, y2 - y1])
            if np.dot(delta, delta) == 0:
                continue
            along = np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0, 1)
            distances.append(np.linalg.norm(point - start - along * delta))
        expected = min(distances) - env.simulator.robots[0].config.radius
        assert meta["static_obstacle_clearance"] == pytest.approx(expected)
    finally:
        env.close()


@pytest.mark.parametrize("custom_reward", [False, True])
def test_default_reward_skips_static_clearance_geometry(monkeypatch, custom_reward):
    """Default and unrelated custom rewards skip static-reward geometry work."""
    from robot_sf.gym_env import robot_env

    def unexpected_geometry(*_args, **_kwargs):
        pytest.fail("default reward requested static-clearance geometry")

    monkeypatch.setattr(robot_env, "robot_obstacle_clearance", unexpected_geometry, raising=False)
    config = RobotSimulationConfig()
    config.sim_config.ped_density_by_area = 0.0
    reward_options = (
        {"reward_func": lambda _meta: 1.0}
        if custom_reward
        else {"reward_name": "route_completion_v3"}
    )
    env = make_robot_env(config=config, seed=1001, **reward_options)
    try:
        env.reset(seed=1001)
        _, reward, _, _, info = env.step((0.5, 0.0))
        meta = info["meta"]
        assert "static_obstacle_clearance" not in meta
        if custom_reward:
            assert reward == pytest.approx(1.0)
    finally:
        env.close()


def test_later_curriculum_stage_enables_static_clearance_metadata():
    """An opt-in later stage must be recognized when the environment is built."""
    config = RobotSimulationConfig()
    config.sim_config.ped_density_by_area = 0.0
    env = make_robot_env(
        config=config,
        seed=1001,
        reward_name="route_completion_v3",
        reward_curriculum={
            "stages": [
                {"until_episodes": 2},
                {"reward_kwargs": {"weights": {"static_speed": -2.0}}},
            ]
        },
    )
    try:
        env.reset(seed=1001)
        meta = env.step((0.5, 0.0))[-1]["meta"]
        assert "static_obstacle_clearance" in meta
        assert meta["robot_speed"] == pytest.approx(abs(env.simulator.robots[0].current_speed[0]))
    finally:
        env.close()
