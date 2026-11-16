"""Waiting must accrue a meaningful penalty after a short progress window."""

import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.reward import route_completion_v3_reward
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def test_waiting_penalty_uses_a_window_and_clears_on_reset():
    config = RobotSimulationConfig()
    config.sim_config.ped_density_by_area = 0.0
    env = make_robot_env(config=config, seed=1001, reward_name="route_completion_v3")
    try:
        env.reset(seed=1001)
        for step in range(1, 13):
            _, _, terminated, truncated, info = env.step((0.0, 0.0))
            assert not (terminated or truncated)
            penalty = info["meta"]["reward_terms"]["stagnation"]
            assert penalty == (0.0 if step < 10 else -1.2)
        env.reset(seed=1001)
        info = env.step((0.0, 0.0))[-1]
        assert info["meta"]["reward_terms"]["stagnation"] == 0.0
    finally:
        env.close()


@pytest.mark.parametrize(
    ("progress", "expected"), [(0.0, -1.2), (-0.2, -1.2), (0.025, -0.6), (0.1, 0.0)]
)
def test_stagnation_penalizes_window_progress_deficit(progress, expected):
    meta = {"goal_progress_window": progress, "goal_progress_window_ready": True}
    route_completion_v3_reward(meta)
    assert meta["reward_terms"]["stagnation"] == pytest.approx(expected)
