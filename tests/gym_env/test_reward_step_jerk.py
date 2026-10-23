"""Reward jerk must describe the current step, independently of historical metrics."""

from types import SimpleNamespace

import pytest

from robot_sf.gym_env.reward import route_completion_v3_reward
from robot_sf.gym_env.snqi_proxy import StepSNQIProxy


def _sim(x):
    return SimpleNamespace(robot_poses=[((x, 0.0), 0.0)])


def test_past_jerk_does_not_penalize_a_later_smooth_step():
    """Finite differences give jerk 0.5 then zero; the metric mean remains 0.25."""
    proxy = StepSNQIProxy()
    proxy.prime(_sim(0.0))
    for x in (1.0, 3.0, 6.5, 11.5):
        meta = proxy.compute_step_metrics(_sim(x), dt=1.0)
    assert meta["jerk_mean"] == pytest.approx(0.25)
    route_completion_v3_reward(meta, weights={"smoothness": -1.0})
    assert meta["reward_terms"]["smoothness"] == 0.0


def test_constant_jerk_gives_constant_step_penalty_after_warmup():
    """Third finite difference of x=t^3/6 is one at dt=1, giving -1/5."""
    proxy = StepSNQIProxy()
    proxy.prime(_sim(0.0))
    for t in range(1, 9):
        meta = proxy.compute_step_metrics(_sim(t**3 / 6.0), dt=1.0)
        route_completion_v3_reward(meta, weights={"smoothness": -1.0})
        expected = -0.2 if t >= 3 else 0.0
        assert meta["reward_terms"]["smoothness"] == pytest.approx(expected)


def test_running_metric_alone_is_not_a_per_step_reward():
    meta = {"jerk_mean": 5.0}
    route_completion_v3_reward(meta, weights={"smoothness": -1.0})
    assert meta["reward_terms"]["smoothness"] == 0.0
