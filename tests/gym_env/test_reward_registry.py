"""Tests for named reward construction utilities."""

from __future__ import annotations

import math

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.reward import build_reward_function, snqi_step_reward


def _meta(*, collision: bool, success: bool, step: int = 50, max_steps: int = 100) -> dict:
    return {
        "step_of_episode": step,
        "max_sim_steps": max_steps,
        "is_pedestrian_collision": collision,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": success,
        "is_robot_at_goal": success,
    }


def test_snqi_step_reward_penalizes_collision():
    """SNQI step reward should be lower when a collision occurs."""
    reward_clean = snqi_step_reward(_meta(collision=False, success=False))
    reward_collision = snqi_step_reward(_meta(collision=True, success=False))
    assert reward_collision < reward_clean


def test_snqi_step_reward_collision_overrides_success_signal():
    """Collision should force success=0 in step-level SNQI projection."""
    reward_collision_success = snqi_step_reward(
        _meta(collision=True, success=True),
        terminal_bonus=5.0,
    )
    reward_collision_failure = snqi_step_reward(
        _meta(collision=True, success=False),
        terminal_bonus=5.0,
    )
    assert math.isclose(reward_collision_success, reward_collision_failure)


def test_build_reward_function_supports_snqi_step_name():
    """Named reward lookup should return a callable SNQI reward."""
    reward_fn = build_reward_function("snqi_step", reward_kwargs={"terminal_bonus": 0.5})
    value = reward_fn(_meta(collision=False, success=True))
    assert math.isfinite(value)


def test_make_robot_env_builds_reward_func_from_reward_name(monkeypatch):
    """Factory should build reward callables from reward_name when reward_func is omitted."""
    captured: dict[str, object] = {}

    class _DummyEnv:
        pass

    def _fake_create_robot_env(**kwargs):
        captured.update(kwargs)
        return _DummyEnv()

    monkeypatch.setattr(
        "robot_sf.gym_env.environment_factory.EnvironmentFactory.create_robot_env",
        staticmethod(_fake_create_robot_env),
    )
    env = make_robot_env(reward_name="snqi_step")
    assert isinstance(env, _DummyEnv)
    assert callable(captured["reward_func"])
