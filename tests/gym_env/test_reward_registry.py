"""Tests for named reward construction utilities."""

from __future__ import annotations

import math

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.reward import (
    build_reward_function,
    punish_action_reward,
    route_completion_v2_reward,
    simple_ped_reward,
    snqi_step_reward,
    social_quality_v1_reward,
)


def _meta(*, collision: bool, success: bool, step: int = 50, max_steps: int = 100) -> dict:
    return {
        "step_of_episode": step,
        "max_sim_steps": max_steps,
        "is_pedestrian_collision": collision,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": success,
        "is_waypoint_complete": success,
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


def test_snqi_step_reward_consumes_extended_proxy_terms():
    """Additional SNQI proxy terms should affect the step reward score."""
    base = snqi_step_reward(_meta(collision=False, success=False))
    enriched_meta = {
        **_meta(collision=False, success=False),
        "near_misses": 2.0,
        "force_exceed_events": 3.0,
        "jerk_mean": 1.5,
    }
    enriched = snqi_step_reward(enriched_meta)
    assert enriched < base


def test_make_robot_env_builds_reward_func_from_reward_name(monkeypatch):
    """Factory should build reward callables from reward_name when reward_func is omitted."""
    captured: dict[str, object] = {}

    class _DummyEnv:
        pass

    def _fake_create_robot_env(**kwargs):
        captured.update(kwargs)
        return _DummyEnv()

    # Patch the exact class object referenced by make_robot_env to avoid module-path alias issues.
    env_factory_cls = make_robot_env.__globals__["EnvironmentFactory"]
    monkeypatch.setattr(env_factory_cls, "create_robot_env", staticmethod(_fake_create_robot_env))
    env = make_robot_env(reward_name="snqi_step")
    assert isinstance(env, _DummyEnv)
    assert callable(captured["reward_func"])


def test_route_completion_and_social_rewards_emit_term_decomposition() -> None:
    """Route/social profile rewards should populate bounded reward decomposition fields."""
    meta = {
        "prev_distance_to_goal": 3.0,
        "distance_to_goal": 2.0,
        "is_pedestrian_collision": False,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": True,
        "near_misses": "not-a-number",
        "time_to_collision": 0.5,
        "comfort_exposure": 0.2,
        "jerk_mean": 3.0,
    }
    value = route_completion_v2_reward(meta, weights={"terminal_bonus": 3.0})
    assert math.isfinite(value)
    assert "reward_terms" in meta
    assert "reward_total" in meta
    assert isinstance(meta["reward_terms"], dict)

    social_meta = dict(meta)
    social_meta["time_to_collision"] = float("inf")
    social_value = social_quality_v1_reward(social_meta, weights={"progress": 2.0})
    assert math.isfinite(social_value)
    assert isinstance(social_meta["reward_terms"], dict)


def test_simple_ped_and_punish_action_rewards_cover_collision_and_action_branches() -> None:
    """Legacy reward paths should remain finite for collision/route/action combinations."""
    ped_meta = {
        "max_sim_steps": 100,
        "distance_to_robot": 1.5,
        "is_pedestrian_collision": True,
        "is_obstacle_collision": True,
        "is_robot_collision": True,
        "is_route_complete": True,
    }
    value = simple_ped_reward(ped_meta)
    assert math.isfinite(value)

    meta = {
        "max_sim_steps": 100,
        "is_pedestrian_collision": False,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": False,
        "action": [1.0, 0.0],
        "last_action": [0.0, 0.0],
    }
    penalized = punish_action_reward(meta, punish_action=True)
    unpenalized = punish_action_reward(meta, punish_action=False)
    assert penalized < unpenalized


def test_build_reward_function_accepts_new_aliases_and_rejects_unknown() -> None:
    """Registry should resolve new aliases and reject unsupported names."""
    assert callable(build_reward_function("route_completion"))
    assert callable(build_reward_function("social_quality"))
    assert callable(build_reward_function("simple"))
    assert callable(build_reward_function("punish_action"))
    try:
        build_reward_function("unknown-reward")
    except ValueError as exc:
        assert "Unknown reward_name" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown reward name")
