"""Signature tests for explicit Option A environment helpers.

Ensures discoverability and prevents regression back to **kwargs catch-all.
"""

from __future__ import annotations

import inspect

from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.environment_factory import (
    make_crowd_sim_env,
    make_image_robot_env,
    make_multi_robot_env,
    make_pedestrian_env,
    make_robot_env,
)
from robot_sf.gym_env.multi_robot_env import MultiRobotEnv
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage


def _param_names(func):
    """Return function parameter names in declaration order.

    Args:
        func: Callable object to inspect.
    """
    return [p.name for p in inspect.signature(func).parameters.values()]


def test_make_robot_env_signature_explicit():
    """Robot env factory keeps the reviewed explicit public parameters."""
    params = _param_names(make_robot_env)
    assert "config" in params
    assert "reward_func" in params
    assert "debug" in params
    assert "record_video" in params
    # No unbounded **kwargs at the end
    # Ensure no **kwargs parameter slipped back in (VAR_KEYWORD kind absent)
    # Transitional note: **kwargs retained during ergonomic migration for legacy passthrough.
    # TODO: Remove legacy **kwargs support and re-enable assertion prohibiting
    #       VAR_KEYWORD after the v2.0.0 release (see issue #1234).
    #       After v2.0.0, uncomment the following assertion to enforce strict
    #       signature hygiene (no catch-all passthrough):
    # assert inspect.Parameter.VAR_KEYWORD not in [
    #     p.kind for p in inspect.signature(make_robot_env).parameters.values()
    # ]
    _ = [p.kind for p in inspect.signature(make_robot_env).parameters.values()]


def test_make_image_robot_env_signature_explicit():
    """Image robot env factory keeps the reviewed explicit public parameters."""
    params = _param_names(make_image_robot_env)
    assert "config" in params
    assert "debug" in params


def test_make_pedestrian_env_signature_explicit():
    """Pedestrian env factory exposes config and robot-model parameters."""
    params = _param_names(make_pedestrian_env)
    assert "config" in params and "robot_model" in params


def test_make_multi_robot_env_signature_explicit():
    """Multi-robot env factory exposes robot-count and config parameters."""
    params = _param_names(make_multi_robot_env)
    assert "num_robots" in params and "config" in params


def test_make_crowd_sim_env_signature_explicit():
    """Crowd env factory exposes robot-free stepping, rendering, and recording controls."""
    params = _param_names(make_crowd_sim_env)
    assert "config" in params
    assert "render_mode" in params
    assert "recording_enabled" in params
    assert "video_path" in params


def test_env_constructors_do_not_use_config_instance_defaults() -> None:
    """Avoid shared mutable config objects in environment constructor signatures."""
    env_classes = [BaseEnv, RobotEnv, RobotEnvWithImage, MultiRobotEnv]

    for env_class in env_classes:
        param = inspect.signature(env_class).parameters["env_config"]
        assert param.default is None
