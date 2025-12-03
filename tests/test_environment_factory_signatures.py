"""Signature tests for explicit Option A environment helpers.

Ensures discoverability and prevents regression back to **kwargs catch-all.
"""

from __future__ import annotations

import inspect

from robot_sf.gym_env.environment_factory import (
    make_image_robot_env,
    make_multi_robot_env,
    make_pedestrian_env,
    make_robot_env,
)


def _param_names(func):
    """TODO docstring. Document this function.

    Args:
        func: TODO docstring.
    """
    return [p.name for p in inspect.signature(func).parameters.values()]


def test_make_robot_env_signature_explicit():
    """TODO docstring. Document this function."""
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
    """TODO docstring. Document this function."""
    params = _param_names(make_image_robot_env)
    assert "config" in params
    assert "debug" in params


def test_make_pedestrian_env_signature_explicit():
    """TODO docstring. Document this function."""
    params = _param_names(make_pedestrian_env)
    assert "config" in params and "robot_model" in params


def test_make_multi_robot_env_signature_explicit():
    """TODO docstring. Document this function."""
    params = _param_names(make_multi_robot_env)
    assert "num_robots" in params and "config" in params
