"""Snapshot test capturing current factory function signatures.

Purpose: Guard against unintended signature drift before ergonomics refactor.
If this test fails after intentional changes, update the expected parameter sets
in the assertions and remove (or migrate) once new baseline accepted.
"""

from __future__ import annotations

import inspect

from robot_sf.gym_env import environment_factory as ef


def _param_names(fn) -> list[str]:
    return [p.name for p in inspect.signature(fn).parameters.values()]


def test_make_robot_env_signature_snapshot():
    params = _param_names(ef.make_robot_env)
    # Adjust this list only when intentional API evolution occurs.
    expected_prefix = [
        "config",
        "seed",
        "peds_have_obstacle_forces",
        "reward_func",
        "debug",
        "recording_enabled",
        "record_video",
        "video_path",
        "video_fps",
        "render_options",
        "recording_options",
    ]
    assert params[: len(expected_prefix)] == expected_prefix, params


def test_make_image_robot_env_signature_snapshot():
    params = _param_names(ef.make_image_robot_env)
    expected_prefix = [
        "config",
        "seed",
        "peds_have_obstacle_forces",
        "reward_func",
        "debug",
        "recording_enabled",
        "record_video",
        "video_path",
        "video_fps",
        "render_options",
        "recording_options",
    ]
    assert params[: len(expected_prefix)] == expected_prefix, params


def test_make_pedestrian_env_signature_snapshot():
    params = _param_names(ef.make_pedestrian_env)
    # Note: pedestrian env may have an additional required robot_model param currently.
    expected_prefix = [
        "config",
        "seed",
        "robot_model",
        "reward_func",
        "debug",
        "recording_enabled",
        "peds_have_obstacle_forces",
    ]
    assert params[: len(expected_prefix)] == expected_prefix, params
