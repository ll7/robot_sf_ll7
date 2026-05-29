"""Snapshot test capturing current factory function signatures.

Purpose: Guard against unintended signature drift before ergonomics refactor.
If this test fails after intentional changes, update the expected parameter sets
in the assertions and remove (or migrate) once new baseline accepted.
"""

from __future__ import annotations

import inspect

from robot_sf.gym_env import environment_factory as ef


def _param_names(fn) -> list[str]:
    """Return parameter names for a callable signature in declaration order.

    Args:
        fn: Callable object whose signature should be inspected.

    Returns:
        list[str]: Ordered parameter names from the callable signature.
    """
    return [p.name for p in inspect.signature(fn).parameters.values()]


def _assert_no_var_keyword(fn) -> None:
    """Assert a public factory has no catch-all keyword parameter."""
    assert inspect.Parameter.VAR_KEYWORD not in [
        p.kind for p in inspect.signature(fn).parameters.values()
    ]


def test_make_robot_env_signature_snapshot():
    """make_robot_env keeps the reviewed public parameter order."""
    params = _param_names(ef.make_robot_env)
    expected = [
        "config",
        "seed",
        "peds_have_obstacle_forces",
        "reward_func",
        "reward_name",
        "reward_kwargs",
        "reward_curriculum",
        "debug",
        "recording_enabled",
        "record_video",
        "video_path",
        "video_fps",
        "render_options",
        "recording_options",
        "jsonl_recording_options",
        "use_jsonl_recording",
        "recording_dir",
        "suite_name",
        "scenario_name",
        "algorithm_name",
        "recording_seed",
        "telemetry_options",
        "enable_telemetry_panel",
        "telemetry_metrics",
        "telemetry_record",
        "telemetry_refresh_hz",
        "telemetry_pane_layout",
        "telemetry_decimation",
        "asymmetric_critic",
    ]
    assert params == expected
    _assert_no_var_keyword(ef.make_robot_env)


def test_make_image_robot_env_signature_snapshot():
    """make_image_robot_env keeps the reviewed public parameter order."""
    params = _param_names(ef.make_image_robot_env)
    expected = [
        "config",
        "seed",
        "peds_have_obstacle_forces",
        "reward_func",
        "reward_name",
        "reward_kwargs",
        "reward_curriculum",
        "debug",
        "recording_enabled",
        "record_video",
        "video_path",
        "video_fps",
        "render_options",
        "recording_options",
        "jsonl_recording_options",
        "use_jsonl_recording",
        "recording_dir",
        "suite_name",
        "scenario_name",
        "algorithm_name",
        "recording_seed",
        "asymmetric_critic",
    ]
    assert params == expected
    _assert_no_var_keyword(ef.make_image_robot_env)


def test_make_pedestrian_env_signature_snapshot():
    """make_pedestrian_env keeps the reviewed public parameter order."""
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
    _assert_no_var_keyword(ef.make_pedestrian_env)
