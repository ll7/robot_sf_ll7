"""Test that RobotEnv.render() raises actionable error when debug disabled.

Covers improved guidance (debug=False path) referencing factory usage.
"""

from __future__ import annotations

import pytest

from robot_sf.gym_env.environment_factory import make_robot_env


def test_render_error_message_contains_guidance():  # noqa: D401
    env = make_robot_env(debug=False)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            env.render()
    finally:
        env.close()
    msg = str(excinfo.value)
    assert "debug=False" in msg
    assert "make_robot_env" in msg
    assert "visualization" in msg or "frame capture" in msg
