"""Tests for ROBOT_SF_MAX_VIDEO_FRAMES environment variable override."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.render.sim_view import SimulationView, VisualizableSimState


def _basic_state(t: int) -> VisualizableSimState:
    """TODO docstring. Document this function.

    Args:
        t: TODO docstring.

    Returns:
        TODO docstring.
    """
    return VisualizableSimState(
        timestep=t,
        robot_action=None,
        robot_pose=((0.0, 0.0), 0.0),
        pedestrian_positions=np.zeros((0, 2)),
        ray_vecs=np.zeros((0, 2, 2)),
        ped_actions=np.zeros((0, 2, 2)),
    )


@pytest.mark.parametrize("override, expected", [("5", 5), ("NONE", None), ("-1", None)])
def test_env_override_max_frames(monkeypatch, override, expected):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
        override: TODO docstring.
        expected: TODO docstring.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("ROBOT_SF_MAX_VIDEO_FRAMES", override)

    view = SimulationView(record_video=True, video_path=None)

    # Determine effective cap
    cap = view.max_frames
    assert cap == expected, f"Expected max_frames={expected} from override={override}, got {cap}"

    # Render frames; ensure enforcement (for numeric cap) or growth (for None) works
    for t in range(12):
        view.render(_basic_state(t))

    if expected is None:
        # Should have > 10 frames (no cap)
        assert len(view.frames) > 10
    else:
        assert len(view.frames) == expected

    view.exit_simulation()


def test_env_override_invalid_ignored(monkeypatch):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("ROBOT_SF_MAX_VIDEO_FRAMES", "not-an-int")

    view = SimulationView(record_video=True, video_path=None)
    # Should keep default (2000) on invalid override
    assert view.max_frames == 2000
    view.exit_simulation()
