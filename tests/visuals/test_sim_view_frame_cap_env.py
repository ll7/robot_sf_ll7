"""Tests for ROBOT_SF_MAX_VIDEO_FRAMES environment variable override."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.render.sim_view import SimulationView, VisualizableSimState


def _basic_state(t: int) -> VisualizableSimState:
    """Basic state.

    Args:
        t: Auto-generated placeholder description.

    Returns:
        VisualizableSimState: Auto-generated placeholder description.
    """
    return VisualizableSimState(
        timestep=t,
        robot_action=None,
        robot_pose=np.array([[0.0, 0.0]]),
        pedestrian_positions=np.zeros((0, 2)),
        ray_vecs=np.zeros((0, 2, 2)),
        ped_actions=np.zeros((0, 2, 2)),
    )


@pytest.mark.parametrize("override, expected", [("5", 5), ("NONE", None), ("-1", None)])
def test_env_override_max_frames(monkeypatch, override, expected):
    """Test env override max frames.

    Args:
        monkeypatch: Auto-generated placeholder description.
        override: Auto-generated placeholder description.
        expected: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
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
    """Test env override invalid ignored.

    Args:
        monkeypatch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("ROBOT_SF_MAX_VIDEO_FRAMES", "not-an-int")

    view = SimulationView(record_video=True, video_path=None)
    # Should keep default (2000) on invalid override
    assert view.max_frames == 2000
    view.exit_simulation()
