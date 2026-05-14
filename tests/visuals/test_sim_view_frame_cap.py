"""Test SimulationView frame capture cap enforcement.

Ensures that when max_frames is set, the view stops recording additional frames
and emits a single warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.render.sim_view import SimulationView, VisualizableSimState


def _basic_state(timestep: int) -> VisualizableSimState:
    """Build a minimal renderable state for frame-buffer tests."""
    return VisualizableSimState(
        timestep=timestep,
        robot_action=None,
        robot_pose=((0.0, 0.0), 0.0),
        pedestrian_positions=np.zeros((0, 2)),
        ray_vecs=np.zeros((0, 2, 2)),
        ped_actions=np.zeros((0, 2, 2)),
        time_per_step_in_secs=0.1,
    )


@pytest.mark.parametrize("cap", [3, 5])
def test_frame_cap_enforced(cap, monkeypatch):
    # Force headless
    """Frame buffering should stop at the configured hard safety cap.

    Args:
        cap: Maximum number of frames to retain.
        monkeypatch: Pytest helper used to force headless rendering.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    view = SimulationView(record_video=True, video_path=None, max_frames=cap, width=64, height=64)

    # Render more frames than the cap
    for t in range(cap + 5):
        view.render(_basic_state(t))

    # Should not exceed cap
    assert len(view.frames) == cap, f"Expected at most {cap} frames but got {len(view.frames)}"

    # Further captures remain no-op
    view.render(_basic_state(999))
    assert len(view.frames) == cap

    # Clean exit (no video writing since video_path=None)
    view.exit_simulation()


def test_recording_frame_buffer_tracks_video_cadence(monkeypatch) -> None:
    """Recording should retain frames at video cadence, not every render call."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    view = SimulationView(
        record_video=True,
        video_path=None,
        video_fps=10.0,
        max_frames=None,
        width=64,
        height=64,
    )

    for t in range(60):
        view.render(_basic_state(t), target_fps=60.0)

    assert len(view.frames) == 10
    view.exit_simulation()
