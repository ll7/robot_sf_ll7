"""Test SimulationView frame capture cap enforcement.

Ensures that when max_frames is set, the view stops recording additional frames
and emits a single warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.render.sim_view import SimulationView, VisualizableSimState


@pytest.mark.parametrize("cap", [3, 5])
def test_frame_cap_enforced(cap, monkeypatch):
    """Test frame cap enforced.

    Args:
        cap: Auto-generated placeholder description.
        monkeypatch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Force headless
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    view = SimulationView(record_video=True, video_path=None, max_frames=cap, width=64, height=64)

    robot_pose = np.array([[0.0, 0.0]])  # shape (1,2) for consistency with drawing code
    pedestrian_positions = np.zeros((0, 2))
    ray_vecs = np.zeros((0, 2, 2))
    ped_actions = np.zeros((0, 2, 2))

    def make_state(t):
        """Make state.

        Args:
            t: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        return VisualizableSimState(
            timestep=t,
            robot_action=None,
            robot_pose=robot_pose,
            pedestrian_positions=pedestrian_positions,
            ray_vecs=ray_vecs,
            ped_actions=ped_actions,
        )

    # Render more frames than the cap
    for t in range(cap + 5):
        view.render(make_state(t))

    # Should not exceed cap
    assert len(view.frames) == cap, f"Expected at most {cap} frames but got {len(view.frames)}"

    # Further captures remain no-op
    view.render(make_state(999))
    assert len(view.frames) == cap

    # Clean exit (no video writing since video_path=None)
    view.exit_simulation()
