"""Test video recording functionality of the simulation view."""

import os
from pathlib import Path
import pytest
import numpy as np

from robot_sf.gym_env.robot_env import RobotEnv, VisualizableSimState
from robot_sf.render.sim_view import MOVIEPY_AVAILABLE, SimulationView
from robot_sf.sim.robot import RobotPose


@pytest.mark.skipif(not MOVIEPY_AVAILABLE, 
                   reason="MoviePy/ffmpeg not available for video recording")
def test_video_recording():
    """Test that video recording works and creates/deletes files properly."""
    # Create recordings directory if it doesn't exist
    recordings_dir = Path("recordings")
    recordings_dir.mkdir(exist_ok=True)

    # Create view directly with recording enabled
    video_path = recordings_dir / "test_video.mp4"
    sim_view = SimulationView(record_video=True, video_path=str(video_path))

    # Create dummy state
    state = VisualizableSimState(
        timestep=0,
        robot_pose=RobotPose(x=0, y=0, theta=0),
        pedestrian_positions=np.zeros((1, 2)),
        ray_vecs=np.zeros((16, 2)),
        robot_action=None,
        ped_actions=np.zeros((1, 2)),
    )

    # Render some frames
    for _ in range(10):
        sim_view.render(state)

    # Close view to trigger video save
    sim_view._handle_quit()

    # Verify video was created
    assert video_path.exists(), "Video file was not created"

    # Clean up
    video_path.unlink()
    assert not video_path.exists(), "Video file was not deleted"

    # Clean up recordings dir if empty
    if not any(recordings_dir.iterdir()):
        recordings_dir.rmdir()
