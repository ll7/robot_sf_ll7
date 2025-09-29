"""T017: Frame recording integration test.

Verifies that using record_video=True with a video_path results in an environment
with a SimulationView (sim_ui) set up for frame capture.
"""

from __future__ import annotations

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig


def test_recording_creates_simulation_view(tmp_path):
    vid_path = tmp_path / "episode.mp4"
    env = make_robot_env(
        config=RobotSimulationConfig(),
        record_video=True,
        video_path=str(vid_path),
    )
    # No stepping required; presence of sim_ui indicates recording path configured
    assert getattr(env, "sim_ui", None) is not None
    env.close()
    # File may not exist yet (depends on capture/flush mechanism), so we do not assert file presence here.
