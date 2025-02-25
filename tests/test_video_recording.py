"""Test video recording functionality of the simulation view."""

from pathlib import Path
import pytest
from loguru import logger

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.render.sim_view import MOVIEPY_AVAILABLE
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.robot.bicycle_drive import BicycleDriveSettings


@pytest.mark.skipif(
    not MOVIEPY_AVAILABLE, reason="MoviePy/ffmpeg not available for video recording"
)
def test_video_recording(
    delete_video: bool = True,
):
    """Test that video recording works and creates/deletes files properly."""

    # Create recordings directory if it doesn't exist
    recordings_dir = Path("recordings")
    recordings_dir.mkdir(exist_ok=True)

    # Create environment settings
    env_config = EnvSettings(
        sim_config=SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.02]),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
    )

    # Create environment with video recording enabled
    video_path = recordings_dir / "test_video.mp4"
    logger.debug(f"Video path: {video_path}")
    env = RobotEnv(
        env_config=env_config,
        debug=True,
        recording_enabled=True,
        record_video=True,
        video_path=str(video_path),
    )

    try:
        # Run simulation for a few frames
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            env.render()  # Need to call render to capture frames
            if done:
                env.reset()

        # Close env to trigger video creation
        env.sim_ui.exit_simulation()
        logger.debug("exit the simulation")

        # Verify video was created
        assert video_path.exists(), "Video file was not created"

    finally:
        # Clean up
        if video_path.exists() and delete_video:
            video_path.unlink()
        assert not video_path.exists(), "Video file was not deleted"

        # Clean up recordings dir if empty
        if recordings_dir.exists() and not any(recordings_dir.iterdir()):
            recordings_dir.rmdir()


if __name__ == "__main__":
    test_video_recording(delete_video=False)
