"""pytest for load_states_and_record_video.py"""

import datetime
from pathlib import Path

import pytest

from robot_sf.render.playback_recording import load_states_and_record_video
from robot_sf.render.sim_view import MOVIEPY_AVAILABLE


@pytest.mark.skipif(
    not MOVIEPY_AVAILABLE,
    reason="MoviePy/ffmpeg not available for video recording",
)
def test_load_states_and_record_video(delete_video: bool = True):
    """Test loading simulation states and recording them as video.

    Args:
        delete_video: Whether to delete the video file after test. Default True.
    """
    # Create recordings directory if it doesn't exist
    recordings_dir = Path("tmp/recording_test")
    recordings_dir.mkdir(exist_ok=True)

    # create a unique video name
    video_name = "playback_test_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"

    output_path = recordings_dir / video_name

    try:
        load_states_and_record_video(
            "test_pygame/recordings/2024-06-04_08-39-59.pkl",
            str(output_path),
        )

        assert output_path.exists(), "Video file was not created"
        assert output_path.stat().st_size > 0, "Video file is empty"
    finally:
        # Clean up
        if output_path.exists() and delete_video:
            output_path.unlink()


if __name__ == "__main__":
    test_load_states_and_record_video(delete_video=False)
