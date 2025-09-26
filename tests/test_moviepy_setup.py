# test_moviepy_setup.py

from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from moviepy import ImageSequenceClip, VideoFileClip

from robot_sf.render.sim_view import MOVIEPY_AVAILABLE


@pytest.mark.skipif(not MOVIEPY_AVAILABLE, reason="Moviepy not installed. Run: uv add moviepy")
def test_moviepy_ffmpeg_setup():
    """Test if moviepy and ffmpeg are properly configured."""
    # Create test directory
    test_dir = Path("test_moviepy")
    test_dir.mkdir(exist_ok=True)

    try:
        # Create frames
        frames = []
        for i in range(11):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            x = int(50 + 30 * np.cos(i / 10 * 2 * np.pi))
            y = int(50 + 30 * np.sin(i / 10 * 2 * np.pi))
            frame[y - 5 : y + 5, x - 5 : x + 5] = [255, 0, 0]
            frames.append(frame)

        # Create and save video
        test_video = test_dir / "test.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(str(test_video), logger=None)

        # Verify video
        video = VideoFileClip(str(test_video))
        logger.info(f"Video duration: {video.duration}")
        assert video.duration in (1.0, 1.1)
        # TODO: On some systems, the video duration is 1.1 instead of 1.0
        video.close()

    finally:
        # Cleanup
        if test_dir.exists():
            for f in test_dir.iterdir():
                f.unlink()
            test_dir.rmdir()
