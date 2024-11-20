# test_moviepy_setup.py

import pytest
from pathlib import Path
import numpy as np
import os
from moviepy import *

def test_moviepy_ffmpeg_setup():
    """Test if moviepy and ffmpeg are properly configured."""
    # Create test directory
    test_dir = Path("test_moviepy")
    test_dir.mkdir(exist_ok=True)

    try:
        # Create frames
        frames = []
        for i in range(10):
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
        assert video.duration == 1.0
        video.close()

    finally:
        # Cleanup
        if test_dir.exists():
            for f in test_dir.iterdir():
                f.unlink()
            test_dir.rmdir()
