"""Return-annotation coverage for benchmark visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_sf.benchmark import visualization


def test_video_encoder_exercises_annotated_frame_generator(monkeypatch, tmp_path: Path) -> None:
    """Run the nested generator definition through a lightweight encoder fake."""
    captured: dict[str, object] = {}

    class FakeImageSequenceClip:
        """Capture the generator output without invoking MoviePy."""

        def __init__(self, frames: list[np.ndarray], *, fps: int) -> None:
            captured["frames"] = frames
            captured["fps"] = fps

        def write_videofile(self, _video_path: str, **kwargs: object) -> None:
            Path(str(kwargs["temp_audiofile"])).unlink()

    monkeypatch.setattr(
        visualization,
        "_load_image_sequence_clip",
        lambda: FakeImageSequenceClip,
    )
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]

    visualization._encode_frames_to_video(frames, str(tmp_path / "video.mp4"), fps=7)

    assert captured["fps"] == 7
    captured_frames = captured["frames"]
    assert isinstance(captured_frames, list)
    assert len(captured_frames) == 1
    np.testing.assert_array_equal(captured_frames[0], frames[0])
