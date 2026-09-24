"""Tests for encoding wrapper (T033).

Validates skip note when moviepy missing and success path via monkeypatching
ImageSequenceClip. We simulate frames with small numpy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robot_sf.benchmark.full_classic import encode

if TYPE_CHECKING:
    from pathlib import Path


def _frame_gen(n=3):
    """Yield small uint8 RGB frames with the red channel varying by index.

    Args:
        n: Number of frames to yield.
    """
    for i in range(n):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 40
        yield arr


def test_encode_skip_when_moviepy_missing(monkeypatch, tmp_path: Path):
    """Assert encode_frames skips when moviepy readiness is false.

    Args:
        monkeypatch: Pytest fixture used to force moviepy_ready false.
        tmp_path: Directory holding the requested output path.
    """
    monkeypatch.setattr(encode, "moviepy_ready", lambda: False)
    res = encode.encode_frames(_frame_gen(), tmp_path / "out.mp4")
    assert res.status == "skipped"
    assert res.note == "moviepy-missing"


def test_encode_success_mocked(monkeypatch, tmp_path: Path):
    # Force readiness
    """Assert encode_frames succeeds with a mocked ImageSequenceClip.

    The fake clip writes a two-byte file; the result reports success with no note,
    the output exists, and an encode time is recorded.

    Args:
        monkeypatch: Pytest fixture used to force readiness and inject the fake clip.
        tmp_path: Directory receiving the output file.
    """
    monkeypatch.setattr(encode, "moviepy_ready", lambda: True)

    class _FakeClip:
        """Minimal ImageSequenceClip stand-in that records inputs and writes a tiny file."""

        def __init__(self, frames, fps):
            """Store the frames and fps passed by the encoder.

            Args:
                frames: Frame sequence handed to the clip.
                fps: Frames-per-second value handed to the clip.
            """
            self._frames = frames
            self.fps = fps

        def write_videofile(self, path, _codec, _fps, _audio, _preset, _logger):
            """Write two bytes to path, ignoring encoder options, to simulate success.

            Args:
                path: Output file path to create.
                _codec: Codec option accepted and ignored.
                _fps: Frames-per-second option accepted and ignored.
                _audio: Audio option accepted and ignored.
                _preset: Encoder preset option accepted and ignored.
                _logger: Logger option accepted and ignored.
            """
            with open(path, "wb") as f:  # tiny file to simulate success
                f.write(b"00")

    def _factory(frames, fps):
        """Build a fake clip from the frames and fps, mimicking ImageSequenceClip.

        Args:
            frames: Frame sequence handed to the clip.
            fps: Frames-per-second value handed to the clip.
        """
        return _FakeClip(frames, fps)

    monkeypatch.setattr(encode, "ImageSequenceClip", _factory)
    out_path = tmp_path / "ok.mp4"
    res = encode.encode_frames(_frame_gen(), out_path)
    assert res.status == "success"
    assert res.note is None
    assert out_path.exists()
    assert res.encode_time_s is not None
