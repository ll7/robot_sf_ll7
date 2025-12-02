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
    """Frame gen.

    Args:
        n: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    for i in range(n):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 40
        yield arr


def test_encode_skip_when_moviepy_missing(monkeypatch, tmp_path: Path):
    """Test encode skip when moviepy missing.

    Args:
        monkeypatch: Auto-generated placeholder description.
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    monkeypatch.setattr(encode, "moviepy_ready", lambda: False)
    res = encode.encode_frames(_frame_gen(), tmp_path / "out.mp4")
    assert res.status == "skipped"
    assert res.note == "moviepy-missing"


def test_encode_success_mocked(monkeypatch, tmp_path: Path):
    """Test encode success mocked.

    Args:
        monkeypatch: Auto-generated placeholder description.
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Force readiness
    monkeypatch.setattr(encode, "moviepy_ready", lambda: True)

    class _FakeClip:
        """FakeClip class."""

        def __init__(self, frames, fps):
            """Init.

            Args:
                frames: Auto-generated placeholder description.
                fps: Auto-generated placeholder description.

            Returns:
                Any: Auto-generated placeholder description.
            """
            self._frames = frames
            self.fps = fps

        def write_videofile(self, path, _codec, _fps, _audio, _preset, _logger):
            """Write videofile.

            Args:
                path: Auto-generated placeholder description.
                _codec: Auto-generated placeholder description.
                _fps: Auto-generated placeholder description.
                _audio: Auto-generated placeholder description.
                _preset: Auto-generated placeholder description.
                _logger: Auto-generated placeholder description.

            Returns:
                Any: Auto-generated placeholder description.
            """
            with open(path, "wb") as f:  # tiny file to simulate success
                f.write(b"00")

    def _factory(frames, fps):
        """Factory.

        Args:
            frames: Auto-generated placeholder description.
            fps: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        return _FakeClip(frames, fps)

    monkeypatch.setattr(encode, "ImageSequenceClip", _factory)
    out_path = tmp_path / "ok.mp4"
    res = encode.encode_frames(_frame_gen(), out_path)
    assert res.status == "success"
    assert res.note is None
    assert out_path.exists()
    assert res.encode_time_s is not None
