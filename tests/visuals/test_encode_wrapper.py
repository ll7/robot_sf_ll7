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
    """TODO docstring. Document this function.

    Args:
        n: TODO docstring.
    """
    for i in range(n):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 40
        yield arr


def test_encode_skip_when_moviepy_missing(monkeypatch, tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
        tmp_path: TODO docstring.
    """
    monkeypatch.setattr(encode, "moviepy_ready", lambda: False)
    res = encode.encode_frames(_frame_gen(), tmp_path / "out.mp4")
    assert res.status == "skipped"
    assert res.note == "moviepy-missing"


def test_encode_success_mocked(monkeypatch, tmp_path: Path):
    # Force readiness
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
        tmp_path: TODO docstring.
    """
    monkeypatch.setattr(encode, "moviepy_ready", lambda: True)

    class _FakeClip:
        """TODO docstring. Document this class."""

        def __init__(self, frames, fps):
            """TODO docstring. Document this function.

            Args:
                frames: TODO docstring.
                fps: TODO docstring.
            """
            self._frames = frames
            self.fps = fps

        def write_videofile(self, path, _codec, _fps, _audio, _preset, _logger):
            """TODO docstring. Document this function.

            Args:
                path: TODO docstring.
                _codec: TODO docstring.
                _fps: TODO docstring.
                _audio: TODO docstring.
                _preset: TODO docstring.
                _logger: TODO docstring.
            """
            with open(path, "wb") as f:  # tiny file to simulate success
                f.write(b"00")

    def _factory(frames, fps):
        """TODO docstring. Document this function.

        Args:
            frames: TODO docstring.
            fps: TODO docstring.
        """
        return _FakeClip(frames, fps)

    monkeypatch.setattr(encode, "ImageSequenceClip", _factory)
    out_path = tmp_path / "ok.mp4"
    res = encode.encode_frames(_frame_gen(), out_path)
    assert res.status == "success"
    assert res.note is None
    assert out_path.exists()
    assert res.encode_time_s is not None
