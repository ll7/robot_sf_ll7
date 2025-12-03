"""TODO docstring. Document this module."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.full_classic import encode as encode_mod

# We test _write_clip indirectly via encode_frames by injecting a fake ImageSequenceClip
# carrying different write_videofile signatures / behaviors.


class _BaseClip:
    """TODO docstring. Document this class."""

    def __init__(self, frames, fps=10):
        """TODO docstring. Document this function.

        Args:
            frames: TODO docstring.
            fps: TODO docstring.
        """
        self.frames = frames
        self.fps = fps


class KeywordClip(_BaseClip):
    """TODO docstring. Document this class."""

    def write_videofile(self, path, *, fps, codec, audio, preset, logger):
        """TODO docstring. Document this function.

        Args:
            path: TODO docstring.
            fps: TODO docstring.
            codec: TODO docstring.
            audio: TODO docstring.
            preset: TODO docstring.
            logger: TODO docstring.
        """
        _ = (fps, codec, audio, preset, logger)
        Path(path).write_bytes(b"kw")


class PositionalClip(_BaseClip):
    """TODO docstring. Document this class."""

    def write_videofile(self, path, codec, fps, audio, preset, logger):
        """TODO docstring. Document this function.

        Args:
            path: TODO docstring.
            codec: TODO docstring.
            fps: TODO docstring.
            audio: TODO docstring.
            preset: TODO docstring.
            logger: TODO docstring.
        """
        _ = (codec, fps, audio, preset, logger)
        Path(path).write_bytes(b"pos")


class MinimalClip(_BaseClip):
    """TODO docstring. Document this class."""

    def write_videofile(self, path):
        """TODO docstring. Document this function.

        Args:
            path: TODO docstring.
        """
        Path(path).write_bytes(b"min")


class AlwaysFailClip(_BaseClip):
    """TODO docstring. Document this class."""

    calls = 0

    def write_videofile(self, *args, **kwargs):
        """TODO docstring. Document this function.

        Args:
            args: TODO docstring.
            kwargs: TODO docstring.
        """
        _ = (args, kwargs)
        self.__class__.calls += 1
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    "clip_cls,expected_bytes",
    [
        (KeywordClip, b"kw"),
        (PositionalClip, b"pos"),
        (MinimalClip, b"min"),
    ],
)
def test_write_clip_fallback_success(monkeypatch, tmp_path, clip_cls, expected_bytes):
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
        tmp_path: TODO docstring.
        clip_cls: TODO docstring.
        expected_bytes: TODO docstring.
    """
    monkeypatch.setattr(encode_mod, "ImageSequenceClip", clip_cls)
    monkeypatch.setattr(encode_mod, "moviepy_ready", lambda: True)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    out = tmp_path / "out.mp4"
    res = encode_mod.encode_frames(
        frames,
        out,
        fps=5,
        codec="libx264",
        preset="ultrafast",
        sample_memory=False,
    )
    assert res.status == "success"
    assert out.read_bytes() == expected_bytes


def test_write_clip_all_fail(monkeypatch, tmp_path):
    # Force all attempts to raise, expecting failed note encode-error:RuntimeError
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
        tmp_path: TODO docstring.
    """
    monkeypatch.setattr(encode_mod, "ImageSequenceClip", AlwaysFailClip)
    monkeypatch.setattr(encode_mod, "moviepy_ready", lambda: True)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    out = tmp_path / "out.mp4"
    res = encode_mod.encode_frames(frames, out, sample_memory=False)
    assert res.status == "failed"
    assert res.note and res.note.startswith("encode-error:RuntimeError")
    assert not out.exists() or out.stat().st_size == 0
