"""Module test_encode_write_clip_fallbacks auto-generated docstring."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.full_classic import encode as encode_mod

# We test _write_clip indirectly via encode_frames by injecting a fake ImageSequenceClip
# carrying different write_videofile signatures / behaviors.


class _BaseClip:
    """BaseClip class."""

    def __init__(self, frames, fps=10):
        """Init.

        Args:
            frames: Auto-generated placeholder description.
            fps: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        self.frames = frames
        self.fps = fps


class KeywordClip(_BaseClip):
    """KeywordClip class."""

    def write_videofile(self, path, *, fps, codec, audio, preset, logger):
        """Write videofile.

        Args:
            path: Auto-generated placeholder description.
            fps: Auto-generated placeholder description.
            codec: Auto-generated placeholder description.
            audio: Auto-generated placeholder description.
            preset: Auto-generated placeholder description.
            logger: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        _ = (fps, codec, audio, preset, logger)
        Path(path).write_bytes(b"kw")


class PositionalClip(_BaseClip):
    """PositionalClip class."""

    def write_videofile(self, path, codec, fps, audio, preset, logger):
        """Write videofile.

        Args:
            path: Auto-generated placeholder description.
            codec: Auto-generated placeholder description.
            fps: Auto-generated placeholder description.
            audio: Auto-generated placeholder description.
            preset: Auto-generated placeholder description.
            logger: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        _ = (codec, fps, audio, preset, logger)
        Path(path).write_bytes(b"pos")


class MinimalClip(_BaseClip):
    """MinimalClip class."""

    def write_videofile(self, path):
        """Write videofile.

        Args:
            path: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        Path(path).write_bytes(b"min")


class AlwaysFailClip(_BaseClip):
    """AlwaysFailClip class."""

    calls = 0

    def write_videofile(self, *args, **kwargs):
        """Write videofile.

        Args:
            args: Auto-generated placeholder description.
            kwargs: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
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
    """Test write clip fallback success.

    Args:
        monkeypatch: Auto-generated placeholder description.
        tmp_path: Auto-generated placeholder description.
        clip_cls: Auto-generated placeholder description.
        expected_bytes: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
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
    """Test write clip all fail.

    Args:
        monkeypatch: Auto-generated placeholder description.
        tmp_path: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Force all attempts to raise, expecting failed note encode-error:RuntimeError
    monkeypatch.setattr(encode_mod, "ImageSequenceClip", AlwaysFailClip)
    monkeypatch.setattr(encode_mod, "moviepy_ready", lambda: True)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    out = tmp_path / "out.mp4"
    res = encode_mod.encode_frames(frames, out, sample_memory=False)
    assert res.status == "failed"
    assert res.note and res.note.startswith("encode-error:RuntimeError")
    assert not out.exists() or out.stat().st_size == 0
