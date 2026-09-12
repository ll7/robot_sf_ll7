"""Encode fallback coverage for ImageSequenceClip write_videofile signatures.

encode_frames is exercised with fake clips implementing keyword-only,
positional, and path-only writer signatures, plus a clip that always fails.
"""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.full_classic import encode as encode_mod

# We test _write_clip indirectly via encode_frames by injecting a fake ImageSequenceClip
# carrying different write_videofile signatures / behaviors.


class _BaseClip:
    """Base fake clip storing frames and fps for the writer-signature variants."""

    def __init__(self, frames, fps=10):
        """Store frames and fps.

        Args:
            frames: Frame sequence handed to the clip.
            fps: Frames-per-second value handed to the clip.
        """
        self.frames = frames
        self.fps = fps


class KeywordClip(_BaseClip):
    """Fake clip whose writer accepts keyword-only options and writes b"kw"."""

    def write_videofile(self, path, *, fps, codec, audio, preset, logger):
        """Write b"kw" to path, ignoring keyword-only options.

        Args:
            path: Output file path to create.
            fps: Frames-per-second option accepted and ignored.
            codec: Codec option accepted and ignored.
            audio: Audio option accepted and ignored.
            preset: Encoder preset option accepted and ignored.
            logger: Logger option accepted and ignored.
        """
        _ = (fps, codec, audio, preset, logger)
        Path(path).write_bytes(b"kw")


class PositionalClip(_BaseClip):
    """Fake clip whose writer accepts positional options and writes b"pos"."""

    def write_videofile(self, path, codec, fps, audio, preset, logger):
        """Write b"pos" to path, ignoring positional options.

        Args:
            path: Output file path to create.
            codec: Codec option accepted and ignored.
            fps: Frames-per-second option accepted and ignored.
            audio: Audio option accepted and ignored.
            preset: Encoder preset option accepted and ignored.
            logger: Logger option accepted and ignored.
        """
        _ = (codec, fps, audio, preset, logger)
        Path(path).write_bytes(b"pos")


class MinimalClip(_BaseClip):
    """Fake clip whose writer accepts only the output path and writes b"min"."""

    def write_videofile(self, path):
        """Write b"min" to path.

        Args:
            path: Output file path to create.
        """
        Path(path).write_bytes(b"min")


class AlwaysFailClip(_BaseClip):
    """Fake clip whose writer always raises RuntimeError and counts invocations."""

    calls = 0

    def write_videofile(self, *args, **kwargs):
        """Increment the class call counter and raise RuntimeError.

        Args:
            args: Positional writer arguments accepted and ignored.
            kwargs: Keyword writer arguments accepted and ignored.
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
    """Assert encode_frames succeeds for each supported writer calling convention.

    Parametrized over fake clips using keyword-only, positional, and path-only
    write_videofile signatures; each clip's bytes are written to the output file
    and the result reports success.

    Args:
        monkeypatch: Pytest fixture used to inject the fake clip and force readiness.
        tmp_path: Directory receiving the output file.
        clip_cls: Fake clip class exercising one writer signature.
        expected_bytes: Bytes the clip is expected to write.
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
    """Assert every writer failure surfaces as an encode-error result.

    The fake clip raises RuntimeError on each attempt; the result is failed with an
    encode-error:RuntimeError note and no non-empty output file.

    Args:
        monkeypatch: Pytest fixture used to inject the always-failing fake clip.
        tmp_path: Directory checked for an absent or empty output file.
    """
    monkeypatch.setattr(encode_mod, "ImageSequenceClip", AlwaysFailClip)
    monkeypatch.setattr(encode_mod, "moviepy_ready", lambda: True)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    out = tmp_path / "out.mp4"
    res = encode_mod.encode_frames(frames, out, sample_memory=False)
    assert res.status == "failed"
    assert res.note and res.note.startswith("encode-error:RuntimeError")
    assert not out.exists() or out.stat().st_size == 0
