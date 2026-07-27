"""Lock the synthetic fallback renderer delegation contract.

``generate_fallback_videos`` is a thin pass-through to the legacy
``videos.generate_videos`` renderer. These tests pin that delegation by mocking
the legacy generator so that:

- ``records``, ``out_dir``, and ``cfg`` are forwarded by identity,
- the legacy artifact list is returned unchanged,
- legacy exceptions propagate unchanged with no retry or fallback, and
- no frames are rendered, no optional encoders are exercised on the test path,
  and no video or filesystem artifacts are created.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from robot_sf.benchmark.full_classic import render_synthetic

# A path object that is never touched by the mocked legacy renderer; it exists
# only so identity forwarding can be asserted without creating any file.
_NEVER_WRITTEN = Path("/nonexistent/never-written-render-synthetic-test")


def test_generate_fallback_videos_forwards_arguments_by_identity(monkeypatch) -> None:
    """Forward records, out_dir, and cfg to the legacy generator by identity."""
    legacy = MagicMock(return_value=["artifact-0"])
    monkeypatch.setattr(render_synthetic._legacy_videos, "generate_videos", legacy)

    records = [{"episode_id": "ep0"}]
    out_dir = _NEVER_WRITTEN
    cfg = object()

    result = render_synthetic.generate_fallback_videos(records, out_dir, cfg)

    # Called exactly once on the success path (no retry or invented fallback).
    assert legacy.call_count == 1
    forwarded = legacy.call_args.args
    assert forwarded[0] is records
    assert forwarded[1] is out_dir
    assert forwarded[2] is cfg
    assert result == ["artifact-0"]


def test_generate_fallback_videos_returns_legacy_list_unchanged(monkeypatch) -> None:
    """Return the legacy generator's list object without copying or wrapping."""
    expected = [{"path": "a.mp4"}, {"path": "b.mp4"}]
    legacy = MagicMock(return_value=expected)
    monkeypatch.setattr(render_synthetic._legacy_videos, "generate_videos", legacy)

    result = render_synthetic.generate_fallback_videos(
        [{"episode_id": "ep0"}], _NEVER_WRITTEN, object()
    )

    assert legacy.call_count == 1
    # The exact artifact list object is returned, not a copy or rebuild.
    assert result is expected


def test_generate_fallback_videos_propagates_legacy_exception(monkeypatch) -> None:
    """Legacy renderer exceptions propagate unchanged with no retry or fallback."""

    class _LegacyRendererError(RuntimeError):
        """Marker exception standing in for a legacy renderer failure."""

    legacy = MagicMock(side_effect=_LegacyRendererError("encoder unavailable"))
    monkeypatch.setattr(render_synthetic._legacy_videos, "generate_videos", legacy)

    with pytest.raises(_LegacyRendererError, match="encoder unavailable"):
        render_synthetic.generate_fallback_videos([{"episode_id": "ep0"}], _NEVER_WRITTEN, object())

    # The exception propagates on the first call; the wrapper does not retry.
    assert legacy.call_count == 1


def test_generate_fallback_videos_passes_through_none_return(monkeypatch) -> None:
    """A None return from the legacy generator passes through untouched."""
    legacy = MagicMock(return_value=None)
    monkeypatch.setattr(render_synthetic._legacy_videos, "generate_videos", legacy)

    result = render_synthetic.generate_fallback_videos([], _NEVER_WRITTEN, object())

    assert legacy.call_count == 1
    assert result is None
