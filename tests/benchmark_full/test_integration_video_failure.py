"""Integration test T043: video generation failure injection.

Goal:
  - Simulate an exception during frame generation to ensure `generate_videos` returns
    an artifact with status 'error' and does not raise.
Strategy:
  - Monkeypatch ImageSequenceClip (if available) or the plotting loop to raise a RuntimeError.
  - Provide a non-smoke config (smoke=False) with videos enabled and one record.
  - Assert artifact status == 'error'.

This test purposefully avoids depending on ffmpeg availability; by monkeypatching
an internal component we guarantee the error path executes deterministically.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic import videos as videos_mod
from robot_sf.benchmark.full_classic.videos import generate_videos


@pytest.mark.skipif(
    videos_mod.plt is None,
    reason="matplotlib not available; cannot test failure path",
)
def test_video_generation_error_path(temp_results_dir, synthetic_episode_record, monkeypatch):
    """Verify a rendering exception becomes an error artifact instead of raising.

    Args:
        temp_results_dir: Temporary directory used as the videos output root.
        synthetic_episode_record: Factory building the single episode record.
        monkeypatch: Fixture injecting the failure into the video pipeline.
    """
    records = [
        synthetic_episode_record(
            episode_id="ep_fail",
            scenario_id="scenario_fail",
            seed=42,
        ),
    ]

    class _Cfg:
        """Minimal non-smoke config stub enabling one video render attempt."""

        smoke = False
        disable_videos = False
        max_videos = 1

    # Force ImageSequenceClip to raise at creation time if present; else raise inside loop.
    if videos_mod.ImageSequenceClip is not None:

        def _boom(*args, **kwargs):
            """Raise to simulate an ImageSequenceClip construction failure.

            Args:
                args: Positional arguments forwarded by the caller (unused).
                kwargs: Keyword arguments forwarded by the caller (unused).
            """
            raise RuntimeError("boom clip")

        monkeypatch.setattr(videos_mod, "ImageSequenceClip", _boom)
    else:
        # Monkeypatch math.cos to raise to trigger the except block early
        def _cos_fail(x):
            """Raise to simulate a failure inside the fallback frame loop.

            Args:
                x: Angle argument passed by the plotting loop (unused).
            """
            raise RuntimeError("boom cos")

        monkeypatch.setattr(videos_mod.math, "cos", _cos_fail)

    artifacts = generate_videos(records, str(temp_results_dir / "videos"), _Cfg())
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.status == "error"
    assert art.episode_id == "ep_fail"
    assert art.scenario_id == "scenario_fail"
    assert art.path_mp4.endswith(".mp4")
