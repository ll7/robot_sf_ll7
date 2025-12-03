"""TODO docstring. Document this module."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import robot_sf.benchmark.runner as runner_mod
from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


@pytest.mark.skipif(
    importlib.util.find_spec("moviepy") is None,
    reason="moviepy/ffmpeg not available",
)
def test_run_batch_with_synthetic_video(tmp_path: Path):
    # Minimal in-memory scenario list (single episode)
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    scenarios = [
        {
            "id": "unit-video-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]

    out_jsonl = tmp_path / "episodes.jsonl"

    summary = run_batch(
        scenarios_or_path=scenarios,
        out_path=out_jsonl,
        schema_path=SCHEMA_PATH,
        base_seed=0,
        repeats_override=None,
        horizon=10,
        dt=0.1,
        record_forces=False,
        video_enabled=True,
        video_renderer="synthetic",
        append=False,
        fail_fast=True,
        workers=1,
        resume=False,
    )

    assert summary["written"] == 1
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    video = rec.get("video")
    assert isinstance(video, dict)
    assert video.get("format") == "mp4"
    assert video.get("status") == "success"
    vpath = Path(video.get("path"))
    assert vpath.exists() and vpath.suffix == ".mp4"
    assert int(video.get("filesize_bytes", 0)) > 0
    assert int(video.get("frames", 0)) >= 1


@pytest.mark.skipif(
    importlib.util.find_spec("moviepy") is None,
    reason="moviepy/ffmpeg not available",
)
def test_zero_step_episode_skips_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        monkeypatch: TODO docstring.
    """

    def fake_encoder(*args, **kwargs):  # type: ignore[no-untyped-def]
        """TODO docstring. Document this function.

        Args:
            args: TODO docstring.
            kwargs: TODO docstring.
        """
        return None, {"reason": "no-frames", "renderer": "synthetic", "steps": 0}

    monkeypatch.setattr(runner_mod, "_try_encode_synthetic_video", fake_encoder)

    scenarios = [
        {
            "id": "unit-video-zero-steps",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]

    out_jsonl = tmp_path / "episodes_zero.jsonl"

    summary = run_batch(
        scenarios_or_path=scenarios,
        out_path=out_jsonl,
        schema_path=SCHEMA_PATH,
        base_seed=0,
        repeats_override=None,
        horizon=10,
        dt=0.1,
        record_forces=False,
        video_enabled=True,
        video_renderer="synthetic",
        append=False,
        fail_fast=True,
        workers=1,
        resume=False,
    )

    assert summary["written"] == 1
    records = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(records[0])
    assert "video" not in rec
    notes = rec.get("notes", "")
    assert "video skipped" in notes
    assert "steps=0" in notes
    videos_dir = out_jsonl.parent / "videos"
    assert not any(videos_dir.glob("*.mp4"))


@pytest.mark.skipif(
    importlib.util.find_spec("moviepy") is None,
    reason="moviepy/ffmpeg not available",
)
def test_metrics_match_with_and_without_video(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    scenarios = [
        {
            "id": "unit-video-metrics",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]

    no_video_path = tmp_path / "episodes_no_video.jsonl"
    with_video_path = tmp_path / "episodes_with_video.jsonl"

    run_batch(
        scenarios_or_path=scenarios,
        out_path=no_video_path,
        schema_path=SCHEMA_PATH,
        base_seed=1,
        repeats_override=None,
        horizon=10,
        dt=0.1,
        record_forces=False,
        video_enabled=False,
        video_renderer="none",
        append=False,
        fail_fast=True,
        workers=1,
        resume=False,
    )

    run_batch(
        scenarios_or_path=scenarios,
        out_path=with_video_path,
        schema_path=SCHEMA_PATH,
        base_seed=1,
        repeats_override=None,
        horizon=10,
        dt=0.1,
        record_forces=False,
        video_enabled=True,
        video_renderer="synthetic",
        append=False,
        fail_fast=True,
        workers=1,
        resume=False,
    )

    rec_no_video = json.loads(no_video_path.read_text(encoding="utf-8").strip())
    rec_with_video = json.loads(with_video_path.read_text(encoding="utf-8").strip())

    assert rec_no_video["metrics"] == rec_with_video["metrics"]
