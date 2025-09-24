from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


@pytest.mark.skipif(
    importlib.util.find_spec("moviepy") is None,
    reason="moviepy/ffmpeg not available",
)
def test_run_batch_with_synthetic_video(tmp_path: Path):
    # Minimal in-memory scenario list (single episode)
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
        }
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
    vpath = Path(video.get("path"))
    assert vpath.exists() and vpath.suffix == ".mp4"
    assert int(video.get("filesize_bytes", 0)) > 0
    assert int(video.get("frames", 0)) >= 1
