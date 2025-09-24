from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.cli import cli_main

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


@pytest.mark.skipif(
    importlib.util.find_spec("moviepy") is None, reason="moviepy/ffmpeg not available"
)
def test_cli_run_with_synthetic_video(tmp_path: Path, capsys):
    # Minimal scenario matrix YAML (single episode)
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "cli-video-uni-low-open",
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
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    out_jsonl = tmp_path / "episodes.jsonl"

    rc = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_jsonl),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--horizon",
            "10",
            "--dt",
            "0.1",
            "--video-renderer",
            "synthetic",
        ]
    )
    cap = capsys.readouterr()
    assert rc == 0, f"CLI run failed: {cap.err}"

    # JSONL exists with a single record
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # Video manifest should be present with expected fields
    video = rec.get("video")
    assert isinstance(video, dict)
    assert video.get("format") == "mp4"
    path = Path(video.get("path"))
    assert path.exists() and path.suffix == ".mp4"
    assert int(video.get("filesize_bytes", 0)) > 0
    assert int(video.get("frames", 0)) == 10
