"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_run_batch_resume_skips_existing(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    scenarios = [
        {
            "id": "resume-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 3,
        },
    ]
    out_file = tmp_path / "episodes.jsonl"

    # First run: fresh file, should write all episodes
    summary1 = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,  # ensure we start from a clean file
        workers=1,
        resume=True,
    )
    assert summary1["total_jobs"] == 3
    assert summary1["written"] == 3
    assert out_file.exists()
    lines1 = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines1) == 3

    # Second run: same setup, with resume enabled; should skip all and append nothing
    summary2 = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=True,  # keep existing file
        workers=1,
        resume=True,
    )
    # No new jobs should be written
    assert summary2["written"] == 0
    lines2 = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines2) == 3


def test_run_batch_resume_identity_includes_benchmark_track(tmp_path: Path):
    """Plain runner resume should not skip rows from a different observation track."""
    scenarios = [
        {
            "id": "track-resume-uni-low-open",
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
    out_file = tmp_path / "episodes.jsonl"

    first = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=1,
        resume=True,
        observation_mode="socnav_state",
        observation_level="tracked_agents_no_noise",
        benchmark_track="grid_socnav_v1",
        track_schema_version="observation-track.v1",
    )
    assert first["written"] == 1

    second = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=True,
        workers=1,
        resume=True,
        observation_mode="socnav_state",
        observation_level="tracked_agents_no_noise",
        benchmark_track="lidar_2d_v1",
        track_schema_version="observation-track.v1",
    )
    assert second["written"] == 1

    records = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert [record["benchmark_track"] for record in records] == [
        "grid_socnav_v1",
        "lidar_2d_v1",
    ]
    assert records[0]["episode_id"] != records[1]["episode_id"]
    assert records[1]["algorithm_metadata"]["benchmark_track"] == {
        "benchmark_track": "lidar_2d_v1",
        "track_schema_version": "observation-track.v1",
        "observation_level": "tracked_agents_no_noise",
        "observation_mode": "socnav_state",
    }
