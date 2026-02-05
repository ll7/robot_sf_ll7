"""TODO docstring. Document this module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_run_batch_resume_parallel_skips_existing(tmp_path: Path):
    """Resume parallel batch runs without duplicating existing episodes."""
    scenarios = [
        {
            "id": "resume-parallel-uni-low-open",
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

    # First run in parallel: should write all episodes
    summary1 = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=3,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=2,  # parallel path
        resume=True,
    )
    assert summary1["total_jobs"] == 1
    assert summary1["written"] == 1
    assert out_file.exists()
    lines1 = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines1) == 1

    # Second run in parallel with resume: should skip all and write nothing
    summary2 = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=3,
        dt=0.1,
        record_forces=False,
        append=True,  # keep file and ensure no new lines are appended
        workers=2,  # parallel path again
        resume=True,
    )
    assert summary2["written"] == 0
    lines2 = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines2) == 1
