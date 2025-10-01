from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def test_run_batch_resume_skips_existing(tmp_path: Path):
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
