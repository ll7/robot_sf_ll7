"""TODO docstring. Document this module."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from robot_sf.benchmark import runner
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


def test_run_batch_parallel_writes_output_in_job_order(tmp_path: Path, monkeypatch) -> None:
    """Parallel batch execution should preserve job order in output files."""

    scenarios = [
        {"id": "job-a", "repeats": 1},
        {"id": "job-b", "repeats": 1},
    ]
    out_file = tmp_path / "episodes.jsonl"

    def fake_worker(job):
        scenario, seed, _ = job
        if scenario["id"] == "job-a":
            time.sleep(0.05)
        return {"episode_id": f"{scenario['id']}-{seed}"}

    def fake_write(out_path, schema, record):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(runner, "_run_job_worker", fake_worker)
    monkeypatch.setattr(runner, "_write_validated_record", fake_write)
    monkeypatch.setattr(runner, "load_schema", lambda path: {})

    summary = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=1,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=2,
        resume=False,
    )

    assert summary["written"] == 2
    lines = out_file.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records[0]["episode_id"].startswith("job-a-")
    assert records[1]["episode_id"].startswith("job-b-")
