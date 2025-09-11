from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def test_run_batch_to_tmp(tmp_path: Path):
    scenarios = [
        {
            "id": "batch-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 2,
        }
    ]
    out_file = tmp_path / "episodes.jsonl"
    summary = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=10,
        dt=0.1,
        record_forces=False,
        append=False,
    )
    assert summary["total_jobs"] == 2
    assert summary["written"] == 2
    assert out_file.exists()
    # file should have 2 lines
    assert out_file.read_text(encoding="utf-8").strip().count("\n") in (0, 1)
    # quick sanity: load lines
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
