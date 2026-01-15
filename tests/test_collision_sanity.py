"""Sanity test: the 'collision_sanity' preset should force a collision.

This test runs a single short episode using the special preset that places a
pedestrian at the default robot start location. The metrics pipeline should
report collisions > 0 for this episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path


def test_collision_sanity(tmp_path: Path):
    # Prepare output path under pytest tmp dir
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    out_path = tmp_path / "episodes_collision_sanity.jsonl"

    # Run a single episode with a short horizon (10 steps is more than enough)
    summary = run_batch(
        scenarios_or_path=[
            {"id": "C00_collision_sanity", "preset": "collision_sanity", "repeats": 1},
        ],
        out_path=str(out_path),
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        base_seed=0,
        repeats_override=1,
        horizon=10,
        dt=0.1,
        record_forces=False,
        workers=1,
        resume=False,
    )
    assert summary["written"] == 1

    # Read the single line JSONL and verify collisions > 0
    data = (out_path.read_text(encoding="utf-8").strip()).splitlines()
    assert len(data) == 1
    import json

    rec = json.loads(data[0])
    assert rec["scenario_id"] == "C00_collision_sanity"
    # collisions is an int after post-processing
    assert isinstance(rec["metrics"]["collisions"], int)
    assert rec["metrics"]["collisions"] > 0
