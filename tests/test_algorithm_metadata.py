"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_algorithm_metadata_present(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    matrix = [
        {
            "id": "algo-meta-smoke",
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
        matrix,
        out_path=out_jsonl,
        schema_path=SCHEMA_PATH,
        base_seed=0,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
    )
    assert summary["written"] == 1
    content = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    rec = json.loads(content[0])
    assert "algorithm_metadata" in rec
    algo_md = rec["algorithm_metadata"]
    assert isinstance(algo_md, dict)
    assert "algorithm" in algo_md
    assert algo_md["baseline_category"] == "classical"
    assert algo_md["policy_semantics"] == "deterministic_goal_seeking"
    planner_meta = algo_md.get("planner_kinematics")
    assert isinstance(planner_meta, dict)
    assert planner_meta.get("execution_mode") == "native"
