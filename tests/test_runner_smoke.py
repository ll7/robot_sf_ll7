"""Smoke test for benchmark runner.

Runs a single tiny episode and validates JSON record against schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_episode, validate_and_write
from robot_sf.benchmark.schema_validator import load_schema, validate_episode

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_runner_single_episode_tmp(tmp_path: Path):
    """Run one benchmark episode and validate schema-compliant JSONL persistence.

    Args:
        tmp_path: Temporary directory for writing the smoke output JSONL.
    """
    scenario = {
        "id": "smoke-uni-low-open",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 1,
    }
    record = run_episode(scenario, seed=123, horizon=15, dt=0.1, record_forces=False)
    # Basic field presence
    assert record["scenario_id"] == scenario["id"]
    assert "metrics" in record
    assert "metric_parameters" in record
    assert "threshold_signature" in record["metric_parameters"]
    # Validate schema
    schema = load_schema(SCHEMA_PATH)
    validate_episode(record, schema)
    # Write and re-read
    out_file = tmp_path / "episode.jsonl"
    validate_and_write(record, SCHEMA_PATH, out_file)
    with out_file.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    reloaded = json.loads(line)
    assert reloaded["episode_id"] == record["episode_id"]
