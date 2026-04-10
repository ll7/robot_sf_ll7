"""Smoke test for benchmark runner.

Runs a single tiny episode and validates JSON record against schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch, run_episode, validate_and_write
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


def _normalize_episode_record(record: dict[str, object]) -> dict[str, object]:
    normalized = dict(record)
    normalized.pop("timestamps", None)
    normalized.pop("wall_time_sec", None)
    normalized.pop("timing", None)
    return normalized


def test_run_batch_repeated_runs_produce_stable_metrics(tmp_path: Path) -> None:
    """Repeat a deterministic run and assert stable episode record contents except runtime metadata."""
    scenario = {
        "id": "repro-sample",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 1,
    }
    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    run_batch(
        [scenario],
        out_path=out1,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=1,
        resume=False,
    )
    run_batch(
        [scenario],
        out_path=out2,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=1,
        resume=False,
    )
    # Parse both outputs
    raw1 = out1.read_text(encoding="utf-8").splitlines()[0]
    raw2 = out2.read_text(encoding="utf-8").splitlines()[0]
    rec1 = json.loads(raw1)
    rec2 = json.loads(raw2)

    # Verify JSON key ordering is consistent (deterministic serialization)
    # by comparing the key order at the top level
    keys1 = list(rec1.keys())
    keys2 = list(rec2.keys())
    assert keys1 == keys2, (
        f"Top-level key order differs (JSON serialization may be non-deterministic):\n"
        f"Run 1 keys: {keys1}\n"
        f"Run 2 keys: {keys2}"
    )

    # Then verify semantic equality (ignoring runtime metadata)
    assert _normalize_episode_record(rec1) == _normalize_episode_record(rec2)
