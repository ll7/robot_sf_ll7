"""Smoke test for benchmark runner.

Runs a single tiny episode and validates JSON record against schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.benchmark import runner as runner_mod
from robot_sf.benchmark.runner import run_batch, run_episode, validate_and_write
from robot_sf.benchmark.schema_validator import load_schema, validate_episode

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


class _ObservationCapture:
    """Minimal Observation stand-in used to inspect runner payload radii."""

    def __init__(self, **payload):
        self.payload = payload


def test_runner_observation_and_episode_data_share_resolved_radii():
    """Synthetic runner helpers should use the same radii for planners and metrics."""
    scenario = {
        "robot_config": {"radius": 0.42},
        "simulation_config": {"ped_radius": 0.24},
    }
    robot_radius = runner_mod._scenario_robot_radius_m(scenario)
    ped_radius = runner_mod._scenario_ped_radius_m(scenario)
    obs = runner_mod._build_observation(
        _ObservationCapture,
        np.zeros(2),
        np.zeros(2),
        np.ones(2),
        np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float),
        0.1,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )
    ep = runner_mod._build_episode_data(
        [np.zeros(2)],
        [np.zeros(2)],
        [np.zeros(2)],
        [np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float)],
        [np.zeros((2, 2), dtype=float)],
        obstacles=None,
        goal=np.ones(2),
        dt=0.1,
        reached_goal_step=None,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )

    assert obs.payload["robot"]["radius"] == 0.42
    assert [agent["radius"] for agent in obs.payload["agents"]] == [0.24, 0.24]
    assert ep.robot_radius == 0.42
    assert ep.ped_radius == 0.24


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
    """Remove runtime-only fields before comparing deterministic episode records."""
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
