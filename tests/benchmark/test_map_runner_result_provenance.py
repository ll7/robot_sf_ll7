"""Result-artifact provenance tests for the map benchmark runner."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner import run_map_batch


def test_run_map_batch_empty_summary_has_result_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even no-episode map summaries should identify code, config, seed, and artifact state."""
    invocation = ["map runner", "--profile", "value with spaces"]
    monkeypatch.setattr(sys, "argv", invocation)

    summary = run_map_batch(
        [],
        tmp_path / "empty.jsonl",
        Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo="goal",
        benchmark_profile="experimental",
        workers=1,
        resume=False,
    )

    provenance = summary["provenance"]
    assert "protocol_version" in provenance
    assert "commit_hash" in provenance
    assert "run_id" in provenance
    assert "python_version" in provenance
    assert provenance["invocation"] == shlex.join(invocation)
    assert provenance["artifact_pointer_status"] == "not_available"

    config_identity = provenance["config_identity"]
    assert config_identity["schema_path"] == "robot_sf/benchmark/schemas/episode.schema.v1.json"
    assert config_identity["scenario_path"] == "."
    assert config_identity["scenario_count"] == 0
    assert config_identity["algo"] == "goal"
    assert config_identity["algo_config_path"] is None
    assert config_identity["benchmark_profile"] == "experimental"
    assert isinstance(config_identity["scenario_matrix_hash"], str)

    seed_identity = provenance["seed_identity"]
    assert seed_identity["suite_key"] == "default"
    assert seed_identity["total_jobs"] == 0
    assert seed_identity["written"] == 0
