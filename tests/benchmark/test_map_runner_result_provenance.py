"""Result-artifact provenance tests for the map benchmark runner."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner import _map_result_provenance, run_map_batch


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
    assert provenance["artifact_pointer_status"] == "local_jsonl_present"

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


def test_map_result_provenance_preserves_planned_skipped_job_count() -> None:
    """Skipped summaries should retain the planned job count in seed identity."""
    provenance = _map_result_provenance(
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=Path("configs/example.yaml"),
        scenarios=[{"name": "one"}, {"name": "two"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="experimental",
        suite_key="default",
        total_jobs=2,
        written=0,
        artifact_pointer_status="not_available",
    )

    assert provenance["seed_identity"]["total_jobs"] == 2
    assert provenance["seed_identity"]["written"] == 0
    assert provenance["artifact_pointer_status"] == "not_available"


def test_map_result_provenance_can_mark_existing_jsonl_available() -> None:
    """Resume summaries with no new writes can still point at an existing JSONL artifact."""
    provenance = _map_result_provenance(
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=Path("configs/example.yaml"),
        scenarios=[{"name": "one"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="experimental",
        suite_key="default",
        total_jobs=1,
        written=0,
        artifact_pointer_status="local_jsonl_present",
    )

    assert provenance["seed_identity"]["total_jobs"] == 1
    assert provenance["seed_identity"]["written"] == 0
    assert provenance["artifact_pointer_status"] == "local_jsonl_present"
