"""Tests for the benchmark result provenance manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.result_provenance import (
    SCHEMA_VERSION,
    ProvenanceRequiredFieldError,
    build_result_provenance_manifest,
    manifest_path_for_result_jsonl,
    validate_result_provenance_manifest,
    write_result_provenance_manifest,
)


def test_manifest_path_convention() -> None:
    """The provenance manifest path appends .provenance.json to the JSONL path."""
    jsonl_path = Path("output/some_run/episodes.jsonl")
    manifest_path = manifest_path_for_result_jsonl(jsonl_path)
    assert manifest_path == Path("output/some_run/episodes.jsonl.provenance.json")


def test_build_manifest_has_correct_schema_version() -> None:
    """Every emitted manifest must carry the expected schema version."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    assert manifest["schema_version"] == SCHEMA_VERSION


def test_build_manifest_records_optional_algo_config_absence() -> None:
    """When algo_config_path is None the input entry should be explicit."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    algo_input = manifest["inputs"]["algo_config"]
    assert algo_input["artifact_status"] == "not_provided"
    assert algo_input["path"] is None
    assert algo_input["sha256"] is None


def test_build_manifest_with_rows(tmp_path: Path) -> None:
    """A manifest built from episode records should link each row correctly."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"scenario-a--7","scenario_id":"scenario-a","seed":7}\n'
        '{"episode_id":"scenario-b--42","scenario_id":"scenario-b","seed":42}\n',
        encoding="utf-8",
    )

    records = [
        {
            "episode_id": "scenario-a--7",
            "scenario_id": "scenario-a",
            "seed": 7,
            "config_hash": "abc123",
            "git_hash": "deadbeef",
        },
        {
            "episode_id": "scenario-b--42",
            "scenario_id": "scenario-b",
            "seed": 42,
            "config_hash": "def456",
            "git_hash": "cafebabe",
        },
    ]

    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=records,
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=2,
        written=2,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    assert len(manifest["rows"]) == 2
    row0 = manifest["rows"][0]
    assert row0["episode_id"] == "scenario-a--7"
    assert row0["scenario_id"] == "scenario-a"
    assert row0["seed"] == 7
    assert row0["config_hash"] == "abc123"
    assert row0["repo_commit"] == "deadbeef"
    assert row0["jsonl_line"] == 0
    assert row0["raw_artifact"] == str(jsonl_path)
    assert row0["simulator_settings"]["horizon"] == 100
    assert row0["simulator_settings"]["dt"] == 0.1
    assert row0["simulator_settings"]["record_forces"] is True
    assert row0["postprocessing"] == [
        {"step": "compute_all_metrics", "status": "completed"},
        {"step": "post_process_metrics", "status": "completed"},
    ]

    row1 = manifest["rows"][1]
    assert row1["episode_id"] == "scenario-b--42"
    assert row1["seed"] == 42
    assert row1["jsonl_line"] == 1


def test_validator_passes_complete_manifest(tmp_path: Path) -> None:
    """A well-formed manifest should pass validation without raising."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"test--0","scenario_id":"test","seed":0}\n',
        encoding="utf-8",
    )

    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "test--0",
                "scenario_id": "test",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "def",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "test"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    # Validation should not raise.
    validate_result_provenance_manifest(manifest)


def test_validator_fails_on_incomplete_manifest() -> None:
    """A manifest with missing required fields should raise."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    # Corrupt the manifest by deleting a required row field.
    manifest["run"]["repo_commit"] = ""

    with pytest.raises(ProvenanceRequiredFieldError, match="run.repo_commit"):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_missing_scenario_matrix_hash() -> None:
    """A manifest without campaign_identity.scenario_matrix_hash should fail."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    del manifest["campaign_identity"]["scenario_matrix_hash"]

    with pytest.raises(
        ProvenanceRequiredFieldError, match="campaign_identity.scenario_matrix_hash"
    ):
        validate_result_provenance_manifest(manifest)


def test_validator_fails_on_missing_episodes_jsonl_artifact() -> None:
    """A manifest without an episodes_jsonl artifact entry should fail validation."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    manifest["raw_artifacts"] = []

    with pytest.raises(ProvenanceRequiredFieldError, match="episodes_jsonl"):
        validate_result_provenance_manifest(manifest)


def test_write_and_reload_roundtrip(tmp_path: Path) -> None:
    """Writing a manifest and reloading it should preserve all fields."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"a--1","scenario_id":"a","seed":1}\n',
        encoding="utf-8",
    )
    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "a--1",
                "scenario_id": "a",
                "seed": 1,
                "config_hash": "abc",
                "git_hash": "def",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "a"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=False,
        active_observation_mode="lidar",
        active_observation_level="full",
    )

    manifest_path = manifest_path_for_result_jsonl(jsonl_path)
    write_result_provenance_manifest(manifest_path, manifest)
    assert manifest_path.exists()

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert len(raw["rows"]) == 1
    assert raw["rows"][0]["episode_id"] == "a--1"
    assert raw["rows"][0]["simulator_settings"]["record_forces"] is False


def test_build_manifest_uses_git_hash_from_records(tmp_path: Path) -> None:
    """Row repo_commit should use the record's git_hash when present."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        '{"episode_id":"x--0","scenario_id":"x","seed":0}\n',
        encoding="utf-8",
    )
    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "x--0",
                "scenario_id": "x",
                "seed": 0,
                "config_hash": "abc",
                "git_hash": "from_record",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "x"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    assert manifest["rows"][0]["repo_commit"] == "from_record"


def test_skipped_manifest_has_not_applicable_completeness() -> None:
    """A manifest for a skipped (preflight) run should have appropriate completeness."""
    manifest = build_result_provenance_manifest(
        out_path=Path("episodes.jsonl"),
        episode_records=[],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test_suite",
        total_jobs=0,
        written=0,
        horizon=100,
        dt=0.1,
        record_forces=True,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    assert manifest["completeness"]["status"] == "not_applicable"
    assert manifest["completeness"]["reason"] == "preflight_skipped"
