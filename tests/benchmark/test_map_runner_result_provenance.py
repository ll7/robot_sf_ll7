"""Result-artifact provenance tests for the map benchmark runner."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark import map_runner
from robot_sf.benchmark.manifest import manifest_path_for, save_manifest
from robot_sf.benchmark.map_runner import _map_result_provenance, run_map_batch
from robot_sf.benchmark.runner import _finalize_batch


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
    # With no scenarios the metric-affecting run-config block is fail-soft but
    # still present, so the manifest is explicit about the missing provenance.
    assert config_identity["metric_affecting_config"] == {
        "status": "not_available",
        "reason": "no scenarios",
    }

    seed_identity = provenance["seed_identity"]
    assert seed_identity["suite_key"] == "default"
    assert seed_identity["total_jobs"] == 0
    assert seed_identity["written"] == 0


def test_run_map_batch_records_result_manifest_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sidecar write failure must not discard an otherwise valid batch summary."""
    monkeypatch.setattr(
        map_runner,
        "_write_result_provenance_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only output")),
    )

    summary = run_map_batch(
        [],
        tmp_path / "empty.jsonl",
        Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo="goal",
        benchmark_profile="experimental",
        workers=1,
        resume=False,
    )

    assert summary["total_jobs"] == 0
    assert summary["written"] == 0
    assert summary["provenance"]["result_manifest_status"] == "error"
    assert "read-only output" in summary["provenance"]["result_manifest_error"]


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
    # Backwards compatible: omitting the optional block leaves config_identity unchanged.
    assert "metric_affecting_config" not in provenance["config_identity"]


def test_map_result_provenance_embeds_metric_affecting_config() -> None:
    """A provided metric-affecting block is embedded under config_identity (issue #3701)."""
    block = {
        "schema": "metric_affecting_run_config.v1",
        "sensor_noise": {"scan_noise": [0.0, 0.0], "scan_noise_enabled": False},
        "collision_regime": {"regime": "terminate_on_contact"},
    }
    provenance = _map_result_provenance(
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=Path("configs/example.yaml"),
        scenarios=[{"name": "one"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="experimental",
        suite_key="default",
        total_jobs=1,
        written=1,
        artifact_pointer_status="local_jsonl_present",
        metric_affecting_config=block,
    )

    assert provenance["config_identity"]["metric_affecting_config"] == block


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


def test_save_manifest_emits_simulation_run_provenance_bundle(tmp_path: Path) -> None:
    """Resume sidecars should link inputs, outputs, reports, and explicit missing optionals."""
    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text('{"episode_id":"scenario-a--7"}\n', encoding="utf-8")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenario: a\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"status":"diagnostic"}\n', encoding="utf-8")

    save_manifest(
        out_path,
        ["scenario-a--7"],
        identity_hash="identity-v1",
        input_paths=[scenario_path],
        report_paths=[summary_path],
    )

    payload = json.loads(manifest_path_for(out_path).read_text(encoding="utf-8"))
    provenance = payload["simulation_run_provenance"]

    assert provenance["schema_version"] == "simulation_run_provenance.v1"
    assert provenance["bundle_status"] == "complete"
    assert provenance["optional_fields"] == {
        "run_id": None,
        "invocation": None,
        "config_path": None,
        "scenario_path": None,
    }
    assert provenance["inputs"][0]["path"] == str(scenario_path)
    assert len(provenance["inputs"][0]["sha256"]) == 64
    assert provenance["outputs"][0]["path"] == str(out_path)
    assert provenance["outputs"][0]["sha256"]
    assert provenance["generated_reports"][0]["path"] == str(summary_path)
    assert provenance["generated_reports"][0]["sha256"]
    stable_identifiers = provenance["stable_identifiers"]
    assert len(stable_identifiers["episode_ids_sha256"]) == 64
    assert stable_identifiers["identity_hash"] == "identity-v1"
    assert stable_identifiers["schema_version"] == "v1"


def test_save_manifest_records_unavailable_optional_artifacts(tmp_path: Path) -> None:
    """Optional provenance inputs should be explicit when missing or not file artifacts."""
    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text('{"episode_id":"scenario-a--7"}\n', encoding="utf-8")
    missing_path = tmp_path / "missing.yaml"
    directory_path = tmp_path / "report_dir"
    directory_path.mkdir()

    save_manifest(
        out_path,
        ["scenario-a--7"],
        identity_hash="identity-v1",
        input_paths=[missing_path],
        report_paths=[directory_path],
    )

    payload = json.loads(manifest_path_for(out_path).read_text(encoding="utf-8"))
    provenance = payload["simulation_run_provenance"]

    assert provenance["inputs"][0] == {
        "path": str(missing_path),
        "artifact_status": "missing",
        "sha256": None,
        "size": None,
        "mtime_ns": None,
    }
    assert provenance["generated_reports"][0] == {
        "path": str(directory_path),
        "artifact_status": "not_file",
        "sha256": None,
        "size": None,
        "mtime_ns": None,
    }
    assert provenance["outputs"][0]["artifact_status"] == "available"
    assert provenance["outputs"][0]["sha256"]


def test_finalize_batch_threads_inputs_into_resume_manifest_provenance(tmp_path: Path) -> None:
    """Batch finalization should preserve schema/scenario inputs in resume sidecar provenance."""
    out_path = tmp_path / "episodes.jsonl"
    out_path.write_text('{"episode_id":"scenario-a--7"}\n', encoding="utf-8")
    schema_path = tmp_path / "episode.schema.json"
    schema_path.write_text('{"schema_version":"v1"}\n', encoding="utf-8")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenario: a\n", encoding="utf-8")

    _finalize_batch(
        out_path,
        wrote=1,
        resume=True,
        provenance_input_paths=[schema_path, scenario_path],
    )

    payload = json.loads(manifest_path_for(out_path).read_text(encoding="utf-8"))
    provenance = payload["simulation_run_provenance"]

    assert [entry["path"] for entry in provenance["inputs"]] == [
        str(schema_path),
        str(scenario_path),
    ]
    assert all(entry["sha256"] for entry in provenance["inputs"])
    assert provenance["outputs"][0]["path"] == str(out_path)
