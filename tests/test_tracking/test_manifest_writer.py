"""Tests for manifest writing + registry primitives."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from robot_sf.telemetry.config import RunTrackerConfig
from robot_sf.telemetry.manifest_writer import ManifestWriter
from robot_sf.telemetry.models import (
    PipelineRunRecord,
    PipelineRunStatus,
    StepExecutionEntry,
    StepStatus,
)
from robot_sf.telemetry.run_registry import RunRegistry


def test_manifest_writer_appends_and_reads_records(run_tracker_config: RunTrackerConfig) -> None:
    """TODO docstring. Document this function.

    Args:
        run_tracker_config: TODO docstring.
    """
    registry = RunRegistry(run_tracker_config)
    writer = ManifestWriter(run_tracker_config, run_id="demo-run", registry=registry)
    step = StepExecutionEntry(
        step_id="collect",
        display_name="Collect Expert Trajectories",
        order=1,
        status=StepStatus.RUNNING,
    )
    record = PipelineRunRecord(
        run_id="demo-run",
        created_at=datetime.now(UTC),
        status=PipelineRunStatus.RUNNING,
        enabled_steps=("collect", "train"),
        artifact_dir=writer.run_directory,
        steps=[step],
    )
    writer.append_run_record(record)
    writer.write_step_index([step])

    entries = writer.iter_run_records()
    assert len(entries) == 1
    assert entries[0]["run_id"] == "demo-run"
    assert (writer.run_directory / run_tracker_config.steps_filename).exists()


def test_run_registry_enforces_unique_ids(run_tracker_config: RunTrackerConfig) -> None:
    """TODO docstring. Document this function.

    Args:
        run_tracker_config: TODO docstring.
    """
    registry = RunRegistry(run_tracker_config)
    registry.create_run_directory("dupe-run")
    with pytest.raises(FileExistsError):
        registry.create_run_directory("dupe-run")


def test_run_registry_prunes_old_runs(tmp_path) -> None:
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    config = RunTrackerConfig(artifact_root=tmp_path / "artifacts", retain_runs=2)
    registry = RunRegistry(config)
    for idx in range(3):
        registry.create_run_directory(f"run-{idx}")
    registry.prune()
    dirs = registry.list_run_directories()
    assert [path.name for path in dirs] == ["run-1", "run-2"]
