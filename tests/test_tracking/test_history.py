"""History API and failure-safe flush tests for the telemetry tracker."""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from robot_sf.telemetry import (
    ManifestWriter,
    PipelineRunStatus,
    PipelineStepDefinition,
    ProgressTracker,
    RunTrackerConfig,
    list_runs,
    load_run,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "history_runs"


def _copy_history_fixtures(target_root: Path) -> None:
    """Copy history fixtures.

    Args:
        target_root: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    shutil.copytree(_FIXTURES, target_root, dirs_exist_ok=True)


def test_history_filters_runs(tmp_path: Path) -> None:
    """Test history filters runs.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    config = RunTrackerConfig(artifact_root=tmp_path)
    _copy_history_fixtures(config.run_tracker_root)

    entries = list_runs(config)
    assert [entry.run_id for entry in entries] == ["gamma", "beta", "alpha"]

    failed = list_runs(config, status="failed")
    assert [entry.run_id for entry in failed] == ["beta"]

    since_cutoff = datetime(2025, 1, 3, tzinfo=UTC)
    recent = list_runs(config, since=since_cutoff)
    assert {entry.run_id for entry in recent} == {"gamma"}

    limited = list_runs(config, limit=1)
    assert limited[0].run_id == "gamma"

    alpha = load_run(config, "alpha")
    assert alpha.status is PipelineRunStatus.COMPLETED
    assert len(alpha.steps) == 2


def test_history_discovers_nested_runs_and_filters_scenarios(tmp_path: Path) -> None:
    """Test history discovers nested runs and filters scenarios.

    Args:
        tmp_path: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    config = RunTrackerConfig(artifact_root=tmp_path)
    _copy_history_fixtures(config.run_tracker_root)

    nested_dir = config.run_tracker_root / "perf-tests" / "latest"
    nested_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = nested_dir / config.manifest_filename
    manifest_record = {
        "run_id": "perf-latest",
        "status": "completed",
        "created_at": "2025-02-01T00:00:00+00:00",
        "completed_at": "2025-02-01T00:05:00+00:00",
        "enabled_steps": ["performance_smoke_test"],
        "artifact_dir": str(nested_dir),
        "scenario_config_path": "configs/validation/minimal.yaml",
        "summary": {"scenario_id": "perf_minimal"},
        "steps": [],
    }
    manifest_path.write_text(json.dumps(manifest_record), encoding="utf-8")

    entries = list_runs(config, limit=0)
    run_ids = {entry.run_id for entry in entries}
    assert "perf-latest" in run_ids

    filtered = list_runs(config, scenario="perf_minimal")
    assert [entry.run_id for entry in filtered] == ["perf-latest"]


def test_failure_guard_marks_failed_steps(run_tracker_config: RunTrackerConfig) -> None:
    """Test failure guard marks failed steps.

    Args:
        run_tracker_config: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    writer = ManifestWriter(run_tracker_config, run_id="guard-demo")
    tracker = ProgressTracker(
        [
            PipelineStepDefinition("collect", "Collect", expected_duration_seconds=5),
            PipelineStepDefinition("train", "Train", expected_duration_seconds=10),
        ],
        writer=writer,
        log_fn=lambda *_: None,
    )
    heartbeats: list[str] = []
    tracker.enable_failure_guard(
        heartbeat=lambda status: heartbeats.append(status.value),
        flush_interval_seconds=0.05,
        signals=(),
    )

    tracker.start_step("collect")
    tracker.trigger_failure_guard(reason="test")
    tracker.disable_failure_guard()

    steps_path = writer.run_directory / run_tracker_config.steps_filename
    payload = json.loads(steps_path.read_text(encoding="utf-8"))
    assert payload[0]["status"] == "failed"
    assert heartbeats[-1] == "failed"

    # Ensure flush happened quickly; guard should not require long sleeps, but
    # add a short wait to avoid flakiness on slow CI hosts.
    time.sleep(0.05)
