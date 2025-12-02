"""Tests for the progress tracker utility."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from robot_sf.telemetry import ManifestWriter, PipelineStepDefinition, ProgressTracker
from robot_sf.telemetry.models import StepStatus


class FakeClock:
    """Deterministic clock helper for tracker tests."""

    def __init__(self) -> None:
        """Init.

        Returns:
            None: Auto-generated placeholder description.
        """
        self._current = datetime(2025, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Call.

        Returns:
            datetime: Auto-generated placeholder description.
        """
        return self._current

    def advance(self, seconds: float) -> datetime:
        """Advance.

        Args:
            seconds: Auto-generated placeholder description.

        Returns:
            datetime: Auto-generated placeholder description.
        """
        self._current = self._current + timedelta(seconds=seconds)
        return self._current


def _make_tracker(*, writer, log_fn, time_provider) -> ProgressTracker:
    """Make tracker.

    Args:
        writer: Auto-generated placeholder description.
        log_fn: Auto-generated placeholder description.
        time_provider: Auto-generated placeholder description.

    Returns:
        ProgressTracker: Auto-generated placeholder description.
    """
    return ProgressTracker(
        [
            PipelineStepDefinition("collect", "Collect Trajectories", expected_duration_seconds=10),
            PipelineStepDefinition("train", "Train Policy", expected_duration_seconds=20),
        ],
        writer=writer,
        log_fn=log_fn,
        time_provider=time_provider,
    )


def test_progress_tracker_emits_eta_and_writes_index(run_tracker_config) -> None:
    """Test progress tracker emits eta and writes index.

    Args:
        run_tracker_config: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    logs: list[str] = []
    clock = FakeClock()
    writer = ManifestWriter(run_tracker_config, run_id="eta-demo")
    tracker = _make_tracker(writer=writer, log_fn=logs.append, time_provider=clock)

    tracker.start_step("collect")
    assert "step 1/2" in logs[0].lower()
    assert tracker.entries[0].eta_snapshot_seconds == 30

    clock.advance(5)
    tracker.complete_step("collect")
    tracker.start_step("train")

    assert tracker.entries[0].status is StepStatus.COMPLETED
    assert tracker.entries[0].duration_seconds == pytest.approx(5)
    assert tracker.entries[1].status is StepStatus.RUNNING
    assert tracker.entries[1].eta_snapshot_seconds == 20

    steps_path = writer.run_directory / run_tracker_config.steps_filename
    data = json.loads(steps_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["status"] == "completed"
    assert data[1]["status"] == "running"
    assert data[0]["eta_snapshot_seconds"] == 20


def test_progress_tracker_handles_skip_and_fail(run_tracker_config) -> None:
    """Test progress tracker handles skip and fail.

    Args:
        run_tracker_config: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    writer = ManifestWriter(run_tracker_config, run_id="skip-demo")
    tracker = _make_tracker(
        writer=writer,
        log_fn=lambda _: None,
        time_provider=lambda: datetime.now(UTC),
    )

    tracker.skip_step("collect", reason="disabled")
    tracker.start_step("train")
    tracker.fail_step("train", reason="boom")

    entries = tracker.clone_entries()
    assert entries[0].status is StepStatus.SKIPPED
    assert entries[1].status is StepStatus.FAILED
    assert entries[1].eta_snapshot_seconds == 0
