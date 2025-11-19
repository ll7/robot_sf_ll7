"""Progress tracking helpers for long-running pipeline workflows."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.telemetry.models import StepExecutionEntry, StepStatus

if TYPE_CHECKING:
    from robot_sf.telemetry.manifest_writer import ManifestWriter

TimeProvider = Callable[[], datetime]
LogFunction = Callable[[str], None]


@dataclass(slots=True)
class PipelineStepDefinition:
    """Declarative definition for a pipeline step."""

    step_id: str
    display_name: str
    expected_duration_seconds: float | None = None


class ProgressTracker:
    """Track step lifecycle events and emit ETA-aware status updates."""

    def __init__(
        self,
        steps: Sequence[PipelineStepDefinition],
        *,
        writer: ManifestWriter | None = None,
        log_fn: LogFunction | None = None,
        time_provider: TimeProvider | None = None,
    ) -> None:
        if not steps:
            raise ValueError("ProgressTracker requires at least one step definition")
        self._definitions = list(steps)
        self._writer = writer
        self._clock = time_provider or (lambda: datetime.now(UTC))
        self._log_fn = log_fn or logger.info
        self._entries: list[StepExecutionEntry] = [
            StepExecutionEntry(
                step_id=definition.step_id,
                display_name=definition.display_name,
                order=index + 1,
            )
            for index, definition in enumerate(self._definitions)
        ]
        self._start_time: datetime | None = None
        self._last_log_messages: dict[str, str] = {}
        self._write_index()

    @property
    def entries(self) -> list[StepExecutionEntry]:
        """Return the live step entries."""

        return self._entries

    def clone_entries(self) -> list[StepExecutionEntry]:
        """Return a deep copy of the step entries for serialization."""

        return deepcopy(self._entries)

    def start_step(self, step_id: str) -> None:
        entry = self._get_entry(step_id)
        if entry.status not in (StepStatus.PENDING, StepStatus.SKIPPED):
            raise ValueError(f"Cannot start step {step_id!r} in state {entry.status.value!r}")
        now = self._clock()
        if self._start_time is None:
            self._start_time = now
        entry.status = StepStatus.RUNNING
        entry.started_at = now
        entry.eta_snapshot_seconds = self._estimate_remaining_seconds(entry)
        message = self._format_status(entry, prefix="Starting")
        self._log_once(step_id, message)
        self._write_index()

    def complete_step(self, step_id: str) -> None:
        entry = self._get_entry(step_id)
        if entry.status not in (StepStatus.RUNNING, StepStatus.PENDING):
            raise ValueError(f"Cannot complete step {step_id!r} in state {entry.status.value!r}")
        now = self._clock()
        entry.status = StepStatus.COMPLETED
        entry.ended_at = now
        if entry.started_at is None:
            entry.started_at = now
        entry.duration_seconds = max(0.0, (entry.ended_at - entry.started_at).total_seconds())
        entry.eta_snapshot_seconds = self._estimate_remaining_seconds(entry, include_current=False)
        message = self._format_status(entry, prefix="Completed")
        self._log_once(step_id, message)
        self._write_index()

    def fail_step(self, step_id: str, *, reason: str | None = None) -> None:
        entry = self._get_entry(step_id)
        now = self._clock()
        entry.status = StepStatus.FAILED
        entry.ended_at = now
        if entry.started_at is None:
            entry.started_at = now
        entry.duration_seconds = max(0.0, (entry.ended_at - entry.started_at).total_seconds())
        entry.eta_snapshot_seconds = 0.0
        message = self._format_status(entry, prefix="Failed", extra=reason)
        self._emit_log(message)
        self._write_index()

    def skip_step(self, step_id: str, *, reason: str | None = None) -> None:
        entry = self._get_entry(step_id)
        if entry.status != StepStatus.PENDING:
            raise ValueError(f"Cannot skip step {step_id!r} in state {entry.status.value!r}")
        entry.status = StepStatus.SKIPPED
        entry.started_at = self._clock()
        entry.ended_at = entry.started_at
        entry.duration_seconds = 0.0
        entry.eta_snapshot_seconds = self._estimate_remaining_seconds(entry, include_current=False)
        message = self._format_status(entry, prefix="Skipped", extra=reason)
        self._emit_log(message)
        self._write_index()

    def current_step(self) -> StepExecutionEntry | None:
        for entry in self._entries:
            if entry.status == StepStatus.RUNNING:
                return entry
        return None

    def completed_steps(self) -> int:
        return sum(1 for entry in self._entries if entry.status == StepStatus.COMPLETED)

    def total_steps(self) -> int:
        return len(self._entries)

    def _get_entry(self, step_id: str) -> StepExecutionEntry:
        for entry in self._entries:
            if entry.step_id == step_id:
                return entry
        raise KeyError(f"Unknown step_id: {step_id}")

    def _estimate_remaining_seconds(
        self,
        current_entry: StepExecutionEntry,
        *,
        include_current: bool = True,
    ) -> float:
        """Estimate remaining runtime based on expected durations and actuals."""

        remaining = 0.0
        for entry, definition in zip(self._entries, self._definitions, strict=True):
            if entry.order < current_entry.order:
                continue
            if entry.order == current_entry.order and not include_current:
                continue
            if entry.status == StepStatus.COMPLETED:
                continue
            expected = definition.expected_duration_seconds
            if entry.duration_seconds:
                expected = entry.duration_seconds
            remaining += expected or 0.0
        return remaining

    def _format_status(
        self,
        entry: StepExecutionEntry,
        *,
        prefix: str,
        extra: str | None = None,
    ) -> str:
        total = self.total_steps()
        elapsed = self._format_duration(entry.duration_seconds)
        eta = self._format_duration(entry.eta_snapshot_seconds)
        message = (
            f"{prefix} step {entry.order}/{total} – {entry.display_name}"
            f" (elapsed={elapsed}, eta={eta})"
        )
        if extra:
            message = f"{message} :: {extra}"
        return message

    def _emit_log(self, message: str) -> None:
        self._log_fn(message)

    def _log_once(self, step_id: str, message: str) -> None:
        last = self._last_log_messages.get(step_id)
        if last == message:
            return
        self._last_log_messages[step_id] = message
        self._emit_log(message)

    def _write_index(self) -> None:
        if self._writer is None:
            return
        self._writer.write_step_index(self._entries)

    @staticmethod
    def _format_duration(value: float | None) -> str:
        if value is None:
            return "--"
        seconds = int(max(value, 0))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m"
        if minutes:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"

    def wait_for_completion(self, poll_interval: float = 0.5) -> None:
        """Busy-wait helper for tests that need deterministic completion."""

        while any(entry.status == StepStatus.RUNNING for entry in self._entries):
            time.sleep(max(poll_interval, 0.05))
