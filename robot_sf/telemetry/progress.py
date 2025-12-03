"""Progress tracking helpers for long-running pipeline workflows."""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import threading
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.telemetry.models import PipelineRunStatus, StepExecutionEntry, StepStatus

_DEFAULT_FAILURE_SIGNALS = tuple(
    (sig.value if hasattr(sig, "value") else sig)
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None))
    if sig is not None
)

if TYPE_CHECKING:
    from types import FrameType

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
        """Init.

        Args:
            steps: Number of steps executed.
            writer: Telemetry writer.
            log_fn: log fn.
            time_provider: time provider.

        Returns:
            None: none.
        """
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
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._failure_guard: _FailureSafeGuard | None = None
        self._failure_guard_callback: Callable[[PipelineRunStatus], None] | None = None
        self._write_index()

    @property
    def entries(self) -> list[StepExecutionEntry]:
        """Return the live step entries."""

        with self._lock:
            return list(self._entries)

    def clone_entries(self) -> list[StepExecutionEntry]:
        """Return a deep copy of the step entries for serialization."""

        with self._lock:
            return deepcopy(self._entries)

    def start_step(self, step_id: str) -> None:
        """Start step.

        Args:
            step_id: identifier for step.

        Returns:
            None: none.
        """
        with self._lock:
            entry = self._get_entry_locked(step_id)
            if entry.status not in (StepStatus.PENDING, StepStatus.SKIPPED):
                msg = f"Cannot start step {step_id!r} in state {entry.status.value!r}"
                raise ValueError(msg)
            now = self._clock()
            if self._start_time is None:
                self._start_time = now
            entry.status = StepStatus.RUNNING
            entry.started_at = now
            entry.eta_snapshot_seconds = self._estimate_remaining_seconds(entry)
            message = self._format_status(
                entry,
                prefix="Starting",
                total_steps_hint=len(self._entries),
            )
            self._write_index_locked()
        self._log_once(step_id, message)

    def complete_step(self, step_id: str) -> None:
        """Complete step.

        Args:
            step_id: identifier for step.

        Returns:
            None: none.
        """
        with self._lock:
            entry = self._get_entry_locked(step_id)
            if entry.status not in (StepStatus.RUNNING, StepStatus.PENDING):
                msg = f"Cannot complete step {step_id!r} in state {entry.status.value!r}"
                raise ValueError(msg)
            now = self._clock()
            entry.status = StepStatus.COMPLETED
            entry.ended_at = now
            if entry.started_at is None:
                entry.started_at = now
            entry.duration_seconds = max(0.0, (entry.ended_at - entry.started_at).total_seconds())
            entry.eta_snapshot_seconds = self._estimate_remaining_seconds(
                entry,
                include_current=False,
            )
            message = self._format_status(
                entry,
                prefix="Completed",
                total_steps_hint=len(self._entries),
            )
            self._write_index_locked()
        self._log_once(step_id, message)

    def fail_step(self, step_id: str, *, reason: str | None = None) -> None:
        """Fail step.

        Args:
            step_id: identifier for step.
            reason: Reason string recorded for diagnostics.

        Returns:
            None: none.
        """
        with self._lock:
            entry = self._get_entry_locked(step_id)
            now = self._clock()
            entry.status = StepStatus.FAILED
            entry.ended_at = now
            if entry.started_at is None:
                entry.started_at = now
            entry.duration_seconds = max(0.0, (entry.ended_at - entry.started_at).total_seconds())
            entry.eta_snapshot_seconds = 0.0
            message = self._format_status(
                entry,
                prefix="Failed",
                extra=reason,
                total_steps_hint=len(self._entries),
            )
            self._write_index_locked()
        self._emit_log(message)

    def skip_step(self, step_id: str, *, reason: str | None = None) -> None:
        """Skip step.

        Args:
            step_id: identifier for step.
            reason: Reason string recorded for diagnostics.

        Returns:
            None: none.
        """
        with self._lock:
            entry = self._get_entry_locked(step_id)
            if entry.status != StepStatus.PENDING:
                msg = f"Cannot skip step {step_id!r} in state {entry.status.value!r}"
                raise ValueError(msg)
            entry.status = StepStatus.SKIPPED
            entry.started_at = self._clock()
            entry.ended_at = entry.started_at
            entry.duration_seconds = 0.0
            entry.eta_snapshot_seconds = self._estimate_remaining_seconds(
                entry,
                include_current=False,
            )
            message = self._format_status(
                entry,
                prefix="Skipped",
                extra=reason,
                total_steps_hint=len(self._entries),
            )
            self._write_index_locked()
        self._emit_log(message)

    def current_step(self) -> StepExecutionEntry | None:
        """Current step.

        Returns:
            StepExecutionEntry | None: Optional tracker step entry.
        """
        with self._lock:
            return self._current_step_locked()

    def completed_steps(self) -> int:
        """Completed steps.

        Returns:
            int: Integer value.
        """
        with self._lock:
            return sum(1 for entry in self._entries if entry.status == StepStatus.COMPLETED)

    def total_steps(self) -> int:
        """Total steps.

        Returns:
            int: Integer value.
        """
        with self._lock:
            return len(self._entries)

    def enable_failure_guard(
        self,
        *,
        heartbeat: Callable[[PipelineRunStatus], None],
        flush_interval_seconds: float = 5.0,
        signals: Sequence[int | signal.Signals] | None = None,
    ) -> None:
        """Start a heartbeat that flushes manifests and handles termination signals."""

        if heartbeat is None:
            raise ValueError("heartbeat callback is required")
        if self._writer is None:
            raise RuntimeError("Failure guard requires a manifest writer")
        if self._failure_guard is not None:
            return
        self._failure_guard_callback = heartbeat
        self._failure_guard = _FailureSafeGuard(
            has_work=self._has_active_steps,
            flush_running=lambda: self._guard_flush(PipelineRunStatus.RUNNING),
            flush_failed=lambda reason: self._guard_flush(
                PipelineRunStatus.FAILED,
                mark_failed=True,
                reason=reason,
            ),
            flush_interval=max(flush_interval_seconds, 1.0),
            signals=signals,
        )

    def disable_failure_guard(self) -> None:
        """Stop the heartbeat thread and restore original signal handlers."""

        if self._failure_guard is None:
            return
        self._failure_guard.close()
        self._failure_guard = None
        self._failure_guard_callback = None

    def trigger_failure_guard(self, *, reason: str | None = None) -> None:
        """Force an immediate failure flush (primarily for tests)."""

        if self._failure_guard is None:
            raise RuntimeError("Failure guard is not enabled")
        self._guard_flush(PipelineRunStatus.FAILED, mark_failed=True, reason=reason)

    def _guard_flush(
        self,
        status: PipelineRunStatus,
        *,
        mark_failed: bool = False,
        reason: str | None = None,
    ) -> None:
        """Guard flush.

        Args:
            status: Status string.
            mark_failed: mark failed.
            reason: Reason string recorded for diagnostics.

        Returns:
            None: none.
        """
        log_message: str | None = None
        with self._lock:
            if mark_failed:
                log_message = self._mark_running_step_failed_locked(reason)
            self._write_index_locked()
        if log_message:
            self._emit_log(log_message)
        callback = self._failure_guard_callback
        if callback is not None:
            callback(status)

    def _mark_running_step_failed_locked(self, reason: str | None) -> str | None:
        """Mark running step failed locked.

        Args:
            reason: Reason string recorded for diagnostics.

        Returns:
            str | None: Optional string value.
        """
        entry = self._current_step_locked()
        if entry is None or entry.status == StepStatus.FAILED:
            return None
        now = self._clock()
        entry.status = StepStatus.FAILED
        if entry.started_at is None:
            entry.started_at = now
        entry.ended_at = now
        entry.duration_seconds = max(0.0, (entry.ended_at - entry.started_at).total_seconds())
        entry.eta_snapshot_seconds = 0.0
        return self._format_status(
            entry,
            prefix="Failed",
            extra=reason,
            total_steps_hint=len(self._entries),
        )

    def _current_step_locked(self) -> StepExecutionEntry | None:
        """Current step locked.

        Returns:
            StepExecutionEntry | None: Optional tracker step entry.
        """
        for entry in self._entries:
            if entry.status == StepStatus.RUNNING:
                return entry
        return None

    def _has_active_steps(self) -> bool:
        """Has active steps.

        Returns:
            bool: Boolean flag.
        """
        with self._lock:
            return self._has_active_steps_locked()

    def _has_active_steps_locked(self) -> bool:
        """Has active steps locked.

        Returns:
            bool: Boolean flag.
        """
        return any(
            entry.status in (StepStatus.PENDING, StepStatus.RUNNING) for entry in self._entries
        )

    def _get_entry_locked(self, step_id: str) -> StepExecutionEntry:
        """Get entry locked.

        Args:
            step_id: identifier for step.

        Returns:
            StepExecutionEntry: Entry describing a tracker step.
        """
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
        total_steps_hint: int | None = None,
    ) -> str:
        """Format status.

        Args:
            entry: Individual manifest or tracker entry.
            prefix: String prefix used for labeling.
            extra: Additional context payload.
            total_steps_hint: total steps hint.

        Returns:
            str: String value.
        """
        total = total_steps_hint if total_steps_hint is not None else self.total_steps()
        elapsed = self._format_duration(entry.duration_seconds)
        eta = self._format_duration(entry.eta_snapshot_seconds)
        message = (
            f"{prefix} step {entry.order}/{total} – {entry.display_name}"
            f" (elapsed={elapsed}, eta={eta})"
        )
        if extra:
            message = f"{message} :: {extra}"
        return message

    def _log_once(self, step_id: str, message: str) -> None:
        """Log once.

        Args:
            step_id: identifier for step.
            message: Human-readable message.

        Returns:
            None: none.
        """
        with self._log_lock:
            last = self._last_log_messages.get(step_id)
            if last == message:
                return
            self._last_log_messages[step_id] = message
        self._emit_log(message)

    def _emit_log(self, message: str) -> None:
        """Emit log.

        Args:
            message: Human-readable message.

        Returns:
            None: none.
        """
        self._log_fn(message)

    def _write_index(self) -> None:
        """Write index.

        Returns:
            None: none.
        """
        with self._lock:
            self._write_index_locked()

    def _write_index_locked(self) -> None:
        """Write index locked.

        Returns:
            None: none.
        """
        if self._writer is None:
            return
        self._writer.write_step_index(self._entries)

    @staticmethod
    def _format_duration(value: float | None) -> str:
        """Format duration.

        Args:
            value: Scalar metric value.

        Returns:
            str: String value.
        """
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

        interval = max(poll_interval, 0.05)
        while True:
            with self._lock:
                running = any(entry.status == StepStatus.RUNNING for entry in self._entries)
            if not running:
                break
            time.sleep(interval)


class _FailureSafeGuard:
    """Background flush + signal handler for `ProgressTracker`."""

    def __init__(
        self,
        *,
        has_work: Callable[[], bool],
        flush_running: Callable[[], None],
        flush_failed: Callable[[str | None], None],
        flush_interval: float,
        signals: Sequence[int | signal.Signals] | None,
    ) -> None:
        """Init.

        Args:
            has_work: has work.
            flush_running: flush running.
            flush_failed: flush failed.
            flush_interval: flush interval.
            signals: Signal set watched by the tracker.

        Returns:
            None: none.
        """
        self._has_work = has_work
        self._flush_running = flush_running
        self._flush_failed = flush_failed
        self._flush_interval = max(flush_interval, 1.0)
        self._signals = self._normalize_signals(signals)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ProgressTrackerFlush",
            daemon=True,
        )
        self._previous_handlers: dict[int, signal.Handlers] = {}
        self._atexit_callback: Callable[[], None] | None = None
        self._thread.start()
        self._install_signal_handlers()

        def _cleanup() -> None:
            """Cleanup.

            Returns:
                None: none.
            """
            self.close()

        self._atexit_callback = _cleanup
        atexit.register(self._atexit_callback)

    def close(self) -> None:
        """Close.

        Returns:
            None: none.
        """
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._flush_interval)
        self._restore_signal_handlers()
        if self._atexit_callback is not None:
            with contextlib.suppress(ValueError):
                atexit.unregister(self._atexit_callback)
            self._atexit_callback = None

    def _run(self) -> None:
        """Run.

        Returns:
            None: none.
        """
        while not self._shutdown.wait(self._flush_interval):
            if not self._has_work():
                continue
            self._flush_running()

    def _install_signal_handlers(self) -> None:
        """Install signal handlers.

        Returns:
            None: none.
        """
        if not self._signals:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in self._signals:
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except (ValueError, OSError):
                continue
            self._previous_handlers[signum] = previous

    def _restore_signal_handlers(self) -> None:
        """Restore signal handlers.

        Returns:
            None: none.
        """
        if not self._previous_handlers:
            return
        for signum, handler in self._previous_handlers.items():
            with contextlib.suppress(ValueError):
                signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle signal.

        Args:
            signum: Signal number as delivered to the handler.
            frame: Frame index.

        Returns:
            None: none.
        """
        signal_name = self._signal_name(signum)
        self._flush_failed(f"Signal {signal_name}")
        previous = self._previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
            return
        if previous == signal.SIG_DFL:
            with contextlib.suppress(OSError):
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

    @staticmethod
    def _signal_name(signum: int) -> str:
        """Signal name.

        Args:
            signum: Signal number as delivered to the handler.

        Returns:
            str: String value.
        """
        try:
            return signal.Signals(signum).name
        except ValueError:
            return str(signum)

    @staticmethod
    def _normalize_signals(signals: Sequence[int | signal.Signals] | None) -> tuple[int, ...]:
        """Normalize signals.

        Args:
            signals: Signal set watched by the tracker.

        Returns:
            tuple[int, ...]: tuple of int, .
        """
        if signals is None:
            return tuple(int(sig) for sig in _DEFAULT_FAILURE_SIGNALS)
        normalized: list[int] = []
        for sig in signals:
            if isinstance(sig, signal.Signals):
                normalized.append(int(sig.value))
            elif isinstance(sig, int):
                normalized.append(int(sig))
        # Preserve order but drop duplicates
        return tuple(dict.fromkeys(normalized))
