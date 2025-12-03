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
        """TODO docstring. Document this function.

        Args:
            steps: TODO docstring.
            writer: TODO docstring.
            log_fn: TODO docstring.
            time_provider: TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.
            reason: TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.
            reason: TODO docstring.
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
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        with self._lock:
            return self._current_step_locked()

    def completed_steps(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        with self._lock:
            return sum(1 for entry in self._entries if entry.status == StepStatus.COMPLETED)

    def total_steps(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            status: TODO docstring.
            mark_failed: TODO docstring.
            reason: TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            reason: TODO docstring.

        Returns:
            TODO docstring.
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
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        for entry in self._entries:
            if entry.status == StepStatus.RUNNING:
                return entry
        return None

    def _has_active_steps(self) -> bool:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        with self._lock:
            return self._has_active_steps_locked()

    def _has_active_steps_locked(self) -> bool:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return any(
            entry.status in (StepStatus.PENDING, StepStatus.RUNNING) for entry in self._entries
        )

    def _get_entry_locked(self, step_id: str) -> StepExecutionEntry:
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.

        Returns:
            TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            entry: TODO docstring.
            prefix: TODO docstring.
            extra: TODO docstring.
            total_steps_hint: TODO docstring.

        Returns:
            TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            step_id: TODO docstring.
            message: TODO docstring.
        """
        with self._log_lock:
            last = self._last_log_messages.get(step_id)
            if last == message:
                return
            self._last_log_messages[step_id] = message
        self._emit_log(message)

    def _emit_log(self, message: str) -> None:
        """TODO docstring. Document this function.

        Args:
            message: TODO docstring.
        """
        self._log_fn(message)

    def _write_index(self) -> None:
        """TODO docstring. Document this function."""
        with self._lock:
            self._write_index_locked()

    def _write_index_locked(self) -> None:
        """TODO docstring. Document this function."""
        if self._writer is None:
            return
        self._writer.write_step_index(self._entries)

    @staticmethod
    def _format_duration(value: float | None) -> str:
        """TODO docstring. Document this function.

        Args:
            value: TODO docstring.

        Returns:
            TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            has_work: TODO docstring.
            flush_running: TODO docstring.
            flush_failed: TODO docstring.
            flush_interval: TODO docstring.
            signals: TODO docstring.
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
            """TODO docstring. Document this function."""
            self.close()

        self._atexit_callback = _cleanup
        atexit.register(self._atexit_callback)

    def close(self) -> None:
        """TODO docstring. Document this function."""
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
        """TODO docstring. Document this function."""
        while not self._shutdown.wait(self._flush_interval):
            if not self._has_work():
                continue
            self._flush_running()

    def _install_signal_handlers(self) -> None:
        """TODO docstring. Document this function."""
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
        """TODO docstring. Document this function."""
        if not self._previous_handlers:
            return
        for signum, handler in self._previous_handlers.items():
            with contextlib.suppress(ValueError):
                signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """TODO docstring. Document this function.

        Args:
            signum: TODO docstring.
            frame: TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            signum: TODO docstring.

        Returns:
            TODO docstring.
        """
        try:
            return signal.Signals(signum).name
        except ValueError:
            return str(signum)

    @staticmethod
    def _normalize_signals(signals: Sequence[int | signal.Signals] | None) -> tuple[int, ...]:
        """TODO docstring. Document this function.

        Args:
            signals: TODO docstring.

        Returns:
            TODO docstring.
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
