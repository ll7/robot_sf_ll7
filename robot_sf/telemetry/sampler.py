"""Resource telemetry sampling loop for the run tracker."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from loguru import logger

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil not installed
    psutil = None  # type: ignore

try:  # pragma: no cover - platform dependent fallback
    import resource
except ImportError:  # pragma: no cover - resource missing on some platforms
    resource = None  # type: ignore

if psutil is not None:  # pragma: no branch - constant per interpreter
    _PSUTIL_ERRORS: tuple[type[Exception], ...] = (psutil.Error, OSError)
else:  # pragma: no cover - psutil missing
    _PSUTIL_ERRORS = (OSError,)

from robot_sf.telemetry.gpu import collect_gpu_sample
from robot_sf.telemetry.models import TelemetrySnapshot

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from collections.abc import Callable

    from robot_sf.telemetry.manifest_writer import ManifestWriter
    from robot_sf.telemetry.progress import ProgressTracker


class SnapshotConsumer(Protocol):
    """Protocol for callbacks that consume telemetry snapshots."""

    def __call__(self, snapshot: TelemetrySnapshot) -> None:
        """Handle a telemetry snapshot."""


class TelemetrySampler:
    """Background sampler that records resource metrics at fixed intervals."""

    def __init__(
        self,
        writer: ManifestWriter,
        *,
        progress_tracker: ProgressTracker | None,
        started_at: datetime,
        interval_seconds: float = 1.0,
        time_provider: Callable[[], datetime] | None = None,
        step_rate_provider: Callable[[], float | None] | None = None,
    ) -> None:
        """Init.

        Args:
            writer: Auto-generated placeholder description.
            progress_tracker: Auto-generated placeholder description.
            started_at: Auto-generated placeholder description.
            interval_seconds: Auto-generated placeholder description.
            time_provider: Auto-generated placeholder description.
            step_rate_provider: Auto-generated placeholder description.

        Returns:
            None: Auto-generated placeholder description.
        """
        self._writer = writer
        self._progress_tracker = progress_tracker
        self._started_at = started_at
        self._interval = max(interval_seconds, 0.5)
        self._clock = time_provider or (lambda: datetime.now(UTC))
        self._step_rate_provider = step_rate_provider or self._default_step_rate
        self._process = self._init_process_handle()
        self._system_cpu_fn = getattr(psutil, "cpu_percent", None)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[SnapshotConsumer] = []
        self._samples_written = 0
        self._last_snapshot: TelemetrySnapshot | None = None

    def start(self) -> None:
        """Start the background sampling loop."""

        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="TelemetrySampler", daemon=True)
        self._thread.start()

    def stop(self, *, flush_final: bool = True) -> None:
        """Stop the sampling loop and optionally emit a final snapshot."""

        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval * 2)
        self._thread = None
        if flush_final:
            with contextlib.suppress(Exception):
                self.emit_snapshot()

    def close(self) -> None:
        """Alias for :meth:`stop` to support context-manager style usage."""

        self.stop()

    def __enter__(self) -> TelemetrySampler:
        """Enter.

        Returns:
            TelemetrySampler: Auto-generated placeholder description.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Exit.

        Args:
            exc_type: Auto-generated placeholder description.
            exc: Auto-generated placeholder description.
            tb: Auto-generated placeholder description.

        Returns:
            None: Auto-generated placeholder description.
        """
        self.stop()

    def add_consumer(self, consumer: SnapshotConsumer) -> None:
        """Register a callback that receives every sampled snapshot."""

        self._callbacks.append(consumer)

    @property
    def samples_written(self) -> int:
        """Return the number of snapshots persisted so far."""

        return self._samples_written

    @property
    def last_snapshot(self) -> TelemetrySnapshot | None:
        """Return the most recent snapshot, if any."""

        return self._last_snapshot

    def emit_snapshot(self) -> TelemetrySnapshot:
        """Collect a snapshot immediately and persist it."""

        snapshot = self._collect_snapshot()
        self._writer.append_telemetry_snapshot(snapshot)
        for consumer in list(self._callbacks):
            with contextlib.suppress(Exception):
                consumer(snapshot)
        self._samples_written += 1
        self._last_snapshot = snapshot
        return snapshot

    def _run_loop(self) -> None:
        """Run loop.

        Returns:
            None: Auto-generated placeholder description.
        """
        while not self._stop_event.is_set():
            start = time.perf_counter()
            with contextlib.suppress(Exception):
                self.emit_snapshot()
            elapsed = time.perf_counter() - start
            sleep_for = max(self._interval - elapsed, 0.1)
            self._stop_event.wait(timeout=sleep_for)

    def _collect_snapshot(self) -> TelemetrySnapshot:
        """Collect snapshot.

        Returns:
            TelemetrySnapshot: Auto-generated placeholder description.
        """
        timestamp = self._clock()
        timestamp_ms = int(timestamp.timestamp() * 1000)
        step_id = self._resolve_current_step()
        steps_per_sec = self._safe_call(self._step_rate_provider)
        notes: set[str] = set()

        cpu_process = self._sample_process_cpu(notes)
        cpu_system = self._sample_system_cpu(notes)
        memory_rss = self._sample_memory_mb(notes)

        gpu_sample = collect_gpu_sample()
        gpu_util = gpu_mem_used = None
        if gpu_sample:
            gpu_util = gpu_sample.util_percent
            gpu_mem_used = gpu_sample.memory_used_mb
            if gpu_sample.notes:
                notes.add(gpu_sample.notes)

        snapshot = TelemetrySnapshot(
            timestamp_ms=timestamp_ms,
            step_id=step_id,
            steps_per_sec=steps_per_sec,
            cpu_percent_process=cpu_process,
            cpu_percent_system=cpu_system,
            memory_rss_mb=memory_rss,
            gpu_util_percent=gpu_util,
            gpu_mem_used_mb=gpu_mem_used,
            notes=", ".join(sorted(notes)) if notes else None,
        )
        return snapshot

    def _resolve_current_step(self) -> str | None:
        """Resolve current step.

        Returns:
            str | None: Auto-generated placeholder description.
        """
        tracker = self._progress_tracker
        if tracker is None:
            return None
        with contextlib.suppress(Exception):
            current = tracker.current_step()
            if current is not None:
                return current.step_id
        return None

    def _default_step_rate(self) -> float | None:
        """Default step rate.

        Returns:
            float | None: Auto-generated placeholder description.
        """
        tracker = self._progress_tracker
        if tracker is None:
            return None
        completed = tracker.completed_steps()
        elapsed = max((self._clock() - self._started_at).total_seconds(), 0.0)
        if elapsed <= 0.0:
            return None
        return completed / elapsed

    def _sample_process_cpu(self, notes: set[str]) -> float | None:
        """Sample process cpu.

        Args:
            notes: Auto-generated placeholder description.

        Returns:
            float | None: Auto-generated placeholder description.
        """
        if self._process is None:
            notes.add("psutil-unavailable")
            return None
        try:
            return float(self._process.cpu_percent(interval=None))
        except _PSUTIL_ERRORS as exc:  # pragma: no cover - unexpected psutil failure
            notes.add(f"cpu-process:{exc.__class__.__name__}")
            self._process = None
            return None

    def _sample_system_cpu(self, notes: set[str]) -> float | None:
        """Sample system cpu.

        Args:
            notes: Auto-generated placeholder description.

        Returns:
            float | None: Auto-generated placeholder description.
        """
        if not self._system_cpu_fn:
            notes.add("system-cpu-unavailable")
            return None
        try:
            return float(self._system_cpu_fn(interval=None))
        except _PSUTIL_ERRORS as exc:  # pragma: no cover - psutil failure
            notes.add(f"cpu-system:{exc.__class__.__name__}")
            self._system_cpu_fn = None
            return None

    def _sample_memory_mb(self, notes: set[str]) -> float | None:
        """Sample memory mb.

        Args:
            notes: Auto-generated placeholder description.

        Returns:
            float | None: Auto-generated placeholder description.
        """
        if self._process is not None:
            try:
                memory_info = self._process.memory_info()
                return float(memory_info.rss) / (1024**2)
            except _PSUTIL_ERRORS as exc:  # pragma: no cover - psutil failure
                notes.add(f"memory:{exc.__class__.__name__}")
                self._process = None
        return self._resource_memory_mb(notes)

    @staticmethod
    def _resource_memory_mb(notes: set[str]) -> float | None:
        """Resource memory mb.

        Args:
            notes: Auto-generated placeholder description.

        Returns:
            float | None: Auto-generated placeholder description.
        """
        if resource is None:  # pragma: no cover - platform without resource
            notes.add("resource-module-unavailable")
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss_kb / (1024**2)
        return rss_kb / 1024.0 if rss_kb else None

    @staticmethod
    def _safe_call(func: Callable[[], float | None] | None) -> float | None:
        """Safe call.

        Args:
            func: Auto-generated placeholder description.

        Returns:
            float | None: Auto-generated placeholder description.
        """
        if func is None:
            return None
        with contextlib.suppress(Exception):
            return func()
        return None

    def _init_process_handle(self):
        """Init process handle.

        Returns:
            Any: Auto-generated placeholder description.
        """
        if psutil is None:
            logger.debug("psutil not available; telemetry sampler will skip CPU/memory metrics")
            return None
        try:
            process = psutil.Process()
            process.cpu_percent(interval=None)
            return process
        except _PSUTIL_ERRORS as exc:  # pragma: no cover - psutil failure
            logger.warning("Unable to initialize psutil.Process for telemetry: {}", exc)
            return None
