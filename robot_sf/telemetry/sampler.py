"""Resource telemetry sampling loop for the run tracker."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from robot_sf.common.optional_import import try_import
from robot_sf.telemetry.gpu import collect_gpu_sample
from robot_sf.telemetry.models import TelemetrySnapshot

psutil = try_import("psutil")  # optional dependency; None when unavailable
resource = try_import("resource")  # platform-dependent stdlib (Unix); None elsewhere

if psutil is not None:  # pragma: no branch - constant per interpreter
    _PSUTIL_ERRORS: tuple[type[Exception], ...] = (psutil.Error, OSError)
else:  # pragma: no cover - psutil missing
    _PSUTIL_ERRORS = (OSError,)

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
        frame_idx_provider: Callable[[], int | None] | None = None,
        status_provider: Callable[[], str | None] | None = None,
    ) -> None:
        """Configure a background sampler for manifest telemetry snapshots.

        Args:
            writer: Manifest writer that receives every collected snapshot.
            progress_tracker: Optional tracker used to resolve the current step and
                default completed-step throughput.
            started_at: Run start time used by the default step-rate calculation.
            interval_seconds: Desired sampling period; values below 0.5 seconds are
                clamped to avoid overly tight polling.
            time_provider: Optional clock callback, primarily for deterministic tests.
            step_rate_provider: Optional callback that returns the current step rate.
            frame_idx_provider: Optional callback providing current frame index.
            status_provider: Optional callback providing run status string.
        """
        self._writer = writer
        self._progress_tracker = progress_tracker
        self._started_at = started_at
        self._interval = max(interval_seconds, 0.5)
        self._clock = time_provider or (lambda: datetime.now(UTC))
        self._step_rate_provider = step_rate_provider or self._default_step_rate
        self._frame_idx_provider = frame_idx_provider
        self._status_provider = status_provider
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
        """Start sampling and return this sampler for context-manager use.


        Returns:
            The running sampler instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Stop sampling when leaving a context-manager block.

        Args:
            exc_type: Exception type raised by the managed block, if any.
            exc: Exception instance raised by the managed block, if any.
            tb: Traceback raised by the managed block, if any.
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
        """Collect a snapshot immediately and persist it.

        Returns:
            TelemetrySnapshot: The collected snapshot that was written and broadcast to consumers.
        """

        snapshot = self._collect_snapshot()
        self._writer.append_telemetry_snapshot(snapshot)
        for consumer in list(self._callbacks):
            with contextlib.suppress(Exception):
                consumer(snapshot)
        self._samples_written += 1
        self._last_snapshot = snapshot
        return snapshot

    def _run_loop(self) -> None:
        """Emit snapshots until stopped, preserving the configured interval."""
        while not self._stop_event.is_set():
            start = time.perf_counter()
            with contextlib.suppress(Exception):
                self.emit_snapshot()
            elapsed = time.perf_counter() - start
            sleep_for = max(self._interval - elapsed, 0.1)
            self._stop_event.wait(timeout=sleep_for)

    def _collect_snapshot(self) -> TelemetrySnapshot:
        """Collect one CPU, memory, GPU, progress, and status snapshot.


        Returns:
            Snapshot with unavailable metrics left as ``None`` and explanatory
            notes for recoverable sampling failures.
        """
        timestamp = self._clock()
        timestamp_ms = int(timestamp.timestamp() * 1000)
        step_id = self._resolve_current_step()
        steps_per_sec = self._safe_call(self._step_rate_provider)
        frame_idx = self._safe_call(self._frame_idx_provider)
        status = self._safe_call(self._status_provider)
        notes: set[str] = set()

        cpu_process = self._sample_process_cpu(notes)
        cpu_system = self._sample_system_cpu(notes)
        memory_rss = self._sample_memory_mb(notes)

        gpu_sample = collect_gpu_sample()
        gpu_util = gpu_mem_used = None
        gpu_devices = None
        if gpu_sample:
            gpu_util = gpu_sample.util_percent
            gpu_mem_used = gpu_sample.memory_used_mb
            if gpu_sample.devices:
                gpu_devices = gpu_sample.devices
            if gpu_sample.notes:
                notes.add(gpu_sample.notes)

        snapshot = TelemetrySnapshot(
            timestamp_ms=timestamp_ms,
            step_id=step_id,
            steps_per_sec=steps_per_sec,
            frame_idx=frame_idx if isinstance(frame_idx, int) else None,
            status=str(status) if status is not None else None,
            cpu_percent_process=cpu_process,
            cpu_percent_system=cpu_system,
            memory_rss_mb=memory_rss,
            gpu_util_percent=gpu_util,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_devices=gpu_devices,
            notes=", ".join(sorted(notes)) if notes else None,
        )
        return snapshot

    def _resolve_current_step(self) -> str | None:
        """Return the currently running progress step id, if one is available.


        Returns:
            Current step id, or ``None`` when no tracker/active step can be read.
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
        """Estimate completed pipeline steps per second from the tracker.


        Returns:
            Completed-step rate, or ``None`` when no tracker or elapsed time is
            available.
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
        """Sample CPU usage for the current process.

        Args:
            notes: Mutable set that receives availability or error markers.

        Returns:
            Process CPU percentage, or ``None`` when psutil is unavailable/fails.
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
        """Sample whole-system CPU usage.

        Args:
            notes: Mutable set that receives availability or error markers.

        Returns:
            System CPU percentage, or ``None`` when sampling is unavailable/fails.
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
        """Sample resident memory in MiB with a resource-module fallback.

        Args:
            notes: Mutable set that receives availability or error markers.

        Returns:
            Resident memory in MiB, or ``None`` if no memory source is available.
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
        """Return peak process memory from ``resource.getrusage`` when psutil is absent.

        Args:
            notes: Mutable set that receives availability markers.

        Returns:
            Peak resident memory in MiB, normalized for Linux and macOS units, or
            ``None`` when unavailable.
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
    def _safe_call(func: Callable[[], object | None] | None) -> object | None:
        """Call an optional provider without letting provider failures escape.

        Args:
            func: Zero-argument provider callback.

        Returns:
            Provider result, or ``None`` when the provider is missing or raises.
        """
        if func is None:
            return None
        with contextlib.suppress(Exception):
            return func()
        return None

    def _init_process_handle(self):
        """Initialize and prime the psutil process handle if available.

        Returns:
            object | None: A ``psutil.Process`` handle ready for sampling, or ``None`` when
            psutil is unavailable or initialization fails.
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
