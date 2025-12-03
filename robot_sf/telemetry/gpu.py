"""Optional GPU telemetry helpers backed by NVIDIA's NVML bindings."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - exercised in integration tests when NVML is present
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - NVML is optional
    pynvml = None  # type: ignore
    _NVML_ERROR = "pynvml not installed"
else:  # pragma: no cover - initialization happens lazily when NVML is present
    _NVML_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - hints only
    from collections.abc import Callable
else:  # pragma: no cover - runtime fallback since annotations are postponed
    Callable = Any  # type: ignore[assignment]


@dataclass(slots=True)
class GpuSample:
    """Snapshot of aggregate GPU utilization across all visible devices."""

    util_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    notes: str | None = None


class _NvmlHandle:
    """Lazy NVML initializer so callers do not pay the cost unless needed."""

    def __init__(self) -> None:
        """TODO docstring. Document this function."""
        self._initialized = False
        self._failed_reason: str | None = _NVML_ERROR

    def ensure_initialized(self) -> bool:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        if pynvml is None:
            return False
        if self._initialized:
            return True
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:  # pragma: no cover - NVML failures are environment specific
            self._failed_reason = str(exc)
            return False
        self._initialized = True
        return True

    def shutdown(self) -> None:
        """TODO docstring. Document this function."""
        if not self._initialized or pynvml is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover - teardown best-effort only
            pynvml.nvmlShutdown()
        self._initialized = False

    @property
    def failed_reason(self) -> str | None:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self._failed_reason


_NVML_HANDLE = _NvmlHandle()


def gpu_support_reason() -> str | None:
    """Return ``None`` if NVML telemetry is available, otherwise the failure reason."""

    if pynvml is None:
        return _NVML_ERROR or "pynvml not installed"
    if not _NVML_HANDLE.ensure_initialized():
        return _NVML_HANDLE.failed_reason or "nvmlInit() failed"
    return None


def collect_gpu_sample() -> GpuSample | None:
    """Collect aggregate GPU utilization metrics if NVML is available.

    Returns:
        GpuSample | None: Aggregated GPU metrics across devices, or None when
        NVML support is unavailable.
    """

    if pynvml is None:
        return None
    if not _NVML_HANDLE.ensure_initialized():
        return GpuSample(notes=_NVML_HANDLE.failed_reason or "nvml-init-failed")
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as exc:  # pragma: no cover - NVML failures are environment specific
        return GpuSample(notes=str(exc))
    if device_count == 0:
        return GpuSample(notes="no-gpus")
    total_util = 0.0
    total_memory_used = 0.0
    total_memory = 0.0
    for index in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_util += float(util.gpu)
        total_memory_used += float(mem.used)
        total_memory += float(mem.total)
    avg_util = total_util / max(device_count, 1)
    used_mb = total_memory_used / (1024**2)
    total_mb = total_memory / (1024**2)
    return GpuSample(util_percent=avg_util, memory_used_mb=used_mb, memory_total_mb=total_mb)


def shutdown_gpu_telemetry() -> None:
    """Release NVML resources (best-effort)."""

    _NVML_HANDLE.shutdown()


def with_gpu_support(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to run a callable only when GPU telemetry is available.

    Returns:
        Callable[..., Any]: A wrapper that returns None when GPU telemetry is
        unavailable, otherwise forwards to the wrapped callable.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """TODO docstring. Document this function.

        Args:
            args: TODO docstring.
            kwargs: TODO docstring.

        Returns:
            TODO docstring.
        """
        if gpu_support_reason() is not None:
            return None
        return func(*args, **kwargs)

    return wrapper
