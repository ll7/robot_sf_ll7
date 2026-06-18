"""Tests for GPU telemetry collection and serialization."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import robot_sf
from robot_sf.telemetry import gpu as gpu_module
from robot_sf.telemetry.gpu import GpuDeviceSample, GpuSample
from robot_sf.telemetry.models import serialize_payload
from robot_sf.telemetry.sampler import TelemetrySampler


class _Writer:
    """Capture emitted telemetry snapshots."""

    def __init__(self) -> None:
        """Initialize empty snapshot storage."""
        self.snapshots = []

    def append_telemetry_snapshot(self, snapshot) -> None:
        """Record one telemetry snapshot."""
        self.snapshots.append(snapshot)


class _FakeNvmlError(Exception):
    """Fake NVML error type."""


class _FakeNvml:
    """Small NVML stand-in with two visible devices."""

    NVMLError = _FakeNvmlError

    def __init__(self) -> None:
        """Initialize fake device metrics."""
        mb = 1024**2
        self.devices = (
            SimpleNamespace(util=10.0, used=2 * mb, total=8 * mb),
            SimpleNamespace(util=70.0, used=4 * mb, total=12 * mb),
        )

    def nvmlInit(self) -> None:
        """Pretend NVML initialized successfully."""

    def nvmlShutdown(self) -> None:
        """Pretend NVML shut down successfully."""

    def nvmlDeviceGetCount(self) -> int:
        """Return fake device count."""
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        """Use the index itself as the fake handle."""
        return index

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        """Return fake utilization for a handle."""
        return SimpleNamespace(gpu=self.devices[handle].util)

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        """Return fake memory info for a handle."""
        device = self.devices[handle]
        return SimpleNamespace(used=device.used, total=device.total)


def test_collect_gpu_sample_includes_per_device_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVML collection keeps aggregate fields and exposes individual devices."""
    monkeypatch.setattr(gpu_module, "_NVML_ERROR", None)
    monkeypatch.setattr(gpu_module, "pynvml", _FakeNvml())
    monkeypatch.setattr(gpu_module, "_NVML_HANDLE", gpu_module._NvmlHandle())

    sample = gpu_module.collect_gpu_sample()

    assert sample is not None
    assert sample.util_percent == pytest.approx(40.0)
    assert sample.memory_used_mb == pytest.approx(6.0)
    assert sample.memory_total_mb == pytest.approx(20.0)
    assert sample.devices == (
        GpuDeviceSample(index=0, util_percent=10.0, memory_used_mb=2.0, memory_total_mb=8.0),
        GpuDeviceSample(index=1, util_percent=70.0, memory_used_mb=4.0, memory_total_mb=12.0),
    )


def test_sampler_serializes_per_device_gpu_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry snapshots include a JSON-ready per-device breakdown."""
    assert robot_sf.telemetry.sampler.collect_gpu_sample is not None
    sample = GpuSample(
        util_percent=40.0,
        memory_used_mb=6.0,
        memory_total_mb=20.0,
        devices=(
            GpuDeviceSample(index=0, util_percent=10.0, memory_used_mb=2.0, memory_total_mb=8.0),
            GpuDeviceSample(index=1, util_percent=70.0, memory_used_mb=4.0, memory_total_mb=12.0),
        ),
    )
    monkeypatch.setattr("robot_sf.telemetry.sampler.collect_gpu_sample", lambda: sample)

    writer = _Writer()
    sampler = TelemetrySampler(
        writer=writer,
        progress_tracker=None,
        started_at=datetime.now(UTC),
        step_rate_provider=lambda: None,
    )

    snapshot = sampler.emit_snapshot()
    payload = serialize_payload(snapshot)

    assert writer.snapshots == [snapshot]
    assert snapshot.gpu_util_percent == pytest.approx(40.0)
    assert snapshot.gpu_mem_used_mb == pytest.approx(6.0)
    assert snapshot.gpu_devices == sample.devices
    assert payload["gpu_devices"] == [
        {
            "index": 0,
            "util_percent": 10.0,
            "memory_used_mb": 2.0,
            "memory_total_mb": 8.0,
        },
        {
            "index": 1,
            "util_percent": 70.0,
            "memory_used_mb": 4.0,
            "memory_total_mb": 12.0,
        },
    ]
