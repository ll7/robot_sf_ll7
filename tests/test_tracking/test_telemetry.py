"""Tests for telemetry sampler fallbacks and recommendation rules."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from robot_sf.telemetry import sampler as sampler_module
from robot_sf.telemetry.models import RecommendationSeverity, TelemetrySnapshot
from robot_sf.telemetry.recommendations import RecommendationEngine, RecommendationRules
from robot_sf.telemetry.sampler import TelemetrySampler


class _DummyWriter:
    def __init__(self) -> None:
        self.snapshots: list[TelemetrySnapshot] = []

    def append_telemetry_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        self.snapshots.append(snapshot)


def test_sampler_emits_fallback_notes_when_psutil_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _DummyWriter()
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=4096.0),
    )
    monkeypatch.setattr(sampler_module, "psutil", None)
    monkeypatch.setattr(sampler_module, "_PSUTIL_ERRORS", (OSError,))
    monkeypatch.setattr(sampler_module, "resource", fake_resource)
    monkeypatch.setattr(sampler_module.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(sampler_module, "collect_gpu_sample", lambda: None)

    sampler = TelemetrySampler(
        writer,
        progress_tracker=None,
        started_at=datetime.now(UTC),
        interval_seconds=0.1,
        time_provider=lambda: datetime.now(UTC),
        step_rate_provider=lambda: None,
    )

    snapshot = sampler.emit_snapshot()

    assert writer.snapshots == [snapshot]
    assert snapshot.cpu_percent_process is None
    assert snapshot.cpu_percent_system is None
    assert snapshot.memory_rss_mb == pytest.approx(4.0)
    assert snapshot.notes is not None
    assert "psutil-unavailable" in snapshot.notes
    assert "system-cpu-unavailable" in snapshot.notes


def test_recommendation_engine_triggers_all_rules() -> None:
    engine = RecommendationEngine(
        rules=RecommendationRules(
            throughput_baseline=10.0,
            throughput_warning_ratio=0.8,
            throughput_critical_ratio=0.5,
            cpu_process_warning=70.0,
            cpu_process_critical=90.0,
            cpu_system_warning=80.0,
        ),
        total_memory_mb=10000.0,
    )

    snapshots = [
        TelemetrySnapshot(
            timestamp_ms=1,
            steps_per_sec=2.0,
            cpu_percent_process=96.0,
            cpu_percent_system=88.0,
            memory_rss_mb=9500.0,
            gpu_util_percent=0.0,
        ),
        TelemetrySnapshot(
            timestamp_ms=2,
            steps_per_sec=3.0,
            cpu_percent_process=97.0,
            cpu_percent_system=92.0,
            memory_rss_mb=9800.0,
            gpu_util_percent=1.0,
        ),
    ]

    for snapshot in snapshots:
        engine.observe_snapshot(snapshot)

    recommendations = engine.generate_recommendations()
    triggers = {recommendation.trigger: recommendation for recommendation in recommendations}

    assert {
        "low_throughput",
        "cpu_saturation",
        "memory_pressure",
        "gpu_idle",
    } <= triggers.keys()
    assert triggers["low_throughput"].severity is RecommendationSeverity.CRITICAL
    assert triggers["cpu_saturation"].severity is RecommendationSeverity.CRITICAL
    assert triggers["memory_pressure"].severity is RecommendationSeverity.CRITICAL
    assert triggers["gpu_idle"].severity is RecommendationSeverity.INFO
