"""Tests for host-safe pytest worker resolution in scripts/dev."""

from __future__ import annotations

import pytest

from scripts.dev.resolve_pytest_workers import (
    LOW_CPU_THRESHOLD,
    MACOS_MAX_WORKERS,
    _cap_workers_for_host,
    _resolve_worker_spec,
)


def test_resolve_worker_spec_defaults_to_auto_on_linux() -> None:
    """Non-macOS hosts should keep xdist auto by default."""
    workers, reason = _resolve_worker_spec(requested=None, cpu_count=32, system="Linux")
    assert workers == "auto"
    assert "default xdist auto worker count" in reason


def test_resolve_worker_spec_caps_macos_auto_workers() -> None:
    """macOS defaults should use a bounded fixed worker count instead of xdist auto."""
    workers, reason = _resolve_worker_spec(requested=None, cpu_count=32, system="Darwin")
    assert workers == "8"
    assert "macOS-safe default" in reason


def test_resolve_worker_spec_applies_macos_floor() -> None:
    """macOS should apply the minimum worker floor on very small hosts."""
    workers, reason = _resolve_worker_spec(requested=None, cpu_count=2, system="Darwin")
    assert workers == "2"
    assert "macOS-safe default" in reason


def test_resolve_worker_spec_caps_explicit_override_on_macos() -> None:
    """Explicit overrides above the macOS cap should be capped, not blindly passed through."""
    workers, reason = _resolve_worker_spec(requested="14", cpu_count=32, system="Darwin")
    assert workers == "8"
    assert "explicit override" in reason
    assert "capped" in reason


def test_resolve_worker_spec_caps_low_cpu_linux_explicit() -> None:
    """Explicit overrides on low-CPU Linux (e.g. GitHub Actions 2-vCPU runner) should be capped."""
    workers, reason = _resolve_worker_spec(requested="32", cpu_count=2, system="Linux")
    assert workers == "16"
    assert "explicit override" in reason
    assert "low-CPU" in reason


def test_resolve_worker_spec_allows_high_cpu_linux_explicit() -> None:
    """High-CPU Linux hosts (>=8 CPUs) should not cap explicit overrides."""
    workers, reason = _resolve_worker_spec(requested="32", cpu_count=32, system="Linux")
    assert workers == "32"
    assert reason == "explicit override via PYTEST_NUM_WORKERS"


def test_resolve_worker_spec_no_cap_at_threshold() -> None:
    """A host exactly at LOW_CPU_THRESHOLD should not trigger the low-CPU cap."""
    workers, reason = _resolve_worker_spec(
        requested="32", cpu_count=LOW_CPU_THRESHOLD, system="Linux"
    )
    assert workers == "32"
    assert reason == "explicit override via PYTEST_NUM_WORKERS"


def test_resolve_worker_spec_caps_just_below_threshold() -> None:
    """A host one below LOW_CPU_THRESHOLD should cap explicit overrides."""
    workers, reason = _resolve_worker_spec(
        requested="32", cpu_count=LOW_CPU_THRESHOLD - 1, system="Linux"
    )
    assert workers == "16"
    assert "low-CPU" in reason


def test_resolve_worker_spec_rejects_non_positive_counts() -> None:
    """Invalid worker overrides should fail loudly instead of silently falling back."""
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_worker_spec(requested="0", cpu_count=32, system="Darwin")


def test_resolve_worker_spec_rejects_non_numeric_counts() -> None:
    """Non-numeric worker overrides should raise the same user-facing error."""
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_worker_spec(requested="foo", cpu_count=32, system="Darwin")


def test_resolve_worker_spec_honors_auto_override() -> None:
    """An explicit auto override should bypass all caps."""
    workers, reason = _resolve_worker_spec(requested="auto", cpu_count=2, system="Linux")
    assert workers == "auto"
    assert reason == "explicit override via PYTEST_NUM_WORKERS=auto"


def test_cap_workers_for_host_no_change_when_within_limit() -> None:
    """_cap_workers_for_host returns unchanged value when requested is within cap."""
    capped, reason = _cap_workers_for_host(requested=8, cpu_count=2, system="Linux")
    assert capped == 8
    assert reason == ""


def test_cap_workers_macos_low_cpu_uses_macos_cap_not_low_cpu_cap() -> None:
    """A low-CPU macOS host must be governed by the macOS cap, never the low-CPU cap.

    Regression guard for the platform-isolation early return: even with a CPU
    count below LOW_CPU_THRESHOLD, macOS overrides must collapse to
    MACOS_MAX_WORKERS (8) rather than LOW_CPU_WORKER_CAP (16).
    """
    capped, reason = _cap_workers_for_host(requested=12, cpu_count=2, system="Darwin")
    assert capped == MACOS_MAX_WORKERS
    assert "macOS" in reason
    assert "low-CPU" not in reason
