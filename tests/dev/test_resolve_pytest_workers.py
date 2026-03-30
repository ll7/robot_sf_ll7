"""Tests for macOS-safe pytest worker resolution in scripts/dev."""

from __future__ import annotations

import pytest

from scripts.dev.resolve_pytest_workers import _resolve_worker_spec


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


def test_resolve_worker_spec_honors_explicit_override() -> None:
    """Explicit worker overrides should win over host-derived defaults."""
    workers, reason = _resolve_worker_spec(requested="14", cpu_count=32, system="Darwin")
    assert workers == "14"
    assert "explicit override" in reason


def test_resolve_worker_spec_rejects_non_positive_counts() -> None:
    """Invalid worker overrides should fail loudly instead of silently falling back."""
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_worker_spec(requested="0", cpu_count=32, system="Darwin")


def test_resolve_worker_spec_rejects_non_numeric_counts() -> None:
    """Non-numeric worker overrides should raise the same user-facing error."""
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_worker_spec(requested="foo", cpu_count=32, system="Darwin")
