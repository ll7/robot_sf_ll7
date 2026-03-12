"""Tests for macOS-safe pytest worker resolution in scripts/dev."""

from __future__ import annotations

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


def test_resolve_worker_spec_honors_explicit_override() -> None:
    """Explicit worker overrides should win over host-derived defaults."""
    workers, reason = _resolve_worker_spec(requested="14", cpu_count=32, system="Darwin")
    assert workers == "14"
    assert "explicit override" in reason


def test_resolve_worker_spec_rejects_non_positive_counts() -> None:
    """Invalid worker overrides should fail loudly instead of silently falling back."""
    try:
        _resolve_worker_spec(requested="0", cpu_count=32, system="Darwin")
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive worker override")
