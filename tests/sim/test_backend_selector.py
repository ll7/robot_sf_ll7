"""Tests for the performance-aware backend selector."""

from __future__ import annotations

import pytest

from robot_sf.sim import registry


def _stub_factory(*_: object, **__: object) -> None:  # pragma: no cover - helper
    """No-op backend factory used to register stub backends in tests.

    Args:
        _: Positional arguments supplied by the registry, ignored.
        __: Keyword arguments supplied by the registry, ignored.
    """
    return None


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate registry mutations per test."""

    monkeypatch.setattr(registry, "_REGISTRY", {})


def test_select_best_backend_prefers_explicit_choice() -> None:
    """An explicitly preferred backend name overrides fastest-backend selection."""
    registry.register_backend("fast-pysf", _stub_factory)
    registry.register_backend("dummy", _stub_factory)

    assert registry.select_best_backend(preferred="dummy") == "dummy"


def test_select_best_backend_respects_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROBOT_SF_BACKEND selects the named backend when no preference is passed.

    Args:
        monkeypatch: Pytest fixture used to set the environment variable.
    """
    registry.register_backend("fast-pysf", _stub_factory)
    registry.register_backend("dummy", _stub_factory)
    monkeypatch.setenv("ROBOT_SF_BACKEND", "dummy")

    assert registry.select_best_backend() == "dummy"


def test_select_best_backend_falls_back_to_fastest() -> None:
    """Without preference or environment override, selection returns the fastest backend."""
    registry.register_backend("dummy", _stub_factory)
    registry.register_backend("fast-pysf", _stub_factory)
    registry.register_backend("experimental", _stub_factory)

    assert registry.select_best_backend() == "fast-pysf"


def test_select_best_backend_errors_when_empty() -> None:
    """Selecting a backend when none are registered raises RuntimeError."""
    with pytest.raises(RuntimeError):
        registry.select_best_backend()
