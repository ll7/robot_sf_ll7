"""Tests for the performance-aware backend selector."""

from __future__ import annotations

import pytest

from robot_sf.sim import registry


def _stub_factory(*_: object, **__: object) -> None:  # pragma: no cover - helper
    """Stub factory.

    Args:
        _: Auto-generated placeholder description.
        __: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    return None


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate registry mutations per test."""

    monkeypatch.setattr(registry, "_REGISTRY", {})


def test_select_best_backend_prefers_explicit_choice() -> None:
    """Test select best backend prefers explicit choice.

    Returns:
        None: Auto-generated placeholder description.
    """
    registry.register_backend("fast-pysf", _stub_factory)
    registry.register_backend("dummy", _stub_factory)

    assert registry.select_best_backend(preferred="dummy") == "dummy"


def test_select_best_backend_respects_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test select best backend respects environment.

    Args:
        monkeypatch: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    registry.register_backend("fast-pysf", _stub_factory)
    registry.register_backend("dummy", _stub_factory)
    monkeypatch.setenv("ROBOT_SF_BACKEND", "dummy")

    assert registry.select_best_backend() == "dummy"


def test_select_best_backend_falls_back_to_fastest() -> None:
    """Test select best backend falls back to fastest.

    Returns:
        None: Auto-generated placeholder description.
    """
    registry.register_backend("dummy", _stub_factory)
    registry.register_backend("fast-pysf", _stub_factory)
    registry.register_backend("experimental", _stub_factory)

    assert registry.select_best_backend() == "fast-pysf"


def test_select_best_backend_errors_when_empty() -> None:
    """Test select best backend errors when empty.

    Returns:
        None: Auto-generated placeholder description.
    """
    with pytest.raises(RuntimeError):
        registry.select_best_backend()
