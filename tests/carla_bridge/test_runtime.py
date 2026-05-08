"""CARLA-free tests for optional CARLA runtime guards."""

from __future__ import annotations

from types import ModuleType

import pytest


def test_require_carla_raises_clear_error_when_api_is_missing(monkeypatch) -> None:
    """Strict replay guards should fail clearly when CARLA is not importable."""
    from robot_sf_carla_bridge.availability import CarlaUnavailableError, require_carla

    def fake_import_module(name):
        """Simulate a missing CARLA module import."""
        if name == "carla":
            raise ModuleNotFoundError("No module named 'carla'", name="carla")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(CarlaUnavailableError, match="CARLA Python API package 'carla'"):
        require_carla()


def test_require_carla_returns_imported_api_module(monkeypatch) -> None:
    """Strict replay guards should return the importable CARLA API module."""
    from robot_sf_carla_bridge.availability import require_carla

    fake_carla = ModuleType("carla")

    def fake_import_module(name):
        """Return the fake CARLA module for strict import checks."""
        if name == "carla":
            return fake_carla
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    assert require_carla() is fake_carla
