"""Contract coverage for lazy Robot SF package exports."""

from __future__ import annotations

import pytest


def test_root_package_resolves_existing_telemetry_exports_lazily() -> None:
    """The root package retains its telemetry convenience exports on demand."""
    import robot_sf

    robot_sf.__dict__.pop("RunTrackerConfig", None)
    robot_sf.__dict__.pop("telemetry", None)

    assert robot_sf.RunTrackerConfig.__module__ == "robot_sf.telemetry.config"
    assert robot_sf.telemetry.RunTrackerConfig is robot_sf.RunTrackerConfig
    assert "ManifestWriter" in dir(robot_sf)
    with pytest.raises(AttributeError, match="has no attribute 'missing_export'"):
        _ = robot_sf.missing_export


def test_telemetry_package_resolves_submodules_and_attribute_errors_lazily() -> None:
    """Telemetry discovery preserves lazy submodule and error behavior."""
    from robot_sf import telemetry

    telemetry.__dict__.pop("models", None)

    assert telemetry.models.__name__ == "robot_sf.telemetry.models"
    assert "models" in dir(telemetry)
    with pytest.raises(AttributeError, match="has no attribute 'missing_export'"):
        _ = telemetry.missing_export
