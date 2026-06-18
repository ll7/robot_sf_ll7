"""Smoke tests for `robot_sf` package import behavior."""

from __future__ import annotations

import sys


def test_robot_sf_import_does_not_mutate_sys_path():
    """Verify importing `robot_sf` does not modify `sys.path`."""
    before = list(sys.path)
    saved_robot_sf_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "robot_sf" or name.startswith("robot_sf.")
    }
    for name in saved_robot_sf_modules:
        del sys.modules[name]

    try:
        import robot_sf

        assert list(sys.path) == before
        assert robot_sf.__all__ == [
            "ManifestWriter",
            "RunRegistry",
            "RunTrackerConfig",
            "generate_run_id",
            "telemetry",
        ]
    finally:
        for name in list(sys.modules):
            if name == "robot_sf" or name.startswith("robot_sf."):
                del sys.modules[name]
        sys.modules.update(saved_robot_sf_modules)
