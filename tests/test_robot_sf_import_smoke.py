"""Smoke tests for `robot_sf` package import behavior."""

from __future__ import annotations

import sys


def test_robot_sf_import_does_not_mutate_sys_path():
    """Verify importing `robot_sf` does not modify `sys.path`."""
    before = list(sys.path)
    if "robot_sf" in sys.modules:
        del sys.modules["robot_sf"]

    import robot_sf

    assert list(sys.path) == before
    assert robot_sf.__all__ == [
        "ManifestWriter",
        "RunRegistry",
        "RunTrackerConfig",
        "generate_run_id",
    ]
