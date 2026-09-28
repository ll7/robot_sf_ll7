"""Focused checks for the optional map plotting seam."""

from __future__ import annotations

import sys
import types

import pytest

from robot_sf.nav.map_config import MapDefinition


def test_plot_map_obstacles_loads_matplotlib_only_on_first_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plotting remains usable while canonical map imports stay lightweight."""

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_axes = types.ModuleType("matplotlib.axes")
    fake_patches = types.ModuleType("matplotlib.patches")
    fake_path = types.ModuleType("matplotlib.path")

    class FakeAxes:
        pass

    fake_axes.Axes = FakeAxes
    fake_matplotlib.axes = fake_axes
    fake_patches.PathPatch = object

    class FakePath:
        MOVETO = 1
        LINETO = 2
        CLOSEPOLY = 79

    fake_path.Path = FakePath
    for name, module in {
        "matplotlib": fake_matplotlib,
        "matplotlib.axes": fake_axes,
        "matplotlib.patches": fake_patches,
        "matplotlib.path": fake_path,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    map_definition = object.__new__(MapDefinition)
    map_definition.obstacles = []
    map_definition.plot_map_obstacles(FakeAxes())
