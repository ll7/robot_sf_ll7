"""Focused checks for the optional map plotting seam."""

from __future__ import annotations

import pytest

from robot_sf.nav.map_config import MapDefinition


def test_plot_map_obstacles_loads_matplotlib_only_on_first_plot() -> None:
    """Plotting remains usable while canonical map imports stay lightweight."""

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pyplot = pytest.importorskip("matplotlib.pyplot")

    map_definition = object.__new__(MapDefinition)
    map_definition.obstacles = []
    figure, axis = pyplot.subplots()
    try:
        map_definition.plot_map_obstacles(axis)
        assert len(axis.patches) == 0
    finally:
        pyplot.close(figure)
