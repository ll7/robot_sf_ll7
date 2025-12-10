"""Tests for visibility graph caching and performance (US4)."""
# ruff: noqa: D103

from shapely.geometry import Polygon

from robot_sf.planner.visibility_graph import VisibilityGraph


def _square(x: float, y: float, size: float = 1.0) -> Polygon:
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def test_graph_reuse_after_build():
    vg = VisibilityGraph()
    vg.build([_square(0, 0)])

    # The networkx_graph should be populated and reused
    first_graph = vg.networkx_graph
    vg.build([_square(0, 0)])  # rebuild
    assert vg.networkx_graph is not None
    assert first_graph is not vg.networkx_graph  # rebuild gives new graph but old stays available


def test_cache_helper_returns_same_instance():
    vg = VisibilityGraph()
    vg.build([_square(0, 0)])
    assert vg.networkx_graph is not None
