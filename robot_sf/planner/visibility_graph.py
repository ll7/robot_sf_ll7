"""Visibility graph wrapper using pyvisgraph and NetworkX."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import networkx as nx
import pyvisgraph as vg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely.geometry import Polygon


class VisibilityGraph:
    """Construct and query visibility graphs for polygonal maps."""

    _graph_cache: dict[str, VisibilityGraph] = {}

    def __init__(self) -> None:
        """Initialize an empty visibility graph."""
        self._vg = vg.VisGraph()
        self._nx_graph: nx.Graph | None = None
        self._built = False

    def build(self, polygons: Sequence[Polygon], workers: int = 1) -> None:
        """Build the visibility graph from buffered obstacle polygons.

        Args:
            polygons: Sequence of buffered Shapely polygons.
            workers: Number of worker threads to pass to pyvisgraph.
        """
        vg_polygons: list[list[vg.Point]] = []
        for poly in polygons:
            if poly.is_empty or poly.area <= 0:
                continue
            coords = list(poly.exterior.coords)
            if coords and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            vg_polygons.append([vg.Point(x, y) for x, y in coords])

        self._vg.build(vg_polygons, workers=workers)
        self._built = True
        self._nx_graph = self._build_networkx_graph()

    def shortest_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Return shortest path waypoints.

        Args:
            start: Start point as (x, y).
            goal: Goal point as (x, y).

        Returns:
            List of waypoints including start and goal.
        """
        if not self._built:
            raise RuntimeError("VisibilityGraph must be built before querying.")

        start_pt = vg.Point(*start)
        goal_pt = vg.Point(*goal)
        points = self._vg.shortest_path(start_pt, goal_pt)
        if not points:
            return []
        return [(p.x, p.y) for p in points]

    def add_waypoint(self, waypoint: tuple[float, float]) -> None:
        """Add a waypoint as a permanent vertex in the visibility graph.

        This ensures the waypoint becomes part of the routing graph, allowing
        shortest_path() to properly route through intermediate waypoints.

        Args:
            waypoint: Waypoint coordinates as (x, y).
        """
        if not self._built:
            raise RuntimeError("VisibilityGraph must be built before adding waypoints.")

        pt = vg.Point(*waypoint)
        # Add the point to the visibility graph permanently
        self._vg.update([pt])

        # Rebuild the NetworkX graph to reflect the new connectivity
        self._nx_graph = self._build_networkx_graph()

    def _build_networkx_graph(self) -> nx.Graph:
        """Create a NetworkX view of the internal graph for diagnostics.

        Returns:
            NetworkX graph mirroring the pyvisgraph edges with distances as weights.
        """
        graph = nx.Graph()
        if not self._vg.visgraph:
            return graph
        for edge in self._vg.visgraph.get_edges():
            p1 = (edge.p1.x, edge.p1.y)
            p2 = (edge.p2.x, edge.p2.y)
            weight = math.hypot(edge.p1.x - edge.p2.x, edge.p1.y - edge.p2.y)
            graph.add_edge(p1, p2, weight=weight)
        return graph

    @property
    def networkx_graph(self) -> nx.Graph | None:
        """Expose a NetworkX graph for testing and benchmarking."""
        return self._nx_graph

    @classmethod
    def _hash_polygons(cls, polygons: Sequence[Polygon]) -> str:
        """Create a stable hash for polygon coordinates.

        Returns:
            MD5 hex digest representing polygon outlines.
        """

        parts = []
        for poly in polygons:
            coords = list(poly.exterior.coords)
            parts.append(";".join(f"{x:.4f},{y:.4f}" for x, y in coords))
        digest = hashlib.md5("|".join(parts).encode()).hexdigest()
        return digest

    @classmethod
    def get_cached(cls, polygons: Sequence[Polygon], workers: int = 1) -> VisibilityGraph:
        """Return a cached VisibilityGraph if available, else build and store."""
        key = cls._hash_polygons(polygons)
        if key in cls._graph_cache:
            return cls._graph_cache[key]
        graph = cls()
        graph.build(polygons, workers=workers)
        cls._graph_cache[key] = graph
        return graph

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the visibility graph cache."""
        cls._graph_cache.clear()
