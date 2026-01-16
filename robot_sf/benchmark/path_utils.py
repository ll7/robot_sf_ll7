"""Helpers for map-aware shortest path calculations used in metrics."""

from __future__ import annotations

from itertools import pairwise
from math import dist
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.planner.classic_global_planner import (
    ClassicGlobalPlanner,
    ClassicPlannerConfig,
    PlanningError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.nav.map_config import MapDefinition


def _path_length(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return float("nan")
    return float(sum(dist(a, b) for a, b in pairwise(pts)))


def compute_shortest_path_length(
    map_def: MapDefinition | None,
    start: np.ndarray,
    goal: np.ndarray,
) -> float:
    """Return shortest path length using the classic Theta* global planner.

    Returns NaN when map definition is missing or planning fails.
    """
    if map_def is None:
        return float("nan")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        return float("nan")
    planner = ClassicGlobalPlanner(map_def, config=ClassicPlannerConfig())
    try:
        waypoints, _info = planner.plan(
            (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
        )
    except (PlanningError, ValueError):
        return float("nan")
    return _path_length(waypoints)


__all__ = ["compute_shortest_path_length"]
