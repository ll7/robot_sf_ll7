"""Optional OMPL geometric planner adapter for Robot SF static-route diagnostics.

This module provides a thin adapter that exposes OMPL geometric sampling-based
planners (RRTConnect, BITstar, RRTstar, InformedRRTstar) as drop-in static-route
diagnostic tools against a Robot SF ``MapDefinition``.

**Purpose (diagnostic only):**
Compare continuous-space sampling-based routes against the existing grid-based
global planner (A*/Theta*) on static maps, to quantify route-length differences,
detect grid-resolution artifacts, and assess OMPL's value for robot_sf_ll7.

**OMPL is an optional dependency** — importing this module when ``ompl`` is not
installed raises ``ImportError`` with a clear message. Do not add ``ompl`` to
``pyproject.toml`` core dependencies; it is only needed for offline diagnostics.

**Install once** (not in ``pyproject.toml``):
    uv pip install ompl  # ompl==2.0.1 from PyPI

**Usage:**
    >>> from robot_sf.nav.svg_map_parser import convert_map
    >>> from robot_sf.planner.ompl_geometric_adapter import OmplGeometricAdapter, OmplPlannerChoice
    >>> map_def = convert_map("maps/svg_maps/classic_bottleneck.svg")
    >>> adapter = OmplGeometricAdapter(map_def, planner=OmplPlannerChoice.BITSTAR)
    >>> result = adapter.plan(start=(20.0, 31.0), goal=(20.0, 8.0))
    >>> result.path_length_m
    23.0

**Known limitations (from feasibility spike, 2026-07-07):**
- ``ompl`` PyPI wheel (v2.0.1) is a nanobind-based binding and produces resource-leak
  warnings at interpreter exit (``nanobind: leaked N instances``). These are benign
  for offline diagnostics but reflect binding immaturity.
- ``STRRTstar`` (Space-Time RRT*) is **not** exposed in the PyPI wheel; see
  ``docs/context/issue_4797_ompl_strrtstar_assessment.md``.
- Performance on first call includes Python import overhead (~0.5 s for shapely).
- OMPL state validity checks are point-containment only (no robot radius inflation).
  For clearance-aware planning, either inflate obstacles or extend the checker.

**Scope (issue #4799):**
Assessment target: geometric-only planning for static Robot SF maps. This adapter
covers the shortlist from issue #4799: RRTConnect, BITstar, RRTstar, InformedRRTstar,
PRMstar. It does not support dynamic pedestrians, kinodynamic feasibility, or
mandatory pipeline integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition


class OmplPlannerChoice(StrEnum):
    """OMPL geometric planner shortlist from issue #4799.

    All members are available in the standard ``ompl`` PyPI wheel (v2.0.1).
    STRRTstar is intentionally excluded — it is missing from the wheel.
    """

    RRTCONNECT = "RRTConnect"
    """Fast feasible single-query planner. Bidirectional RRT."""
    BITSTAR = "BITstar"
    """Batch Informed Trees*: anytime optimizing, typically fastest to optimum."""
    RRTSTAR = "RRTstar"
    """Optimizing single-query planner. Slower convergence than BITstar."""
    INFORMED_RRTSTAR = "InformedRRTstar"
    """Informed RRT*: focuses sampling on prolate hyperspheroid after initial solution."""
    PRMSTAR = "PRMstar"
    """Probabilistic Roadmap*: good for repeated-query fixed-map routing."""


@dataclass(slots=True)
class OmplGeometricResult:
    """Result from an OMPL geometric plan call.

    Attributes:
        solved: True if a solution was found within the time budget.
        planner_name: Name of the planner that was used.
        planning_time_s: Wall-clock seconds spent in ``planner.solve()``.
        path_length_m: Euclidean path length in metres (0.0 if not solved).
        waypoints: Sequence of (x, y) world-coordinate waypoints.
        exact_solution: True when OMPL reports an exact (non-approximate) solution.
    """

    solved: bool
    planner_name: str
    planning_time_s: float
    path_length_m: float
    waypoints: list[tuple[float, float]]
    exact_solution: bool


@dataclass
class OmplGeometricConfig:
    """Configuration for OmplGeometricAdapter.

    Attributes:
        planner: Which OMPL planner to use from the shortlist.
        time_budget_s: Maximum seconds allowed for ``planner.solve()``.
        interpolate_waypoints: Number of waypoints to interpolate along the path.
            Set to 0 to skip interpolation and return the sparse sampled path.
        robot_radius_m: Optional obstacle inflation for validity checking.
            Defaults to 0.0 (point-robot). Inflation uses ``shapely.buffer``.
    """

    planner: OmplPlannerChoice = OmplPlannerChoice.BITSTAR
    time_budget_s: float = 5.0
    interpolate_waypoints: int = 50
    robot_radius_m: float = 0.0
    _obstacle_polygons: list[Polygon] = field(default_factory=list, repr=False)


def _build_obstacle_union(map_def: MapDefinition, robot_radius_m: float = 0.0):
    """Convert MapDefinition obstacles into a shapely union for validity checks.

    Args:
        map_def: Robot SF map definition.
        robot_radius_m: Buffer radius to inflate each obstacle polygon.

    Returns:
        shapely.geometry.base.BaseGeometry: Union of all obstacle polygons.
    """
    polys = []
    for obs in map_def.obstacles:
        # Use the obstacle's own polygon components (handles MultiPolygon and
        # representative-vertex fallbacks) rather than Polygon(obs.vertices),
        # which silently mis-constructs complex/holed obstacles.
        for poly in obs.iter_polygons():
            try:
                if poly.is_empty:
                    continue
                if robot_radius_m > 0.0:
                    poly = poly.buffer(robot_radius_m)
                polys.append(poly)
            except Exception:  # noqa: BLE001  # invalid geometry — skip
                continue
    # Include map bounds as boundary obstacles
    w, h = map_def.width, map_def.height
    margin = 0.05  # thin boundary wall
    bound_polys = [
        Polygon([(0, 0), (w, 0), (w, margin), (0, margin)]),
        Polygon([(0, h - margin), (w, h - margin), (w, h), (0, h)]),
        Polygon([(0, 0), (margin, 0), (margin, h), (0, h)]),
        Polygon([(w - margin, 0), (w, 0), (w, h), (w - margin, h)]),
    ]
    return unary_union(polys + bound_polys)


class OmplGeometricAdapter:
    """Thin adapter wrapping OMPL geometric planners for Robot SF static maps.

    Converts a ``MapDefinition`` into an OMPL 2D ``RealVectorStateSpace`` with a
    Shapely-based obstacle validity checker, then exposes a ``plan()`` interface
    consistent with the rest of the robot_sf planner family.

    **Raises ImportError** at construction time if ``ompl`` is not installed.

    Args:
        map_def: Robot SF map definition (from ``convert_map``).
        config: Optional adapter configuration. Defaults to BITstar, 5 s budget.
        planner: Convenience shorthand for ``config.planner``. Ignored when
            ``config`` is provided explicitly.

    Example::

        adapter = OmplGeometricAdapter(map_def, planner=OmplPlannerChoice.RRTCONNECT)
        result = adapter.plan(start=(5.0, 5.0), goal=(35.0, 35.0))
        if result.solved:
            print(result.path_length_m, result.planning_time_s)
    """

    def __init__(
        self,
        map_def: MapDefinition,
        config: OmplGeometricConfig | None = None,
        *,
        planner: OmplPlannerChoice = OmplPlannerChoice.BITSTAR,
    ) -> None:
        """Initialize the adapter with a map definition and optional config."""
        try:
            from ompl import base as ob  # noqa: PLC0415  # lazy import (optional dep)
            from ompl import geometric as og  # noqa: PLC0415

            self._ob = ob
            self._og = og
        except ImportError as exc:
            raise ImportError(
                "ompl is not installed. Install it with: uv pip install ompl\n"
                "Note: ompl is an optional diagnostic dependency; do not add it to "
                "pyproject.toml core dependencies."
            ) from exc

        self._config = config if config is not None else OmplGeometricConfig(planner=planner)
        self._map_def = map_def
        self._obstacle_union = _build_obstacle_union(map_def, self._config.robot_radius_m)
        self._setup_ompl_space()

    def _setup_ompl_space(self) -> None:
        """Build and cache the OMPL SpaceInformation for this map."""
        ob = self._ob
        w, h = self._map_def.width, self._map_def.height

        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, 0.0)
        bounds.setHigh(0, w)
        bounds.setLow(1, 0.0)
        bounds.setHigh(1, h)
        space.setBounds(bounds)

        self._space = space
        self._ss = self._og.SimpleSetup(space)

        obstacle_union = self._obstacle_union

        class _Checker(ob.StateValidityChecker):
            def isValid(self, state) -> bool:  # type: ignore[override]
                x, y = state[0], state[1]
                return not obstacle_union.contains(Point(x, y))

        self._checker = _Checker(self._ss.getSpaceInformation())
        self._ss.setStateValidityChecker(self._checker)

    def _make_planner(self):
        """Instantiate the chosen OMPL planner against the cached space info.

        Returns:
            OMPL planner instance configured for the current space information.
        """
        og = self._og
        si = self._ss.getSpaceInformation()
        mapping = {
            OmplPlannerChoice.RRTCONNECT: og.RRTConnect,
            OmplPlannerChoice.BITSTAR: og.BITstar,
            OmplPlannerChoice.RRTSTAR: og.RRTstar,
            OmplPlannerChoice.INFORMED_RRTSTAR: og.InformedRRTstar,
            OmplPlannerChoice.PRMSTAR: og.PRMstar,
        }
        cls = mapping[self._config.planner]
        return cls(si)

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> OmplGeometricResult:
        """Compute a route between ``start`` and ``goal`` using the configured planner.

        Args:
            start: World-coordinate (x, y) start position in metres.
            goal: World-coordinate (x, y) goal position in metres.

        Returns:
            OmplGeometricResult with solve status, timing, and waypoints.
        """
        self._ss.clear()

        start_state = self._space.allocState()
        start_state[0] = float(start[0])
        start_state[1] = float(start[1])

        goal_state = self._space.allocState()
        goal_state[0] = float(goal[0])
        goal_state[1] = float(goal[1])

        self._ss.setStartAndGoalStates(start_state, goal_state)
        self._ss.setPlanner(self._make_planner())

        t0 = time.perf_counter()
        status = self._ss.solve(self._config.time_budget_s)
        planning_time_s = time.perf_counter() - t0

        if not status:
            return OmplGeometricResult(
                solved=False,
                planner_name=self._config.planner.value,
                planning_time_s=planning_time_s,
                path_length_m=0.0,
                waypoints=[],
                exact_solution=False,
            )

        path = self._ss.getSolutionPath()
        if self._config.interpolate_waypoints > 0:
            path.interpolate(self._config.interpolate_waypoints)

        states = path.getStates()
        waypoints = [(s[0], s[1]) for s in states]

        if len(waypoints) > 1:
            pts = np.asarray(waypoints, dtype=np.float64)
            path_length_m = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        else:
            path_length_m = 0.0

        exact = bool(self._ss.haveExactSolutionPath())

        return OmplGeometricResult(
            solved=True,
            planner_name=self._config.planner.value,
            planning_time_s=planning_time_s,
            path_length_m=path_length_m,
            waypoints=waypoints,
            exact_solution=exact,
        )
