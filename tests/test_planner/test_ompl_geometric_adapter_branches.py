"""Branch-coverage tests for the OMPL geometric planner adapter (issue #6371).

These tests lock the adapter's planning-outcome contract using tiny in-memory
``MapDefinition`` geometry and mocked OMPL bindings. They do **not** require the
real ``ompl`` package and do **not** load any repository SVG maps, so they run
on every machine in the exact-green CI lane.

Covered outcomes:
- Every ``OmplPlannerChoice`` routes to the matching OMPL planner class.
- State-validity checker rejects obstacle interiors and map-boundary walls.
- ``RealVectorStateSpace`` bounds match the map width/height.
- ``plan`` configures start/goal states and forwards the time budget.
- Solved plans convert OMPL path states to waypoints and compute path length.
- The ``interpolate_waypoints`` toggle enables/skips path interpolation.
- Unsolved plans return a stable ``solved=False`` result.
- Missing ``ompl`` raises ``ImportError`` with install guidance.
- Planning exceptions currently propagate (locked contract; see test docstring).
"""

from __future__ import annotations

import sys
import types

import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.ompl_geometric_adapter import (
    OmplGeometricAdapter,
    OmplGeometricConfig,
    OmplPlannerChoice,
)

# ---------------------------------------------------------------------------
# Mock OMPL bindings
#
# The fake OMPL classes are defined at module scope and read the active
# ``_active_recorder`` global, which each ``_FakeOmpl.install()`` points at the
# per-test recorder via ``monkeypatch``. This keeps the mock factory functions
# simple (low cyclomatic complexity) while still giving every test an isolated
# recorder that captures bounds, start/goal, solve budget, planner routing, and
# path interpolation. Tests run sequentially within a process, so a single
# module-global recorder is safe and is always restored by ``monkeypatch``.
# ---------------------------------------------------------------------------


class _OmplState:
    """Indexable stand-in for an OMPL RealVector state.

    OMPL states expose ``state[0]`` / ``state[1]`` coordinate access; this double
    mirrors that protocol so the adapter's start/goal assignment and waypoint
    conversion exercise the same indexing path as the real binding.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.coords: list[float] = [float(x), float(y)]

    def __getitem__(self, index: int) -> float:
        return self.coords[index]

    def __setitem__(self, index: int, value: float) -> None:
        self.coords[index] = float(value)


class _FakeBounds:
    """Stand-in for ``ompl.base.RealVectorBounds`` recording low/high calls."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.lows: dict[int, float] = {}
        self.highs: dict[int, float] = {}

    def setLow(self, index: int, value: float) -> None:
        self.lows[index] = value

    def setHigh(self, index: int, value: float) -> None:
        self.highs[index] = value


class _FakeStateSpace:
    """Stand-in for ``ompl.base.RealVectorStateSpace``."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.bounds: _FakeBounds | None = None

    def setBounds(self, bounds: _FakeBounds) -> None:
        self.bounds = bounds
        _active_recorder.bounds = bounds

    def allocState(self) -> _OmplState:
        return _OmplState()


class _FakeValidityChecker:
    """Base class mirrored after ``ompl.base.StateValidityChecker``."""

    def __init__(self, space_info: object) -> None:
        self.space_info = space_info

    def isValid(self, state: object) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError


class _FakePath:
    """Stand-in for an OMPL solution path capturing interpolation calls."""

    def __init__(self, states: list[_OmplState]) -> None:
        self._states = list(states)
        self.interpolate_calls: list[int] = []

    def interpolate(self, count: int) -> None:
        self.interpolate_calls.append(count)

    def getStates(self) -> list[_OmplState]:
        return list(self._states)


class _FakeSimpleSetup:
    """Stand-in for ``ompl.geometric.SimpleSetup`` recording all plan inputs."""

    def __init__(self, space: _FakeStateSpace) -> None:
        _active_recorder.space = space
        _active_recorder.setup = self
        self._space_info = object()
        self.cleared = False
        self.start_state: object | None = None
        self.goal_state: object | None = None
        self.planner: object | None = None
        self.solve_budget: object | None = None
        self.checker: object | None = None
        self.solution_path = _FakePath(_active_recorder.solution_states)

    def getSpaceInformation(self) -> object:
        return self._space_info

    def setStateValidityChecker(self, checker: object) -> None:
        self.checker = checker

    def clear(self) -> None:
        self.cleared = True

    def setStartAndGoalStates(self, start: object, goal: object) -> None:
        self.start_state = start
        self.goal_state = goal

    def setPlanner(self, planner: object) -> None:
        self.planner = planner

    def solve(self, budget: object) -> object:
        self.solve_budget = budget
        if _active_recorder.solve_exc is not None:
            raise _active_recorder.solve_exc
        return _active_recorder.solve_result

    def getSolutionPath(self) -> _FakePath:
        return self.solution_path

    def haveExactSolutionPath(self) -> bool:
        return _active_recorder.exact_solution


def _planner_class(name: str) -> type:
    """Build a distinct OMPL planner double that records its name on use."""

    class _Planner:
        def __init__(self, space_info: object) -> None:
            self.space_info = space_info
            _active_recorder.planner_log.append(name)

    _Planner.__name__ = name
    return _Planner


class _FakeOmpl:
    """Per-test recorder and installer for mocked OMPL bindings.

    Attributes:
        solve_result: Value returned by the fake ``SimpleSetup.solve`` (truthy =
            solved).
        solve_exc: Optional exception raised by ``solve`` to exercise failure.
        solution_states: States the fake solution path exposes.
        exact_solution: Value returned by ``haveExactSolutionPath``.
        planner_log: Ordered list of OMPL planner class names instantiated.
        space/bounds/setup: References captured from the constructed OMPL stack.
    """

    def __init__(
        self,
        *,
        solve_result: object = True,
        solve_exc: BaseException | None = None,
        solution_states: list[_OmplState] | None = None,
        exact_solution: bool = True,
    ) -> None:
        self.solve_result = solve_result
        self.solve_exc = solve_exc
        self.solution_states = (
            solution_states if solution_states is not None else [_OmplState(0.0, 0.0)]
        )
        self.exact_solution = exact_solution
        self.planner_log: list[str] = []
        self.bounds: _FakeBounds | None = None
        self.space: _FakeStateSpace | None = None
        self.setup: _FakeSimpleSetup | None = None
        self.ob = types.ModuleType("ompl.base")
        self.ob.RealVectorBounds = _FakeBounds
        self.ob.RealVectorStateSpace = _FakeStateSpace
        self.ob.StateValidityChecker = _FakeValidityChecker
        self.og = types.ModuleType("ompl.geometric")
        self.og.SimpleSetup = _FakeSimpleSetup
        for planner_name in ("RRTConnect", "BITstar", "RRTstar", "InformedRRTstar", "PRMstar"):
            setattr(self.og, planner_name, _planner_class(planner_name))
        self.ompl_pkg = types.ModuleType("ompl")
        self.ompl_pkg.__path__ = []  # type: ignore[attr-defined]
        self.ompl_pkg.base = self.ob  # type: ignore[attr-defined]
        self.ompl_pkg.geometric = self.og  # type: ignore[attr-defined]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _FakeOmpl:
        """Register the fakes in ``sys.modules`` and as the active recorder."""
        monkeypatch.setitem(sys.modules, "ompl", self.ompl_pkg)
        monkeypatch.setitem(sys.modules, "ompl.base", self.ob)
        monkeypatch.setitem(sys.modules, "ompl.geometric", self.og)
        monkeypatch.setattr(sys.modules[__name__], "_active_recorder", self)
        return self


_active_recorder: _FakeOmpl | None = None
"""Module-global recorder pointed at the active ``_FakeOmpl`` by ``install``."""


@pytest.fixture()
def fake_ompl(monkeypatch: pytest.MonkeyPatch) -> _FakeOmpl:
    """Install mocked OMPL bindings solved with a three-state path."""
    return _FakeOmpl(
        solve_result=True,
        solution_states=[_OmplState(0.0, 0.0), _OmplState(3.0, 0.0), _OmplState(3.0, 4.0)],
        exact_solution=True,
    ).install(monkeypatch)


# ---------------------------------------------------------------------------
# In-memory map fixture (no SVG, no repository maps)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_map_def() -> MapDefinition:
    """Minimal 10x10 MapDefinition with one central rectangular obstacle."""
    obstacle = Obstacle([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)])
    zone = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.0, 1.0), (9.0, 9.0)],
        spawn_zone=zone,
        goal_zone=zone,
    )
    # Flat (x_start, x_end, y_start, y_end) bounds; exactly four edges.
    bounds = [
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
    ]
    return MapDefinition(
        width=10.0,
        height=10.0,
        obstacles=[obstacle],
        robot_spawn_zones=[zone],
        ped_spawn_zones=[zone],
        robot_goal_zones=[zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
    )


# ---------------------------------------------------------------------------
# Planner-choice routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("choice", "expected_planner"),
    [
        (OmplPlannerChoice.RRTCONNECT, "RRTConnect"),
        (OmplPlannerChoice.BITSTAR, "BITstar"),
        (OmplPlannerChoice.RRTSTAR, "RRTstar"),
        (OmplPlannerChoice.INFORMED_RRTSTAR, "InformedRRTstar"),
        (OmplPlannerChoice.PRMSTAR, "PRMstar"),
    ],
)
def test_plan_routes_each_planner_choice(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
    choice: OmplPlannerChoice,
    expected_planner: str,
) -> None:
    """Each OmplPlannerChoice instantiates the matching OMPL planner class."""
    adapter = OmplGeometricAdapter(tiny_map_def, planner=choice)
    result = adapter.plan(start=(1.0, 1.0), goal=(9.0, 9.0))

    assert fake_ompl.planner_log == [expected_planner]
    assert result.solved is True
    assert result.planner_name == expected_planner


def test_default_constructor_uses_bitstar_shorthand(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """Omitting config/planner defaults to BITstar via the keyword shorthand."""
    adapter = OmplGeometricAdapter(tiny_map_def)
    adapter.plan(start=(1.0, 1.0), goal=(9.0, 9.0))

    assert fake_ompl.planner_log == ["BITstar"]


def test_explicit_config_overrides_planner_shorthand(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """An explicit config wins over the ``planner`` convenience shorthand."""
    config = OmplGeometricConfig(planner=OmplPlannerChoice.RRTSTAR)
    adapter = OmplGeometricAdapter(
        tiny_map_def,
        config=config,
        planner=OmplPlannerChoice.RRTCONNECT,
    )
    adapter.plan(start=(1.0, 1.0), goal=(9.0, 9.0))

    assert fake_ompl.planner_log == ["RRTstar"]


# ---------------------------------------------------------------------------
# State validity (collision + bounds)
# ---------------------------------------------------------------------------


def test_state_validity_rejects_obstacle_interior(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """The adapter checker marks states inside an obstacle as invalid."""
    adapter = OmplGeometricAdapter(tiny_map_def)

    assert adapter._checker.isValid(_OmplState(5.0, 5.0)) is False
    assert adapter._checker.isValid(_OmplState(4.5, 5.5)) is False


def test_state_validity_accepts_free_space(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """The adapter checker marks states in free space as valid."""
    adapter = OmplGeometricAdapter(tiny_map_def)

    assert adapter._checker.isValid(_OmplState(1.0, 1.0)) is True
    assert adapter._checker.isValid(_OmplState(8.0, 2.0)) is True


def test_state_validity_rejects_boundary_wall(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """Map-boundary walls are part of the obstacle union, so edges are invalid."""
    adapter = OmplGeometricAdapter(tiny_map_def)

    # The left boundary wall occupies x in [0, 0.05] (margin = 0.05 in the adapter).
    assert adapter._checker.isValid(_OmplState(0.02, 5.0)) is False
    # The bottom boundary wall occupies y in [0, 0.05].
    assert adapter._checker.isValid(_OmplState(5.0, 0.02)) is False


# ---------------------------------------------------------------------------
# Bounds configuration
# ---------------------------------------------------------------------------


def test_space_bounds_match_map_dimensions(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """The RealVectorStateSpace is 2D and bounded to the map extent."""
    OmplGeometricAdapter(tiny_map_def)

    assert fake_ompl.space is not None
    assert fake_ompl.space.dim == 2
    assert fake_ompl.bounds is not None
    assert fake_ompl.bounds.lows == {0: 0.0, 1: 0.0}
    assert fake_ompl.bounds.highs == {0: 10.0, 1: 10.0}


# ---------------------------------------------------------------------------
# Start / goal / time-budget configuration
# ---------------------------------------------------------------------------


def test_plan_configures_start_goal_and_budget(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """plan() clears setup, assigns start/goal coords, and forwards the budget."""
    config = OmplGeometricConfig(planner=OmplPlannerChoice.BITSTAR, time_budget_s=2.5)
    adapter = OmplGeometricAdapter(tiny_map_def, config=config)
    adapter.plan(start=(1.0, 2.0), goal=(8.0, 9.0))

    setup = fake_ompl.setup
    assert setup is not None
    assert setup.cleared is True
    assert setup.start_state.coords == [1.0, 2.0]
    assert setup.goal_state.coords == [8.0, 9.0]
    assert setup.solve_budget == 2.5


# ---------------------------------------------------------------------------
# Solved path conversion
# ---------------------------------------------------------------------------


def test_solved_result_converts_path_states(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """A solved plan converts OMPL states to waypoints and sums segment lengths."""
    adapter = OmplGeometricAdapter(tiny_map_def, planner=OmplPlannerChoice.BITSTAR)
    result = adapter.plan(start=(0.0, 0.0), goal=(3.0, 4.0))

    assert result.solved is True
    assert result.waypoints == [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    # 3-4-5 triangle legs: 3.0 + 4.0 = 7.0.
    assert result.path_length_m == pytest.approx(7.0)
    assert result.exact_solution is True
    assert result.planning_time_s >= 0.0


def test_solved_single_waypoint_has_zero_length(
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_def: MapDefinition,
) -> None:
    """A solved path with a single state yields zero length and one waypoint."""
    _FakeOmpl(solve_result=True, solution_states=[_OmplState(2.0, 2.0)]).install(monkeypatch)
    adapter = OmplGeometricAdapter(tiny_map_def)
    result = adapter.plan(start=(2.0, 2.0), goal=(2.0, 2.0))

    assert result.solved is True
    assert result.waypoints == [(2.0, 2.0)]
    assert result.path_length_m == 0.0


def test_solved_exact_solution_flag_reflects_setup(
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_def: MapDefinition,
) -> None:
    """exact_solution mirrors SimpleSetup.haveExactSolutionPath()."""
    _FakeOmpl(
        solve_result=True,
        solution_states=[_OmplState(0.0, 0.0), _OmplState(1.0, 0.0)],
        exact_solution=False,
    ).install(monkeypatch)
    adapter = OmplGeometricAdapter(tiny_map_def)
    result = adapter.plan(start=(0.0, 0.0), goal=(1.0, 0.0))

    assert result.solved is True
    assert result.exact_solution is False


# ---------------------------------------------------------------------------
# Path interpolation toggle
# ---------------------------------------------------------------------------


def test_interpolate_enabled_calls_path_interpolate(
    fake_ompl: _FakeOmpl,
    tiny_map_def: MapDefinition,
) -> None:
    """interpolate_waypoints > 0 triggers path.interpolate with that count."""
    config = OmplGeometricConfig(planner=OmplPlannerChoice.BITSTAR, interpolate_waypoints=50)
    adapter = OmplGeometricAdapter(tiny_map_def, config=config)
    adapter.plan(start=(0.0, 0.0), goal=(1.0, 1.0))

    assert fake_ompl.setup is not None
    assert fake_ompl.setup.solution_path.interpolate_calls == [50]


def test_interpolate_disabled_skips_path_interpolate(
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_def: MapDefinition,
) -> None:
    """interpolate_waypoints == 0 skips interpolation entirely."""
    fake = _FakeOmpl(
        solve_result=True,
        solution_states=[_OmplState(0.0, 0.0), _OmplState(1.0, 0.0)],
    ).install(monkeypatch)
    config = OmplGeometricConfig(planner=OmplPlannerChoice.BITSTAR, interpolate_waypoints=0)
    adapter = OmplGeometricAdapter(tiny_map_def, config=config)
    adapter.plan(start=(0.0, 0.0), goal=(1.0, 0.0))

    assert fake.setup is not None
    assert fake.setup.solution_path.interpolate_calls == []


# ---------------------------------------------------------------------------
# Unsolved result
# ---------------------------------------------------------------------------


def test_unsolved_returns_stable_failure_result(
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_def: MapDefinition,
) -> None:
    """A falsy solve status yields solved=False with zero length and no waypoints."""
    fake = _FakeOmpl(solve_result=False).install(monkeypatch)
    config = OmplGeometricConfig(planner=OmplPlannerChoice.RRTSTAR, time_budget_s=0.001)
    adapter = OmplGeometricAdapter(tiny_map_def, config=config)
    result = adapter.plan(start=(1.0, 1.0), goal=(9.0, 9.0))

    assert fake.setup is not None
    assert fake.setup.solve_budget == 0.001
    assert result.solved is False
    assert result.planner_name == "RRTstar"
    assert result.path_length_m == 0.0
    assert result.waypoints == []
    assert result.exact_solution is False
    assert result.planning_time_s >= 0.0


# ---------------------------------------------------------------------------
# Dependency unavailable
# ---------------------------------------------------------------------------


def test_construction_raises_when_ompl_missing(
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_def: MapDefinition,
) -> None:
    """When ompl is unimportable, construction raises ImportError with guidance.

    Setting ``sys.modules['ompl'] = None`` makes ``import ompl`` raise ImportError
    (CPython treats a None entry as a halted import), so this exercises the
    adapter's dependency guard in-process even where ompl is installed.
    """
    monkeypatch.setitem(sys.modules, "ompl", None)
    monkeypatch.setitem(sys.modules, "ompl.base", None)
    monkeypatch.setitem(sys.modules, "ompl.geometric", None)

    with pytest.raises(ImportError, match="ompl is not installed") as exc_info:
        OmplGeometricAdapter(tiny_map_def)

    assert "uv pip install ompl" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Exception propagation (locked current contract)
# ---------------------------------------------------------------------------


def test_plan_propagates_solve_exception(
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_def: MapDefinition,
) -> None:
    """A solve() exception currently propagates from plan() (locked contract).

    The adapter does not swallow exceptions raised by OMPL during ``solve()``.
    This test pins that current behaviour and captures the diagnostic evidence
    (exception type and message) deterministically. Converting such exceptions
    into a ``solved=False`` result would be a deliberate future hardening and is
    intentionally out of scope here, to avoid changing planning semantics.
    """
    _FakeOmpl(solve_exc=RuntimeError("ompl internal failure")).install(monkeypatch)
    adapter = OmplGeometricAdapter(tiny_map_def)

    with pytest.raises(RuntimeError, match="ompl internal failure"):
        adapter.plan(start=(1.0, 1.0), goal=(9.0, 9.0))
