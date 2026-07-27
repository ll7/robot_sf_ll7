"""Branch-coverage tests for the optional OMPL kinodynamic smoke diagnostic.

These tests lock the differential-drive propagation helper and the
``smoke_plan`` planning lifecycle without requiring a real OMPL install:

* ``_differential_drive_propagate`` is covered directly for straight motion,
  turning motion, and heading values beyond ``+/-pi`` (the helper performs no
  heading wrapping), plus the ``float64`` return contract.
* ``smoke_plan`` is exercised through mocked ``ompl.base`` / ``ompl.control``
  modules injected into ``sys.modules`` so the fail-closed import path, the
  solved and unsolved planner outcomes, solution-path extraction, and the
  obstacle/shapely configuration branches are all reachable without OMPL and
  without loading any map or running a benchmark.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from robot_sf.planner import ompl_smoke
from robot_sf.planner.ompl_smoke import OmplSmokeConfig, smoke_plan

_PROPAGATE = ompl_smoke._differential_drive_propagate


class _FakeState:
    """Indexable stand-in for an OMPL ``State``/``Control`` object.

    Supports both ``obj[i]`` reads (used by the solution-path extraction loop)
    and ``obj[i] = value`` writes (used when seeding start/goal states), so the
    mocked lifecycle drives the same indexing code paths as real OMPL.
    """

    __slots__ = ("_values",)

    def __init__(self, values=(0.0, 0.0, 0.0)) -> None:
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value) -> None:
        self._values[index] = value


@pytest.fixture
def ompl_env(monkeypatch):
    """Inject mocked ``ompl.base``/``ompl.control`` modules and enable the lifecycle.

    Returns a namespace of mock handles (state/control spaces, captured bounds,
    the ``SimpleSetup``/``SpaceInformation`` pair, and the RNG) so each test can
    assert exactly which configured resources the planner received.
    """
    mock_base = MagicMock(name="ompl.base")
    mock_control = MagicMock(name="ompl.control")

    bounds_by_dim: dict[int, MagicMock] = {}

    def _make_bounds(dim):
        created = MagicMock(name=f"RealVectorBounds({dim})")
        bounds_by_dim[dim] = created
        return created

    mock_base.RealVectorBounds.side_effect = _make_bounds

    state_space = mock_base.RealVectorStateSpace.return_value
    control_space = mock_control.RealVectorControlSpace.return_value
    simple_setup = mock_control.SimpleSetup.return_value
    space_info = simple_setup.getSpaceInformation.return_value
    rng = mock_base.RNG.return_value

    # Each allocState() call must hand back a fresh writable state object.
    def _alloc_state():
        return _FakeState()

    state_space.allocState.side_effect = _alloc_state

    ompl_pkg = ModuleType("ompl")
    ompl_pkg.base = mock_base
    ompl_pkg.control = mock_control
    monkeypatch.setitem(sys.modules, "ompl", ompl_pkg)
    monkeypatch.setitem(sys.modules, "ompl.base", mock_base)
    monkeypatch.setitem(sys.modules, "ompl.control", mock_control)
    monkeypatch.setattr(ompl_smoke, "_OMPL_AVAILABLE", True)
    monkeypatch.setattr(ompl_smoke, "_OMPL_IMPORT_ERROR", None)

    return SimpleNamespace(
        base=mock_base,
        control=mock_control,
        bounds_by_dim=bounds_by_dim,
        state_space=state_space,
        control_space=control_space,
        simple_setup=simple_setup,
        space_info=space_info,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# _differential_drive_propagate: pure kinematics
# ---------------------------------------------------------------------------


def test_propagate_straight_motion_advances_along_heading():
    """omega=0 advances position along the current heading without turning."""
    state = np.array([0.0, 0.0, 0.0])
    result = _PROPAGATE(state, np.array([1.0, 0.0]), 2.0)
    assert result[0] == pytest.approx(2.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)


def test_propagate_straight_motion_holds_nonzero_heading():
    """A nonzero heading with omega=0 moves along that heading; theta is unchanged."""
    state = np.array([0.0, 0.0, math.pi / 2])
    result = _PROPAGATE(state, np.array([2.0, 0.0]), 1.0)
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(2.0)
    assert result[2] == pytest.approx(math.pi / 2)


def test_propagate_turning_motion_updates_heading_and_displacement():
    """Nonzero v and omega turn the heading and displace along the old heading."""
    state = np.array([0.0, 0.0, math.pi / 4])
    result = _PROPAGATE(state, np.array([1.0, 0.5]), 1.0)
    assert result[0] == pytest.approx(math.cos(math.pi / 4))
    assert result[1] == pytest.approx(math.sin(math.pi / 4))
    assert result[2] == pytest.approx(math.pi / 4 + 0.5)


def test_propagate_heading_grows_beyond_positive_pi_without_wrapping():
    """Heading beyond +pi grows linearly; the helper performs no wrapping."""
    theta0 = 3.5  # beyond +pi
    state = np.array([2.0, -1.0, theta0])
    result = _PROPAGATE(state, np.array([1.0, 0.2]), 1.0)
    assert result[0] == pytest.approx(2.0 + math.cos(theta0))
    assert result[1] == pytest.approx(-1.0 + math.sin(theta0))
    assert result[2] == pytest.approx(theta0 + 0.2)
    # The propagated value is genuinely beyond +pi and was NOT wrapped.
    assert result[2] > math.pi


def test_propagate_heading_grows_below_negative_pi_without_wrapping():
    """Heading below -pi decreases linearly; the helper performs no wrapping."""
    theta0 = -3.5  # below -pi
    state = np.array([1.0, 4.0, theta0])
    result = _PROPAGATE(state, np.array([1.0, -0.3]), 1.0)
    assert result[0] == pytest.approx(1.0 + math.cos(theta0))
    assert result[1] == pytest.approx(4.0 + math.sin(theta0))
    assert result[2] == pytest.approx(theta0 - 0.3)
    assert result[2] < -math.pi


def test_propagate_returns_float64_without_mutating_input():
    """The returned state is a fresh float64 array; the input array is untouched."""
    state = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    control = np.array([0.5, 0.25], dtype=np.float64)
    original = state.copy()
    result = _PROPAGATE(state, control, 0.5)

    assert result.dtype == np.float64
    assert result is not state
    np.testing.assert_array_equal(state, original)


# ---------------------------------------------------------------------------
# smoke_plan: fail-closed import path
# ---------------------------------------------------------------------------


def test_smoke_plan_fails_closed_with_unavailable_detail(monkeypatch):
    """When OMPL is unavailable the error names OMPL and the import-error detail."""
    detail = "No module named 'ompl'"
    monkeypatch.setattr(ompl_smoke, "_OMPL_AVAILABLE", False)
    monkeypatch.setattr(ompl_smoke, "_OMPL_IMPORT_ERROR", detail)

    result = smoke_plan(start=(0.0, 0.0), goal=(1.0, 1.0))

    assert result.success is False
    assert result.path_length == 0
    assert result.path_states == []
    assert result.error is not None
    assert "OMPL not available" in result.error
    assert detail in result.error


# ---------------------------------------------------------------------------
# smoke_plan: solved lifecycle forwards every config value
# ---------------------------------------------------------------------------


def test_solved_lifecycle_wires_all_configured_resources(ompl_env):
    """A solved plan forwards every config value to the mocked OMPL resources."""
    cfg = OmplSmokeConfig(
        state_bounds=(0.0, 20.0, 0.0, 20.0, -3.1416, 3.1416),
        control_bounds=(0.0, 1.2, -1.5, 1.5),
        dt=0.2,
        max_planning_time_sec=4.0,
        state_tolerance=0.35,
        min_control_duration=2,
        max_control_duration=12,
    )
    path_states = [
        _FakeState((1.0, 1.0, 0.0)),
        _FakeState((3.0, 1.0, 0.0)),
        _FakeState((5.0, 1.0, 0.0)),
    ]
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = path_states

    result = smoke_plan(start=(1.0, 1.0), goal=(5.0, 1.0), config=cfg)

    # --- Result shape ---
    assert result.success is True
    assert result.error is None
    assert result.path_length == 3
    assert result.path_states == [(1.0, 1.0, 0.0), (3.0, 1.0, 0.0), (5.0, 1.0, 0.0)]

    # --- State space + bounds ---
    ompl_env.base.RealVectorStateSpace.assert_called_once_with(3)
    state_bounds = ompl_env.bounds_by_dim[3]
    state_bounds.setLow.assert_has_calls(
        [
            call(0, cfg.state_bounds[0]),
            call(1, cfg.state_bounds[2]),
            call(2, cfg.state_bounds[4]),
        ]
    )
    state_bounds.setHigh.assert_has_calls(
        [
            call(0, cfg.state_bounds[1]),
            call(1, cfg.state_bounds[3]),
            call(2, cfg.state_bounds[5]),
        ]
    )
    ompl_env.state_space.setBounds.assert_called_once_with(state_bounds)

    # --- Control space + bounds ---
    ompl_env.control.RealVectorControlSpace.assert_called_once_with(ompl_env.state_space, 2)
    control_bounds = ompl_env.bounds_by_dim[2]
    control_bounds.setLow.assert_has_calls(
        [
            call(0, cfg.control_bounds[0]),
            call(1, cfg.control_bounds[2]),
        ]
    )
    control_bounds.setHigh.assert_has_calls(
        [
            call(0, cfg.control_bounds[1]),
            call(1, cfg.control_bounds[3]),
        ]
    )
    ompl_env.control_space.setBounds.assert_called_once_with(control_bounds)

    # --- SimpleSetup + SpaceInformation ---
    ompl_env.control.SimpleSetup.assert_called_once_with(ompl_env.control_space)
    ompl_env.simple_setup.getSpaceInformation.assert_called_once_with()

    # --- Propagation wiring ---
    ompl_env.space_info.setStatePropagator.assert_called_once()
    propagator = ompl_env.space_info.setStatePropagator.call_args[0][0]
    assert callable(propagator)
    ompl_env.space_info.setMinMaxControlDuration.assert_called_once_with(
        cfg.min_control_duration, cfg.max_control_duration
    )
    ompl_env.space_info.setPropagationStepSize.assert_called_once_with(cfg.dt)

    # --- Start/goal with tolerance and neutral heading ---
    ompl_env.simple_setup.setStartAndGoalStates.assert_called_once()
    start_state, goal_state, tolerance = ompl_env.simple_setup.setStartAndGoalStates.call_args[0]
    assert tolerance == cfg.state_tolerance
    assert start_state[0] == 1.0
    assert start_state[1] == 1.0
    assert start_state[2] == 0.0
    assert goal_state[0] == 5.0
    assert goal_state[1] == 1.0
    assert goal_state[2] == 0.0

    # --- Planner wiring + solve budget (seed unset -> RNG untouched) ---
    ompl_env.control.RRT.assert_called_once_with(ompl_env.space_info)
    ompl_env.simple_setup.setPlanner.assert_called_once_with(ompl_env.control.RRT.return_value)
    ompl_env.simple_setup.setup.assert_called_once_with()
    ompl_env.simple_setup.solve.assert_called_once_with(cfg.max_planning_time_sec)
    ompl_env.base.RNG.assert_not_called()


def test_solved_path_states_are_xyz_float_triples(ompl_env):
    """Each extracted state is a three-component (x, y, theta) tuple of floats."""
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = [
        _FakeState((1.5, 2.5, -0.5)),
        _FakeState((4.0, 3.0, 1.25)),
    ]

    result = smoke_plan(start=(1.5, 2.5), goal=(4.0, 3.0))

    assert result.success is True
    assert result.path_length == 2
    assert len(result.path_states) == 2
    for state in result.path_states:
        assert isinstance(state, tuple)
        assert len(state) == 3
        for component in state:
            assert isinstance(component, float)
    assert result.path_states[0] == (1.5, 2.5, -0.5)
    assert result.path_states[1] == (4.0, 3.0, 1.25)


def test_propagator_closure_applies_differential_drive(ompl_env):
    """The registered propagator mirrors the differential-drive update."""
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = []
    smoke_plan(start=(0.0, 0.0), goal=(1.0, 0.0))

    propagator = ompl_env.space_info.setStatePropagator.call_args[0][0]

    # Straight segment from the origin along +x.
    out = _FakeState()
    propagator(_FakeState((0.0, 0.0, 0.0)), _FakeState((1.0, 0.0)), 2.0, out)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)

    # Turning segment: nonzero heading + angular velocity updates theta and y.
    out = _FakeState()
    propagator(_FakeState((0.0, 0.0, math.pi / 2)), _FakeState((2.0, 0.5)), 1.0, out)
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(math.pi / 2 + 0.5)


def test_seed_is_applied_when_configured(ompl_env):
    """A configured seed is forwarded to OMPL's RNG before planning."""
    cfg = OmplSmokeConfig(seed=12345)
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = []

    smoke_plan(start=(0.0, 0.0), goal=(1.0, 0.0), config=cfg)

    ompl_env.base.RNG.assert_called_once_with()
    ompl_env.rng.setSeed.assert_called_once_with(12345)


def test_rng_is_not_touched_when_seed_unset(ompl_env):
    """A None seed leaves the OMPL RNG untouched."""
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = []

    smoke_plan(start=(0.0, 0.0), goal=(1.0, 0.0))

    ompl_env.base.RNG.assert_not_called()
    ompl_env.rng.setSeed.assert_not_called()


# ---------------------------------------------------------------------------
# smoke_plan: unsolved outcome
# ---------------------------------------------------------------------------


def test_unsolved_returns_time_budget_diagnostic(ompl_env):
    """An unsolved plan returns the time-budget diagnostic without extracting a path."""
    cfg = OmplSmokeConfig(max_planning_time_sec=0.5)
    ompl_env.simple_setup.solve.return_value = False

    result = smoke_plan(start=(0.0, 0.0), goal=(10.0, 10.0), config=cfg)

    assert result.success is False
    assert result.path_length == 0
    assert result.path_states == []
    assert result.error is not None
    assert "did not find a solution within the time budget" in result.error
    assert result.planning_time_sec >= 0.0
    ompl_env.simple_setup.solve.assert_called_once_with(cfg.max_planning_time_sec)
    ompl_env.simple_setup.getSolutionPath.assert_not_called()


# ---------------------------------------------------------------------------
# smoke_plan: obstacle / shapely configuration branches
# ---------------------------------------------------------------------------


def test_obstacle_polygons_install_validity_checker(ompl_env):
    """Obstacle polygons wire a state-validity checker when shapely is importable."""
    shapely_geom = pytest.importorskip("shapely.geometry")
    wall = shapely_geom.box(4.0, 0.0, 5.0, 10.0)
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = []

    result = smoke_plan(
        start=(1.0, 5.0),
        goal=(9.0, 5.0),
        config=OmplSmokeConfig(state_bounds=(0.0, 10.0, 0.0, 10.0, -3.1416, 3.1416)),
        obstacle_polygons=[wall],
    )

    assert result.success is True
    ompl_env.space_info.setStateValidityChecker.assert_called_once()
    validity_checker = ompl_env.space_info.setStateValidityChecker.call_args[0][0]
    assert callable(validity_checker)


def test_obstacle_polygons_skip_validity_when_shapely_missing(ompl_env, monkeypatch):
    """Missing shapely logs a warning and continues without a validity checker."""
    monkeypatch.setitem(sys.modules, "shapely", None)
    monkeypatch.setitem(sys.modules, "shapely.geometry", None)
    ompl_env.simple_setup.solve.return_value = True
    ompl_env.simple_setup.getSolutionPath.return_value.getStates.return_value = []

    result = smoke_plan(
        start=(1.0, 5.0),
        goal=(9.0, 5.0),
        obstacle_polygons=[MagicMock(name="obstacle")],
    )

    assert result.success is True
    ompl_env.space_info.setStateValidityChecker.assert_not_called()
    ompl_env.simple_setup.solve.assert_called_once()
