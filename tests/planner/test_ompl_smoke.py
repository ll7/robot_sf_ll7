"""Tests for the optional OMPL kinodynamic smoke diagnostic.

All tests that require OMPL to be installed are guarded with
``@pytest.mark.skipif(not check_ompl_available()[0], ...)``.

Tests that exercise the fail-closed path (OMPL absent) run unconditionally.
Tests that exercise the comparison utilities run unconditionally since they
use only numpy and standard Python.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest import mock

import numpy as np
import pytest

from robot_sf.planner import ompl_smoke
from robot_sf.planner.ompl_smoke import (
    OmplSmokeConfig,
    OmplSmokeResult,
    check_ompl_available,
    compare_with_classic_route,
    smoke_plan,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """Default config should use reasonable planning bounds."""
    cfg = OmplSmokeConfig()
    assert cfg.state_bounds[0] == 0.0
    assert cfg.state_bounds[1] == 50.0
    assert cfg.control_bounds[1] == 1.5  # max linear speed
    assert cfg.dt == 0.1
    assert cfg.robot_radius == 0.25


def test_config_custom_bounds() -> None:
    """Custom config should allow tighter or looser planning bounds."""
    cfg = OmplSmokeConfig(
        state_bounds=(0.0, 10.0, 0.0, 10.0, -3.1416, 3.1416),
        control_bounds=(0.0, 0.5, -1.0, 1.0),
        max_planning_time_sec=2.0,
    )
    assert cfg.state_bounds[1] == 10.0
    assert cfg.control_bounds[1] == 0.5
    assert cfg.max_planning_time_sec == 2.0


# ---------------------------------------------------------------------------
# Availability check tests
# ---------------------------------------------------------------------------


def test_check_ompl_available_returns_bool() -> None:
    """check_ompl_available should return a (bool, str|None) tuple."""
    available, error = check_ompl_available()
    assert isinstance(available, bool)
    if available:
        assert error is None
    else:
        assert isinstance(error, str)


# ---------------------------------------------------------------------------
# Smoke plan tests (when OMPL is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not check_ompl_available()[0],
    reason="OMPL not installed; skipping integration tests",
)
class TestOmplSmokePlan:
    """Tests that require OMPL to be available."""

    def test_straight_line_feasible(self) -> None:
        """A straight-line route in open space should be feasible."""
        result = smoke_plan(start=(1.0, 1.0), goal=(8.0, 1.0))
        assert result.success
        assert result.path_length > 0
        assert result.error is None

    def test_short_route_returns_quickly(self) -> None:
        """A short route should complete planning quickly."""
        result = smoke_plan(
            start=(2.0, 2.0),
            goal=(4.0, 2.0),
            config=OmplSmokeConfig(max_planning_time_sec=5.0),
        )
        assert result.success
        assert result.planning_time_sec < 5.0

    def test_path_connects_start_to_goal(self) -> None:
        """The returned path should approximately connect start and goal."""
        result = smoke_plan(start=(2.0, 2.0), goal=(8.0, 2.0))
        assert result.success
        first = result.path_states[0]
        last = result.path_states[-1]
        assert np.isclose(first[0], 2.0, atol=0.6)
        assert np.isclose(first[1], 2.0, atol=0.6)
        assert np.isclose(last[0], 8.0, atol=1.0)
        assert np.isclose(last[1], 2.0, atol=1.0)

    def test_path_states_have_three_components(self) -> None:
        """Each path state should be (x, y, theta)."""
        result = smoke_plan(start=(1.0, 1.0), goal=(5.0, 5.0))
        assert result.success
        for state in result.path_states:
            assert len(state) == 3
            _x, _y, theta = state
            # Theta bounds are soft — OMPL may propagate slightly beyond bounds
            assert -7.0 < theta < 7.0

    def test_obstacle_avoidance_with_polygons(self) -> None:
        """Planner should route around obstacle polygons when provided."""
        import shapely.geometry as sg

        # Wall between start and goal
        wall = sg.box(4.0, 0.0, 5.0, 10.0)
        result = smoke_plan(
            start=(2.0, 5.0),
            goal=(8.0, 5.0),
            config=OmplSmokeConfig(
                state_bounds=(0.0, 10.0, 0.0, 10.0, -3.1416, 3.1416),
                max_planning_time_sec=5.0,
                robot_radius=0.2,
            ),
            obstacle_polygons=[wall],
        )
        # Path should still be found (go around the wall)
        assert result.success
        assert result.path_length > 0


# ---------------------------------------------------------------------------
# Fail-closed tests (when OMPL is not available)
# ---------------------------------------------------------------------------


class TestOmplFailClosed:
    """Tests that simulate OMPL being unavailable."""

    def test_smoke_plan_fails_closed_when_ompl_missing(self) -> None:
        """smoke_plan should return a failed result when OMPL is missing."""
        with mock.patch("robot_sf.planner.ompl_smoke._OMPL_AVAILABLE", False):
            with mock.patch(
                "robot_sf.planner.ompl_smoke._OMPL_IMPORT_ERROR",
                "ModuleNotFoundError",
            ):
                result = smoke_plan(start=(0.0, 0.0), goal=(1.0, 1.0))

        assert not result.success
        assert result.path_length == 0
        assert result.path_states == []
        assert result.error is not None
        assert "OMPL not available" in result.error

    def test_check_returns_unavailable_when_mocked(self) -> None:
        """check_ompl_available should reflect mocked unavailability."""
        # Structural test: check_ompl_available always returns (bool, str|None).
        available, error = check_ompl_available()
        assert isinstance(available, bool)
        assert isinstance(error, (str, type(None)))


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------


def test_compare_returns_impossible_when_ompl_failed() -> None:
    """Comparison should report impossible when OMPL didn't produce a path."""
    result = OmplSmokeResult(
        success=False,
        path_length=0,
        path_states=[],
        planning_time_sec=0.0,
        error="no solution",
    )
    comparison = compare_with_classic_route(result, [(0, 0), (1, 1)])
    assert not comparison["comparison_possible"]
    assert "OMPL did not produce a valid path" in comparison["reason"]


def test_compare_returns_impossible_when_classic_empty() -> None:
    """Comparison should report impossible when classic path is empty."""
    result = OmplSmokeResult(
        success=True,
        path_length=5,
        path_states=[(0, 0, 0), (1, 1, 0.5)],
        planning_time_sec=0.5,
    )
    comparison = compare_with_classic_route(result, [])
    assert not comparison["comparison_possible"]
    assert "Classic path is empty" in comparison["reason"]


def test_compare_computes_diagnostics() -> None:
    """Comparison should compute path length and deviation diagnostics."""
    ompl_result = OmplSmokeResult(
        success=True,
        path_length=4,
        path_states=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ],
        planning_time_sec=0.3,
    )
    classic_path = [(0.0, 0.0), (1.5, 0.1), (3.0, 0.0)]

    comparison = compare_with_classic_route(ompl_result, classic_path)

    assert comparison["comparison_possible"]
    assert comparison["ompl_path_steps"] == 4
    assert comparison["classic_path_steps"] == 3
    assert isinstance(comparison["ompl_length_m"], float)
    assert isinstance(comparison["classic_length_m"], float)
    assert isinstance(comparison["max_lateral_deviation_m"], float)


# ---------------------------------------------------------------------------
# Residual branch coverage (issue #6392): the obstacle state-validity-checker
# closure body inside ``smoke_plan`` and the module-level OMPL import-success
# branch. Neither requires a real OMPL install: ``ompl.base``/``ompl.control``
# are injected into ``sys.modules`` as ``MagicMock`` modules attached to a real
# ``types.ModuleType("ompl")`` package (the same technique as open PR #6389),
# so the inline imports inside ``smoke_plan`` and the module-level ``try`` body
# both resolve to the mocks. No planning semantics are changed; the source
# module is not edited.
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_ompl_lifecycle(monkeypatch):
    """Inject mocked ``ompl.base``/``ompl.control`` and enable the smoke lifecycle.

    ``smoke_plan``'s inline ``import ompl.base``/``ompl.control`` resolve to
    ``MagicMock`` objects without a real OMPL install. ``monkeypatch`` restores
    ``sys.modules`` and the availability flags on teardown.
    """
    mock_base = mock.MagicMock(name="ompl.base")
    mock_control = mock.MagicMock(name="ompl.control")
    ompl_pkg = types.ModuleType("ompl")
    ompl_pkg.base = mock_base
    ompl_pkg.control = mock_control
    monkeypatch.setitem(sys.modules, "ompl", ompl_pkg)
    monkeypatch.setitem(sys.modules, "ompl.base", mock_base)
    monkeypatch.setitem(sys.modules, "ompl.control", mock_control)
    monkeypatch.setattr(ompl_smoke, "_OMPL_AVAILABLE", True)
    monkeypatch.setattr(ompl_smoke, "_OMPL_IMPORT_ERROR", None)

    simple_setup = mock_control.SimpleSetup.return_value
    space_info = simple_setup.getSpaceInformation.return_value
    # A solved plan with an empty extracted path keeps the focus on the
    # obstacle-validity wiring rather than solution-path extraction.
    simple_setup.solve.return_value = True
    simple_setup.getSolutionPath.return_value.getStates.return_value = []

    return types.SimpleNamespace(
        base=mock_base,
        control=mock_control,
        simple_setup=simple_setup,
        space_info=space_info,
    )


def test_obstacle_validity_checker_closure_uses_buffered_polygons(mocked_ompl_lifecycle):
    """The captured state-validity checker rejects points inside the
    ``robot_radius``-buffered obstacle polygon (including the buffer margin) and
    accepts a clear point, exercising the ``is_state_valid`` closure body that
    open PR #6389 only asserts is registered.
    """
    import shapely.geometry as sg

    obstacle = sg.box(4.0, 4.0, 5.0, 6.0)
    robot_radius = 0.3
    buffered = obstacle.buffer(robot_radius)

    result = smoke_plan(
        start=(1.0, 5.0),
        goal=(9.0, 5.0),
        config=OmplSmokeConfig(
            state_bounds=(0.0, 10.0, 0.0, 10.0, -3.1416, 3.1416),
            robot_radius=robot_radius,
        ),
        obstacle_polygons=[obstacle],
    )

    # smoke_plan completed against the mocks and registered a validity checker.
    assert result.success is True
    mocked_ompl_lifecycle.space_info.setStateValidityChecker.assert_called_once()
    is_state_valid = mocked_ompl_lifecycle.space_info.setStateValidityChecker.call_args[0][0]
    assert callable(is_state_valid)

    # A point inside the buffer margin but OUTSIDE the raw obstacle: this is
    # only rejected because the closure checks the robot_radius-buffered polygon.
    margin_point = (4.0 - robot_radius / 2.0, 5.0)
    assert not obstacle.contains(sg.Point(*margin_point))
    assert buffered.contains(sg.Point(*margin_point))
    assert is_state_valid(margin_point) is False

    # A point deep inside the raw obstacle is trivially inside the buffer too.
    assert is_state_valid((4.5, 5.0)) is False

    # A clear point well away from every buffered polygon is valid.
    assert is_state_valid((1.0, 1.0)) is True


def test_module_level_ompl_import_success_branch_via_reload():
    """Reload ``ompl_smoke`` with mocked ``ompl`` in ``sys.modules`` to exercise
    the module-level import-success ``try`` body (the imports succeed,
    ``_OMPL_AVAILABLE`` becomes ``True``, ``_OMPL_IMPORT_ERROR`` stays ``None``),
    then restore the prior ``sys.modules`` entries and reload the module back to
    its real import state so other tests and files are unaffected.
    """
    sentinel = object()
    ompl_keys = ("ompl", "ompl.base", "ompl.control")
    saved_modules = {
        key: (sys.modules[key] if key in sys.modules else sentinel) for key in ompl_keys
    }
    # ``ompl_base``/``ompl_control`` are module-level names bound only on a
    # successful import (unused by smoke_plan, which local-imports them).
    saved_bindings = {
        name: (ompl_smoke.__dict__[name] if name in ompl_smoke.__dict__ else sentinel)
        for name in ("ompl_base", "ompl_control")
    }
    saved_available = ompl_smoke._OMPL_AVAILABLE
    saved_error = ompl_smoke._OMPL_IMPORT_ERROR

    try:
        mock_base = mock.MagicMock(name="ompl.base")
        mock_control = mock.MagicMock(name="ompl.control")
        ompl_pkg = types.ModuleType("ompl")
        ompl_pkg.base = mock_base
        ompl_pkg.control = mock_control
        sys.modules["ompl"] = ompl_pkg
        sys.modules["ompl.base"] = mock_base
        sys.modules["ompl.control"] = mock_control

        importlib.reload(ompl_smoke)

        # The module-level try-body ran against the mocks.
        assert ompl_smoke._OMPL_AVAILABLE is True
        assert ompl_smoke._OMPL_IMPORT_ERROR is None
        assert ompl_smoke.check_ompl_available() == (True, None)
    finally:
        for key in ompl_keys:
            value = saved_modules[key]
            if value is sentinel:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        # Reload against the restored sys.modules so the availability flags and
        # module-level bindings return to their real prior import state.
        importlib.reload(ompl_smoke)
        for name in ("ompl_base", "ompl_control"):
            prior = saved_bindings[name]
            if prior is sentinel:
                ompl_smoke.__dict__.pop(name, None)
            else:
                setattr(ompl_smoke, name, prior)
        assert ompl_smoke._OMPL_AVAILABLE == saved_available
        assert ompl_smoke._OMPL_IMPORT_ERROR == saved_error
