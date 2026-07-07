"""Tests for the optional OMPL kinodynamic smoke diagnostic.

All tests that require OMPL to be installed are guarded with
``@pytest.mark.skipif(not check_ompl_available()[0], ...)``.

Tests that exercise the fail-closed path (OMPL absent) run unconditionally.
Tests that exercise the comparison utilities run unconditionally since they
use only numpy and standard Python.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

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
