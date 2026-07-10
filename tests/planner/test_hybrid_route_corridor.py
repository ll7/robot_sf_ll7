"""Unit tests for the extracted route-corridor geometry helpers (issue #4987).

These cover ``robot_sf/planner/hybrid_route_corridor.py`` directly: the pure,
``self``-independent route-geometry functions extracted from
``HybridRuleLocalPlannerAdapter`` as the first god-class decomposition move.
They assert fail-closed behavior for malformed / missing inputs and correct
geometry for well-formed inputs, and verify the adapter still delegates to them
so ``plan()`` / ``diagnostics()`` outputs stay byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner import hybrid_route_corridor as hrc
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)

# ---------------------------------------------------------------------------
# route_point
# ---------------------------------------------------------------------------


def test_route_point_reads_finite_xy_pair() -> None:
    """A finite 2-element value reads back as an (x, y) array."""
    point = hrc.route_point({"p": [1.0, 2.0]}, "p")
    assert point is not None
    np.testing.assert_allclose(point, [1.0, 2.0])


def test_route_point_reads_nested_xy_pair() -> None:
    """A nested 2-element value is flattened to an (x, y) array."""
    point = hrc.route_point({"p": [[1.0, 2.0]]}, "p")
    assert point is not None
    np.testing.assert_allclose(point, [1.0, 2.0])


def test_route_point_fail_closed_on_missing_inputs() -> None:
    """Non-dict corridor and missing keys return None."""
    assert hrc.route_point(None, "p") is None
    assert hrc.route_point("not a dict", "p") is None  # type: ignore[arg-type]
    assert hrc.route_point({}, "missing") is None


def test_route_point_fail_closed_on_non_finite_or_short() -> None:
    """Non-finite or single-element values return None."""
    assert hrc.route_point({"p": [float("nan"), 2.0]}, "p") is None
    assert hrc.route_point({"p": [1.0]}, "p") is None  # only one element
    assert hrc.route_point({"p": object()}, "p") is None


# ---------------------------------------------------------------------------
# route_float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1.5, 2, np.float64(3.25)])
def test_route_float_reads_finite_scalar(value: float) -> None:
    """Finite numeric scalars read back as floats."""
    assert hrc.route_float({"k": value}, "k") == float(value)


def test_route_float_fail_closed_on_missing_or_non_numeric() -> None:
    """Missing, non-numeric, or non-finite values return None."""
    assert hrc.route_float(None, "k") is None
    assert hrc.route_float({}, "missing") is None
    assert hrc.route_float({"k": "not numeric"}, "k") is None
    assert hrc.route_float({"k": float("inf")}, "k") is None
    assert hrc.route_float({"k": None}, "k") is None


# ---------------------------------------------------------------------------
# route_progress_pair
# ---------------------------------------------------------------------------


def test_route_progress_pair_reads_windows() -> None:
    """Well-formed 1s/3s windows read back as a float tuple."""
    assert hrc.route_progress_pair({"route_arc_progress_windows": {"1s": 0.1, "3s": 0.3}}) == (
        0.1,
        0.3,
    )


def test_route_progress_pair_fail_closed_on_malformed() -> None:
    """Missing, non-dict, partial, or non-finite windows return None."""
    assert hrc.route_progress_pair(None) is None
    assert hrc.route_progress_pair({}) is None
    assert hrc.route_progress_pair({"route_arc_progress_windows": "nope"}) is None
    assert hrc.route_progress_pair({"route_arc_progress_windows": {"1s": 0.1}}) is None
    assert (
        hrc.route_progress_pair({"route_arc_progress_windows": {"1s": float("nan"), "3s": 0.3}})
        is None
    )


# ---------------------------------------------------------------------------
# route_tangent_heading
# ---------------------------------------------------------------------------


def test_route_tangent_heading_uses_explicit_heading_when_present() -> None:
    """An explicit route_tangent_heading scalar is returned directly."""
    corridor = {"route_tangent_heading": 0.7}
    assert hrc.route_tangent_heading(corridor) == 0.7


def test_route_tangent_heading_derives_from_route_points() -> None:
    """When no explicit heading, it derives atan2 from start->next points."""
    corridor = {
        "route_start_world": [0.0, 0.0],
        "route_next_world": [1.0, 0.0],
    }
    assert hrc.route_tangent_heading(corridor) == 0.0
    # Points straight up in +y -> heading pi/2.
    corridor["route_next_world"] = [0.0, 1.0]
    assert hrc.route_tangent_heading(corridor) == pytest.approx(np.pi / 2)


def test_route_tangent_heading_falls_back_to_waypoint_when_next_missing() -> None:
    """Missing route_next falls back to route_waypoint for derivation."""
    corridor = {
        "route_start_world": [0.0, 0.0],
        "route_waypoint_world": [1.0, 1.0],
    }
    assert hrc.route_tangent_heading(corridor) == pytest.approx(np.pi / 4)


def test_route_tangent_heading_fail_closed_on_degenerate_or_missing() -> None:
    """Missing points or a zero-length delta return None."""
    assert hrc.route_tangent_heading(None) is None
    assert hrc.route_tangent_heading({}) is None
    # Coincident start/next -> zero-length delta -> None.
    assert (
        hrc.route_tangent_heading({"route_start_world": [1.0, 1.0], "route_next_world": [1.0, 1.0]})
        is None
    )


# ---------------------------------------------------------------------------
# lateral_offset_to_segment
# ---------------------------------------------------------------------------


def test_lateral_offset_to_segment_computes_perpendicular_distance() -> None:
    """Returns the perpendicular distance from point to segment line."""
    point = np.array([0.0, 3.0])
    start = np.array([0.0, 0.0])
    stop = np.array([5.0, 0.0])
    assert hrc.lateral_offset_to_segment(point, start, stop) == pytest.approx(3.0)


def test_lateral_offset_to_segment_fail_closed_on_degenerate_segment() -> None:
    """A zero-length segment returns None (cannot define a lateral offset)."""
    start = np.array([1.0, 1.0])
    stop = np.array([1.0, 1.0])  # zero length
    assert hrc.lateral_offset_to_segment(np.array([2.0, 2.0]), start, stop) is None


# ---------------------------------------------------------------------------
# Adapter delegation (behavior preservation)
# ---------------------------------------------------------------------------


def _adapter() -> HybridRuleLocalPlannerAdapter:
    """Build a default adapter for delegation checks."""
    return HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config({}),
    )


def test_adapter_route_helpers_delegate_to_module_functions() -> None:
    """Adapter methods are thin wrappers over the extracted module functions."""
    planner = _adapter()
    corridor = {
        "route_remaining_distance": 4.2,
        "route_waypoint_world": [3.0, 4.0],
        "route_arc_progress_windows": {"1s": 0.5, "3s": 1.5},
        "route_tangent_heading": 1.25,
    }

    # The adapter method results must equal the module-level function results
    # for the same inputs (byte-identical delegation).
    np.testing.assert_allclose(
        planner._route_point(corridor, "route_waypoint_world"),
        hrc.route_point(corridor, "route_waypoint_world"),
    )
    assert planner._route_float(corridor, "route_remaining_distance") == hrc.route_float(
        corridor, "route_remaining_distance"
    )
    assert planner._route_progress_pair(corridor) == hrc.route_progress_pair(corridor)
    assert planner._route_tangent_heading(corridor) == hrc.route_tangent_heading(corridor)

    point = np.array([0.0, 2.0])
    start = np.array([0.0, 0.0])
    stop = np.array([4.0, 0.0])
    assert planner._lateral_offset_to_segment(point, start, stop) == (
        hrc.lateral_offset_to_segment(point, start, stop)
    )


def test_adapter_route_helpers_fail_closed_consistently() -> None:
    """Adapter wrappers preserve the fail-closed None contract."""
    planner = _adapter()
    assert planner._route_point(None, "x") is None
    assert planner._route_float(None, "x") is None
    assert planner._route_progress_pair(None) is None
    assert planner._route_tangent_heading(None) is None
