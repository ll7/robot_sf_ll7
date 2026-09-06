"""Focused contract tests for deterministic route geometry."""

from __future__ import annotations

import math

import pytest

from robot_sf.nav.global_route import RouteGeometry


def test_route_geometry_normalizes_duplicates_and_exposes_stable_identity() -> None:
    """Adjacent duplicate points should not create zero-length sections."""
    route = RouteGeometry([(0, 0), (1, 0), (1, 0), (2, 0)])
    equivalent = RouteGeometry([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    assert route.waypoints == ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    assert route.section_lengths == pytest.approx((1.0, 1.0))
    assert route.section_offsets == pytest.approx((0.0, 1.0))
    assert route.total_length_m == pytest.approx(2.0)
    assert route.route_hash == equivalent.route_hash


def test_point_at_arc_length_clamps_and_interpolates_across_unequal_sections() -> None:
    """Arc-length sampling should be continuous at corners and endpoints."""
    route = RouteGeometry([(0, 0), (3, 0), (3, 4)])

    assert route.point_at_arc_length(-1.0) == (0.0, 0.0)
    assert route.point_at_arc_length(3.0) == (3.0, 0.0)
    assert route.point_at_arc_length(5.0) == (3.0, 2.0)
    assert route.point_at_arc_length(10.0) == (3.0, 4.0)

    with pytest.raises(ValueError, match="arc_length_m must be finite"):
        route.point_at_arc_length(math.nan)


def test_projection_reports_arc_length_and_left_positive_lateral_offset() -> None:
    """Projection should use route-relative distance and a directed left frame."""
    route = RouteGeometry([(0, 0), (4, 0)])

    projection = route.project((2.0, 1.0))
    assert projection.status == "ok"
    assert projection.is_valid
    assert projection.projected_point == pytest.approx((2.0, 0.0))
    assert projection.segment_index == 0
    assert projection.arc_length_m == pytest.approx(2.0)
    assert projection.distance_m == pytest.approx(1.0)
    assert projection.lateral_offset_m == pytest.approx(1.0)

    right_projection = route.project((2.0, -1.0))
    assert right_projection.lateral_offset_m == pytest.approx(-1.0)


def test_projection_handles_rotated_routes_and_corner_endpoint_ties() -> None:
    """Rotation should not change semantics, and a shared corner is one branch."""
    route = RouteGeometry([(0, 0), (2, 2), (4, 2)])

    rotated = route.project((0.0, 2.0))
    assert rotated.status == "ok"
    assert rotated.projected_point == pytest.approx((1.0, 1.0))
    assert rotated.arc_length_m == pytest.approx(math.sqrt(2.0))
    assert rotated.lateral_offset_m == pytest.approx(math.sqrt(2.0))

    corner = route.project((2.0, 2.0))
    assert corner.status == "ok"
    assert corner.projected_point == pytest.approx((2.0, 2.0))
    assert corner.arc_length_m == pytest.approx(2.0 * math.sqrt(2.0))


def test_projection_fails_closed_for_invalid_queries() -> None:
    """Malformed or non-finite query points must not become route progress."""
    route = RouteGeometry([(0, 0), (1, 0)])

    projection = route.project((math.nan, 0.0))
    assert projection.status == "invalid_query"
    assert not projection.is_valid
    assert projection.arc_length_m is None

    with pytest.raises(ValueError, match="at least two distinct"):
        RouteGeometry([(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="finite values"):
        RouteGeometry([(0, 0), (math.inf, 1)])


def test_projection_marks_self_intersection_as_ambiguous() -> None:
    """Equal-distance branches must not silently select one route arc."""
    route = RouteGeometry([(0, 0), (2, 2), (0, 2), (2, 0)])

    projection = route.project((1.0, 1.0))
    assert projection.status == "ambiguous"
    assert not projection.is_valid
    assert projection.arc_length_m is None
    assert projection.segment_index is None


def test_projection_marks_equal_distance_parallel_branches_as_ambiguous() -> None:
    """A query equidistant from separate route branches must fail closed."""
    route = RouteGeometry([(0, 0), (2, 0), (2, 2), (0, 2)])

    projection = route.project((1.0, 1.0))
    assert projection.status == "ambiguous"
    assert projection.distance_m == pytest.approx(1.0)
