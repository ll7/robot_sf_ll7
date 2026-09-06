"""Focused contract tests for deterministic route geometry."""

from __future__ import annotations

import json
import math

import pytest

from robot_sf.nav.global_route import (
    RouteGeometry,
    RouteProjectionHint,
    RouteProjectionTracker,
)


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


def test_projection_fails_closed_for_integer_conversion_overflow() -> None:
    """Oversized integers must follow the invalid-input contract, not crash."""
    overflowing = 10**400
    route = RouteGeometry([(0, 0), (5, 0), (10, 0)])

    projection = route.project((overflowing, 0))
    assert projection.status == "invalid_query"

    tracker = RouteProjectionTracker(route)
    assert tracker.project((1.0, 0.0), step=0).is_valid
    failed = tracker.project((overflowing, 0), step=1)
    assert failed.status == "invalid_query"
    assert failed.failure_count == 1
    assert tracker.previous_s_m == pytest.approx(1.0)

    with pytest.raises(ValueError, match="finite"):
        RouteGeometry([(0, 0), (overflowing, 0)])
    with pytest.raises(ValueError, match="finite"):
        RouteProjectionHint(previous_s_m=overflowing)

    snapshot = tracker.snapshot()
    with pytest.raises(ValueError, match="max_forward_jump_m"):
        RouteProjectionTracker.restore(route, dict(snapshot, max_forward_jump_m=overflowing))


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


def test_hinted_projection_rejects_forward_jump_to_nearby_parallel_branch() -> None:
    """Continuity bounds must prevent a nearer later parallel branch from winning."""
    route = RouteGeometry([(0, 0), (10, 0), (10, 1), (0, 1)])
    query = (8.5, 0.9)

    unconstrained = route.project(query)
    hinted = route.project(
        query,
        hint=RouteProjectionHint(
            previous_s_m=8.0,
            previous_segment_index=0,
            max_forward_jump_m=0.75,
            max_backtrack_m=0.5,
        ),
    )

    assert unconstrained.status == "ok"
    assert unconstrained.segment_index == 2
    assert unconstrained.arc_length_m == pytest.approx(12.5)
    assert hinted.status == "ok"
    assert hinted.segment_index == 0
    assert hinted.arc_length_m == pytest.approx(8.5)


def test_hinted_projection_stays_on_the_continuous_route_branch() -> None:
    """A bounded hint should reject a later self-intersection branch."""
    route = RouteGeometry([(0, 0), (2, 2), (0, 2), (2, 0)])
    hint = RouteProjectionHint(
        previous_s_m=0.5,
        previous_segment_index=0,
        max_forward_jump_m=1.5,
        max_backtrack_m=0.5,
    )

    projection = route.project((1.0, 1.0), hint=hint)

    assert projection.status == "ok"
    assert projection.segment_index == 0
    assert projection.arc_length_m == pytest.approx(math.sqrt(2.0))


def test_hinted_projection_reports_discontinuity_without_overclaiming_progress() -> None:
    """A query outside the continuity window must not return route progress."""
    route = RouteGeometry([(0, 0), (2, 0)])
    hint = RouteProjectionHint(
        previous_s_m=0.0,
        max_forward_jump_m=0.5,
        max_backtrack_m=0.0,
    )

    projection = route.project((2.0, 0.0), hint=hint)

    assert projection.status == "discontinuous"
    assert projection.projected_point is None
    assert projection.arc_length_m is None


def test_hint_tolerance_does_not_expand_the_continuity_window() -> None:
    """Tie tolerance must not permit motion outside explicit continuity bounds."""
    route = RouteGeometry([(0, 0), (10, 0)], tie_tolerance_m=0.25)
    hint = RouteProjectionHint(
        previous_s_m=0.0,
        max_forward_jump_m=0.0,
        max_backtrack_m=0.0,
    )

    projection = route.project((0.1, 0.0), hint=hint)

    assert projection.status == "discontinuous"


def test_route_tracker_preserves_last_valid_state_and_rejects_bad_step_order() -> None:
    """Failures and duplicate/out-of-order steps must not rewrite continuity."""
    route = RouteGeometry([(0, 0), (10, 0)])
    tracker = RouteProjectionTracker(route, max_forward_jump_m=2.0, max_backtrack_m=0.5)

    first = tracker.project((1.0, 0.0), step=0)
    failed = tracker.project((5.0, 0.0), step=1)
    duplicate = tracker.project((1.0, 0.0), step=1)
    out_of_order = tracker.project((1.0, 0.0), step=0)
    recovered = tracker.project((1.5, 0.0), step=2)

    assert first.is_valid
    assert failed.status == "discontinuous"
    assert failed.failure_count == 1
    assert tracker.previous_s_m == pytest.approx(1.5)
    assert duplicate.status == "duplicate_step"
    assert duplicate.failure_count == 1
    assert out_of_order.status == "out_of_order"
    assert recovered.is_valid
    assert recovered.failure_count == 0


def test_route_tracker_route_change_resets_continuity_explicitly() -> None:
    """A route hash change must clear the prior projection before the next step."""
    tracker = RouteProjectionTracker(RouteGeometry([(0, 0), (10, 0)]))
    assert tracker.project((1.0, 0.0), step=4).is_valid

    reset = tracker.update_route(RouteGeometry([(0, 0), (0, 10)]))
    result = tracker.project((0.0, 1.0), step=0)

    assert reset is not None
    assert reset.status == "reset"
    assert reset.step is None
    assert result.is_valid
    assert result.last_reset_reason == "route_changed"
    assert tracker.previous_s_m == pytest.approx(1.0)

    manual_reset = tracker.reset()
    assert manual_reset.status == "reset"
    assert tracker.previous_s_m is None


def test_route_tracker_resets_when_projection_policy_changes() -> None:
    """A same-waypoint route with different projection policy must reset state."""
    route = RouteGeometry([(0, 0), (10, 0)])
    tracker = RouteProjectionTracker(route)
    assert tracker.project((1.0, 0.0), step=4).is_valid

    changed_policy = RouteGeometry([(0, 0), (10, 0)], tie_tolerance_m=0.5)
    reset = tracker.update_route(changed_policy)

    assert changed_policy.route_hash == route.route_hash
    assert reset is not None
    assert reset.status == "reset"
    assert reset.last_reset_reason == "route_changed"
    assert tracker.previous_s_m is None


def test_route_tracker_snapshot_restore_is_json_safe_and_replayable() -> None:
    """A validated snapshot must reproduce the next projection exactly."""
    route = RouteGeometry([(0, 0), (5, 0), (10, 0)])
    tracker = RouteProjectionTracker(route)
    tracker.project((1.0, 0.0), step=3)
    snapshot = tracker.snapshot()
    restored = RouteProjectionTracker.restore(route, snapshot)

    assert json.loads(json.dumps(snapshot)) == snapshot
    assert restored.snapshot() == snapshot
    assert restored.project((2.0, 0.0), step=4) == tracker.project((2.0, 0.0), step=4)

    invalid_hash = dict(snapshot, route_hash="changed")
    with pytest.raises(ValueError, match="route_hash"):
        RouteProjectionTracker.restore(route, invalid_hash)

    invalid_number = dict(snapshot, previous_s_m=math.nan)
    with pytest.raises(ValueError, match="previous_s_m"):
        RouteProjectionTracker.restore(route, invalid_number)

    invalid_segment = dict(snapshot, previous_segment_index=99)
    with pytest.raises(ValueError, match="previous_segment_index"):
        RouteProjectionTracker.restore(route, invalid_segment)

    invalid_pair = dict(snapshot, previous_s_m=6.0, previous_segment_index=0)
    with pytest.raises(ValueError, match="does not belong"):
        RouteProjectionTracker.restore(route, invalid_pair)

    invalid_failure_state = dict(snapshot, last_step=None, failure_count=1)
    with pytest.raises(ValueError, match="last_step"):
        RouteProjectionTracker.restore(route, invalid_failure_state)

    changed_policy = RouteGeometry([(0, 0), (5, 0), (10, 0)], tie_tolerance_m=0.5)
    with pytest.raises(ValueError, match="tie_tolerance_m"):
        RouteProjectionTracker.restore(changed_policy, snapshot)
