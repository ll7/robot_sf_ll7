"""Tests for RouteNavigator waypoint/destination detection and route rebasing."""

from math import dist

import pytest

from robot_sf.nav.map_config import (
    GOAL_COMPLETION_POLICY_GOAL_ZONE_ENTRY_V1,
    GOAL_COMPLETION_POLICY_WAYPOINT_RADIUS_V1,
)
from robot_sf.nav.navigation import RouteNavigator, sample_route
from robot_sf.nav.svg_map_parser import convert_map


def west_east_route():
    """Return a four-waypoint west-to-east route along y = 1."""
    return [(0, 1), (2, 1), (4, 1), (6, 1)]


def test_can_detect_when_waypoint_reached():
    """A position within the first waypoint's radius sets reached_waypoint."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((0.5, 1.5))
    assert navi.reached_waypoint


def test_can_detect_when_waypoint_not_reached():
    """A position far from the route leaves reached_waypoint false."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((-0.5, -1.5))
    assert not navi.reached_waypoint


def test_can_detect_when_destination_reached():
    """A position near the final waypoint sets reached_destination."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((6.5, 1.5))
    assert navi.completion_policy == GOAL_COMPLETION_POLICY_WAYPOINT_RADIUS_V1
    assert navi.reached_destination


def test_can_detect_when_destination_not_reached():
    """A position near the first waypoint leaves reached_destination false."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((0.5, 1.5))
    assert not navi.reached_destination


def test_can_drive_route_from_start_to_finish():
    """Stepping east along the route reaches every waypoint and then the destination."""
    route = west_east_route()
    navi = RouteNavigator(route)
    step = 0.1
    spawn_pos = (-1.0, 1.2)

    reached_waypoint_count = 0
    for i in range(10_000):
        new_pos = (spawn_pos[0] + i * step, spawn_pos[1])
        navi.update_position(new_pos)
        reached_waypoint_count += 1 if navi.reached_waypoint else 0
        if navi.reached_destination:
            break

    assert navi.reached_destination
    assert reached_waypoint_count == len(route)


def test_new_route_rebases_initial_handoff_target_when_spawn_starts_inside_threshold():
    """Route reset should push the first active target outside the completion radius."""
    route = [(2.25, 10.0), (9.0, 10.0)]
    spawn = (2.9696724038395113, 10.957236684465528)

    navi = RouteNavigator(proximity_threshold=2.0)
    navi.new_route(route, start_pos=spawn)

    assert dist(navi.current_waypoint, spawn) > navi.proximity_threshold
    assert navi.current_waypoint[1] == pytest.approx(10.0)
    assert spawn[0] < navi.current_waypoint[0] < route[-1][0]


def test_new_route_rebases_from_endpoint_clamped_projection():
    """Endpoint-clamped projections should still exit the threshold on the first segment."""
    route = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    spawn = (-1.0, 1.0)

    navi = RouteNavigator(proximity_threshold=2.0)
    navi.new_route(route, start_pos=spawn)

    assert navi.current_waypoint[1] == pytest.approx(0.0)
    assert 0.0 < navi.current_waypoint[0] < 1.0
    assert dist(navi.current_waypoint, spawn) == pytest.approx(
        navi.proximity_threshold + 1e-6,
        abs=1e-5,
    )


def test_new_route_keeps_first_waypoint_when_spawn_starts_outside_threshold():
    """Route reset should preserve the original first waypoint when no rebasing is needed."""
    route = [(2.0, 1.0), (4.0, 1.0)]
    spawn = (-1.0, 1.2)

    navi = RouteNavigator(proximity_threshold=1.0)
    navi.new_route(route, start_pos=spawn)

    assert navi.current_waypoint == route[0]


def test_initial_orientation_uses_spawn_reference_for_single_waypoint_routes():
    """Single-waypoint routes should still produce a heading after spawn-aware rebasing."""
    navi = RouteNavigator(proximity_threshold=2.0)
    navi.new_route([(4.0, 1.0)], start_pos=(1.0, 1.0))

    assert navi.initial_orientation == pytest.approx(0.0)


@pytest.mark.parametrize("position", [(4.0, 1.0), (4.1, 1.9)])
def test_goal_zone_entry_completion_accepts_rectangle_boundary_and_interior(position):
    """The versioned zone policy completes on entry, even away from the sampled point."""
    goal_zone = ((4.0, 0.0), (6.0, 0.0), (6.0, 2.0))
    navi = RouteNavigator(
        [(0.0, 0.0), (10.0, 10.0)],
        proximity_threshold=0.1,
        completion_policy=GOAL_COMPLETION_POLICY_GOAL_ZONE_ENTRY_V1,
        goal_zone=goal_zone,
    )

    navi.update_position(position)

    assert navi.reached_destination


def test_goal_zone_entry_completion_rejects_outside_position():
    """The zone policy does not complete until the robot enters its bound rectangle."""
    goal_zone = ((4.0, 0.0), (6.0, 0.0), (6.0, 2.0))
    navi = RouteNavigator(
        [(0.0, 0.0), (10.0, 10.0)],
        completion_policy=GOAL_COMPLETION_POLICY_GOAL_ZONE_ENTRY_V1,
        goal_zone=goal_zone,
    )

    navi.update_position((3.9, 1.0))

    assert not navi.reached_destination


def test_goal_completion_policy_fails_closed_for_unknown_or_missing_goal_zone():
    """Unknown policies and unbound goal-zone policies must not silently downgrade."""
    with pytest.raises(ValueError, match="Unknown goal_completion_policy"):
        RouteNavigator(completion_policy="goal_zone_entry_v9")

    with pytest.raises(ValueError, match="requires a goal_zone"):
        RouteNavigator(
            [(0.0, 0.0), (1.0, 1.0)],
            completion_policy=GOAL_COMPLETION_POLICY_GOAL_ZONE_ENTRY_V1,
        )


def test_sampled_route_binds_source_goal_zone_to_navigator_metadata():
    """A sampled route carries the exact source goal-zone and route identifiers."""
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")
    route = sample_route(map_def, spawn_id=0)
    navigator = RouteNavigator(
        completion_policy=GOAL_COMPLETION_POLICY_GOAL_ZONE_ENTRY_V1,
    )
    navigator.new_route(
        route[1:],
        start_pos=route[0],
        goal_zone=route.goal_zone,
        spawn_id=route.spawn_id,
        goal_id=route.goal_id,
    )

    metadata = navigator.completion_metadata()

    assert metadata["policy"] == GOAL_COMPLETION_POLICY_GOAL_ZONE_ENTRY_V1
    assert metadata["route_binding"] == {
        "spawn_id": 0,
        "goal_id": 0,
        "goal_zone": [[17.0, 4.0], [18.5, 4.0], [18.5, 5.5]],
    }
