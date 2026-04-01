"""TODO docstring. Document this module."""

from math import dist

import pytest

from robot_sf.nav.navigation import RouteNavigator


def west_east_route():
    """TODO docstring. Document this function."""
    return [(0, 1), (2, 1), (4, 1), (6, 1)]


def test_can_detect_when_waypoint_reached():
    """TODO docstring. Document this function."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((0.5, 1.5))
    assert navi.reached_waypoint


def test_can_detect_when_waypoint_not_reached():
    """TODO docstring. Document this function."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((-0.5, -1.5))
    assert not navi.reached_waypoint


def test_can_detect_when_destination_reached():
    """TODO docstring. Document this function."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((6.5, 1.5))
    assert navi.reached_destination


def test_can_detect_when_destination_not_reached():
    """TODO docstring. Document this function."""
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((0.5, 1.5))
    assert not navi.reached_destination


def test_can_drive_route_from_start_to_finish():
    """TODO docstring. Document this function."""
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
