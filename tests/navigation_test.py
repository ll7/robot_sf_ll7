"""Module navigation_test auto-generated docstring."""

from robot_sf.nav.navigation import RouteNavigator


def west_east_route():
    """West east route.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return [(0, 1), (2, 1), (4, 1), (6, 1)]


def test_can_detect_when_waypoint_reached():
    """Test can detect when waypoint reached.

    Returns:
        Any: Auto-generated placeholder description.
    """
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((0.5, 1.5))
    assert navi.reached_waypoint


def test_can_detect_when_waypoint_not_reached():
    """Test can detect when waypoint not reached.

    Returns:
        Any: Auto-generated placeholder description.
    """
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((-0.5, -1.5))
    assert not navi.reached_waypoint


def test_can_detect_when_destination_reached():
    """Test can detect when destination reached.

    Returns:
        Any: Auto-generated placeholder description.
    """
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((6.5, 1.5))
    assert navi.reached_destination


def test_can_detect_when_destination_not_reached():
    """Test can detect when destination not reached.

    Returns:
        Any: Auto-generated placeholder description.
    """
    route = west_east_route()
    navi = RouteNavigator(route)
    navi.update_position((0.5, 1.5))
    assert not navi.reached_destination


def test_can_drive_route_from_start_to_finish():
    """Test can drive route from start to finish.

    Returns:
        Any: Auto-generated placeholder description.
    """
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
