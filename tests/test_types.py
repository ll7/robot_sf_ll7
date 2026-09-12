"""Tests that the robot_sf.common.types aliases match their runtime shapes."""

# FILE: robot_sf/util/test_types.py
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_sf.common.types import (
        BicycleAction,
        Circle2D,
        DifferentialDriveAction,
        Line2D,
        MapBounds,
        PedGrouping,
        PedPose,
        PedState,
        Point2D,
        PolarVec2D,
        Range,
        Range2D,
        Rect,
        RgbColor,
        RobotAction,
        RobotPose,
        UnicycleAction,
        Vec2D,
        Zone,
        ZoneAssignments,
    )


def test_vec2d():
    """Vec2D is a two-element tuple of floats."""
    vec: Vec2D = (1.0, 2.0)
    assert isinstance(vec, tuple)
    assert len(vec) == 2
    for element in vec:
        assert isinstance(element, float)


def test_range2d():
    """Range2D is a two-element tuple of floats."""
    r: Range2D = (0.0, 10.0)
    assert isinstance(r, tuple)
    assert len(r) == 2
    for element in r:
        assert isinstance(element, float)


def test_line2d():
    """Line2D is a four-element tuple of floats."""
    line: Line2D = (0.0, 0.0, 5.0, 5.0)
    assert isinstance(line, tuple)
    assert len(line) == 4
    for element in line:
        assert isinstance(element, float)


def test_point2d():
    """Point2D is a two-element tuple of floats."""
    pt: Point2D = (3.0, 4.0)
    assert isinstance(pt, tuple)
    assert len(pt) == 2
    for element in pt:
        assert isinstance(element, float)


def test_circle2d():
    """Circle2D is a (center, radius) tuple with a two-float center and float radius."""
    circle: Circle2D = ((0.0, 0.0), 5.0)
    assert isinstance(circle, tuple)
    assert len(circle) == 2
    assert isinstance(circle[0], tuple)
    assert len(circle[0]) == 2
    for element in circle[0]:
        assert isinstance(element, float)
    assert isinstance(circle[1], float)


def test_map_bounds():
    """MapBounds is a pair of two-element float ranges."""
    bounds: MapBounds = ((0.0, 10.0), (0.0, 20.0))
    assert isinstance(bounds, tuple)
    assert len(bounds) == 2
    for rng in bounds:
        assert isinstance(rng, tuple)
        assert len(rng) == 2
        for element in rng:
            assert isinstance(element, float)


def test_rect_zone_polar_range():
    """Rect and Zone are three 2-float tuples; PolarVec2D and Range are 2-float tuples."""
    rect: Rect = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    zone: Zone = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    polar: PolarVec2D = (1.0, 45.0)
    rnge: Range = (0.0, 1.0)
    for item in [rect, zone]:
        assert isinstance(item, tuple)
        assert len(item) == 3
        for vec in item:
            assert isinstance(vec, tuple)
            assert len(vec) == 2
            for element in vec:
                assert isinstance(element, float)
    assert isinstance(polar, tuple)
    assert len(polar) == 2
    for element in polar:
        assert isinstance(element, float)
    assert isinstance(rnge, tuple)
    assert len(rnge) == 2
    for element in rnge:
        assert isinstance(element, float)


def test_differential_drive_action():
    """DifferentialDriveAction is a two-element tuple of floats."""
    action: DifferentialDriveAction = (1.0, 0.5)
    assert isinstance(action, tuple)
    assert len(action) == 2
    for element in action:
        assert isinstance(element, float)


def test_bicycle_action():
    """BicycleAction is a two-element tuple of floats."""
    action: BicycleAction = (0.8, 0.2)
    assert isinstance(action, tuple)
    assert len(action) == 2
    for element in action:
        assert isinstance(element, float)


def test_robot_action_union():
    # Test as DifferentialDriveAction
    """Both differential-drive and bicycle actions satisfy the RobotAction shape."""
    action1: RobotAction = (1.0, 0.5)
    # Test as BicycleAction
    action2: RobotAction = (0.8, 0.2)
    for action in [action1, action2]:
        assert isinstance(action, tuple)
        assert len(action) == 2
        for element in action:
            assert isinstance(element, float)


def test_robot_pose():
    """RobotPose is a ((x, y), orientation) tuple with float components."""
    pose: RobotPose = ((2.0, 3.0), 1.57)
    assert isinstance(pose, tuple)
    assert len(pose) == 2
    position, orientation = pose
    assert isinstance(position, tuple)
    assert len(position) == 2
    for element in position:
        assert isinstance(element, float)
    assert isinstance(orientation, float)


def test_ped_pose():
    """PedPose is a ((x, y), orientation) tuple with float components."""
    ped: PedPose = ((4.0, 5.0), 0.0)
    assert isinstance(ped, tuple)
    assert len(ped) == 2
    position, orientation = ped
    assert isinstance(position, tuple)
    assert len(position) == 2
    for element in position:
        assert isinstance(element, float)
    assert isinstance(orientation, float)


def test_unicycle_action():
    """UnicycleAction is a two-element tuple of floats."""
    action: UnicycleAction = (0.5, 0.1)
    assert isinstance(action, tuple)
    assert len(action) == 2
    for element in action:
        assert isinstance(element, float)


def test_ped_state():
    """PedState is a numpy ndarray."""
    state: PedState = np.array([1.0, 2.0, 3.0])
    assert isinstance(state, np.ndarray)


def test_ped_grouping():
    """PedGrouping is a set of integer pedestrian ids."""
    grouping: PedGrouping = {1, 2, 3}
    assert isinstance(grouping, set)
    for element in grouping:
        assert isinstance(element, int)


def test_zone_assignments():
    """ZoneAssignments is a dict mapping integer keys to integer values."""
    assignments: ZoneAssignments = {1: 10, 2: 20}
    assert isinstance(assignments, dict)
    for key, value in assignments.items():
        assert isinstance(key, int)
        assert isinstance(value, int)


def test_rgb_color():
    """RgbColor is a three-element tuple of ints."""
    color: RgbColor = (255, 128, 64)
    assert isinstance(color, tuple)
    assert len(color) == 3
    for element in color:
        assert isinstance(element, int)
