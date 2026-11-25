"""Independent analytic surface-gap, contact, and selector contracts."""

from math import pi

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.footprint import CapsuleFootprint, segment_segment_distance


def test_t60_dimensions_and_hand_computed_gaps():
    """Total length includes the rounded caps; side/end clearances are distinct."""
    body = CapsuleFootprint(1.183, 0.478)
    pose = ((0.0, 0.0), 0.0)
    assert body.radius == pytest.approx(0.239)
    assert body.half_segment == pytest.approx(0.3525)
    assert body.covering_radius == pytest.approx(0.5915)
    assert body.circle_gap(pose, (0.0, 1.0), 0.4) == pytest.approx(0.361)
    assert body.circle_gap(pose, (1.0, 0.0), 0.4) == pytest.approx(0.0085)
    assert body.segment_gap(pose, ((-2.0, 1.0), (2.0, 1.0))) == pytest.approx(0.761)
    assert body.segment_gap(pose, ((0.0, -1.0), (0.0, 1.0))) == pytest.approx(-0.239)


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (((-1.0, 0.0), (1.0, 0.0)), ((0.0, -1.0), (0.0, 1.0)), 0.0),
        (((-1.0, 0.0), (1.0, 0.0)), ((-0.5, 0.0), (2.0, 0.0)), 0.0),
        (((-1.0, 0.0), (1.0, 0.0)), ((2.0, 0.0), (3.0, 0.0)), 1.0),
        (((-1.0, 0.0), (1.0, 0.0)), ((-1.0, 2.0), (1.0, 2.0)), 2.0),
        (((0.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), (1.0, 0.0)), 1.0),
    ],
)
def test_segment_distances_cross_parallel_overlap_and_degenerate(first, second, expected):
    """A closest-endpoint-only algorithm misses crossing interiors."""
    assert segment_segment_distance(first, second) == pytest.approx(expected)
    assert segment_segment_distance(second, first) == pytest.approx(expected)


def _occupancy(body, *, pedestrians=(), walls=(), dynamic=()):
    return ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (5.0, 5.0),
        get_agent_pose=lambda: ((5.0, 5.0), 0.0),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: walls,
        get_pedestrian_coords=lambda: pedestrians,
        get_dynamic_objects=lambda: dynamic,
        capsule=body,
    )


def test_contact_includes_tangency_and_rejects_positive_surface_gap():
    """Touching side/caps count as contact; a positive gap does not."""
    body = CapsuleFootprint(2.0, 1.0)
    assert _occupancy(body, walls=[((4.0, 5.5), (6.0, 5.5))]).is_obstacle_collision
    assert not _occupancy(body, walls=[((4.0, 5.5001), (6.0, 5.5001))]).is_obstacle_collision
    assert _occupancy(body, pedestrians=[(6.4, 5.0)]).is_pedestrian_collision
    assert not _occupancy(body, pedestrians=[(6.4001, 5.0)]).is_pedestrian_collision
    assert _occupancy(body, walls=[((6.0, 5.0), (6.0, 5.0))]).is_obstacle_collision
    assert not _occupancy(body, walls=[((6.001, 5.0), (6.001, 5.0))]).is_obstacle_collision
    assert _occupancy(body, dynamic=[((6.0, 5.0), 0.1)]).is_dynamic_collision


def test_ego_pedestrian_capsule_collision_is_symmetric():
    """The ego pedestrian must see the same side/end body as the robot."""
    pose = [((5.0, 5.0), 0.0)]
    ego = EgoPedContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (5.0, 5.7),
        get_goal_coords=lambda: (9.0, 9.0),
        get_obstacle_coords=lambda: [],
        get_pedestrian_coords=lambda: [],
        agent_radius=0.4,
        get_enemy_coords=lambda: pose[0][0],
        get_enemy_pose=lambda: pose[0],
        enemy_capsule=CapsuleFootprint(1.183, 0.478),
        enemy_radius=0.64,
    )
    assert ego.is_agent_agent_collision is False
    pose[0] = ((5.0, 5.0), pi / 2)
    assert ego.is_agent_agent_collision is True


def test_capsule_requires_heading_and_conservative_proxy():
    """Missing pose and undersized proxy cannot silently omit contact geometry."""
    with pytest.raises(ValueError, match="radius must cover"):
        BicycleDriveSettings(radius=0.239, capsule={"length": 1.183, "width": 0.478})
    with pytest.raises(ValueError, match="get_agent_pose"):
        ContinuousOccupancy(
            10,
            10,
            lambda: (5, 5),
            lambda: (9, 9),
            lambda: [],
            lambda: [],
            capsule=CapsuleFootprint(1.183, 0.478),
        )
    with pytest.raises(ValueError, match="get_enemy_pose"):
        EgoPedContinuousOccupancy(
            10,
            10,
            lambda: (5, 5),
            lambda: (9, 9),
            lambda: [],
            lambda: [],
            enemy_capsule=CapsuleFootprint(1.183, 0.478),
        )


@pytest.mark.parametrize("seed", range(1001, 1201))
def test_surface_distances_match_independent_shapely_segments(seed):
    """Cross-check translated/rotated geometry against GEOS on authorized dev seeds."""
    rng = np.random.default_rng(seed)
    body = CapsuleFootprint(1.183, 0.478)
    x, y, heading = rng.uniform(-3, 3, 3)
    pose = ((x, y), heading)
    # Independent endpoint construction, without using CapsuleFootprint.axis.
    offset = np.array([np.cos(heading), np.sin(heading)]) * 0.3525
    line = LineString([np.array([x, y]) - offset, np.array([x, y]) + offset])
    point = rng.uniform(-3, 3, 2)
    segment = rng.uniform(-3, 3, (2, 2))
    assert body.circle_gap(pose, point, 0.4) == pytest.approx(line.distance(Point(point)) - 0.639)
    assert body.segment_gap(pose, segment) == pytest.approx(
        line.distance(LineString(segment)) - 0.239
    )


def test_capsule_and_disc_dynamic_checks_use_true_caps_not_proxy():
    """Mixed bodies agree on contact regardless of which one is queried."""
    body = CapsuleFootprint(2.0, 1.0)
    pose = ((5.0, 5.0), 0.0)
    disc = ContinuousOccupancy(
        10,
        10,
        lambda: (5.0, 6.0),
        lambda: (9.0, 9.0),
        lambda: [],
        lambda: [],
        agent_radius=0.4,
        get_dynamic_capsules=lambda: [(pose, body)],
    )
    assert not disc.is_dynamic_collision
    touching = ContinuousOccupancy(
        10,
        10,
        lambda: (5.0, 5.9),
        lambda: (9.0, 9.0),
        lambda: [],
        lambda: [],
        agent_radius=0.4,
        get_dynamic_capsules=lambda: [(pose, body)],
    )
    assert touching.is_dynamic_collision


def test_circular_capsule_and_boundary_tangency():
    """Equal dimensions degenerate to a disc without losing bounds contact."""
    body = CapsuleFootprint(1.0, 1.0)
    assert body.circle_gap(((0.0, 0.0), 1.2), (1.0, 0.0), 0.4) == pytest.approx(0.1)
    occ = _occupancy(body)
    assert not occ.is_in_bounds(0.5, 5.0)
    assert occ.is_in_bounds(0.5001, 5.0)
