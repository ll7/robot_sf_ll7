"""Tests that sample_zone keeps sampled points inside triangular zones."""

from robot_sf.ped_npc.ped_zone import sample_zone


def is_within_zone(p):
    """Return whether point p lies in the [0, 10] x [0, 10] square.

    Args:
        p: Two-element point as (x, y) coordinates.
    """
    return 0 <= p[0] <= 10 and 0 <= p[1] <= 10


def prepare_zones():
    """Build the four triangular zones and reversed-winding variants used by these tests."""
    zone_topleft = ((0, 0), (0, 10), (10, 10))
    zone_botleft = ((0, 10), (0, 0), (10, 0))
    zone_botright = ((0, 0), (10, 0), (10, 10))
    zone_topright = ((10, 0), (10, 10), (0, 10))
    zones = [zone_topleft, zone_botleft, zone_botright, zone_topright]
    zones_rev = [(z[2], z[1], z[0]) for z in zones]
    return list(zip(zones, zones_rev, strict=False))


def test_must_not_spawn_outside_of_topleft_zone():
    """Sample 1000 points from the top-left zone and its reverse and checks all stay in bounds."""
    (zone, zone_rev), _, _, _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)


def test_must_not_spawn_outside_of_botleft_zone():
    """Sample 1000 points from the bottom-left zone and its reverse and checks all stay in bounds."""
    _, (zone, zone_rev), _, _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)


def test_must_not_spawn_outside_of_botright_zone():
    """Sample 1000 points from the bottom-right zone and its reverse and checks all stay in bounds."""
    _, _, (zone, zone_rev), _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)


def test_must_not_spawn_outside_of_topright_zone():
    """Sample 1000 points from the top-right zone and its reverse and checks all stay in bounds."""
    _, _, _, (zone, zone_rev) = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)
