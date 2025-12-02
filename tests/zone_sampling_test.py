"""Module zone_sampling_test auto-generated docstring."""

from robot_sf.ped_npc.ped_zone import sample_zone


def is_within_zone(p):
    """Is within zone.

    Args:
        p: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return 0 <= p[0] <= 10 and 0 <= p[1] <= 10


def prepare_zones():
    """Prepare zones.

    Returns:
        Any: Auto-generated placeholder description.
    """
    zone_topleft = ((0, 0), (0, 10), (10, 10))
    zone_botleft = ((0, 10), (0, 0), (10, 0))
    zone_botright = ((0, 0), (10, 0), (10, 10))
    zone_topright = ((10, 0), (10, 10), (0, 10))
    zones = [zone_topleft, zone_botleft, zone_botright, zone_topright]
    zones_rev = [(z[2], z[1], z[0]) for z in zones]
    return list(zip(zones, zones_rev, strict=False))


def test_must_not_spawn_outside_of_topleft_zone():
    """Test must not spawn outside of topleft zone.

    Returns:
        Any: Auto-generated placeholder description.
    """
    (zone, zone_rev), _, _, _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)


def test_must_not_spawn_outside_of_botleft_zone():
    """Test must not spawn outside of botleft zone.

    Returns:
        Any: Auto-generated placeholder description.
    """
    _, (zone, zone_rev), _, _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)


def test_must_not_spawn_outside_of_botright_zone():
    """Test must not spawn outside of botright zone.

    Returns:
        Any: Auto-generated placeholder description.
    """
    _, _, (zone, zone_rev), _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)


def test_must_not_spawn_outside_of_topright_zone():
    """Test must not spawn outside of topright zone.

    Returns:
        Any: Auto-generated placeholder description.
    """
    _, _, _, (zone, zone_rev) = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(is_within_zone(p) for p in points)
    assert all(is_within_zone(p) for p in points_rev)
