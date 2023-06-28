from robot_sf.ped_npc.ped_spawn_generator import sample_zone


def is_within_zone(p):
    return 0 <= p[0] <= 10 and 0 <= p[1] <= 10


def prepare_zones():
    zone_topleft = ((0, 0), (0, 10), (10, 10))
    zone_botleft = ((0, 10), (0, 0), (10, 0))
    zone_botright = ((0, 0), (10, 0), (10, 10))
    zone_topright = ((10, 0), (10, 10), (0, 10))
    zones = [zone_topleft, zone_botleft, zone_botright, zone_topright]
    zones_rev = [(z[2], z[1], z[0]) for z in zones]
    return list(zip(zones, zones_rev))


def test_must_not_spawn_outside_of_topleft_zone():
    (zone, zone_rev), _, _, _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(map(lambda p: is_within_zone(p), points))
    assert all(map(lambda p: is_within_zone(p), points_rev))


def test_must_not_spawn_outside_of_botleft_zone():
    _, (zone, zone_rev), _, _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(map(lambda p: is_within_zone(p), points))
    assert all(map(lambda p: is_within_zone(p), points_rev))


def test_must_not_spawn_outside_of_botright_zone():
    _, _, (zone, zone_rev), _ = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(map(lambda p: is_within_zone(p), points))
    assert all(map(lambda p: is_within_zone(p), points_rev))


def test_must_not_spawn_outside_of_topright_zone():
    _, _, _, (zone, zone_rev) = prepare_zones()

    points = [sample_zone(zone, 1)[0] for i in range(1000)]
    points_rev = [sample_zone(zone_rev, 1)[0] for i in range(1000)]

    assert all(map(lambda p: is_within_zone(p), points))
    assert all(map(lambda p: is_within_zone(p), points_rev))
