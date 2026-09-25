"""Unit tests for clearance-aware spawns and the respawn robot exclusion (issue #9725)."""

from __future__ import annotations

from math import dist
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.prepared import prep

from robot_sf.benchmark.spawn_validity import build_spawn_validity, record_has_spawn_overlap
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.navigation import sample_route
from robot_sf.nav.spawn_clearance import (
    SPAWN_CLEARANCE_MARGIN_M,
    relocate_overlapping_pedestrians,
    robot_obstacle_clearance,
    robot_start_exclusions,
)
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.ped_behavior import FollowRouteBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.ped_npc.ped_zone import sample_zone

MAPS = Path(__file__).resolve().parents[2] / "maps" / "svg_maps"
ROBOT_RADIUS = 1.0
PED_RADIUS = 0.4


@pytest.fixture(scope="module")
def corridor_map():
    """Return the head-on corridor map (walls at y=2 and y=38, x=2 and x=30)."""
    return convert_map(str(MAPS / "classic_head_on_corridor.svg"))


def test_sample_zone_exclusions_keep_draws_for_valid_candidates() -> None:
    """A candidate that clears the exclusions is the same sample as without them."""
    zone = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    far_away = [prep(Point(100.0, 100.0).buffer(1.0))]
    plain = sample_zone(zone, 3, rng=np.random.default_rng(7))
    guarded = sample_zone(zone, 3, rng=np.random.default_rng(7), exclusions=far_away)
    assert guarded == plain


def test_sample_zone_exclusions_reject_blocked_candidates() -> None:
    """No sample may land inside an exclusion geometry."""
    zone = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    blocker = Point(8.0, 2.0).buffer(3.0)
    samples = sample_zone(zone, 50, rng=np.random.default_rng(3), exclusions=[prep(blocker)])
    assert len(samples) == 50
    assert not any(blocker.intersects(Point(p)) for p in samples)


def test_robot_start_exclusions_keep_footprint_off_walls(corridor_map) -> None:
    """Samples from a zone touching the wall keep robot radius plus margin of clearance."""
    zone_on_wall = ((12.0, 2.1), (16.0, 2.1), (16.0, 6.1))
    exclusions = robot_start_exclusions(corridor_map, ROBOT_RADIUS)
    samples = sample_zone(
        zone_on_wall,
        200,
        rng=np.random.default_rng(116),
        max_attempts_per_point=50,
        exclusions=exclusions,
    )
    clearances = [robot_obstacle_clearance(corridor_map, p, ROBOT_RADIUS) for p in samples]
    assert min(clearances) >= SPAWN_CLEARANCE_MARGIN_M - 1e-9


def test_robot_start_fails_loudly_when_zone_cannot_fit_robot(corridor_map) -> None:
    """A spawn zone entirely within the wall clearance raises instead of spawning in the wall."""
    zone_in_wall_band = ((12.0, 2.1), (16.0, 2.1), (16.0, 2.9))
    with pytest.raises(RuntimeError, match="clearance"):
        sample_zone(
            zone_in_wall_band,
            1,
            rng=np.random.default_rng(0),
            exclusions=robot_start_exclusions(corridor_map, ROBOT_RADIUS),
        )


def test_sample_route_with_radius_clears_walls(corridor_map) -> None:
    """Robot starts from the map spawn zone always clear the corridor walls."""
    np.random.seed(0)
    for _ in range(100):
        route = sample_route(corridor_map, 0, robot_radius=ROBOT_RADIUS)
        clearance = robot_obstacle_clearance(corridor_map, route[0], ROBOT_RADIUS)
        assert clearance >= SPAWN_CLEARANCE_MARGIN_M - 1e-9


def test_corridor_map_itself_is_unchanged_and_violates_clearance(corridor_map) -> None:
    """The corridor map keeps its geometry; only the sampler enforces clearance.

    Part of the spawn zone lies within the robot radius of the wall, which is exactly
    why the padded sampler is needed (mechanism C).
    """
    corners = [c for zone in corridor_map.robot_spawn_zones for c in zone]
    assert min(robot_obstacle_clearance(corridor_map, c, ROBOT_RADIUS) for c in corners) < 0.0


def test_station_platform_ped_spawn_zones_avoid_robot_spawn_zone() -> None:
    """The successor station map spawns no route inside the padded robot zone (mechanism B)."""
    from shapely.geometry import Polygon

    map_def = convert_map(
        str(MAPS.parent / "successor_svg_maps" / "classic_station_platform_v2.svg")
    )

    def rect(zone):
        a, b, c = zone
        return Polygon((a, b, c, (a[0] + c[0] - b[0], a[1] + c[1] - b[1])))

    padding = ROBOT_RADIUS + PED_RADIUS
    for robot_zone in map_def.robot_spawn_zones:
        padded = rect(robot_zone).buffer(padding)
        for route in map_def.ped_routes:
            assert not padded.intersects(rect(route.spawn_zone))


def test_relocation_moves_only_overlapping_pedestrians(corridor_map) -> None:
    """Overlapping rows move onto the exclusion circle; clear rows stay bit-identical."""
    robot = ((15.0, 20.0), ROBOT_RADIUS)
    peds = [(15.5, 20.2), (20.0, 20.0), (14.9, 21.0)]
    report = relocate_overlapping_pedestrians(peds, PED_RADIUS, [robot], corridor_map)
    assert sorted(report.relocated) == [0, 2]
    assert report.unresolved == []
    required = ROBOT_RADIUS + PED_RADIUS + SPAWN_CLEARANCE_MARGIN_M
    for row, (_old, new) in report.relocated.items():
        assert dist(new, robot[0]) >= required - 1e-9
        assert robot_obstacle_clearance(corridor_map, new, PED_RADIUS) >= -1e-9
        assert all(
            dist(new, other) >= 2 * PED_RADIUS - 1e-9
            for idx, other in enumerate(peds)
            if idx != row
        )
    first = report.relocated[0][1]
    # The first candidate lies on the ray from the robot through the old position.
    assert np.isclose(np.arctan2(first[1] - 20.0, first[0] - 15.0), np.arctan2(0.2, 0.5))
    again = relocate_overlapping_pedestrians(peds, PED_RADIUS, [robot], corridor_map)
    assert again.relocated == report.relocated


def test_relocation_respects_walls(corridor_map) -> None:
    """A pedestrian pushed toward a wall is rotated to a clear direction instead."""
    robot = ((3.2, 20.0), ROBOT_RADIUS)
    peds = [(2.7, 20.0)]
    report = relocate_overlapping_pedestrians(peds, PED_RADIUS, [robot], corridor_map)
    new = report.relocated[0][1]
    assert robot_obstacle_clearance(corridor_map, new, PED_RADIUS) >= -1e-9


def _route_behavior(robot_xy):
    """Build a one-group route behavior whose spawn zone contains ``robot_xy``."""
    states = np.zeros((2, 7))
    states[:, 0:2] = [(30.0, 0.0), (30.0, 1.0)]
    ped_states = PedestrianStates(lambda: states)
    groups = PedestrianGroupings(ped_states)
    gid = groups.new_group({0, 1})
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(4.0, 2.0), (30.0, 2.0)],
        spawn_zone=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)),
        goal_zone=((29.0, 0.0), (31.0, 0.0), (31.0, 2.0)),
    )
    behavior = FollowRouteBehavior(groups, {gid: route}, [0])
    exclusion = ROBOT_RADIUS + PED_RADIUS + SPAWN_CLEARANCE_MARGIN_M
    behavior.set_robot_exclusion(lambda: [(robot_xy, 0.0)], exclusion)
    return behavior, groups, gid, states, exclusion


def test_respawn_excludes_robot_footprint() -> None:
    """Route-end respawns never land inside the robot exclusion circle."""
    robot_xy = (3.0, 1.0)
    behavior, _groups, gid, states, exclusion = _route_behavior(robot_xy)
    np.random.seed(115)
    for _ in range(200):
        behavior.respawn_group_at_start(gid)
        for row in states[:, 0:2]:
            assert dist(tuple(row), robot_xy) >= exclusion - 1e-9
    assert behavior.respawn_overlap_events == []


def test_respawn_records_event_and_keeps_legacy_sample_when_zone_is_fully_covered() -> None:
    """If the robot covers the whole spawn zone, the first (legacy) sample is kept."""
    behavior, _groups, gid, states, _exclusion = _route_behavior((2.7, 1.3))
    behavior.robot_exclusion_radius = 10.0
    np.random.seed(0)
    legacy = sample_zone(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)), 2)
    np.random.seed(0)
    behavior.respawn_group_at_start(gid)
    assert [tuple(row) for row in states[:, 0:2]] == legacy
    assert len(behavior.respawn_overlap_events) == 1
    assert behavior.respawn_overlap_events[0]["ped_rows"] == [0, 1]
    behavior.reset()
    assert behavior.respawn_overlap_events == []


def test_respawn_keeps_random_stream_when_legacy_sample_is_clear() -> None:
    """A respawn far from the robot is identical to the unguarded one, draws included."""
    behavior, _groups, gid, states, _exclusion = _route_behavior((100.0, 100.0))
    np.random.seed(5)
    legacy = sample_zone(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)), 2)
    after_legacy = np.random.uniform()
    np.random.seed(5)
    behavior.respawn_group_at_start(gid)
    assert [tuple(row) for row in states[:, 0:2]] == legacy
    assert np.random.uniform() == after_legacy


def test_reset_respawn_does_not_use_stale_robot_pose() -> None:
    """Reset-time respawns skip the robot guard; the simulator relocates after sampling."""
    behavior, _groups, _gid, _states, _exclusion = _route_behavior((2.7, 1.3))
    behavior.robot_exclusion_radius = 10.0
    behavior.reset_at_start = True
    behavior.reset()
    assert behavior.respawn_overlap_events == []


def test_spawn_validity_marks_overlap_invalid() -> None:
    """Reset overlap marks the row invalid unless the route was completed."""
    clean = build_spawn_validity({"overlap": False}, [])
    assert clean["invalid_run"] is False and clean["invalid_reason"] is None
    reset = build_spawn_validity({"overlap": True}, [])
    assert reset["invalid_run"] is True and reset["invalid_reason"] == "spawn_overlap"
    unmatched = build_spawn_validity({"overlap": False}, [{"group_id": 0, "ped_rows": [1]}])
    assert unmatched["invalid_run"] is False
    assert record_has_spawn_overlap({"spawn_validity": reset})
    assert not record_has_spawn_overlap({"spawn_validity": clean})
    assert not record_has_spawn_overlap({})


def test_spawn_overlap_rows_are_ledger_invalid_and_excluded_from_rates() -> None:
    """A spawn-overlap row sets ledger invalid_run and leaves the aggregate rates."""
    from robot_sf.benchmark.aggregate import compute_aggregates
    from robot_sf.benchmark.event_ledger import build_event_ledger

    invalid = build_spawn_validity({"overlap": True}, [])
    overlap_row = {
        "episode_id": "overlap",
        "scenario_id": "scenario",
        "algo": "orca",
        "termination_reason": "collision",
        "outcome": {"collision_event": True, "route_complete": False, "timeout_event": False},
        "metrics": {"success": 0.0, "collisions": 1.0},
        "spawn_validity": invalid,
    }
    clean_row = {
        "episode_id": "clean",
        "scenario_id": "scenario",
        "algo": "orca",
        "termination_reason": "success",
        "outcome": {"collision_event": False, "route_complete": True, "timeout_event": False},
        "metrics": {"success": 1.0, "collisions": 0.0},
        "spawn_validity": build_spawn_validity({"overlap": False}, []),
    }
    ledger = build_event_ledger(overlap_row)
    assert ledger["exact_events"]["invalid_run"] is True
    assert ledger["provenance"]["invalid_reason"] == "spawn_overlap"
    assert build_event_ledger(clean_row)["exact_events"]["invalid_run"] is False

    summary = compute_aggregates([overlap_row, clean_row], group_by="algo")
    assert summary["orca"]["success"]["mean"] == 1.0
    assert summary["orca"]["collisions"]["mean"] == 0.0
    assert summary["_meta"]["evidence_eligibility"]["excluded_record_count"] == 1


def test_map_definition_pickles_after_spawn_sampling() -> None:
    """Spawn-clearance caches hold prepared geometries; pickling must drop and rebuild them."""
    import pickle

    map_def = convert_map(str(MAPS / "classic_head_on_corridor.svg"))
    np.random.seed(0)
    sample_route(map_def, 0, robot_radius=ROBOT_RADIUS)
    relocate_overlapping_pedestrians(
        [(15.0, 20.0)], PED_RADIUS, [((15.2, 20.0), ROBOT_RADIUS)], map_def
    )
    restored = pickle.loads(pickle.dumps(map_def))
    np.random.seed(1)
    start = sample_route(restored, 0, robot_radius=ROBOT_RADIUS)[0]
    assert (
        robot_obstacle_clearance(restored, start, ROBOT_RADIUS) >= SPAWN_CLEARANCE_MARGIN_M - 1e-9
    )
