"""Regression tests for ``force_population_size`` accounting in ``populate_simulation``.

Issue #4618 R4: when a scenario declares BOTH pedestrian routes and crowded zones, the
route spawner and the crowd spawner each independently honored ``force_population_size``,
so the merged population was ``2 * force_population_size`` (a silent double-count). The
forced size must instead be the exact TOTAL, split across the active spawners.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from robot_sf.nav.navigation import get_prepared_obstacles
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.ped_npc import ped_behavior, ped_population
from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels
from robot_sf.ped_npc.ped_behavior import CrowdedZoneBehavior
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIC_BOTTLENECK_MEDIUM_SVG = _REPO_ROOT / "maps" / "svg_maps" / "classic_bottleneck_medium.svg"
_FRANCIS2023_BLIND_CORNER_SVG = (
    _REPO_ROOT / "maps" / "svg_maps" / "francis2023" / "francis2023_blind_corner.svg"
)


@pytest.fixture(scope="module")
def _route_map_with_explicit_pedestrian():
    """Load a real map containing both a route and one explicit pedestrian."""
    return SvgMapConverter(str(_CLASSIC_BOTTLENECK_MEDIUM_SVG)).get_map_definition()


@pytest.fixture(scope="module")
def _geometryless_francis_map():
    """Load a real single-interaction map with no background spawn geometry."""
    map_def = SvgMapConverter(str(_FRANCIS2023_BLIND_CORNER_SVG)).get_map_definition()
    assert not map_def.ped_routes
    assert not map_def.ped_crowded_zones
    assert len(map_def.single_pedestrians) == 1
    return map_def


def _forced_francis_population(map_def, seed: int):
    """Instantiate the issue #5829 regression population with production geometry inputs."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.0,
        max_group_members=3,
        initial_speed=0.5,
        force_population_size=12,
        route_spawn_seed=seed,
        archetype_composition={"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
        archetype_speed_factors={"cautious": 0.7, "standard": 1.0, "hurried": 1.4},
        archetype_seed=3574,
    )
    return spawn_config, populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=map_def.ped_routes,
        ped_crowded_zones=map_def.ped_crowded_zones,
        obstacle_polygons=get_prepared_obstacles(map_def),
        single_pedestrians=map_def.single_pedestrians,
        map_bounds=map_def.get_map_bounds(),
        reserved_zones=[*map_def.robot_spawn_zones, *map_def.robot_goal_zones],
        ped_radius=0.4,
        reserved_zone_radius=0.3,
    )


def _zone_polygon(zone) -> Polygon:
    """Expand the map's three-corner rectangle encoding into a polygon."""
    a, b, c = zone
    d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
    return Polygon((a, b, c, d))


@pytest.fixture
def _captured_spawner_sizes(monkeypatch):
    """Patch the two spawners/behaviors and capture the forced size each spawner sees.

    Returns a dict with ``crowd`` and ``route`` keys holding the
    ``force_population_size`` value passed to each spawner (``None`` when a spawner is
    skipped or receives no override).
    """
    captured: dict[str, int | None] = {"crowd": None, "route": None}

    def _fake_crowded(config, crowded_zones, *, obstacle_polygons=None):
        captured["crowd"] = config.force_population_size
        count = int(config.force_population_size or 0) if crowded_zones else 0
        return np.zeros((count, 6)), [], {}

    def _fake_routes(config, routes, *, obstacle_polygons=None):
        captured["route"] = config.force_population_size
        count = int(config.force_population_size or 0) if routes else 0
        return np.zeros((count, 6)), [], {}, {}

    monkeypatch.setattr(ped_population, "populate_crowded_zones", _fake_crowded)
    monkeypatch.setattr(ped_population, "populate_ped_routes", _fake_routes)
    # Replace behavior constructors with no-ops; the spawners return count-only states.
    monkeypatch.setattr(ped_population, "CrowdedZoneBehavior", lambda *a, **k: object())
    monkeypatch.setattr(ped_population, "FollowRouteBehavior", lambda *a, **k: object())
    return captured


def test_force_population_size_split_when_routes_and_zones_present(_captured_spawner_sizes):
    """Both spawners active: the forced total is split, summing to the requested size."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=10,
    )
    populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=[object()],  # sentinel routes; spawner is stubbed
        ped_crowded_zones=[object()],  # sentinel zones; spawner is stubbed
    )

    crowd_size = _captured_spawner_sizes["crowd"]
    route_size = _captured_spawner_sizes["route"]
    assert crowd_size is not None and route_size is not None
    # No double-count: the two shares reconstruct the exact requested total.
    assert crowd_size + route_size == 10
    # Odd remainder is assigned to routes (5 + 5 here, but the contract is the sum).
    assert route_size >= crowd_size


def test_force_population_size_odd_remainder_goes_to_routes(_captured_spawner_sizes):
    """An odd forced total assigns the extra pedestrian to the route spawner."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=7,
    )
    populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=[object()],
        ped_crowded_zones=[object()],
    )

    assert _captured_spawner_sizes["crowd"] == 3
    assert _captured_spawner_sizes["route"] == 4


def test_force_population_size_not_split_when_only_routes(_captured_spawner_sizes):
    """Only routes active: the route spawner receives the full forced size, unsplit."""
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=10,
    )
    populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=[object()],
        ped_crowded_zones=[],
    )

    assert _captured_spawner_sizes["route"] == 10


@pytest.mark.parametrize("population_size", [1, 3, 6, 12, 24])
def test_force_population_size_includes_explicit_pedestrians(
    _route_map_with_explicit_pedestrian,
    population_size: int,
) -> None:
    """The forced total includes map-defined pedestrians as well as route spawns."""
    map_def = _route_map_with_explicit_pedestrian
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        force_population_size=population_size,
        route_spawn_distribution="spread",
        route_spawn_seed=3574,
    )

    ped_states, _groups, _behaviors = populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=map_def.ped_routes,
        ped_crowded_zones=map_def.ped_crowded_zones,
        obstacle_polygons=get_prepared_obstacles(map_def),
        single_pedestrians=map_def.single_pedestrians,
    )

    assert ped_states.num_peds == population_size


def test_force_population_size_preserves_full_population_archetype_mix(
    _route_map_with_explicit_pedestrian,
) -> None:
    """The forced 12-person population realizes the declared 3/6/3 speed split."""
    map_def = _route_map_with_explicit_pedestrian
    spawn_config = PedSpawnConfig(
        peds_per_area_m2=0.02,
        max_group_members=3,
        initial_speed=0.5,
        force_population_size=12,
        route_spawn_distribution="spread",
        route_spawn_seed=3574,
        archetype_composition={"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
        archetype_speed_factors={"cautious": 0.7, "standard": 1.0, "hurried": 1.4},
        archetype_seed=3574,
    )

    ped_states, _groups, _behaviors = populate_simulation(
        tau=0.5,
        spawn_config=spawn_config,
        ped_routes=map_def.ped_routes,
        ped_crowded_zones=map_def.ped_crowded_zones,
        obstacle_polygons=get_prepared_obstacles(map_def),
        single_pedestrians=map_def.single_pedestrians,
    )

    speeds = np.linalg.norm(ped_states.ped_velocities, axis=1)
    labels = assign_archetype_labels(
        12,
        spawn_config.archetype_composition,
        seed=spawn_config.archetype_seed,
    )
    expected_speeds = np.asarray(
        [0.5 * spawn_config.archetype_speed_factors[label] for label in labels],
    )
    assert np.allclose(speeds, expected_speeds)
    assert Counter(np.round(speeds, 6).tolist()) == {0.35: 3, 0.5: 6, 0.7: 3}


def test_force_population_scatter_spawns_geometryless_francis_map(
    _geometryless_francis_map,
) -> None:
    """A Francis-style forced population is exact, composed, and uses crowd wandering."""
    spawn_config, (ped_states, _groups, behaviors) = _forced_francis_population(
        _geometryless_francis_map,
        seed=219,
    )

    assert ped_states.num_peds == 12
    speeds = np.linalg.norm(ped_states.ped_velocities, axis=1)
    labels = assign_archetype_labels(
        12,
        spawn_config.archetype_composition,
        seed=spawn_config.archetype_seed,
    )
    expected_speeds = np.asarray(
        [0.5 * spawn_config.archetype_speed_factors[label] for label in labels],
    )
    assert np.allclose(speeds, expected_speeds)
    assert Counter(np.round(speeds, 6).tolist()) == {0.35: 3, 0.5: 6, 0.7: 3}
    assert isinstance(behaviors[0], CrowdedZoneBehavior)
    assert behaviors[0].groups.states.num_peds == 11


def test_force_population_scatter_spawn_is_seed_deterministic(
    _geometryless_francis_map,
) -> None:
    """Identical population seeds reproduce positions while another seed changes them."""
    _, (same_a, _groups_a, behaviors_a) = _forced_francis_population(
        _geometryless_francis_map,
        seed=219,
    )
    _, (same_b, _groups_b, behaviors_b) = _forced_francis_population(
        _geometryless_francis_map,
        seed=219,
    )
    _, (different, _groups_c, _behaviors_c) = _forced_francis_population(
        _geometryless_francis_map,
        seed=220,
    )

    assert np.array_equal(same_a.ped_positions, same_b.ped_positions)
    assert not np.array_equal(same_a.ped_positions[:11], different.ped_positions[:11])

    behaviors_a[0].reset()
    behaviors_b[0].reset()
    assert np.array_equal(same_a.pysf_states()[:11, 4:6], same_b.pysf_states()[:11, 4:6])
    for goal in same_a.pysf_states()[:11, 4:6]:
        assert all(
            not exclusion.intersects(Point(goal)) for exclusion in behaviors_a[0].obstacle_polygons
        )


def test_native_crowded_zone_goal_sampling_keeps_legacy_call(monkeypatch) -> None:
    """Native crowd zones keep the pre-existing unconstrained sampler invocation."""
    zone = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    calls = []

    def _record_sample(sampled_zone, num_samples):
        calls.append((sampled_zone, num_samples))
        return [(0.25, 0.75)]

    monkeypatch.setattr(ped_behavior, "sample_zone", _record_sample)
    behavior = CrowdedZoneBehavior(groups=None, zone_assignments={}, crowded_zones=[])

    assert behavior._sample_goal(zone) == (0.25, 0.75)
    assert calls == [(zone, 1)]


def test_force_population_scatter_spawn_has_radius_clearance(
    _geometryless_francis_map,
) -> None:
    """Synthesized backgrounds start clear of static geometry and every t=0 agent."""
    ped_radius = 0.4
    _, (ped_states, _groups, _behaviors) = _forced_francis_population(
        _geometryless_francis_map,
        seed=219,
    )
    positions = ped_states.ped_positions
    background_positions = positions[:11]
    prepared_obstacles = get_prepared_obstacles(_geometryless_francis_map)
    reserved_polygons = [
        _zone_polygon(zone)
        for zone in [
            *_geometryless_francis_map.robot_spawn_zones,
            *_geometryless_francis_map.robot_goal_zones,
        ]
    ]

    for position in background_positions:
        point = Point(position)
        assert all(
            obstacle.context.distance(point) >= ped_radius - 1e-9 for obstacle in prepared_obstacles
        )
        assert all(zone.distance(point) >= ped_radius + 0.3 - 1e-9 for zone in reserved_polygons)

    for left_index, left in enumerate(positions):
        for right in positions[left_index + 1 :]:
            assert np.linalg.norm(left - right) >= 2 * ped_radius - 1e-9
