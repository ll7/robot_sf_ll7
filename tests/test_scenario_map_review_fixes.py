"""Regression tests for the 2026-09-30 scenario and map review findings 2, 3, 4, 6, 7 and 8.

Each test goes through the production function that carried the defect, so it fails on the
pre-fix code. Seeds come from the development band (1001-1030); no environment is stepped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_with_episode_seed_defaults,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.nav_types import SvgRectangle
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels
from robot_sf.ped_npc.ped_behavior import CrowdedZoneBehavior, FollowRouteBehavior
from robot_sf.ped_npc.ped_population import (
    PedSpawnConfig,
    _synthetic_crowd_zones,
    populate_simulation,
)
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.training.scenario_loader import _rebase_scenario_paths, load_scenarios

if TYPE_CHECKING:
    from pathlib import Path

PED_RADIUS = 0.4
WALL = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)]
WALL_GEOM = Polygon(WALL)


def _behaviors(**kwargs):
    """Return the crowd and route behaviors built by ``populate_simulation``."""
    _state, _groups, behaviors = populate_simulation(0.5, **kwargs)
    crowd = next(b for b in behaviors if isinstance(b, CrowdedZoneBehavior))
    route = next(b for b in behaviors if isinstance(b, FollowRouteBehavior))
    return _state, crowd, route


# Finding 2: SVG rectangles were sampled as their lower-right triangle.


def test_svg_rectangle_zone_samples_the_full_rectangle() -> None:
    """A 4 m x 4 m SVG rectangle must be sampled uniformly over both halves."""
    zone = SvgRectangle(0.0, 0.0, 4.0, 4.0, "robot_spawn_zone", "z").get_zone()
    points = np.asarray(sample_zone(zone, 4000, rng=np.random.default_rng(1001)))
    above_diagonal = float(np.mean(points[:, 1] > points[:, 0]))
    assert 0.45 < above_diagonal < 0.55
    assert np.allclose(points.mean(axis=0), [2.0, 2.0], atol=0.1)
    assert points.min() >= 0.0 and points.max() <= 4.0


def test_parsed_svg_spawn_zone_covers_the_whole_rectangle(tmp_path: Path) -> None:
    """The parser-to-sampler path reaches the corner opposite the implied fourth corner."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="2" height="2" />
      <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="7" y="7" width="2" height="2" />
      <path id="robot_route_0_0" inkscape:label="robot_route_0_0" d="M 2 2 L 8 8" />
    </svg>
    """
    svg_path = tmp_path / "map.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")
    map_def = convert_map(str(svg_path))
    zone = map_def.robot_routes[0].spawn_zone
    points = np.asarray(sample_zone(zone, 2000, rng=np.random.default_rng(1002)))
    rect = box(1.0, 1.0, 3.0, 3.0)
    assert all(rect.buffer(1e-9).contains(Point(p)) for p in points)
    # Uniform over the rectangle: every quadrant receives roughly a quarter of the samples.
    for qx in (0, 1):
        for qy in (0, 1):
            quadrant = (
                (points[:, 0] >= 1 + qx)
                & (points[:, 0] < 2 + qx)
                & (points[:, 1] >= 1 + qy)
                & (points[:, 1] < 2 + qy)
            )
            assert 0.2 < float(np.mean(quadrant)) < 0.3


def test_true_triangle_zones_stay_triangular() -> None:
    """Synthetic crowd triangles must not grow into parallelograms outside the map."""
    zones = _synthetic_crowd_zones((0.0, 10.0, 0.0, 10.0), PED_RADIUS)
    rng = np.random.default_rng(1003)
    for zone in zones:
        triangle = Polygon(zone).buffer(1e-9)
        points = sample_zone(zone, 500, rng=rng)
        assert all(triangle.contains(Point(p)) for p in points)


def test_triangular_svg_crowded_zone_path_stays_triangular(tmp_path: Path) -> None:
    """An authored three-vertex crowded-zone path is a true triangle."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="8" y="8" width="1" height="1" />
      <path id="robot_route_0_0" inkscape:label="robot_route_0_0" d="M 1.5 1.5 L 8.5 8.5" />
      <path id="crowd" inkscape:label="crowded_zone" d="M 2 2 L 6 2 L 6 6 Z" />
    </svg>
    """
    svg_path = tmp_path / "tri.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")
    map_def = convert_map(str(svg_path))
    (zone,) = map_def.ped_crowded_zones
    triangle = Polygon(zone).buffer(1e-9)
    points = sample_zone(zone, 500, rng=np.random.default_rng(1004))
    assert all(triangle.contains(Point(p)) for p in points)


# Finding 3: ordinary crowded-zone behaviours lost their obstacle constraints.


def test_crowded_zone_behavior_keeps_obstacles_for_later_goals() -> None:
    """Goals re-sampled after spawn must avoid the obstacles used at spawn time."""
    zone = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    wall = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]
    np.random.seed(1005)
    _state, crowd, _route = _behaviors(
        spawn_config=PedSpawnConfig(peds_per_area_m2=0.05, max_group_members=1),
        ped_routes=[],
        ped_crowded_zones=[zone],
        obstacle_polygons=[wall],
    )
    assert crowd.obstacle_polygons is not None
    np.random.seed(1006)
    goals = [crowd._sample_goal(zone) for _ in range(300)]
    assert not any(Polygon(wall).contains(Point(goal)) for goal in goals)


# Finding 4: pedestrian spawns validated centres, not footprints.


def test_crowded_zone_spawns_keep_the_pedestrian_footprint_off_walls() -> None:
    """Crowd spawns next to a wall must keep a full pedestrian radius of clearance."""
    zone = ((1.0, 1.1), (9.0, 1.1), (9.0, 3.0))
    np.random.seed(1007)
    state, _crowd, _route = _behaviors(
        spawn_config=PedSpawnConfig(peds_per_area_m2=3.0, max_group_members=1),
        ped_routes=[],
        ped_crowded_zones=[zone],
        obstacle_polygons=[WALL],
        ped_radius=PED_RADIUS,
    )
    positions = state.pysf_states()[:, 0:2]
    assert len(positions) > 20
    clearances = [WALL_GEOM.distance(Point(p)) for p in positions]
    assert min(clearances) >= PED_RADIUS - 1e-9


def _wall_route() -> GlobalRoute:
    spawn_zone = ((1.0, 1.1), (2.5, 1.1), (2.5, 2.2))
    goal_zone = ((7.5, 1.1), (9.0, 1.1), (9.0, 2.2))
    return GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.6), (8.5, 1.6)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )


def test_route_spawns_and_respawns_keep_the_pedestrian_footprint_off_walls() -> None:
    """Route spawns and route-start respawns must keep a pedestrian radius off walls."""
    np.random.seed(1008)
    state, _crowd, route = _behaviors(
        spawn_config=PedSpawnConfig(
            peds_per_area_m2=3.0, max_group_members=1, sidewalk_width=1.0, route_spawn_seed=1008
        ),
        ped_routes=[_wall_route()],
        ped_crowded_zones=[],
        obstacle_polygons=[WALL],
        ped_radius=PED_RADIUS,
    )
    positions = state.pysf_states()[:, 0:2]
    assert len(positions) > 5
    assert min(WALL_GEOM.distance(Point(p)) for p in positions) >= PED_RADIUS - 1e-9

    np.random.seed(1009)
    respawned = []
    for gid in list(route.navigators)[:10]:
        route.respawn_group_at_start(gid, guard_robot=False)
        respawned.extend(route.groups.states.pos_of_many(route.groups.groups[gid]))
    assert respawned
    assert min(WALL_GEOM.distance(Point(p)) for p in respawned) >= PED_RADIUS - 1e-9


# Finding 6: the episode seed did not seed optional archetype assignment.


def test_archetype_seed_defaults_to_the_episode_seed() -> None:
    """Archetype scenarios without an explicit seed are reproducible per episode seed."""
    composition = {"slow": 0.5, "fast": 0.5}
    scenario = {
        "name": "archetypes",
        "simulation_config": {
            "archetype_composition": composition,
            "archetype_speed_factors": {"slow": 0.8, "fast": 1.2},
        },
    }
    first = _scenario_with_episode_seed_defaults(scenario, seed=1010)
    second = _scenario_with_episode_seed_defaults(scenario, seed=1010)
    seed = first["simulation_config"].get("archetype_seed")
    assert seed == 1010
    labels_a = assign_archetype_labels(64, composition, seed=seed)
    labels_b = assign_archetype_labels(
        64, composition, seed=second["simulation_config"]["archetype_seed"]
    )
    assert np.array_equal(labels_a, labels_b)


def test_archetype_seed_default_keeps_explicit_values_and_plain_scenarios() -> None:
    """Explicit seeds (including 0) stay; scenarios without archetypes gain no new key."""
    explicit = {
        "simulation_config": {"archetype_composition": {"a": 1.0}, "archetype_seed": 0},
    }
    assert (
        _scenario_with_episode_seed_defaults(explicit, seed=1011)["simulation_config"][
            "archetype_seed"
        ]
        == 0
    )
    plain = _scenario_with_episode_seed_defaults({"simulation_config": {}}, seed=1011)
    assert "archetype_seed" not in plain["simulation_config"]


# Finding 7: an included scenario's relative map was shadowed by a root-level file.


def _include_tree(tmp_path: Path) -> tuple[Path, Path]:
    child = tmp_path / "child"
    child.mkdir()
    (child / "map.svg").write_text("CHILD", encoding="utf-8")
    (child / "scenarios.yaml").write_text(
        "scenarios:\n- name: included\n  map_file: map.svg\n", encoding="utf-8"
    )
    root = tmp_path / "suite.yaml"
    root.write_text("includes:\n- child/scenarios.yaml\n", encoding="utf-8")
    return root, child


def _rebase(root: Path, child: Path) -> dict:
    return dict(
        _rebase_scenario_paths(
            {"name": "included", "map_file": "map.svg"},
            source=child / "scenarios.yaml",
            root=root,
            map_search_paths=[],
            map_registry={},
        )
    )


def test_included_scenario_map_resolves_beside_its_own_file(tmp_path: Path) -> None:
    """Without a collision the included map resolves relative to its own manifest."""
    root, child = _include_tree(tmp_path)
    assert _rebase(root, child)["map_file"] == "child/map.svg"


def test_included_scenario_map_name_collision_is_rejected(tmp_path: Path) -> None:
    """A same-named file beside the root manifest must not silently shadow the child's map."""
    root, child = _include_tree(tmp_path)
    (tmp_path / "map.svg").write_text("ROOT", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        _rebase(root, child)
    with pytest.raises(ValueError, match="ambiguous"):
        load_scenarios(root)


# Finding 8: sparse SVG zone indices were compacted, shifting route bindings.


def test_sparse_svg_zone_indices_are_rejected(tmp_path: Path) -> None:
    """A gap in explicit zone ids cannot be compacted: routes would bind the wrong zone."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="8" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_2" inkscape:label="robot_goal_zone_2" x="8" y="8" width="1" height="1" />
      <path id="robot_route_0_2" inkscape:label="robot_route_0_2" d="M 1.5 1.5 L 8.5 8.5" />
    </svg>
    """
    svg_path = tmp_path / "sparse.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")
    with pytest.raises(ValueError, match="robot_goal_zone index 1"):
        convert_map(str(svg_path))


def test_gap_filled_by_unindexed_zone_still_binds_by_position(tmp_path: Path) -> None:
    """The documented fill rule (unindexed zones fill gaps) keeps working."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
         width="10" height="10">
      <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
      <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="8" y="1" width="1" height="1" />
      <rect id="goal_free" inkscape:label="robot_goal_zone" x="5" y="5" width="1" height="1" />
      <rect id="robot_goal_zone_2" inkscape:label="robot_goal_zone_2" x="8" y="8" width="1" height="1" />
      <path id="robot_route_0_2" inkscape:label="robot_route_0_2" d="M 1.5 1.5 L 8.5 8.5" />
    </svg>
    """
    svg_path = tmp_path / "filled.svg"
    svg_path.write_text(svg.strip(), encoding="utf-8")
    map_def = convert_map(str(svg_path))
    assert map_def.robot_routes[0].goal_zone[0] == (8.0, 8.0)
