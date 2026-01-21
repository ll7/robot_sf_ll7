"""Tests for scenario-level single pedestrian overrides."""

from __future__ import annotations

import pytest

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.training.scenario_loader import (
    _apply_single_pedestrian_overrides,
    apply_single_pedestrian_overrides,
)


def _build_map_with_pois() -> MapDefinition:
    """Create a minimal map definition with POIs for override tests."""
    width, height = 10.0, 10.0
    obstacles = [Obstacle([(0, 0), (10, 0), (10, 1), (0, 1)])]
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    ped_spawn_zones = [((3, 3), (4, 3), (4, 4))]
    robot_goal_zones = [((8, 8), (9, 8), (9, 9))]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    ped_goal_zones = [((6, 6), (7, 6), (7, 7))]
    ped_crowded_zones: list = []
    robot_routes: list = []
    ped_routes: list = []
    poi_labels = {"poi1": "start", "poi2": "mid", "poi3": "end"}
    poi_positions = [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]

    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        [SinglePedestrianDefinition(id="ped1", start=(2.0, 2.0), goal=(4.0, 4.0))],
        poi_positions=poi_positions,
        poi_labels=poi_labels,
    )


def test_apply_single_pedestrian_overrides_goal_poi():
    """Verify POI-based goal overrides set deterministic targets for scenario configs."""
    map_def = _build_map_with_pois()

    apply_single_pedestrian_overrides(
        map_def,
        [
            {
                "id": "ped1",
                "goal_poi": "end",
                "speed_m_s": 0.4,
                "note": "slow",
            }
        ],
    )

    ped = map_def.single_pedestrians[0]
    assert ped.goal == (9.0, 9.0)
    assert ped.speed_m_s == pytest.approx(0.4)
    assert ped.note == "slow"


def test_apply_single_pedestrian_overrides_trajectory_poi_wait():
    """Verify POI-based trajectories and wait rules resolve to correct waypoint indices."""
    map_def = _build_map_with_pois()

    apply_single_pedestrian_overrides(
        map_def,
        [
            {
                "id": "ped1",
                "goal": None,
                "trajectory_poi": ["start", "mid", "end"],
                "wait_at": [{"poi": "mid", "wait_s": 1.5, "note": "yield"}],
            }
        ],
    )

    ped = map_def.single_pedestrians[0]
    assert ped.goal is None
    assert ped.trajectory == [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]
    assert ped.wait_at is not None
    assert ped.wait_at[0].waypoint_index == 1
    assert ped.wait_at[0].wait_s == pytest.approx(1.5)
    assert ped.wait_at[0].note == "yield"


def test_apply_single_pedestrian_overrides_role_fields():
    """Verify role-related overrides are applied to single pedestrians."""
    map_def = _build_map_with_pois()

    apply_single_pedestrian_overrides(
        map_def,
        [
            {
                "id": "ped1",
                "role": "follow",
                "role_target_id": "robot:0",
                "role_offset": [1.0, -0.5],
            }
        ],
    )

    ped = map_def.single_pedestrians[0]
    assert ped.role == "follow"
    assert ped.role_target_id == "robot:0"
    assert ped.role_offset == (1.0, -0.5)


def test_single_pedestrian_overrides_do_not_mutate_shared_map_defs():
    """Ensure scenario overrides clone map defs to avoid cross-scenario contamination."""
    map_def = _build_map_with_pois()
    config = RobotSimulationConfig()
    config.map_pool = MapDefinitionPool(map_defs={"scenario_map": map_def})

    overrides = [
        {
            "id": "ped1",
            "goal": None,
            "trajectory": [(1.0, 1.0), (5.0, 5.0)],
        }
    ]
    _apply_single_pedestrian_overrides(config, overrides)

    updated_map = config.map_pool.map_defs["scenario_map"]
    assert updated_map is not map_def
    assert map_def.single_pedestrians[0].trajectory is None
    assert updated_map.single_pedestrians[0].trajectory == [(1.0, 1.0), (5.0, 5.0)]
