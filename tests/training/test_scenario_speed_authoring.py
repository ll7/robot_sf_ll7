"""Regression checks for live speed settings and successor scenario authoring."""

import random
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_authoring_0_1_0_v1.yaml")


def _config(name, overrides=None):
    scenario = next(s for s in load_scenarios(MATRIX) if s["name"] == name)
    if overrides:
        scenario = dict(scenario)
        scenario["simulation_config"] = {**scenario["simulation_config"], **overrides}
    return build_robot_config_from_scenario(scenario, scenario_path=MATRIX)


@pytest.mark.parametrize(
    ("overrides", "mean", "std"),
    [
        ({"ped_speed_tier": "Typical", "desired_speed_seed": 1002}, 1.3, 0.2),
        ({"ped_speed_tier": "typical", "desired_speed_std": 0.0}, 1.3, 0.0),
        (
            {"desired_speed_mean": 1.1, "desired_speed_std": 0.0, "desired_speed_seed": 1001},
            1.1,
            0.0,
        ),
        (
            {"ped_speed_tier": "typical", "desired_speed_mean": 1.1, "desired_speed_std": 0.0},
            1.1,
            0.0,
        ),
    ],
)
def test_scenario_speed_settings_reach_live_caps(overrides, mean, std):
    """Loaded tiers and explicit overrides must change actual pedestrian caps."""
    config = _config("classic_realworld_double_bottleneck_high", overrides)
    config.sim_config.pedestrian_seed = 1001
    config.sim_config.route_spawn_seed = 1001
    random.seed(1001)
    np.random.seed(1001)
    assert config.sim_config.desired_speed_mean == pytest.approx(mean)
    assert config.sim_config.desired_speed_std == pytest.approx(std)
    if "desired_speed_seed" in overrides:
        assert config.sim_config.desired_speed_seed == overrides["desired_speed_seed"]
    sim = init_simulators(config, next(iter(config.map_pool.map_defs.values())))[0]
    caps = sim.pysf_sim.peds.max_speeds
    assert len(caps) >= 8
    assert caps.mean() == pytest.approx(mean, abs=0.15)
    if std == 0:
        np.testing.assert_allclose(caps, mean)
    elif "desired_speed_seed" in overrides:
        # These draws lie within the documented clipping bounds. The speed seed
        # differs from the population seed so a dropped override is observable.
        expected = np.random.default_rng(overrides["desired_speed_seed"]).normal(
            mean, std, len(caps)
        )
        np.testing.assert_allclose(caps, expected)


def test_bottleneck_routes_and_markers_clear_both_openings():
    """High density requires route spawns and obstacle-clear marker paths."""
    config = _config("classic_realworld_double_bottleneck_high")
    assert config.sim_config.peds_per_area_m2 == pytest.approx(0.08)
    map_def = next(iter(config.map_pool.map_defs.values()))
    obstacles = [
        Polygon(o.vertices).buffer(config.sim_config.ped_radius) for o in map_def.obstacles
    ]
    assert len(map_def.ped_routes) == 2
    for ped in map_def.single_pedestrians:
        path = LineString([ped.start, ped.goal])
        assert all(not path.intersects(obstacle) for obstacle in obstacles), ped.id


def test_platform_pause_waypoint_has_body_clearance():
    """The platform pause point must be reachable outside the stair block."""
    config = _config("classic_station_platform_medium")
    map_def = next(iter(config.map_pool.map_defs.values()))
    ped = next(p for p in map_def.single_pedestrians if p.id == "p3")
    obstacles = [
        Polygon(o.vertices).buffer(config.sim_config.ped_radius) for o in map_def.obstacles
    ]
    assert ped.wait_at[0].wait_s == 5.0
    pause = Point(ped.trajectory[1])
    path = LineString(ped.trajectory)
    assert all(not pause.intersects(o) and not path.intersects(o) for o in obstacles)
