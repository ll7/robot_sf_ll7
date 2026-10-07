"""Single radius must reach production force, physical, placement and metric sites."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from robot_sf.benchmark import metrics
from robot_sf.benchmark.runner import _scenario_ped_radius_m
from robot_sf.ped_npc.ped_population import _synthetic_crowd_zones
from robot_sf.sim.sim_config import SimulationSettings
from tests.sim.test_pedestrian_desired_speed import _build_simulator


def test_explicit_radius_reaches_force_physical_placement_and_metric_consumers():
    np.random.seed(1001)
    settings = SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.04], population_size=8)
    settings.pedestrian_radius_m = 0.28  # Old class accepts but silently ignores assignment.
    sim = _build_simulator(settings)
    assert settings.ped_radius == 0.28, "physical radius ignores explicit parameter"
    assert sim.pysf_sim.peds.agent_radius == 0.28, "force kernel ignores explicit parameter"
    assert sim.config.ped_radius == 0.28, "physical geometry ignores explicit parameter"
    zone = _synthetic_crowd_zones((0, 10, 0, 10), settings.ped_radius)[0]
    assert zone[0] == (0.28, 0.28), "placement margin ignores explicit parameter"
    data = metrics.EpisodeData(
        robot_pos=np.zeros((3, 2)),
        robot_vel=np.zeros((3, 2)),
        robot_acc=np.zeros((3, 2)),
        peds_pos=np.array([[[0.6, 0]]] * 3),
        ped_forces=np.zeros((3, 1, 2)),
        goal=np.ones(2),
        dt=0.1,
        robot_radius=0.3,
        ped_radius=settings.ped_radius,
    )
    assert metrics.human_collisions(data) == 0, "metric radius ignores explicit parameter"
    assert _scenario_ped_radius_m({"pedestrian_radius_m": 0.28, "ped_radius": 0.4}) == 0.28


def test_radius_roundtrip_and_unset_default_hash_payload():
    default = SimulationSettings()
    assert default.ped_radius == 0.4
    assert "pedestrian_radius_m" not in asdict(default)
    assert "pedestrian_radius_m" not in default.to_dict()
    settings = SimulationSettings(pedestrian_radius_m=0.28)
    for clone in (replace(settings), deepcopy(settings), SimulationSettings(**settings.to_dict())):
        assert clone == settings
        assert clone.ped_radius == 0.28
    settings.ped_radius = 0.5
    assert settings.ped_radius == 0.28
    settings.pedestrian_radius_m = None
    assert settings.ped_radius == 0.5


def test_shared_radius_sets_gradient_profile_body_edge_origin():
    np.random.seed(1001)
    settings = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.04],
        population_size=8,
        obstacle_force_profile="gradient_v3",
    )
    settings.pedestrian_radius_m = 0.28
    sim = _build_simulator(settings)
    assert sim.pysf_sim.config.obstacle_force_config.threshold == 0.28
    assert sim.pysf_sim.config.obstacle_force_config.sigma == 0.0
    assert sim.pysf_sim.config.scene_config.agent_radius == 0.28


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True, "0.28"])
def test_invalid_radius_is_rejected_before_simulation(radius):
    with pytest.raises((TypeError, ValueError), match="pedestrian_radius_m"):
        SimulationSettings(pedestrian_radius_m=radius)


def test_scenario_loader_forwards_shared_radius_to_actual_simulation_config():
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.training.scenario_loader import _apply_simulation_overrides

    config = RobotSimulationConfig()
    _apply_simulation_overrides(config, {"pedestrian_radius_m": 0.28})
    assert config.sim_config.ped_radius == 0.28, "scenario radius was silently dropped"


@pytest.mark.parametrize("value", [False, "0.28", float("nan")])
def test_benchmark_metric_radius_rejects_malformed_explicit_selector(value):
    with pytest.raises(ValueError, match="pedestrian_radius_m"):
        _scenario_ped_radius_m({"sim_config": {"pedestrian_radius_m": value}, "ped_radius": 0.4})
