"""Versioned interaction wiring and literal legacy-output compatibility."""

import hashlib

import numpy as np
import pytest
from pysocialforce.config import GroupGazeForceConfig, GroupReplusiveForceConfig, SceneConfig
from pysocialforce.force_trace import stable_config_hash
from pysocialforce.forces import GroupGazeForceAlt, GroupRepulsiveForce
from pysocialforce.scene import PedState

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.ped_npc.ped_robot_force import PedRobotForce, PedRobotForceConfig
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators
from tests.sim.test_residual_adversary_wiring import _minimal_map


def test_robot_steering_starts_outside_contact_and_scales_with_radius():
    """Robot steering starts outside contact and scales with radius."""
    for radius in [0.5, 1.0, 1.5]:
        contact = radius + 0.35
        state = np.array([[-contact - 1.9, 0.0, 1.0, 0.0, 8.0, 0.0, 0.5]])
        peds = PedState(state, [[0]], SceneConfig(agent_radius=0.35))
        config = PedRobotForceConfig(robot_radius=radius)
        config.law_version = "anticipatory_v2"
        config.edge_onset = 2.0
        config.steering_factor = 1.0
        force = PedRobotForce(config, peds, lambda: (0.0, 0.0))()[0]
        assert force[0] < 0.0, "brake with 1.9 m remaining before contact"
        assert abs(force[1]) > 0.0, "head-on approach needs a passing direction"
        # Same law must not turn a pedestrian already walking away or on a safe bypass.
        peds.state[0, 2] = -1.0
        assert PedRobotForce(config, peds, lambda: (0.0, 0.0))()[0, 1] == 0.0
        peds.state[0] = [-contact - 1.9, contact + 0.1, 1.0, 0.0, 8.0, 0.0, 0.5]
        assert PedRobotForce(config, peds, lambda: (0.0, 0.0))()[0, 1] >= 0.0
        peds.state[0, 0] = -contact - 2.01
        peds.state[0, 1] = 0.0
        assert PedRobotForce(config, peds, lambda: (0.0, 0.0))() == pytest.approx(np.zeros((1, 2)))


def test_profile_reaches_real_force_factory_and_roundtrips():
    """Profile reaches real force factory and roundtrips."""
    settings = SimulationSettings(population_size=2, route_spawn_seed=1001)
    settings.pedestrian_force_profile = "pedestrian_interaction_v2"
    settings.__post_init__()
    map_def = _minimal_map()
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=settings,
    )
    sim = init_simulators(
        config, map_def, num_robots=1, random_start_pos=False, peds_have_obstacle_forces=True
    )[0]
    forces = sim.pysf_sim.forces
    gaze = next(f for f in forces if isinstance(f, GroupGazeForceAlt))
    repulsion = next(f for f in forces if isinstance(f, GroupRepulsiveForce))
    robot = next(f for f in forces if isinstance(f, PedRobotForce))
    # Runtime behaviour checks first so base red proof names the ignored profile.
    sim.pysf_sim.peds.groups = [[0, 1]]
    sim.pysf_sim.peds.state[:, :6] = [
        [3.0, 0.0, 1.0, 0.0, 10.0, 0.0],
        [4.0, 0.0, 1.0, 0.0, 10.0, 0.0],
    ]
    assert gaze()[0] == pytest.approx([0.0, 0.0])
    assert repulsion.config.law_version == "unit_vectors_v2"
    assert robot.config.law_version == "anticipatory_v2"
    assert robot.config.robot_radius == sim.robots[0].config.radius
    restored = SimulationSettings(**settings.to_dict())
    assert restored.pedestrian_force_profile == settings.pedestrian_force_profile


def test_absent_profile_preserves_base_force_and_trajectory_bytes():
    """Absent profile preserves base force and trajectory bytes."""
    peds = PedState(
        np.array(
            [
                [2.7, 0.0, 0.6, 0.2, 8.0, 1.0, 0.5],
                [2.8, 0.3, 0.5, 0.0, 8.0, 1.0, 0.5],
                [4.0, 1.0, -0.2, 0.3, 1.0, 2.0, 0.5],
            ]
        ),
        [[0, 1, 2]],
        SceneConfig(),
    )
    configs = [GroupGazeForceConfig(), GroupReplusiveForceConfig(), PedRobotForceConfig()]
    forces = [
        GroupGazeForceAlt(configs[0], peds),
        GroupRepulsiveForce(configs[1], peds),
        PedRobotForce(configs[2], peds, lambda: (0.0, 0.0)),
    ]
    digest = hashlib.sha256()
    for _ in range(12):
        arrays = [f() for f in forces]
        for array in arrays:
            digest.update(array.tobytes())
        peds.step(sum(arrays))
        digest.update(peds.state.tobytes())
    # Captured on untouched dada635b (also compared with freeze 66f402ba).
    assert digest.hexdigest() == "a52fdadb3065aea30c5c7d66988e15acdc07b4016e0bb001435eacc69c7111a4"
    assert [stable_config_hash(c) for c in configs] == [
        "9b2bbad9979dfeee6481511740d73e2ad576185e066c710c8ad262b7a8839f86",
        "d8a093040b0164a54fc4213df6fb0315f983094f66d660989cfa19799faa304b",
        "51e2a3bf276c11bcba65fda54d66e089212cfb39cb794cdd3e9acf2d851c862d",
    ]


def test_scenario_profile_and_edge_onset_reach_runtime_settings():
    """Scenario overrides retain opt-in onset geometry and reject ambiguous knobs."""
    from robot_sf.training.scenario_loader import _apply_simulation_overrides

    config = RobotSimulationConfig()
    _apply_simulation_overrides(
        config,
        {
            "pedestrian_force_profile": "pedestrian_interaction_v2",
            "prf_config": {"edge_onset": 1.65, "steering_factor": 0.5},
        },
    )
    restored = SimulationSettings(**config.sim_config.to_dict())
    assert restored.prf_config.edge_onset == 1.65
    assert restored.prf_config.steering_factor == 0.5
    assert restored.prf_config.law_version == "anticipatory_v2"
    with pytest.raises(ValueError, match="require"):
        _apply_simulation_overrides(RobotSimulationConfig(), {"prf_config": {"edge_onset": 1.65}})
    with pytest.raises(ValueError, match="Unsupported pedestrian_force_profile"):
        SimulationSettings(pedestrian_force_profile="typo")


def test_legacy_settings_serialization_and_hash_stay_identical():
    """Missing selectors preserve both serialized settings and diagnostic hash identity."""
    settings = SimulationSettings()
    assert "pedestrian_force_profile" not in settings.to_dict()
    assert (
        stable_config_hash(settings)
        == "0d054279f4a6e090d40e1fa54accaf58e955e624a3309472b44107f8790bb78d"
    )
