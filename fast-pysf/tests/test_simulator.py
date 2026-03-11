"""Integration-style simulator smoke tests for the fast-pysf package."""

import warnings

import numpy as np
import pysocialforce as pysf
from pysocialforce.scene import PedState


def test_can_simulate_with_empty_map_no_peds():
    """Simulator should step without errors on an empty default map."""
    simulator = pysf.Simulator_v2()
    for _ in range(10):
        simulator.step()
        print(simulator)


def test_can_simulate_with_populated_map():
    """Simulator should step without errors when obstacles/routes/zones are present."""
    obstacle01 = pysf.map_config.Obstacle([(10, 10), (15, 10), (15, 15), (10, 15)])
    obstacle02 = pysf.map_config.Obstacle([(20, 10), (25, 10), (25, 15), (20, 15)])

    route01 = pysf.map_config.GlobalRoute([(0, 0), (10, 10), (20, 10), (30, 0)])
    crowded_zone01 = ((10, 10), (20, 10), (20, 20))

    map_def = pysf.map_config.MapDefinition(
        obstacles=[obstacle01, obstacle02], routes=[route01], crowded_zones=[crowded_zone01]
    )

    simulator = pysf.Simulator_v2(map_def)

    for _ in range(10):
        simulator.step()
        print(simulator.states.ped_positions)


def test_capped_velocity_handles_zero_desired_speed_without_runtime_warning():
    """Velocity capping should remain finite when desired speed is zero."""
    desired_velocity = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    max_velocity = np.array([0.0, 0.5], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        capped = PedState.capped_velocity(desired_velocity, max_velocity)

    assert np.all(np.isfinite(capped))
    assert np.allclose(capped[0], np.array([0.0, 0.0], dtype=float))
    assert np.allclose(capped[1], np.array([0.5, 0.0], dtype=float))
