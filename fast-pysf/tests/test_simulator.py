"""Integration-style simulator smoke tests for the fast-pysf package."""

import warnings

import numpy as np
import pysocialforce as pysf
from pysocialforce.ped_grouping import PedestrianGroupings, PedestrianStates
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


def test_compute_forces_accumulates_multiple_force_components():
    """compute_forces should sum each force component explicitly into an array."""
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    def make_forces(_, __):
        return [
            lambda: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            lambda: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        ]

    simulator = pysf.Simulator(state=state, make_forces=make_forces)
    forces = simulator.compute_forces()

    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 2)
    assert np.array_equal(forces, np.array([[6.0, 8.0], [10.0, 12.0]], dtype=float))


def test_simulator_v2_step_accumulates_multiple_force_components():
    """Simulator_v2 stepping should feed the combined force array to pedestrian state."""
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    captured_forces = []

    def populate(_, __):
        states = PedestrianStates(state)
        groupings = PedestrianGroupings(states, {})
        return states, groupings, []

    def make_forces(_, __):
        return [
            lambda: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            lambda: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        ]

    simulator = pysf.Simulator_v2(make_forces=make_forces, populate=populate)
    simulator.peds.step = lambda force: captured_forces.append(force.copy())

    simulator.step()

    assert len(captured_forces) == 1
    assert np.array_equal(captured_forces[0], np.array([[6.0, 8.0], [10.0, 12.0]], dtype=float))


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
