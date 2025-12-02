"""Module test_simulator auto-generated docstring."""

import pysocialforce as pysf


def test_can_simulate_with_empty_map_no_peds():
    """Test can simulate with empty map no peds.

    Returns:
        Any: Auto-generated placeholder description.
    """
    simulator = pysf.Simulator_v2()
    for _ in range(10):
        simulator.step()
        print(simulator)


def test_can_simulate_with_populated_map():
    """Test can simulate with populated map.

    Returns:
        Any: Auto-generated placeholder description.
    """
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
