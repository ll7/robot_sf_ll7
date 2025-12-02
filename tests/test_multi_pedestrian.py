"""
Test: Multi-Pedestrian Scenario (T020)

Validates that multiple single pedestrians (goal, trajectory, static) are correctly loaded and simulated in a single map.
"""

from examples.example_multi_pedestrian import create_multi_pedestrian_map
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool


def test_multi_pedestrian_map_definition():
    """Test multi pedestrian map definition.

    Returns:
        Any: Auto-generated placeholder description.
    """
    map_def = create_multi_pedestrian_map()
    assert len(map_def.single_pedestrians) == 4  # T033: exactly 4 single pedestrians
    ids = {ped.id for ped in map_def.single_pedestrians}
    assert "ped_goal_1" in ids
    assert "ped_goal_2" in ids
    assert "ped_static" in ids
    assert "ped_traj_1" in ids
    # Check types
    for ped in map_def.single_pedestrians:
        if ped.trajectory:
            assert isinstance(ped.trajectory, list)
        elif ped.goal:
            assert isinstance(ped.goal, tuple)
        else:
            assert ped.goal is None and ped.trajectory is None


def test_multi_pedestrian_env_smoke():
    """Test multi pedestrian env smoke.

    Returns:
        Any: Auto-generated placeholder description.
    """
    map_def = create_multi_pedestrian_map()
    pool = MapDefinitionPool(map_defs={"multi_ped": map_def})
    config = RobotSimulationConfig()
    config.map_pool = pool
    env = make_robot_env(config=config, debug=True)
    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    # If we reach here, env runs with multiple pedestrians
