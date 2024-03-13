import pytest
from robot_sf.gym_env.robot_env import init_collision_and_sensors
from robot_sf.sim.simulator import Simulator
from robot_sf.gym_env.env_config import EnvSettings
from gymnasium import spaces

def test_init_collision_and_sensors():
    # Create a mock simulator with two robots
    sim = Simulator()
    sim.robots = ['robot1', 'robot2']

    # Create a mock environment configuration
    env_config = EnvSettings()
    env_config.sim_config = 'sim_config'
    env_config.robot_config = 'robot_config'
    env_config.lidar_config = 'lidar_config'

    # Create a mock observation space
    orig_obs_space = spaces.Dict({})

    # Call the function with the mock parameters
    occupancies, sensor_fusions = init_collision_and_sensors(sim, env_config, orig_obs_space)

    # Check that the function returns two lists of the correct length
    assert isinstance(occupancies, list)
    assert len(occupancies) == len(sim.robots)
    assert isinstance(sensor_fusions, list)
    assert len(sensor_fusions) == len(sim.robots)

pytest.main()