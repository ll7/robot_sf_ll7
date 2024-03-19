"""
env_util
"""
from typing import List

from gymnasium import spaces

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.occupancy import ContinuousOccupancy
from robot_sf.sensor.range_sensor import lidar_ray_scan, lidar_sensor_space
from robot_sf.sensor.goal_sensor import target_sensor_obs, target_sensor_space
from robot_sf.sensor.sensor_fusion import fused_sensor_space, SensorFusion
from robot_sf.sim.simulator import Simulator

def init_collision_and_sensors(
        sim: Simulator,
        env_config: EnvSettings,
        orig_obs_space: spaces.Dict
        ):
    """
    Initialize collision detection and sensor fusion for the robots in the
    simulator.

    Parameters:
    sim (Simulator): The simulator object.
    env_config (EnvSettings): Configuration settings for the environment.
    orig_obs_space (spaces.Dict): Original observation space.

    Returns:
    Tuple[List[ContinuousOccupancy], List[SensorFusion]]:
        A tuple containing a list of occupancy objects for collision detection
        and a list of sensor fusion objects for sensor data handling.
    """

    # Get the number of robots, simulation configuration,
    # robot configuration, and lidar configuration
    num_robots = len(sim.robots)
    sim_config = env_config.sim_config
    robot_config = env_config.robot_config
    lidar_config = env_config.lidar_config

    # Initialize occupancy objects for each robot for collision detection
    occupancies = [ContinuousOccupancy(
            sim.map_def.width, sim.map_def.height,
            lambda: sim.robot_pos[i], lambda: sim.goal_pos[i],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4], lambda: sim.ped_pos,
            robot_config.radius, sim_config.ped_radius, sim_config.goal_radius)
        for i in range(num_robots)]

    # Initialize sensor fusion objects for each robot for sensor data handling
    sensor_fusions: List[SensorFusion] = []
    for r_id in range(num_robots):
        # Define the ray sensor, target sensor, and speed sensor for each robot
        ray_sensor = lambda r_id=r_id: lidar_ray_scan(
            sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]
        target_sensor = lambda r_id=r_id: target_sensor_obs(
            sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id])
        speed_sensor = lambda r_id=r_id: sim.robots[r_id].current_speed

        # Create the sensor fusion object and add it to the list
        sensor_fusions.append(SensorFusion(
            ray_sensor, speed_sensor, target_sensor,
            orig_obs_space, sim_config.use_next_goal))

    return occupancies, sensor_fusions


def init_spaces(env_config: EnvSettings, map_def: MapDefinition):
    """
    Initialize the action and observation spaces for the environment.

    This function creates action and observation space using the factory method
    provided in the environment
    configuration, and then uses the robot's action space and observation space
    as the basis for the environment's action and observation spaces.
    The observation space is
    further extended with additional sensors.

    Parameters
    ----------
    env_config : EnvSettings
        The configuration settings for the environment.
    map_def : MapDefinition
        The definition of the map for the environment.

    Returns
    -------
    Tuple[Space, Space, Space]
        A tuple containing the action space, the extended observation space, and
        the original observation space of the robot.
    """

    # Create a robot using the factory method in the environment configuration
    robot = env_config.robot_factory()

    # Get the action space from the robot
    action_space = robot.action_space

    # Extend the robot's observation space with additional sensors
    observation_space, orig_obs_space = fused_sensor_space(
        env_config.sim_config.stack_steps,
        robot.observation_space,
        target_sensor_space(map_def.max_target_dist),
        lidar_sensor_space(
            env_config.lidar_config.num_rays,
            env_config.lidar_config.max_scan_dist)
        )

    # Return the action space, the extended observation space, and the original
    # observation space
    return action_space, observation_space, orig_obs_space
