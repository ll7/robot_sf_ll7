"""
env_util
"""
from typing import List, Union

from gymnasium import spaces

from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.sensor.range_sensor import lidar_ray_scan, lidar_sensor_space
from robot_sf.sensor.goal_sensor import target_sensor_obs, target_sensor_space
from robot_sf.sensor.sensor_fusion import fused_sensor_space, SensorFusion
from robot_sf.sim.simulator import Simulator, PedSimulator

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
    action_space, obs_space, orig_obs_space = create_spaces(env_config, map_def, create_robot=True)
    # Return the action space, the extended observation space, and the original
    # observation space
    return action_space, obs_space, orig_obs_space

def create_spaces(env_config: Union[EnvSettings, PedEnvSettings], map_def: MapDefinition,
                  create_robot: bool = True):
    # Create a agent using the factory method in the environment configuration
    if create_robot:
        agent = env_config.robot_factory()
    else:
        agent = env_config.pedestrian_factory()

    # Get the action space from the agent
    action_space = agent.action_space

    # Extend the agent's observation space with additional sensors
    observation_space, orig_obs_space = fused_sensor_space(
        env_config.sim_config.stack_steps,
        agent.observation_space,
        target_sensor_space(map_def.max_target_dist),
        lidar_sensor_space(
            env_config.lidar_config.num_rays,
            env_config.lidar_config.max_scan_dist)
        )
    return action_space, observation_space, orig_obs_space

def init_ped_spaces(env_config: PedEnvSettings, map_def: MapDefinition):
    """
    Initialize the action and observation spaces for the environment.

    This function creates action and observation space using the factory method
    provided in the environment
    configuration, and then uses the robot's and the pedestrian's action space and observation space
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
    action_space_robot, obs_space_robot, orig_obs_space_robot = create_spaces(env_config, map_def, create_robot=True)
    action_space_ped, obs_space_ped, orig_obs_space_ped = create_spaces(env_config, map_def, create_robot=False)

    # As a list [robot, pedestrian]
    # Return the action space, the extended observation space, and the original
    # observation space
    return [action_space_robot, action_space_ped], [obs_space_robot, obs_space_ped], [orig_obs_space_robot, orig_obs_space_ped]


def init_ped_collision_and_sensors(
        sim: PedSimulator,
        env_config: PedEnvSettings,
        orig_obs_space: List[spaces.Dict]
        ):
    """
    Initialize collision detection and sensor fusion for the robot and the pedestrian in the
    simulator.

    Parameters:
    sim (PedSimulator): The simulator object.
    env_config (PedEnvSettings): Configuration settings for the environment.
    orig_obs_space (spaces.Dict): Original observation space.

    Returns:
    Tuple[List[ContinuousOccupancy], List[SensorFusion]]:
        A tuple containing a list of occupancy objects for collision detection
        and a list of sensor fusion objects for sensor data handling.
    """

    # Get the simulation configuration, pedestrian configuration,
    # robot configuration, and lidar configuration
    # Only one robot will be present in the pedestrian simulator
    sim_config = env_config.sim_config
    robot_config = env_config.robot_config
    lidar_config = env_config.lidar_config
    ego_ped_config = env_config.ego_ped_config

    occupancies: List[Union[ContinuousOccupancy,EgoPedContinuousOccupancy]] = []
    sensor_fusions: List[SensorFusion] = []

    # Initialize a occupancy object for the robot for collision detection
    occupancies.append(ContinuousOccupancy(
            sim.map_def.width, sim.map_def.height,
            lambda: sim.robot_pos[0], lambda: sim.goal_pos[0],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4], lambda: sim.ped_pos,
            robot_config.radius, sim_config.ped_radius, sim_config.goal_radius))

    # Define the ray sensor, target sensor, and speed sensor for the robot
    ray_sensor = lambda r_id=0: lidar_ray_scan(
        sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]
    target_sensor = lambda r_id=0: target_sensor_obs(
        sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id])
    speed_sensor = lambda r_id=0: sim.robots[r_id].current_speed

    # Initialize a sensor fusion object for the robot for sensor data handling
    sensor_fusions.append(SensorFusion(
        ray_sensor, speed_sensor, target_sensor,
        orig_obs_space[0], sim_config.use_next_goal))

    # Initalize occupancy and sensor fusion for the ego pedestrian
    occupancies.append(EgoPedContinuousOccupancy(
            sim.map_def.width, sim.map_def.height,
            lambda: sim.ego_ped_pos, lambda: sim.ego_ped_goal_pos,
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4], lambda: sim.ped_pos,
            ego_ped_config.radius, sim_config.ped_radius,
            sim_config.goal_radius,
            lambda: sim.robot_pos[0], robot_config.radius))

    ray_sensor = lambda: lidar_ray_scan(
        sim.ego_ped.pose, occupancies[1], lidar_config)[0]
    target_sensor = lambda: target_sensor_obs(
        sim.ego_ped.pose, sim.ego_ped_goal_pos, sim.ego_ped_goal_pos) # TODO: What next goal to choose?
    speed_sensor = lambda: sim.ego_ped.current_speed

    sensor_fusions.append(SensorFusion(
        ray_sensor, speed_sensor, target_sensor,
        orig_obs_space[1], False)) # Ego pedestrian does not have a next goal

    # Format: [robot, ego_pedestrian]
    return occupancies, sensor_fusions
