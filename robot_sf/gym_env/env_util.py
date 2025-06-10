"""
env_util
"""

from enum import Enum
from typing import List, Union

import numpy as np
from gymnasium import spaces

from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings, RobotEnvSettings
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.sensor.goal_sensor import target_sensor_obs, target_sensor_space
from robot_sf.sensor.image_sensor import image_sensor_space
from robot_sf.sensor.image_sensor_fusion import ImageSensorFusion
from robot_sf.sensor.range_sensor import lidar_ray_scan, lidar_sensor_space
from robot_sf.sensor.sensor_fusion import SensorFusion, fused_sensor_space
from robot_sf.sim.simulator import PedSimulator, Simulator


class AgentType(Enum):
    ROBOT = 1
    PEDESTRIAN = 2


def init_collision_and_sensors(
    sim: Simulator, env_config: EnvSettings, orig_obs_space: spaces.Dict
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
    occupancies = [
        ContinuousOccupancy(
            sim.map_def.width,
            sim.map_def.height,
            lambda: sim.robot_pos[i],
            lambda: sim.goal_pos[i],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4],
            lambda: sim.ped_pos,
            robot_config.radius,
            sim_config.ped_radius,
            sim_config.goal_radius,
        )
        for i in range(num_robots)
    ]

    # Initialize sensor fusion objects for each robot for sensor data handling
    sensor_fusions: List[SensorFusion] = []
    for r_id in range(num_robots):
        # Define the ray sensor, target sensor, and speed sensor for each robot
        def ray_sensor(r_id=r_id):
            return lidar_ray_scan(sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]

        def target_sensor(r_id=r_id):
            return target_sensor_obs(
                sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id]
            )

        def speed_sensor(r_id=r_id):
            return sim.robots[r_id].current_speed

        # Create the sensor fusion object and add it to the list
        sensor_fusions.append(
            SensorFusion(
                ray_sensor,
                speed_sensor,
                target_sensor,
                orig_obs_space,
                sim_config.use_next_goal,
            )
        )

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
    action_space, obs_space, orig_obs_space = create_spaces(
        env_config, map_def, agent_type=AgentType.ROBOT
    )
    # Return the action space, the extended observation space, and the original
    # observation space
    return action_space, obs_space, orig_obs_space


def create_spaces(
    env_config: Union[EnvSettings, PedEnvSettings],
    map_def: MapDefinition,
    agent_type: AgentType = AgentType.ROBOT,
):
    # Create a agent using the factory method in the environment configuration
    if agent_type == AgentType.ROBOT:
        agent = env_config.robot_factory()
    elif agent_type == AgentType.PEDESTRIAN:
        agent = env_config.pedestrian_factory()
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Get the action space from the agent
    action_space = agent.action_space

    # Extend the agent's observation space with additional sensors
    observation_space, orig_obs_space = fused_sensor_space(
        env_config.sim_config.stack_steps,
        agent.observation_space,
        target_sensor_space(map_def.max_target_dist),
        lidar_sensor_space(env_config.lidar_config.num_rays, env_config.lidar_config.max_scan_dist),
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
    env_config : PedEnvSettings
        The configuration settings for the environment.
    map_def : MapDefinition
        The definition of the map for the environment.

    Returns
    -------
    Tuple[List[Space], List[Space], List[Space]]
        A tuple containing a list of action space, the extended observation space, and
        the original observation space of the robot and the pedestrian.
    """
    action_space_robot, obs_space_robot, orig_obs_space_robot = create_spaces(
        env_config, map_def, agent_type=AgentType.ROBOT
    )
    action_space_ped, obs_space_ped, orig_obs_space_ped = create_spaces(
        env_config, map_def, agent_type=AgentType.PEDESTRIAN
    )

    # As a list [robot, pedestrian]
    return (
        [action_space_robot, action_space_ped],
        [obs_space_robot, obs_space_ped],
        [orig_obs_space_robot, orig_obs_space_ped],
    )


def init_ped_collision_and_sensors(
    sim: PedSimulator, env_config: PedEnvSettings, orig_obs_space: List[spaces.Dict]
):
    """
    Initializes collision detection and sensor fusion for both the robot and the ego pedestrian in the pedestrian simulator.
    
    Creates occupancy objects for collision detection and sensor fusion objects for sensor data integration for the robot and the ego pedestrian, using their respective positions, goals, and environment configuration. Returns lists of occupancy and sensor fusion objects for both agents.
    
    Returns:
        A tuple containing two lists: occupancy objects and sensor fusion objects, ordered as [robot, ego pedestrian].
    """

    # Get the simulation configuration, pedestrian configuration,
    # robot configuration, and lidar configuration
    # Only one robot will be present in the pedestrian simulator
    sim_config = env_config.sim_config
    robot_config = env_config.robot_config
    lidar_config = env_config.lidar_config
    ego_ped_config = env_config.ego_ped_config

    occupancies: List[Union[ContinuousOccupancy, EgoPedContinuousOccupancy]] = []
    sensor_fusions: List[SensorFusion] = []

    # Initialize a occupancy object for the robot for collision detection
    occupancies.append(
        ContinuousOccupancy(
            sim.map_def.width,
            sim.map_def.height,
            lambda: sim.robot_pos[0],
            lambda: sim.goal_pos[0],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4],
            lambda: np.vstack((sim.ped_pos, np.array([sim.ego_ped_pos]))),
            # Add ego pedestrian to pedestrian positions, np.vstack might lead to performance issues
            robot_config.radius,
            sim_config.ped_radius,
            sim_config.goal_radius,
        )
    )

    # Define the ray sensor, target sensor, and speed sensor for the robot
    def ray_sensor(r_id=0):
        return lidar_ray_scan(sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]

    def target_sensor(r_id=0):
        return target_sensor_obs(sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id])

    def speed_sensor(r_id=0):
        return sim.robots[r_id].current_speed

    # Initialize a sensor fusion object for the robot for sensor data handling
    sensor_fusions.append(
        SensorFusion(
            ray_sensor,
            speed_sensor,
            target_sensor,
            orig_obs_space[0],
            sim_config.use_next_goal,
        )
    )

    # Initalize occupancy and sensor fusion for the ego pedestrian
    occupancies.append(
        EgoPedContinuousOccupancy(
            sim.map_def.width,
            sim.map_def.height,
            lambda: sim.ego_ped_pos,
            lambda: sim.ego_ped_goal_pos,
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4],
            lambda: sim.ped_pos,
            ego_ped_config.radius,
            sim_config.ped_radius,
            sim_config.goal_radius,
            lambda: sim.robot_pos[0],
            robot_config.radius,
        )
    )

    def ray_sensor_ego_ped():
        return lidar_ray_scan(sim.ego_ped.pose, occupancies[1], lidar_config)[0]

    def target_sensor_ego_ped():
        return target_sensor_obs(
            sim.ego_ped.pose, sim.ego_ped_goal_pos, None
        )  # TODO: What next goal to choose?

    def speed_sensor_ego_ped():
        return sim.ego_ped.current_speed

    sensor_fusions.append(
        SensorFusion(
            ray_sensor_ego_ped,
            speed_sensor_ego_ped,
            target_sensor_ego_ped,
            orig_obs_space[1],
            False,
        )
    )  # Ego pedestrian does not have a next goal

    # Format: [robot, ego_pedestrian]
    return occupancies, sensor_fusions


def create_spaces_with_image(
    env_config: Union[EnvSettings, PedEnvSettings, RobotEnvSettings],
    map_def: MapDefinition,
    agent_type: AgentType = AgentType.ROBOT,
):
    """
    Creates action and observation spaces for an agent, optionally including image-based observations.
    
    If image observations are enabled in the environment configuration and an image configuration is provided, the observation space is extended to include image sensor data in addition to target and lidar sensors. Raises a ValueError if a pedestrian agent is requested but no pedestrian factory is available.
    
    Returns:
        A tuple containing the agent's action space, the extended observation space (with optional image sensor), and the original observation space.
    """
    # Create a agent using the factory method in the environment configuration
    if agent_type == AgentType.ROBOT:
        agent = env_config.robot_factory()
    elif agent_type == AgentType.PEDESTRIAN:
        if hasattr(env_config, "pedestrian_factory"):
            agent = env_config.pedestrian_factory()
        else:
            raise ValueError(
                "Pedestrian agent type requires an env_config with pedestrian_factory method"
            )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Get the action space from the agent
    action_space = agent.action_space

    # Check if image observations are enabled
    use_image_obs = getattr(env_config, "use_image_obs", False)
    image_obs_space = None

    if use_image_obs and hasattr(env_config, "image_config"):
        image_obs_space = image_sensor_space(env_config.image_config)

    # Extend the agent's observation space with additional sensors
    if image_obs_space is not None:
        from robot_sf.sensor.sensor_fusion import fused_sensor_space_with_image

        observation_space, orig_obs_space = fused_sensor_space_with_image(
            env_config.sim_config.stack_steps,
            agent.observation_space,
            target_sensor_space(map_def.max_target_dist),
            lidar_sensor_space(
                env_config.lidar_config.num_rays, env_config.lidar_config.max_scan_dist
            ),
            image_obs_space,
        )
    else:
        observation_space, orig_obs_space = fused_sensor_space(
            env_config.sim_config.stack_steps,
            agent.observation_space,
            target_sensor_space(map_def.max_target_dist),
            lidar_sensor_space(
                env_config.lidar_config.num_rays, env_config.lidar_config.max_scan_dist
            ),
        )

    return action_space, observation_space, orig_obs_space


def init_collision_and_sensors_with_image(
    sim: Simulator,
    env_config: Union[EnvSettings, RobotEnvSettings],
    orig_obs_space: spaces.Dict,
    sim_view=None,
):
    """
    Initializes collision detection and sensor fusion for robots, including optional image sensors.
    
    Creates occupancy objects for collision detection and, for each robot, constructs either a standard sensor fusion or an image sensor fusion object depending on whether image observations are enabled and a simulation view is provided. Returns lists of occupancy and sensor fusion objects for all robots.
    
    Args:
        sim: The simulator instance containing robots and environment state.
        env_config: Environment configuration specifying robot, lidar, and optional image settings.
        orig_obs_space: The original observation space definition.
        sim_view: Optional simulation view used for image sensor input.
    
    Returns:
        A tuple containing a list of occupancy objects for collision detection and a list of sensor fusion objects (standard or image-based) for each robot.
    """
    # Get the number of robots, simulation configuration,
    # robot configuration, and lidar configuration
    num_robots = len(sim.robots)
    sim_config = env_config.sim_config
    robot_config = env_config.robot_config
    lidar_config = env_config.lidar_config

    # Check if image observations are enabled
    use_image_obs = getattr(env_config, "use_image_obs", False)

    # Initialize occupancy objects for each robot for collision detection
    occupancies = [
        ContinuousOccupancy(
            sim.map_def.width,
            sim.map_def.height,
            lambda i=i: sim.robot_pos[i],
            lambda i=i: sim.goal_pos[i],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4],
            lambda: sim.ped_pos,
            robot_config.radius,
            sim_config.ped_radius,
            sim_config.goal_radius,
        )
        for i in range(num_robots)
    ]

    # Initialize sensor fusion objects for each robot
    sensor_fusions = []
    for r_id in range(num_robots):
        # Define the ray sensor, target sensor, and speed sensor for each robot
        def ray_sensor(r_id=r_id):
            """
            Returns the lidar ray scan readings for the specified robot.
            
            Args:
                r_id: The index of the robot for which to compute the ray scan.
            
            Returns:
                The lidar ray scan data for the robot at the given index.
            """
            return lidar_ray_scan(sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]

        def target_sensor(r_id=r_id):
            """
            Generates target sensor observations for a specified robot.
            
            Args:
                r_id: The index of the robot for which to generate the target sensor observation.
            
            Returns:
                The observation from the target sensor for the specified robot, including current pose, goal position, and next goal position.
            """
            return target_sensor_obs(
                sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id]
            )

        def speed_sensor(r_id=r_id):
            """
            Retrieves the current speed of the specified robot.
            
            Args:
                r_id: The index of the robot whose speed is to be returned.
            
            Returns:
                The current speed of the robot.
            """
            return sim.robots[r_id].current_speed

        # Create appropriate sensor fusion based on configuration
        if use_image_obs and sim_view is not None:
            from robot_sf.sensor.image_sensor import ImageSensor

            # Create image sensor
            image_config = getattr(env_config, "image_config", None)
            if image_config is not None:
                image_sensor = ImageSensor(image_config, sim_view)
            else:
                from robot_sf.sensor.image_sensor import ImageSensorSettings

                image_sensor = ImageSensor(ImageSensorSettings(), sim_view)

            # Create image sensor fusion
            sensor_fusion = ImageSensorFusion(
                ray_sensor,
                speed_sensor,
                target_sensor,
                image_sensor,
                orig_obs_space,
                sim_config.use_next_goal,
                use_image_obs=True,
            )
        else:
            # Create regular sensor fusion
            sensor_fusion = SensorFusion(
                ray_sensor,
                speed_sensor,
                target_sensor,
                orig_obs_space,
                sim_config.use_next_goal,
            )

        sensor_fusions.append(sensor_fusion)

    return occupancies, sensor_fusions
