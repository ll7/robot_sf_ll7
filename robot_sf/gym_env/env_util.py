"""
env_util
"""

from enum import Enum

import numpy as np
from gymnasium import spaces

from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings, RobotEnvSettings
from robot_sf.gym_env.unified_config import RobotSimulationConfig
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
    sim: Simulator,
    env_config: EnvSettings | RobotSimulationConfig,
    orig_obs_space: spaces.Dict,
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
    sensor_fusions: list[SensorFusion] = []
    for r_id in range(num_robots):
        # Define the ray sensor, target sensor, and speed sensor for each robot
        def ray_sensor(r_id=r_id):
            return lidar_ray_scan(sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]

        def target_sensor(r_id=r_id):
            return target_sensor_obs(
                sim.robots[r_id].pose,
                sim.goal_pos[r_id],
                sim.next_goal_pos[r_id],
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
            ),
        )

    return occupancies, sensor_fusions


def init_spaces(env_config: EnvSettings | RobotSimulationConfig, map_def: MapDefinition):
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
        env_config,
        map_def,
        agent_type=AgentType.ROBOT,
    )
    # Return the action space, the extended observation space, and the original
    # observation space
    return action_space, obs_space, orig_obs_space


def create_spaces(
    env_config: EnvSettings | PedEnvSettings | RobotSimulationConfig,
    map_def: MapDefinition,
    agent_type: AgentType = AgentType.ROBOT,
):
    # Create a agent using the factory method in the environment configuration
    if agent_type == AgentType.ROBOT:
        agent = env_config.robot_factory()
    elif agent_type == AgentType.PEDESTRIAN:
        if hasattr(env_config, "pedestrian_factory"):
            agent = env_config.pedestrian_factory()  # type: ignore[union-attr]
        else:
            raise ValueError(
                "Pedestrian agent type requires an env_config with pedestrian_factory method",
            )
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
        env_config,
        map_def,
        agent_type=AgentType.ROBOT,
    )
    action_space_ped, obs_space_ped, orig_obs_space_ped = create_spaces(
        env_config,
        map_def,
        agent_type=AgentType.PEDESTRIAN,
    )

    # As a list [robot, pedestrian]
    return (
        [action_space_robot, action_space_ped],
        [obs_space_robot, obs_space_ped],
        [orig_obs_space_robot, orig_obs_space_ped],
    )


def init_ped_collision_and_sensors(
    sim: PedSimulator,
    env_config: PedEnvSettings,
    orig_obs_space: list[spaces.Dict],
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

    occupancies: list[ContinuousOccupancy | EgoPedContinuousOccupancy] = []
    sensor_fusions: list[SensorFusion] = []

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
        ),
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
        ),
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
        ),
    )

    def ray_sensor_ego_ped():
        return lidar_ray_scan(sim.ego_ped.pose, occupancies[1], lidar_config)[0]

    def target_sensor_ego_ped():
        return target_sensor_obs(
            sim.ego_ped.pose,
            sim.ego_ped_goal_pos,
            None,
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
        ),
    )  # Ego pedestrian does not have a next goal

    # Format: [robot, ego_pedestrian]
    return occupancies, sensor_fusions


def create_spaces_with_image(
    env_config: EnvSettings | PedEnvSettings | RobotEnvSettings,
    map_def: MapDefinition,
    agent_type: AgentType = AgentType.ROBOT,
):
    """
    Create observation and action spaces including optional image observations.

    Parameters
    ----------
    env_config : Union[EnvSettings, PedEnvSettings]
        The configuration settings for the environment.
    map_def : MapDefinition
        The definition of the map for the environment.
    agent_type : AgentType
        The type of agent (robot or pedestrian).

    Returns
    -------
    Tuple[Space, Space, Space]
        A tuple containing the action space, the extended observation space, and
        the original observation space of the agent.
    """
    # Create a agent using the factory method in the environment configuration
    if agent_type == AgentType.ROBOT:
        agent = env_config.robot_factory()
    elif agent_type == AgentType.PEDESTRIAN:
        if hasattr(env_config, "pedestrian_factory"):
            agent = env_config.pedestrian_factory()  # type: ignore[union-attr]
        else:
            raise ValueError(
                "Pedestrian agent type requires an env_config with pedestrian_factory method",
            )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Get the action space from the agent
    action_space = agent.action_space

    # Check if image observations are enabled
    use_image_obs = getattr(env_config, "use_image_obs", False)
    image_obs_space = None

    if use_image_obs and hasattr(env_config, "image_config"):
        image_obs_space = image_sensor_space(env_config.image_config)  # type: ignore[arg-type]

    # Extend the agent's observation space with additional sensors
    if image_obs_space is not None:
        from robot_sf.sensor.sensor_fusion import fused_sensor_space_with_image

        observation_space, orig_obs_space = fused_sensor_space_with_image(
            env_config.sim_config.stack_steps,
            agent.observation_space,
            target_sensor_space(map_def.max_target_dist),
            lidar_sensor_space(
                env_config.lidar_config.num_rays,
                env_config.lidar_config.max_scan_dist,
            ),
            image_obs_space,
        )
    else:
        observation_space, orig_obs_space = fused_sensor_space(
            env_config.sim_config.stack_steps,
            agent.observation_space,
            target_sensor_space(map_def.max_target_dist),
            lidar_sensor_space(
                env_config.lidar_config.num_rays,
                env_config.lidar_config.max_scan_dist,
            ),
        )

    return action_space, observation_space, orig_obs_space


def init_collision_and_sensors_with_image(
    sim: Simulator,
    env_config: EnvSettings | RobotEnvSettings,
    orig_obs_space: spaces.Dict,
    sim_view=None,
):
    """
    Initialize collision detection and sensor fusion including image sensors for the robots.

    Parameters:
    sim (Simulator): The simulator object.
    env_config (EnvSettings): Configuration settings for the environment.
    orig_obs_space (spaces.Dict): Original observation space.
    sim_view: The simulation view for capturing images (optional).

    Returns:
    Tuple[List[ContinuousOccupancy], List[Union[SensorFusion, ImageSensorFusion]]]:
        A tuple containing a list of occupancy objects for collision detection
        and a list of sensor fusion objects for sensor data handling.
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
            return lidar_ray_scan(sim.robots[r_id].pose, occupancies[r_id], lidar_config)[0]

        def target_sensor(r_id=r_id):
            return target_sensor_obs(
                sim.robots[r_id].pose,
                sim.goal_pos[r_id],
                sim.next_goal_pos[r_id],
            )

        def speed_sensor(r_id=r_id):
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


def prepare_pedestrian_actions(simulator) -> np.ndarray:
    """
    Prepare pedestrian action visualization data.

    This helper function creates pedestrian action vectors for visualization
    by combining pedestrian positions with their velocity vectors.

    Args:
        simulator: The simulator object containing pysf_sim with pedestrian data

    Returns:
        np.ndarray: Array of shape (n_peds, 2, 2) where each pedestrian has
                   a start position [x, y] and end position [x, y] representing
                   their current position and position + velocity vector
    """
    ped_actions = zip(
        simulator.pysf_sim.peds.pos(),
        simulator.pysf_sim.peds.pos() + simulator.pysf_sim.peds.vel(),
        strict=False,
    )
    return np.array([[pos, vel] for pos, vel in ped_actions])
