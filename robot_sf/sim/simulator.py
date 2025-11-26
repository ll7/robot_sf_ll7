from dataclasses import dataclass, field
from math import ceil, cos, pi, sin
from random import sample, uniform

import numpy as np
from loguru import logger
from pysocialforce import Simulator as PySFSimulator
from pysocialforce.config import SimulatorConfig as PySFSimConfig
from pysocialforce.forces import Force as PySFForce
from pysocialforce.forces import ObstacleForce
from pysocialforce.simulator import make_forces as pysf_make_forces

from robot_sf.common.types import RobotAction, RobotPose, Vec2D
from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings, SimulationSettings
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.navigation import RouteNavigator, sample_route
from robot_sf.nav.occupancy import is_circle_line_intersection
from robot_sf.ped_ego.unicycle_drive import UnicycleAction, UnicycleDrivePedestrian
from robot_sf.ped_npc.ped_behavior import PedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation
from robot_sf.ped_npc.ped_robot_force import PedRobotForce
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.robot.robot_state import Robot


@dataclass
class Simulator:
    """
    Simulator class to manage the simulation environment, including robots,
    pedestrians, and their interactions based on the provided configuration.

    Args:
        config (SimulationSettings): Configuration settings for the simulation.
        map_def (MapDefinition): Definition of the map for the environment.
        robots (List[Robot]): List of robots in the environment.
        goal_proximity_threshold (float): Proximity to the goal for the robots.
        random_start_pos (bool): Whether to start the robots at random positions.

        robot_navs (List[RouteNavigator]): List of robot routes.
        pysf_sim (PySFSimulator): PySocialForce simulator object.
        pysf_state (PedestrianStates): PySocialForce pedestrian states.
        groups (PedestrianGroupings): PySocialForce pedestrian groups.
        peds_behaviors (List[PedestrianBehavior]): List of pedestrian behaviors.
        peds_have_obstacle_forces (bool): Whether pedestrians have obstacle forces.
            Activating this increases the simulation duration by 40%.
    """

    config: SimulationSettings
    map_def: MapDefinition
    robots: list[Robot]
    goal_proximity_threshold: float
    random_start_pos: bool
    robot_navs: list[RouteNavigator] = field(init=False)
    pysf_sim: PySFSimulator = field(init=False)
    pysf_state: PedestrianStates = field(init=False)
    groups: PedestrianGroupings = field(init=False)
    peds_behaviors: list[PedestrianBehavior] = field(init=False)
    peds_have_obstacle_forces: bool
    # Last pedestrian force vectors used to step the simulation (K,2)
    last_ped_forces: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the simulator components after dataclass initialization.
        Sets up pedestrian states, groups, behaviors, simulation forces,
        and robot navigators.
        """
        pysf_config = PySFSimConfig()
        spawn_config = PedSpawnConfig(self.config.peds_per_area_m2, self.config.max_peds_per_group)
        self.pysf_state, self.groups, self.peds_behaviors = populate_simulation(
            pysf_config.scene_config.tau,
            spawn_config,
            self.map_def.ped_routes,
            self.map_def.ped_crowded_zones,
            self.map_def.single_pedestrians,
        )

        if self.peds_have_obstacle_forces is None:
            logger.warning(
                "The peds_have_obstacle_forces attribute is not set. "
                "This may lead to unexpected behavior."
                "Setting it to False by default.",
            )
            self.peds_have_obstacle_forces = False

        def make_forces(sim: PySFSimulator, config: PySFSimConfig) -> list[PySFForce]:
            """
            Creates and configures the forces to be applied in the simulation,
            excluding obstacle forces and adding pedestrian-robot interaction forces
            if PRF is active.
            """
            forces = pysf_make_forces(sim, config)

            if self.peds_have_obstacle_forces is False:
                logger.info("Peds have no obstacle forces.")
                # if peds have no obstacle forces, we filter the obstacle forces and remove them
                forces = [f for f in forces if not isinstance(f, ObstacleForce)]
            if self.config.prf_config.is_active:
                for robot in self.robots:
                    self.config.prf_config.robot_radius = robot.config.radius
                    forces.append(
                        PedRobotForce(self.config.prf_config, sim.peds, lambda: robot.pos),
                    )
            return forces

        self.pysf_sim = PySFSimulator(
            self.pysf_state.pysf_states(),
            self.groups.groups_as_lists,
            self.map_def.obstacles_pysf,
            config=pysf_config,
            make_forces=make_forces,
        )
        self.pysf_sim.peds.step_width = self.config.time_per_step_in_secs
        self.pysf_sim.peds.max_speed_multiplier = self.config.peds_speed_mult
        self.robot_navs = [
            RouteNavigator(proximity_threshold=self.goal_proximity_threshold) for _ in self.robots
        ]

        self.last_ped_forces = np.zeros((0, 2), dtype=float)

        self.reset_state()
        for behavior in self.peds_behaviors:
            behavior.reset()

    @property
    def goal_pos(self) -> list[Vec2D]:
        """
        Returns the current goal positions for all robot navigators.
        """
        return [n.current_waypoint for n in self.robot_navs]

    @property
    def next_goal_pos(self) -> list[Vec2D | None]:
        """
        Returns the next goal positions for all robot navigators.
        """
        return [n.next_waypoint for n in self.robot_navs]

    @property
    def robot_poses(self) -> list[RobotPose]:
        """
        Returns the current poses of all robots.
        """
        return [r.pose for r in self.robots]

    @property
    def robot_pos(self) -> list[Vec2D]:
        """
        Returns the current positions of all robots.
        """
        return [r.pose[0] for r in self.robots]

    @property
    def ped_pos(self):
        """
        Returns the current positions of all pedestrians.
        """
        return self.pysf_state.ped_positions

    def reset_state(self):
        """
        Resets the state of all robots and their navigators.
        Assigns new routes and resets robot positions if collision occurs
        or a robot has reached its final goal.
        """
        for i, (robot, nav) in enumerate(zip(self.robots, self.robot_navs, strict=False)):
            collision = not nav.reached_waypoint
            is_at_final_goal = nav.reached_destination
            if collision or is_at_final_goal:
                waypoints = sample_route(self.map_def, None if self.random_start_pos else i)
                nav.new_route(waypoints[1:])
                robot.reset_state((waypoints[0], nav.initial_orientation))

    def step_once(self, actions: list[RobotAction]):
        """
        Performs a single simulation step by updating pedestrian behaviors,
        computing and applying forces, updating pedestrian positions,
        and applying robot actions and navigation updates.
        """
        for behavior in self.peds_behaviors:
            behavior.step()
        ped_forces = self.pysf_sim.compute_forces()
        self.last_ped_forces = np.asarray(ped_forces, dtype=float)
        groups = self.groups.groups_as_lists
        self.pysf_sim.peds.step(ped_forces, groups)
        for robot, nav, action in zip(self.robots, self.robot_navs, actions, strict=False):
            robot.apply_action(action, self.config.time_per_step_in_secs)
            nav.update_position(robot.pos)


def init_simulators(
    env_config: EnvSettings | RobotSimulationConfig,
    map_def: MapDefinition,
    num_robots: int = 1,
    random_start_pos: bool = True,
    peds_have_obstacle_forces: bool = False,
) -> list[Simulator]:
    """
    Initialize simulators for the robot environment.

    Parameters:
    env_config (EnvSettings): Configuration settings for the environment.
    map_def (MapDefinition): Definition of the map for the environment.
    num_robots (int): Number of robots in the environment.
    random_start_pos (bool): Whether to start the robots at random positions.


    Returns:
    List[Simulator]: A list of initialized Simulator objects.
    """
    # assert that the map_def has the correct type
    try:
        assert isinstance(map_def, MapDefinition)
    except AssertionError:
        # rasie type error and print the type of map_def
        raise TypeError(f"map_def should be of type MapDefinition, got {type(map_def)}")

    # Calculate the number of simulators needed based on the number of robots and start positions
    if map_def.num_start_pos <= 0:
        # Defensive guard: division-by-zero would occur and route sampling will fail later anyway.
        raise ValueError(
            "Cannot initialize simulators: map definition provides zero robot start positions "
            "(no robot routes detected). Ensure the map JSON/SVG conversion produced robot_routes "
            "and that spawn/goal zones plus routes are present.",
        )

    num_sims = ceil(num_robots / map_def.num_start_pos)

    # Calculate the proximity to the goal based on the robot radius and goal radius
    goal_proximity = env_config.robot_config.radius + env_config.sim_config.goal_radius

    # Initialize an empty list to hold the simulators
    sims: list[Simulator] = []

    # Create the required number of simulators
    for i in range(num_sims):
        # Determine the number of robots for this simulator
        n = (
            map_def.num_start_pos
            if i < num_sims - 1
            else max(1, num_robots % map_def.num_start_pos)
        )

        # Create the robots for this simulator
        sim_robots = [env_config.robot_factory() for _ in range(n)]

        # Create the simulator with the robots and add it to the list
        sim = Simulator(
            config=env_config.sim_config,
            map_def=map_def,
            robots=sim_robots,
            goal_proximity_threshold=goal_proximity,
            random_start_pos=random_start_pos,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )
        sims.append(sim)

    return sims


@dataclass
class PedSimulator(Simulator):
    """
    A pedestrian simulator, which extends the base simulator.

    Args:
        ego_ped (UnicycleDrivePedestrian): The ego pedestrian in the environment.

    """

    ego_ped: UnicycleDrivePedestrian

    @property
    def ego_ped_pos(self) -> Vec2D:
        return self.ego_ped.pos

    @property
    def ego_ped_pose(self) -> Vec2D:
        return self.ego_ped.pose

    @property
    def ego_ped_goal_pos(self) -> Vec2D:
        return self.robots[0].pos

    def reset_state(self):
        for i, (robot, nav) in enumerate(zip(self.robots, self.robot_navs, strict=False)):
            collision = not nav.reached_waypoint
            is_at_final_goal = nav.reached_destination
            if collision or is_at_final_goal:
                waypoints = sample_route(self.map_def, None if self.random_start_pos else i)
                nav.new_route(waypoints[1:])
                robot.reset_state((waypoints[0], nav.initial_orientation))
        # Ego_pedestrian reset
        robot_spawn = self.robot_pos[0]
        ped_spawn = self.get_proximity_point(robot_spawn, 10, 15)
        self.ego_ped.reset_state((ped_spawn, self.ego_ped.pose[1]))

    def step_once(self, actions: list[RobotAction], ego_ped_actions: list[UnicycleAction]):
        for behavior in self.peds_behaviors:
            behavior.step()
        ped_forces = self.pysf_sim.compute_forces()
        self.last_ped_forces = np.asarray(ped_forces, dtype=float)
        groups = self.groups.groups_as_lists
        self.pysf_sim.peds.step(ped_forces, groups)
        for robot, nav, action in zip(self.robots, self.robot_navs, actions, strict=False):
            robot.apply_action(action, self.config.time_per_step_in_secs)
            nav.update_position(robot.pos)

        self.ego_ped.apply_action(ego_ped_actions[0], self.config.time_per_step_in_secs)

    def get_proximity_point(
        self,
        fixed_point: tuple[float, float],
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[float, float]:
        """
        Calculate a point in the proximity of another point with specified distance bounds.

        Args:
            fixed_point (tuple): (x, y) The original point.
            lower_bound (float): The minimum distance from the original point.
            upper_bound (float): The maximum distance from the original point.

        Returns:
            tuple: A tuple containing the new x and y coordinates.
        """
        x, y = fixed_point
        for _ in range(10):
            angle = uniform(0, 2 * pi)
            distance = uniform(lower_bound, upper_bound)

            new_x = x + distance * cos(angle)
            new_y = y + distance * sin(angle)
            if not self.is_obstacle_collision(new_x, new_y):
                return new_x, new_y

        logger.warning(f"Could not find a valid proximity point: {fixed_point}.")
        spawn_id = sample(self.map_def.ped_spawn_zones, k=1)[0]  # Spawn in pedestrian spawn_zone
        initial_spawn = sample_zone(spawn_id, 1)[0]
        return initial_spawn

    def is_obstacle_collision(self, x: float, y: float) -> bool:
        """
        TODO: this method is copied from occupancy.py
        """
        if not (0 <= x <= self.map_def.width and 0 <= y <= self.map_def.height):
            return True

        collision_distance = self.ego_ped.config.radius
        circle_agent = ((x, y), collision_distance)
        for s_x, s_y, e_x, e_y in self.pysf_sim.env.obstacles_raw[:, :4]:
            if is_circle_line_intersection(circle_agent, ((s_x, s_y), (e_x, e_y))):
                return True
        return False


def init_ped_simulators(
    env_config: PedEnvSettings,
    map_def: MapDefinition,
    random_start_pos: bool = False,
    peds_have_obstacle_forces: bool = False,
) -> list[PedSimulator]:
    """
    Initialize simulators for the pedestrian environment.

    Parameters:
    env_config (PedEnvSettings): Configuration settings for the environment.
    map_def (MapDefinition): Definition of the map for the environment.
    num_robots (int): Number of robots in the environment.
    random_start_pos (bool): Whether to start the robots at random positions.

    Returns:
    sim (PedSimulator): A Simulator object for the pedestrian environment.
    """

    # Calculate the proximity to the goal based on the robot radius and goal radius
    goal_proximity = env_config.robot_config.radius + env_config.sim_config.goal_radius

    # Create the robots for this simulator
    sim_robot = env_config.robot_factory()

    # Create the pedestrian for this simulator
    sim_ped = env_config.pedestrian_factory()

    # Create the simulator with the robots and add it to the list
    sim = PedSimulator(
        env_config.sim_config,
        map_def,
        [sim_robot],
        goal_proximity,
        random_start_pos,
        ego_ped=sim_ped,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
    )

    return [sim]
