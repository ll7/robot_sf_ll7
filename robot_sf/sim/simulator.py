"""Robot and pedestrian simulation management.

This module provides the core simulation infrastructure for managing robots,
pedestrians, and their interactions in a shared environment. It integrates the
PySocialForce physics engine for pedestrian dynamics and supports both robot-only
and pedestrian-robot interaction scenarios.

Key Components:
    - Simulator: Base simulation engine managing robots, pedestrian physics,
      and navigation waypoints.
    - PedSimulator: Extended simulator with ego pedestrian (robot-as-pedestrian)
      for pedestrian-centric environments.
    - init_simulators: Factory for creating robot-only simulator instances.
    - init_ped_simulators: Factory for creating pedestrian simulator instances.

Example:
    >>> from robot_sf.gym_env.unified_config import RobotSimulationConfig
    >>> from robot_sf.nav.svg_map_parser import load_svg_maps
    >>> config = RobotSimulationConfig()
    >>> maps = load_svg_maps("maps/svg_maps/")
    >>> sims = init_simulators(config, maps["hallway"], num_robots=2)
    >>> for sim in sims:
    ...     sim.step_once([action1, action2])"""

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

from robot_sf.common.types import PedPose, RobotAction, RobotPose, Vec2D
from robot_sf.gym_env.env_config import EnvSettings, PedEnvSettings, SimulationSettings
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.navigation import RouteNavigator, get_prepared_obstacles, sample_route
from robot_sf.nav.occupancy import is_circle_line_intersection
from robot_sf.ped_ego.unicycle_drive import UnicycleAction, UnicycleDrivePedestrian
from robot_sf.ped_npc.ped_behavior import PedestrianBehavior, SinglePedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation
from robot_sf.ped_npc.ped_robot_force import PedRobotForce
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.robot.robot_state import Robot


@dataclass
class Simulator:
    """Manages robot and pedestrian simulation in a shared environment.

    Coordinates robot navigation, pedestrian dynamics via PySocialForce,
    collision detection, and timestep synchronization. Automatically initializes
    pedestrian spawn locations, behaviors, and navigation routes on creation.

    Attributes:
        config: Simulation settings (timestep, pedestrian density, forces).
        map_def: Map definition with obstacles, spawn zones, routes.
        robots: List of Robot instances in the environment.
        goal_proximity_threshold: Distance threshold for waypoint arrival (robot radius + goal radius).
        random_start_pos: If True, robots spawn at random valid positions; else, assigned positions.
        robot_navs: (init=False) RouteNavigator instances tracking waypoints.
        pysf_sim: (init=False) PySocialForce simulator managing pedestrian physics.
        pysf_state: (init=False) Pedestrian state snapshots (positions, velocities).
        groups: (init=False) Pedestrian group assignments for crowd behavior.
        peds_behaviors: (init=False) Behavior instances (goal selection, group dynamics).
        peds_have_obstacle_forces: Enable pedestrian-obstacle collision forces.
            Note: Activating increases simulation duration by ~40%.
        last_ped_forces: (init=False, repr=False) Last computed pedestrian forces (K, 2).
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
        """Initialize simulator components after dataclass construction.

        Sets up pedestrian spawn locations, groups, and behaviors; configures
        PySocialForce physics engine with optional obstacle/interaction forces;
        initializes robot navigation paths; and resets all agents to start state.
        Route spawning honors SimulationSettings route spawn options when provided.
        """
        pysf_config = PySFSimConfig()
        spawn_config = PedSpawnConfig(
            self.config.peds_per_area_m2,
            self.config.max_peds_per_group,
            route_spawn_distribution=self.config.route_spawn_distribution,
            route_spawn_jitter_frac=self.config.route_spawn_jitter_frac,
            route_spawn_seed=self.config.route_spawn_seed,
        )
        self.pysf_state, self.groups, self.peds_behaviors = populate_simulation(
            pysf_config.scene_config.tau,
            spawn_config,
            self.map_def.ped_routes,
            self.map_def.ped_crowded_zones,
            obstacle_polygons=get_prepared_obstacles(self.map_def),
            single_pedestrians=self.map_def.single_pedestrians,
            time_step_s=self.config.time_per_step_in_secs,
            single_ped_goal_threshold=pysf_config.desired_force_config.goal_threshold,
        )
        for behavior in self.peds_behaviors:
            if isinstance(behavior, SinglePedestrianBehavior):
                behavior.set_robot_pose_provider(lambda: self.robot_poses)

        if self.peds_have_obstacle_forces is None:
            logger.warning(
                "The peds_have_obstacle_forces attribute is not set. "
                "This may lead to unexpected behavior."
                "Setting it to False by default.",
            )
            self.peds_have_obstacle_forces = False

        def make_forces(sim: PySFSimulator, config: PySFSimConfig) -> list[PySFForce]:
            """Configure pedestrian forces for the physics engine.

            Creates default SocialForce forces, optionally filters obstacle forces,
            and adds pedestrian-robot interaction forces if enabled.

            Args:
                sim: PySocialForce simulator instance.
                config: PySocialForce configuration object.

            Returns:
                List of Force objects including social, goal attraction, obstacle
                (conditional), and pedestrian-robot interaction (conditional) forces.
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
        """Current goal waypoint for each robot navigator."""
        return [n.current_waypoint for n in self.robot_navs]

    @property
    def next_goal_pos(self) -> list[Vec2D | None]:
        """Next waypoint for each robot navigator (None if at route end)."""
        return [n.next_waypoint for n in self.robot_navs]

    @property
    def robot_poses(self) -> list[RobotPose]:
        """Current poses (position + orientation) of all robots."""
        return [r.pose for r in self.robots]

    @property
    def robot_pos(self) -> list[Vec2D]:
        """Current (x, y) positions of all robots."""
        return [r.pose[0] for r in self.robots]

    @property
    def ped_pos(self):
        """Current (x, y) positions of all pedestrians."""
        return self.pysf_state.ped_positions

    @property
    def ped_vel(self):
        """Current (vx, vy) velocities of all pedestrians."""
        return self.pysf_state.ped_velocities

    def reset_state(self):
        """Reset robot navigation and spawn positions.

        Reassigns routes and respawns robots when they collide or reach
        their destination goal. Updates are necessary for episodic reset
        or continuous replay scenarios.
        """
        for i, (robot, nav) in enumerate(zip(self.robots, self.robot_navs, strict=False)):
            collision = not nav.reached_waypoint
            is_at_final_goal = nav.reached_destination
            if collision or is_at_final_goal:
                waypoints = sample_route(self.map_def, None if self.random_start_pos else i)
                nav.new_route(waypoints[1:])
                robot.reset_state((waypoints[0], nav.initial_orientation))

    def step_once(self, actions: list[RobotAction]):
        """Advance simulation by one timestep.

        Updates pedestrian behaviors and physics (via PySocialForce), applies
        robot actions, and updates navigation state. Called once per episode
        timestep.

        Args:
            actions: Control actions for each robot (velocity, angular velocity, etc.).
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
    peds_have_obstacle_forces: bool = True,
) -> list[Simulator]:
    """Initialize one or more simulator instances for the robot environment.

    Args:
        env_config: Environment configuration containing simulator/robot settings.
        map_def: Map definition describing start positions, goals, and obstacles.
        num_robots: Total number of robots to simulate across instances.
        random_start_pos: Whether robots start at random spawn positions.
        peds_have_obstacle_forces: Whether pedestrians experience obstacle forces.

    Returns:
        list[Simulator]: Simulator instances sized to cover ``num_robots`` robots.
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
    """Extended simulator with ego pedestrian in a multi-agent scenario.

    Inherits robot and NPC pedestrian management from Simulator, adding a
    controllable ego pedestrian (e.g., human surrogate or trained robot-as-ped).
    Supports pedestrian-centric observations and action spaces.

    Attributes:
        ego_ped: Controllable pedestrian instance (typically UnicycleDrivePedestrian).
    """

    ego_ped: UnicycleDrivePedestrian

    @property
    def ego_ped_pos(self) -> Vec2D:
        """Return the current 2D position of the ego pedestrian.

        Returns:
            Vec2D: The (x, y) coordinates of the ego pedestrian.
        """
        return self.ego_ped.pos

    @property
    def ego_ped_pose(self) -> PedPose:
        """Return the current pose of the ego pedestrian.

        Returns:
            PedPose: The full pose including position and orientation of the ego pedestrian.
        """
        return self.ego_ped.pose

    @property
    def ego_ped_goal_pos(self) -> Vec2D:
        """Return the goal position for the ego pedestrian (robot position).

        Returns:
            Vec2D: The (x, y) coordinates of the first robot, which serves as the goal.
        """
        return self.robots[0].pos

    def reset_state(self):
        """Reset robot and ego pedestrian state.

        Calls parent reset_state() to reassign robot routes, then spawns
        the ego pedestrian at a random valid location 10-15 units away
        from the first robot.
        """
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
        """Advance simulation with robot and ego pedestrian actions.

        Updates pedestrian behaviors and physics, applies robot actions,
        applies ego pedestrian actions, and updates navigation.

        Args:
            actions: Control actions for each robot.
            ego_ped_actions: Control actions for the ego pedestrian.
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

        self.ego_ped.apply_action(ego_ped_actions[0], self.config.time_per_step_in_secs)

    def get_proximity_point(
        self,
        fixed_point: tuple[float, float],
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[float, float]:
        """Sample a collision-free point at a given distance from a reference point.

        Attempts up to 10 times to find a valid point within the distance bounds,
        checking for obstacle collisions and map bounds. Falls back to random
        pedestrian spawn zone if unsuccessful.

        Args:
            fixed_point: Reference (x, y) coordinates.
            lower_bound: Minimum distance from fixed_point.
            upper_bound: Maximum distance from fixed_point.

        Returns:
            Tuple of (x, y) for collision-free point, or fallback spawn location.
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
        """Check if a position collides with obstacles or is outside map bounds.

        Validates both map boundary containment and obstacle collision using
        circle-line intersection with ego pedestrian radius. Adapted from
        occupancy.py for spawn validation.

        Args:
            x: X coordinate to check.
            y: Y coordinate to check.

        Returns:
            True if position is out of bounds or collides with an obstacle,
            False if position is collision-free and within bounds.
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
    peds_have_obstacle_forces: bool = True,
) -> list[PedSimulator]:
    """Create a pedestrian-centric simulator instance.

    Factory function for initializing a PedSimulator with one robot and one
    controllable ego pedestrian. Validates map definition and initializes
    navigation, physics, and spawn configurations.

    Args:
        env_config: Pedestrian environment settings (robot/ped config, physics).
        map_def: Map with obstacles, spawn zones, routes.
        random_start_pos: If False, use deterministic spawn; if True, randomize.
        peds_have_obstacle_forces: Enable pedestrian-obstacle collision forces.

    Returns:
        Single-element list containing initialized PedSimulator instance.
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
