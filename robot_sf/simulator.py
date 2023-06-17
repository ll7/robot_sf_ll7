import random
from math import dist
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Callable

import pysocialforce as pysf
from pysocialforce.utils import SimulatorConfig as PySFSimConfig

from robot_sf.sim_config import SimulationSettings
from robot_sf.map_config import MapDefinition
from robot_sf.ped_spawn_generator \
    import ZonePointsGenerator, PedSpawnConfig, initialize_pedestrians
from robot_sf.ped_robot_force import PedRobotForce
from robot_sf.pedestrian_grouping \
    import GroupRedirectBehavior, PySFPedestrianStates, PedestrianGroupings
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveAction
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleAction


Vec2D = Tuple[float, float]
RobotAction = Union[DifferentialDriveAction, BicycleAction]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
Robot = Union[DifferentialDriveRobot, BicycleDriveRobot]


@dataclass
class RouteNavigator:
    waypoints: List[Vec2D] = field(default_factory=list)
    waypoint_id: int = 0
    proximity_threshold: float = 1.0 # info: should be set to vehicle radius + goal radius
    pos: Vec2D = field(default=(0, 0))
    reached_waypoint: bool = False

    @property
    def reached_destination(self) -> bool:
        return len(self.waypoints) == 0 or \
            dist(self.waypoints[-1], self.pos) <= self.proximity_threshold

    @property
    def current_waypoint(self) -> Vec2D:
        return self.waypoints[self.waypoint_id]

    @property
    def next_waypoint(self) -> Union[Vec2D, None]:
        return self.waypoints[self.waypoint_id + 1] \
            if self.waypoint_id + 1 < len(self.waypoints) else None

    def update_position(self, pos: Vec2D):
        reached_waypoint = dist(self.current_waypoint, pos) <= self.proximity_threshold
        if reached_waypoint:
            self.waypoint_id = min(len(self.waypoints) - 1, self.waypoint_id + 1)
        self.pos = pos
        self.reached_waypoint = reached_waypoint

    def new_route(self, route: List[Vec2D]):
        self.waypoints = route
        self.waypoint_id = 0


def sample_route(
        map_def: MapDefinition,
        spawn_gens: List[ZonePointsGenerator],
        goal_gens: List[ZonePointsGenerator]) -> List[Vec2D]:

    def generate_point(zone: ZonePointsGenerator) -> Vec2D:
            points, zone_id = zone.generate(num_samples=1)
            return points[0]

    spawn_id = random.randint(0, len(spawn_gens) - 1)
    goal_id = random.randint(0, len(goal_gens) - 1)
    route = map_def.find_route(spawn_id, goal_id)
    initial_spawn = generate_point(spawn_gens[route.spawn_id])
    final_goal = generate_point(goal_gens[route.goal_id])
    route = [initial_spawn] + route.waypoints + [final_goal]
    # TODO: think of adding a bit of noise to the exact waypoint positions as well
    return route


@dataclass
class Simulator:
    config: SimulationSettings
    map_def: MapDefinition
    robot: Robot
    goal_proximity_threshold: float
    navigator: RouteNavigator = field(init=False)
    pysf_sim: pysf.Simulator = field(init=False)
    sample_route: Callable[[], List[Vec2D]] = field(init=False)
    pysf_state: PySFPedestrianStates = field(init=False)
    groups: PedestrianGroupings = field(init=False)
    peds_behavior: GroupRedirectBehavior = field(init=False)

    def __post_init__(self):
        ped_spawn_gens = [ZonePointsGenerator([z]) for z in self.map_def.ped_spawn_zones]
        robot_spawn_gens = [ZonePointsGenerator([z]) for z in self.map_def.robot_spawn_zones]
        robot_goal_gens = [ZonePointsGenerator([z]) for z in self.map_def.goal_zones]

        self.sample_route = lambda: sample_route(
            self.map_def, robot_spawn_gens, robot_goal_gens)

        spawn_config = PedSpawnConfig(self.config.peds_per_area_m2, self.config.max_peds_per_group)
        ped_states_np, initial_groups, zone_assignments = \
            initialize_pedestrians(spawn_config, self.map_def.ped_spawn_zones)

        self.pysf_state = PySFPedestrianStates(lambda: self.pysf_sim.peds.state)
        self.groups = PedestrianGroupings(self.pysf_state)
        self.peds_behavior = GroupRedirectBehavior(self.groups, zone_assignments, ped_spawn_gens)

        for ped_ids in initial_groups:
            self.groups.new_group(ped_ids)

        def make_forces(sim: pysf.Simulator, config: PySFSimConfig) -> List[pysf.forces.Force]:
            forces = pysf.simulator.make_forces(sim, config)
            forces = [f for f in forces if type(f) != pysf.forces.ObstacleForce]
            if self.config.prf_config.is_active:
                forces.append(PedRobotForce(
                    self.config.prf_config, sim.peds, lambda: self.robot.pos))
            return forces

        self.pysf_sim = pysf.Simulator(
            ped_states_np, self.groups.groups_as_lists,
            self.map_def.obstacles_pysf, make_forces=make_forces)
        self.pysf_sim.peds.step_width = self.config.time_per_step_in_secs
        self.pysf_sim.peds.max_speed_multiplier = self.config.peds_speed_mult
        self.navigator = RouteNavigator(proximity_threshold=self.goal_proximity_threshold)
        self.reset_state()

    @property
    def goal_pos(self) -> Vec2D:
        return self.navigator.current_waypoint

    @property
    def next_goal_pos(self) -> Union[Vec2D, None]:
        return self.navigator.next_waypoint

    @property
    def robot_pose(self) -> RobotPose:
        return self.robot.pose

    @property
    def ped_positions(self):
        return self.pysf_state.ped_positions

    def reset_state(self):
        self.peds_behavior.pick_new_goals()
        collision = not self.navigator.reached_waypoint
        is_at_final_goal = self.navigator.reached_destination
        if collision or is_at_final_goal:
            waypoints = self.sample_route()
            self.navigator.new_route(waypoints[1:])
            self.robot.reset_state((waypoints[0], 0))

    def step_once(self, action: RobotAction):
        self.peds_behavior.redirect_groups_if_at_goal()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups.groups_as_lists
        self.pysf_sim.peds.step(ped_forces, groups)
        self.robot.apply_action(action, self.config.time_per_step_in_secs)
        self.navigator.update_position(self.robot.pos)
