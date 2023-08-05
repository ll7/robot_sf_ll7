from math import atan2
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Callable

from pysocialforce import Simulator as PySFSimulator
from pysocialforce.simulator import make_forces as pysf_make_forces
from pysocialforce.utils import SimulatorConfig as PySFSimConfig
from pysocialforce.forces import Force as PySFForce, ObstacleForce

from robot_sf.sim_config import SimulationSettings
from robot_sf.nav.map_config import MapDefinition
from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation
from robot_sf.ped_npc.ped_robot_force import PedRobotForce
from robot_sf.ped_npc.ped_grouping import PedestrianStates, PedestrianGroupings
from robot_sf.ped_npc.ped_behavior import PedestrianBehavior
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveAction
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleAction
from robot_sf.nav.navigation import RouteNavigator, sample_route


Vec2D = Tuple[float, float]
RobotAction = Union[DifferentialDriveAction, BicycleAction]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
Robot = Union[DifferentialDriveRobot, BicycleDriveRobot]


@dataclass
class Simulator:
    config: SimulationSettings
    map_def: MapDefinition
    robots: List[Robot]
    goal_proximity_threshold: float
    robot_navs: List[RouteNavigator] = field(init=False)
    pysf_sim: PySFSimulator = field(init=False)
    sample_route: Callable[[], List[Vec2D]] = field(init=False)
    pysf_state: PedestrianStates = field(init=False)
    groups: PedestrianGroupings = field(init=False)
    peds_behaviors: List[PedestrianBehavior] = field(init=False)

    def __post_init__(self):
        pysf_config = PySFSimConfig()
        spawn_config = PedSpawnConfig(self.config.peds_per_area_m2, self.config.max_peds_per_group)
        self.pysf_state, self.groups, self.peds_behaviors = populate_simulation(
            pysf_config.scene_config.tau, spawn_config,
            self.map_def.ped_routes, self.map_def.ped_crowded_zones)

        def make_forces(sim: PySFSimulator, config: PySFSimConfig) -> List[PySFForce]:
            forces = pysf_make_forces(sim, config)
            forces = [f for f in forces if type(f) != ObstacleForce]
            if self.config.prf_config.is_active:
                for robot in self.robots:
                    self.config.prf_config.robot_radius = robot.config.radius
                    forces.append(PedRobotForce(self.config.prf_config, sim.peds, lambda: robot.pos))
            return forces

        self.pysf_sim = PySFSimulator(
            self.pysf_state.pysf_states(), self.groups.groups_as_lists,
            self.map_def.obstacles_pysf, config=pysf_config, make_forces=make_forces)
        self.pysf_sim.peds.step_width = self.config.time_per_step_in_secs
        self.pysf_sim.peds.max_speed_multiplier = self.config.peds_speed_mult
        self.robot_navs = [RouteNavigator(proximity_threshold=self.goal_proximity_threshold) for _ in self.robots]

        self.reset_state()
        for behavior in self.peds_behaviors:
            behavior.reset()

    @property
    def goal_pos(self) -> List[Vec2D]:
        return [n.current_waypoint for n in self.robot_navs]

    @property
    def next_goal_pos(self) -> List[Union[Vec2D, None]]:
        return [n.next_waypoint for n in self.robot_navs]

    @property
    def robot_poses(self) -> List[RobotPose]:
        return [r.pose for r in self.robots]

    @property
    def robot_pos(self) -> List[Vec2D]:
        return [r.pose[0] for r in self.robots]

    @property
    def ped_pos(self):
        return self.pysf_state.ped_positions

    def reset_state(self):
        for i, (robot, nav) in enumerate(zip(self.robots, self.robot_navs)):
            collision = not nav.reached_waypoint
            is_at_final_goal = nav.reached_destination
            if collision or is_at_final_goal:
                waypoints = sample_route(self.map_def, i)
                nav.new_route(waypoints[1:])
                new_orient = atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
                robot.reset_state((waypoints[0], new_orient))

    def step_once(self, actions: List[RobotAction]):
        for behavior in self.peds_behaviors:
            behavior.step()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups.groups_as_lists
        self.pysf_sim.peds.step(ped_forces, groups)
        for robot, nav, action in zip(self.robots, self.robot_navs, actions):
            robot.apply_action(action, self.config.time_per_step_in_secs)
            nav.update_position(robot.pos)
