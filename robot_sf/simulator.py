from dataclasses import dataclass, field
from typing import List, Tuple, Union, Callable

import numpy as np
from pysocialforce import Simulator as PySFSimulator
from pysocialforce.simulator import make_forces as pysf_make_forces
from pysocialforce.utils import SimulatorConfig as PySFSimConfig
from pysocialforce.forces import Force as PySFForce, ObstacleForce

from robot_sf.sim_config import SimulationSettings
from robot_sf.map_config import MapDefinition
from robot_sf.ped_spawn_generator \
    import PedSpawnConfig, populate_crowded_zones, populate_ped_routes
from robot_sf.ped_robot_force import PedRobotForce
from robot_sf.ped_grouping import PedestrianStates, PedestrianGroupings
from robot_sf.ped_behavior import PedestrianBehavior, CrowdedZoneBehavior, FollowRouteBehavior
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveAction
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleAction
from robot_sf.navigation import RouteNavigator, sample_route


Vec2D = Tuple[float, float]
RobotAction = Union[DifferentialDriveAction, BicycleAction]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]
Robot = Union[DifferentialDriveRobot, BicycleDriveRobot]


def populate_simulation(
        pysf_config: PySFSimConfig, spawn_config: PedSpawnConfig, map_def: MapDefinition
    ) -> Tuple[PedestrianStates, PedestrianGroupings, List[PedestrianBehavior]]:

    crowd_ped_states_np, crowd_groups, zone_assignments = \
        populate_crowded_zones(spawn_config, map_def.ped_spawn_zones)
    route_ped_states_np, route_groups, route_assignments, initial_sections = \
        populate_ped_routes(spawn_config, map_def.ped_routes)

    tau = pysf_config.scene_config.tau
    combined_ped_states_np = np.concatenate((crowd_ped_states_np, route_ped_states_np))
    taus = np.full((combined_ped_states_np.shape[0]), tau)
    ped_states = np.concatenate((combined_ped_states_np, np.expand_dims(taus, -1)), axis=-1)
    id_offset = ped_states.shape[0]
    combined_groups = crowd_groups + [{id + id_offset for id in peds} for peds in route_groups]

    pysf_state = PedestrianStates(lambda: ped_states)
    crowd_pysf_state = PedestrianStates(lambda: ped_states[:id_offset])
    route_pysf_state = PedestrianStates(lambda: ped_states[id_offset:])

    groups = PedestrianGroupings(pysf_state)
    for ped_ids in combined_groups:
        groups.new_group(ped_ids)
    crowd_groupings = PedestrianGroupings(crowd_pysf_state)
    for ped_ids in crowd_groups:
        crowd_groupings.new_group(ped_ids)
    route_groupings = PedestrianGroupings(route_pysf_state)
    for ped_ids in route_groups:
        route_groupings.new_group(ped_ids)

    crowd_behavior = CrowdedZoneBehavior(
        crowd_groupings, zone_assignments, map_def.ped_spawn_zones)
    route_behavior = FollowRouteBehavior(route_groupings, route_assignments, initial_sections)
    ped_behaviors: List[PedestrianBehavior] = [crowd_behavior] # [crowd_behavior, route_behavior]
    # TODO: enable "follow route behavior" once it's ready
    return pysf_state, groups, ped_behaviors


@dataclass
class Simulator:
    config: SimulationSettings
    map_def: MapDefinition
    robot: Robot
    goal_proximity_threshold: float
    robot_nav: RouteNavigator = field(init=False)
    pysf_sim: PySFSimulator = field(init=False)
    sample_route: Callable[[], List[Vec2D]] = field(init=False)
    pysf_state: PedestrianStates = field(init=False)
    groups: PedestrianGroupings = field(init=False)
    peds_behaviors: List[PedestrianBehavior] = field(init=False)

    def __post_init__(self):
        pysf_config = PySFSimConfig()
        spawn_config = PedSpawnConfig(self.config.peds_per_area_m2, self.config.max_peds_per_group)
        self.pysf_state, self.groups, self.peds_behaviors = \
              populate_simulation(pysf_config, spawn_config, self.map_def)

        def make_forces(sim: PySFSimulator, config: PySFSimConfig) -> List[PySFForce]:
            forces = pysf_make_forces(sim, config)
            # TODO: enable obstacle force in case pedestrian routes require it
            forces = [f for f in forces if type(f) != ObstacleForce]
            if self.config.prf_config.is_active:
                forces.append(PedRobotForce(self.config.prf_config, sim.peds, lambda: self.robot.pos))
            return forces

        self.pysf_sim = PySFSimulator(
            self.pysf_state.pysf_states(), self.groups.groups_as_lists,
            self.map_def.obstacles_pysf, config=pysf_config, make_forces=make_forces)
        self.pysf_sim.peds.step_width = self.config.time_per_step_in_secs
        self.pysf_sim.peds.max_speed_multiplier = self.config.peds_speed_mult
        self.robot_nav = RouteNavigator(proximity_threshold=self.goal_proximity_threshold)

        self.reset_state()
        for behavior in self.peds_behaviors:
            behavior.reset()

    @property
    def goal_pos(self) -> Vec2D:
        return self.robot_nav.current_waypoint

    @property
    def next_goal_pos(self) -> Union[Vec2D, None]:
        return self.robot_nav.next_waypoint

    @property
    def robot_pose(self) -> RobotPose:
        return self.robot.pose

    @property
    def ped_positions(self):
        return self.pysf_state.ped_positions

    def reset_state(self):
        collision = not self.robot_nav.reached_waypoint
        is_at_final_goal = self.robot_nav.reached_destination
        if collision or is_at_final_goal:
            waypoints = sample_route(self.map_def)
            self.robot_nav.new_route(waypoints[1:])
            self.robot.reset_state((waypoints[0], 0))

    def step_once(self, action: RobotAction):
        for behavior in self.peds_behaviors:
            behavior.step()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups.groups_as_lists
        self.pysf_sim.peds.step(ped_forces, groups)
        self.robot.apply_action(action, self.config.time_per_step_in_secs)
        self.robot_nav.update_position(self.robot.pos)
