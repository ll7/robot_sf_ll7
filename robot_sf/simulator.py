import random
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Protocol, Union

import pysocialforce as pysf
from pysocialforce.utils import SimulatorConfig as PySFSimConfig

from robot_sf.map_config import MapDefinition
from robot_sf.ped_spawn_generator \
    import ZonePointsGenerator, PedSpawnConfig, initialize_pedestrians
from robot_sf.ped_robot_force \
    import PedRobotForce, PedRobotForceConfig
from robot_sf.pedestrian_grouping \
    import GroupRedirectBehavior, PySFPedestrianStates, PedestrianGroupings


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


@dataclass
class MovingRobot(Protocol):
    @property
    def pos(self) -> Vec2D:
        raise NotImplementedError()

    @property
    def pose(self) -> RobotPose:
        raise NotImplementedError()

    @property
    def current_speed(self) -> PolarVec2D:
        raise NotImplementedError()

    def apply_action(self, action: PolarVec2D, d_t: float):
        raise NotImplementedError()


class SimulationSettings(Protocol):
    @property
    def peds_per_area_m2(self) -> float:
        raise NotImplementedError()

    @property
    def max_peds_per_group(self) -> int:
        raise NotImplementedError()

    @property
    def step_time_in_secs(self) -> float:
        raise NotImplementedError()

    @property
    def peds_speed_mult(self) -> float:
        raise NotImplementedError()

    @property
    def prf_config(self) -> PedRobotForceConfig:
        raise NotImplementedError()


@dataclass
class Simulator:
    config: SimulationSettings
    map_def: MapDefinition
    robot_factory: Callable[[RobotPose], MovingRobot]
    is_robot_at_goal: Callable[[], bool]
    robot: MovingRobot = field(init=False)
    waypoints: List[Vec2D] = field(init=False, default_factory=list)
    waypoint_id: int = field(init=False, default=-1)

    def __post_init__(self):
        ped_spawn_gens = [ZonePointsGenerator([z]) for z in self.map_def.ped_spawn_zones]
        self.robot_spawn_gens = [ZonePointsGenerator([z]) for z in self.map_def.robot_spawn_zones]
        self.robot_goal_gens = [ZonePointsGenerator([z]) for z in self.map_def.goal_zones]

        spawn_config = PedSpawnConfig(self.config.peds_per_area_m2, self.config.max_peds_per_group)
        ped_states_np, initial_groups, zone_assignments = \
            initialize_pedestrians(spawn_config, self.map_def.ped_spawn_zones)

        get_state = lambda: self.pysf_sim.peds.state
        self.pysf_state = PySFPedestrianStates(get_state)
        groups = PedestrianGroupings(self.pysf_state)
        self.peds_behavior = GroupRedirectBehavior(groups, zone_assignments, ped_spawn_gens)
        self.groups_as_list = lambda: [list(ped_ids) for ped_ids in groups.groups.values()]

        for ped_ids in initial_groups:
            groups.new_group(ped_ids)

        def make_forces(sim: pysf.Simulator, config: PySFSimConfig) -> List[pysf.forces.Force]:
            forces = pysf.simulator.make_forces(sim, config)
            forces = [f for f in forces if type(f) != pysf.forces.ObstacleForce]
            if self.config.prf_config.is_active:
                forces.append(PedRobotForce(
                    self.config.prf_config, sim.peds, lambda: self.robot.pos))
            return forces

        self.pysf_sim = pysf.Simulator(
            ped_states_np, self.groups_as_list(),
            self.map_def.obstacles_pysf, make_forces=make_forces)
        self.pysf_sim.peds.step_width = self.config.step_time_in_secs
        self.pysf_sim.peds.max_speed_multiplier = self.config.peds_speed_mult
        self.reset_state()

    @property
    def goal_pos(self) -> Vec2D:
        return self.waypoints[self.waypoint_id]

    @property
    def next_goal_pos(self) -> Union[Vec2D, None]:
        return self.waypoints[self.waypoint_id + 1] \
            if self.waypoint_id + 1 < len(self.waypoints) else None

    @property
    def robot_pose(self) -> RobotPose:
        return self.robot.pose

    @property
    def ped_positions(self):
        return self.pysf_state.ped_positions

    def reset_state(self):
        self.peds_behavior.pick_new_goals()

        def generate_point(zone: ZonePointsGenerator) -> Vec2D:
            points, zone_id = zone.generate(num_samples=1)
            return points[0]

        def generate_route() -> List[Vec2D]:
            spawn_id = random.randint(0, len(self.robot_spawn_gens) - 1)
            goal_id = random.randint(0, len(self.robot_goal_gens) - 1)
            route = self.map_def.find_route(spawn_id, goal_id)
            initial_spawn = generate_point(self.robot_spawn_gens[route.spawn_id])
            final_goal = generate_point(self.robot_goal_gens[route.goal_id])
            route = [initial_spawn] + route.waypoints + [final_goal]
            # TODO: think of adding a bit of noise to the exact waypoint positions as well
            return route

        collision = not self.is_robot_at_goal
        last_waypoint_reached = self.waypoint_id == len(self.waypoints) - 1
        if collision or last_waypoint_reached:
            self.waypoints = generate_route()
            self.waypoint_id = 1
            self.robot = self.robot_factory((self.waypoints[0], 0))
        else:
            self.waypoint_id += 1

    def step_once(self, action: PolarVec2D):
        self.peds_behavior.redirect_groups_if_at_goal()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups_as_list()
        self.pysf_sim.peds.step(ped_forces, groups)
        self.robot.apply_action(action, self.config.step_time_in_secs)

    def get_pedestrians_groups(self):
        _, groups = self.pysf_sim.current_state
        return groups
