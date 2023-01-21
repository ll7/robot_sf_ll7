import random
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Protocol

import pysocialforce as pysf
from pysocialforce.utils import SimulatorConfig as PySFSimConfig

from robot_sf.sim_config import MapDefinition, GlobalRoute
from robot_sf.ped_spawn_generator \
    import SpawnGenerator, PedSpawnConfig, initialize_pedestrians
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
    def dist_to_goal(self) -> float:
        raise NotImplementedError()

    @property
    def pos(self) -> Vec2D:
        raise NotImplementedError()

    @property
    def pose(self) -> RobotPose:
        raise NotImplementedError()

    @property
    def goal(self) -> Vec2D:
        raise NotImplementedError()

    @goal.setter
    def goal(self, new_goal: Vec2D):
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
    robot_factory: Callable[[RobotPose, Vec2D], MovingRobot]
    robot: MovingRobot = field(init=False)
    spawn_id: int = field(init=False, default=0)
    goal_id: int = field(init=False, default=0)
    waypoint_id: int = field(init=False, default=-1)

    def __post_init__(self):
        ped_spawn_gens = [SpawnGenerator([z]) for z in self.map_def.ped_spawn_zones]
        self.robot_spawn_gens = [SpawnGenerator([z]) for z in self.map_def.robot_spawn_zones]
        self.robot_goal_gens = [SpawnGenerator([z]) for z in self.map_def.goal_zones]

        spawn_config = PedSpawnConfig(self.config.peds_per_area_m2, self.config.max_peds_per_group)
        ped_states_np, initial_groups, zone_assignments = \
            initialize_pedestrians(spawn_config, self.map_def.ped_spawn_zones)

        get_state = lambda: self.pysf_sim.peds.state
        pysf_state = PySFPedestrianStates(get_state)
        groups = PedestrianGroupings(pysf_state)
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
        return self.robot.goal

    @property
    def robot_pose(self) -> RobotPose:
        return self.robot.pose

    @property
    def dist_to_goal(self) -> float:
        return self.robot.dist_to_goal

    @property
    def current_positions(self):
        ped_states, _ = self.pysf_sim.current_state
        return ped_states[:, 0:2]

    def reset_state(self):
        self.peds_behavior.pick_new_goals()

        def get_route(spawn_id: int, goal_id: int) -> GlobalRoute:
            return next(filter(
                lambda r: r.goal_id == goal_id and r.spawn_id == spawn_id,
                self.map_def.robot_routes))

        reset_required = self.waypoint_id == -1 or self.dist_to_goal >= 1.0
        if reset_required:
            self.spawn_id = random.randint(0, len(self.robot_spawn_gens) - 1)
            self.goal_id = random.randint(0, len(self.robot_goal_gens) - 1)
            route = get_route(self.spawn_id, self.goal_id)
            self.waypoint_id = 0
            goal_pos = route.waypoints[self.waypoint_id]
            robot_pose = (self.robot_spawn_gens[self.spawn_id].generate(1)[0][0], 0)
            self.robot = self.robot_factory(robot_pose, goal_pos)
        else:
            route = get_route(self.spawn_id, self.goal_id)
            if self.waypoint_id == len(route.waypoints) - 1:
                self.waypoint_id = -1
                self.robot.goal = self.robot_goal_gens[self.goal_id].generate(1)[0][0]
            else:
                self.waypoint_id += 1
                self.robot.goal = route.waypoints[self.waypoint_id]

    def step_once(self, action: PolarVec2D):
        self.peds_behavior.redirect_groups_if_at_goal()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups_as_list()
        self.pysf_sim.peds.step(ped_forces, groups)
        self.robot.apply_action(action, self.config.step_time_in_secs)

    def get_pedestrians_groups(self):
        _, groups = self.pysf_sim.current_state
        return groups
