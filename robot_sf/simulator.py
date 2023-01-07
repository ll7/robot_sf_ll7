from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Union, Protocol

import pysocialforce as pysf
from pysocialforce.utils import SimulatorConfig as PySFSimConfig

from robot_sf.ped_spawn_generator \
    import SpawnGenerator, PedSpawnConfig, initialize_pedestrians
from robot_sf.ped_robot_force import PedRobotForce
from robot_sf.simulation_config import RobotForceConfig
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

    @property
    def current_speed(self) -> PolarVec2D:
        raise NotImplementedError()

    def apply_action(self, action: PolarVec2D, d_t: float) -> Tuple[PolarVec2D, bool]:
        raise NotImplementedError()


@dataclass
class Simulator:
    box_size: float
    config: RobotForceConfig
    obstacles: List[Tuple[float, float, float, float]]
    robot_factory: Callable[[RobotPose, Vec2D], MovingRobot]
    peds_speed_mult: float = 1.3
    custom_d_t: Union[float, None] = field(default=1)
    robot: MovingRobot = field(init=False)

    def __post_init__(self):
        box_rect = (
            (-self.box_size, self.box_size),
            (-self.box_size, -self.box_size),
            (self.box_size, -self.box_size))
        spawn_gen = SpawnGenerator([box_rect])
        spawn_config = PedSpawnConfig(20, 6)
        ped_states_np, initial_groups = initialize_pedestrians(spawn_config, spawn_gen, spawn_gen)
        pick_ped_goal = lambda: spawn_gen.generate(1)[0]
        self.pick_robot_spawn = lambda: spawn_gen.generate(1)[0]
        self.pick_robot_goal = lambda: spawn_gen.generate(1)[0]

        get_state = lambda: self.pysf_sim.peds.state
        pysf_state = PySFPedestrianStates(get_state)
        groups = PedestrianGroupings(pysf_state)
        self.peds_behavior = GroupRedirectBehavior(groups, pick_ped_goal)
        self.groups_as_list = lambda: [list(ped_ids) for ped_ids in groups.groups.values()]

        for ped_ids in initial_groups:
            groups.new_group(ped_ids)

        def make_forces(sim: pysf.Simulator, config: PySFSimConfig) -> List[pysf.forces.Force]:
            forces = pysf.simulator.make_forces(sim, config)
            if self.config.is_active:
                forces.append(PedRobotForce(self.config, sim.peds, lambda: self.robot.pos))
            return forces

        self.pysf_sim = pysf.Simulator(
            ped_states_np, self.groups_as_list(),
            self.obstacles, make_forces=make_forces)
        self.pysf_sim.peds.step_width = self.custom_d_t \
            if self.custom_d_t else self.pysf_sim.peds.step_width
        self.pysf_sim.peds.max_speed_multiplier = self.peds_speed_mult
        self.reset_state()

    @property
    def d_t(self) -> float:
        return self.pysf_sim.peds.step_width

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
        robot_pose = (self.pick_robot_spawn(), 0)
        goal_pos = self.pick_robot_goal()
        self.robot = self.robot_factory(robot_pose, goal_pos)

    def step_once(self, action: PolarVec2D) -> Tuple[PolarVec2D, float, float, bool]:
        self.peds_behavior.redirect_groups_if_at_goal()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups_as_list()
        self.pysf_sim.peds.step(ped_forces, groups)

        dist_before = self.robot.dist_to_goal
        movement, is_overdrive = self.robot.apply_action(action, self.d_t)
        dist_after = self.robot.dist_to_goal
        return movement, dist_before, dist_after, is_overdrive

    def get_pedestrians_groups(self):
        _, groups = self.pysf_sim.current_state
        return groups
