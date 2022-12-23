from typing import List, Callable, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pysocialforce as pysf

from robot_sf.extender_force import PedRobotForce
from robot_sf.simulation_config \
    import load_config, SimulationConfiguration
from robot_sf.pedestrian_grouping \
    import GroupRedirectBehavior, PySFPedestrianStates, PedestrianGroupings
from robot_sf.robot import RobotPose


Vec2D = Tuple[float, float]


@dataclass
class RobotObject:
    pose: RobotPose
    radius: float


def make_forces(sim_config_user: SimulationConfiguration, enable_groups: bool,
                get_robot_pos: Callable[[], Tuple[float, float]]) -> List[pysf.forces.Force]:
    ped_rob_force = PedRobotForce(
        get_robot_pos,
        sim_config_user.robot_force_config.robot_radius,
        sim_config_user.robot_force_config.activation_threshold,
        sim_config_user.robot_force_config.force_multiplier)

    force_list: List[pysf.forces.Force] = [
        pysf.forces.DesiredForce(),
        pysf.forces.SocialForce(),
        pysf.forces.ObstacleForce(),
    ]

    if sim_config_user.flags.activate_ped_robot_force:
        force_list.append(ped_rob_force)

    group_forces = [
        pysf.forces.GroupCoherenceForceAlt(),
        pysf.forces.GroupRepulsiveForce(),
        pysf.forces.GroupGazeForceAlt(),
    ]
    if enable_groups:
        force_list += group_forces

    return force_list


class Simulator:
    # TODO: include robot kinematics and occupancy here, make RobotEnv just the Gym wrapper

    def __init__(self, difficulty: int, peds_sparsity: int,
                 d_t: Union[float, None], peds_speed_mult: float):
        path_to_config: str = None

        config, state, _, obstacles = load_config(path_to_config, difficulty)
        self.box_size = config.box_size
        self.peds_sparsity = peds_sparsity

        def pick_goal() -> Tuple[float, float]:
            pos_x, pos_y = np.random.uniform(-self.box_size, self.box_size, size=(2))
            return (pos_x, pos_y)

        get_state = lambda: self.pysf_sim.peds.state
        self.pysf_state = PySFPedestrianStates(get_state)
        self.groups = PedestrianGroupings(self.pysf_state)
        self.peds_behavior = GroupRedirectBehavior(self.groups, pick_goal)
        self.groups_as_list = lambda: [list(ped_ids) for ped_ids in self.groups.groups.values()]

        robot_radius = config.robot_force_config.robot_radius
        self.robot = RobotObject(RobotPose((1e5, 1e5), 0), robot_radius)

        get_robot_pos = lambda: self.robot.pose.pos
        sim_forces = self.forces = make_forces(config, True, get_robot_pos)
        self.pysf_sim = pysf.Simulator(sim_forces, state, self.groups_as_list(), obstacles)
        self.pysf_sim.peds.step_width = d_t if d_t else self.pysf_sim.peds.step_width
        self.pysf_sim.peds.max_speed_multiplier = peds_speed_mult

        if self.pysf_state.num_peds:
            self.groups.cluster_groups(self.pysf_state.num_peds // 5)
        self.reset_state()

    @property
    def d_t(self) -> float:
        return self.pysf_sim.peds.step_width

    def reset_state(self):
        self.peds_behavior.pick_new_goals()

    def step_once(self):
        self.peds_behavior.redirect_groups_if_at_goal()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups_as_list()
        self.pysf_sim.peds.step(ped_forces, groups)

    @property
    def current_positions(self):
        ped_states, _ = self.pysf_sim.current_state
        return ped_states[:, 0:2]

    def get_pedestrians_groups(self):
        _, groups = self.pysf_sim.current_state
        return groups

    def move_robot(self, pose: RobotPose):
        self.robot.pose = pose
