from typing import List, Callable, Tuple
from dataclasses import dataclass

import numpy as np

from pysocialforce import Simulator, forces
from pysocialforce.scene import PedState

from robot_sf.extenders_py_sf.extender_force \
    import DesiredForce, GroupRepulsiveForce, PedRobotForce
from robot_sf.extenders_py_sf.simulation_config \
    import load_config, SimulationConfiguration
from robot_sf.extenders_py_sf.pedestrian_grouping \
    import GroupRedirectBehavior, PySFPedestrianStates, PedestrianGroupings

from robot_sf.robot import RobotPose
from robot_sf.vector import Vec2D


@dataclass
class RobotObject:
    pose: RobotPose
    radius: float


def make_forces(sim_config_user: SimulationConfiguration, enable_groups: bool,
                get_robot_pos: Callable[[], Tuple[float, float]]) -> List[forces.Force]:
    """Construct forces"""
    if sim_config_user.obstacle_avoidance_params is None:
        force_des = forces.DesiredForce()
    else:
        force_des = DesiredForce(
            obstacle_avoidance= True,
            angles = sim_config_user.obstacle_avoidance_params[0],
            p_0 = sim_config_user.obstacle_avoidance_params[1],
            p_1 = sim_config_user.obstacle_avoidance_params[2],
            view_distance = sim_config_user.obstacle_avoidance_params[3],
            forgetting_factor = sim_config_user.obstacle_avoidance_params[4])

    ped_rob_force = PedRobotForce(
        get_robot_pos,
        sim_config_user.robot_force_config.robot_radius,
        sim_config_user.robot_force_config.activation_threshold,
        sim_config_user.robot_force_config.force_multiplier)

    force_list: List[forces.Force] = [
        force_des,
        forces.SocialForce(),
        forces.ObstacleForce(),
    ]

    if sim_config_user.flags.activate_ped_robot_force:
        force_list.append(ped_rob_force)

    group_forces = [
        forces.GroupCoherenceForceAlt(),
        GroupRepulsiveForce(),
        forces.GroupGazeForceAlt(),
    ]
    if enable_groups:
        force_list += group_forces

    return force_list


class ExtdSimulator:
    def __init__(self, difficulty: int=0, peds_sparsity: int=0):
        path_to_config: str = None

        config, state, _, obstacles = load_config(path_to_config, difficulty)
        self.box_size = config.box_size
        self.peds_sparsity = peds_sparsity

        def pick_goal() -> Tuple[float, float]:
            x, y = np.random.uniform(-self.box_size, self.box_size, size=(2))
            return (x, y)

        get_state = lambda: self.pysf_sim.peds.state
        self.pysf_state = PySFPedestrianStates(get_state)
        self.groups = PedestrianGroupings(self.pysf_state)
        self.peds_behavior = GroupRedirectBehavior(self.groups, pick_goal)
        self.groups_as_list = lambda: self.groups.groups.values()

        robot_radius = config.robot_force_config.robot_radius
        self.robot = RobotObject(RobotPose(Vec2D(1e5, 1e5), 0), robot_radius)

        get_robot_pos = lambda: self.robot.pose.pos.as_tuple()
        forces = self.forces = make_forces(config, True, get_robot_pos)
        self.pysf_sim = Simulator(forces, state, self.groups_as_list(), obstacles)
        self.reset_state()

    @property
    def peds(self) -> PedState:
        return self.pysf_sim.peds

    def reset_state(self):
        self.peds_behavior.pick_new_goals()

    def step_once(self):
        self.peds_behavior.redirect_groups_if_at_goal()
        ped_forces = self.pysf_sim.compute_forces()
        groups = self.groups_as_list()
        self.peds.step(ped_forces, groups)

    @property
    def current_positions(self):
        ped_states, _ = self.pysf_sim.current_state
        return ped_states[:, 0:2]

    def get_pedestrians_groups(self):
        _, groups = self.pysf_sim.current_state
        return groups

    def move_robot(self, coordinates: List[float]):
        pos_x, pos_y, orient = coordinates
        self.robot.pose = RobotPose(Vec2D(pos_x, pos_y), orient)
