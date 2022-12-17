import copy
import random
from typing import List
from dataclasses import dataclass

import numpy as np

from pysocialforce import Simulator, forces
from pysocialforce.utils import stateutils

from robot_sf.utils.utilities import fill_state, fun_reduce_index
from robot_sf.extenders_py_sf.extender_scene import RSFPedState
from robot_sf.extenders_py_sf.extender_force import DesiredForce, GroupRepulsiveForce, PedRobotForce
from robot_sf.extenders_py_sf.simulation_config import load_config

from robot_sf.robot import RobotPose
from robot_sf.vector import Vec2D


@dataclass
class RobotObject:
    pose: RobotPose
    radius: float


# info: this implementation is not ideal, but at least it's not the performance bottleneck
class ExtdSimulator(Simulator):
    # TODO: don't use inheritance, it makes the code very complex

    def __init__(self, config_file=None, path_to_config: str=None, difficulty: int=0, peds_sparsity: int=0):
        self.robot = RobotObject(RobotPose(Vec2D(0, 0), 0), 0)

        self.sim_config_user, state, groups, obstacles = load_config(path_to_config, difficulty)
        self.box_size = self.sim_config_user.box_size

        self.groups_vect_idxs = []

        # filter duplicate obstacle line segments
        obs_filtered = [l for l in obstacles if not (l[0] == l[1] and l[2] == l[3])]

        # TODO: why is the robot's position initialized in such a ridiculous way?
        self.move_robot([1e5, 1e5, 0])

        # TODO: get rid of inheritance (if possible)
        super().__init__(state, groups, obs_filtered, config_file)
        self.peds_sparsity = peds_sparsity
        self.state_factory = lambda: RSFPedState(state, groups, self.config)
        self.peds: RSFPedState = None
        self.reset_state()

    def reset_state(self):
        self.peds = self.state_factory()
        self.forces = self.make_forces(self.config)

        # new state with current positions of "active" pedestrians are stored
        self._active_peds_update()
        self.set_ped_sparsity(self.peds_sparsity)

    # make_forces overrides existing function: accounts for obstacle avoidance
    def make_forces(self, force_configs):
        """Construct forces"""
        if self.sim_config_user.obstacle_avoidance_params is None:
            force_des = forces.DesiredForce()
        else:
            force_des = DesiredForce(
                obstacle_avoidance= True,
                angles = self.sim_config_user.obstacle_avoidance_params[0],
                p_0 = self.sim_config_user.obstacle_avoidance_params[1],
                p_1 = self.sim_config_user.obstacle_avoidance_params[2],
                view_distance = self.sim_config_user.obstacle_avoidance_params[3],
                forgetting_factor = self.sim_config_user.obstacle_avoidance_params[4])

        ped_rob_force = PedRobotForce(
            lambda: self.robot.pose.pos.as_tuple(),
            self.sim_config_user.robot_force_config.robot_radius,
            self.sim_config_user.robot_force_config.activation_threshold,
            self.sim_config_user.robot_force_config.force_multiplier)

        force_list: List[forces.Force] = [
            force_des,
            forces.SocialForce(),
            forces.ObstacleForce(),
        ]

        if self.sim_config_user.flags.activate_ped_robot_force:
            force_list.append(ped_rob_force)

        group_forces = [
            forces.GroupCoherenceForceAlt(),
            GroupRepulsiveForce(),
            forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)

        return force_list

    # overwrite function to account for pedestrians update
    def step_once(self):
        self._update_peds_on_scene()
        forces = self.compute_forces() if self.current_positions.shape[0] > 0 else 0
        self.peds.step(forces)

    def get_pedestrians_positions(self):
        last_peds_states = self.get_states()[0][-1, :, :]
        return last_peds_states[:, 0:2]

    def get_pedestrians_groups(self):
        return self.get_states()[1][-1]

    # update positions of currently active pedestrians
    def _active_peds_update(self):
        self.current_positions = self.get_pedestrians_positions()
        self.current_groups = self.get_pedestrians_groups()

        # define vector with all groups belongings (for plotting reasons)
        groups_vect = np.zeros(self.current_positions.shape[0], dtype = int)
        for j in range(len(groups_vect)):
            for num,group in enumerate(self.current_groups, start=1):
                if j in group:
                    groups_vect[j] = num
                    break
        self.groups_vect_idxs = groups_vect

    def move_robot(self, coordinates):
        self.robot.radius = self.sim_config_user.robot_force_config.robot_radius
        self.robot.pose = RobotPose(Vec2D(coordinates[0], coordinates[1]), coordinates[2])

    def _update_peds_on_scene(self):
        box_size = self.box_size

        # mask of pedestrians to drop because they reached their target outside of the square
        # row of booleans indicating which pedestrian is out of bound (True)
        # and which is still valid (False)
        ped_positions = self.peds.state[:, :2]
        drop_out_of_bounds: np.ndarray = np.any(
            np.absolute(ped_positions) > box_size * 1.2, axis=1) # TODO: use ped radius instead!!!
        drop_zeroes = (ped_positions == [0, 0]).all(axis=1)
        # TODO: what does position = (0, 0) mean? model this properly!

        mask = np.ones(drop_out_of_bounds.shape, dtype=bool)
        mask[drop_out_of_bounds] = False
        mask[drop_zeroes] = False

        # create new pedestrians by applying the mask
        new_state = self.peds.state[mask, :]
        new_groups = self.peds.groups

        # remove indexes of removed pedestrians from groups
        index_drop = np.where(~mask)[0]
        mask2 = np.ones(self._stopped_peds.shape, dtype=bool)
        mask2[index_drop] = False

        # clean pedestrians memory
        self._stopped_peds = self._stopped_peds[mask2]
        self._timer_stopped_peds = self._timer_stopped_peds[mask2]
        self._last_known_ped_target = self._last_known_ped_target[mask2]

        # TODO: use numpy operations instead of this loop
        for i in -np.sort(-index_drop):
            for num,group in enumerate(new_groups):
                group = np.array(group)
                if i in group: # if the index removed is in the group, remove it from there
                    new_groups[num] = group[~(group==i)].tolist()
            new_groups = fun_reduce_index(new_groups, i)

        # IMPORTANT!!before adding new pedestrians, the vector of initial speeds also has to be cleaned
        self.peds.initial_speeds = self.peds.initial_speeds[mask]
        self.peds.update(new_state, new_groups)

        #ACTIONS
        self._generation_action_selector()
        self._group_action_selector()
        self._stop_action_selector()
        self._active_peds_update()
        # re-initialize forces
        self.forces = self.make_forces(self.config)

    def set_ped_sparsity(self, new_ped_sparsity: int):
        """ This method updates the variables related to peds sparsity.
        If arg is given, overrides default value"""
        self.peds_sparsity = new_ped_sparsity
        self.av_max_people = round((2*self.box_size)**2 / self.peds_sparsity)
        self.max_population_for_new_group = int(self.av_max_people - \
            round((self.sim_config_user.new_peds_params.max_grp_size+2)/2) )
        self.max_population_for_new_individual = self.max_population_for_new_group - \
            (1 + self.sim_config_user.new_peds_params.max_single_peds)
