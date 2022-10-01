# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:30:55 2020

@author: enric
"""

import sys 
import os

csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)

from ..utils.utilities import change_direction

from pysocialforce import forces
from pysocialforce.utils import stateutils

import numpy as np


def normalize(vecs):
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = np.linalg.norm(vecs, axis=1)
    normalized = vecs / (norm_factors[:, np.newaxis] + 1e-8)
    return normalized, norm_factors


class PedRobotForce(forces.Force):
    def __init__(self, robot_radius=1, activation_treshold=0.5, force_multiplier=1):
        self.robot_radius = robot_radius
        self.activation_treshold = activation_treshold
        super().__init__()
        self.robot_state = np.array([[1e5, 1e5]], dtype=float)
        self.force_multiplier = force_multiplier

    def updateRobotState(self, pos):
        self.robot_state = pos

    def _get_force(self) -> np.ndarray:
        sigma = self.config("sigma", 0.2)
        threshold = self.activation_treshold + self.peds.agent_radius
        force = np.zeros((self.peds.size(), 2))
        pos = self.peds.pos()

        for i, p in enumerate(pos):
            diff = p - self.robot_state
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.peds.agent_radius -self.robot_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)
        return force * self.force_multiplier


class DesiredForce(forces.Force):
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """

    def __init__(self, 
                 obstacle_avoidance= False, 
                 angles =np.pi*np.array([-1, -0.5, -0.25, 0.25, 0.5, 1]), 
                 p0 = np.empty((0, 2)),
                 p1 = np.empty((0, 2)),
                 view_distance = 15,
                 forgetting_factor = .8):
        super().__init__()
        self.obstacle_avoidance = obstacle_avoidance
        if self.obstacle_avoidance:
            self.angles = angles
            self.p0 = p0
            self.p1 = p1
            self.view_distance = view_distance
            self.forgetting_factor = forgetting_factor

    def _get_force(self):
        relexation_time = self.config("relaxation_time", 0.5)
        goal_threshold = self.config("goal_threshold", 0.1)
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()

        direction, dist = normalize(goal - pos)
        ### in the following, direction is changed if obstacle is detected
        if self.obstacle_avoidance:
            direction,peds_collision_indices = change_direction(
                self.p0,
                self.p1,
                self.peds.state[:, :2],   # current positions
                self.peds.state[:, 4:6],  # current destinations
                self.view_distance, 
                self.angles, 
                direction,
                self.peds.desired_directions()) # current desired directions

        force = np.zeros((self.peds.size(), 2))
        force[dist > goal_threshold] = (
            direction * self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
            )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]
        force /= relexation_time

        if self.obstacle_avoidance:
            # in case of correction of direction, some "memory" has to be used
            # on the direction of the pedestrians in order to reduce "chattering"
            forces_intensities = np.linalg.norm(force, axis=-1)

            # TODO: fix division bug
            previous_directions = vel / np.tile(np.linalg.norm(vel, axis=-1), (2, 1)).T
            previous_directions = np.nan_to_num(previous_directions)
            #print(previous_directions)
            previous_forces = previous_directions * np.tile(forces_intensities, (2, 1)).T
            force[peds_collision_indices] = self.forgetting_factor * force[peds_collision_indices] \
                + (1 - self.forgetting_factor) * previous_forces[peds_collision_indices] 

        return force * self.factor


class GroupRepulsiveForce(forces.Force):
    """Group repulsive force"""

    def _get_force(self):
        threshold = self.config("threshold", 0.5)
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                size = len(group)
                member_pos = self.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = normalize(diff)
                diff[norms > threshold, :] = 0
                try:
                    forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)
                except Exception:
                    pass

        return forces * self.factor
