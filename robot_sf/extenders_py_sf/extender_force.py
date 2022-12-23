from typing import Callable, Tuple

import numpy as np

from pysocialforce.forces import Force
from pysocialforce.utils import stateutils


Vec2D = Tuple[float, float]


class PedRobotForce(Force):
    """This force represents a behavior that's making the pedestrians
    aware of the robot such that they try to avoid it.

    If the force multiplier is parameterized with a negative factor,
    this force can be used for adverserial trainings as well."""

    def __init__(self, get_robot_pos: Callable[[], Vec2D], robot_radius: float=1,
                 activation_treshold: float=0.5, force_multiplier: float=1):
        self.get_robot_pos = get_robot_pos
        self.robot_radius = robot_radius
        self.activation_treshold = activation_treshold
        super().__init__()
        self.force_multiplier = force_multiplier

    def _get_force(self) -> np.ndarray:
        sigma = self.config("sigma", 0.2)
        threshold = self.activation_treshold + self.peds.agent_radius
        force = np.zeros((self.peds.size(), 2))
        ped_positions = self.peds.pos()
        robot_pos = self.get_robot_pos()

        for i, pos in enumerate(ped_positions):
            diff = pos - robot_pos
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.peds.agent_radius -self.robot_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)
        return force * self.force_multiplier
