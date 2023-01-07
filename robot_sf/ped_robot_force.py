from typing import Tuple, Callable

import numpy as np
from pysocialforce.scene import PedState
from pysocialforce.utils import stateutils

from robot_sf.simulation_config import RobotForceConfig

Vec2D = Tuple[float, float]


class PedRobotForce:
    """This force represents a behavior that's making the pedestrians
    aware of the robot such that they try to avoid it.

    If the force multiplier is parameterized with a negative factor,
    this force can be used for adverserial trainings as well."""

    def __init__(self, config: RobotForceConfig, peds: PedState,
                 get_robot_pos: Callable[[], Vec2D]):
        self.config = config
        self.peds = peds
        self.get_robot_pos = get_robot_pos

    def __call__(self) -> np.ndarray:
        threshold = self.config.activation_threshold + self.peds.agent_radius
        force = np.zeros((self.peds.size(), 2))
        ped_positions = self.peds.pos()
        robot_pos = self.get_robot_pos()

        for i, pos in enumerate(ped_positions):
            diff = pos - robot_pos
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.peds.agent_radius -self.config.robot_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / self.config.sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)
        return force * self.config.force_multiplier
