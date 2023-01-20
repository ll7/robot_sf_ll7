from typing import Tuple, Callable
from dataclasses import dataclass

import numpy as np
from pysocialforce.scene import PedState
from pysocialforce.utils import stateutils

Vec2D = Tuple[float, float]


@dataclass
class PedRobotForceConfig:
    is_active: bool = False
    robot_radius: float = 1.0
    activation_threshold: float = 1.0
    force_multiplier: float = 10.0
    sigma: float=0.2


class PedRobotForce:
    """This force represents a behavior that's making the pedestrians
    aware of the robot such that they try to avoid it.

    If the force multiplier is parameterized with a negative factor,
    this force can be used for adverserial trainings as well."""

    def __init__(self, config: PedRobotForceConfig, peds: PedState,
                 get_robot_pos: Callable[[], Vec2D]):
        self.config = config
        self.peds = peds
        self.get_robot_pos = get_robot_pos

    def __call__(self) -> np.ndarray:
        threshold = self.config.activation_threshold + self.peds.agent_radius
        forces = np.zeros((self.peds.size(), 2))
        ped_positions = self.peds.pos()
        robot_pos = self.get_robot_pos()

        diff = (ped_positions - robot_pos).astype(np.float64)
        directions, dists = stateutils.normalize(diff)

        dists = dists - self.peds.agent_radius - self.config.robot_radius
        if np.all(dists >= threshold):
            return forces

        dist_mask = dists < threshold
        directions[dist_mask] *= np.exp(-dists[dist_mask].reshape(-1, 1) / self.config.sigma)
        forces[dist_mask] = np.sum(directions[dist_mask], axis=0)

        return forces * self.config.force_multiplier
