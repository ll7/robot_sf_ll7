from math import dist, atan2
from typing import Tuple
from dataclasses import dataclass

import numpy as np


Vec2D = Tuple[float, float]


@dataclass
class PolarVec2D:
    """Representing directed movement as 2D polar coords"""
    dist: float
    orient: float


@dataclass
class RobotPose:
    pos: Vec2D
    orient: float

    def rel_pos(self, target_coords: Vec2D) -> Tuple[float, float]:
        t_x, t_y = target_coords
        r_x, r_y = self.pos
        distance = dist(target_coords, self.pos)

        angle = atan2(t_y - r_y, t_x - r_x) - self.orient
        angle = (angle + np.pi) % (2 * np.pi) -np.pi
        return distance, angle
