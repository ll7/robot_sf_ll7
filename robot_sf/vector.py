from math import sin, cos, dist, atan2
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np


Vec2D = Tuple[float, float]


def norm_angle(angle: float) -> float:
    """Normalize angles between [-pi, pi)"""
    return (angle + np.pi) % (2 * np.pi) -np.pi


@dataclass
class PolarVec2D:
    """Representing directed movement as 2D polar coords"""
    dist: float
    orient: float

    @property
    def as_unit_vec(self) -> Vec2D:
        return (self.dist * cos(self.orient), self.dist * sin(self.orient))


@dataclass
class RobotPose:
    pos: Vec2D
    orient: float

    @property
    def coords(self) -> List[float]:
        pos_x, pos_y = self.pos
        return [pos_x ,pos_y]

    @property
    def coords_with_orient(self) -> List[float]:
        pos_x, pos_y = self.pos
        return [pos_x, pos_y, self.orient]

    def rel_pos(self, target_coords: Vec2D) -> Tuple[float, float]:
        t_x, t_y = target_coords
        r_x, r_y = self.pos
        distance = dist(target_coords, self.pos)

        angle = atan2(t_y - r_y, t_x - r_x) - self.orient
        angle = norm_angle(angle)
        return distance, angle
