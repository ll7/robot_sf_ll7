from math import sin, cos
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from robot_sf.utils.utilities import norm_angles


@dataclass
class Vec2D:
    x: float
    y: float

    @property
    def as_list(self) -> List[float]:
        return [self.x, self.y]


@dataclass
class PolarVec2D:
    """Representing directed movement as 2D polar coords"""
    dist: float
    orient: float

    @property
    def vector(self) -> Vec2D:
        return Vec2D(self.dist * cos(self.orient), self.dist * sin(self.orient))


@dataclass
class RobotPose:
    pos: Vec2D
    orient: float

    @property
    def coords(self) -> List[float]:
        return [self.pos.x, self.pos.y]

    @property
    def coords_with_orient(self) -> List[float]:
        return [self.pos.x, self.pos.y, self.orient]

    def rel_pos(self, target_coords: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        dists: np.ndarray = np.linalg.norm(target_coords - np.array(self.coords))
        x_offsets = target_coords[0] - self.pos.x
        y_offsets = target_coords[1] - self.pos.y
        angles = np.arctan2(y_offsets, x_offsets) - self.orient
        angles = norm_angles(angles)
        return dists, angles
