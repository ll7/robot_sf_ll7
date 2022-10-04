from math import sin, cos
from typing import List
from dataclasses import dataclass


@dataclass
class Vec2D:
    x: float
    y: float

    @property
    def as_list(self) -> List[float]:
        return [self.x, self.y]


@dataclass
class MovementVec2D:
    """Representing directed movement as 2D polar coords"""
    dist: float
    orient: float

    @property
    def vector(self) -> Vec2D:
        return Vec2D(self.dist * cos(self.orient), self.dist * sin(self.orient))
