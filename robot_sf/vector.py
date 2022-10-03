from typing import List
from dataclasses import dataclass


@dataclass
class Vec2D:
    x: float
    y: float

    @property
    def as_list(self) -> List[float]:
        return [self.x, self.y]
