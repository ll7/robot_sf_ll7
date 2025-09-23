from dataclasses import dataclass
from typing import Tuple

from robot_sf.util.types import Vec2D, Zone


@dataclass
class SvgRectangle:
    """
    A class to represent a rectangle in an SVG file.
    """

    x: float
    y: float
    width: float
    height: float
    label: str
    id_: str

    def __post_init__(self):
        """
        Validates the width and height.
        Raises a ValueError if width or height is less than 0.
        """
        if self.width < 0:
            raise ValueError("Width needs to be a float >= 0!")
        if self.height < 0:
            raise ValueError("Height needs to be a float >= 0!")

    def get_zone(self) -> Zone:
        """
        Returns the zone of the rectangle.
        """
        # TODO: Is this a correct zone definition?
        return (
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
        )


@dataclass
class SvgPath:
    """Represents a path in an SVG file (sequence of 2D waypoints)."""

    coordinates: Tuple[Vec2D, ...]
    label: str
    id: str
