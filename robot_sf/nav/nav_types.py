from typing import Tuple
from dataclasses import dataclass

Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]
# TODO: Is there a difference between a Rect and a Zone?
# rect ABC with sides |A B|, |B C| and diagonal |A C|
Zone = Tuple[Vec2D, Vec2D, Vec2D]


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
            raise ValueError('Width needs to be a float >= 0!')
        if self.height < 0:
            raise ValueError('Height needs to be a float >= 0!')

    def get_zone(self) -> Zone:
        """
        Returns the zone of the rectangle.
        """
        # TODO: Is this a correct zone definition?
        return (
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height)
            )


@dataclass
class SvgPath:
    """
    A class to represent a path in an SVG file.
    """
    coordinates: Tuple[Vec2D]
    label: str
    id: str
