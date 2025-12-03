"""TODO docstring. Document this module."""

from dataclasses import dataclass

from robot_sf.common.types import Vec2D, Zone


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

        Returns:
            Zone: A tuple of 3 corner points defining the rectangular zone boundaries.
        """
        # TODO: Is this a correct zone definition?
        return (
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
        )


@dataclass
class SvgCircle:
    """Represents a circle in an SVG file (for single pedestrian markers)."""

    cx: float
    cy: float
    r: float
    label: str
    id_: str

    def __post_init__(self):
        """
        Validates the radius.
        Raises a ValueError if radius is less than or equal to 0.
        """
        if self.r <= 0:
            raise ValueError("Radius needs to be a float > 0!")

    def get_center(self) -> Vec2D:
        """Returns the center point of the circle.

        Returns:
            Vec2D: A tuple (cx, cy) representing the center coordinates.
        """
        return (self.cx, self.cy)


@dataclass
class SvgPath:
    """Represents a path in an SVG file (sequence of 2D waypoints)."""

    coordinates: tuple[Vec2D, ...]
    label: str
    id: str
