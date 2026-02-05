"""SVG-derived geometry types used for navigation maps.

These dataclasses normalize SVG elements (rectangles, circles, paths) into simple
Python structures consumed by map parsing and obstacle construction logic.
"""

from dataclasses import dataclass

from robot_sf.common.types import Vec2D, Zone


@dataclass
class SvgRectangle:
    """Axis-aligned rectangle extracted from an SVG map.

    Attributes:
        x: X coordinate of the top-left corner.
        y: Y coordinate of the top-left corner.
        width: Rectangle width in world units.
        height: Rectangle height in world units.
        label: Human-readable label from the SVG metadata.
        id_: Raw SVG element identifier.
    """

    x: float
    y: float
    width: float
    height: float
    label: str
    id_: str

    def __post_init__(self):
        """Validate rectangle dimensions are non-negative.

        Raises:
            ValueError: If ``width`` or ``height`` is negative.
        """
        if self.width < 0:
            raise ValueError("Width needs to be a float >= 0!")
        if self.height < 0:
            raise ValueError("Height needs to be a float >= 0!")

    def get_zone(self) -> Zone:
        """Convert the rectangle into a zone tuple.

        Returns:
            Zone: Three corner points defining the rectangular zone boundaries.
                The order is top-left, top-right, bottom-right (axis-aligned).
        """
        return (
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
        )


@dataclass
class SvgCircle:
    """Circle extracted from an SVG map (e.g., pedestrian markers).

    Attributes:
        cx: X coordinate of the circle center.
        cy: Y coordinate of the circle center.
        r: Radius of the circle.
        label: Human-readable label from the SVG metadata.
        id_: Raw SVG element identifier.
        cls: Raw SVG class attribute, used for POI detection.
    """

    cx: float
    cy: float
    r: float
    label: str
    id_: str
    cls: str = ""

    def __post_init__(self):
        """Validate radius is positive.

        Raises:
            ValueError: If ``r`` is less than or equal to zero.
        """
        if self.r <= 0:
            raise ValueError("Radius needs to be a float > 0!")

    def get_center(self) -> Vec2D:
        """Return the circle center coordinates.

        Returns:
            Vec2D: Tuple ``(cx, cy)`` representing the center point.
        """
        return (self.cx, self.cy)


@dataclass
class SvgPath:
    """Polyline path extracted from an SVG file.

    Attributes:
        coordinates: Ordered 2D waypoints composing the path.
        label: Human-readable label from the SVG metadata.
        id: Raw SVG element identifier.
    """

    coordinates: tuple[Vec2D, ...]
    label: str
    id: str
