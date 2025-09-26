"""
Module specifying types used in robot_sf
"""

from typing import Union

import numpy as np

# Geometry types
Vec2D = tuple[float, float]
"""Type alias for a 2D vector represented as a tuple of two floats"""

Range2D = tuple[float, float]  # (low, high)
"""Type alias for a range represented as a tuple of two floats
`(low, high)``
"""

Line2D = tuple[Vec2D, Vec2D]
"""Type alias for a 2D line represented as a tuple of two 2D vectors ``((x1, y1), (x2, y2))``."""

Point2D = tuple[float, float]

Circle2D = tuple[Vec2D, float]
"""Type alias for a 2D circle represented as a tuple of a 2D vector (center) and a float (radius)"""

MapBounds = tuple[Range2D, Range2D]  # ((min_x, max_x), (min_y, max_y))
"""
Type alias for the bounds of a map represented as a tuple of two ranges (x and y)
`((min_x, max_x), (min_y, max_y))`
"""

Rect = tuple[Vec2D, Vec2D, Vec2D]
"""
# TODO: Is there a difference between a Rect and a Zone?
# rect ABC with sides |A B|, |B C| and diagonal |A C|
"""

Zone = tuple[Vec2D, Vec2D, Vec2D]
PolarVec2D = tuple[float, float]
Range = tuple[float, float]
"""Type alias for a range represented as a tuple of two floats"""

# Robot types
DifferentialDriveAction = tuple[float, float]  # (linear velocity, angular velocity)
BicycleAction = tuple[float, float]  # (acceleration, steering angle)
RobotAction = Union[DifferentialDriveAction, BicycleAction]
RobotPose = tuple[Vec2D, float]
"""
Type alias for a robot's pose represented as a tuple of a 2D vector (position)
and a float (orientation)
"""


# Pedestrian types
PedPose = tuple[Vec2D, float]
UnicycleAction = tuple[float, float]  # (acceleration, steering angle)
PedState = np.ndarray
PedGrouping = set[int]
ZoneAssignments = dict[int, int]

# Visualization types
RgbColor = tuple[int, int, int]
