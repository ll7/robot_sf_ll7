"""
Module specifying types used in robot_sf
"""

from typing import Tuple, Union, Set, Dict
import numpy as np
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveAction
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleAction

# Geometry types
Vec2D = Tuple[float, float]
"""Type alias for a 2D vector represented as a tuple of two floats"""

Range2D = Tuple[float, float]  # (low, high)
"""Type alias for a range represented as a tuple of two floats
`(low, high)``
"""

Line2D = Tuple[float, float, float, float]
"""Type alias for a 2D line represented as a tuple of two 2D vectors (start and end points)"""

Point2D = Tuple[float, float]

Circle2D = Tuple[Vec2D, float]
"""Type alias for a 2D circle represented as a tuple of a 2D vector (center) and a float (radius)"""

MapBounds = Tuple[Range2D, Range2D]  # ((min_x, max_x), (min_y, max_y))
"""
Type alias for the bounds of a map represented as a tuple of two ranges (x and y)
`((min_x, max_x), (min_y, max_y))`
"""

Rect = Tuple[Vec2D, Vec2D, Vec2D]
"""
# TODO: Is there a difference between a Rect and a Zone?
# rect ABC with sides |A B|, |B C| and diagonal |A C|
"""

Zone = Tuple[Vec2D, Vec2D, Vec2D]
PolarVec2D = Tuple[float, float]
Range = Tuple[float, float]
"""Type alias for a range represented as a tuple of two floats"""

# Robot types
RobotAction = Union[DifferentialDriveAction, BicycleAction]
RobotPose = Tuple[Vec2D, float]
"""
Type alias for a robot's pose represented as a tuple of a 2D vector (position)
and a float (orientation)
"""

Robot = Union[DifferentialDriveRobot, BicycleDriveRobot]

# Pedestrian types
PedPose = Tuple[Vec2D, float]
UnicycleAction = Tuple[float, float]  # (acceleration, steering angle)
PedState = np.ndarray
PedGrouping = Set[int]
ZoneAssignments = Dict[int, int]

# Visualization types
RgbColor = Tuple[int, int, int]
