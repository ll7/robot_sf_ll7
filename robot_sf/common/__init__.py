"""
Common utilities for robot_sf package.

This module provides shared type definitions, error handling,
seed management, and compatibility shims.
"""

# Re-export commonly used symbols
from robot_sf.common.compat import validate_compatibility
from robot_sf.common.errors import raise_fatal_with_remedy, warn_soft_degrade
from robot_sf.common.seed import SeedReport, set_global_seed
from robot_sf.common.types import (
    Circle2D,
    Line2D,
    MapBounds,
    Point2D,
    Range,
    Range2D,
    RobotPose,
    Vec2D,
    Zone,
)

__all__ = [  # noqa: RUF022 - Grouped by source module for clarity
    # Types (from .types)
    "Circle2D",
    "Line2D",
    "MapBounds",
    "Point2D",
    "Range",
    "Range2D",
    "RobotPose",
    "Vec2D",
    "Zone",
    # Seed management (from .seed)
    "SeedReport",
    "set_global_seed",
    # Errors (from .errors)
    "raise_fatal_with_remedy",
    "warn_soft_degrade",
    # Compatibility (from .compat)
    "validate_compatibility",
]
