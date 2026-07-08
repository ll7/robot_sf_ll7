"""Central default values for robot configuration.

This module provides the authoritative default robot radius used across
the codebase. All robot-radius defaults should reference this constant
to ensure consistency and make future changes easier.

IMPORTANT: The robot_radius default currently diverges across modules
for historical reasons. Any change that would alter computed benchmark
metrics is deferred per issue #4856. This module documents the current
state and provides the authoritative value.

Current State:
    Authoritative default: DEFAULT_ROBOT_RADIUS = 1.0 (meters)
    Used by:
        - DifferentialDriveSettings.radius (robot_sf/robot/differential_drive.py)
        - BicycleDriveSettings.radius (robot_sf/robot/bicycle_drive.py)
        - HolonomicDriveSettings.radius (robot_sf/robot/holonomic_drive.py)
        - StepSNQIProxy._DEFAULT_ROBOT_RADIUS (robot_sf/gym_env/snqi_proxy.py)

    Divergent defaults (metric-affecting changes deferred):
        - GridConfig.robot_radius = 0.3 (robot_sf/nav/occupancy_grid.py)
          Reason: Historical; affects grid rasterization visualization.
        - base_env planner setup fallback = 0.4 (robot_sf/gym_env/base_env.py)
          Reason: Historical; affects planner clearance margins.

To align all defaults to the authoritative value, propose a separate issue
after 2026-07-18 with benchmark impact analysis.
"""

from __future__ import annotations

# Authoritative default robot radius in meters.
# This is the collision radius used by physics and metrics computations.
DEFAULT_ROBOT_RADIUS: float = 1.0


def get_default_robot_radius() -> float:
    """Return the authoritative default robot radius in meters.

    Returns:
        float: Default robot radius (1.0 meters).

    Example:
        >>> from robot_sf.common.robot_defaults import get_default_robot_radius
        >>> radius = get_default_robot_radius()
        >>> assert radius == 1.0
    """
    return DEFAULT_ROBOT_RADIUS
