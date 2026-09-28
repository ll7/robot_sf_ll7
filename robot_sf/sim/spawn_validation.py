"""Reset-time spawn validation shared by the simulator, benchmark runner, and preflight.

Issue #9725: every benchmark episode records its reset clearance, and a reset in
contact (robot-pedestrian or robot-obstacle) marks the row ``invalid_run`` with
reason ``spawn_overlap``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.nav.spawn_clearance import robot_obstacle_clearance
from robot_sf.ped_npc.ped_population import validate_spawn_footprints

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D


def robot_positions_and_radii(simulator: Any) -> list[tuple[Vec2D, float]]:
    """Return ``((x, y), radius)`` for every robot of a simulator."""
    result: list[tuple[Vec2D, float]] = []
    for robot in getattr(simulator, "robots", []) or []:
        (x, y), _theta = robot.pose
        result.append(((float(x), float(y)), float(robot.config.radius)))
    return result


def reset_spawn_clearance(simulator: Any) -> dict[str, Any]:
    """Measure robot-pedestrian and robot-obstacle clearance of the current state.

    Intended to run right after a reset. Uses the same radii and geometry as the
    collision checks, so a negative clearance means the episode starts in contact.

    Returns:
        JSON-serializable clearance block with an ``overlap`` verdict.
    """

    ped_radius = float(simulator.config.ped_radius)
    ped_xy = [tuple(map(float, row)) for row in list(simulator.ped_pos)]
    robots = robot_positions_and_radii(simulator)
    ped_clearances: list[float] = []
    overlapping_rows: set[int] = set()
    obstacle_clearances: list[float] = []
    for robot_xy, robot_radius in robots:
        report = validate_spawn_footprints(robot_xy, robot_radius, ped_xy, ped_radius)
        if report.min_clearance_m is not None:
            ped_clearances.append(float(report.min_clearance_m))
        overlapping_rows.update(report.overlapping_rows)
        obstacle_clearances.append(
            robot_obstacle_clearance(simulator.map_def, robot_xy, robot_radius)
        )
    min_ped = min(ped_clearances) if ped_clearances else None
    min_obstacle = min(obstacle_clearances) if obstacle_clearances else None
    ped_overlap = bool(overlapping_rows)
    obstacle_overlap = min_obstacle is not None and min_obstacle < 0.0
    return {
        "robot_pedestrian_min_surface_clearance_m": min_ped,
        "robot_obstacle_min_surface_clearance_m": min_obstacle,
        "overlapping_pedestrian_rows": sorted(overlapping_rows),
        "pedestrian_overlap": ped_overlap,
        "obstacle_overlap": bool(obstacle_overlap),
        "overlap": bool(ped_overlap or obstacle_overlap),
    }


__all__ = ["reset_spawn_clearance", "robot_positions_and_radii"]
