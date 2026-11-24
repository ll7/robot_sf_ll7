"""Wall-aware proposals for the risk-aware controller's control samples."""

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np
import shapely
from numba import njit

from robot_sf.planner.clearance_geometry import StaticObstacleClearance, swept_pedestrian_clearance
from robot_sf.planner.drive_rollout import native_drive_rollout_batch
from robot_sf.robot.differential_drive import DifferentialDriveSettings


@dataclass(frozen=True)
class PassingLane:
    """A line along the route, offset toward one wall."""

    point: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    side: int


def passing_lane(
    obstacles: StaticObstacleClearance,
    position: np.ndarray,
    route: np.ndarray,
    pedestrians: np.ndarray,
    velocities: np.ndarray,
    robot_radius: float,
    contact_radius: float,
    previous_side: int,
) -> PassingLane | None:
    """Find lateral room on both sides of an approaching encounter.

    Returns:
        An offset line between parallel walls, or None outside such an encounter.
    """
    start, end = route
    segment = end - start
    length = np.linalg.norm(segment)
    if length <= robot_radius:
        return None
    direction = segment / length
    normal = np.array([-direction[1], direction[0]])
    longitudinal = (pedestrians - position) @ direction
    lateral = (pedestrians - start) @ normal
    within_reach = (
        (longitudinal > -contact_radius)
        & (longitudinal < 8.0 * robot_radius)
        & (np.abs(lateral) < contact_radius + 0.6 * robot_radius)
        & ((velocities @ direction < 0.3) | (longitudinal < contact_radius))
    )
    if not within_reach.any():
        return None

    # Check two cross-sections so a doorpost or a wall corner does not set the lane.
    center = position - ((position - start) @ normal) * normal
    centers = center + np.array([[0.0], [1.5 * robot_radius]]) * direction
    origins = np.repeat(centers[:, None, :], 2, axis=1)
    width_limit = 2.0 * (robot_radius + contact_radius)
    ends = origins + np.array([-width_limit, width_limit])[None, :, None] * normal
    cross_sections = shapely.linestrings(np.stack((origins, ends), axis=-2))
    walls = shapely.intersection(cross_sections, obstacles.geometry)
    room = shapely.distance(shapely.points(origins), walls)
    margin = 0.10 * robot_radius
    if (
        not np.isfinite(room).all()
        or np.any(room.sum(axis=1) > width_limit)
        or np.any(np.abs(room[1] - room[0]) > 2.0 * margin)
    ):
        return None
    offsets = room.min(axis=0) - robot_radius - margin
    if (offsets <= 0.0).any():
        return None

    first = np.argmin(np.where(within_reach, np.abs(longitudinal), np.inf))
    options = offsets * [-1, 1]
    side = previous_side or (-1, 1)[np.argmax(np.abs(options - lateral[first]))]
    return PassingLane(center + options[int(side > 0)] * normal, direction, normal, side)


@njit(cache=True)
def _steering_error(lateral_error: float, heading_error: float, lookahead: float) -> float:
    """Return the wrapped angle toward the offset line."""
    angle = atan2(lateral_error, lookahead) - heading_error
    return atan2(sin(angle), cos(angle))


@njit(cache=True)
def _steering_target(error: float, turn_limit: float) -> float:
    """Convert heading error to a bounded turn target.

    Returns:
        Angular speed within the configured limit.
    """
    return min(turn_limit, max(-turn_limit, 2.0 * error))


@njit(cache=True)
def passing_controls(
    lane_error: float,
    heading_error: float,
    horizon: int,
    dt: float,
    speed_limit: float,
    turn_limit: float,
    lookahead: float,
) -> tuple[np.ndarray, float]:
    """Propose a range of approaches and stops along an offset line.

    Returns:
        Control samples for native rollout and the turn target when yielding.
    """
    controls = np.empty((60, horizon, 2))
    row = 0
    for speed_fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        for duration in (0.25, 0.50, 0.75, 1.0):
            for shift_fraction in (0.65, 0.85, 1.0):
                error = shift_fraction * lane_error
                heading = heading_error
                for step in range(horizon):
                    steer = _steering_error(error, heading, lookahead)
                    turn = _steering_target(steer, turn_limit)
                    speed = 0.0
                    if step < horizon * duration:
                        speed = speed_fraction * speed_limit * max(0.0, cos(steer))
                    controls[row, step] = (speed, turn)
                    error -= speed * dt * sin(heading + 0.5 * turn * dt)
                    heading += turn * dt
                row += 1
    yield_turn = _steering_target(_steering_error(lane_error, heading_error, lookahead), turn_limit)
    return controls, yield_turn


def braking_path_is_clear(  # noqa: PLR0913
    command: tuple[float, float],
    position: np.ndarray,
    heading: float,
    velocity: np.ndarray,
    dt: float,
    drive: DifferentialDriveSettings,
    obstacles: StaticObstacleClearance,
    pedestrians: np.ndarray,
    pedestrian_velocities: np.ndarray,
    contact_radius: float,
) -> bool:
    """Check one control tick and its braking path against walls and moving pedestrians.

    Returns:
        Whether the swept footprint is clear through the first step at rest.
    """
    local, _, velocities = native_drive_rollout_batch(
        np.asarray(command).reshape(1, 1, 2), drive, velocity[0], velocity[1], dt, braking_tail=True
    )
    stopped = np.flatnonzero(np.abs(velocities[0, :, 0]) <= 1e-9)
    last = int(stopped[0]) + 1 if len(stopped) else len(local[0])
    c, s = cos(heading), sin(heading)
    points = position + local[0, :last] @ np.array([[c, s], [-s, c]])
    path = np.vstack((position, points))[None, ...]
    if obstacles.rollout_clearance(path, drive.radius)[0] <= 0.0:
        return False
    times = dt * np.arange(path.shape[1])
    clearance = swept_pedestrian_clearance(
        path, times, pedestrians, pedestrian_velocities, contact_radius
    )
    return bool((clearance > 0.0).all())
