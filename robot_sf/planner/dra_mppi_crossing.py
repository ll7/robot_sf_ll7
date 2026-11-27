"""Find crossings in the active route and check whether a command can stop clear."""

from __future__ import annotations

from math import ceil, cos, sin
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.planner.clearance_geometry import swept_pedestrian_clearance
from robot_sf.planner.drive_rollout import native_drive_rollout_batch

if TYPE_CHECKING:
    from robot_sf.planner.clearance_geometry import StaticObstacleClearance
    from robot_sf.robot.differential_drive import DifferentialDriveSettings

# Probe for a side passage that opens beyond the robot's approach corridor.
_OPENING_HALF_WIDTH_RADII = 4.5
_STOP_DWELL_S = 1.2
_PEDESTRIAN_MARGIN_M = 0.18


def crossing_spans(
    obstacles: StaticObstacleClearance,
    start: np.ndarray,
    goal: np.ndarray,
    robot_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find two-sided openings bracketed by narrower sections of the segment.

    Returns:
        Along-segment intervals in metres and the segment's unit direction.
    """
    segment = goal - start
    length = np.linalg.norm(segment)
    if length < 4.0 * robot_radius:
        return np.empty((0, 2)), np.zeros(2)
    direction = segment / length
    normal = np.array([-direction[1], direction[0]])
    progress = np.linspace(0.0, length, ceil(length / robot_radius) + 1)
    centers = start + progress[:, None] * direction
    half_width = _OPENING_HALF_WIDTH_RADII * robot_radius
    sides = centers[:, None, :] + np.array([-half_width, half_width])[:, None] * normal
    wide = obstacles.rollout_clearance(sides, 0.0) > 0.0
    edges = np.diff(np.r_[False, wide, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    spans = np.column_stack((progress[starts], progress[ends]))
    bracketed = (starts > 0) & (ends < len(progress) - 1)
    return spans[bracketed & (spans[:, 1] - spans[:, 0] >= robot_radius)], direction


def _stopping_clearance(  # noqa: PLR0913
    commands: np.ndarray,
    position: np.ndarray,
    heading: float,
    velocity: np.ndarray,
    dt: float,
    drive: DifferentialDriveSettings,
    obstacles: StaticObstacleClearance,
    pedestrians: np.ndarray,
    pedestrian_velocities: np.ndarray,
    contact_radius: float,
) -> np.ndarray:
    """Measure swept separation through one control tick, braking and a dwell.

    Returns:
        The minimum wall or pedestrian surface clearance for each first command.
    """
    local, _, speeds = native_drive_rollout_batch(
        commands[:, None, :], drive, velocity[0], velocity[1], dt, braking_tail=True
    )
    hold = ceil(_STOP_DWELL_S / dt)
    local = np.concatenate((local, np.repeat(local[:, -1:], hold, axis=1)), axis=1)
    c, s = cos(heading), sin(heading)
    points = position + local @ np.array([[c, s], [-s, c]])
    initial = np.broadcast_to(position, (len(points), 1, 2))
    paths = np.concatenate((initial, points), axis=1)
    times = dt * np.arange(paths.shape[1])
    separation = swept_pedestrian_clearance(
        paths, times, pedestrians, pedestrian_velocities, contact_radius
    )
    stopped = np.abs(speeds[..., 0]) <= 1e-9
    last_step = np.where(stopped.any(axis=1), stopped.argmax(axis=1) + 1, speeds.shape[1])
    through_dwell = np.arange(separation.shape[1]) < (last_step + hold)[:, None]
    pedestrian_gap = np.where(through_dwell, separation, np.inf).min(axis=1)
    return np.minimum(pedestrian_gap, obstacles.rollout_clearance(paths, drive.radius))


def select_stoppable_plan(  # noqa: PLR0913
    plan: np.ndarray,
    controls: np.ndarray,
    total_cost: np.ndarray,
    position: np.ndarray,
    heading: float,
    velocity: np.ndarray,
    dt: float,
    drive: DifferentialDriveSettings,
    obstacles: StaticObstacleClearance,
    pedestrians: np.ndarray,
    pedestrian_velocities: np.ndarray,
    contact_radius: float,
) -> np.ndarray | None:
    """Keep the proposed plan or find an existing sample that can stop clear.

    Returns:
        The original plan, the cheapest wall-clear and stoppable sample, or None to yield.
    """
    # A distant walker cannot enter either the braking path or the stopped footprint.
    speed = abs(velocity[0]) + drive.max_linear_accel * dt
    assert drive.max_linear_decel is not None
    deceleration = min(drive.max_linear_decel, drive.max_linear_accel)
    duration = dt * (1 + ceil(speed / (deceleration * dt)) + ceil(_STOP_DWELL_S / dt))
    distance = np.linalg.norm(pedestrians - position, axis=1)
    closing_speed_bound = speed + np.linalg.norm(pedestrian_velocities, axis=1)
    in_reach = distance < contact_radius + _PEDESTRIAN_MARGIN_M + duration * closing_speed_bound
    if not in_reach.any():
        return plan

    # Preserve positive clearance when the observation starts inside the extra margin.
    margin = min(_PEDESTRIAN_MARGIN_M, max(0.0, distance.min() - contact_radius - 0.01))
    stopping_state = (
        position,
        heading,
        velocity,
        dt,
        drive,
        obstacles,
        pedestrians[in_reach],
        pedestrian_velocities[in_reach],
        contact_radius + margin,
    )
    if _stopping_clearance(plan[:1], *stopping_state)[0] > 0.0:
        return plan
    candidates = np.flatnonzero(np.isfinite(total_cost))
    clear = _stopping_clearance(controls[candidates, 0], *stopping_state) > 0.0
    if not clear.any():
        return None
    best = np.argmin(np.where(clear, total_cost[candidates], np.inf))
    return controls[candidates[best]]
