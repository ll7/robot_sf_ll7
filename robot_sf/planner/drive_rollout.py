"""Native differential-drive command forecasts shared by physical safety gates."""

from math import ceil, cos, sin

import numpy as np
from numba import njit

from robot_sf.common.math_utils import clip_scalar
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings

_clip = njit(inline="always")(clip_scalar)


def native_drive_rollout(
    sequence: np.ndarray,
    settings: DifferentialDriveSettings,
    speed: float,
    angular_speed: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local positions, headings and endpoint velocities, including braking.

    Reuse FX4's velocity-to-acceleration conversion and native wheel odometry.
    A terminal zero command is integrated until translation stops; the caller
    separates that static viability tail from its pedestrian forecast horizon.
    """
    drive = DifferentialDriveRobot(settings)
    drive.state.velocity = (speed, angular_speed)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    positions, headings, velocities = [], [], []

    def advance(command) -> None:
        velocity = np.asarray(drive.current_speed)
        drive.apply_action(tuple((np.asarray(command) - velocity) / dt), dt)
        positions.append(drive.pos)
        headings.append(drive.pose[1])
        velocities.append(drive.current_speed)

    for action in sequence:
        advance(action)
    while abs(drive.current_speed[0]) > 1e-9:
        advance((0.0, 0.0))
    return np.asarray(positions), np.asarray(headings), np.asarray(velocities)


@njit(cache=True)
def _native_drive_batch(commands, speed, angular_speed, dt, limits, brake_steps):
    """Keep wheel odometry and target-to-acceleration conversion in native order.

    Returns:
        Local position, heading and velocity sequences.
    """
    min_v, max_v, max_w, accel, decel, angular_accel, wheel_radius, axle = limits
    count, horizon, _ = commands.shape
    length = horizon + brake_steps
    positions = np.zeros((count, length, 2))
    velocities = np.zeros_like(positions)
    angles = np.zeros((count, length))
    for i in range(count):
        x = y = heading = 0.0
        v, w = speed, angular_speed
        left = (v - axle * w / 2) / wheel_radius
        right = (v + axle * w / 2) / wheel_radius
        for t in range(length):
            if t < horizon or abs(v) > 1e-9:
                target_v = target_w = 0.0
                if t < horizon:
                    target_v, target_w = commands[i, t]
                a = _clip((target_v - v) / dt, -decel, accel)
                aw = _clip((target_w - w) / dt, -angular_accel, angular_accel)
                v = _clip(v + a * dt, min_v, max_v)
                w = _clip(w + aw * dt, -max_w, max_w)
                new_left = (v - axle * w / 2) / wheel_radius
                new_right = (v + axle * w / 2) / wheel_radius
                avg_left = (left + new_left) / 2
                avg_right = (right + new_right) / 2
                distance = wheel_radius * ((avg_left + avg_right) / 2) * dt
                new_angle = heading + (wheel_radius / axle) * (avg_right - avg_left) * dt
                mid_angle = (heading + new_angle) / 2
                x += distance * cos(mid_angle)
                y += distance * sin(mid_angle)
                left, right, heading = new_left, new_right, new_angle
            positions[i, t, 0], positions[i, t, 1] = x, y
            velocities[i, t, 0], velocities[i, t, 1] = v, w
            angles[i, t] = heading
    return positions, angles, velocities


def native_drive_rollout_batch(
    commands: np.ndarray,
    settings: DifferentialDriveSettings,
    speed: float,
    angular_speed: float,
    dt: float,
    *,
    braking_tail: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out velocity targets for a batch, optionally padding each braking tail.

    Returns:
        Local positions, headings and polar velocities at the end of each step.
        Tail entries after translation stops repeat the stopped pose.
    """
    assert settings.max_linear_decel is not None
    decel = float(settings.max_linear_decel)
    limits = (
        settings.min_linear_speed,
        settings.max_linear_speed,
        settings.max_angular_speed,
        settings.max_linear_accel,
        decel,
        settings.max_angular_accel,
        settings.wheel_radius,
        settings.interaxis_length,
    )
    stopping_time = max(
        settings.max_linear_speed / decel, -settings.min_linear_speed / settings.max_linear_accel
    )
    tail = ceil(stopping_time / dt) if braking_tail else 0
    return _native_drive_batch(commands, speed, angular_speed, dt, limits, tail)
