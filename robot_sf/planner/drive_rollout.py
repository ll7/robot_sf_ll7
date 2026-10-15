"""Native differential-drive command forecasts shared by physical safety gates."""

import numpy as np

from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


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


def native_drive_rollout_batch(
    sequences: np.ndarray,
    settings: DifferentialDriveSettings,
    speed: float,
    angular_speed: float,
    dt: float,
    *,
    braking_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast batches using native wheel trapezoids and midpoint odometry.

    Returns local positions, headings and feasible endpoint commands. The initial
    pose is omitted, as in ``native_drive_rollout``. Optional zero commands append
    a fixed braking tail. Callers must include time zero in swept checks.

    Returns:
        Arrays with shapes (batch, time, 2), (batch, time), (batch, time, 2).
    """
    commands = np.asarray(sequences, dtype=float)
    if (
        commands.ndim != 3
        or commands.shape[-1] != 2
        or commands.shape[1] < 1
        or not np.isfinite(commands).all()
        or not np.isfinite([speed, angular_speed, dt]).all()
        or dt <= 0
        or braking_steps < 0
    ):
        raise ValueError("Require finite (batch,time,2) commands and positive dt")
    count, horizon, _ = commands.shape
    total = horizon + braking_steps
    positions = np.zeros((count, total, 2))
    headings = np.zeros((count, total))
    velocities = np.zeros((count, total, 2))
    velocity = np.tile([speed, angular_speed], (count, 1)).astype(float)
    point = np.zeros((count, 2))
    heading = np.zeros(count)
    radius, axle = settings.wheel_radius, settings.interaxis_length

    def wheels(vw: np.ndarray) -> np.ndarray:
        return (
            np.column_stack((vw[:, 0] - axle * vw[:, 1] / 2, vw[:, 0] + axle * vw[:, 1] / 2))
            / radius
        )

    wheel = wheels(velocity)
    for index in range(total):
        target = commands[:, index] if index < horizon else np.zeros((count, 2))
        acceleration = np.clip(
            (target - velocity) / dt,
            [-settings.max_linear_decel, -settings.max_angular_accel],
            [settings.max_linear_accel, settings.max_angular_accel],
        )
        velocity = np.clip(
            velocity + acceleration * dt,
            [settings.min_linear_speed, -settings.max_angular_speed],
            [settings.max_linear_speed, settings.max_angular_speed],
        )
        new_wheel = wheels(velocity)
        average_wheel = (wheel + new_wheel) / 2
        distance = radius * np.mean(average_wheel, axis=1) * dt
        new_heading = heading + radius / axle * (average_wheel[:, 1] - average_wheel[:, 0]) * dt
        middle = (heading + new_heading) / 2
        point = point + distance[:, None] * np.column_stack((np.cos(middle), np.sin(middle)))
        positions[:, index], headings[:, index], velocities[:, index] = point, new_heading, velocity
        wheel, heading = new_wheel, new_heading
    return positions, headings, velocities
