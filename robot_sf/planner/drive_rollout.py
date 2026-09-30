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
