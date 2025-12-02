"""Module goal_sensor auto-generated docstring."""

from math import atan2, dist

import numpy as np
from gymnasium import spaces

from robot_sf.common.types import PolarVec2D, RobotPose, Vec2D


def norm_angle(angle: float) -> float:
    """Norm angle.

    Args:
        angle: Auto-generated placeholder description.

    Returns:
        float: Auto-generated placeholder description.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_trajectory_angle(p_1: Vec2D, p_2: Vec2D, p_3: Vec2D) -> float:
    """
    Compute the difference between the linear projection of a vehicle at p1
    moving straight towards the goal p2 and the straight trajectory from p2
    towards the next goal p3. For a smooth trajectory, the driving agent
    has to steer such that the angle is about 0.
    """

    v1_x, v1_y = p_2[0] - p_1[0], p_2[1] - p_1[1]
    v2_x, v2_y = p_3[0] - p_2[0], p_3[1] - p_2[1]
    o_1, o_2 = atan2(v1_y, v1_x), atan2(v2_y, v2_x)
    return norm_angle(o_2 - o_1)


def rel_pos(pose: RobotPose, target_coords: Vec2D) -> PolarVec2D:
    """
    Compute the relative position of a target with respect to the robot.
    """
    # Extract the target coordinates
    t_x, t_y = target_coords

    # Extract the robot's coordinates and orientation
    (r_x, r_y), orient = pose

    # Calculate the distance to the target
    distance = dist(target_coords, (r_x, r_y))

    # Calculate the angle to the target, normalized to the robot's orientation
    angle = norm_angle(atan2(t_y - r_y, t_x - r_x) - orient)
    return distance, angle


def target_sensor_obs(
    robot_pose: RobotPose,
    goal_pos: Vec2D,
    next_goal_pos: Vec2D | None,
) -> tuple[float, float, float]:
    """
    Calculate the observations for the target sensor.

    Parameters
    ----------
    robot_pose : RobotPose
        The current pose of the robot, consisting of its coordinates and orientation.
    goal_pos : Vec2D
        The coordinates of the current goal.
    next_goal_pos : Union[Vec2D, None]
        The coordinates of the next goal, or None if there is no next goal.

    Returns
    -------
    Tuple[float, float, float]
        The distance and angle to the current goal, and the trajectory angle to the next goal.
    """
    # Extract the robot's position from its pose
    robot_pos, _ = robot_pose

    # Calculate the distance and angle to the current goal
    target_distance, target_angle = rel_pos(robot_pose, goal_pos)

    # Calculate the trajectory angle to the next goal, or 0.0 if there is no next goal
    next_target_angle = (
        calculate_trajectory_angle(robot_pos, goal_pos, next_goal_pos)
        if next_goal_pos is not None
        else 0.0
    )

    return target_distance, target_angle, next_target_angle


def target_sensor_space(max_target_dist: float) -> spaces.Box:
    """
    Create a Box space for the target sensor.

    The Box space represents the possible observations from the target sensor. It is
    a 3-dimensional space, with the dimensions representing the distance to the target,
    the angle to the target, and the trajectory angle to the next target.

    Parameters
    ----------
    max_target_dist : float
        The maximum possible distance to the target.

    Returns
    -------
    spaces.Box
        The Box space for the target sensor.
    """
    # Define the upper bounds for the distance, target angle, and next target angle
    high = np.array([max_target_dist, np.pi, np.pi], dtype=np.float32)

    # Define the lower bounds for the distance, target angle, and next target angle
    low = np.array([0.0, -np.pi, -np.pi], dtype=np.float32)

    # Return a Box space defined by the lower and upper bounds
    return spaces.Box(low=low, high=high, dtype=np.float32)
