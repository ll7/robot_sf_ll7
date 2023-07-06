from math import atan2, dist
from typing import Tuple, Union

import numpy as np
from gym import spaces

Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


def norm_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle(p_1: Vec2D, p_2: Vec2D, p_3: Vec2D) -> float:
    """Compute the difference between the linear projection of a vehicle at p1
    moving straight towards the goal p2 and the straight trajectory from p2
    towards the next goal p3."""

    v1_x, v1_y = p_2[0] - p_1[0], p_2[1] - p_1[1]
    v2_x, v2_y = p_3[0] - p_2[0], p_3[1] - p_2[1]
    o_1, o_2 = atan2(v1_y, v1_x), atan2(v2_y, v2_x)
    return norm_angle(o_2 - o_1)


def rel_pos(pose: RobotPose, target_coords: Vec2D) -> PolarVec2D:
    t_x, t_y = target_coords
    (r_x, r_y), orient = pose
    distance = dist(target_coords, (r_x, r_y))
    angle = norm_angle(atan2(t_y - r_y, t_x - r_x) - orient)
    return distance, angle


def target_sensor_obs(
        robot_pose: RobotPose,
        goal_pos: Vec2D,
        next_goal_pos: Union[Vec2D, None]) -> Tuple[float, float, float]:
    robot_pos, _ = robot_pose
    target_distance, target_angle = rel_pos(robot_pose, goal_pos)
    next_target_angle = angle(robot_pos, goal_pos, next_goal_pos) \
        if next_goal_pos is not None else 0.0
    return target_distance, target_angle, next_target_angle


def target_sensor_space(max_target_dist: float) -> spaces.Box:
    high = np.array([max_target_dist, np.pi, np.pi], dtype=np.float32)
    low = np.array([0.0, -np.pi, -np.pi], dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)
