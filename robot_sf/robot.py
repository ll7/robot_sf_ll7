from math import dist, atan2, sin, cos
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[PolarVec2D, float]
WheelSpeedState = Tuple[float, float] # tuple of (left, right) speeds


def rel_pos(pose: RobotPose, target_coords: Vec2D) -> PolarVec2D:
    t_x, t_y = target_coords
    (r_x, r_y), orient = pose
    distance = dist(target_coords, (r_x, r_y))

    angle = atan2(t_y - r_y, t_x - r_x) - orient
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return distance, angle


@dataclass
class RobotSettings:
    radius: float = 1.0
    max_linear_speed: float = 0.5
    max_angular_speed: float = 0.5
    rob_collision_radius: float = 0.7
    wheel_radius: float = 0.05
    interaxis_length: float = 0.3


@dataclass
class DifferentialDriveState:
    pose: RobotPose
    velocity: PolarVec2D = field(default=(0, 0))
    last_wheels_speed: WheelSpeedState = field(default=(0, 0))
    wheels_speed: WheelSpeedState = field(default=(0, 0))


@dataclass
class DifferentialDriveMotion:
    config: RobotSettings

    def move(self, state: DifferentialDriveState, action: PolarVec2D, d_t: float):
        robot_vel = self._robot_velocity(state.velocity, action)
        new_wheel_speeds = self._resulting_wheel_speeds(robot_vel)
        distance = self._covered_distance(state.wheels_speed, new_wheel_speeds, d_t)
        new_orient = self._new_orientation(state.pose[1], state.wheels_speed, new_wheel_speeds, d_t)
        state.pose = self._compute_odometry(state.pose, (distance, new_orient))
        state.last_wheels_speed = state.wheels_speed
        state.wheels_speed = new_wheel_speeds
        state.velocity = robot_vel

    def _robot_velocity(self, velocity: PolarVec2D, action: PolarVec2D) -> PolarVec2D:
        dot_x = velocity[0] + action[0]
        dot_orient = velocity[1] + action[1]
        dot_x = np.clip(dot_x, 0, self.config.max_linear_speed)
        angular_max = self.config.max_angular_speed
        dot_orient = np.clip(dot_orient, -angular_max, angular_max)
        return dot_x, dot_orient

    def _resulting_wheel_speeds(self, movement: PolarVec2D) -> WheelSpeedState:
        dot_x, dot_orient = movement
        diff = self.config.interaxis_length * dot_orient / 2
        new_left_wheel_speed = (dot_x - diff) / self.config.wheel_radius
        new_right_wheel_speed = (dot_x + diff) / self.config.wheel_radius
        return new_left_wheel_speed, new_right_wheel_speed

    def _covered_distance(self, last_wheel_speeds: WheelSpeedState,
                                new_wheel_speeds: WheelSpeedState, d_t: float) -> float:
        last_wheel_speed_left, last_wheel_speed_right = last_wheel_speeds
        wheel_speed_left, wheel_speed_right = new_wheel_speeds
        velocity = ((last_wheel_speed_left + wheel_speed_left) / 2 \
            + (last_wheel_speed_right + wheel_speed_right) / 2)
        distance_covered = self.config.wheel_radius / 2 * velocity * d_t
        return distance_covered

    def _new_orientation(self, robot_orient: float, last_wheel_speeds: WheelSpeedState,
                         wheel_speeds: WheelSpeedState, d_t: float) -> float:
        last_wheel_speed_left, last_wheel_speed_right = last_wheel_speeds
        wheel_speed_left, wheel_speed_right = wheel_speeds

        right_left_diff = (last_wheel_speed_right + wheel_speed_right) / 2 \
            - (last_wheel_speed_left + wheel_speed_left) / 2
        diff = self.config.wheel_radius / self.config.interaxis_length * right_left_diff * d_t
        new_orient = robot_orient + diff
        return new_orient

    def _compute_odometry(self, old_pose: RobotPose, movement: PolarVec2D) -> RobotPose:
        distance_covered, new_orient = movement
        (robot_x, robot_y), old_orient = old_pose
        rel_rotation = (old_orient + new_orient) / 2
        new_x = robot_x + distance_covered * cos(rel_rotation)
        new_y = robot_y + distance_covered * sin(rel_rotation)
        return ((new_x, new_y), new_orient)


@dataclass
class DifferentialDriveRobot():
    """Representing a robot with differential driving behavior"""

    spawn_pose: RobotPose
    goal: Vec2D
    config: RobotSettings
    state: DifferentialDriveState = field(init=False)
    movement: DifferentialDriveMotion = field(init=False)

    def __post_init__(self):
        self.state = DifferentialDriveState(self.spawn_pose)
        self.movement = DifferentialDriveMotion(self.config)

    @property
    def pos(self) -> Vec2D:
        return self.state.pose[0]

    @property
    def pose(self) -> RobotPose:
        return self.state.pose

    @property
    def dist_to_goal(self) -> float:
        return dist(self.pose[0], self.goal)

    @property
    def current_speed(self) -> PolarVec2D:
        return self.state.velocity

    def apply_action(self, action: PolarVec2D, d_t: float):
        self.movement.move(self.state, action, d_t)
