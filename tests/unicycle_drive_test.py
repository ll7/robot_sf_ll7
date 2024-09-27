from math import pi
from robot_sf.ped_ego.unicycle_drive import UnicycleMotion, UnicycleDriveState, UnicycleDriveSettings, UnicycleAction


def norm_angle(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle >= 2 * pi:
        angle -= 2 * pi
    return angle


def test_unicycle_can_drive_right_curve():
    motion = UnicycleMotion(UnicycleDriveSettings())
    pose_before, vel_before = ((0, 0), 0), 1
    state = UnicycleDriveState(pose_before, vel_before)

    right_curve_action: UnicycleAction = (0, -0.78)
    motion.move(state, right_curve_action, 1.0)

    steer_straight_action: UnicycleAction = (0, 0)
    motion.move(state, steer_straight_action, 1.0)
    pos_after, orient_after = state.pose

    assert pos_after[0] > 0 and pos_after[1] < 0    # position in 4th quadrant
    assert 1.5 * pi < norm_angle(orient_after) < 2 * pi  # orientation in 4th quadrant


def test_unicycle_can_drive_left_curve():
    motion = UnicycleMotion(UnicycleDriveSettings())
    pose_before, vel_before = ((0, 0), 0), 1
    state = UnicycleDriveState(pose_before, vel_before)

    left_curve_action: UnicycleAction = (0, 0.78)
    motion.move(state, left_curve_action, 1.0)

    steer_straight_action: UnicycleAction = (0, 0)
    motion.move(state, steer_straight_action, 1.0)
    pos_after, orient_after = state.pose

    assert pos_after[0] > 0 and pos_after[1] > 0  # position in 1st quadrant
    assert 0 < norm_angle(orient_after) < 0.5 * pi  # orientation in 1st quadrant


def test_unicycle_acceleration():
    motion = UnicycleMotion(UnicycleDriveSettings(allow_backwards=True))
    pose_before, vel_before = ((0, 0), 0), 0
    state = UnicycleDriveState(pose_before, vel_before)

    acceleration_action: UnicycleAction = (1, 0)
    motion.move(state, acceleration_action, 1.0)
    speed = state.velocity
    assert speed == 1  # assert gained speed

    motion.move(state, acceleration_action, 1.0)
    motion.move(state, acceleration_action, 1.0)
    pos_after, orient_after = state.pose
    speed = state.velocity

    assert pos_after[1] == 0 and orient_after == 0  # assert steering straight
    assert pos_after[0] > 0 and speed == 3  # assert moving forward and max speed

    deceleration_action: UnicycleAction = (-1, 0)
    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == 2  # assert lost speed

    motion.move(state, deceleration_action, 1.0)
    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == 0  # assert stopped

    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == -1  # assert moving backwards

    motion = UnicycleMotion(UnicycleDriveSettings(allow_backwards=False))
    motion.move(state, deceleration_action, 1.0)
    speed = state.velocity
    assert speed == 0  # assert not moving backwards
