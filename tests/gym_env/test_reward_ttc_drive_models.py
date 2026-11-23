"""TTC uses production drivetrain velocities, including holonomic reverse motion."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.reward import route_completion_v3_reward
from robot_sf.gym_env.snqi_proxy import StepSNQIProxy
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings


def _robot(drive, heading, sign):
    if drive == "differential":
        settings = DifferentialDriveSettings(radius=0.2, allow_backwards=True)
    elif drive == "bicycle":
        settings = BicycleDriveSettings(radius=0.2, allow_backwards=True)
    else:
        settings = HolonomicDriveSettings(radius=0.2, command_mode=drive)
    robot = RobotSimulationConfig(robot_config=settings).robot_factory()
    robot.reset_state(((0.0, 0.0), heading))
    action = (
        (sign * math.cos(heading), sign * math.sin(heading)) if drive == "vx_vy" else (sign, 0.0)
    )
    robot.apply_action(action, 0.1)
    return robot


@pytest.mark.parametrize(
    ("drive", "contact_seconds"),
    [("differential", 44.95), ("bicycle", 45.0), ("vx_vy", 4.4), ("unicycle_vw", 4.4)],
)
@pytest.mark.parametrize("heading", [0.0, math.pi / 2.0])
@pytest.mark.parametrize("pedestrian_side", [-1.0, 1.0], ids=["behind", "ahead"])
@pytest.mark.parametrize("travel_sign", [-1.0, 1.0], ids=["reverse", "forward"])
def test_production_drive_ttc_respects_world_motion(
    drive, contact_seconds, heading, pedestrian_side, travel_sign
):
    """Reverse contacts behind, forward contacts ahead; receding discs never contact.

    dt=0.1, initial speed zero, unit command and a stationary pedestrian 5 m away:
    holonomic speed is 1 m/s with 0.1 m displacement, differential speed is 0.1 m/s
    with 0.005 m displacement, and bicycle speed is 0.1 m/s with zero displacement.
    Subtracting the 0.5 m combined footprint gives the literal TTC values above.
    """
    robot = _robot(drive, heading, travel_sign)
    direction = np.array([math.cos(heading), math.sin(heading)])
    simulator = SimpleNamespace(
        robots=[robot],
        robot_poses=[robot.pose],
        ped_radius=0.3,
        ped_pos=np.array([5.0 * pedestrian_side * direction]),
        ped_vel=np.zeros((1, 2)),
        last_ped_forces=[],
    )
    meta = StepSNQIProxy().compute_step_metrics(simulator, dt=0.1)
    approaching = pedestrian_side == travel_sign
    expected = contact_seconds if approaching else math.inf
    assert meta["time_to_collision"] == pytest.approx(expected)
    route_completion_v3_reward(meta, weights={"ttc_risk": -1.0})
    assert meta["reward_terms"]["ttc_risk"] == pytest.approx(
        -1.0 / contact_seconds if approaching else 0.0
    )
